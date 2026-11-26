"""Stage-1 ONNX Compatibility Bridge for Armiger Ultra v3.17.0 Clean-Sheet Graphs.

Allows `ArmigerStandaloneONNX` (`armiger_inference.py`) and `ArmigerUltraPipeline`
(`armiger_ultra_pipeline.py`) to run the v3.17.0 clean-sheet Stage-1 ONNX graph
(`armiger_ultra.fp32.onnx` and `armiger.int8.onnx` taking `subword_ids`, `audio_subword_ids`,
`audio_byte_ids`, `audio_p_feats`, `byte_ids`, `file_tok_ids`, `file_num_feats`, `file_mask`,
`tree_summary`) and return the 9-head logit dictionary (`category`, `adult`, `junk`,
`anime`, `english_subs`, `lang`, `owner_scope`, `title_language`, `span_bio`) in the
exact `NEURAL_HEAD_CLASSES` order expected by callers and plugins.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from audio_features_p import AUDIO_BYTE_L, encode_audio_p_bytes, extract_audio_p_features
from canonical_views import project_audio_only
from featurizer_v317 import featurize_single_row
from tokenizer_v317 import PurePythonUnigramTokenizer

# CAT11 in v3.17.0: ["movie", "tv_show", "music", "game", "software", "ebook", "audiobook", "comic", "course", "xxx", "unknown"]
# NEURAL_HEAD_CLASSES["category"]: ["movie", "tv_show", "xxx", "music", "comic", "game", "audiobook", "software", "ebook", "course", "unknown"]
_V317_TO_LEGACY_CAT11 = [0, 1, 9, 2, 7, 3, 6, 4, 5, 8, 10]

_TOK_CACHE: Dict[str, PurePythonUnigramTokenizer] = {}


def is_v317_session(session: Any) -> bool:
    if session is None:
        return False
    try:
        return any(getattr(inp, "name", "") == "subword_ids" for inp in session.get_inputs())
    except Exception:
        return False


def _get_v317_tokenizer(vocab_path: Optional[Path] = None, model_dir: Optional[Path] = None) -> PurePythonUnigramTokenizer:
    candidates: List[Path] = []
    if vocab_path is not None:
        candidates.append(Path(vocab_path))
        candidates.append(Path(vocab_path).with_name("tokenizer_v317_32k.json"))
    if model_dir is not None:
        candidates.append(Path(model_dir) / "tokenizer_v317_32k.json")
        candidates.append(Path(model_dir) / "vocab.json")
    candidates.append(_THIS_DIR / "tokenizer_v317_32k.json")
    candidates.append(_THIS_DIR / "vocab.json")
    for p in candidates:
        if not p.is_file():
            continue
        key = str(p.resolve())
        tok = _TOK_CACHE.get(key)
        if tok is not None:
            return tok
        try:
            tok = PurePythonUnigramTokenizer.from_json(p)
            if tok.vocab_size >= 1000:
                _TOK_CACHE[key] = tok
                return tok
        except Exception:
            continue
    raise FileNotFoundError("Could not locate v3.17.0 32K SentencePiece Unigram tokenizer in vocab.json or tokenizer_v317_32k.json")


def _join_audio_p_text(clean_title: str, media_names: list[str]) -> str:
    if not media_names:
        return clean_title
    return clean_title + " " + " ".join(media_names[:4])


def run_v317_stage1_as_legacy_8heads(
    session: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    vocab_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    B = len(rows)
    if B == 0:
        return {
            "category": np.zeros((0, 11), dtype=np.float64),
            "adult": np.zeros((0, 2), dtype=np.float64),
            "junk": np.zeros((0, 2), dtype=np.float64),
            "anime": np.zeros((0, 2), dtype=np.float64),
            "english_subs": np.zeros((0, 3), dtype=np.float64),
            "lang": np.zeros((0, 3), dtype=np.float64),
            "owner_scope": np.zeros((0, 2), dtype=np.float64),
            "title_language": np.zeros((0, 2), dtype=np.float64),
            "span_bio": np.zeros((0, 256, 7), dtype=np.float64),
        }

    tok = _get_v317_tokenizer(vocab_path=vocab_path, model_dir=model_dir)
    norm_rows = []
    for r in rows:
        d = dict(r)
        if "name" not in d and "title" in d:
            d["name"] = d["title"]
        norm_rows.append(d)

    px_pairs = [project_audio_only(str(r.get("name") or ""), r.get("files") or []) for r in norm_rows]
    aud_txts = [_join_audio_p_text(ct, mn) for ct, mn in px_pairs]
    feats = [featurize_single_row(r, tok, aud_txt=aud_txts[i]) for i, r in enumerate(norm_rows)]

    sub_b = np.stack([x["subword_ids"] for x in feats])
    aud_b = np.stack([x["audio_subword_ids"] for x in feats])
    byt_b = np.stack([x["byte_ids"] for x in feats])
    ftok_b = np.stack([x["file_tok_ids"] for x in feats])
    fnum_b = np.stack([x["file_num_feats"] for x in feats])
    fmsk_b = np.stack([x["file_mask"] for x in feats])
    tsum_b = np.stack([x["tree_summary"] for x in feats])
    ab_b = np.stack([encode_audio_p_bytes(t, AUDIO_BYTE_L) for t in aud_txts])
    ap_b = np.stack([extract_audio_p_features(t) for t in aud_txts])

    mf = max(1, int(fmsk_b.sum(axis=1).max()))
    feed = {
        "subword_ids": sub_b,
        "audio_subword_ids": aud_b,
        "audio_byte_ids": ab_b,
        "audio_p_feats": ap_b,
        "byte_ids": byt_b,
        "file_tok_ids": ftok_b[:, :mf, :],
        "file_num_feats": fnum_b[:, :mf, :],
        "file_mask": fmsk_b[:, :mf],
        "tree_summary": tsum_b,
    }
    c11_l, jk2_l, sf_l, an_l, aud_l, bio_l, ptr_l, _ = session.run(None, feed)

    # 1. category: permute from v3.17.0 CAT11 order to NEURAL_HEAD_CLASSES["category"]
    cat_leg = c11_l[:, _V317_TO_LEGACY_CAT11].astype(np.float64).copy()
    for i, r in enumerate(norm_rows):
        fl = r.get("files") or []
        nm_low = str(r.get("name") or "").lower()
        is_space_year_pack = (" (" in nm_low and ")" in nm_low)
        if len(fl) >= 2 and is_space_year_pack and cat_leg[i, 0] > cat_leg[i, 1]:
            vid_cnt = 0
            for f_it in fl:
                fp = str(f_it.get("path") if isinstance(f_it, Mapping) else f_it).lower()
                if fp.endswith((".mkv", ".mp4", ".avi", ".m4v", ".ts")):
                    vid_cnt += 1
            if vid_cnt >= 2 and (cat_leg[i, 0] - cat_leg[i, 1]) > 1.25:
                mid = 0.5 * (cat_leg[i, 0] + cat_leg[i, 1])
                cat_leg[i, 0] = mid + 0.60
                cat_leg[i, 1] = mid - 0.60

    # 2. adult: ["non_adult", "adult"] derived from xxx logit vs max non-xxx logit
    xxx_l = c11_l[:, 9].astype(np.float64)
    non_xxx_max = np.max(c11_l[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]], axis=1).astype(np.float64)
    adult_leg = np.stack([non_xxx_max, xxx_l - 1.0], axis=1)

    # 3. junk: ["clean", "junk"] — jk2_l in v3.17.0 Stage-1 contains two BCE binary logits
    # (col 0 = general junk logit, col 1 = structural/trap junk logit).
    z_jk = np.maximum(jk2_l[:, 0], jk2_l[:, 1]).astype(np.float64).copy()
    is_tiny_stub = [False] * B
    for i, r in enumerate(norm_rows):
        fl = r.get("files") or []
        if fl:
            max_vid_sz = 0
            max_exec_sz = 0
            has_dll_or_lnk = False
            for f_it in fl:
                if isinstance(f_it, Mapping):
                    fp = str(f_it.get("path") or f_it.get("name") or "").lower()
                    fsz = int(f_it.get("size") or 0)
                else:
                    fp = str(f_it).lower()
                    fsz = 0
                if fp.endswith((".mkv", ".mp4", ".avi", ".m4v")):
                    max_vid_sz = max(max_vid_sz, fsz)
                elif fp.endswith((".exe", ".scr", ".bat", ".cmd", ".com", ".msi")):
                    max_exec_sz = max(max_exec_sz, fsz)
                elif fp.endswith((".dll", ".lnk", ".ps1", ".vbs", ".hta")):
                    has_dll_or_lnk = True
            if max_vid_sz >= 50_000_000 and 0 < max_exec_sz <= 4096 and not has_dll_or_lnk:
                is_tiny_stub[i] = True
                z_jk[i] = min(float(jk2_l[i, 0]), -6.0)
    junk_leg = np.stack([-0.5 * z_jk, 0.5 * z_jk], axis=1)
    for i, f in enumerate(feats):
        sig = f.get("p2") or {}
        top_cat = int(np.argmax(cat_leg[i]))
        is_sg = top_cat in (5, 7)  # 5=game, 7=software in NEURAL_HEAD_CLASSES["category"]
        nm_i_low = str(norm_rows[i].get("name") or "").lower()
        fl_i = norm_rows[i].get("files") or []
        has_double_ext_dropper = any(
            any(
                s.endswith(v_ext + e_ext) or s.endswith(v_ext + " " + e_ext)
                for v_ext in (".mkv", ".mp4", ".avi", ".m4v", ".mov", ".wmv", ".srt")
                for e_ext in (".exe", ".scr", ".bat", ".cmd", ".com", ".lnk", ".msi", ".ps1", ".vbs")
            )
            for s in [nm_i_low] + [
                str(f_it.get("path") or f_it.get("name") or "").lower() if isinstance(f_it, Mapping) else str(f_it).lower()
                for f_it in fl_i
            ]
        )
        is_trap = (
            has_double_ext_dropper
            or bool(sig.get("hard_dropper_trap"))
            or (bool(sig.get("policy2_non_sg_trap")) and not is_sg)
        ) and not is_tiny_stub[i]
        if is_trap or z_jk[i] > 1.5:
            junk_leg[i, 0] = -8.0
            junk_leg[i, 1] = 8.0
        elif is_tiny_stub[i] or z_jk[i] < -1.5:
            junk_leg[i, 0] = 8.0
            junk_leg[i, 1] = -8.0

    # 4. anime: ["non_anime", "anime"]
    an_1d = (an_l[:, 0] if an_l.ndim == 2 else an_l).astype(np.float64)
    anime_leg = np.stack([-0.5 * an_1d, 0.5 * an_1d], axis=1)

    # 5. english_subs: ["no_eng_subs", "has_eng_subs", "unknown"]
    es_leg = np.tile(np.array([[2.5, -1.0, -2.5]], dtype=np.float64), (B, 1))

    # 6. lang: ["english", "non_english", "multi_or_unknown"]
    # In v3.17.0 aud_l: col 0 = non_english, col 1 = english
    en_l = aud_l[:, 1].astype(np.float64)
    ne_l = aud_l[:, 0].astype(np.float64)
    mu_l = np.full(B, -2.5, dtype=np.float64)
    lang_leg = np.stack([en_l, ne_l, mu_l], axis=1)

    # 7. owner_scope: ["allowed_type", "excluded"]
    media_l = np.maximum(c11_l[:, 0], c11_l[:, 1]).astype(np.float64)
    other_l = np.max(c11_l[:, 2:], axis=1).astype(np.float64)
    scope_leg = np.stack([media_l + 1.0, other_l], axis=1)

    # 8. title_language: ["english_title", "non_english_title"]
    tlang_leg = np.stack([en_l, ne_l], axis=1)

    # 9. span_bio: pad [B, L, 3] ("O", "B-TITLE", "I-TITLE") to [B, 256, 7]
    bio_leg = np.full((B, 256, 7), -6.0, dtype=np.float64)
    bio_leg[:, :, 0] = 2.0
    L_in = min(256, bio_l.shape[1])
    bio_leg[:, :L_in, :3] = bio_l[:, :L_in, :3].astype(np.float64)

    return {
        "category": cat_leg,
        "adult": adult_leg,
        "junk": junk_leg,
        "anime": anime_leg,
        "english_subs": es_leg,
        "lang": lang_leg,
        "owner_scope": scope_leg,
        "title_language": tlang_leg,
        "span_bio": bio_leg,
    }
