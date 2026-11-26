"""
Clean-Sheet Single-Pass Consolidated Runtime for Armiger Ultra v3.17.0-rc3.
Zero torch, zero sklearn, zero scipy, zero lightgbm, zero pickle — runs inside
Provides the production 4-plane hybrid classifier runtime:
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from audio_features_p import (
    AUDIO_BYTE_L,
    encode_audio_p_bytes,
    extract_audio_p_extra_20d,
    extract_audio_p_features,
)
from authority_aidx import ExternalAuthorityV317
from canonical_views import (
    policy2_v2_signals,
    project_audio_only,
    view_c_byte_obf_stats,
)
from featurizer_v317 import (
    decode_span_utf8_safe,
    extract_opaque_6d,
    extract_tree_fine_18d,
    featurize_single_row,
)
from hashed_lex_v317 import (
    build_enriched_lexical_text,
    clean_lang_title_p,
    extract_char_only_counts,
    extract_hashed_counts,
    score_linear_sparse_vec,
    score_linear_sparse_vec_T,
)
from safety_plane0_v317 import Plane0SafetyEngine
from tokenizer_v317 import PurePythonUnigramTokenizer

CAT11 = [
    "movie", "tv_show", "music", "game", "software",
    "ebook", "audiobook", "comic", "course", "xxx", "unknown",
]
CAT_TO_IDX = {c: i for i, c in enumerate(CAT11)}
MEDIA_ADMIT_CATS = {"movie", "tv_show"}
_TOK_RE = re.compile(r"[a-z0-9]+", re.I)

_EXPLICIT_ANIME_TOKS = frozenset({
    "anime", "ova", "ona", "oav", "fansub", "horriblesubs",
    "subsplease", "erai", "judas", "ember", "commie", "nyaa", "gekijouban",
    "shingeki", "seiyuu", "animax", "crunchyroll", "funimation",
})
_ANIME_FRANCHISE_PHRASES = (
    "grendizer", "doraemon", "digimon", "pokemon", "pokémon",
    "yu gi oh", "yugioh", "beyblade", "sailor moon", "dragon ball",
)
_NON_JP_ANIM_PHRASES = (
    "russian animation", "soviet animation", "korean animation", "french animation",
    "american animation", "western animation", "british animation", "canadian animation",
    "chinese animation", "donghua", "blender open movie",
)
_CAM_UPLOAD_PREFIXES = (
    "km202", "km_202", "km-202", "youcut202", "youcut_202", "youcut-202",
    "inshot20", "inshot_20", "inshot-20", "1080p4000k", "1080p_4000k",
    "video00", "video_00", "videoplayback", "getvid",
)


def _clean_utf8(s: Any) -> str:
    st = str(s or "")
    try:
        st.encode("utf-8")
        return st
    except UnicodeEncodeError:
        return st.encode("utf-8", "replace").decode("utf-8")


def _sanitize_row(r: dict[str, Any]) -> dict[str, Any]:
    nm = r.get("name", "")
    fl = r.get("files") or []
    need_clean = False
    try:
        str(nm or "").encode("utf-8")
    except UnicodeEncodeError:
        need_clean = True
    if not need_clean:
        for f in fl[:64]:
            fp = f.get("path") or f.get("name") or f.get("p") or "" if isinstance(f, dict) else f
            try:
                str(fp or "").encode("utf-8")
            except UnicodeEncodeError:
                need_clean = True
                break
    if not need_clean:
        return r
    out = dict(r)
    out["name"] = _clean_utf8(nm)
    new_fl = []
    for f in fl:
        if isinstance(f, dict):
            nf = dict(f)
            for k in ("path", "name", "p"):
                if k in nf:
                    nf[k] = _clean_utf8(nf[k])
            new_fl.append(nf)
        else:
            new_fl.append(_clean_utf8(f))
    out["files"] = new_fl
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


def _ort_probs(outs: list[np.ndarray], binary: bool = False) -> np.ndarray:
    arr = np.asarray(outs[-1], dtype=np.float32)
    if binary:
        if arr.ndim == 2:
            return arr[:, 1] if arr.shape[1] >= 2 else arr[:, 0]
        return arr.reshape(-1)
    return arr


def _join_audio_p_text(ct: str, media: list[str]) -> str:
    if not media:
        return ct
    ct_l = ct.lower()
    extras: list[str] = []
    for m in media:
        if m.lower() != ct_l and m not in extras:
            extras.append(m)
        if len(extras) >= 3:
            break
    return ct if not extras else f"{ct} | " + " ; ".join(extras)


def _boost_prob(cp: np.ndarray, target_idx: int, min_p: float = 0.88) -> np.ndarray:
    out = cp.copy()
    cur = float(out[target_idx])
    if cur >= min_p:
        return out
    rem = max(1e-7, 1.0 - cur)
    scale = (1.0 - min_p) / rem
    out *= scale
    out[target_idx] = min_p
    out /= out.sum()
    return out


_CONCERT_KWS = (
    "concert", "live in ", "live at ", "live from ", "mtv unplugged", "unplugged",
    "music video", "official video", "boiler room", "tiny desk", "glastonbury",
    "wacken", "coachella", "woodstock", "rock am ring", "symphony", "philharmonic",
    "opera ", "ballet ", "tour 19", "tour 20", "월드투어", "演唱会", "ライヴ", "ライブ",
)


def _post_resolve_category(
    nm: str,
    fl: list[Any],
    sz: int,
    cp_in: np.ndarray,
    s1_p: np.ndarray,
    tab_p: np.ndarray,
    tsum: np.ndarray,
    au: np.ndarray,
    tf: np.ndarray,
    opq: np.ndarray,
    l_cx: float,
    l_xxx: float,
) -> tuple[int, np.ndarray]:
    idx_m, idx_tv, idx_mus, idx_gm, idx_sw, idx_eb, idx_ab, idx_com, idx_crs, idx_xxx, idx_unk = range(11)
    cp = cp_in.copy()
    pred_i = int(np.argmax(cp))
    nl = (nm or "").lower()
    p_xxx_3way = 0.45 * float(cp[idx_xxx]) + 0.25 * float(s1_p[idx_xxx]) + 0.30 * float(tab_p[idx_xxx])
    not_strong_non_xxx_auth = not (au[2] > 0.5 and au[5] < 0.5 and au[18] == 0.0 and au[6] >= 3.5)

    # 1. Strict opaque (pure hex / warc / sql dump) -> unknown (unless xxx)
    if (opq[0] > 0.5 or opq[1] > 0.5) and pred_i != idx_xxx:
        cp = _boost_prob(cp, idx_unk, 0.90)
        return idx_unk, cp

    # 2. Unknown false-abstention gate: rescue adult rows & borderline known rows while keeping genuine unknown recall >= 60%
    if pred_i == idx_unk:
        top10_i = int(np.argmax(cp[:10]))
        top10_p = float(cp[top10_i])
        unk_p = float(cp[idx_unk])
        if (
            p_xxx_3way >= 0.22
            or float(cp[idx_xxx]) >= 0.18
            or float(s1_p[idx_xxx]) >= 0.30
            or (l_xxx > 0.35 and float(cp[idx_xxx]) >= 0.08)
            or tf[16] > 0.5
        ):
            cp[idx_unk] = min(unk_p, 0.05)
            cp = _boost_prob(cp, idx_xxx, 0.80)
            return idx_xxx, cp
        is_mf = len(fl) >= 2
        tau_u = 0.58 if is_mf else 0.50
        tau_m = 0.18 if is_mf else 0.12
        if opq[2] < 0.5 and (unk_p < tau_u or (unk_p - top10_p) < tau_m):
            cp[idx_unk] = max(0.01, top10_p * 0.5)
            cp /= cp.sum()
            pred_i = int(np.argmax(cp[:10]))

    # 3. 3-Way (Meta + Stage-1 + Tabular) Adult/XXX Disambiguation (protects SFW .cbz/comics & high-score IMDb titles!)
    if pred_i != idx_xxx and pred_i != idx_unk and tf[12] < 0.5 and not_strong_non_xxx_auth:
        if pred_i in (idx_com, idx_eb):
            if (
                tf[16] > 0.5
                or float(s1_p[idx_xxx]) >= 0.32
                or (p_xxx_3way >= 0.26 and float(s1_p[idx_xxx]) >= 0.12)
                or (l_cx > 1.80 and float(cp[idx_xxx]) >= 0.12 and float(s1_p[idx_xxx]) >= 0.10)
            ):
                cp = _boost_prob(cp, idx_xxx, 0.85)
                return idx_xxx, cp
        else:
            if (
                tf[16] > 0.5
                or (p_xxx_3way >= 0.25 and (float(cp[idx_xxx]) >= 0.09 or float(s1_p[idx_xxx]) >= 0.14))
                or (float(cp[idx_xxx]) >= 0.28 and float(s1_p[idx_xxx]) >= 0.18)
            ):
                cp = _boost_prob(cp, idx_xxx, 0.85)
                return idx_xxx, cp

    # 4. High-precision Structural Packaging & External Authority (.aidx + ZCB3 + s3 + ext_games)
    top_p = float(cp[pred_i])
    mw = math.expm1(max(0.0, float(au[8])))
    tvw = math.expm1(max(0.0, float(au[9])))
    gw = math.expm1(max(0.0, float(au[10])))
    sw = math.expm1(max(0.0, float(au[12])))
    score = math.expm1(max(0.0, float(au[6])))
    nt = int(round(float(au[1]) * 8.0))
    is_ex = au[2] > 0.5
    is_dict = au[3] > 0.5
    yr_ex = au[4] > 0.5
    yr_conf = au[5] > 0.5
    has_ep = (tf[0] > 0.5 or tf[1] > 0.1 or tf[2] > 0.15 or tf[5] > 0.5 or tf[6] > 0.5)

    # 4a. Pure-video (>1.2 GB DVD/BDMV/MKV with 0 audio files & no concert keywords) Music -> Movie/TV rescue
    if (
        pred_i == idx_mus
        and float(tsum[4]) >= 0.85
        and float(tsum[5]) == 0.0
        and sz >= 1_200_000_000
        and float(tab_p[idx_m] + tab_p[idx_tv]) >= 0.35
        and float(au[14]) == 0.0
        and not any(k in nl for k in _CONCERT_KWS)
    ):
        tgt_v = idx_tv if (has_ep or float(tab_p[idx_tv]) > float(tab_p[idx_m])) else idx_m
        cp = _boost_prob(cp, tgt_v, 0.86)
        return tgt_v, cp

    # 4b. Movie <-> TV Show
    if pred_i == idx_m:
        if (
            ((tf[5] > 0.5 or tf[6] > 0.5 or tf[1] > 0.35 or tf[2] >= 0.2) and top_p < 0.94)
            or (au[25] > 0.5 and au[24] < 0.5 and top_p < 0.90)
            or (
                is_ex and not yr_conf and tvw >= 10.0 and mw == 0.0 and gw == 0.0 and sw == 0.0
                and (not is_dict or nt >= 2 or yr_ex)
                and (has_ep or tf[4] > 0.5 or score >= 120.0)
                and top_p < 0.92
            )
        ):
            cp = _boost_prob(cp, idx_tv, 0.86)
            return idx_tv, cp
    elif pred_i == idx_tv:
        if (
            not has_ep and tf[4] < 0.5
            and (
                (au[24] > 0.5 and au[25] < 0.5 and top_p < 0.88)
                or (
                    is_ex and not yr_conf and mw >= 10.0 and (tvw == 0.0 or mw >= 3.5 * tvw)
                    and gw == 0.0 and sw == 0.0 and (not is_dict or nt >= 2 or yr_ex)
                    and top_p < 0.90
                )
            )
        ):
            cp = _boost_prob(cp, idx_m, 0.86)
            return idx_m, cp

    # 4c. Game <-> Software (protect macOS .pkg installers via tf[11] version / sw cues!)
    if pred_i == idx_sw and tf[12] < 0.5:
        has_pkg_only = any(str(f.get("path") if isinstance(f, dict) else f).lower().endswith(".pkg") for f in fl[:16])
        rom_strict = (tf[8] > 0.5) and not (has_pkg_only and (tf[10] > 0.5 or tf[11] > 0.5))
        if (
            ((tf[7] > 0.5 or rom_strict) and tf[10] < 0.5 and top_p < 0.94)
            or (au[36] > 0.5 and sw == 0.0 and tf[10] < 0.5 and top_p < 0.92)
            or (is_ex and not yr_conf and gw >= 10.0 and sw == 0.0 and mw == 0.0 and tvw == 0.0 and (not is_dict or nt >= 2) and tf[10] < 0.5 and top_p < 0.92)
        ):
            cp = _boost_prob(cp, idx_gm, 0.86)
            return idx_gm, cp
    elif pred_i == idx_gm and tf[12] < 0.5 and tf[7] < 0.5 and tf[8] < 0.5:
        if (
            (tf[10] > 0.5 and tf[11] > 0.5 and gw == 0.0 and au[36] < 0.5 and top_p < 0.90)
            or (is_ex and not yr_conf and sw >= 10.0 and gw == 0.0 and au[36] < 0.5 and top_p < 0.90)
        ):
            cp = _boost_prob(cp, idx_sw, 0.86)
            return idx_sw, cp

    # 4d. Comic <-> Ebook
    if pred_i == idx_eb and tf[12] > 0.5:
        cp = _boost_prob(cp, idx_com, 0.88)
        return idx_com, cp
    elif pred_i == idx_com and tf[12] < 0.5:
        if any(ln_k in nl for ln_k in ("[light novel]", "(light novel)", "[ln]", "(ln)", "light novel epub", "web novel", "二次元ドリームノベルズ")):
            cp = _boost_prob(cp, idx_eb, 0.88)
            return idx_eb, cp

    return pred_i, cp


def _resolve_anime_post(nm: str, fl: list[Any], pred_cat_i: int, p_an: float, au: np.ndarray) -> float:
    if pred_cat_i not in (0, 1, 10):
        return min(p_an, 0.12)
    nl = (nm or "").lower()
    w_toks = set(_TOK_RE.findall(nl))
    f_low = " ".join(str(f.get("path") or "") if isinstance(f, dict) else str(f) for f in (fl or [])[:8]).lower()
    all_low = nl + (" " + f_low if f_low else "")
    has_exp = bool(w_toks & _EXPLICIT_ANIME_TOKS) or ("anime" in f_low) or ("fansub" in f_low)
    has_fran = any(ph in nl for ph in _ANIME_FRANCHISE_PHRASES)
    has_non_jp = any(ph in all_low for ph in _NON_JP_ANIM_PHRASES) or any(nl.startswith(pr) for pr in _CAM_UPLOAD_PREFIXES)
    if au[23] > 0.5 or (has_non_jp and not has_exp and au[22] < 0.35 and au[27] < 0.5):
        return min(p_an, 0.12)
    if has_fran:
        return max(p_an, 0.90)
    if au[0] > 0.5 and au[5] < 0.5 and au[2] > 0.5:
        is_dict = au[3] > 0.5
        yr_ex = au[4] > 0.5
        n_toks = int(round(float(au[1]) * 8.0))
        sc = math.expm1(max(0.0, float(au[6])))
        aw = math.expm1(max(0.0, float(au[11])))
        pa_ac = float(au[22])
        if sc >= 35.0 and pa_ac >= 0.85 and aw >= 6.5 and (not is_dict or yr_ex or n_toks >= 3) and p_an >= 0.25:
            return max(p_an, 0.88)
    if (has_exp or au[27] > 0.5) and p_an >= 0.30 and not has_non_jp:
        return max(p_an, 0.88)
    return p_an


class ArmigerUltraV317Runtime:
    """Single-pass 4-Plane Pure-FP32 Runtime (Plane 0 + CandB3 FP32 + ONNX TreeEnsemble + mmap .aidx/ZCB3 + 2-Regime Admission)."""

    def __init__(self, bundle_dir: str | Path, intra_threads: int = 1):
        b = Path(bundle_dir)
        self.cfg = json.loads((b / "v317_runtime_config.json").read_text(encoding="utf-8"))
        self.tok = PurePythonUnigramTokenizer.from_json(b / "tokenizer_v317_32k.json")
        self.auth = ExternalAuthorityV317(
            b / "authority_v317.aidx",
            b / "validated_zero_cue_catalog.zcb3",
            b / "external_catalogs_v317.json",
        )
        self.plane0 = Plane0SafetyEngine()

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        self.sess_s1 = ort.InferenceSession(str(b / "armiger_ultra.fp32.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_tab = ort.InferenceSession(str(b / "tab_only_cat11.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_cat = ort.InferenceSession(str(b / "meta_cat11.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_jk = ort.InferenceSession(str(b / "meta_junk.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_aud = ort.InferenceSession(str(b / "meta_audio_p.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_ani = ort.InferenceSession(str(b / "meta_anime.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_saf = ort.InferenceSession(str(b / "safety_tier2.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

        m_npz = np.load(b / "mahal_state.npz", allow_pickle=False)
        self.pca_mean = m_npz["pca_mean"].astype(np.float32)
        self.pca_comp = m_npz["pca_components"].astype(np.float32)
        self.prec = m_npz["precision"].astype(np.float32)
        self.centroids = m_npz["centroids"].astype(np.float32)
        self.jk_centroid = m_npz["jk_centroid"].astype(np.float32)
        self.mahal_z_mean = m_npz["mahal_z_mean"].astype(np.float32) if "mahal_z_mean" in m_npz else np.zeros(4, dtype=np.float32)
        self.mahal_z_std = m_npz["mahal_z_std"].astype(np.float32) if "mahal_z_std" in m_npz else np.ones(4, dtype=np.float32)

        lw = np.load(b / "lexical_weights_fp16.npz", allow_pickle=False)
        self.raw_W_T = np.ascontiguousarray(lw["raw_W"].T, dtype=np.float32)
        self.raw_b = np.ascontiguousarray(lw["raw_b"], dtype=np.float32)
        self.aud_full_W_T = np.ascontiguousarray(lw["aud_full_W"].T, dtype=np.float32)
        self.aud_full_b = np.ascontiguousarray(lw["aud_full_b"], dtype=np.float32)
        self.aud_pure_W32 = np.ascontiguousarray(lw["aud_pure_W"], dtype=np.float32)
        self.aud_pure_b = float(lw["aud_pure_b"][0])
        self.aud_char_W32 = np.ascontiguousarray(lw["aud_char_W"], dtype=np.float32)
        self.aud_char_b = float(lw["aud_char_b"][0])
        lw.close()

        gz = json.loads((b / "audio_p_gazetteer_compact.json").read_text(encoding="utf-8"))
        self.gaz_order = [
            "anime_dub_only", "anime_dual", "anime_sub_only", "non_anime_dual",
            "non_anime_sub_only", "bad_non_en_dual", "fr_anime", "de_anime", "anime_bd_web",
        ]
        self.gaz_sets = {k: frozenset(gz.get(k, ())) for k in self.gaz_order}
        self.anime_phrases = frozenset(gz.get("anime_phrases", ()))

        self.tau_jk = float(self.cfg["tau_jk"])
        self.tau_saf_lo = float(self.cfg["tau_saf_lo"])
        self.tau_saf_hi = float(self.cfg["tau_saf_hi"])
        self.tau_admit_bare = float(self.cfg["tau_admit_bare"])
        self.tau_admit_mf = float(self.cfg["tau_admit_mf"])
        self.tau_xxx_bare = float(self.cfg.get("tau_xxx_bare", 1.0))
        self.tau_xxx_mf = float(self.cfg.get("tau_xxx_mf", 1.0))
        self.s1_temp_scale = float(self.cfg.get("s1_temp_scale", 1.0))
        self.p_saf_adm_floor = float(self.cfg.get("p_saf_adm_floor", 0.0))
        self.admit_coef = np.asarray(self.cfg["admit_coef"], dtype=np.float32)
        self.admit_intercept = float(self.cfg["admit_intercept"])

    def close(self) -> None:
        self.auth.close()

    def _mahal_4d(self, emb: np.ndarray, pred_cat_idx: np.ndarray) -> np.ndarray:
        z = (emb - self.pca_mean) @ self.pca_comp.T
        prec = self.prec

        def _md(mu: np.ndarray) -> np.ndarray:
            d = z - mu
            return np.sqrt(np.maximum(0.0, np.sum((d @ prec) * d, axis=1))).astype(np.float32)

        all_cl = np.stack([_md(self.centroids[c]) for c in range(9)], axis=1)
        all_11 = np.concatenate([all_cl, _md(self.centroids[9])[:, None], _md(self.centroids[10])[:, None]], axis=1)
        return np.stack([
            all_cl.min(axis=1),
            all_11[np.arange(len(z)), pred_cat_idx],
            all_11[:, 9],
            _md(self.jk_centroid),
        ], axis=1)

    def _gaz_p_feats(self, aud_txts: list[str]) -> np.ndarray:
        out = np.zeros((len(aud_txts), 10), dtype=np.float32)
        for i, txt in enumerate(aud_txts):
            low = (txt or "").lower()
            words = _TOK_RE.findall(low)
            toks = set(words)
            if toks:
                for gi, gk in enumerate(self.gaz_order):
                    if toks & self.gaz_sets[gk]:
                        out[i, gi] = 1.0
            if self.anime_phrases and len(words) >= 2:
                w8 = words[:8]
                hit = 0.0
                for L in (2, 3, 4):
                    for s in range(0, max(0, len(w8) - L + 1)):
                        if " ".join(w8[s : s + L]) in self.anime_phrases:
                            hit = 1.0
                            break
                    if hit > 0:
                        break
                out[i, 9] = hit
        return out

    def _safety_surf_8d(self, rows: list[dict[str, Any]], vcs: list[dict[str, Any]] | None = None) -> np.ndarray:
        out = np.zeros((len(rows), 8), dtype=np.float32)
        for i, r in enumerate(rows):
            nm = str(r.get("name") or "")
            vc = vcs[i] if vcs is not None else view_c_byte_obf_stats(nm)
            ascii_letters = sum(1 for c in nm if ("a" <= c.lower() <= "z"))
            toks = [t for t in nm.replace(".", " ").replace("_", " ").replace("-", " ").split() if t]
            short_alnum = sum(1 for t in toks if 2 <= len(t) <= 6 and any(c.isdigit() for c in t) and any(c.isalpha() for c in t))
            out[i, 0] = 1.0 if vc["byte_len"] < 20 else 0.0
            out[i, 1] = 1.0 if vc["byte_len"] < 35 else 0.0
            out[i, 2] = 1.0 if ascii_letters == 0 else 0.0
            out[i, 3] = vc["digit_ratio"]
            out[i, 4] = vc["non_ascii_ratio"]
            out[i, 5] = vc["mixed_script_flag"]
            out[i, 6] = float(short_alnum) / max(1, len(toks))
            out[i, 7] = vc["punct_ratio"]
        return out

    def classify_batch(self, rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Single-pass end-to-end 4-plane inference emitting consumer-compatible 8-head contract."""
        M = len(rows_in)
        if M == 0:
            return []
        rows = [_sanitize_row(r) for r in rows_in]

        # Plane 0: Deterministic full-input security scan
        p0_list = self.plane0.inspect_plane0_batch(rows)

        # Strictly P(x)-only audio projection computed ONCE per row & reused in featurize_single_row!
        px_pairs = [project_audio_only(r.get("name", ""), r.get("files") or []) for r in rows]
        aud_txts = [_join_audio_p_text(ct, mn) for ct, mn in px_pairs]
        feats = [featurize_single_row(r, self.tok, aud_txt=aud_txts[i]) for i, r in enumerate(rows)]
        sub_b = np.stack([x["subword_ids"] for x in feats])
        aud_b = np.stack([x["audio_subword_ids"] for x in feats])
        byt_b = np.stack([x["byte_ids"] for x in feats])
        ftok_b = np.stack([x["file_tok_ids"] for x in feats])
        fnum_b = np.stack([x["file_num_feats"] for x in feats])
        fmsk_b = np.stack([x["file_mask"] for x in feats])
        tsum_b = np.stack([x["tree_summary"] for x in feats])
        ab_b = np.stack([encode_audio_p_bytes(t, AUDIO_BYTE_L) for t in aud_txts])
        ap_b = np.stack([extract_audio_p_features(t) for t in aud_txts])

        # Plane 1: Pure-FP32 ONNX forward with dynamic T/Ta/L/La/F trim + bare/MF split
        def _run_s1_sub(idxs: np.ndarray) -> tuple[np.ndarray, ...]:
            sb = sub_b[idxs]
            aub = aud_b[idxs]
            byb = byt_b[idxs]
            abb = ab_b[idxs]
            apb = ap_b[idxs]
            tsb = tsum_b[idxs]
            fmb = fmsk_b[idxs]
            nz_t = np.where(sb != 0)
            mt = max(2, int(nz_t[1].max()) + 1) if len(nz_t[1]) else 2
            nz_ta = np.where(aub != 0)
            mta = max(2, int(nz_ta[1].max()) + 1) if len(nz_ta[1]) else 2
            nz_l = np.where(byb != 0)
            ml = min(byb.shape[1], max(32, (int(nz_l[1].max()) + 17) if len(nz_l[1]) else 32))
            nz_la = np.where(abb != 0)
            mla = min(abb.shape[1], max(32, (int(nz_la[1].max()) + 17) if len(nz_la[1]) else 32))
            mf = int(fmb.sum(axis=1).max())
            if mf > 0:
                ft_in = ftok_b[idxs, :mf, :]
                fn_in = fnum_b[idxs, :mf, :]
                fm_in = fmb[:, :mf]
            else:
                k = len(idxs)
                ft_in = np.zeros((k, 0, 8), dtype=np.int64)
                fn_in = np.zeros((k, 0, 12), dtype=np.float32)
                fm_in = np.zeros((k, 0), dtype=np.float32)
            return tuple(self.sess_s1.run(None, {
                "subword_ids": sb[:, :mt], "audio_subword_ids": aub[:, :mta],
                "audio_byte_ids": abb[:, :mla], "audio_p_feats": apb,
                "byte_ids": byb[:, :ml], "file_tok_ids": ft_in,
                "file_num_feats": fn_in, "file_mask": fm_in,
                "tree_summary": tsb,
            }))

        f_counts = fmsk_b.sum(axis=1)
        buckets = [
            np.where(f_counts == 0)[0],
            np.where((f_counts >= 1) & (f_counts <= 8))[0],
            np.where(f_counts > 8)[0],
        ]
        non_empty = [ix for ix in buckets if len(ix) > 0]
        if len(non_empty) == 1:
            c11_l, jk2_l, sf_l, an_l, aud_l, bio_l, ptr_l, emb_b = _run_s1_sub(non_empty[0])
        else:
            outs_list = [(ix, _run_s1_sub(ix)) for ix in non_empty]
            o0 = outs_list[0][1]
            max_L_out = max(oo[5].shape[1] for _, oo in outs_list)
            c11_l = np.empty((M, o0[0].shape[1]), dtype=np.float32)
            jk2_l = np.empty((M, o0[1].shape[1]), dtype=np.float32)
            sf_l = np.empty((M,) + o0[2].shape[1:], dtype=np.float32)
            an_l = np.empty((M,) + o0[3].shape[1:], dtype=np.float32)
            aud_l = np.empty((M, o0[4].shape[1]), dtype=np.float32)
            bio_l = np.zeros((M, max_L_out, o0[5].shape[2]), dtype=np.float32)
            ptr_l = np.full((M, max_L_out, o0[6].shape[2]), -1e4, dtype=np.float32)
            emb_b = np.empty((M, o0[7].shape[1]), dtype=np.float32)
            for ix_s, oo in outs_list:
                c11_l[ix_s] = oo[0]; jk2_l[ix_s] = oo[1]; sf_l[ix_s] = oo[2]
                an_l[ix_s] = oo[3]; aud_l[ix_s] = oo[4]; emb_b[ix_s] = oo[7]
                lo = oo[5].shape[1]
                bio_l[ix_s, :lo, :] = oo[5]
                ptr_l[ix_s, :lo, :] = oo[6]
        if an_l.ndim == 1:
            an_l_col = an_l[:, None].astype(np.float32)
        else:
            an_l_col = an_l.astype(np.float32)
        sf_1d = sf_l[:, 0].astype(np.float32) if sf_l.ndim == 2 else sf_l.astype(np.float32)

        # Plane 2: Structural + External Authority (.aidx + ZCB3 + s3 + ext_games) + Single-Matmul FP16 Lexical!
        tf_l, opq_l, au38_l, p2_l, lex_raw_l = [], [], [], [], []
        ap_ext_l, ap_auth_l, lex_aud_full_l, lex_aud_pure_l, lex_aud_char_l = [], [], [], [], []
        p2_sigs = []

        for i, r in enumerate(rows):
            nm = r.get("name", "")
            fl = r.get("files") or []
            sz = int(r.get("size") or 0)
            tf_l.append(extract_tree_fine_18d(nm, fl))
            opq_l.append(extract_opaque_6d(nm, fl))
            au38_l.append(self.auth.extract_full_authority_38d(nm, fl))
            sig = feats[i]["p2"]
            p2_sigs.append(sig)
            p2_l.append([
                1.0 if sig["hard_dropper_trap"] else 0.0,
                1.0 if sig["policy2_non_sg_trap"] else 0.0,
                1.0 if sig["game_demotion_eligible"] else 0.0,
                1.0 if sig["pure_chm_only"] else 0.0,
                1.0 if sig["has_ia"] else 0.0,
                float(sig["game_asset_byte_ratio"]),
                float(sig["exec_byte_ratio"]),
                float(sig["archive_byte_ratio"]),
            ])
            ltxt = build_enriched_lexical_text(nm, fl, sz)
            lex_raw_l.append(score_linear_sparse_vec_T(extract_hashed_counts(ltxt), self.raw_W_T, self.raw_b))

            # Strictly P(x)-only audio signals (ZERO non-P(x) inputs!)
            ct, mn = px_pairs[i]
            ap_ext_l.append(extract_audio_p_extra_20d(ct, mn))
            ap_auth_l.append(self.auth.extract_audio_p_authority_12d(ct))
            lex_aud_full_l.append(score_linear_sparse_vec_T(extract_hashed_counts(aud_txts[i]), self.aud_full_W_T, self.aud_full_b))
            ct_pure = clean_lang_title_p(ct)
            lex_aud_pure_l.append(score_linear_sparse_vec_T(extract_hashed_counts(ct_pure), self.aud_pure_W32, self.aud_pure_b))
            lex_aud_char_l.append(score_linear_sparse_vec_T(extract_char_only_counts(ct_pure), self.aud_char_W32, self.aud_char_b))

        tf_b = np.stack(tf_l).astype(np.float32)
        opq_b = np.stack(opq_l).astype(np.float32)
        au38_b = np.stack(au38_l).astype(np.float32)
        p2_b = np.asarray(p2_l, dtype=np.float32)
        lex_raw_b = np.stack(lex_raw_l).astype(np.float32)  # [M, 20]: 0..17=RC2 exact, 18=unk_clean, 19=lex_cx

        x_tab = np.concatenate([tsum_b, tf_b, opq_b, au38_b, p2_b], axis=1).astype(np.float32)  # 118d
        # meta_junk.onnx uses exact 150d RC2 feature layout (cols 0..17 of lex_raw_b)
        x_meta_jk = np.concatenate([c11_l, jk2_l, an_l_col, x_tab, lex_raw_b[:, :18]], axis=1).astype(np.float32)  # 150d
        # meta_cat11.onnx uses 151d RC3 feature layout (col 15 replaced by unk_clean [col 18], plus lex_cx [col 19])
        lex_cat19 = np.concatenate([
            lex_raw_b[:, :15],
            lex_raw_b[:, 18:19],
            lex_raw_b[:, 16:18],
            lex_raw_b[:, 19:20],
        ], axis=1).astype(np.float32)
        x_meta_cat = np.concatenate([c11_l, jk2_l, an_l_col, x_tab, lex_cat19], axis=1).astype(np.float32)  # 151d

        ap_ext_b = np.stack(ap_ext_l).astype(np.float32)
        ap_auth_b = np.stack(ap_auth_l).astype(np.float32)
        ap_gaz_b = self._gaz_p_feats(aud_txts)
        laf_b = np.stack(lex_aud_full_l).astype(np.float32)  # [M, 2]: [aud_full, clean_anime_px]
        lap_b = np.asarray(lex_aud_pure_l, dtype=np.float32)[:, None]
        lac_b = np.asarray(lex_aud_char_l, dtype=np.float32)[:, None]
        x_aud = np.concatenate([
            aud_l, ap_b, ap_ext_b, ap_auth_b, ap_gaz_b,
            laf_b[:, 0:1], lap_b, lac_b, laf_b[:, 1:2],
        ], axis=1).astype(np.float32)  # 72d strictly P(x)-only!

        x_ani = np.concatenate([
            an_l_col, c11_l, au38_b, ap_gaz_b,
            laf_b[:, 1:2], tsum_b, tf_b,
        ], axis=1).astype(np.float32)  # 128d Stage-2 Anime GBDT!

        tab_p = _ort_probs(self.sess_tab.run(None, {"X": x_tab}), binary=False)
        m_cat_p = _ort_probs(self.sess_cat.run(None, {"X": x_meta_cat}), binary=False)
        m_jk_p = _ort_probs(self.sess_jk.run(None, {"X": x_meta_jk}), binary=True)
        m_aud_en = _ort_probs(self.sess_aud.run(None, {"X": x_aud}), binary=True)
        m_ani_p = _ort_probs(self.sess_ani.run(None, {"X": x_ani}), binary=True)

        # Epistemic uncertainty + Mahalanobis + Tier-2 Safety
        c11_p = _softmax(c11_l / self.s1_temp_scale if self.s1_temp_scale != 1.0 else c11_l)
        ent_src = m_cat_p if self.cfg.get("adm_use_meta_uncertainty", False) else c11_p
        c_ent = -np.sum(ent_src * np.log(np.clip(ent_src, 1e-7, 1.0)), axis=1).astype(np.float32)
        t2 = np.partition(ent_src, -2, axis=1)[:, -2:]
        c_mar = (t2[:, 1] - t2[:, 0]).astype(np.float32)
        m_mix = 0.5 * (np.clip(c11_p, 1e-7, 1.0) + np.clip(tab_p, 1e-7, 1.0))
        js_b = (0.5 * (
            np.sum(np.clip(c11_p, 1e-7, 1.0) * np.log(np.clip(c11_p, 1e-7, 1.0) / m_mix), axis=1)
            + np.sum(np.clip(tab_p, 1e-7, 1.0) * np.log(np.clip(tab_p, 1e-7, 1.0) / m_mix), axis=1)
        )).astype(np.float32)
        mah_b = self._mahal_4d(emb_b, m_cat_p.argmax(axis=1))

        surf_b = self._safety_surf_8d(rows, [x["vc"] for x in feats])
        x_saf = np.concatenate([
            m_cat_p, m_jk_p[:, None], sf_1d[:, None], jk2_l, mah_b,
            tsum_b[:, [0, 1, 2, 3, 4, 5, 7, 10, 11, 23, 42, 43, 44, 45, 46, 47]],
            au38_b[:, [0, 6, 14, 18]], surf_b,
        ], axis=1).astype(np.float32)  # 47d
        p_saf_b = _ort_probs(self.sess_saf.run(None, {"X": x_saf}), binary=True)

        idx_game, idx_sw, idx_eb, idx_xxx, idx_unk = 3, 4, 5, 9, 10
        outputs: list[dict[str, Any]] = []
        for i, r in enumerate(rows):
            nm_i = r.get("name", "")
            fl_i = r.get("files") or []
            pred_cat_i, cp = _post_resolve_category(
                nm_i, fl_i, int(r.get("size") or 0), m_cat_p[i], c11_p[i], tab_p[i], tsum_b[i],
                au38_b[i], tf_b[i], opq_b[i],
                float(lex_raw_b[i, 19]), float(lex_raw_b[i, 14]),
            )
            pred_cat = CAT11[pred_cat_i]

            sig = p2_sigs[i]
            p_jk = float(m_jk_p[i])
            is_sg = (pred_cat_i in (idx_game, idx_sw)) or (float(cp[idx_game] + cp[idx_sw]) >= 0.45)
            is_pure_chm_eb = (pred_cat_i == idx_eb) and bool(sig["pure_chm_only"])
            if sig["hard_dropper_trap"]:
                p_jk = max(p_jk, 0.999)
                pred_is_junk = True
            elif sig["policy2_non_sg_trap"] and (not is_sg) and (not is_pure_chm_eb):
                p_jk = max(p_jk, 0.995)
                pred_is_junk = True
            else:
                pred_is_junk = p_jk >= self.tau_jk

            # Monotonic Safety Precedence Lattice (Plane 0 NEVER downgraded!)
            is_clean_ctx = (float(cp[2] + cp[5] + cp[6] + cp[0] + cp[1]) >= 0.85) and (float(cp[idx_xxx]) < 0.05)
            saf_fin = self.plane0.fuse_monotonic_lattice(
                p0_list[i], float(p_saf_b[i]),
                tau_hi=self.tau_saf_hi, tau_lo=self.tau_saf_lo,
                is_clean_creative_context=is_clean_ctx,
            )
            saf_action = saf_fin["action"]
            if saf_action in ("hard_block_tombstone", "local_hard_block"):
                pred_is_junk = True
                p_jk = max(p_jk, 0.9999)

            saf_blocked = saf_action != "safe"
            p_en = float(np.clip(m_aud_en[i], 0.0, 1.0))
            p_ne = 1.0 - p_en
            pred_lang = "english" if p_en >= 0.50 else "non_english"
            p_an_raw = float(np.clip(m_ani_p[i], 0.0, 1.0))
            p_an = float(np.clip(_resolve_anime_post(nm_i, fl_i, pred_cat_i, p_an_raw, au38_b[i]), 0.0, 1.0))
            p_ad = float(np.clip(cp[idx_xxx], 0.0, 1.0))
            p_scope_ok = float(np.clip(cp[0] + cp[1], 0.0, 1.0))

            # Unconditional UTF-8 codepoint & whole-word boundary-snapping title span decoder
            title_str, s_b, e_b, utf8_ok = decode_span_utf8_safe(nm_i, bio_l[i], ptr_l[i])

            # 2-Regime Calibrated Selective Admission
            is_bare = 1.0 if len(fl_i) == 0 else 0.0
            p_saf_adm = max(float(p_saf_b[i]), self.p_saf_adm_floor) if self.p_saf_adm_floor > 0.0 else float(p_saf_b[i])
            x_adm = np.array([
                float(cp[0] + cp[1]), float(cp[idx_xxx]), float(cp[idx_unk]),
                p_jk, p_saf_adm, float(js_b[i]),
                float((mah_b[i, 0] - self.mahal_z_mean[0]) / max(1e-6, float(self.mahal_z_std[0]))),
                float((mah_b[i, 2] - self.mahal_z_mean[2]) / max(1e-6, float(self.mahal_z_std[2]))),
                float(c_ent[i]), float(c_mar[i]),
                float(au38_b[i, 0]), is_bare,
            ], dtype=np.float32)
            admit_score = _sigmoid(float(np.dot(self.admit_coef, x_adm) + self.admit_intercept))
            tau_adm = self.tau_admit_bare if is_bare > 0.5 else self.tau_admit_mf
            tau_xxx = self.tau_xxx_bare if is_bare > 0.5 else self.tau_xxx_mf
            p_xxx_eff = 0.75 * float(cp[idx_xxx]) + 0.25 * float(c11_p[i, idx_xxx])

            if saf_blocked or pred_is_junk or pred_cat not in MEDIA_ADMIT_CATS:
                outcome = "blocked" if (saf_blocked or pred_is_junk or pred_cat == "xxx") else "excluded_type"
                admitted = False
            elif (admit_score >= tau_adm) and (p_xxx_eff <= tau_xxx):
                outcome = "admitted"
                admitted = True
            else:
                outcome = "abstained"
                admitted = False

            cat_probs = {CAT11[k]: round(float(cp[k]), 6) for k in range(11)}
            top_k = CAT11[int(np.argmax(cp))]
            cat_probs[top_k] = round(cat_probs[top_k] + (1.0 - sum(cat_probs.values())), 6)
            p_jk_c = float(np.clip(p_jk, 0.0, 1.0))

            outputs.append({
                "category": pred_cat,
                "junk": "junk" if pred_is_junk else "clean",
                "lang": pred_lang,
                "adult": "adult" if p_ad >= 0.5 else "non_adult",
                "anime": "anime" if p_an >= 0.5 else "non_anime",
                "owner_scope": "allowed_type" if pred_cat in MEDIA_ADMIT_CATS else "excluded_type",
                "admission": {
                    "outcome": outcome,
                    "admitted": admitted,
                    "score": round(admit_score, 6),
                    "regime": "bare" if is_bare > 0.5 else "with_files",
                    "threshold": round(tau_adm, 6),
                },
                "safety": {
                    "action": saf_action,
                    "level": saf_fin["level"],
                    "rule_code": saf_fin["rule_code"],
                    "incomplete_inspection": saf_fin["incomplete_inspection"],
                    "is_safe": not saf_blocked,
                    "overall_safe_to_index": (not saf_blocked) and (not pred_is_junk) and (pred_cat != "xxx"),
                },
                "span_bio": {
                    "title": title_str,
                    "title_byte_span": (s_b, e_b),
                    "utf8_valid": utf8_ok,
                },
                "heads": {
                    "category": {"label": pred_cat, "probs": cat_probs},
                    "adult": {"label": "adult" if p_ad >= 0.5 else "non_adult", "probs": {"non_adult": round(1.0 - p_ad, 6), "adult": round(p_ad, 6)}},
                    "junk": {"label": "junk" if pred_is_junk else "clean", "probs": {"clean": round(1.0 - p_jk_c, 6), "junk": round(p_jk_c, 6)}},
                    "anime": {"label": "anime" if p_an >= 0.5 else "non_anime", "probs": {"non_anime": round(1.0 - p_an, 6), "anime": round(p_an, 6)}},
                    "english_subs": {"label": "no_eng_subs", "probs": {"no_eng_subs": 1.0, "has_eng_subs": 0.0}},
                    "lang": {"label": pred_lang, "probs": {"english": round(p_en, 6), "non_english": round(p_ne, 6)}},
                    "owner_scope": {"label": "allowed_type" if pred_cat in MEDIA_ADMIT_CATS else "excluded_type", "probs": {"allowed_type": round(p_scope_ok, 6), "excluded_type": round(1.0 - p_scope_ok, 6)}},
                    "title_language": {"label": "english_title" if p_en >= 0.5 else "non_english_title", "probs": {"english_title": round(p_en, 6), "non_english_title": round(p_ne, 6)}},
                },
            })
        return outputs
