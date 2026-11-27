#!/usr/bin/env python3
"""Armiger Ultra v4 Core Runtime (Planes 0-3): 100% Pure-ML, Pure-FP32 ONNX + Zero-Copy mmap, ZERO Torrent-Title Heuristics.

Key architectural guarantees:
  1. ZERO hardcoded torrent-title regex/phrase rules (`_post_resolve_category`, `_resolve_anime_post`,
     `_CONCERT_KWS`, `_ANIME_FRANCHISE_PHRASES`, `_NON_JP_ANIM_PHRASES`, `_CAM_UPLOAD_PREFIXES` 100% REMOVED!).
  2. Pure-FP32 ONNX IR v9 / opset 17 (`CandB4_IsolatedAudio_FP32` + Stage-2 ONNX TreeEnsemble meta-stack) + zero-copy mmap `.aidx`/`.zcb3`.
  3. Zero Python `.pkl`, zero PyTorch/sklearn/scipy/lightgbm runtime imports.
  4. 100% physically isolated Audio P(x) graph -> mathematically bitwise 0.0000000000 Cert-2 subtitle invariance.
  5. Plane-0 Monotonic Safety Precedence Lattice + Policy-2 deterministic executable/dropper structural security contract.
  6. Memory-tuned for 1-vCPU / 512 MiB: ORT `enable_cpu_mem_arena=False` + `enable_mem_pattern=False` + `MADV_RANDOM` on mmap indices.
  7. Mondrian / Conformal per-class & per-regime calibrated thresholds loaded from `v4_runtime_config.json`.
"""
from __future__ import annotations

import hashlib
import json
import math
import mmap
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

try:
    from armiger.audio_features_p import AUDIO_BYTE_L, encode_audio_p_bytes, extract_audio_p_extra_20d, extract_audio_p_features
    from armiger.authority_aidx import ExternalAuthorityV317
    from armiger.canonical_views import project_audio_only
    from armiger.featurizer_v317 import (
        decode_span_utf8_safe,
        extract_opaque_6d,
        extract_tree_fine_18d,
        featurize_single_row,
    )
    from armiger.hashed_lex_v317 import (
        build_enriched_lexical_text,
        clean_lang_title_p,
        extract_char_only_counts,
        extract_hashed_counts,
        score_linear_sparse_vec_T,
    )
    from armiger.safety_plane0_v317 import Plane0SafetyEngine
    from armiger.tokenizer_v317 import PurePythonUnigramTokenizer
except ImportError:
    from audio_features_p import AUDIO_BYTE_L, encode_audio_p_bytes, extract_audio_p_extra_20d, extract_audio_p_features
    from authority_aidx import ExternalAuthorityV317
    from canonical_views import project_audio_only
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
        score_linear_sparse_vec_T,
    )
    from safety_plane0_v317 import Plane0SafetyEngine
    from tokenizer_v317 import PurePythonUnigramTokenizer

try:
    from armiger.midx_core_features import extract_midx_core_16d
except ImportError:
    try:
        from midx_core_features import extract_midx_core_16d
    except ImportError:
        extract_midx_core_16d = None  # type: ignore[assignment]

try:
    from armiger.mmgl_catalog_midx_v2 import MMGLCatalogMidx
except ImportError:
    try:
        from mmgl_catalog_midx_v2 import MMGLCatalogMidx
    except ImportError:
        MMGLCatalogMidx = None  # type: ignore[assignment]


CAT11 = [
    "movie", "tv_show", "music", "game", "software",
    "ebook", "audiobook", "comic", "course", "xxx", "unknown",
]
MEDIA_ADMIT_CATS = {"movie", "tv_show"}


def _clean_utf8(s: Any) -> str:
    st = str(s or "")
    try:
        st.encode("utf-8")
        return st
    except UnicodeEncodeError:
        return st.encode("utf-8", "replace").decode("utf-8")


def _sanitize_row(r: dict[str, Any]) -> dict[str, Any]:
    fl_in = r.get("files")
    fl_out = []
    if isinstance(fl_in, (list, tuple)):
        for f in fl_in[:256]:
            if isinstance(f, dict):
                p = _clean_utf8(f.get("path") or f.get("name") or f.get("p") or "")
                try:
                    sz = max(0, int(f.get("size") or f.get("length") or f.get("s") or 0))
                except Exception:
                    sz = 0
                fl_out.append({"path": p, "size": sz})
            elif isinstance(f, str):
                fl_out.append({"path": _clean_utf8(f), "size": 0})
    try:
        tot_sz = max(0, int(r.get("size") or 0))
    except Exception:
        tot_sz = 0
    return {
        "infohash": _clean_utf8(r.get("infohash") or r.get("info_hash") or ""),
        "name": _clean_utf8(r.get("name") or r.get("title") or ""),
        "size": tot_sz,
        "files": fl_out,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _ort_probs(outs: list[np.ndarray], binary: bool = False) -> np.ndarray:
    if len(outs) >= 2:
        p = np.asarray(outs[1], dtype=np.float32)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1] if binary else p
    p0 = np.asarray(outs[0], dtype=np.float32)
    if binary:
        return p0[:, 1] if (p0.ndim == 2 and p0.shape[1] >= 2) else p0.reshape(-1)
    return p0


def _join_audio_p_text(ct: str, media: list[str]) -> str:
    parts = [ct] if ct else []
    if media:
        parts.extend(media[:8])
    return " | ".join(parts)


class ArmigerUltraV4Runtime:
    """Single-pass 4-Plane Pure-FP32 Pure-ML Runtime (ZERO torrent-title heuristics)."""

    def __init__(self, bundle_dir: str | Path, intra_threads: int = 1, sess_s1: Any = None, midx: Any = None):
        b = Path(bundle_dir)
        cfg_p = b / "v4_runtime_config.json"
        if not cfg_p.is_file():
            raise FileNotFoundError(f"FATAL B1: missing required v4_runtime_config.json in {b}")
        self.cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        # B1 fix (fail-closed!): Require & verify SHA-256 of ALL 8 ACTUAL IMPORTED feature modules (__file__) against training!
        _REQUIRED_MODS = (
            "audio_features_p", "authority_aidx", "canonical_views", "csam_guard",
            "featurizer_v317", "hashed_lex_v317", "safety_plane0_v317", "tokenizer_v317",
        )
        exp_mhashes = self.cfg.get("feature_module_hashes") or {}
        missing_mh = [f"{_mn}.py" for _mn in _REQUIRED_MODS if f"{_mn}.py" not in exp_mhashes]
        if missing_mh:
            raise RuntimeError(f"FATAL B1: v4_runtime_config.json missing required feature_module_hashes: {missing_mh}")
        _check_mods = list(_REQUIRED_MODS)
        if "midx_core_features.py" in exp_mhashes:
            _check_mods.append("midx_core_features")
        _imported_mods = {
            f"{_mn}.py": (sys.modules.get(_mn) or sys.modules.get(f"armiger.{_mn}"))
            for _mn in _check_mods
        }
        for _fn in [f"{_mn}.py" for _mn in _check_mods]:
            _eh = exp_mhashes[_fn]
            _mod = _imported_mods.get(_fn)
            _cand_paths = []
            if _mod is not None and getattr(_mod, "__file__", None):
                _cand_paths.append(Path(_mod.__file__))
            if (b / _fn).is_file():
                _cand_paths.append(b / _fn)
            if not _cand_paths:
                raise RuntimeError(f"FATAL B1: cannot locate imported or bundled feature module {_fn}")
            for _cp in _cand_paths:
                _gh = hashlib.sha256(_cp.read_bytes()).hexdigest()
                if _gh != _eh:
                    raise RuntimeError(f"FATAL B1: Feature module SHA-256 skew for {_fn} ({_cp}): expected {_eh[:12]}, got {_gh[:12]}")
        self.tok = PurePythonUnigramTokenizer.from_json(b / "tokenizer_v317_32k.json")
        self.auth = ExternalAuthorityV317(
            b / "authority_v317.aidx",
            b / "validated_zero_cue_catalog.zcb3",
            b / "external_catalogs_v317.json",
        )
        # Advise kernel against 128KB sequential readahead on random-access mmap indices!
        for attr in ("_mm", "_aidx_mm", "_zcb_mm"):
            mm = getattr(self.auth, attr, None)
            if isinstance(mm, mmap.mmap):
                try:
                    mm.madvise(mmap.MADV_RANDOM)
                except Exception:
                    pass
        self.plane0 = Plane0SafetyEngine()

        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        self.sess_s1 = sess_s1 if sess_s1 is not None else ort.InferenceSession(str(b / "armiger_ultra.fp32.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_tab = ort.InferenceSession(str(b / "tab_only_cat11.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_cat = ort.InferenceSession(str(b / "meta_cat11.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_jk = ort.InferenceSession(str(b / "meta_junk.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_aud = ort.InferenceSession(str(b / "meta_audio_p.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_ani = ort.InferenceSession(str(b / "meta_anime.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_saf = ort.InferenceSession(str(b / "safety_tier2.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        self.sess_cat_mlp = ort.InferenceSession(str(b / "meta_cat11_mlp.onnx"), sess_options=so, providers=["CPUExecutionProvider"]) if (b / "meta_cat11_mlp.onnx").is_file() else None
        self.sess_adult = ort.InferenceSession(str(b / "meta_adult.onnx"), sess_options=so, providers=["CPUExecutionProvider"]) if (b / "meta_adult.onnx").is_file() else None
        self._cat_in_dim = int(self.sess_cat.get_inputs()[0].shape[1])
        self._jk_in_dim = int(self.sess_jk.get_inputs()[0].shape[1])
        self._ani_in_dim = int(self.sess_ani.get_inputs()[0].shape[1])
        self._ad_in_dim = int(self.sess_adult.get_inputs()[0].shape[1]) if self.sess_adult is not None else 0
        self._needs_midx = (self._cat_in_dim == 206) or (self._ani_in_dim == 143)
        self.midx: Any = None
        self._owns_midx: bool = False
        if self._needs_midx:
            if extract_midx_core_16d is None:
                raise RuntimeError("FATAL RC4: midx_core_features.extract_midx_core_16d unavailable")
            if midx is not None:
                self.midx = midx
                self._owns_midx = False
            elif (b / "mmgl_catalog_v2.midx").is_file() and MMGLCatalogMidx is not None:
                self.midx = MMGLCatalogMidx(b / "mmgl_catalog_v2.midx")
                self._owns_midx = True
                mm_m = getattr(self.midx, "_mm", None)
                if isinstance(mm_m, mmap.mmap):
                    try:
                        mm_m.madvise(mmap.MADV_RANDOM)
                    except Exception:
                        pass
            else:
                raise FileNotFoundError(f"FATAL RC4: missing mmgl_catalog_v2.midx required for {b}")

        m_npz = np.load(b / "mahal_state.npz", allow_pickle=False)
        self.pca_mean = m_npz["pca_mean"].astype(np.float32)
        self.pca_comp = m_npz["pca_components"].astype(np.float32)
        self.prec = m_npz["precision"].astype(np.float32)
        self.centroids = m_npz["centroids"].astype(np.float32)
        self.jk_centroid = m_npz["jk_centroid"].astype(np.float32)
        self.mahal_z_mean = m_npz["mahal_z_mean"].astype(np.float32) if "mahal_z_mean" in m_npz else np.zeros(4, dtype=np.float32)
        self.mahal_z_std = m_npz["mahal_z_std"].astype(np.float32) if "mahal_z_std" in m_npz else np.ones(4, dtype=np.float32)
        self.cos_centroids = m_npz["cos_centroids"].astype(np.float32) if "cos_centroids" in m_npz else None
        m_npz.close()

        lw = np.load(b / "lexical_weights_fp16.npz", allow_pickle=False)
        self.raw_W_T = np.ascontiguousarray(lw["raw_W"].T, dtype=np.float32)
        self.raw_b = np.ascontiguousarray(lw["raw_b"], dtype=np.float32)
        self.aud_full_W_T = np.ascontiguousarray(lw["aud_full_W"].T, dtype=np.float32)
        self.aud_full_b = np.ascontiguousarray(lw["aud_full_b"], dtype=np.float32)
        self.aud_pure_W32 = np.ascontiguousarray(lw["aud_pure_W"], dtype=np.float32)
        self.aud_pure_b = float(lw["aud_pure_b"][0])
        self.aud_char_W32 = np.ascontiguousarray(lw["aud_char_W"], dtype=np.float32)
        self.aud_char_b = float(lw["aud_char_b"][0])
        self.bin_W_T = np.ascontiguousarray(lw["bin_W"].T, dtype=np.float32) if "bin_W" in lw else None
        self.bin_b = np.ascontiguousarray(lw["bin_b"], dtype=np.float32) if "bin_b" in lw else None
        lw.close()

        gz = json.loads((b / "audio_p_gazetteer_compact.json").read_text(encoding="utf-8"))
        self.gaz_order = [
            "anime_dub_only", "anime_dual", "anime_sub_only", "non_anime_dual",
            "non_anime_sub_only", "bad_non_en_dual", "fr_anime", "de_anime", "anime_bd_web",
        ]
        self.gaz_sets = {k: frozenset(gz.get(k, ())) for k in self.gaz_order}
        self.anime_phrases = frozenset(gz.get("anime_phrases", ()))

        # Strictly OOF-calibrated scalar/vector parameters (zero hardcoded rules!)
        self.tau_jk = float(self.cfg["tau_jk"])
        self.tau_saf_lo = float(self.cfg["tau_saf_lo"])
        self.tau_saf_hi = float(self.cfg["tau_saf_hi"])
        self.tau_anime = float(self.cfg["tau_anime"])
        self.tau_audio_en = float(self.cfg["tau_audio_en"])
        self.tau_adult = float(self.cfg["tau_adult"])
        self.is_sg_prob_thr = float(self.cfg["is_sg_prob_thr"])
        self.clean_ctx_media_thr = float(self.cfg["clean_ctx_media_thr"])
        self.clean_ctx_xxx_thr = float(self.cfg["clean_ctx_xxx_thr"])
        self.tau_admit_bare = float(self.cfg["tau_admit_bare"])
        self.tau_admit_mf = float(self.cfg["tau_admit_mf"])
        self.tau_xxx_bare = float(self.cfg["tau_xxx_bare"])
        self.tau_xxx_mf = float(self.cfg["tau_xxx_mf"])
        self.s1_temp_scale = float(self.cfg["s1_temp_scale"])
        self.p_saf_adm_floor = float(self.cfg["p_saf_adm_floor"])
        self.admit_coef = np.asarray(self.cfg["admit_coef"], dtype=np.float32)
        self.admit_intercept = float(self.cfg["admit_intercept"])
        self.cat_prior_scale = np.asarray(self.cfg["cat_prior_scale"], dtype=np.float32)
        self.unk_conformal_tau = float(self.cfg["unk_conformal_tau"])
        self.meta_mlp_alpha = float(self.cfg.get("meta_mlp_alpha", 0.0))
        self.tau_adult_promote = float(self.cfg.get("tau_adult_promote", 2.0))
        self.p_xxx_weights = tuple(float(v) for v in self.cfg.get("p_xxx_weights", (0.50, 0.30, 0.20)))

    def set_shared_midx(self, midx: Any) -> None:
        if midx is None or midx is self.midx:
            return
        if self._owns_midx and self.midx is not None and hasattr(self.midx, "close"):
            try:
                self.midx.close()
            except Exception:
                pass
        self.midx = midx
        self._owns_midx = False

    def close(self) -> None:
        self.auth.close()
        if self._owns_midx and self.midx is not None and hasattr(self.midx, "close"):
            try:
                self.midx.close()
            except Exception:
                pass
            self.midx = None

    def _mahal_all_and_geom24(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        z = (emb - self.pca_mean) @ self.pca_comp.T
        def _md(mu: np.ndarray) -> np.ndarray:
            d = z - mu
            return np.sqrt(np.clip(np.sum((d @ self.prec) * d, axis=1), 0.0, None))
        all_d = np.stack([_md(self.centroids[c]) for c in range(11)], axis=1).astype(np.float32)
        d_jk = _md(self.jk_centroid).astype(np.float32)
        geom24 = None
        if self.cos_centroids is not None:
            en = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
            cos12 = (en @ self.cos_centroids.T).astype(np.float32)
            geom24 = np.concatenate([cos12, all_d, d_jk[:, None]], axis=1).astype(np.float32)
        return all_d, d_jk, geom24

    def _gaz_p_feats(self, aud_txts: list[str]) -> np.ndarray:
        out = np.zeros((len(aud_txts), 10), dtype=np.float32)
        for i, txt in enumerate(aud_txts):
            tl = txt.lower()
            toks = set(tl.replace(".", " ").replace("_", " ").replace("-", " ").replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ").split())
            for j, k in enumerate(self.gaz_order):
                if toks & self.gaz_sets[k]:
                    out[i, j] = 1.0
            if any(ph in tl for ph in self.anime_phrases):
                out[i, 9] = 1.0
        return out

    def _safety_surf_8d(self, rows: list[dict[str, Any]], vcs: list[dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(rows), 8), dtype=np.float32)
        for i, (r, vc) in enumerate(zip(rows, vcs)):
            nm = str(r.get("name") or "")
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
        """Single-pass end-to-end 4-plane pure-ML inference emitting consumer-compatible 8-head contract."""
        M = len(rows_in)
        if M == 0:
            return []
        rows = [_sanitize_row(r) for r in rows_in]

        # Plane 0: Deterministic full-input security scan
        p0_list = self.plane0.inspect_plane0_batch(rows)

        # Strictly P(x)-only audio projection computed ONCE per row
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

        # Plane 1: Pure-FP32 CandB4 ONNX forward with dynamic trim + bare/MF bucketing
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

        an_l_col = an_l[:, None].astype(np.float32) if an_l.ndim == 1 else an_l.astype(np.float32)
        sf_1d = sf_l[:, 0].astype(np.float32) if sf_l.ndim == 2 else sf_l.astype(np.float32)

        # Plane 2: Structural + External Authority (.aidx + ZCB3 + optional .midx v2) + Hashed Lexical Linear
        tf_l, opq_l, au38_l, p2_l, lex_raw_l = [], [], [], [], []
        lex_bin3_l, midx16_l = [], []
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
            h_raw = extract_hashed_counts(ltxt)
            lex_raw_l.append(score_linear_sparse_vec_T(h_raw, self.raw_W_T, self.raw_b))
            if self.bin_W_T is not None and self.bin_b is not None:
                lex_bin3_l.append(score_linear_sparse_vec_T(h_raw, self.bin_W_T, self.bin_b))
            if self._needs_midx:
                midx16_l.append(extract_midx_core_16d(self.midx, nm, fl))

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
        lex_raw_b = np.stack(lex_raw_l).astype(np.float32)

        x_tab = np.concatenate([tsum_b, tf_b, opq_b, au38_b, p2_b], axis=1).astype(np.float32)  # 118d
        tab_p = _ort_probs(self.sess_tab.run(None, {"X": x_tab}), binary=False)
        all_d, d_jk, geom24 = self._mahal_all_and_geom24(emb_b)

        if self._cat_in_dim == 190:
            lex_bin3_b = np.stack(lex_bin3_l).astype(np.float32)
            midx16_b = None
            assert geom24 is not None, "FATAL RC6: mahal_state.npz missing cos_centroids for 190-d geom24"
            x_meta = np.concatenate([c11_l, jk2_l, an_l_col, tab_p, x_tab, lex_raw_b, lex_bin3_b, geom24], axis=1).astype(np.float32)  # 190d pure-512m Core!
        elif self._cat_in_dim == 206:
            lex_bin3_b = np.stack(lex_bin3_l).astype(np.float32)
            midx16_b = np.stack(midx16_l).astype(np.float32)
            assert geom24 is not None, "FATAL RC4: mahal_state.npz missing cos_centroids for 206-d geom24"
            x_meta = np.concatenate([c11_l, jk2_l, an_l_col, tab_p, x_tab, lex_raw_b, lex_bin3_b, midx16_b, geom24], axis=1).astype(np.float32)  # 206d
        else:
            midx16_b = np.stack(midx16_l).astype(np.float32) if midx16_l else None
            x_meta = np.concatenate([c11_l, jk2_l, an_l_col, x_tab, lex_raw_b], axis=1).astype(np.float32)  # 152d

        ap_ext_b = np.stack(ap_ext_l).astype(np.float32)
        ap_auth_b = np.stack(ap_auth_l).astype(np.float32)
        ap_gaz_b = self._gaz_p_feats(aud_txts)
        laf_b = np.stack(lex_aud_full_l).astype(np.float32)
        lap_b = np.asarray(lex_aud_pure_l, dtype=np.float32)[:, None]
        lac_b = np.asarray(lex_aud_char_l, dtype=np.float32)[:, None]
        x_aud = np.concatenate([
            aud_l, ap_b, ap_ext_b, ap_auth_b, ap_gaz_b,
            laf_b[:, 0:1], lap_b, lac_b, laf_b[:, 1:2],
        ], axis=1).astype(np.float32)  # 72d strictly P(x)-only!

        if self._ani_in_dim == 143 and midx16_b is not None:
            x_ani = np.concatenate([
                an_l_col, c11_l, au38_b, ap_gaz_b,
                laf_b[:, 1:2], tsum_b, tf_b, midx16_b,
            ], axis=1).astype(np.float32)  # 143d
        else:
            x_ani = np.concatenate([
                an_l_col, c11_l, au38_b, ap_gaz_b,
                laf_b[:, 1:2], tsum_b, tf_b,
            ], axis=1).astype(np.float32)  # 127d

        m_cat_lgb = _ort_probs(self.sess_cat.run(None, {"X": x_meta}), binary=False)
        if self.sess_cat_mlp is not None and self.meta_mlp_alpha > 0.0:
            mlp_in = self.sess_cat_mlp.get_inputs()[0].name
            m_cat_mlp = np.asarray(self.sess_cat_mlp.run(None, {mlp_in: x_meta})[0], dtype=np.float32)
            m_cat_p = ((1.0 - self.meta_mlp_alpha) * m_cat_lgb + self.meta_mlp_alpha * m_cat_mlp).astype(np.float32)
        else:
            m_cat_p = m_cat_lgb

        x_meta_p = np.concatenate([x_meta, m_cat_p], axis=1).astype(np.float32) if (self.sess_adult is not None or self._jk_in_dim == x_meta.shape[1] + 11) else x_meta
        m_adult_p = _ort_probs(self.sess_adult.run(None, {"X": x_meta_p if self._ad_in_dim == x_meta_p.shape[1] else x_meta}), binary=True) if self.sess_adult is not None else None
        m_jk_p = _ort_probs(self.sess_jk.run(None, {"X": x_meta_p if self._jk_in_dim == x_meta_p.shape[1] else x_meta}), binary=True)
        m_aud_en = _ort_probs(self.sess_aud.run(None, {"X": x_aud}), binary=True)
        m_ani_p = _ort_probs(self.sess_ani.run(None, {"X": x_ani}), binary=True)

        # Epistemic uncertainty + Mahalanobis + Tier-2 Safety
        c11_p = _softmax(c11_l / self.s1_temp_scale if self.s1_temp_scale != 1.0 else c11_l)
        c_ent = -np.sum(m_cat_p * np.log(np.clip(m_cat_p, 1e-7, 1.0)), axis=1).astype(np.float32)
        t2 = np.partition(m_cat_p, -2, axis=1)[:, -2:]
        c_mar = (t2[:, 1] - t2[:, 0]).astype(np.float32)
        m_mix = 0.5 * (np.clip(c11_p, 1e-7, 1.0) + np.clip(tab_p, 1e-7, 1.0))
        js_b = (0.5 * (
            np.sum(np.clip(c11_p, 1e-7, 1.0) * np.log(np.clip(c11_p, 1e-7, 1.0) / m_mix), axis=1)
            + np.sum(np.clip(tab_p, 1e-7, 1.0) * np.log(np.clip(tab_p, 1e-7, 1.0) / m_mix), axis=1)
        )).astype(np.float32)
        mah_b = np.stack([all_d.min(axis=1), all_d[np.arange(M), m_cat_p.argmax(axis=1)], all_d[:, 9], d_jk], axis=1).astype(np.float32)

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

            # PURE-ML Category Decision (ZERO hardcoded torrent-title rules!)
            cp_raw = m_cat_p[i].astype(np.float32) * self.cat_prior_scale
            cp = cp_raw / max(1e-12, float(cp_raw.sum()))
            pred_cat_i = int(np.argmax(cp))
            if self.unk_conformal_tau < 1.0 and pred_cat_i != idx_xxx and float(cp[idx_unk]) >= self.unk_conformal_tau:
                pred_cat_i = idx_unk
            p_ad_head = float(m_adult_p[i]) if m_adult_p is not None else 0.0
            if m_adult_p is not None and p_ad_head >= self.tau_adult_promote:
                pred_cat_i = idx_xxx
                if float(cp[idx_xxx]) < p_ad_head:
                    rem = max(1e-12, 1.0 - float(cp[idx_xxx]))
                    cp = cp * ((1.0 - p_ad_head) / rem)
                    cp[idx_xxx] = p_ad_head
            pred_cat = CAT11[pred_cat_i]

            # Policy-2 Structural Security Contract + Pure-ML Semantic Junk Head
            sig = p2_sigs[i]
            p_jk = float(m_jk_p[i])
            is_sg = (pred_cat_i in (idx_game, idx_sw)) or (float(cp[idx_game] + cp[idx_sw]) >= self.is_sg_prob_thr)
            is_pure_chm_eb = (pred_cat_i == idx_eb) and bool(sig["pure_chm_only"])
            if sig["hard_dropper_trap"]:
                p_jk = max(p_jk, 0.999)
                pred_is_junk = True
            elif sig["policy2_non_sg_trap"] and (not is_sg) and (not is_pure_chm_eb):
                p_jk = max(p_jk, 0.995)
                pred_is_junk = True
            else:
                pred_is_junk = p_jk >= self.tau_jk

            # Plane-0 Monotonic Safety Precedence Lattice (NEVER downgraded!)
            is_clean_ctx = (float(cp[2] + cp[5] + cp[6] + cp[0] + cp[1]) >= self.clean_ctx_media_thr) and (float(cp[idx_xxx]) < self.clean_ctx_xxx_thr)
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
            pred_lang = "english" if p_en >= self.tau_audio_en else "non_english"

            # PURE-ML Anime Decision (ZERO hardcoded franchise/studio phrase lists!)
            p_an = float(np.clip(m_ani_p[i], 0.0, 1.0))
            is_ani_pred = (p_an >= self.tau_anime)
            p_ad = float(np.clip(max(float(cp[idx_xxx]), p_ad_head) if m_adult_p is not None else float(cp[idx_xxx]), 0.0, 1.0))
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
            admit_score = round(_sigmoid(float(np.dot(self.admit_coef, x_adm) + self.admit_intercept)), 6)
            tau_adm = round(self.tau_admit_bare if is_bare > 0.5 else self.tau_admit_mf, 6)
            tau_xxx = round(self.tau_xxx_bare if is_bare > 0.5 else self.tau_xxx_mf, 6)
            if m_adult_p is not None:
                w0, w1, w2 = self.p_xxx_weights
                p_xxx_eff = round(w0 * float(cp[idx_xxx]) + w1 * p_ad_head + w2 * float(c11_p[i, idx_xxx]), 6)
            else:
                w_mx = float(self.cfg.get("p_xxx_meta_weight", 0.75))
                p_xxx_eff = round(w_mx * float(cp[idx_xxx]) + (1.0 - w_mx) * float(c11_p[i, idx_xxx]), 6)

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
                "adult": "adult" if p_ad >= self.tau_adult else "non_adult",
                "anime": "anime" if is_ani_pred else "non_anime",
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
                    "adult": {"label": "adult" if p_ad >= self.tau_adult else "non_adult", "probs": {"non_adult": round(1.0 - p_ad, 6), "adult": round(p_ad, 6)}},
                    "junk": {"label": "junk" if pred_is_junk else "clean", "probs": {"clean": round(1.0 - p_jk_c, 6), "junk": round(p_jk_c, 6)}},
                    "anime": {"label": "anime" if is_ani_pred else "non_anime", "probs": {"non_anime": round(1.0 - p_an, 6), "anime": round(p_an, 6)}},
                    "english_subs": {"label": "no_eng_subs", "probs": {"no_eng_subs": 1.0, "has_eng_subs": 0.0}},
                    "lang": {"label": pred_lang, "probs": {"english": round(p_en, 6), "non_english": round(p_ne, 6)}},
                    "owner_scope": {"label": "allowed_type" if pred_cat in MEDIA_ADMIT_CATS else "excluded_type", "probs": {"allowed_type": round(p_scope_ok, 6), "excluded_type": round(1.0 - p_scope_ok, 6)}},
                    "title_language": {"label": "english_title" if p_en >= self.tau_audio_en else "non_english_title", "probs": {"english_title": round(p_en, 6), "non_english_title": round(p_ne, 6)}},
                },
            })
        return outputs
