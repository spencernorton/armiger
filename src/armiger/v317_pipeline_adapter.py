"""Clean-Sheet Armiger Ultra v3.17.0 / v3.18.0-MMGL Pipeline Adapter (`V317UltraPipelineAdapter`).

Zero `.pkl`, zero `torch`, zero `scikit-learn` runtime adapter wrapping `ArmigerUltraV317Runtime`
into the 8-head + `csam_safety` + `honeypot_trap` + `mondrian_gates` + `plugin_trace` contract
expected by `armiger/release_worker.py` and `armiger/release_engine.py`.

When `mmgl_config.json` is absent from `model_dir` (e.g., frozen `v3.17.0-rc3`), behavior and
outputs are byte-identical to `v3.17.0`. When an MMGL bundle is present, `MMGLEngine` runs
strictly post-classify with zero feedback into frozen Planes 0-3 and a fail-closed entry gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


_YEAR_RE = re.compile(r"(?:19[2-9]\d|20[0-3]\d)")
_RES_RE = re.compile(r"\b(2160p|1080p|720p|480p|4k|uhd)\b", re.IGNORECASE)

_SHARED_CACHE_LOCK = threading.RLock()
_SHARED_RUNTIME_CACHE: Dict[Tuple[Any, ...], Any] = {}
_SHARED_RUNTIME_LOCKS: Dict[Tuple[Any, ...], threading.RLock] = {}
_SHARED_MMGL_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _cache_key(model_dir: Path, sentinel_names: Union[str, Sequence[str]]) -> Tuple[Any, ...]:
    names = (sentinel_names,) if isinstance(sentinel_names, str) else tuple(sentinel_names)
    sig: List[Any] = [str(model_dir)]
    for nm in names:
        p = model_dir / nm
        try:
            st = p.stat()
            sig.append((nm, int(st.st_ino), int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            sig.append((nm, 0, 0, 0))
    return tuple(sig)


def _strip_mmgl_keys(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Ensure frozen v3.17.0 runtime NEVER sees offered candidates or MMGL control flags."""
    return {
        k: v
        for k, v in row.items()
        if k != "candidates" and k != "catalog_exact_episodes" and not str(k).startswith("_mmgl_")
    }


class UltraPredictionResult:
    """Wraps a single v3.17.0 classification dictionary with `.to_dict()` for `release_worker.py`."""

    __slots__ = ("_payload",)

    def __init__(self, raw: Mapping[str, Any], tau_jk: float) -> None:
        heads_in = raw["heads"]
        adm = raw["admission"]
        saf = raw["safety"]
        span_in = raw.get("span_bio") or {}

        cat_label = str(raw["category"])
        cat_probs = dict(heads_in["category"]["probs"])

        ad_label = str(heads_in["adult"]["label"])
        ad_probs = dict(heads_in["adult"]["probs"])

        jk_label = str(heads_in["junk"]["label"])
        jk_probs = dict(heads_in["junk"]["probs"])

        an_label = str(heads_in["anime"]["label"])
        an_probs = dict(heads_in["anime"]["probs"])

        es_in = heads_in["english_subs"]["probs"]
        p_no_sub = float(es_in.get("no_eng_subs", 1.0))
        p_has_sub = float(es_in.get("has_eng_subs", 0.0))
        p_unk_sub = max(0.0, round(1.0 - p_no_sub - p_has_sub, 6))
        es_probs = {
            "no_eng_subs": round(p_no_sub, 6),
            "has_eng_subs": round(p_has_sub, 6),
            "unknown": p_unk_sub,
        }
        es_label = str(heads_in["english_subs"]["label"])
        if es_label not in es_probs:
            es_label = "no_eng_subs"

        lg_in = heads_in["lang"]["probs"]
        p_en = float(lg_in.get("english", 0.5))
        p_ne = float(lg_in.get("non_english", 0.5))
        p_mu = max(0.0, round(1.0 - p_en - p_ne, 6))
        lg_probs = {
            "english": round(p_en, 6),
            "non_english": round(p_ne, 6),
            "multi_or_unknown": p_mu,
        }
        lg_label = str(heads_in["lang"]["label"])
        if lg_label not in lg_probs:
            lg_label = "english" if p_en >= p_ne else "non_english"

        sc_in = heads_in["owner_scope"]["probs"]
        p_allow = float(sc_in.get("allowed_type", 0.0))
        p_excl = round(1.0 - p_allow, 6)
        admitted = bool(adm.get("admitted", False))
        sc_label = "allowed_type" if (cat_label in ("movie", "tv_show") and admitted) else "excluded"
        sc_probs = {
            "allowed_type": round(p_allow, 6),
            "excluded": p_excl,
        }

        tl_in = heads_in["title_language"]["probs"]
        tl_label = str(heads_in["title_language"]["label"])
        tl_probs = {
            "english_title": round(float(tl_in.get("english_title", p_en)), 6),
            "non_english_title": round(float(tl_in.get("non_english_title", p_ne)), 6),
        }

        title_str = str(span_in.get("title") or "")
        span_trace: Dict[str, Any] = {
            "title": title_str,
            "title_byte_span": list(span_in.get("title_byte_span") or [0, 0]),
            "utf8_valid": bool(span_in.get("utf8_valid", True)),
        }
        ym = _YEAR_RE.search(title_str)
        if ym:
            span_trace["year"] = int(ym.group(0))
        rm = _RES_RE.search(title_str)
        if rm:
            span_trace["res"] = rm.group(1).lower()

        is_safe = bool(saf.get("is_safe", True))
        overall_safe = bool(saf.get("overall_safe_to_index", False)) and admitted

        self._payload: Dict[str, Any] = {
            "category": cat_label,
            "heads": {
                "category": {"label": cat_label, "probs": cat_probs},
                "adult": {"label": ad_label, "probs": ad_probs},
                "junk": {"label": jk_label, "probs": jk_probs},
                "anime": {"label": an_label, "probs": an_probs},
                "english_subs": {"label": es_label, "probs": es_probs},
                "lang": {"label": lg_label, "probs": lg_probs},
                "owner_scope": {"label": sc_label, "probs": sc_probs},
                "title_language": {"label": tl_label, "probs": tl_probs},
            },
            "csam_safety": {
                "is_safe": is_safe,
                "overall_safe_to_index": overall_safe,
                "action": str(saf.get("action", "safe")),
                "level": int(saf.get("level", 4)),
                "rule_code": str(saf.get("rule_code", "L4_SAFE")),
                "incomplete_inspection": bool(saf.get("incomplete_inspection", False)),
            },
            "honeypot_trap": {
                "is_honeypot_trap": False,
            },
            "mondrian_gates": {
                "passed_all": admitted,
                "thresholds": {
                    "adult": 0.5,
                    "junk": round(float(tau_jk), 4),
                    "owner_scope": round(float(adm.get("threshold", 0.32)), 4),
                },
            },
            "swarm_viability": {},
            "plugin_trace": {
                "span_bio": span_trace,
                "v317_plane3_admission": dict(adm),
                "v317_plane0_safety": dict(saf),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return self._payload

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]


class V317UltraPipelineAdapter:
    """Drop-in `ArmigerUltraPipeline` adapter backed by `ArmigerUltraV317Runtime` + optional post-classify MMGL."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        *,
        variant: str = "ultra",
        onnx_Session: Any = None,
        onnx_session: Any = None,
        enable_mmgl: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.variant = variant
        cfg_path = self.model_dir / "config.json"
        if cfg_path.is_file():
            self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            self.config = {"version": "3.17.0"}

        rt_key = _cache_key(
            self.model_dir,
            (
                "config.json", "v4_runtime_config.json", "v317_runtime_config.json",
                "MANIFEST_V4_MASTER.sha256", "MANIFEST_V3_18_0_MMGL.sha256",
                "armiger_ultra.fp32.onnx", "lexical_weights_fp16.npz", "mahal_state.npz",
                "meta_cat11.onnx", "meta_cat11_mlp.onnx", "meta_adult.onnx", "meta_junk.onnx", "meta_anime.onnx", "meta_audio_p.onnx",
                "safety_tier2.onnx", "tab_only_cat11.onnx", "authority_v317.aidx",
            ),
        )
        with _SHARED_CACHE_LOCK:
            shared = _SHARED_RUNTIME_CACHE.get(rt_key)
            if shared is None:
                if (self.model_dir / "v4_runtime_config.json").is_file():
                    if (self.model_dir / "armiger_v4_runtime.py").is_file() and str(self.model_dir) not in sys.path[:2]:
                        sys.path.insert(0, str(self.model_dir))
                    try:
                        from armiger.armiger_v4_runtime import ArmigerUltraV4Runtime
                    except ImportError:
                        from armiger_v4_runtime import ArmigerUltraV4Runtime
                    _init_s1 = onnx_Session if onnx_Session is not None else onnx_session
                    shared = ArmigerUltraV4Runtime(self.model_dir, intra_threads=1, sess_s1=_init_s1)
                else:
                    if (self.model_dir / "armiger_v317_runtime.py").is_file() and str(self.model_dir) not in sys.path[:2]:
                        sys.path.insert(0, str(self.model_dir))
                    try:
                        from armiger.armiger_v317_runtime import ArmigerUltraV317Runtime
                    except ImportError:
                        from armiger_v317_runtime import ArmigerUltraV317Runtime
                    shared = ArmigerUltraV317Runtime(self.model_dir, intra_threads=1)
                _SHARED_RUNTIME_CACHE[rt_key] = shared
                _SHARED_RUNTIME_LOCKS[rt_key] = threading.RLock()
            self._rt = shared
            self._rt_lock = _SHARED_RUNTIME_LOCKS[rt_key]

            self._mmgl = None
            mmgl_cfg = self.model_dir / "mmgl_config.json"
            if enable_mmgl is None:
                _mmgl_env = os.environ.get("ULTRA_ENABLE_MMGL_STAGES")
                if _mmgl_env is not None:
                    enable_mmgl = str(_mmgl_env).strip().lower() in ("1", "true", "yes")
                else:
                    enable_mmgl = bool(getattr(self._rt, "_needs_midx", False))
            if enable_mmgl and mmgl_cfg.is_file():
                mmgl_key = _cache_key(
                    self.model_dir,
                    (
                        "MANIFEST_MMGL.sha256", "mmgl_config.json",
                        "mmgl_pack_clf.onnx", "mmgl_pack_lex.onnx", "mmgl_pack_char.onnx",
                        "mmgl_match_reranker.onnx", "mmgl_match_reranker_e2e.onnx",
                        "mmgl_work_embedder.onnx", "mmgl_group_pair_clf.onnx",
                        "mmgl_catalog_v2.midx", "mmgl_catalog_v1.midx",
                    ),
                )
                mmgl_eng = _SHARED_MMGL_CACHE.get(mmgl_key)
                if mmgl_eng is None:
                    if (self.model_dir / "mmgl_catalog_v2.midx").is_file():
                        try:
                            from armiger.mmgl_engine_v2 import MMGLEngineV2
                        except ImportError:
                            from mmgl_engine_v2 import MMGLEngineV2
                        mmgl_eng = MMGLEngineV2(self.model_dir, verify_manifest=True, load_midx=True, enable_cpu_mem_arena=False)
                    else:
                        try:
                            from armiger.mmgl_engine import MMGLEngine
                        except ImportError:
                            from mmgl_engine import MMGLEngine
                        mmgl_eng = MMGLEngine(self.model_dir, verify_manifest=True, load_midx=True)
                    _SHARED_MMGL_CACHE[mmgl_key] = mmgl_eng
                self._mmgl = mmgl_eng
                if getattr(mmgl_eng, "midx", None) is not None and hasattr(self._rt, "set_shared_midx"):
                    self._rt.set_shared_midx(mmgl_eng.midx)

        self.mmgl_enabled = self._mmgl is not None
        self._s1_session = (
            onnx_Session
            if onnx_Session is not None
            else (onnx_session if onnx_session is not None else shared.sess_s1)
        )

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        *,
        variant: str = "ultra",
        **kwargs: Any,
    ) -> "V317UltraPipelineAdapter":
        return cls(directory, variant=variant, **kwargs)

    @classmethod
    def from_Directory(
        cls,
        directory: Union[str, Path],
        *,
        variant: str = "ultra",
        **kwargs: Any,
    ) -> "V317UltraPipelineAdapter":
        return cls(directory, variant=variant, **kwargs)

    def classify_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        ingest_to_dht_db: bool = False,
        **kwargs: Any,
    ) -> List[UltraPredictionResult]:
        row_list = list(rows)
        clean_rows = [_strip_mmgl_keys(r) if isinstance(r, Mapping) else r for r in row_list]
        with self._rt_lock:
            prev_sess = self._rt.sess_s1
            try:
                if self._s1_session is not None:
                    self._rt.sess_s1 = self._s1_session
                raw_list = self._rt.classify_batch(clean_rows)
            finally:
                self._rt.sess_s1 = prev_sess
        tau_jk = float(self._rt.tau_jk)
        results = [UltraPredictionResult(r, tau_jk) for r in raw_list]

        if self._mmgl is None:
            return results

        # Post-classify MMGL plane: ZERO feedback into frozen Planes 0-3!
        for orig_row, res in zip(row_list, results):
            if not isinstance(orig_row, Mapping) or orig_row.get("_mmgl_skip") is True:
                continue
            cands = orig_row.get("_mmgl_candidates") if "_mmgl_candidates" in orig_row else orig_row.get("candidates")
            job_in: Dict[str, Any] = {
                "id": str(orig_row.get("id") or orig_row.get("info_hash") or ""),
                "name": str(orig_row.get("name") or ""),
                "files": list(orig_row.get("files") or []),
                "files_complete": bool(orig_row.get("files_complete", True)),
                "file_count": orig_row.get("file_count"),
                "size": int(orig_row.get("size") or 0),
                "candidates": list(cands) if isinstance(cands, list) else [],
            }
            if orig_row.get("catalog_exact_episodes") is not None:
                job_in["catalog_exact_episodes"] = orig_row["catalog_exact_episodes"]

            stage = str(orig_row.get("_mmgl_stage") or "classify")
            need_embed = bool(orig_row.get("_mmgl_need_embed", False) or stage == "embed")
            need_match = bool(orig_row.get("_mmgl_need_match", False) or stage == "match" or bool(job_in["candidates"]))

            payload = res.to_dict()
            mmgl_out = self._mmgl.run_row(
                job_in,
                payload,
                stage=stage,
                need_embed=need_embed,
                need_match=need_match,
                allow_local_retrieval=bool(need_match and not job_in["candidates"]),
            )
            pt = payload["plugin_trace"]
            pt["mmgl"] = mmgl_out["trace_mmgl"]
            pt["mmgl_proposals"] = {
                "gated": mmgl_out["gated"],
                "gate_reason": mmgl_out["gate_reason"],
                "extraction_wire": mmgl_out["extraction_wire"],
                "match_wire": mmgl_out["match_wire"],
                "embedding_wire": mmgl_out["embedding_wire"],
            }
            if not mmgl_out["gated"] and isinstance(mmgl_out.get("extraction_wire"), dict):
                ext_w = mmgl_out["extraction_wire"]
                new_title = ext_w.get("title")
                new_span = mmgl_out["trace_mmgl"].get("title_span_bytes")
                if isinstance(new_title, str) and new_title and isinstance(new_span, list) and len(new_span) == 2:
                    raw_b = job_in["name"].encode("utf-8", errors="replace")
                    sb, eb = int(new_span[0]), int(new_span[1])
                    if 0 <= sb < eb <= len(raw_b):
                        try:
                            raw_b[sb:eb].decode("utf-8")
                            span_trace = pt["span_bio"]
                            span_trace["title"] = new_title
                            span_trace["title_byte_span"] = [sb, eb]
                            span_trace["utf8_valid"] = True
                            if isinstance(ext_w.get("year"), int):
                                span_trace["year"] = ext_w["year"]
                            if isinstance(ext_w.get("video_resolution"), str):
                                span_trace["res"] = ext_w["video_resolution"].lower()
                        except UnicodeDecodeError:
                            pass
        return results

    def classify(self, row: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return self.classify_batch([row], **kwargs)[0].to_dict()

ArmigerUltraPipeline = V317UltraPipelineAdapter
