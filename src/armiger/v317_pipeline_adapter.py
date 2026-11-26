"""Clean-Sheet Armiger Ultra v3.17.0 Pipeline Adapter (`V317UltraPipelineAdapter`).

Zero `.pkl`, zero `torch`, zero `scikit-learn` runtime adapter wrapping `ArmigerUltraV317Runtime`
into the 8-head + `csam_safety` + `honeypot_trap` + `mondrian_gates` + `plugin_trace` contract
expected by `armiger/release_worker.py` and `armiger/release_engine.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Sequence, Union

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


_YEAR_RE = re.compile(r"(?:19[2-9]\d|20[0-3]\d)")
_RES_RE = re.compile(r"\b(2160p|1080p|720p|480p|4k|uhd)\b", re.IGNORECASE)

_SHARED_RUNTIME_CACHE: Dict[str, Any] = {}


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
    """Drop-in `ArmigerUltraPipeline` adapter backed by `ArmigerUltraV317Runtime`."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        *,
        variant: str = "ultra",
        onnx_Session: Any = None,
        onnx_session: Any = None,
        **kwargs: Any,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.variant = variant
        cfg_path = self.model_dir / "config.json"
        if cfg_path.is_file():
            self.config = json.loads(cfg_path.read_text(encoding="utf-8"))
        else:
            self.config = {"version": "3.17.0"}

        key = str(self.model_dir)
        shared = _SHARED_RUNTIME_CACHE.get(key)
        if shared is None:
            from armiger_v317_runtime import ArmigerUltraV317Runtime
            shared = ArmigerUltraV317Runtime(self.model_dir, intra_threads=1)
            _SHARED_RUNTIME_CACHE[key] = shared
        self._rt = shared
        self._s1_session = onnx_Session if onnx_Session is not None else (onnx_session if onnx_session is not None else shared.sess_s1)

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
        prev_sess = self._rt.sess_s1
        try:
            if self._s1_session is not None:
                self._rt.sess_s1 = self._s1_session
            raw_list = self._rt.classify_batch(list(rows))
        finally:
            self._rt.sess_s1 = prev_sess
        tau_jk = float(self._rt.tau_jk)
        return [UltraPredictionResult(r, tau_jk) for r in raw_list]

    def classify(self, row: Mapping[str, Any], **kwargs: Any) -> Dict[str, Any]:
        return self.classify_batch([row], **kwargs)[0].to_dict()
