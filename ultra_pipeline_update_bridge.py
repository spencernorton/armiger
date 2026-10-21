"""Distributed Model Update Bridge & Federation Aggregator for ArmigerUltraPipeline."""
from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import zlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Ensure sibling plugin modules in the same directory are importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Plugin #1: SwarmSurvivalNet & Residual Adapter
from swarm_survival import (
    SwarmResidualAdapter,
    SwarmSurvivalNet,
)

# Plugin #2: SwarmPriors (4-Pathway Empirical Bayes & Tracker Scrape Priors)
from swarm_priors import (
    PUBLIC_TRACKER_DOMAINS,
    TRACKER_DOMAIN_PRIORS,
    WAY2_FEATURE_NAMES,
    WAY3_FEATURE_NAMES,
    SwarmViabilityReport,
    apply_modality_dropout,
    compute_sampling_priority_weight,
    compute_swarm_viability,
    extract_prior_features,
    extract_swarm_side_vector,
    extract_way2_source_features,
    extract_way3_anomaly_features,
    is_public_safe_source,
    normalise_source,
    normalise_tracker_domain,
    sanitize_sources_for_export,
)

# Plugin #3: ZeroCueCatalog (929.4K Offline Title Lookup + Structural Disambiguator)
from zero_cue_plugin import (
    CATS as ZERO_CUE_CATS,
    ZeroCueCatalogPlugin,
    apply_structural_and_zero_cue_disambiguation,
    hash_catalog_key,
    lookup_zero_cue_catalog,
    norm_title,
    parse_clean_candidates,
)

# Plugin #4: AudioLanguageGuard (Cross-Head Audio & Anime Dub Guard)
from audio_language_guard import (
    apply_audio_language_guard,
    classify_audio_and_anime_dub,
)

# Plugin #5: CalibrationAndDHT + ArmigerDHTRSIEngine
from calibration_and_dht import (
    ArmigerCalibrator,
    ArmigerDHTUpdater,
    SignedDHTManifest,
)
from armiger_dht_rsi import (
    AnonymityGuard,
    AnonymityViolationError,
    AuthoritativeASNVerifier,
    resolve_opt_in_sharing,
    AnonymizedTelemetryRecord,
    ArmigerDHTRSIEngine,
    AudioWebhookAdapter,
    BundleFormat,
    CanaryItem,
    CanaryRegressionReport,
    CandidateObservation,
    DHTRSIMode,
    EncryptedTelemetryBundle,
    GoldenCanaryRegressionGate,
    HybridPQCAEADEngine,
    MLKEM768,
    MovieWebhookAdapter,
    PeerAttestation,
    PhysicalGroundTruthVerifier,
    PublicationWebhookAdapter,
    QBittorrentAdapter,
    SafeFormatValidator,
    SeriesWebhookAdapter,
    SybilQuorumVerifier,
    TelemetryAnonymizer,
    ThresholdDHTManifest,
    ThresholdSignatureVerifier,
    WOTSPlusSHAKE256,
    X25519ChaCha20Poly1305Engine,
    generate_golden_canary_suite,
)

# Plugin #6: ArmigerDHTDBSync + CSAMGuard
from csam_guard import (
    ArmigerSafetyGate,
    ContentFilterResult,
    CSAMGuard,
    CSAMViolationError,
    CSAMViolationType,
    SignedTombstoneList,
    TombstoneStore,
    blind_infohash,
    generate_ed25519_keypair,
    normalize_text,
    sanitize_title,
    sign_tombstone_list,
)
from armiger_dht_db_sync import (
    ALL_CATEGORIES,
    CATEGORIES_MAINSTREAM,
    CATEGORY_ADULT,
    DEFAULT_EPOCH,
    LANGUAGES,
    QUALITY_TIERS,
    ArmigerDHTDatabaseSync,
    DHTShardManifest,
    MerkleTree,
    QuorumAttestation,
    QuorumVerifier,
    ShardCryptoEngine,
    ShardDatabase,
    SubscriptionConfig,
    SyncMode,
    TorrentRecord,
    create_mock_quorum_attestations,
)

# Friendly SDK aliases matching plugin suite documentation
ZeroCueCatalog = ZeroCueCatalogPlugin
CalibrationAndDHT = ArmigerCalibrator
ArmigerDHTDBSync = ArmigerDHTDatabaseSync
CSAMAndAbuseGuard = CSAMGuard



NEURAL_HEAD_CLASSES: Dict[str, List[str]] = {
    "category": [
        "movie", "tv_show", "xxx", "music", "comic",
        "game", "audiobook", "software", "ebook", "course", "unknown",
    ],
    "adult": ["non_adult", "adult"],
    "junk": ["clean", "junk"],
    "anime": ["non_anime", "anime"],
    "english_subs": ["no_eng_subs", "has_eng_subs", "unknown"],
    "lang": ["english", "non_english", "multi_or_unknown"],
    "owner_scope": ["allowed_type", "excluded"],
    "title_language": ["english_title", "non_english_title"],
}

ALL_11_HEAD_NAMES: Tuple[str, ...] = (
    "category",
    "adult",
    "junk",
    "anime",
    "english_subs",
    "lang",
    "owner_scope",
    "title_language",
    "swarm_viability",
    "honeypot_trap",
    "csam_safety",
)


# =====================================================================
# Distributed Model Update Bridge: Closed-Loop RSI Telemetry <-> BEP-46 Updates
# =====================================================================
#
# Instance A (contributor) -> `export_rsi_telemetry_bundle` -> hybrid PQ encrypted telemetry
# Aggregator               -> `RSIUpdateAggregator` (decrypt, consensus, fit, train, sign)
# Peers (distinct ASNs)    -> `attest_dht_update` (dry-run review + signed attestation)
# Instance B (receiver)    -> `stage_and_apply_dht_update` (verify, snapshot, stage,
#                             500-item canary non-regression, commit or roll back)

DHT_UPDATE_SCHEMA = "armiger-dht-update/v1"
RSI_TELEMETRY_SCHEMA = "armiger-rsi-telemetry/v1"
DHT_UPDATE_COMPONENTS: Tuple[str, ...] = (
    "zero_cue_catalog_delta",
    "calibration_delta",
    "swarm_adapter_safetensors_delta",
)
ZERO_CUE_DELTA_MAPS: Tuple[str, ...] = ("title_year_map", "title_only_map", "anime_map")
DEFAULT_TEMPERATURE_GRID: Tuple[float, ...] = tuple(round(0.5 + 0.05 * k, 2) for k in range(71))  # 0.50 .. 4.00
TEMPERATURE_BOUNDS: Tuple[float, float] = (0.25, 10.0)
MAX_ZERO_CUE_DELTA_ENTRIES = 50_000
MAX_ADAPTER_DELTA_ABS = 64.0
SWARM_ADAPTER_TRAINABLE: Tuple[str, ...] = ("junk_gate.bias", "head_scales.junk", "junk_encoder.4.weight")
DEFAULT_CANARY_VIEWS: Tuple[str, ...] = ("bare_title", "file_tree_swarm")
CANARY_HEALTHY_SWARM: Dict[str, Any] = {
    "tracker_known": True,
    "tracker_seeders": 120,
    "tracker_leechers": 10,
    "tracker_completed": 900,
}
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_CANARY_SAMPLE_SUFFIX_RE = re.compile(r"\.sample\d+$")
_MISSING = object()

# Pipeline-facing name for the BEP-46 threshold manifest. There is no separate type:
# `ThresholdDHTManifest` (armiger_dht_rsi.py) is the signed update manifest.
SignedUpdateManifest = ThresholdDHTManifest


class DHTUpdateRejected(ValueError):
    """Raised internally when an update payload fails strict schema or format validation."""


@dataclasses.dataclass
class DHTUpdateResult:
    """Outcome of `stage_and_apply_dht_update` or a dry-run `evaluate_dht_update_payload`."""

    applied: bool
    status: str  # committed | rejected_verification | rolled_back | review_passed | review_failed
    reason: str
    seq_before: int
    seq_after: int
    verification_trace: List[Dict[str, Any]]
    components: List[str]
    baseline_canary: Dict[str, Dict[str, Any]]
    candidate_canary: Dict[str, Dict[str, Any]]
    regressions: List[str]
    state_fingerprint_before: str
    state_fingerprint_after: str
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def encode_safetensors(
    tensors: Mapping[str, Any],
    metadata: Optional[Mapping[str, str]] = None,
) -> bytes:
    """Serialize float32 tensors to the SafeTensors layout (8-byte LE header length + JSON header + raw buffer)."""
    header: Dict[str, Any] = {}
    buffers: List[bytes] = []
    offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float32))
        raw = arr.astype("<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": [int(d) for d in arr.shape], "data_offsets": [offset, offset + len(raw)]}
        buffers.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    return len(header_bytes).to_bytes(8, "little") + header_bytes + b"".join(buffers)


def decode_safetensors(data: bytes) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """Strictly parse a float32 SafeTensors blob. Tensor bytes are only ever read as little-endian float32.

    Structural validation reuses `SafeFormatValidator._validate_safetensors`. The pickle-opcode byte scan is
    applied to the JSON header only: arbitrary float32 payloads legitimately contain byte pairs such as
    0x80 0x03, so scanning the numeric buffer for pickle opcodes produces false positives without adding
    safety (the buffer is never unpickled or executed).
    """
    if not isinstance(data, (bytes, bytearray)) or len(data) < 8:
        raise DHTUpdateRejected("safetensors_too_short")
    data = bytes(data)
    ok, reason = SafeFormatValidator._validate_safetensors(data)
    if not ok:
        raise DHTUpdateRejected(f"safe_format_violation:{reason}")
    header_len = int.from_bytes(data[:8], "little")
    header_raw = data[8:8 + header_len]
    for pattern in SafeFormatValidator.PROHIBITED_BYTE_PATTERNS:
        if pattern in header_raw:
            raise DHTUpdateRejected("safe_format_violation:prohibited_bytecode_pattern_in_header")
    header = json.loads(header_raw.decode("utf-8"))
    buf = data[8 + header_len:]
    meta = header.pop("__metadata__", {}) or {}
    if not isinstance(meta, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
        raise DHTUpdateRejected("safetensors_invalid_metadata")
    out: Dict[str, np.ndarray] = {}
    covered = 0
    for name, meta_t in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        if meta_t.get("dtype") != "F32":
            raise DHTUpdateRejected(f"adapter_delta_dtype_not_f32:{name}")
        shape = meta_t.get("shape")
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise DHTUpdateRejected(f"safetensors_invalid_shape:{name}")
        start, end = (int(x) for x in meta_t["data_offsets"])
        n_elem = int(np.prod(shape)) if shape else 1
        if end - start != 4 * n_elem:
            raise DHTUpdateRejected(f"safetensors_size_mismatch:{name}")
        if start != covered:
            raise DHTUpdateRejected(f"safetensors_non_contiguous:{name}")
        covered = end
        out[name] = np.frombuffer(buf[start:end], dtype="<f4").reshape(shape).astype(np.float32)
    if covered != len(buf):
        raise DHTUpdateRejected("safetensors_trailing_bytes")
    return out, meta


def _validate_zero_cue_delta(delta: Any) -> Dict[str, Dict[str, str]]:
    if not isinstance(delta, dict) or not delta:
        raise DHTUpdateRejected("zero_cue_delta_not_object")
    clean: Dict[str, Dict[str, str]] = {}
    total = 0
    for map_name, entries in delta.items():
        if map_name not in ZERO_CUE_DELTA_MAPS:
            raise DHTUpdateRejected(f"zero_cue_delta_unknown_map:{map_name}")
        if not isinstance(entries, dict):
            raise DHTUpdateRejected(f"zero_cue_delta_map_not_object:{map_name}")
        clean_map: Dict[str, str] = {}
        for key, cat in entries.items():
            if not isinstance(key, str) or not _HEX16_RE.match(key):
                raise DHTUpdateRejected("zero_cue_delta_key_not_blake2b64_hex")
            if cat not in ZERO_CUE_CATS:
                raise DHTUpdateRejected(f"zero_cue_delta_unknown_category:{cat}")
            clean_map[key] = str(cat)
        total += len(clean_map)
        clean[map_name] = clean_map
    if total == 0 or total > MAX_ZERO_CUE_DELTA_ENTRIES:
        raise DHTUpdateRejected(f"zero_cue_delta_entry_count_out_of_bounds:{total}")
    return clean


def _validate_calibration_delta(delta: Any) -> Dict[str, Any]:
    if not isinstance(delta, dict) or not delta:
        raise DHTUpdateRejected("calibration_delta_not_object")
    unknown = set(delta) - {"heads", "mondrian_thresholds"}
    if unknown:
        raise DHTUpdateRejected(f"calibration_delta_unknown_keys:{sorted(unknown)}")
    clean: Dict[str, Any] = {"heads": {}, "mondrian_thresholds": {}}
    if "heads" in delta:
        raw_heads = delta["heads"]
        if not isinstance(raw_heads, dict):
            raise DHTUpdateRejected("calibration_delta_heads_not_object")
    else:
        raw_heads = {}
    if "mondrian_thresholds" in delta:
        raw_mondrian = delta["mondrian_thresholds"]
        if not isinstance(raw_mondrian, dict):
            raise DHTUpdateRejected("calibration_delta_mondrian_thresholds_not_object")
    else:
        raw_mondrian = {}
    for head, cfg in raw_heads.items():
        if head not in NEURAL_HEAD_CLASSES and head != "span_bio":
            raise DHTUpdateRejected(f"calibration_delta_unknown_head:{head}")
        if not isinstance(cfg, dict) or set(cfg) != {"temperature"}:
            raise DHTUpdateRejected(f"calibration_delta_head_schema:{head}")
        temp = cfg["temperature"]
        if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not math.isfinite(float(temp)):
            raise DHTUpdateRejected(f"calibration_delta_temperature_not_finite:{head}")
        if not (TEMPERATURE_BOUNDS[0] <= float(temp) <= TEMPERATURE_BOUNDS[1]):
            raise DHTUpdateRejected(f"calibration_delta_temperature_out_of_bounds:{head}")
        clean["heads"][head] = {"temperature": float(temp)}
    for name, thr in raw_mondrian.items():
        if name not in ("adult", "junk", "owner_scope"):
            raise DHTUpdateRejected(f"calibration_delta_unknown_threshold:{name}")
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not (0.0 < float(thr) < 1.0):
            raise DHTUpdateRejected(f"calibration_delta_threshold_out_of_bounds:{name}")
        clean["mondrian_thresholds"][name] = float(thr)
    if not clean["heads"] and not clean["mondrian_thresholds"]:
        raise DHTUpdateRejected("calibration_delta_empty")
    return clean


def parse_dht_update_payload(payload_bytes: bytes, bundle_format: str) -> Dict[str, Any]:
    """Strictly parse a BEP-46 update payload into validated components. Unknown keys are rejected.

    Formats:
      * `json`: {"schema", "epoch", "zero_cue_catalog_delta"?, "calibration_delta"?,
                 "swarm_adapter_safetensors_delta"?: {"encoding": "safetensors+base64", "data", "sha256"}}
      * `safetensors`: additive adapter delta tensors only.
    """
    fmt = str(bundle_format).lower().strip()
    components: Dict[str, Any] = {}
    if fmt == "safetensors":
        tensors, _meta = decode_safetensors(payload_bytes)
        if not tensors:
            raise DHTUpdateRejected("adapter_delta_empty")
        components["swarm_adapter_safetensors_delta"] = tensors
        return components
    if fmt == "onnx":
        ok_onnx, reason_onnx = SafeFormatValidator.validate(payload_bytes, "onnx")
        if not ok_onnx:
            raise DHTUpdateRejected(f"onnx_validation_failed:{reason_onnx}")
        components["onnx_model_bytes"] = payload_bytes
        return components
    if fmt != "json":
        raise DHTUpdateRejected(f"unsupported_update_format:{fmt}")
    try:
        doc = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - any decode error is a rejection
        raise DHTUpdateRejected(f"json_parse_error:{type(exc).__name__}") from None
    if not isinstance(doc, dict) or doc.get("schema") != DHT_UPDATE_SCHEMA:
        raise DHTUpdateRejected("update_schema_mismatch")
    unknown = set(doc) - {"schema", "epoch", *DHT_UPDATE_COMPONENTS}
    if unknown:
        raise DHTUpdateRejected(f"update_unknown_keys:{sorted(unknown)}")
    if "zero_cue_catalog_delta" in doc:
        components["zero_cue_catalog_delta"] = _validate_zero_cue_delta(doc["zero_cue_catalog_delta"])
    if "calibration_delta" in doc:
        components["calibration_delta"] = _validate_calibration_delta(doc["calibration_delta"])
    if "swarm_adapter_safetensors_delta" in doc:
        blob = doc["swarm_adapter_safetensors_delta"]
        if not isinstance(blob, dict) or set(blob) != {"encoding", "data", "sha256"} or blob["encoding"] != "safetensors+base64":
            raise DHTUpdateRejected("adapter_delta_envelope_schema")
        try:
            raw = base64.b64decode(str(blob["data"]).encode("ascii"), validate=True)
        except Exception:  # noqa: BLE001
            raise DHTUpdateRejected("adapter_delta_base64_invalid") from None
        if hashlib.sha256(raw).hexdigest() != str(blob["sha256"]).lower():
            raise DHTUpdateRejected("adapter_delta_sha256_mismatch")
        tensors, _meta = decode_safetensors(raw)
        if not tensors:
            raise DHTUpdateRejected("adapter_delta_empty")
        components["swarm_adapter_safetensors_delta"] = tensors
    if not components:
        raise DHTUpdateRejected("update_has_no_components")
    return components


def _canary_files_for(item: CanaryItem) -> List[Dict[str, Any]]:
    """Deterministic, physically plausible file tree for a golden canary item (exercises structural paths)."""
    title = _CANARY_SAMPLE_SUFFIX_RE.sub("", item.title)
    kind = item.item_id.split("_")[1]
    if kind in ("mov", "tv", "ani", "xxx"):
        return [{"path": f"{title}.mkv", "size": 1_400_000_000}]
    if kind == "mus":
        return [{"path": f"{title}/0{j}.flac", "size": 40_000_000} for j in range(1, 5)]
    if kind in ("com", "ebk", "jnk"):
        return [{"path": title, "size": 30_000_000}]
    if kind == "aud":
        return [{"path": f"{title}.m4b", "size": 300_000_000}]
    if kind == "sw":
        if title.endswith(".iso"):
            return [{"path": title, "size": 900_000_000}]
        return [{"path": f"{title}/setup.exe", "size": 900_000_000}, {"path": f"{title}/data.bin", "size": 2_000_000_000}]
    if kind == "crs":
        return [{"path": f"{title}/lecture0{j}.mp4", "size": 300_000_000} for j in range(1, 4)]
    return [{"path": title, "size": 100_000_000}]


def build_canary_rows(items: Sequence[CanaryItem], view: str) -> List[Dict[str, Any]]:
    """Materialise the golden canary suite as pipeline rows for a given view."""
    if view == "bare_title":
        return [{"name": it.title} for it in items]
    if view == "file_tree_swarm":
        rows = []
        for it in items:
            files = _canary_files_for(it)
            rows.append({
                "name": it.title,
                "size": int(sum(f["size"] for f in files)),
                "files": files,
                "swarm": dict(CANARY_HEALTHY_SWARM),
            })
        return rows
    raise ValueError(f"unknown canary view: {view}")


def _canary_summary(report: CanaryRegressionReport) -> Dict[str, Any]:
    return {
        "absolute_gate_passed": bool(report.passed),
        "absolute_gate_reason": report.reason,
        "total_items": int(report.total_items),
        "junk_false_negatives": int(report.junk_false_negatives),
        "junk_balanced_accuracy": round(float(report.junk_balanced_accuracy), 6),
        "adult_accuracy": round(float(report.adult_accuracy), 6),
        "category_accuracy": round(float(report.category_accuracy), 6),
        "owner_scope_accuracy": round(float(report.owner_scope_accuracy), 6),
    }


def _prediction_labels(pred: "ArmigerUltraPrediction") -> Dict[str, Any]:
    junk_head = pred.heads.get("junk", {})
    pre_csam_p = float(junk_head.get("pre_csam_p_junk", junk_head.get("probs", {}).get("junk", 0.0)))
    neural_p = float(junk_head.get("neural_p_junk", pre_csam_p))
    junk_thr = float(junk_head.get("threshold", 0.5))
    return {
        "category": pred.category,
        "adult": pred.adult == "adult",
        "junk": pred.junk == "junk",
        "neural_junk": neural_p >= junk_thr,
        "neural_junk_margin": neural_p - junk_thr,
        "pre_csam_junk": pre_csam_p >= junk_thr,
        "pre_csam_junk_margin": pre_csam_p - junk_thr,
        "owner_scope": pred.owner_scope,
    }


def _zero_cue_catalog_key(raw_title: str, authoritative_title: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Derive the catalog's own BLAKE2b-64 `title|year` key for a verified release name.

    Uses `parse_clean_candidates` exactly as `lookup_zero_cue_catalog` does, so the receiver's unchanged
    lookup path finds the entry. Prefers the authoritative library title (Series/Movie webhook) when it
    matches a parsed candidate; otherwise the shortest year-bearing candidate.
    """
    cands, _year = parse_clean_candidates(raw_title)
    year_cands = [(ct, cy) for ct, cy in cands if cy is not None and len(ct) >= 4]
    if not year_cands:
        return None
    chosen: Optional[Tuple[str, int]] = None
    if authoritative_title:
        auth_norm = norm_title(authoritative_title)
        for ct, cy in year_cands:
            if ct == auth_norm:
                chosen = (ct, int(cy))
                break
    if chosen is None:
        ct, cy = min(year_cands, key=lambda c: (len(c[0]), c[0]))
        chosen = (ct, int(cy))
    return {"map": "title_year_map", "key": hash_catalog_key(f"{chosen[0]}|{chosen[1]}")}


def _nll_profile(logits: Sequence[float], label_idx: int, grid: Sequence[float]) -> List[float]:
    z = np.asarray(logits, dtype=np.float64)
    out = []
    for temp in grid:
        s = z / float(temp)
        s = s - s.max()
        out.append(float(-(s[label_idx] - math.log(float(np.exp(s).sum())))))
    return out


def _verified_head_labels(obs: Any) -> Dict[str, str]:
    """Map a VerifiedObservation to neural-head class labels."""
    return {
        "category": obs.category if obs.category in NEURAL_HEAD_CLASSES["category"] else "unknown",
        "adult": "adult" if obs.adult else "non_adult",
        "junk": "junk" if obs.junk else "clean",
        "anime": "anime" if obs.anime else "non_anime",
        "lang": obs.lang if obs.lang in NEURAL_HEAD_CLASSES["lang"] else "multi_or_unknown",
        "owner_scope": obs.owner_scope if obs.owner_scope in NEURAL_HEAD_CLASSES["owner_scope"] else "allowed_type",
        "title_language": obs.title_language if obs.title_language in NEURAL_HEAD_CLASSES["title_language"] else "english_title",
        "english_subs": "has_eng_subs" if obs.english_subs else "no_eng_subs",
    }



# =====================================================================
# Federation Aggregator: Decrypt -> Consensus -> Fit/Train -> Sign
# =====================================================================

def generate_federation_signer() -> Tuple[Any, str, bytes, str]:
    """(Ed25519PrivateKey, ed25519_pub_hex, wots_priv_seed, wots_root_pub_hex) for a federation signer or peer.

    Uses `WOTSPlusSHAKE256` Merkle-authenticated one-time leaf trees (`AWM2`) so each manifest or attestation
    allocates a fresh one-time WOTS+ leaf keypair authenticated to the pinned `wots_root_pub_hex`.
    """
    return ThresholdSignatureVerifier.generate_hybrid_signer()


class RSIUpdateAggregator:
    """Federation aggregator for the closed-loop RSI -> BEP-46 update bridge.

    Holds a hybrid X25519 + ML-KEM-768 keypair; contributors encrypt to (`x25519_pub`, `mlkem_ek`). The
    aggregator only ever sees anonymized records, catalog keys, NLL profiles and swarm feature vectors.
    """

    def __init__(
        self,
        *,
        epoch: str = DEFAULT_EPOCH,
        trusted_contributors: Optional[Mapping[str, int]] = None,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        require_contributor_auth: bool = True,
        require_trusted_contributor: bool = True,
    ) -> None:
        self._hybrid_priv, self._hybrid_pub = X25519ChaCha20Poly1305Engine.generate_keypair()
        self.epoch = str(epoch)
        self.require_contributor_auth = bool(require_contributor_auth)
        self.trusted_contributors: Dict[str, int] = {
            AuthoritativeASNVerifier.normalize_pubkey(k): int(v)
            for k, v in (trusted_contributors or {}).items()
            if AuthoritativeASNVerifier.normalize_pubkey(k)
        }
        self.require_trusted_contributor = bool(require_trusted_contributor)
        self.asn_verifier = asn_verifier or AuthoritativeASNVerifier(
            pinned_pubkey_asn=self.trusted_contributors,
            require_authoritative_binding=self.require_trusted_contributor,
        )
        self.observations: List[Dict[str, Any]] = []
        self.rejected: List[str] = []
        self._seen_ciphertexts: set = set()
        self._seen_plaintext_digests: set = set()
        self._seen_contributor_items: set = set()

    def register_contributor(self, pubkey_hex: str, asn: int) -> None:
        """Pin a contributor Ed25519 public key to its authoritative ASN."""
        norm_pk = AuthoritativeASNVerifier.normalize_pubkey(pubkey_hex)
        if not norm_pk or len(norm_pk) != 64:
            raise ValueError("Invalid contributor Ed25519 public key hex")
        self.trusted_contributors[norm_pk] = int(asn)
        self.asn_verifier.register_pinned_key(norm_pk, int(asn))

    @staticmethod
    def _validate_telemetry_doc_components(doc: Mapping[str, Any]) -> Tuple[bool, str]:
        """Validate record, zero_cue_catalog_key, calibration_nll_profile, and swarm_features schemas."""
        has_component = False

        if "record" in doc and doc["record"] is not None:
            rec = doc["record"]
            if not isinstance(rec, dict):
                return False, "invalid_record_not_dict"
            bh = rec.get("blinded_hash")
            if not isinstance(bh, str) or not bh.strip() or len(bh.strip()) > 128:
                return False, "invalid_record_blinded_hash"
            for num_f in ("dp_size_mb", "dp_duration_mins", "timestamp"):
                if num_f in rec and rec[num_f] is not None:
                    val = rec[num_f]
                    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                        return False, f"invalid_record_numeric_field:{num_f}"
            has_component = True

        if "zero_cue_catalog_key" in doc and doc["zero_cue_catalog_key"] is not None:
            zk = doc["zero_cue_catalog_key"]
            if not isinstance(zk, dict):
                return False, "invalid_zero_cue_catalog_key_not_dict"
            m, k, c = zk.get("map"), zk.get("key"), zk.get("category")
            if (
                m not in ZERO_CUE_DELTA_MAPS
                or not isinstance(k, str)
                or not _HEX16_RE.match(k)
                or c not in ZERO_CUE_CATS
            ):
                return False, "invalid_zero_cue_catalog_key_fields"
            has_component = True

        if "calibration_nll_profile" in doc and doc["calibration_nll_profile"] is not None:
            prof = doc["calibration_nll_profile"]
            if not isinstance(prof, dict):
                return False, "invalid_calibration_profile_not_dict"
            grid = prof.get("grid")
            heads_map = prof.get("heads")
            if (
                not isinstance(grid, list)
                or not (1 <= len(grid) <= 128)
                or any(
                    isinstance(t, bool)
                    or not isinstance(t, (int, float))
                    or not math.isfinite(float(t))
                    or not (TEMPERATURE_BOUNDS[0] <= float(t) <= TEMPERATURE_BOUNDS[1])
                    for t in grid
                )
            ):
                return False, "invalid_calibration_profile_grid"
            if not isinstance(heads_map, dict) or not heads_map:
                return False, "invalid_calibration_profile_heads"
            for h_name, vec in heads_map.items():
                if not isinstance(h_name, str) or h_name not in NEURAL_HEAD_CLASSES:
                    return False, f"invalid_calibration_profile_head_name:{h_name}"
                if (
                    not isinstance(vec, list)
                    or len(vec) != len(grid)
                    or any(
                        isinstance(v, bool)
                        or not isinstance(v, (int, float))
                        or not math.isfinite(float(v))
                        or float(v) < 0.0
                        or float(v) > 1e6
                        for v in vec
                    )
                ):
                    return False, f"invalid_calibration_profile_vector:{h_name}"
            has_component = True

        if "swarm_features" in doc and doc["swarm_features"] is not None:
            sf = doc["swarm_features"]
            if not isinstance(sf, dict):
                return False, "invalid_swarm_features_not_dict"
            w2 = sf.get("way2")
            w3 = sf.get("way3")
            pj = sf.get("p_junk_neural")
            jk = sf.get("junk")
            if (
                not isinstance(w2, list)
                or len(w2) not in (8, len(WAY2_FEATURE_NAMES))
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in w2)
            ):
                return False, "invalid_swarm_features_way2"
            if (
                not isinstance(w3, list)
                or len(w3) not in (6, len(WAY3_FEATURE_NAMES))
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in w3)
            ):
                return False, "invalid_swarm_features_way3"
            if (
                isinstance(pj, bool)
                or not isinstance(pj, (int, float))
                or not math.isfinite(float(pj))
                or not (0.0 <= float(pj) <= 1.0)
            ):
                return False, "invalid_swarm_features_p_junk_neural"
            if not isinstance(jk, bool):
                return False, "invalid_swarm_features_junk_flag"
            has_component = True

        if not has_component:
            return False, "empty_telemetry_components"
        return True, "valid"

    @property
    def x25519_pub(self) -> bytes:
        return self._hybrid_pub[:32]

    @property
    def mlkem_ek(self) -> bytes:
        return self._hybrid_pub[32:]

    def ingest_bundles(
        self,
        bundles: Sequence[Any],
        *,
        require_contributor_auth: Optional[bool] = None,
        require_trusted_contributor: Optional[bool] = None,
    ) -> Dict[str, int]:
        """Decrypt + authenticate bundles; drop replays, duplicate observations, classical-only bundles, AEAD failures and wrong epochs."""
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ed25519

        eff_req_auth = self.require_contributor_auth if require_contributor_auth is None else bool(require_contributor_auth)
        eff_req_trusted = (
            self.require_trusted_contributor
            if require_trusted_contributor is None
            else bool(require_trusted_contributor)
        )

        accepted = rejected = 0
        for item in bundles:
            att: Optional[PeerAttestation] = None
            if isinstance(item, tuple) and len(item) == 2:
                bundle, att = item
            else:
                bundle = item
            try:
                digest = hashlib.sha256(bytes.fromhex(bundle.ciphertext_hex)).hexdigest()
            except ValueError:
                self.rejected.append("malformed_ciphertext_hex")
                rejected += 1
                continue
            if digest in self._seen_ciphertexts:
                self.rejected.append("duplicate_bundle_replay")
                rejected += 1
                continue
            if not bundle.is_post_quantum:
                self.rejected.append("classical_only_bundle_refused")
                rejected += 1
                continue
            att_verified_pk = ""
            if att is not None:
                target_h = hashlib.sha256(bundle.ciphertext_hex.encode("utf-8")).hexdigest()
                try:
                    att_verified_pk = att.peer_pubkey_hex.strip().lower()
                    pk_b = bytes.fromhex(att_verified_pk)
                    sig_b = bytes.fromhex(att.signature_hex.strip())
                    ed25519.Ed25519PublicKey.from_public_bytes(pk_b).verify(sig_b, att.canonical_bytes(target_h))
                except (ValueError, InvalidSignature):
                    self.rejected.append("invalid_peer_attestation_signature")
                    rejected += 1
                    continue
            try:
                raw = X25519ChaCha20Poly1305Engine.decrypt_bundle(bundle, self._hybrid_priv)
                doc = json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                self.rejected.append(f"aead_or_json_rejected:{str(exc)[:60]}")
                rejected += 1
                continue
            if not isinstance(doc, dict) or doc.get("schema") != RSI_TELEMETRY_SCHEMA or doc.get("epoch") != self.epoch:
                self.rejected.append("schema_or_epoch_mismatch")
                rejected += 1
                continue

            contrib_pk = str(doc.get("contributor_pubkey_hex", "")).strip().lower()
            contrib_sig = str(doc.get("contributor_signature_hex", "")).strip().lower()
            unsigned_doc = {
                k: v
                for k, v in doc.items()
                if k not in ("contributor_pubkey_hex", "contributor_signature_hex") and not str(k).startswith("_")
            }
            unsigned_bytes = json.dumps(unsigned_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
            doc_sig_verified = False
            if contrib_pk or contrib_sig:
                if not contrib_pk or not contrib_sig:
                    self.rejected.append("missing_contributor_signature")
                    rejected += 1
                    continue
                try:
                    ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(contrib_pk)).verify(
                        bytes.fromhex(contrib_sig), unsigned_bytes
                    )
                    doc_sig_verified = True
                except (ValueError, InvalidSignature):
                    self.rejected.append("invalid_contributor_signature")
                    rejected += 1
                    continue

            if att_verified_pk and contrib_pk and att_verified_pk != contrib_pk:
                self.rejected.append("contributor_attestation_pubkey_mismatch")
                rejected += 1
                continue

            eff_contrib = att_verified_pk or (contrib_pk if doc_sig_verified else "")
            if eff_req_auth and not eff_contrib:
                self.rejected.append("missing_contributor_signature")
                rejected += 1
                continue

            pinned_asn: Optional[int] = None
            if eff_contrib:
                pinned_asn = (
                    self.trusted_contributors.get(eff_contrib)
                    or self.asn_verifier.pinned_pubkey_asn.get(eff_contrib)
                    or AuthoritativeASNVerifier._GLOBAL_PINNED_PUBKEY_ASN.get(eff_contrib)
                )
            if att is not None:
                if pinned_asn is not None and int(att.asn) != int(pinned_asn):
                    self.rejected.append("contributor_attestation_asn_mismatch")
                    rejected += 1
                    continue
                if att.ip_address is not None or eff_req_trusted:
                    b_ok, _ = self.asn_verifier.verify_binding(
                        claimed_asn=int(att.asn),
                        pubkey_hex=eff_contrib or None,
                        ip_address=att.ip_address,
                        require_authoritative=eff_req_trusted,
                    )
                    if not b_ok:
                        self.rejected.append("untrusted_or_mismatched_contributor_asn")
                        rejected += 1
                        continue
                    if pinned_asn is None and int(att.asn) > 0:
                        pinned_asn = int(att.asn)

            if eff_req_trusted and (not eff_contrib or pinned_asn is None):
                self.rejected.append("untrusted_contributor_pubkey")
                rejected += 1
                continue

            comp_ok, comp_reason = self._validate_telemetry_doc_components(doc)
            if not comp_ok:
                self.rejected.append(f"malformed_telemetry_component:{comp_reason}")
                rejected += 1
                continue

            plaintext_fp = hashlib.sha256(unsigned_bytes).hexdigest()
            if plaintext_fp in self._seen_plaintext_digests:
                self.rejected.append("duplicate_observation_content_replay")
                rejected += 1
                continue

            rec_obj = doc.get("record")
            rec_bh = str(rec_obj.get("blinded_hash", "")).strip() if isinstance(rec_obj, dict) else ""
            zk_obj = doc.get("zero_cue_catalog_key")
            zk_id = f"{zk_obj.get('map')}:{zk_obj.get('key')}" if isinstance(zk_obj, dict) else ""
            voter_scope = f"asn:{pinned_asn}" if pinned_asn is not None else (f"pk:{eff_contrib}" if eff_contrib else "")
            if voter_scope:
                if rec_bh and (voter_scope, "bh", rec_bh) in self._seen_contributor_items:
                    self.rejected.append("duplicate_contributor_observation_replay")
                    rejected += 1
                    continue
                if zk_id and (voter_scope, "zk", zk_id) in self._seen_contributor_items:
                    self.rejected.append("duplicate_contributor_catalog_vote_replay")
                    rejected += 1
                    continue
                if rec_bh:
                    self._seen_contributor_items.add((voter_scope, "bh", rec_bh))
                if zk_id:
                    self._seen_contributor_items.add((voter_scope, "zk", zk_id))

            stored_doc = dict(doc)
            stored_doc["_authenticated_voter_id"] = voter_scope
            stored_doc["_authenticated_pubkey"] = eff_contrib
            stored_doc["_authenticated_asn"] = pinned_asn
            self._seen_ciphertexts.add(digest)
            self._seen_plaintext_digests.add(plaintext_fp)
            self.observations.append(stored_doc)
            accepted += 1
        return {"accepted": accepted, "rejected": rejected}

    def _extract_verified_voter_id(self, doc: Mapping[str, Any]) -> str:
        comp_ok, _ = self._validate_telemetry_doc_components(doc)
        if not comp_ok:
            return ""
        auth_asn = doc.get("_authenticated_asn")
        auth_pk = str(doc.get("_authenticated_pubkey") or "").strip().lower()
        if isinstance(auth_asn, int) and not isinstance(auth_asn, bool) and auth_asn > 0:
            return f"asn:{auth_asn}"
        if auth_pk and not self.require_trusted_contributor:
            return f"pk:{auth_pk}"
        contrib_pk = str(doc.get("contributor_pubkey_hex") or "").strip().lower()
        contrib_sig = str(doc.get("contributor_signature_hex") or "").strip().lower()
        if len(contrib_pk) == 64 and contrib_sig:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric import ed25519

            unsigned_doc = {
                k: v
                for k, v in doc.items()
                if k not in ("contributor_pubkey_hex", "contributor_signature_hex") and not str(k).startswith("_")
            }
            unsigned_bytes = json.dumps(unsigned_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
            try:
                ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(contrib_pk)).verify(
                    bytes.fromhex(contrib_sig), unsigned_bytes
                )
            except (ValueError, InvalidSignature):
                return ""
            pinned = (
                self.trusted_contributors.get(contrib_pk)
                or self.asn_verifier.pinned_pubkey_asn.get(contrib_pk)
                or AuthoritativeASNVerifier._GLOBAL_PINNED_PUBKEY_ASN.get(contrib_pk)
            )
            if pinned is not None:
                return f"asn:{pinned}"
            if not self.require_trusted_contributor:
                return f"pk:{contrib_pk}"
        return ""

    def build_zero_cue_delta(self, *, min_support: int = 3) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
        """Unanimous-vote consensus on catalog keys across distinct authenticated contributors/ASNs; conflicting keys are dropped."""
        votes: Dict[Tuple[str, str], Dict[str, int]] = {}
        seen_voters: Dict[Tuple[str, str], set] = {}
        for doc in self.observations:
            voter_id = self._extract_verified_voter_id(doc)
            if not voter_id:
                continue
            zk = doc.get("zero_cue_catalog_key")
            if not isinstance(zk, dict):
                continue
            m, k, c = zk.get("map"), str(zk.get("key", "")), zk.get("category")
            if m in ZERO_CUE_DELTA_MAPS and _HEX16_RE.match(k) and c in ZERO_CUE_CATS:
                voters = seen_voters.setdefault((m, k), set())
                if voter_id in voters:
                    continue
                voters.add(voter_id)
                votes.setdefault((m, k), {}).setdefault(c, 0)
                votes[(m, k)][c] += 1
        delta: Dict[str, Dict[str, str]] = {}
        stats = {"keys_seen": len(votes), "accepted": 0, "conflicting": 0, "insufficient_support": 0}
        for (m, k), ctr in sorted(votes.items()):
            if len(ctr) > 1:
                stats["conflicting"] += 1
            elif sum(ctr.values()) < min_support:
                stats["insufficient_support"] += 1
            else:
                delta.setdefault(m, {})[k] = next(iter(ctr))
                stats["accepted"] += 1
        return delta, stats

    def fit_calibration_delta(
        self,
        *,
        heads: Sequence[str] = ("category",),
        min_samples: int = 20,
        min_rel_improvement: float = 0.02,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Federated temperature scaling: sum authenticated contributors' NLL(T) profiles, pick argmin T per head."""
        delta: Dict[str, Any] = {"heads": {}}
        stats: Dict[str, Any] = {}
        for head in heads:
            by_grid: Dict[Tuple[float, ...], List[List[float]]] = {}
            seen_profile_keys: set = set()
            for doc in self.observations:
                voter_id = self._extract_verified_voter_id(doc)
                if not voter_id:
                    continue
                prof = doc.get("calibration_nll_profile")
                if isinstance(prof, dict) and head in prof.get("heads", {}):
                    vec = prof["heads"][head]
                    rec_bh = str((doc.get("record") or {}).get("blinded_hash", "")).strip()
                    dedup_key = (voter_id, rec_bh) if rec_bh else (voter_id, tuple(round(float(x), 6) for x in vec))
                    if dedup_key in seen_profile_keys:
                        continue
                    seen_profile_keys.add(dedup_key)
                    by_grid.setdefault(tuple(float(t) for t in prof["grid"]), []).append(vec)
            if not by_grid:
                stats[head] = {"n": 0, "skipped": "no_profiles"}
                continue
            grid, profiles = max(by_grid.items(), key=lambda kv: len(kv[1]))
            n = len(profiles)
            if n < min_samples or 1.0 not in grid:
                stats[head] = {"n": n, "skipped": "insufficient_samples_or_grid"}
                continue
            mean = np.asarray(profiles, dtype=np.float64).mean(axis=0)
            valid_indices = [
                idx_g for idx_g, g_val in enumerate(grid)
                if TEMPERATURE_BOUNDS[0] <= float(g_val) <= TEMPERATURE_BOUNDS[1]
            ]
            if not valid_indices:
                stats[head] = {"n": n, "skipped": "no_in_bounds_grid_points"}
                continue
            best = min(valid_indices, key=lambda idx_g: float(mean[idx_g]))
            nll_t1 = float(mean[grid.index(1.0)])
            entry = {"n": n, "temperature": float(grid[best]), "mean_nll_T1": round(nll_t1, 6), "mean_nll_Tstar": round(float(mean[best]), 6)}
            if float(mean[best]) > nll_t1 * (1.0 - float(min_rel_improvement)):
                entry["skipped"] = "improvement_below_threshold"
            else:
                delta["heads"][head] = {"temperature": float(grid[best])}
            stats[head] = entry
        return (delta if delta["heads"] else {}), stats

    def train_swarm_adapter_delta(
        self,
        base_adapter: SwarmResidualAdapter,
        *,
        trainable: Sequence[str] = SWARM_ADAPTER_TRAINABLE,
        steps: int = 300,
        lr: float = 0.05,
        l2: float = 1e-3,
        logit_scale: float = 10.0,
        min_samples: int = 20,
        seed: int = 0,
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Fine-tune the adapter's junk residual path on contributed (Way-2, Way-3, p_junk, verified junk) samples.

        Trains only `trainable` tensors on a copy of `base_adapter` (inputs mirror the pipeline's
        `evaluate_swarm` call: zero pooled/side21 context) and returns additive float32 deltas.
        """
        import torch
        import torch.nn.functional as TF

        exp_w2 = int(getattr(base_adapter, "d_way2", 8))
        exp_w3 = int(getattr(base_adapter, "d_way3", 6))
        samples: List[Dict[str, Any]] = []
        seen_swarm_keys: set = set()
        for doc in self.observations:
            voter_id = self._extract_verified_voter_id(doc)
            if not voter_id:
                continue
            sf = doc.get("swarm_features")
            if not isinstance(sf, dict):
                continue
            if len(sf.get("way2", [])) != exp_w2 or len(sf.get("way3", [])) != exp_w3:
                continue
            rec_bh = str((doc.get("record") or {}).get("blinded_hash", "")).strip()
            sf_key = (
                (voter_id, rec_bh)
                if rec_bh
                else (
                    voter_id,
                    tuple(round(float(v), 6) for v in sf.get("way2", [])),
                    tuple(round(float(v), 6) for v in sf.get("way3", [])),
                    bool(sf.get("junk")),
                )
            )
            if sf_key in seen_swarm_keys:
                continue
            seen_swarm_keys.add(sf_key)
            samples.append(sf)
        n_pos = sum(1 for s in samples if s.get("junk"))
        stats: Dict[str, Any] = {"n_samples": len(samples), "n_junk": n_pos}
        if len(samples) < min_samples or n_pos == 0 or n_pos == len(samples):
            stats["skipped"] = "insufficient_or_single_class_samples"
            return None, stats
        torch.manual_seed(int(seed))
        adapter = copy.deepcopy(base_adapter)
        adapter.eval()
        named = dict(adapter.named_parameters())
        missing = [n for n in trainable if n not in named]
        if missing:
            raise ValueError(f"unknown trainable tensors: {missing}")
        for q in adapter.parameters():
            q.requires_grad_(False)
        params = [named[n] for n in trainable]
        for q in params:
            q.requires_grad_(True)
        orig = [q.detach().clone() for q in params]

        w2 = torch.tensor([s["way2"] for s in samples], dtype=torch.float32)
        w3 = torch.tensor([s["way3"] for s in samples], dtype=torch.float32)
        y = torch.tensor([1.0 if s["junk"] else 0.0 for s in samples], dtype=torch.float32)
        pj = torch.tensor([min(max(float(s["p_junk_neural"]), 1e-6), 1.0 - 1e-6) for s in samples], dtype=torch.float32)
        zl = torch.log(pj / (1.0 - pj))
        n = len(samples)
        pooled = torch.zeros((n, adapter.d_pooled))
        side21 = torch.zeros((n, adapter.d_side21))

        def adapter_margin() -> "torch.Tensor":
            out = adapter.forward(pooled, side21, {"junk": torch.stack([torch.zeros_like(zl), zl], dim=-1)}, w2, w3)
            return (out["junk"][:, 1] - out["junk"][:, 0]) - zl

        with torch.no_grad():
            stats["bce_before"] = round(float(TF.binary_cross_entropy_with_logits(adapter_margin() * logit_scale, y)), 6)
        opt = torch.optim.Adam(params, lr=float(lr))
        for _ in range(int(steps)):
            loss = TF.binary_cross_entropy_with_logits(adapter_margin() * logit_scale, y)
            loss = loss + float(l2) * sum(((q - o) ** 2).sum() for q, o in zip(params, orig))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                if "head_scales.junk" in named:
                    named["head_scales.junk"].clamp_(0.0, 2.0)
        with torch.no_grad():
            stats["bce_after"] = round(float(TF.binary_cross_entropy_with_logits(adapter_margin() * logit_scale, y)), 6)
            deltas = {n_: (named[n_].detach() - o).cpu().numpy().astype(np.float32) for n_, o in zip(trainable, orig)}
        stats["delta_l2"] = {k: round(float(np.linalg.norm(v)), 6) for k, v in deltas.items()}
        return deltas, stats

    def build_update_payload(
        self,
        *,
        zero_cue_catalog_delta: Optional[Mapping[str, Mapping[str, str]]] = None,
        calibration_delta: Optional[Mapping[str, Any]] = None,
        swarm_adapter_delta: Optional[Mapping[str, np.ndarray]] = None,
    ) -> bytes:
        """Canonical JSON update bundle; adapter deltas travel as SHA-256-pinned base64 SafeTensors."""
        doc: Dict[str, Any] = {"schema": DHT_UPDATE_SCHEMA, "epoch": self.epoch}
        if zero_cue_catalog_delta:
            _validate_zero_cue_delta(dict(zero_cue_catalog_delta))
            doc["zero_cue_catalog_delta"] = {m: dict(v) for m, v in zero_cue_catalog_delta.items()}
        if calibration_delta:
            _validate_calibration_delta(dict(calibration_delta))
            doc["calibration_delta"] = copy.deepcopy(dict(calibration_delta))
        if swarm_adapter_delta:
            raw = encode_safetensors(swarm_adapter_delta, metadata={"kind": "additive_delta", "base": "swarm_adapter.pt"})
            doc["swarm_adapter_safetensors_delta"] = {
                "encoding": "safetensors+base64",
                "data": base64.b64encode(raw).decode("ascii"),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        if len(doc) == 2:
            raise ValueError("update payload has no components")
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def sign_manifest(
        payload_bytes: bytes,
        *,
        seq: int,
        signers: Sequence[Tuple[Any, str, bytes, str]],
        version: str,
        min_asn_quorum: int = 5,
        bundle_format: str = "json",
        timestamp: Optional[int] = None,
    ) -> ThresholdDHTManifest:
        """Produce a BEP-46 `ThresholdDHTManifest` signed by every signer with Ed25519 + WOTS+-SHAKE256.

        `infohash` is the SHA-1 content id of the payload (BEP-46 mutable-item target derivation is out of scope).
        """
        unsigned = ThresholdDHTManifest(
            seq=int(seq),
            version=str(version),
            infohash=hashlib.sha1(payload_bytes).hexdigest(),
            bundle_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            bundle_format=str(bundle_format),
            min_asn_quorum=int(min_asn_quorum),
            timestamp=int(timestamp if timestamp is not None else time.time()),
            signatures={},
        )
        msg = unsigned.canonical_payload()
        sigs: Dict[str, str] = {}

        class _PQSignatureMap(dict):
            def __init__(self) -> None:
                super().__init__()
                self._root_to_leaf: Dict[str, str] = {}

            def __getitem__(self, key: str) -> str:
                norm = str(key).lower()
                if norm in self._root_to_leaf and not super().__contains__(key):
                    return super().__getitem__(self._root_to_leaf[norm])
                return super().__getitem__(key)

        pq_sigs = _PQSignatureMap()
        for ed_priv, ed_pub_hex, pq_priv, _pq_pub_hex in signers:
            sigs[ed_pub_hex.lower()] = ed_priv.sign(msg).hex()
            leaf_pub, pq_sig_bytes, _, _ = WOTSPlusSHAKE256.allocate_and_sign(
                pq_priv, msg, reserve_fresh=True
            )
            leaf_hex = leaf_pub.hex()
            pq_sigs[leaf_hex] = pq_sig_bytes.hex()
            if _pq_pub_hex:
                pq_sigs._root_to_leaf[_pq_pub_hex.lower()] = leaf_hex
        return dataclasses.replace(unsigned, signatures=sigs, pq_signatures=pq_sigs)


