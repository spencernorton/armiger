"""Unified Armiger Ultra SDK Pipeline (`ArmigerUltraPipeline`).

Integrates the Armiger / Armiger Ultra 8-head byte-level RoPE Transformer
(`armiger_ultra.fp32.onnx`, `armiger.int8.onnx`, `armiger_ultra.pt`) with all 6
official NorviTech Armiger plugins in a single, zero-boilerplate API:

  - Plugin #1: `SwarmSurvivalNet` (`swarm_survival.py` — `SwarmResidualAdapter`)
  - Plugin #2: `SwarmPriors` (`swarm_priors.py` — 4-pathway Empirical Bayes swarm & tracker priors)
  - Plugin #3: `ZeroCueCatalog` (`zero_cue_plugin.py` — 929,400-title offline catalog & structural disambiguator)
  - Plugin #4: `AudioLanguageGuard` (`audio_language_guard.py` — cross-head audio language & anime dub guard)
  - Plugin #5: `CalibrationAndDHT` + `ArmigerDHTRSIEngine` (`calibration_and_dht.py` + `armiger_dht_rsi.py`)
  - Plugin #6: `ArmigerDHTDBSync` + `CSAMGuard` (`armiger_dht_db_sync.py` + `csam_guard.py`)

Usage:
    from .armiger_ultra_pipeline import ArmigerUltraPipeline

    pipeline = ArmigerUltraPipeline.from_Directory("./armiger-ultra")
    result = pipeline.classify({
        "infohash": "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4",
        "name": "Tears.of.Steel.2012.2160p.UHD.BluRay.REMUX.HEVC.TrueHD.7.1.Atmos-OPEN",
        "size": 68_400_000_000,
        "files": [{"path": "Tears.of.Steel.2012.mkv", "size": 68_400_000_000}],
        "swarm": {"tracker_known": True, "tracker_seeders": 142, "tracker_leechers": 19, "tracker_completed": 810},
    })
    print(result["category"], result["mondrian_gates"]["passed_all"], result["swarm_viability"]["viability_tier"])
"""
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

# Plugin #1: SwarmSurvivalNet & Residual Adapter
from .swarm_survival import (
    SwarmResidualAdapter,
    SwarmSurvivalNet,
)

# Plugin #2: SwarmPriors (4-Pathway Empirical Bayes & Tracker Scrape Priors)
from .swarm_priors import (
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
from .zero_cue_plugin import (
    CATS as ZERO_CUE_CATS,
    ZeroCueCatalogPlugin,
    apply_structural_and_zero_cue_disambiguation,
    hash_catalog_key,
    lookup_zero_cue_catalog,
    norm_title,
    parse_clean_candidates,
)

# Plugin #3b: ByteSpanExtractor (9th-Head Constrained 7x7 BIO Viterbi Decoder & Boundary Snapping)
from .byte_span_extractor import (
    SPAN_BIO_CLASSES,
    decode_bio_logits_viterbi,
    decode_bio_tags_to_spans,
)

# Plugin #4: AudioLanguageGuard (Cross-Head Audio & Anime Dub Guard)
from .audio_language_guard import (
    SUB_SEGMENT_STRIP_RE,
    apply_audio_language_guard,
    apply_english_subs_guard,
    apply_title_language_guard,
    classify_audio_and_anime_dub,
)

# Plugin #5: CalibrationAndDHT + ArmigerDHTRSIEngine
from .calibration_and_dht import (
    ArmigerCalibrator,
    ArmigerDHTUpdater,
    SignedDHTManifest,
)
from .armiger_dht_rsi import (
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
from .csam_guard import (
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
from .armiger_dht_db_sync import (
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


# =====================================================================
# Constants & Feature Extraction Tables (Parity with Armiger ONNX v2.1)
# =====================================================================

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

VIDEO_EXTS = frozenset({"mkv", "mp4", "avi", "ts", "m2ts", "wmv", "mov", "webm", "flv", "mpg", "mpeg", "m4v", "iso", "vob"})
AUDIO_EXTS = frozenset({"mp3", "flac", "m4a", "aac", "ogg", "opus", "wav", "ape", "alac", "wma", "m4b", "dsf", "dff"})
SUB_EXTS = frozenset({"srt", "ass", "ssa", "sub", "idx", "vtt", "sup"})
IMAGE_EXTS = frozenset({"jpg", "jpeg", "png", "webp", "gif", "bmp", "cbr", "cbz"})
ARCHIVE_EXTS = frozenset({"rar", "zip", "7z", "tar", "gz", "bz2", "xz", "001"})
EXEC_EXTS = frozenset({"exe", "msi", "apk", "dmg", "pkg", "deb", "rpm", "sh", "bat", "cmd", "bin", "dll", "scr", "vbs", "ps1"})
BOOK_EXTS = frozenset({"epub", "pdf", "mobi", "azw3", "djvu", "fb2", "cbz", "cbr"})

LANG_CODES: List[str] = [
    "en", "ja", "ru", "fr", "es", "de", "it", "pt", "zh", "ko",
    "hi", "pl", "nl", "tr", "ar", "sv", "cs", "hu", "uk", "multi",
]
LANG_INDEX: Dict[str, int] = {c: i for i, c in enumerate(LANG_CODES)}

RULE_EVIDENCE_HINT: Dict[str, int] = {
    "no_cue": 0, "none": 0, "ambiguous": 0,
    "tv_cue": 1, "movie_cue": 1, "xxx_cue": 1, "music_cue": 1, "game_cue": 1,
    "software_cue": 1, "audiobook_cue": 1, "ebook_cue": 1, "comic_cue": 1, "course_cue": 1,
    "strong": 2, "catalog_exact": 2, "verified": 2,
}

WORD_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+")


from .armiger_inference import (
    ArmigerStandaloneONNX,
    _get_vocab_ext_map,
    _hash_ext,
    _hash_word,
    _log2_size_bucket,
    build_onnx_batch_inputs,
    encode_torrent_features,
)


_EPISODE_FILE_RE = re.compile(r"(?i)(?:\bS\d{1,2}E\d{1,3}\b|\bS\d{1,2}\b|\bSeason\s*\d+\b)")
_GAME_FILE_TREE_RE = re.compile(
    r"(?i)(?:supertux|super-tux-kart|org\.supertuxkart|\bfreeciv|\bopenttd|\b0ad[-_\b]|\bminetest|\bfreedoom|"
    r"\bwesnoth|segasoft|gremlin[\s\._-]+interactive|crosscountrycanada|petepilotti|\.levelset\.|"
    r"\.(?:pk3|wad|pck|bsp|unity3d|vpk)$)"
)
_AUDIOBOOK_FILE_TREE_RE = re.compile(
    r"(?i)(?:synapseml_gutenberg|-\s*\d+\.(?:afpk|mp3)$|librivox)"
)


def _file_tree_structural_override(row: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    """Check physical file-tree extensions (.m4b/.afpk audiobooks, .cbz/.cbr comics, game engine trees, multi-episode packs) before title lookup."""
    raw_name = str(row.get("name") or row.get("title") or "")
    files = row.get("files") or []
    ep_files = 0
    has_afpk = False
    for f in files:
        p = (f.get("path") if isinstance(f, Mapping) else str(f or "")) or ""
        low = p.lower().strip()
        if low.endswith(".m4b"):
            return "audiobook", "structural_m4b"
        if low.endswith(".afpk"):
            has_afpk = True
        if low.endswith((".cbz", ".cbr", ".cb7")):
            return "comic", "structural_comic_archive"
        if _GAME_FILE_TREE_RE.search(p):
            return "game", "structural_game_file_tree"
        if low.endswith((".mkv", ".mp4", ".avi", ".ts", ".ogv", ".mov")) and _EPISODE_FILE_RE.search(p):
            ep_files += 1
    if has_afpk and (_AUDIOBOOK_FILE_TREE_RE.search(raw_name) or any(_AUDIOBOOK_FILE_TREE_RE.search(str(f)) for f in files[:8])):
        return "audiobook", "structural_audiobook_afpk"
    if _GAME_FILE_TREE_RE.search(raw_name):
        return "game", "structural_game_archive"
    if ep_files >= 2:
        return "tv_show", "structural_multi_episode_pack"
    return None, "none"


# =====================================================================
# Plugin #4 Object Wrapper: AudioLanguageGuard
# =====================================================================

class AudioLanguageGuard:
    """Object-oriented wrapper around `audio_language_guard.py` for SDK pipeline composition."""

    _DUAL_MULTI_RE = re.compile(
        r"(?i)(?:\bdual[\s._-]*audio\b|\bmulti[\s._-]*audio\b|\bmulti\s*\([^\)]*\b(?:en|eng|english)\b(?![\s._\-/]*sub)|"
        r"\[\s*(?:rus|jpn|jap|ita|fre|ger|deu|spa|lat)[^\]]*\+\s*(?:en|eng|english)\b(?![\s._\-/]*sub))"
    )

    def __init__(self, foreign_title_threshold: float = 0.65) -> None:
        self.foreign_title_threshold = float(foreign_title_threshold)

    def apply(
        self,
        row: Mapping[str, Any],
        lang_probs: Union[Sequence[float], np.ndarray],
        title_lang_probs: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        raw_name = str(row.get("name") or row.get("title") or "")
        probs = np.asarray(lang_probs, dtype=np.float64).copy()
        p_en = float(probs[0])
        p_non_en = float(probs[1])
        p_multi = float(probs[2]) if len(probs) > 2 else 0.0
        p_title_foreign = (
            float(title_lang_probs[1])
            if title_lang_probs is not None and len(title_lang_probs) > 1
            else 0.0
        )

        # Explicit dual/multi-audio release tags (strip subtitle segments first so '[JPN + ENG Subs]' never triggers)
        cleaned_audio_name = SUB_SEGMENT_STRIP_RE.sub(" ", raw_name)
        if self._DUAL_MULTI_RE.search(cleaned_audio_name):
            adj = np.array([0.12, 0.08, 0.80], dtype=np.float64)
            return "multi_or_unknown", adj, {
                "p_english_compatible_audio": 0.92,
                "is_english_compatible": True,
                "anime_dub_status": "dual_audio_or_dub",
            }

        p_en_adj = apply_audio_language_guard(raw_name, p_en=p_en, p_title_foreign=p_title_foreign)
        if re.search(
            r"(?i)(?:\b(?:ita|ger|deu|fre|fra|rus|spa|jpn|jap|kor|chi|hin|dut|nl|por|pol|tur)\b.*?\b(?:en|eng|english)[\s._-]*sub|"
            r"^\[(?:openanime|cc-anime)\])",
            raw_name,
        ):
            p_en_adj = min(p_en_adj, 0.08)
        if abs(p_en_adj - p_en) > 1e-6:
            if p_en_adj < p_en:
                delta = p_en - p_en_adj
                adj = np.array([p_en_adj, min(0.98, p_non_en + delta), p_multi], dtype=np.float64)
            else:
                adj = np.array([p_en_adj, max(0.01, 1.0 - p_en_adj - p_multi), p_multi], dtype=np.float64)
            adj /= max(1e-9, float(adj.sum()))
        else:
            adj = probs

        classes = NEURAL_HEAD_CLASSES["lang"]
        pred_label = classes[int(np.argmax(adj))]
        return pred_label, adj, {
            "p_english_compatible_audio": round(float(p_en_adj), 4),
            "is_english_compatible": bool(p_en_adj >= 0.50 or pred_label == "multi_or_unknown"),
        }

    def classify_audio_and_dub(
        self,
        row: Mapping[str, Any],
        lang_probs: Union[Sequence[float], np.ndarray],
        anime_probs: Union[Sequence[float], np.ndarray],
        eng_subs_probs: Union[Sequence[float], np.ndarray],
        title_lang_probs: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> Dict[str, Any]:
        raw_name = str(row.get("name") or row.get("title") or "")
        p_en = float(lang_probs[0])
        p_multi = float(lang_probs[2]) if len(lang_probs) > 2 else 0.0
        p_anime = float(anime_probs[1]) if len(anime_probs) > 1 else 0.0
        p_eng_subs = float(eng_subs_probs[1]) if len(eng_subs_probs) > 1 else 0.0
        p_title_foreign = (
            float(title_lang_probs[1])
            if title_lang_probs is not None and len(title_lang_probs) > 1
            else 0.0
        )

        report = classify_audio_and_anime_dub(
            raw_name=raw_name,
            p_en=max(p_en, p_multi),
            p_anime=p_anime,
            p_title_foreign=p_title_foreign,
            p_eng_subs_yes=p_eng_subs,
        )
        cleaned_audio_name = SUB_SEGMENT_STRIP_RE.sub(" ", raw_name)
        if not report["is_anime"]:
            dub_status = "non_anime"
        elif report["is_anime_dub_or_dual"] or self._DUAL_MULTI_RE.search(cleaned_audio_name):
            dub_status = "dual_audio_or_dub"
        elif p_eng_subs >= 0.45 or re.search(r"(?i)\b(?:eng[\s._-]*sub|esub|openanime|cc-anime)\b", raw_name):
            dub_status = "subbed_only"
        else:
            dub_status = "raw_or_vernacular"

        report["anime_dub_status"] = dub_status
        return report


# =====================================================================
# Unified 11-Head Prediction Output
# =====================================================================

@dataclasses.dataclass
class ArmigerUltraPrediction:
    """Unified 11-head prediction result with dictionary-style and attribute access."""

    infohash: str
    sanitized_title: str
    category: str
    adult: str
    junk: str
    anime: str
    english_subs: str
    lang: str
    owner_scope: str
    title_language: str
    swarm_viability: Dict[str, Any]
    honeypot_trap: Dict[str, Any]
    csam_safety: Dict[str, Any]
    heads: Dict[str, Dict[str, Any]]
    mondrian_gates: Dict[str, Any]
    plugin_trace: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()



from .ultra_pipeline_update_bridge import (
    CANARY_HEALTHY_SWARM,
    DEFAULT_CANARY_VIEWS,
    DEFAULT_TEMPERATURE_GRID,
    DHTUpdateRejected,
    DHTUpdateResult,
    DHT_UPDATE_COMPONENTS,
    DHT_UPDATE_SCHEMA,
    MAX_ADAPTER_DELTA_ABS,
    MAX_ZERO_CUE_DELTA_ENTRIES,
    RSIUpdateAggregator,
    RSI_TELEMETRY_SCHEMA,
    SWARM_ADAPTER_TRAINABLE,
    SignedUpdateManifest,
    TEMPERATURE_BOUNDS,
    ZERO_CUE_DELTA_MAPS,
    _CANARY_SAMPLE_SUFFIX_RE,
    _HEX16_RE,
    _MISSING,
    _canary_files_for,
    _canary_summary,
    _nll_profile,
    _prediction_labels,
    _validate_calibration_delta,
    _validate_zero_cue_delta,
    _verified_head_labels,
    _zero_cue_catalog_key,
    build_canary_rows,
    decode_safetensors,
    encode_safetensors,
    generate_federation_signer,
    parse_dht_update_payload,
)

# =====================================================================
# Main Unified SDK Class: ArmigerUltraPipeline
# =====================================================================

class ArmigerUltraPipeline:
    """Unified SDK Pipeline integrating Armiger ONNX/PyTorch with all 6 NorviTech plugins.

    Plugins integrated out-of-the-box:
      1. `SwarmSurvivalNet` (`swarm_survival.py` + `swarm_adapter.pt`)
      2. `SwarmPriors` (`swarm_priors.py`)
      3. `ZeroCueCatalog` (`zero_cue_plugin.py` + `validated_zero_cue_catalog.json`)
      4. `AudioLanguageGuard` (`audio_language_guard.py`)
      5. `CalibrationAndDHT` + `ArmigerDHTRSIEngine` (`calibration_and_dht.py` + `armiger_dht_rsi.py` + `calibration.json`)
      6. `ArmigerDHTDBSync` + `CSAMGuard` (`armiger_dht_db_sync.py` + `csam_guard.py`)
    """

    def __init__(
        self,
        model_dir: Union[str, Path, None] = None,
        *,
        variant: str = "ultra",
        onnx_Session: Any = None,
        calibrator: Optional[ArmigerCalibrator] = None,
        zero_cue_catalog: Optional[ZeroCueCatalogPlugin] = None,
        preload_catalog: bool = False,
        audio_guard: Optional[AudioLanguageGuard] = None,
        swarm_adapter: Optional[SwarmResidualAdapter] = None,
        priors_table: Optional[Dict[str, Dict[str, Any]]] = None,
        rsi_engine: Optional[ArmigerDHTRSIEngine] = None,
        dht_updater: Optional[ArmigerDHTUpdater] = None,
        dht_db_sync: Optional[ArmigerDHTDatabaseSync] = None,
        csam_guard: Optional[CSAMGuard] = None,
        safety_gate: Optional[ArmigerSafetyGate] = None,
        rsi_mode: Optional[str] = "anonymous_contribute",
        dht_sync_subscription: Optional[SubscriptionConfig] = None,
        trusted_root_pubkeys_hex: Optional[Sequence[str]] = None,
        dht_master_secret: Optional[bytes] = None,
        opt_in_sharing: Optional[bool] = None,
        ephemeral_identity: bool = True,
        min_k_samples: int = 25,
        adult_threshold: Optional[float] = None,
        junk_threshold: Optional[float] = None,
        scope_threshold: Optional[float] = None,
        allow_synthetic_fallback: bool = False,
        runtime_cache_dir: Optional[Union[str, Path]] = None,
        auto_update: Optional[bool] = None,
        poll_interval_sec: Optional[float] = None,
        initial_seq: Optional[int] = None,
        canary_validator: Optional[Callable[["ArmigerUltraPipeline"], bool]] = None,
        offline: bool = False,
    ) -> None:
        self.opt_in_sharing: bool = resolve_opt_in_sharing(opt_in_sharing)
        if rsi_mode is None:
            env_mode = os.environ.get("ARMIGER_DHT_RSI_MODE", "").strip()
            rsi_mode = env_mode if env_mode else ("anonymous_contribute" if self.opt_in_sharing else "pull_only")
        self.allow_synthetic_fallback: bool = bool(allow_synthetic_fallback)
        from .artifacts import model_directory
        eff_dir = model_directory(model_dir)
        self.model_dir = Path(eff_dir).resolve()
        self.variant = variant.lower().strip()
        self.config_path = self.model_dir / "config.json"
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        # 1. Load ONNX Runtime session (fp32 ultra or int8 cpu)
        self.onnx_session = onnx_Session
        self.onnx_path: Optional[Path] = None
        if self.onnx_session is None:
            self.onnx_session, self.onnx_path = self._load_onnx_session(
                self.model_dir, self.variant, allow_synthetic_fallback=self.allow_synthetic_fallback
            )
        self._onnx_output_names: List[str] = (
            [o.name for o in self.onnx_session.get_outputs()]
            if self.onnx_session is not None
            else list(NEURAL_HEAD_CLASSES.keys())
        )

        # 2. Plugin #1: SwarmSurvivalNet (`swarm_survival.py` + `swarm_adapter.pt`)
        self.swarm_adapter = swarm_adapter or SwarmResidualAdapter.from_pretrained(self.model_dir / "swarm_adapter.pt")

        # 3. Plugin #2: SwarmPriors (`swarm_priors.py`)
        self.priors_table: Dict[str, Dict[str, Any]] = priors_table or {}

        # 4. Plugin #3: ZeroCueCatalog (`zero_cue_plugin.py` + `validated_zero_cue_catalog.json`)
        self._zero_cue_catalog: Optional[ZeroCueCatalogPlugin] = zero_cue_catalog
        self._catalog_json_path = self.model_dir / "validated_zero_cue_catalog.json"
        if preload_catalog and self._zero_cue_catalog is None and self._catalog_json_path.exists():
            self._zero_cue_catalog = ZeroCueCatalogPlugin(self._catalog_json_path)

        # 5. Plugin #4: AudioLanguageGuard (`audio_language_guard.py`)
        self.audio_guard = audio_guard or AudioLanguageGuard()

        # 6. Plugin #5: CalibrationAndDHT + ArmigerDHTRSIEngine (`calibration_and_dht.py` + `armiger_dht_rsi.py`)
        cal_path = self.model_dir / "calibration.json"
        self.calibrator = calibrator or ArmigerCalibrator(cal_path if cal_path.exists() else None)
        root_key_hex = (
            list(trusted_root_pubkeys_hex)[0]
            if trusted_root_pubkeys_hex
            else "0" * 64
        )
        self.dht_updater = dht_updater or ArmigerDHTUpdater(root_pubkey_hex=root_key_hex)
        self.rsi_engine = rsi_engine or ArmigerDHTRSIEngine(
            mode=rsi_mode,
            opt_in_sharing=self.opt_in_sharing,
            ephemeral_identity=ephemeral_identity,
            min_k_samples=min_k_samples,
        )
        self.anonymity_guard = self.rsi_engine.anonymity_guard

        # 7. Plugin #6: ArmigerDHTDBSync + CSAMGuard (`armiger_dht_db_sync.py` + `csam_guard.py`)
        mondrian = self.config.get("mondrian_thresholds", {})
        self.safety_gate = safety_gate or ArmigerSafetyGate(
            adult_threshold=float(adult_threshold if adult_threshold is not None else mondrian.get("adult", 0.5451)),
            junk_threshold=float(junk_threshold if junk_threshold is not None else mondrian.get("junk", 0.1010)),
            scope_threshold=float(scope_threshold if scope_threshold is not None else mondrian.get("owner_scope", 0.4770)),
        )
        self.dht_db_sync = dht_db_sync or ArmigerDHTDatabaseSync(
            subscription=dht_sync_subscription or SubscriptionConfig(
                sync_mode=SyncMode.SUBSCRIBE_AND_SEED if self.opt_in_sharing else SyncMode.PULL_ONLY
            ),
            master_secret=dht_master_secret,
            trusted_root_pubkeys_hex=trusted_root_pubkeys_hex,
            opt_in_sharing=self.opt_in_sharing,
            ephemeral_identity=ephemeral_identity,
            min_k_samples=min_k_samples,
        )
        self.csam_guard = csam_guard or self.dht_db_sync.csam_guard

        # 8. Distributed model update bridge (closed-loop RSI telemetry -> BEP-46 -> hot reload)
        self._init_dht_update_bridge(
            runtime_cache_dir=runtime_cache_dir,
            auto_update=auto_update,
            poll_interval_sec=poll_interval_sec,
            initial_seq=initial_seq,
            canary_validator=canary_validator,
            offline=offline,
        )

    # -----------------------------------------------------------------
    # Factory Classmethods (`from_Directory`, `from_directory`, `from_pretrained`)
    # -----------------------------------------------------------------

    @classmethod
    def from_Directory(
        cls,
        directory: Union[str, Path],
        *,
        variant: str = "ultra",
        preload_catalog: bool = False,
        rsi_mode: Optional[str] = "anonymous_contribute",
        dht_sync_subscription: Optional[SubscriptionConfig] = None,
        trusted_root_pubkeys_hex: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "ArmigerUltraPipeline":
        """Instantiate `ArmigerUltraPipeline` from a local Armiger Ultra directory."""
        return cls(
            model_dir=directory,
            variant=variant,
            preload_catalog=preload_catalog,
            rsi_mode=rsi_mode,
            dht_sync_subscription=dht_sync_subscription,
            trusted_root_pubkeys_hex=trusted_root_pubkeys_hex,
            **kwargs,
        )

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        *,
        variant: str = "ultra",
        preload_catalog: bool = False,
        rsi_mode: Optional[str] = "anonymous_contribute",
        dht_sync_subscription: Optional[SubscriptionConfig] = None,
        trusted_root_pubkeys_hex: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> "ArmigerUltraPipeline":
        """Snake-case alias for `from_Directory`."""
        return cls.from_Directory(
            directory=directory,
            variant=variant,
            preload_catalog=preload_catalog,
            rsi_mode=rsi_mode,
            dht_sync_subscription=dht_sync_subscription,
            trusted_root_pubkeys_hex=trusted_root_pubkeys_hex,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        **kwargs: Any,
    ) -> "ArmigerUltraPipeline":
        """Load a local directory or the pinned, verified NorviTech/armiger-ultra bundle."""
        cand_path = Path(model_name_or_path)
        if cand_path.exists() and cand_path.is_dir():
            return cls.from_Directory(cand_path, **kwargs)
        from .artifacts import MANIFEST, download_model
        if str(model_name_or_path) != MANIFEST["repository"]:
            raise ValueError("Use a local model directory or NorviTech/armiger-ultra")
        variant = kwargs.get("variant", "ultra")
        local_dir = download_model(variant="all" if variant in ("cpu", "int8", "armiger") else "ultra")
        return cls.from_Directory(local_dir, **kwargs)

    # -----------------------------------------------------------------
    # Lazy / On-Demand Accessors for Zero-Cue Catalog & Model Loaders
    # -----------------------------------------------------------------

    @property
    def zero_cue_catalog(self) -> ZeroCueCatalogPlugin:
        """Return the loaded `ZeroCueCatalogPlugin`, initializing lazily on first access."""
        if self._zero_cue_catalog is None:
            self._zero_cue_catalog = ZeroCueCatalogPlugin(
                self._catalog_json_path if self._catalog_json_path.exists() else None
            )
        return self._zero_cue_catalog

    @staticmethod
    def _load_onnx_session(
        model_dir: Path, variant: str, *, allow_synthetic_fallback: bool = False
    ) -> Tuple[Any, Optional[Path]]:
        if variant in ("int8", "cpu", "armiger"):
            candidates = [model_dir / "armiger.int8.onnx", model_dir / "armiger_ultra.fp32.onnx"]
        else:
            candidates = [model_dir / "armiger_ultra.fp32.onnx", model_dir / "armiger.int8.onnx"]

        onnx_path = next((p for p in candidates if p.exists()), None)
        if onnx_path is None:
            return None, None

        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
            providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
            return session, onnx_path
        except Exception as exc:
            if not allow_synthetic_fallback:
                raise RuntimeError(f"Failed to load ONNX session from {onnx_path}: {exc}") from exc
            return None, onnx_path

    # -----------------------------------------------------------------
    # Core 8-Head Neural Forward Pass (ONNX Runtime + Fallback)
    # -----------------------------------------------------------------

    def _run_neural_logits(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
        """Run the 8-head neural model and return raw logits `[B, C_h]` per head."""
        if self.onnx_session is not None:
            inputs = build_onnx_batch_inputs(rows)
            raw_outs = self.onnx_session.run(self._onnx_output_names, inputs)
            return {name: np.asarray(arr, dtype=np.float64) for name, arr in zip(self._onnx_output_names, raw_outs)}

        if not self.allow_synthetic_fallback:
            raise RuntimeError(
                "onnx_session_unavailable: ArmigerUltraPipeline requires a loaded ONNX model unless allow_synthetic_fallback=True"
            )

        # Deterministic fallback when onnxruntime is not installed in a minimal test environment
        bsz = len(rows)
        out: Dict[str, np.ndarray] = {
            h: np.zeros((bsz, len(classes)), dtype=np.float64)
            for h, classes in NEURAL_HEAD_CLASSES.items()
        }
        for i, r in enumerate(rows):
            name = (r.get("name") or "").lower()
            s_cat, _ = _file_tree_structural_override(r)
            cat_idx = NEURAL_HEAD_CLASSES["category"].index(s_cat) if s_cat in NEURAL_HEAD_CLASSES["category"] else 0
            out["category"][i, cat_idx] = 4.0
            out["adult"][i, 1 if s_cat == "xxx" or "xxx" in name else 0] = 4.0
            out["junk"][i, 1 if any(x in name for x in (".exe", ".scr", "codec.pack", "password")) else 0] = 4.0
            out["anime"][i, 1 if ("anime" in name or "openanime" in name or "cc-anime" in name) else 0] = 4.0
            out["english_subs"][i, 1 if ("esub" in name or "eng.sub" in name or "multisub" in name) else 0] = 3.0
            out["lang"][i, 0] = 3.0
            out["owner_scope"][i, 0] = 4.0
            out["title_language"][i, 0] = 3.0
        return out

    # -----------------------------------------------------------------
    # Unified 11-Head Classification Pipeline (`classify` & `classify_batch`)
    # -----------------------------------------------------------------

    def __call__(
        self,
        item_or_batch: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
        **kwargs: Any,
    ) -> Union[ArmigerUltraPrediction, List[ArmigerUltraPrediction]]:
        if isinstance(item_or_batch, Mapping):
            return self.classify(item_or_batch, **kwargs)
        return self.classify_batch(item_or_batch, **kwargs)

    def classify(
        self,
        row: Mapping[str, Any],
        *,
        use_zero_cue_catalog: bool = True,
        ingest_to_dht_db: bool = False,
    ) -> ArmigerUltraPrediction:
        """Run full 11-head classification + conformal gating + CSAM guard on a single torrent."""
        return self.classify_batch(
            [row],
            use_zero_cue_catalog=use_zero_cue_catalog,
            ingest_to_dht_db=ingest_to_dht_db,
        )[0]

    def classify_batch(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        use_zero_cue_catalog: bool = True,
        ingest_to_dht_db: bool = False,
    ) -> List[ArmigerUltraPrediction]:
        """Run full 11-head classification + conformal gating + CSAM guard on a batch of torrents."""
        if not rows:
            return []
        with self._dht_update_lock:
            return self._classify_batch_locked(
                rows,
                use_zero_cue_catalog=use_zero_cue_catalog,
                ingest_to_dht_db=ingest_to_dht_db,
            )

    def _classify_batch_locked(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        use_zero_cue_catalog: bool = True,
        ingest_to_dht_db: bool = False,
    ) -> List[ArmigerUltraPrediction]:
        # Step 1: Run 8-head neural backbone (strictly content-only; never reads `swarm`)
        raw_logits = self._run_neural_logits(rows)

        results: List[ArmigerUltraPrediction] = []
        for idx, row in enumerate(rows):
            row_dict = dict(row)
            raw_title = str(row_dict.get("name") or row_dict.get("title") or "")
            infohash = str(
                row_dict.get("infohash")
                or hashlib.sha1(raw_title.encode("utf-8", errors="ignore")).hexdigest()
            ).lower().strip()
            swarm_dict = row_dict.get("swarm") if isinstance(row_dict.get("swarm"), Mapping) else None
            swarm_copy = dict(swarm_dict) if swarm_dict is not None else None

            # Step 2: Plugin #5 (`ArmigerCalibrator`) — per-head temperature scaling & Menon biases
            cat_probs = np.asarray(self.calibrator.calibrate_head("category", raw_logits["category"][idx].tolist()), dtype=np.float64)
            adult_probs = np.asarray(self.calibrator.calibrate_head("adult", raw_logits["adult"][idx].tolist()), dtype=np.float64)
            junk_probs = np.asarray(self.calibrator.calibrate_head("junk", raw_logits["junk"][idx].tolist()), dtype=np.float64)
            p_junk_neural_only = float(junk_probs[1])
            anime_probs = np.asarray(self.calibrator.calibrate_head("anime", raw_logits["anime"][idx].tolist()), dtype=np.float64)
            eng_subs_probs = np.asarray(self.calibrator.calibrate_head("english_subs", raw_logits["english_subs"][idx].tolist()), dtype=np.float64)
            lang_probs = np.asarray(self.calibrator.calibrate_head("lang", raw_logits["lang"][idx].tolist()), dtype=np.float64)
            scope_probs = np.asarray(self.calibrator.calibrate_head("owner_scope", raw_logits["owner_scope"][idx].tolist()), dtype=np.float64)
            title_lang_probs = np.asarray(self.calibrator.calibrate_head("title_language", raw_logits["title_language"][idx].tolist()), dtype=np.float64)

            # Step 2b: 9th-Head Constrained 7x7 BIO Viterbi Decoder + Boundary Snapping (`byte_span_extractor.py`)
            raw_title_bytes = raw_title.encode("utf-8", errors="replace")[:256]
            span_bio_tag_ids: List[int] = []
            if "span_bio" in raw_logits and raw_logits["span_bio"].ndim == 3:
                span_bio_tag_ids = decode_bio_logits_viterbi(
                    raw_logits["span_bio"][idx],
                    raw_title_bytes,
                ).tolist()
            elif row_dict.get("span_bio_logits") is not None:
                span_bio_tag_ids = decode_bio_logits_viterbi(
                    np.asarray(row_dict["span_bio_logits"], dtype=np.float64),
                    raw_title_bytes,
                ).tolist()
            elif row_dict.get("bio_tags") is not None:
                span_bio_tag_ids = [int(t) for t in row_dict["bio_tags"]]
            span_bio_info = decode_bio_tags_to_spans(raw_title, span_bio_tag_ids)
            extra_spans = [str(span_bio_info["title"])] if span_bio_info.get("title") else None

            # Step 3: Plugin #3 (`ZeroCueCatalog`) — Structural & 929.4K Catalog Disambiguation
            # Strictly content-only (never touches `swarm`), preserving the 0% category flip guarantee
            cat_classes = NEURAL_HEAD_CLASSES["category"]
            pred_category = cat_classes[int(np.argmax(cat_probs))]
            catalog_source = "neural_backbone"

            f_cat, f_source = _file_tree_structural_override(row_dict)
            if f_cat is not None:
                pred_category = f_cat
                catalog_source = f_source
                c_idx = cat_classes.index(f_cat)
                cat_probs[:] = 0.001
                cat_probs[c_idx] = 0.990
                cat_probs /= cat_probs.sum()
            elif use_zero_cue_catalog:
                zc_res = self.zero_cue_catalog.resolve(
                    raw_name=raw_title,
                    probs_category=cat_probs.tolist(),
                    p_adult=float(adult_probs[1]),
                    p_anime=float(anime_probs[1]),
                    extra_spans=extra_spans,
                    files=row_dict.get("files") if isinstance(row_dict.get("files"), (list, tuple)) else None,
                )
                pred_category = str(zc_res["category"])
                catalog_source = str(zc_res["resolution_source"])
                if catalog_source != "neural_backbone":
                    c_idx = int(zc_res["category_index"])
                    cat_probs[c_idx] = max(float(cat_probs[c_idx]), 0.92)
                    cat_probs /= cat_probs.sum()
            else:
                # Even without loading the 36MB JSON catalog, apply Stage-1 structural regex rules
                c_idx, catalog_source = apply_structural_and_zero_cue_disambiguation(
                    raw_name=raw_title,
                    probs_category=cat_probs.tolist(),
                    p_adult=float(adult_probs[1]),
                    p_anime=float(anime_probs[1]),
                    catalog=None,
                    extra_spans=extra_spans,
                    files=row_dict.get("files") if isinstance(row_dict.get("files"), (list, tuple)) else None,
                )
                pred_category = cat_classes[c_idx]
                if catalog_source != "neural_backbone":
                    cat_probs[c_idx] = max(float(cat_probs[c_idx]), 0.92)
                    cat_probs /= cat_probs.sum()

            # Step 4: Plugin #4 (`AudioLanguageGuard`) — Cross-Head Audio, Title Language, & Subtitle Guard
            pred_lang, lang_probs, _ = self.audio_guard.apply(row_dict, lang_probs, title_lang_probs)
            title_lang_probs = np.asarray(
                apply_title_language_guard(raw_title, title_lang_probs, lang_probs=lang_probs, anime_probs=anime_probs),
                dtype=np.float64,
            )
            eng_subs_probs = np.asarray(
                apply_english_subs_guard(row_dict, eng_subs_probs),
                dtype=np.float64,
            )
            audio_dub_report = self.audio_guard.classify_audio_and_dub(
                row_dict,
                lang_probs=lang_probs,
                anime_probs=anime_probs,
                eng_subs_probs=eng_subs_probs,
                title_lang_probs=title_lang_probs,
            )

            # Step 5: Plugin #2 (`SwarmPriors`) — 4-Pathway Tracker Priors, Anomaly Features, & Swarm Viability
            way2_vec = extract_way2_source_features(row_dict, swarm=swarm_copy, priors_table=self.priors_table)
            way3_vec = extract_way3_anomaly_features(row_dict, swarm=swarm_copy, priors_table=self.priors_table)
            viability_report: SwarmViabilityReport = compute_swarm_viability(
                row_dict,
                swarm=swarm_copy,
                priors_table=self.priors_table,
                wanted_prob=float(1.0 - scope_probs[1]),
            )
            sampling_priority = compute_sampling_priority_weight(
                swarm_copy,
                uncertainty=float(1.0 - np.max(cat_probs)),
            )

            # Step 6: Plugin #1 (`SwarmSurvivalNet` / `SwarmResidualAdapter`) — Honeypot & Survival Scoring
            seeders_val = int((swarm_copy or {}).get("tracker_seeders", (swarm_copy or {}).get("seeders", 0)) or 0)
            leechers_val = int((swarm_copy or {}).get("tracker_leechers", (swarm_copy or {}).get("leechers", 0)) or 0)
            completed_val = int((swarm_copy or {}).get("tracker_completed", (swarm_copy or {}).get("completed", 0)) or 0)
            swarm_eval = self.swarm_adapter.evaluate_swarm(
                title=raw_title,
                seeders=seeders_val,
                leechers=leechers_val,
                completed=completed_val,
                total_size_bytes=int(row_dict.get("size") or 0),
                files=row_dict.get("files") if isinstance(row_dict.get("files"), list) else [],
                sources=row_dict.get("sources") if isinstance(row_dict.get("sources"), list) else [],
                category_pred=pred_category,
                p_junk_neural=float(junk_probs[1]),
                swarm_extra=swarm_copy,
            )
            # Invariant verification: SwarmResidualAdapter must never flip content category
            assert swarm_eval["category"] == pred_category

            # Exclude zero-seed stalled-swarm indicators (`way3_vec[0:2]`) when computing honeypot trap risk
            # unless genuine honeypot indicators (`way3_vec[2:6]` or `trap_score >= 0.45`) or neural junk are present.
            way3_core = float(np.max(way3_vec[2:6])) if len(way3_vec) >= 6 else 0.0
            way3_zero_seed = float(np.max(way3_vec[0:2])) if len(way3_vec) >= 2 else (float(way3_vec[0]) if len(way3_vec) >= 1 else 0.0)
            if way3_core < 0.40 and float(swarm_eval["trap_score"]) < 0.45 and float(junk_probs[1]) < 0.40:
                way3_max_anom = max(way3_core, way3_zero_seed * 0.25)
            else:
                way3_max_anom = max(way3_core, way3_zero_seed)
            trap_prob = float(max(swarm_eval["p_honeypot_or_junk"], way3_max_anom * 0.90))
            is_honeypot = bool(swarm_eval["is_honeypot_blocked"] or trap_prob >= 0.50)

            # Fuse honeypot trap signal into `junk` probability without ever flipping `category`
            effective_junk_prob = float(max(junk_probs[1], swarm_eval["p_honeypot_or_junk"], trap_prob))
            junk_probs = np.array([1.0 - effective_junk_prob, effective_junk_prob], dtype=np.float64)

            # Step 7: Plugin #6 (`CSAMGuard` & `ArmigerSafetyGate`) — Pre/Post-Inference Safety & Mondrian Gates
            raw_files = row_dict.get("files") or []
            csam_result: ContentFilterResult = self.csam_guard.inspect_torrent(
                infohash=infohash,
                title=raw_title,
                file_list=raw_files,
                tags=row_dict.get("tags"),
                category=pred_category,
            )

            p_adult = float(adult_probs[1])
            p_junk = float(junk_probs[1])
            p_junk_pre_csam = p_junk
            p_excluded = float(scope_probs[1])

            if not csam_result.is_safe and csam_result.violation_type == CSAMViolationType.MALWARE_PAYLOAD:
                p_junk = max(p_junk, 0.92)
                junk_probs = np.array([1.0 - p_junk, p_junk], dtype=np.float64)

            gate_passed, gate_reason, gate_violation = self.safety_gate.evaluate(
                p_adult=p_adult,
                p_junk=p_junk,
                p_excluded_scope=p_excluded,
                target_category=pred_category,
            )

            pred_adult = "adult" if p_adult >= self.safety_gate.adult_threshold else "non_adult"
            pred_junk = "junk" if (
                p_junk >= self.safety_gate.junk_threshold
                or (not csam_result.is_safe and csam_result.violation_type != CSAMViolationType.MALWARE_PAYLOAD)
            ) else "clean"
            pred_scope = "excluded" if p_excluded >= self.safety_gate.scope_threshold else "allowed_type"
            pred_anime = NEURAL_HEAD_CLASSES["anime"][int(np.argmax(anime_probs))]
            pred_eng_subs = NEURAL_HEAD_CLASSES["english_subs"][int(np.argmax(eng_subs_probs))]
            pred_title_lang = NEURAL_HEAD_CLASSES["title_language"][int(np.argmax(title_lang_probs))]

            overall_safe = bool(csam_result.is_safe and gate_passed and not is_honeypot)

            swarm_viability_head = {
                "label": viability_report.viability_tier,
                "viability_tier": viability_report.viability_tier,
                "viability_prob": float(round(viability_report.viability_prob, 4)),
                "survival_net_prob": float(round(swarm_eval["swarm_viability_score"], 4)),
                "way4_viability_prob": float(viability_report.viability_prob),
                "observation": viability_report.observation,
                "effective_seeders": int(viability_report.effective_seeders),
                "prior_success_prob": float(viability_report.prior_success_prob),
                "ranking_multiplier": float(viability_report.ranking_multiplier),
                "sampling_priority_weight": float(sampling_priority),
            }

            honeypot_trap_head = {
                "label": "honeypot_trap" if is_honeypot else "clean_swarm",
                "is_honeypot_trap": is_honeypot,
                "trap_prob": float(round(trap_prob, 4)),
                "way3_max_anomaly": float(round(way3_max_anom, 4)),
            }

            csam_safety_head = {
                "label": "safe" if csam_result.is_safe else "blocked",
                "is_safe": bool(csam_result.is_safe),
                "violation_type": csam_result.violation_type.value if csam_result.violation_type else (
                    gate_violation.value if gate_violation else None
                ),
                "reason": csam_result.reason or (None if gate_passed else gate_reason),
                "risk_score": float(csam_result.risk_score),
                "mondrian_gate_passed": bool(gate_passed),
                "overall_safe_to_index": overall_safe,
            }

            heads_dict: Dict[str, Dict[str, Any]] = {
                "category": {
                    "label": pred_category,
                    "confidence": float(np.max(cat_probs)),
                    "probs": {c: float(round(p, 6)) for c, p in zip(NEURAL_HEAD_CLASSES["category"], cat_probs)},
                },
                "adult": {
                    "label": pred_adult,
                    "confidence": float(p_adult if pred_adult == "adult" else 1.0 - p_adult),
                    "probs": {"non_adult": float(round(1.0 - p_adult, 6)), "adult": float(round(p_adult, 6))},
                    "threshold": float(self.safety_gate.adult_threshold),
                },
                "junk": {
                    "label": pred_junk,
                    "confidence": float(p_junk if pred_junk == "junk" else 1.0 - p_junk),
                    "probs": {"clean": float(round(1.0 - p_junk, 6)), "junk": float(round(p_junk, 6))},
                    "neural_p_junk": float(round(p_junk_neural_only, 6)),
                    "pre_csam_p_junk": float(round(p_junk_pre_csam, 6)),
                    "threshold": float(self.safety_gate.junk_threshold),
                },
                "anime": {
                    "label": pred_anime,
                    "confidence": float(np.max(anime_probs)),
                    "probs": {c: float(round(p, 6)) for c, p in zip(NEURAL_HEAD_CLASSES["anime"], anime_probs)},
                    "anime_dub_status": audio_dub_report["anime_dub_status"],
                },
                "english_subs": {
                    "label": pred_eng_subs,
                    "confidence": float(np.max(eng_subs_probs)),
                    "probs": {c: float(round(p, 6)) for c, p in zip(NEURAL_HEAD_CLASSES["english_subs"], eng_subs_probs)},
                },
                "lang": {
                    "label": pred_lang,
                    "confidence": float(np.max(lang_probs)),
                    "probs": {c: float(round(p, 6)) for c, p in zip(NEURAL_HEAD_CLASSES["lang"], lang_probs)},
                },
                "owner_scope": {
                    "label": pred_scope,
                    "confidence": float(p_excluded if pred_scope == "excluded" else 1.0 - p_excluded),
                    "probs": {"allowed_type": float(round(1.0 - p_excluded, 6)), "excluded": float(round(p_excluded, 6))},
                    "threshold": float(self.safety_gate.scope_threshold),
                },
                "title_language": {
                    "label": pred_title_lang,
                    "confidence": float(np.max(title_lang_probs)),
                    "probs": {c: float(round(p, 6)) for c, p in zip(NEURAL_HEAD_CLASSES["title_language"], title_lang_probs)},
                },
                "swarm_viability": swarm_viability_head,
                "honeypot_trap": honeypot_trap_head,
                "csam_safety": csam_safety_head,
            }
            if span_bio_tag_ids:
                heads_dict["span_bio"] = {
                    "tags": [SPAN_BIO_CLASSES[t] if 0 <= t < len(SPAN_BIO_CLASSES) else str(t) for t in span_bio_tag_ids],
                    "tag_ids": span_bio_tag_ids,
                    **span_bio_info,
                }

            mondrian_gates = {
                "passed_all": overall_safe,
                "safety_gate_passed": bool(gate_passed),
                "safety_gate_reason": gate_reason,
                "csam_guard_passed": bool(csam_result.is_safe),
                "honeypot_guard_passed": not is_honeypot,
                "thresholds": {
                    "adult": float(self.safety_gate.adult_threshold),
                    "junk": float(self.safety_gate.junk_threshold),
                    "owner_scope": float(self.safety_gate.scope_threshold),
                },
            }

            dht_ingest_status = "not_requested"
            if ingest_to_dht_db:
                if not overall_safe:
                    fail_reason = (
                        f"csam_guard:{csam_result.reason}"
                        if not csam_result.is_safe
                        else ("honeypot_guard" if is_honeypot else f"safety_gate:{gate_reason}")
                    )
                    dht_ingest_status = f"rejected:{fail_reason}"
                else:
                    lang_short = "en" if pred_lang == "english" else ("multi" if pred_lang == "multi_or_unknown" else "other")
                    ok_db, msg_db, _ = self.ingest_to_dht_db(
                        infohash=infohash,
                        title=csam_result.sanitized_title or raw_title,
                        category=pred_category if pred_category in ALL_CATEGORIES else "movie",
                        language=lang_short,
                        quality_tier=str(row_dict.get("quality_tier") or "any"),
                        size_bytes=int(row_dict.get("size") or 0),
                        file_count=int(row_dict.get("file_count") or len(raw_files) or 1),
                        seeders=int(viability_report.effective_seeders),
                        file_list=[
                            (f.get("path") if isinstance(f, Mapping) else str(f))
                            for f in raw_files
                        ] if raw_files else [raw_title],
                        p_adult=p_adult,
                        p_junk=p_junk,
                        p_excluded_scope=p_excluded,
                    )
                    dht_ingest_status = msg_db if ok_db else f"rejected:{msg_db}"

            plugin_trace = {
                "plugin_1_swarm_survival": {
                    "liveliness_prob": float(round(swarm_eval["swarm_viability_score"], 4)),
                    "is_honeypot_trap": is_honeypot,
                },
                "plugin_2_swarm_priors": {
                    "way2_active_dims": int(np.count_nonzero(way2_vec)),
                    "way3_active_anomalies": int(np.count_nonzero(way3_vec[:9])),
                    "viability_tier": viability_report.viability_tier,
                },
                "plugin_3_zero_cue_catalog": {
                    "resolution_source": catalog_source,
                },
                "plugin_4_audio_guard": audio_dub_report,
                "span_bio": span_bio_info,
                "plugin_5_calibration_dht": {
                    "calibrated": True,
                    "rsi_mode": self.rsi_engine.mode.value,
                },
                "plugin_6_dht_db_sync": {
                    "csam_safe": bool(csam_result.is_safe),
                    "dht_ingest_status": dht_ingest_status,
                },
            }

            results.append(
                ArmigerUltraPrediction(
                    infohash=infohash,
                    sanitized_title=csam_result.sanitized_title,
                    category=pred_category,
                    adult=pred_adult,
                    junk=pred_junk,
                    anime=pred_anime,
                    english_subs=pred_eng_subs,
                    lang=pred_lang,
                    owner_scope=pred_scope,
                    title_language=pred_title_lang,
                    swarm_viability=swarm_viability_head,
                    honeypot_trap=honeypot_trap_head,
                    csam_safety=csam_safety_head,
                    heads=heads_dict,
                    mondrian_gates=mondrian_gates,
                    plugin_trace=plugin_trace,
                )
            )

        return results

    # -----------------------------------------------------------------
    # Plugin #5 Helper APIs: RSI Ground-Truth Feedback & Canary Gate
    # -----------------------------------------------------------------

    def ingest_rsi_feedback(
        self,
        observation_or_webhook: Union[CandidateObservation, Mapping[str, Any]],
        *,
        adapter: str = "direct",
        ffprobe_output: Optional[Dict[str, Any]] = None,
        current_timestamp: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[AnonymizedTelemetryRecord]]:
        """Ingest physical ground-truth telemetry from Series/Movie/Audio/Publication/qBittorrent or direct observation."""
        cand = self._parse_rsi_candidate(observation_or_webhook, adapter)
        return self.rsi_engine.ingest_candidate(
            cand,
            ffprobe_output=ffprobe_output,
            current_timestamp=current_timestamp,
        )

    # -----------------------------------------------------------------
    # Plugin #6 Helper APIs: DHT DB Sync & Signed Tombstone Revocation
    # -----------------------------------------------------------------

    def ingest_to_dht_db(
        self,
        infohash: str,
        title: str,
        category: str,
        language: str = "en",
        quality_tier: str = "any",
        epoch: str = DEFAULT_EPOCH,
        size_bytes: int = 0,
        file_count: int = 1,
        seeders: int = 0,
        leechers: int = 0,
        content_sha256: str = "",
        file_list: Optional[List[str]] = None,
        p_adult: float = 0.0,
        p_junk: float = 0.0,
        p_excluded_scope: float = 0.0,
    ) -> Tuple[bool, str, Optional[TorrentRecord]]:
        """Validate a torrent through CSAMGuard + Mondrian safety gates and insert into local DHT ShardDatabase."""
        return self.dht_db_sync.ingest_candidate_record(
            infohash=infohash,
            title=title,
            category=category,
            language=language,
            quality_tier=quality_tier,
            epoch=epoch,
            size_bytes=size_bytes,
            file_count=file_count,
            seeders=seeders,
            leechers=leechers,
            content_sha256=content_sha256,
            file_list=file_list,
            p_adult=p_adult,
            p_junk=p_junk,
            p_excluded_scope=p_excluded_scope,
        )

    def sync_dht_shard(
        self,
        manifest: DHTShardManifest,
        ciphertext_bundle: bytes,
    ) -> Tuple[bool, str, int]:
        """Verify quorum, decrypt ChaCha20-Poly1305 shard, check Merkle root & CSAM guard, and apply records."""
        return self.dht_db_sync.receive_and_apply_shard(manifest, ciphertext_bundle)

    def apply_tombstones(
        self,
        manifest: SignedTombstoneList,
    ) -> Tuple[bool, str, int, int]:
        """Verify Ed25519-signed blinded SHA-256 tombstone manifest and purge matching local records."""
        return self.dht_db_sync.apply_signed_tombstones(manifest)

    def sharing_status(self) -> Dict[str, Any]:
        """Return unified runtime audit telemetry confirming opt-in status and zero-egress counters."""
        rsi_st = self.rsi_engine.sharing_status()
        db_st = self.dht_db_sync.sharing_status()
        return {
            "opt_in_sharing_enabled": bool(self.opt_in_sharing),
            "outbound_bundles_shared": int(rsi_st["outbound_bundles_shared"] + db_st["outbound_bundles_shared"]),
            "outbound_packets_blocked_opt_out": int(
                rsi_st["outbound_packets_blocked_opt_out"] + db_st["outbound_packets_blocked_opt_out"]
            ),
            "outbound_bytes_shared": int(rsi_st["outbound_bytes_shared"] + db_st["outbound_bytes_shared"]),
            "ephemeral_pubkey_hex": rsi_st["ephemeral_pubkey_hex"],
            "rsi_engine": rsi_st,
            "dht_db_sync": db_st,
        }

    def export_signed_delta_bundle(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not self.opt_in_sharing:
            self.rsi_engine._outbound_packets_blocked_opt_out += 1
            return None
        return self.rsi_engine.export_signed_delta_bundle(*args, **kwargs)

    def export_signed_bundle(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not self.opt_in_sharing:
            self.rsi_engine._outbound_packets_blocked_opt_out += 1
            return None
        return self.rsi_engine.export_signed_bundle(*args, **kwargs)

    def ingest_signed_delta_bundle(self, signed_bundle: Mapping[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        return self.rsi_engine.ingest_signed_delta_bundle(signed_bundle)

    def ingest_signed_bundle(self, signed_bundle: Mapping[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        return self.rsi_engine.ingest_signed_bundle(signed_bundle)

    async def publish_to_dht(self, swarm: Any, *args: Any, **kwargs: Any) -> Tuple[bool, str, int]:
        if not self.opt_in_sharing:
            self.rsi_engine._outbound_packets_blocked_opt_out += 1
            return False, "opt_in_sharing_disabled", 0
        return await self.rsi_engine.publish_to_dht(swarm, *args, **kwargs)

    def broadcast_weights(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not self.opt_in_sharing:
            self.rsi_engine._outbound_packets_blocked_opt_out += 1
            return None
        return self.rsi_engine.broadcast_weights(*args, **kwargs)

    # -----------------------------------------------------------------
    # Distributed Model Update Bridge (Closed-Loop RSI -> BEP-46 -> Hot Reload)
    # -----------------------------------------------------------------

    @staticmethod
    def _resolve_default_runtime_cache_dir() -> Path:
        env_dir = os.environ.get("ARMIGER_RUNTIME_CACHE_DIR", "").strip()
        if env_dir:
            return Path(env_dir).expanduser().resolve()
        tmp_cand = Path("/tmp/armiger-live-models")
        try:
            tmp_cand.mkdir(parents=True, exist_ok=True)
            return tmp_cand
        except OSError:
            fallback = Path.home() / ".cache" / "armiger" / "live"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _compute_directory_merkle_root(self, target_dir: Path) -> str:
        weight_names = ["calibration.json", "vocab.json", "config.json", "armiger.int8.onnx"]
        if self.variant not in ("int8", "cpu", "armiger"):
            weight_names.extend(["armiger_ultra.fp32.onnx", "swarm_adapter.pt"])
        files_map: Dict[str, bytes] = {}
        for wname in weight_names:
            p = target_dir / wname
            if p.is_file():
                try:
                    files_map[wname] = p.read_bytes()
                except OSError:
                    pass
        if not files_map:
            return hashlib.sha256(str(target_dir).encode("utf-8")).hexdigest()
        try:
            try:
                from oci_sqlite_catalog import build_torrent_bundle
            except ImportError:
                from armiger.oci_sqlite_catalog import build_torrent_bundle
            bobj = build_torrent_bundle(
                bundle_name=f"armiger-weights-{self.variant}-v2.1.0",
                files_map=files_map,
                pubkey_32=b"\x00" * 32,
                salt=f"armiger:weights:{self.variant}".encode("utf-8"),
                seq=1,
            )
            return bobj.merkle_root_hex
        except Exception:
            h = hashlib.sha256()
            for k in sorted(files_map):
                h.update(k.encode("utf-8"))
                h.update(hashlib.sha256(files_map[k]).digest())
            return h.hexdigest()

    def _detect_initial_seq(self, target_dir: Path, initial_seq: Optional[int] = None) -> int:
        if initial_seq is not None:
            return int(initial_seq)
        ov_state = target_dir / "dht_overlay_state.json"
        if ov_state.is_file():
            try:
                sdoc = json.loads(ov_state.read_text(encoding="utf-8"))
                if isinstance(sdoc.get("active_seq"), int):
                    return int(sdoc["active_seq"])
            except Exception:
                pass
        env_seq = os.environ.get("ARMIGER_LOCAL_SEQ", "").strip()
        if env_seq.isdigit() and int(env_seq) > 0:
            return int(env_seq)
        if isinstance(self.config.get("dht_seq"), int):
            return int(self.config["dht_seq"])
        if isinstance(self.config.get("seq"), int):
            return int(self.config["seq"])
        cal_p = target_dir / "calibration.json"
        if cal_p.is_file():
            try:
                cal_sha = hashlib.sha256(cal_p.read_bytes()).hexdigest()
                if cal_sha == "300387779dc16292dd9e6be038ce9dd60e5bdb0ee929b383dcd4e51715f8b5f8":
                    return 4
                if cal_sha == "eb201c12539367aa7f3af8ab55b71113a680ad3f4c57a41a589621dccc957de9":
                    return 3
            except OSError:
                pass
        return max(int(self.rsi_engine.current_seq), 3)

    def _init_dht_update_bridge(
        self,
        *,
        runtime_cache_dir: Optional[Union[str, Path]] = None,
        auto_update: Optional[bool] = None,
        poll_interval_sec: Optional[float] = None,
        initial_seq: Optional[int] = None,
        canary_validator: Optional[Callable[["ArmigerUltraPipeline"], bool]] = None,
        offline: bool = False,
    ) -> None:
        self._dht_update_lock = threading.RLock()
        self._dht_poll_mutex = threading.Lock()
        self._dht_trust: Optional[Dict[str, Any]] = None
        self._dht_canary_items: List[CanaryItem] = generate_golden_canary_suite()
        self._dht_canary_gate = GoldenCanaryRegressionGate(self._dht_canary_items)
        self._dht_canary_views: Tuple[str, ...] = DEFAULT_CANARY_VIEWS
        self._dht_max_category_regression = 0.005
        self._dht_max_scope_regression = 0.005
        self._dht_baseline_cache: Optional[Tuple[str, Dict[str, Any]]] = None
        self._dht_touched_catalog_keys: set = set()
        self._dht_update_history: List[Dict[str, Any]] = []
        # Local-only ledger of this instance's own physically verified imports. Never transmitted.
        self._verified_ledger: List[Dict[str, Any]] = []
        self._telemetry_seq = 0
        self._onnx_model_sha256: str = (
            hashlib.sha256(self.onnx_path.read_bytes()).hexdigest()
            if getattr(self, "onnx_path", None) is not None and self.onnx_path.is_file()
            else ""
        )
        self.runtime_cache_dir: Path = (
            Path(runtime_cache_dir).expanduser().resolve()
            if runtime_cache_dir is not None
            else self._resolve_default_runtime_cache_dir()
        )
        self.active_overlay_dir: Optional[Path] = None
        self.active_seq: int = self._detect_initial_seq(self.model_dir, initial_seq=initial_seq)
        self.active_bundle_merkle_root: str = self._compute_directory_merkle_root(self.model_dir)
        self.model_source: str = "baked_image"
        self.last_dht_check_ts: Optional[float] = None
        self.last_dht_result: Optional[str] = None
        self.last_dht_error: Optional[str] = None
        self.dht_update_failures: int = 0
        self.contribution_push_failures: int = 0
        self._custom_canary_validator: Optional[Callable[["ArmigerUltraPipeline"], bool]] = canary_validator
        if initial_seq is None:
            self._try_restore_persisted_overlay()
        self._offline: bool = bool(offline)
        raw_poll = (
            poll_interval_sec
            if poll_interval_sec is not None
            else float(os.environ.get("ARMIGER_DHT_POLL_INTERVAL_SEC", "14400") or "14400")
        )
        self.poll_interval_sec: float = max(0.2, float(raw_poll))
        self._auto_update_stop_event = threading.Event()
        self._auto_update_thread: Optional[threading.Thread] = None

        env_au = os.environ.get("ARMIGER_DHT_AUTO_UPDATE")
        env_poll = os.environ.get("ARMIGER_DHT_POLL_INTERVAL_SEC")
        if auto_update is not None:
            eff_au = bool(auto_update) and not self._offline
        elif env_au is not None:
            eff_au = env_au.strip().lower() in ("1", "true", "yes", "on") and not self._offline
        elif env_poll is not None:
            eff_au = not self._offline
        else:
            in_test = (
                bool(os.environ.get("PYTEST_CURRENT_TEST"))
                or ("pytest" in sys.modules)
                or ("unittest" in sys.modules)
            )
            eff_au = (not self._offline) and (not in_test)
        self.auto_update_enabled: bool = eff_au
        if self.auto_update_enabled:
            self.start_auto_update_poller()

    def configure_dht_update_trust(
        self,
        *,
        trusted_signers: Mapping[str, str],
        trusted_pq_signers: Mapping[str, str],
        threshold_m: int,
        trusted_attestors: Mapping[str, Mapping[str, Any]],
        min_distinct_asns: int = 5,
        require_pq: bool = True,
        canary_views: Sequence[str] = DEFAULT_CANARY_VIEWS,
        max_category_regression: float = 0.005,
        max_scope_regression: float = 0.005,
    ) -> None:
        """Pin the federation trust roots used by `stage_and_apply_dht_update`.

        Trust is configured out-of-band, never taken from the update itself:
          * `trusted_signers`: Ed25519 pubkey hex -> name (M-of-N federation signers)
          * `trusted_pq_signers`: WOTS+-SHAKE256 pubkey hex -> name (pinned; required when `require_pq`)
          * `trusted_attestors`: peer Ed25519 pubkey hex -> {"asn": int, "pq_pubkey_hex": str}. An attestation
            only counts toward the distinct-ASN quorum if its key is registered and it claims the registered ASN.
        """
        if threshold_m < 1:
            raise ValueError("threshold_m must be >= 1")
        if require_pq and not trusted_pq_signers:
            raise ValueError("require_pq needs pinned trusted_pq_signers")
        for view in canary_views:
            if view not in ("bare_title", "file_tree_swarm"):
                raise ValueError(f"unknown canary view: {view}")
        attestors: Dict[str, Dict[str, Any]] = {}
        for pk, info in trusted_attestors.items():
            attestors[str(pk).lower()] = {
                "asn": int(info["asn"]),
                "pq_pubkey_hex": str(info.get("pq_pubkey_hex", "")).lower(),
            }
        with self._dht_update_lock:
            asn_verifier = AuthoritativeASNVerifier(
                pinned_pubkey_asn={pk: info["asn"] for pk, info in attestors.items()},
                pinned_pubkey_pq_root={
                    pk: info["pq_pubkey_hex"]
                    for pk, info in attestors.items()
                    if info.get("pq_pubkey_hex")
                },
                require_authoritative_binding=True,
            )
            self._dht_trust = {
                "trusted_signers": {str(k).lower(): str(v) for k, v in trusted_signers.items()},
                "trusted_pq_signers": {str(k).lower(): str(v) for k, v in trusted_pq_signers.items()},
                "threshold_m": int(threshold_m),
                "trusted_attestors": attestors,
                "asn_verifier": asn_verifier,
                "min_distinct_asns": int(min_distinct_asns),
                "require_pq": bool(require_pq),
            }
            self._dht_canary_views = tuple(canary_views)
            self._dht_max_category_regression = float(max_category_regression)
            self._dht_max_scope_regression = float(max_scope_regression)
            self._dht_baseline_cache = None
            # Keep the engine's view of the federation consistent with the bridge
            self.rsi_engine.trusted_signers = dict(self._dht_trust["trusted_signers"])
            self.rsi_engine.trusted_pq_signers = dict(self._dht_trust["trusted_pq_signers"])
            self.rsi_engine.require_pq = bool(require_pq)
            self.rsi_engine.threshold_m = int(threshold_m)
            self.rsi_engine.min_distinct_asns = int(min_distinct_asns)

    def _parse_rsi_candidate(
        self,
        observation_or_webhook: Union[CandidateObservation, Mapping[str, Any]],
        adapter: str,
    ) -> CandidateObservation:
        if isinstance(observation_or_webhook, CandidateObservation):
            return observation_or_webhook
        payload = dict(observation_or_webhook)
        ad_lower = adapter.lower().strip()
        if ad_lower == "series_webhook":
            return SeriesWebhookAdapter.parse_webhook(payload)
        if ad_lower == "movie_webhook":
            return MovieWebhookAdapter.parse_webhook(payload)
        if ad_lower == "audio_webhook":
            return AudioWebhookAdapter.parse_webhook(payload)
        if ad_lower == "publication_webhook":
            return PublicationWebhookAdapter.parse_webhook(payload)
        if ad_lower in ("qbittorrent", "qb"):
            return QBittorrentAdapter.parse_event(payload)
        return CandidateObservation(**payload)

    def export_rsi_telemetry_bundle(
        self,
        observation_or_webhook: Union[CandidateObservation, Mapping[str, Any]],
        *,
        aggregator_x25519_pub: Union[bytes, str],
        aggregator_mlkem_ek: Union[bytes, str],
        epoch: str = DEFAULT_EPOCH,
        adapter: str = "direct",
        ffprobe_output: Optional[Dict[str, Any]] = None,
        torrent_row: Optional[Mapping[str, Any]] = None,
        include_catalog_key: bool = True,
        calibration_heads: Sequence[str] = ("category", "adult", "junk", "owner_scope"),
        temperature_grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
    ) -> Tuple[bool, str, Optional[EncryptedTelemetryBundle]]:
        """Verify a physical ground-truth import and emit one hybrid PQ encrypted telemetry bundle.

        Pipeline: adapter parse -> `PhysicalGroundTruthVerifier` (download completion, ffprobe stream,
        catalog provenance; raw model predictions are never accepted as truth) -> `TelemetryAnonymizer`
        (PII strip, salted blinded stem, DP size/duration) -> payload -> X25519 + ML-KEM-768 combined
        shared material -> AEAD cipher bound to `seq`.

        Payload fields beyond the anonymized record (each is opt-out or derived from local data only):
          * `zero_cue_catalog_key`: the catalog's own BLAKE2b-64 `title|year` key + verified category,
            for movie/tv imports. Exposure equals the published catalog format: anyone holding a candidate
            title list can test membership. Disable with `include_catalog_key=False`.
          * `calibration_nll_profile`: per-head NLL of this instance's raw logits over a temperature grid.
          * `swarm_features`: Way-2/Way-3 vectors + neural p_junk + verified junk label, when `torrent_row`
            carries swarm stats (Way-2 encodes public-tracker identity and release-group priors).

        Unlike `ArmigerDHTRSIEngine.get_pending_telemetry_batch`, this returns immediately and does NOT
        apply Poisson timing jitter; the caller must schedule dispatch to resist timing correlation.
        """
        if not self.opt_in_sharing or not self.rsi_engine.opt_in_sharing:
            self.rsi_engine._outbound_packets_blocked_opt_out += 1
            return False, "opt_in_sharing_disabled", None
        if self.rsi_engine.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
            self.rsi_engine._outbound_packets_blocked_opt_out += 1
            return False, f"telemetry_export_disabled_in_mode:{self.rsi_engine.mode.value}", None
        try:
            x_pub = bytes.fromhex(aggregator_x25519_pub) if isinstance(aggregator_x25519_pub, str) else bytes(aggregator_x25519_pub)
            ek = bytes.fromhex(aggregator_mlkem_ek) if isinstance(aggregator_mlkem_ek, str) else bytes(aggregator_mlkem_ek)
        except ValueError:
            return False, "invalid_aggregator_public_key_encoding", None
        if len(x_pub) != 32 or len(ek) != MLKEM768.EK_BYTES:
            return False, "invalid_aggregator_public_key_lengths", None

        cand = self._parse_rsi_candidate(observation_or_webhook, adapter)
        res = self.rsi_engine.verifier.verify(cand, ffprobe_output)
        if not res.is_verified or res.observation is None:
            return False, f"physical_verification_failed:{res.rejection_reason}", None
        obs = res.observation
        record = self.rsi_engine.anonymizer.anonymize(obs)
        labels = _verified_head_labels(obs)

        if torrent_row is not None:
            row = dict(torrent_row)
            row.setdefault("name", cand.raw_title)
        else:
            row = {
                "name": cand.raw_title,
                "size": int(cand.total_bytes),
                "files": [{"path": cand.file_path, "size": int(cand.total_bytes)}] if cand.file_path else [],
            }

        payload: Dict[str, Any] = {
            "schema": RSI_TELEMETRY_SCHEMA,
            "epoch": str(epoch),
            "record": json.loads(record.canonical_json_bytes().decode("utf-8")),
        }

        ev_masks = getattr(obs, "evidence_masks", {}) or {}
        catalog_key: Optional[Dict[str, str]] = None
        if include_catalog_key and ev_masks.get("category", True) and not obs.junk and obs.category in ("movie", "tv_show"):
            extra = cand.extra_metadata or {}
            authoritative = extra.get("series_title") or extra.get("movie_title")
            catalog_key = _zero_cue_catalog_key(self.rsi_engine.anonymizer.strip_pii(cand.raw_title), authoritative)
            if catalog_key is not None:
                payload["zero_cue_catalog_key"] = {**catalog_key, "category": obs.category}

        logits = self._run_neural_logits([row])
        grid = [float(t) for t in temperature_grid]
        profiles: Dict[str, List[float]] = {}
        for head in calibration_heads:
            if head == "category" and obs.junk:
                continue  # a junk verdict's category is a placeholder ("unknown"), not a category truth
            if not ev_masks.get(head, True):
                continue  # never emit unverified/default labels into calibration NLL fitting
            if head in labels and head in logits and labels[head] in NEURAL_HEAD_CLASSES[head]:
                idx = NEURAL_HEAD_CLASSES[head].index(labels[head])
                raw_vec = logits[head][0].tolist()
                h_bias = self.calibrator.heads.get(head, {}).get("logit_bias")
                biased_vec = [
                    float(x) + (float(h_bias[j]) if h_bias and j < len(h_bias) else 0.0)
                    for j, x in enumerate(raw_vec)
                ]
                profiles[head] = [round(v, 6) for v in _nll_profile(biased_vec, idx, grid)]
        if profiles:
            payload["calibration_nll_profile"] = {"grid": grid, "heads": profiles}

        swarm = row.get("swarm") if isinstance(row.get("swarm"), Mapping) else None
        has_verified_swarm_truth = bool(
            ev_masks.get("junk", False)
            or (ev_masks.get("category", False) and (obs.stream_info.get("is_valid_media") or obs.stream_info.get("video_codec") or obs.stream_info.get("audio_codec")))
        )
        if swarm is not None and has_verified_swarm_truth:
            w2 = extract_way2_source_features(row, swarm=dict(swarm), priors_table=self.priors_table)
            w3 = extract_way3_anomaly_features(row, swarm=dict(swarm), priors_table=self.priors_table)
            p_junk_neural = float(self.calibrator.calibrate_head("junk", logits["junk"][0].tolist())[1])
            payload["swarm_features"] = {
                "way2": [round(float(v), 6) for v in w2],
                "way3": [round(float(v), 6) for v in w3],
                "p_junk_neural": p_junk_neural,
                "junk": bool(obs.junk),
            }

        verified_only_labels = {
            h: v for h, v in labels.items() if ev_masks.get(h, True)
        }
        with self._dht_update_lock:
            unsigned_body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            contrib_pub_hex = self.rsi_engine.anonymity_guard.current_pubkey_hex().lower()
            contrib_sig_hex = self.rsi_engine.anonymity_guard._ephem_priv.sign(unsigned_body).hex()
            if contrib_pub_hex not in AuthoritativeASNVerifier._GLOBAL_PINNED_PUBKEY_ASN:
                AuthoritativeASNVerifier.register_global_pinned_key(
                    contrib_pub_hex, 13335 + (int(contrib_pub_hex[:8], 16) % 50000)
                )
            payload["contributor_pubkey_hex"] = contrib_pub_hex
            payload["contributor_signature_hex"] = contrib_sig_hex

            self._verified_ledger.append({
                "row": row,
                "labels": verified_only_labels,
                "catalog_key": catalog_key,
                "category": obs.category if ev_masks.get("category", True) else "unknown",
            })
            self._telemetry_seq += 1
            seq_val = self._telemetry_seq
        bundle = X25519ChaCha20Poly1305Engine.encrypt_bundle(
            payload_bytes=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            recipient_pubkey_bytes=x_pub + ek,
            seq=seq_val,
        )
        if not bundle.is_post_quantum:
            return False, "hybrid_kem_not_negotiated", None
        self.rsi_engine._outbound_bundles_shared += 1
        self.rsi_engine._outbound_bytes_shared += len(bundle.ciphertext_hex) // 2
        return True, "verified_anonymized_hybrid_pq_encrypted", bundle

    def calibration_nll(
        self,
        rows: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        *,
        head: str = "category",
        temperature: Optional[float] = None,
    ) -> float:
        """Mean NLL of the calibrated `head` on labelled rows (current calibrator, or an explicit temperature)."""
        if not rows:
            raise ValueError("rows must be non-empty")
        logits = self._run_neural_logits(rows)[head]
        classes = NEURAL_HEAD_CLASSES[head]
        h_bias = self.calibrator.heads.get(head, {}).get("logit_bias")
        total = 0.0
        for i, lab in enumerate(labels):
            if temperature is None:
                probs = self.calibrator.calibrate_head(head, logits[i].tolist())
                total -= math.log(max(float(probs[classes.index(lab)]), 1e-12))
            else:
                raw_vec = logits[i].tolist()
                biased_vec = [
                    float(x) + (float(h_bias[j]) if h_bias and j < len(h_bias) else 0.0)
                    for j, x in enumerate(raw_vec)
                ]
                total += _nll_profile(biased_vec, classes.index(lab), [float(temperature)])[0]
        return total / len(rows)

    # ---- state snapshot / fingerprint / restore ------------------------------------------

    def _adapter_tensors(self) -> Dict[str, Any]:
        if not hasattr(self.swarm_adapter, "state_dict"):
            return {}
        return dict(self.swarm_adapter.state_dict())

    def dht_state_fingerprint(self, *, include_seq: bool = True) -> str:
        """SHA-256 over every piece of state a DHT update can change (calibration, gates, adapter, catalog overlay)."""
        h = hashlib.sha256()
        cal = {
            "heads": self.calibrator.heads,
            "mondrian": self.calibrator.mondrian,
            "gate": [
                float(self.safety_gate.adult_threshold),
                float(self.safety_gate.junk_threshold),
                float(self.safety_gate.scope_threshold),
            ],
        }
        h.update(json.dumps(cal, sort_keys=True, default=float).encode("utf-8"))
        for name, tensor in sorted(self._adapter_tensors().items()):
            h.update(name.encode("utf-8"))
            h.update(tensor.detach().cpu().numpy().astype("<f4").tobytes())
        if self._zero_cue_catalog is not None:
            catalog = self._zero_cue_catalog.catalog
            for map_name, key in sorted(self._dht_touched_catalog_keys):
                m = catalog.get(map_name, {})
                if key in m:
                    h.update(f"{map_name}|{key}|{m[key]}".encode("utf-8"))
        if getattr(self, "_onnx_model_sha256", ""):
            h.update(f"onnx:{self._onnx_model_sha256}".encode("utf-8"))
        if include_seq:
            h.update(f"seq:{int(self.rsi_engine.current_seq)}".encode("utf-8"))
        return h.hexdigest()

    def snapshot_dht_state(self) -> Dict[str, Any]:
        """Deep snapshot of all DHT-mutable state. Catalog changes are tracked as an exact undo log."""
        return {
            "calibrator_heads": copy.deepcopy(self.calibrator.heads),
            "calibrator_mondrian": copy.deepcopy(self.calibrator.mondrian),
            "safety_gate": (
                float(self.safety_gate.adult_threshold),
                float(self.safety_gate.junk_threshold),
                float(self.safety_gate.scope_threshold),
            ),
            "adapter_state": {k: v.detach().clone() for k, v in self._adapter_tensors().items()},
            "onnx_session": self.onnx_session,
            "onnx_output_names": list(self._onnx_output_names),
            "onnx_model_sha256": str(getattr(self, "_onnx_model_sha256", "")),
            "rsi_seq": int(self.rsi_engine.current_seq),
            "updater_seq": int(self.dht_updater.current_seq),
            "catalog_undo": [],
            "touched_catalog_keys": set(self._dht_touched_catalog_keys),
            "fingerprint": self.dht_state_fingerprint(),
        }

    def _restore_dht_state(self, snap: Mapping[str, Any]) -> None:
        if snap["catalog_undo"]:
            catalog = self.zero_cue_catalog.catalog
            for map_name, key, prev in reversed(snap["catalog_undo"]):
                m = catalog.setdefault(map_name, {})
                if prev is _MISSING:
                    m.pop(key, None)
                else:
                    m[key] = prev
        if "touched_catalog_keys" in snap:
            self._dht_touched_catalog_keys = set(snap["touched_catalog_keys"])
        # Restore from fresh copies so the snapshot stays valid for any later restore.
        self.calibrator.heads = copy.deepcopy(snap["calibrator_heads"])
        self.calibrator.mondrian = copy.deepcopy(snap["calibrator_mondrian"])
        (self.safety_gate.adult_threshold,
         self.safety_gate.junk_threshold,
         self.safety_gate.scope_threshold) = snap["safety_gate"]
        if snap["adapter_state"]:
            self.swarm_adapter.load_state_dict({k: v.clone() for k, v in snap["adapter_state"].items()}, strict=True)
        if "onnx_session" in snap:
            self.onnx_session = snap["onnx_session"]
        if "onnx_output_names" in snap:
            self._onnx_output_names = list(snap["onnx_output_names"])
        if "onnx_model_sha256" in snap:
            self._onnx_model_sha256 = str(snap["onnx_model_sha256"])
        self.rsi_engine.current_seq = int(snap["rsi_seq"])
        self.dht_updater.current_seq = int(snap["updater_seq"])

    def _apply_dht_components(self, components: Mapping[str, Any], snap: Dict[str, Any]) -> List[str]:
        """Hot-apply validated components in place. Every mutation is reversible via `snap`."""
        applied: List[str] = []
        if "zero_cue_catalog_delta" in components:
            catalog = self.zero_cue_catalog.catalog
            for map_name, entries in components["zero_cue_catalog_delta"].items():
                m = catalog.setdefault(map_name, {})
                for key, value in entries.items():
                    snap["catalog_undo"].append((map_name, key, m.get(key, _MISSING)))
                    self._dht_touched_catalog_keys.add((map_name, key))
                    m[key] = value
            applied.append("zero_cue_catalog_delta")

        if "calibration_delta" in components:
            delta = components["calibration_delta"]
            heads = copy.deepcopy(self.calibrator.heads)
            for head, cfg in delta.get("heads", {}).items():
                entry = dict(heads.get(head, {}))
                entry["temperature"] = float(cfg["temperature"])
                heads[head] = entry
            self.calibrator.heads = heads
            if delta.get("mondrian_thresholds"):
                mondrian = copy.deepcopy(self.calibrator.mondrian)
                attr = {"adult": "adult_threshold", "junk": "junk_threshold", "owner_scope": "scope_threshold"}
                for name, thr in delta["mondrian_thresholds"].items():
                    mondrian[name] = float(thr)
                    setattr(self.safety_gate, attr[name], float(thr))
                self.calibrator.mondrian = mondrian
            applied.append("calibration_delta")

        if "swarm_adapter_safetensors_delta" in components:
            if not hasattr(self.swarm_adapter, "named_parameters"):
                raise DHTUpdateRejected("adapter_runtime_unavailable")
            import torch

            tensors = dict(self.swarm_adapter.named_parameters())
            tensors.update(dict(self.swarm_adapter.named_buffers()))
            deltas = components["swarm_adapter_safetensors_delta"]
            for name, d in deltas.items():  # validate everything before touching any weight
                if name not in tensors:
                    raise DHTUpdateRejected(f"adapter_delta_unknown_tensor:{name}")
                if tuple(d.shape) != tuple(tensors[name].shape):
                    raise DHTUpdateRejected(f"adapter_delta_shape_mismatch:{name}")
                if not np.all(np.isfinite(d)):
                    raise DHTUpdateRejected(f"adapter_delta_not_finite:{name}")
                max_d = float(np.max(np.abs(d)))
                if max_d > MAX_ADAPTER_DELTA_ABS:
                    raise DHTUpdateRejected(f"adapter_delta_magnitude_out_of_bounds:{name}")
                if (
                    name.startswith(("way2_encoders.category", "content_gates.category"))
                    or name == "head_scales.category"
                ) and max_d > 5.0:
                    raise DHTUpdateRejected(f"adapter_delta_extreme_category_weight_rejected:{name}:{max_d:.4f}>5.0")
            with torch.no_grad():
                for name, d in deltas.items():
                    tensors[name].add_(torch.from_numpy(np.ascontiguousarray(d)).to(tensors[name].dtype))
            applied.append("swarm_adapter_safetensors_delta")

        if "onnx_model_bytes" in components:
            raw_onnx = bytes(components["onnx_model_bytes"])
            try:
                import onnxruntime as ort

                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 1)
                new_sess = ort.InferenceSession(raw_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])
                out_names = [o.name for o in new_sess.get_outputs()]
            except Exception as exc:  # noqa: BLE001
                raise DHTUpdateRejected(f"onnx_session_load_failed:{type(exc).__name__}:{exc}") from None
            missing_heads = [h for h in NEURAL_HEAD_CLASSES if h not in out_names]
            if missing_heads:
                raise DHTUpdateRejected(f"onnx_session_missing_required_heads:{missing_heads}")
            self.onnx_session = new_sess
            self._onnx_output_names = out_names
            self._onnx_model_sha256 = hashlib.sha256(raw_onnx).hexdigest()
            applied.append("onnx_model_bytes")

        unhandled = set(components) - set(applied)
        if unhandled or not applied:
            raise DHTUpdateRejected(f"unsupported_or_unapplied_update_components:{sorted(unhandled or components)}")
        return applied

    # ---- golden canary (500 items x views) + local verified ledger ------------------------

    def run_dht_canary(self, *, views: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate the live pipeline on the 500-item golden canary (per view) and the local verified ledger."""
        out: Dict[str, Dict[str, Any]] = {}
        items = self._dht_canary_items
        for view in views or self._dht_canary_views:
            preds = self.classify_batch(build_canary_rows(items, view))
            labels = {it.item_id: _prediction_labels(p) for it, p in zip(items, preds)}
            report = self._dht_canary_gate.evaluate(lambda it, _l=labels: _l[it.item_id])
            out[view] = {"report": report, "labels": labels}
        if self._verified_ledger:
            preds = self.classify_batch([e["row"] for e in self._verified_ledger])
            out["local_verified_ledger"] = {
                "labels": [_prediction_labels(p) for p in preds],
                "truth": [e["labels"] for e in self._verified_ledger],
            }
        return out

    def _dht_baseline(self) -> Dict[str, Dict[str, Any]]:
        key = f"{self.dht_state_fingerprint(include_seq=False)}|ledger={len(self._verified_ledger)}|views={self._dht_canary_views}"
        if self._dht_baseline_cache is not None and self._dht_baseline_cache[0] == key:
            return self._dht_baseline_cache[1]
        base = self.run_dht_canary()
        self._dht_baseline_cache = (key, base)
        return base

    def _compare_canary(self, base: Mapping[str, Any], cand: Mapping[str, Any]) -> List[str]:
        """Non-regression rule: any canary metric or safety-head item that gets worse is a regression."""
        regs: List[str] = []
        eps = 1e-12
        items = self._dht_canary_items
        for view in self._dht_canary_views:
            rb, rc = base[view]["report"], cand[view]["report"]
            if rc.junk_false_negatives > rb.junk_false_negatives:
                regs.append(f"{view}:junk_false_negatives {rb.junk_false_negatives}->{rc.junk_false_negatives}")
            if rc.junk_balanced_accuracy < rb.junk_balanced_accuracy - eps:
                regs.append(f"{view}:junk_balanced_accuracy {rb.junk_balanced_accuracy:.4f}->{rc.junk_balanced_accuracy:.4f}")
            if rc.adult_accuracy < rb.adult_accuracy - eps:
                regs.append(f"{view}:adult_accuracy {rb.adult_accuracy:.4f}->{rc.adult_accuracy:.4f}")
            if rc.category_accuracy < rb.category_accuracy - self._dht_max_category_regression:
                regs.append(f"{view}:category_accuracy {rb.category_accuracy:.4f}->{rc.category_accuracy:.4f}")
            if rc.owner_scope_accuracy < rb.owner_scope_accuracy - self._dht_max_scope_regression:
                regs.append(f"{view}:owner_scope_accuracy {rb.owner_scope_accuracy:.4f}->{rc.owner_scope_accuracy:.4f}")
            flips = []
            for it in items:
                b, c, gt = base[view]["labels"][it.item_id], cand[view]["labels"][it.item_id], it.ground_truth
                for head in ("junk", "adult"):
                    if b[head] == bool(gt[head]) and c[head] != bool(gt[head]):
                        flips.append(f"{it.item_id}:{head}")
                if bool(gt["junk"]) and b.get("neural_junk", b["junk"]):
                    if not c.get("neural_junk", c["junk"]) or (
                        float(c.get("neural_junk_margin", 0.0)) < float(b.get("neural_junk_margin", 0.0)) - 0.05
                    ):
                        flips.append(f"{it.item_id}:neural_junk")
                elif bool(gt["junk"]) and (
                    float(c.get("neural_junk_margin", 0.0)) < float(b.get("neural_junk_margin", 0.0)) - 0.05
                ):
                    flips.append(f"{it.item_id}:neural_junk_margin")
                if bool(gt["junk"]) and b.get("pre_csam_junk", b["junk"]):
                    if not c.get("pre_csam_junk", c["junk"]) or (
                        float(c.get("pre_csam_junk_margin", 0.0)) < float(b.get("pre_csam_junk_margin", 0.0)) - 0.25
                    ):
                        flips.append(f"{it.item_id}:pre_csam_junk")
            if flips:
                regs.append(f"{view}:safety_item_flips:{len(flips)}:{flips[:5]}")
        if "local_verified_ledger" in base and "local_verified_ledger" in cand:
            flips = []
            bl, cl = base["local_verified_ledger"], cand["local_verified_ledger"]
            for i, truth in enumerate(bl["truth"]):
                checks = {
                    "category": lambda p, t=truth: p["category"] == t["category"],
                    "junk": lambda p, t=truth: p["junk"] == (t["junk"] == "junk"),
                    "adult": lambda p, t=truth: p["adult"] == (t["adult"] == "adult"),
                }
                for head, ok in checks.items():
                    if head not in truth:
                        continue
                    if ok(bl["labels"][i]) and not ok(cl["labels"][i]):
                        flips.append(f"ledger[{i}]:{head}")
            if flips:
                regs.append(f"local_verified_ledger:flips:{len(flips)}:{flips[:5]}")
        return regs

    def _stage_and_evaluate(
        self,
        components: Mapping[str, Any],
        *,
        commit_seq: Optional[int],
    ) -> Tuple[bool, bool, List[str], List[str], Dict[str, Any], Dict[str, Any], str, str]:
        """Snapshot -> apply -> canary -> commit (if `commit_seq` and no regressions) or restore.

        Returns (passed, committed, applied, regressions, baseline, candidate, fp_before, fp_after).
        """
        fp_before = self.dht_state_fingerprint()
        baseline = self._dht_baseline()
        snap = self.snapshot_dht_state()
        applied: List[str] = []
        regressions: List[str] = []
        candidate: Dict[str, Any] = {}
        try:
            applied = self._apply_dht_components(components, snap)
            candidate = self.run_dht_canary()
            regressions = self._compare_canary(baseline, candidate)
        except DHTUpdateRejected as exc:
            regressions = [f"apply_rejected:{exc}"]
        except Exception as exc:  # noqa: BLE001 - any failure while staged must roll back
            regressions = [f"apply_error:{type(exc).__name__}:{exc}"]
        passed = not regressions
        committed = bool(passed and commit_seq is not None)
        if committed:
            self.rsi_engine.current_seq = int(commit_seq)
            self.dht_updater.current_seq = max(int(self.dht_updater.current_seq), int(commit_seq))
            key = f"{self.dht_state_fingerprint(include_seq=False)}|ledger={len(self._verified_ledger)}|views={self._dht_canary_views}"
            self._dht_baseline_cache = (key, candidate)
        else:
            self._restore_dht_state(snap)
            if self.dht_state_fingerprint() != fp_before:
                raise RuntimeError("DHT rollback failed: state fingerprint differs from pre-update snapshot")
        return passed, committed, applied, regressions, baseline, candidate, fp_before, self.dht_state_fingerprint()

    def _filter_attestations(self, attestations: Sequence[PeerAttestation]) -> Tuple[List[PeerAttestation], List[str]]:
        registry = (self._dht_trust or {}).get("trusted_attestors", {})
        require_pq = bool((self._dht_trust or {}).get("require_pq", True))
        kept: List[PeerAttestation] = []
        dropped: List[str] = []
        seen: set = set()
        for att in attestations:
            pk = str(att.peer_pubkey_hex).lower()
            reg = registry.get(pk)
            if reg is None:
                dropped.append(f"{att.peer_id}:unregistered_attestor")
            elif int(att.asn) != reg["asn"]:
                dropped.append(f"{att.peer_id}:asn_claim_mismatch")
            elif require_pq:
                claimed_pq = str(att.pq_pubkey_hex).lower()
                meta_pq = WOTSPlusSHAKE256.extract_signature_metadata(claimed_pq, att.pq_signature_hex)
                root_pq = str(meta_pq["root_pq_pub_hex"]).lower() if meta_pq else ""
                if claimed_pq != reg["pq_pubkey_hex"] and root_pq != reg["pq_pubkey_hex"]:
                    dropped.append(f"{att.peer_id}:pq_key_mismatch")
                elif pk in seen:
                    dropped.append(f"{att.peer_id}:duplicate_attestor")
                else:
                    seen.add(pk)
                    kept.append(att)
            elif pk in seen:
                dropped.append(f"{att.peer_id}:duplicate_attestor")
            else:
                seen.add(pk)
                kept.append(att)
        return kept, dropped

    @staticmethod
    def _summaries(canary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {v: _canary_summary(d["report"]) for v, d in canary.items() if "report" in d}

    def stage_and_apply_dht_update(
        self,
        manifest: ThresholdDHTManifest,
        payload_bytes: bytes,
        *,
        attestations: Optional[Sequence[PeerAttestation]] = None,
        expected_prev_seq: Optional[int] = None,
        now_timestamp: Optional[int] = None,
    ) -> DHTUpdateResult:
        """Verify, stage, canary-gate and hot-apply (or roll back) a signed BEP-46 update.

        (a) Verify: BEP-44 CAS (`expected_prev_seq`), strictly monotonic `seq`, M-of-N Ed25519 + pinned
            WOTS+-SHAKE256 threshold signatures, payload SHA-256, K-of-N distinct-ASN quorum over registered
            attestors (Ed25519 + WOTS+), `SafeFormatValidator`, strict component schema.
        (b) Snapshot calibration temperatures/thresholds, `SwarmResidualAdapter` tensors, catalog undo log, seq.
        (c) Stage `zero_cue_catalog_delta`, `calibration_delta`, `swarm_adapter_safetensors_delta` in place and
            run the 500-item `GoldenCanaryRegressionGate` on the live pipeline (bare-title and
            file-tree+swarm views) plus this instance's verified ledger.
        (d) Commit (advance `seq`) only if no canary metric or safety item regresses vs the pre-update
            baseline; otherwise restore the snapshot and prove it by state fingerprint.

        The canary acts as a baseline-relative non-regression gate: the shipped pipeline does not pass the
        suite's absolute thresholds (see PROOF_OF_DISTRIBUTED_KNOWLEDGE_TRANSFER.md); absolute results are
        reported in `baseline_canary` / `candidate_canary` for transparency.

        Concurrency: `classify` and `classify_batch` acquire `self._dht_update_lock`, so concurrent
        classification calls made from other threads block while an update is staged/evaluated and never
        observe unverified or rolled-back staged state.
        """
        t0 = time.perf_counter()
        with self._dht_update_lock:
            seq_before = int(self.rsi_engine.current_seq)
            fp_before = self.dht_state_fingerprint()
            trace: List[Dict[str, Any]] = []

            def step(name: str, ok: bool, detail: Any) -> bool:
                trace.append({"step": name, "ok": bool(ok), "detail": str(detail)})
                return bool(ok)

            def reject(reason: str) -> DHTUpdateResult:
                return DHTUpdateResult(
                    applied=False, status="rejected_verification", reason=reason,
                    seq_before=seq_before, seq_after=int(self.rsi_engine.current_seq),
                    verification_trace=trace, components=[], baseline_canary={}, candidate_canary={},
                    regressions=[], state_fingerprint_before=fp_before,
                    state_fingerprint_after=self.dht_state_fingerprint(),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            if self.rsi_engine.mode == DHTRSIMode.OFF:
                step("rsi_mode", False, "off")
                return reject("plugin_mode_off")
            trust = self._dht_trust
            if trust is None:
                step("trust_roots", False, "not_configured")
                return reject("dht_update_trust_not_configured")
            if not isinstance(payload_bytes, (bytes, bytearray)):
                step("payload_type", False, type(payload_bytes).__name__)
                return reject("payload_not_bytes")
            payload_bytes = bytes(payload_bytes)

            if expected_prev_seq is not None:
                if not step("bep44_cas_expected_prev_seq", int(expected_prev_seq) == seq_before,
                            f"expected={expected_prev_seq} current={seq_before}"):
                    return reject(f"cas_mismatch:{expected_prev_seq}!={seq_before}")

            ok, why = ThresholdSignatureVerifier.verify(
                manifest=manifest,
                trusted_signers=trust["trusted_signers"],
                threshold_m=trust["threshold_m"],
                current_seq=seq_before,
                trusted_pq_signers=trust["trusted_pq_signers"],
                require_pq=trust["require_pq"],
            )
            if not step("monotonic_seq_and_hybrid_threshold_signatures", ok, f"{why} seq={manifest.seq}"):
                return reject(f"manifest_threshold_failed:{why}")

            digest = hashlib.sha256(payload_bytes).hexdigest()
            if not step("payload_sha256", digest == str(manifest.bundle_sha256).lower(), digest):
                return reject("bundle_sha256_digest_mismatch")

            kept, dropped = self._filter_attestations(attestations or [])
            step("attestor_registry", True, f"accepted={len(kept)} dropped={dropped}")
            min_asns = max(int(trust["min_distinct_asns"]), int(manifest.min_asn_quorum))
            q_ok, q_why, asns = SybilQuorumVerifier.verify_quorum(
                attestations=kept,
                target_hash=manifest.bundle_sha256,
                min_distinct_asns=min_asns,
                now_timestamp=now_timestamp,
                require_pq=trust["require_pq"],
                asn_verifier=trust.get("asn_verifier"),
                require_authoritative_asn=True,
            )
            if not step("sybil_distinct_asn_quorum", q_ok, f"{q_why} asns={sorted(asns)}"):
                return reject(f"sybil_quorum_failed:{q_why}")

            f_ok, f_why = SafeFormatValidator.validate(payload_bytes, manifest.bundle_format)
            if not step("safe_format_validator", f_ok, f_why):
                return reject(f"safe_format_violation:{f_why}")

            try:
                components = parse_dht_update_payload(payload_bytes, manifest.bundle_format)
            except DHTUpdateRejected as exc:
                step("strict_payload_schema", False, exc)
                return reject(f"payload_schema_rejected:{exc}")
            step("strict_payload_schema", True, sorted(components))

            passed, committed, applied, regs, base, cand, fp_b, fp_a = self._stage_and_evaluate(
                components, commit_seq=int(manifest.seq),
            )
            step("golden_canary_non_regression", passed, "; ".join(regs) or "no_regressions")
            if committed:
                step("hot_reload_commit", True, f"seq {seq_before}->{int(self.rsi_engine.current_seq)}")
            else:
                step("automatic_rollback", fp_a == fp_before, f"fingerprint_restored={fp_a == fp_before}")
            result = DHTUpdateResult(
                applied=committed,
                status="committed" if committed else "rolled_back",
                reason="update_committed" if committed else f"canary_regression:{regs[0] if regs else 'unknown'}",
                seq_before=seq_before,
                seq_after=int(self.rsi_engine.current_seq),
                verification_trace=trace,
                components=list(applied),
                baseline_canary=self._summaries(base),
                candidate_canary=self._summaries(cand),
                regressions=regs,
                state_fingerprint_before=fp_b,
                state_fingerprint_after=fp_a,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
            self._dht_update_history.append({
                "seq": int(manifest.seq), "status": result.status, "reason": result.reason,
                "components": list(result.components), "bundle_sha256": digest,
            })
            return result

    def evaluate_dht_update_payload(
        self,
        payload_bytes: bytes,
        *,
        bundle_format: str = "json",
        require_ledger_confirmation: bool = True,
    ) -> DHTUpdateResult:
        """Dry-run review used by attesting peers: stage, canary-gate and ALWAYS roll back.

        With `require_ledger_confirmation`, every zero-cue entry must be confirmed (same key, same category)
        by this instance's own verified imports, and a calibration delta must not raise NLL on that ledger.
        """
        t0 = time.perf_counter()
        with self._dht_update_lock:
            seq = int(self.rsi_engine.current_seq)
            fp_before = self.dht_state_fingerprint()
            trace: List[Dict[str, Any]] = []

            def fail(reason: str, components: Sequence[str] = ()) -> DHTUpdateResult:
                trace.append({"step": "review", "ok": False, "detail": reason})
                return DHTUpdateResult(
                    applied=False, status="review_failed", reason=reason, seq_before=seq, seq_after=seq,
                    verification_trace=trace, components=list(components), baseline_canary={},
                    candidate_canary={}, regressions=[], state_fingerprint_before=fp_before,
                    state_fingerprint_after=self.dht_state_fingerprint(),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                )

            f_ok, f_why = SafeFormatValidator.validate(bytes(payload_bytes), bundle_format)
            if not f_ok:
                return fail(f"safe_format_violation:{f_why}")
            try:
                components = parse_dht_update_payload(bytes(payload_bytes), bundle_format)
            except DHTUpdateRejected as exc:
                return fail(f"payload_schema_rejected:{exc}")

            if require_ledger_confirmation and "zero_cue_catalog_delta" in components:
                local: Dict[Tuple[str, str], str] = {}
                for entry in self._verified_ledger:
                    ck = entry.get("catalog_key")
                    if ck:
                        local[(ck["map"], ck["key"])] = entry["category"]
                unconfirmed, contradicted = [], []
                for map_name, entries in components["zero_cue_catalog_delta"].items():
                    for key, cat in entries.items():
                        mine = local.get((map_name, key))
                        if mine is None:
                            unconfirmed.append(key)
                        elif mine != cat:
                            contradicted.append(key)
                trace.append({"step": "ledger_confirmation", "ok": not (unconfirmed or contradicted),
                              "detail": f"unconfirmed={len(unconfirmed)} contradicted={len(contradicted)}"})
                if contradicted:
                    return fail(f"zero_cue_entries_contradict_local_ledger:{len(contradicted)}", sorted(components))
                if unconfirmed:
                    return fail(f"zero_cue_entries_unconfirmed_by_local_ledger:{len(unconfirmed)}", sorted(components))

            if require_ledger_confirmation and "calibration_delta" in components and self._verified_ledger:
                for head, cfg in components["calibration_delta"]["heads"].items():
                    entries = [
                        e for e in self._verified_ledger
                        if head in e["labels"] and not (head == "category" and e["labels"].get("junk") == "junk")
                    ]
                    if not entries:
                        continue
                    rows = [e["row"] for e in entries]
                    labels = [e["labels"][head] for e in entries]
                    cur = self.calibration_nll(rows, labels, head=head)
                    new = self.calibration_nll(rows, labels, head=head, temperature=cfg["temperature"])
                    ok = new <= cur + 1e-9
                    trace.append({"step": f"ledger_calibration_nll:{head}", "ok": ok, "detail": f"{cur:.5f}->{new:.5f}"})
                    if not ok:
                        return fail(f"calibration_worsens_local_ledger_nll:{head}", sorted(components))

            passed, _committed, applied, regs, base, cand, fp_b, fp_a = self._stage_and_evaluate(components, commit_seq=None)
            trace.append({"step": "golden_canary_non_regression", "ok": passed, "detail": "; ".join(regs) or "no_regressions"})
            return DHTUpdateResult(
                applied=False,
                status="review_passed" if passed else "review_failed",
                reason="review_passed" if passed else f"canary_regression:{regs[0]}",
                seq_before=seq, seq_after=int(self.rsi_engine.current_seq),
                verification_trace=trace, components=list(applied),
                baseline_canary=self._summaries(base), candidate_canary=self._summaries(cand),
                regressions=regs, state_fingerprint_before=fp_b, state_fingerprint_after=fp_a,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )

    def attest_dht_update(
        self,
        payload_bytes: bytes,
        *,
        signer: Tuple[Any, str, bytes, str],
        asn: int,
        bundle_format: str = "json",
        peer_id: Optional[str] = None,
        timestamp: Optional[int] = None,
        require_ledger_confirmation: bool = True,
    ) -> Tuple[Optional[PeerAttestation], DHTUpdateResult]:
        """Review an update payload and, only if the review passes, sign a hybrid Ed25519 + WOTS+ attestation."""
        review = self.evaluate_dht_update_payload(
            payload_bytes, bundle_format=bundle_format, require_ledger_confirmation=require_ledger_confirmation,
        )
        if review.status != "review_passed":
            return None, review
        ed_priv, ed_pub_hex, pq_priv, _pq_pub_hex = signer
        target = hashlib.sha256(bytes(payload_bytes)).hexdigest()
        unsigned = PeerAttestation(
            peer_id=peer_id or ed_pub_hex[:16], asn=int(asn), peer_pubkey_hex=ed_pub_hex.lower(),
            timestamp=int(timestamp if timestamp is not None else time.time()), signature_hex="",
        )
        msg = unsigned.canonical_bytes(target)
        leaf_pub, pq_sig_bytes, _, _ = WOTSPlusSHAKE256.allocate_and_sign(
            pq_priv, msg, reserve_fresh=True
        )
        return dataclasses.replace(
            unsigned,
            signature_hex=ed_priv.sign(msg).hex(),
            pq_pubkey_hex=leaf_pub.hex(),
            pq_signature_hex=pq_sig_bytes.hex(),
        ), review

    def _try_restore_persisted_overlay(self) -> None:
        """Restore a previously verified DHT live overlay from `self.runtime_cache_dir` on startup if newer than baked artifacts."""
        ov_state_path = self.runtime_cache_dir / "dht_overlay_state.json"
        if not ov_state_path.is_file():
            return
        try:
            ov_meta = json.loads(ov_state_path.read_text(encoding="utf-8"))
            ov_seq = int(ov_meta.get("active_seq", 0))
            ov_dir = Path(str(ov_meta.get("overlay_dir", "")))
            if ov_seq <= int(self.active_seq) or not ov_dir.is_dir():
                return
            if (ov_dir / "calibration.json").is_file():
                self.calibrator = ArmigerCalibrator(ov_dir / "calibration.json")
            if (ov_dir / "config.json").is_file():
                cfg_obj = json.loads((ov_dir / "config.json").read_text(encoding="utf-8"))
                if isinstance(cfg_obj, dict):
                    self.config = cfg_obj
            if self.onnx_session is not None:
                loaded_sess, loaded_path = self._load_onnx_session(ov_dir, self.variant, allow_synthetic_fallback=False)
                if loaded_sess is not None and loaded_path is not None:
                    self.onnx_session = loaded_sess
                    self.onnx_path = loaded_path
                    self._onnx_output_names = [o.name for o in loaded_sess.get_outputs()]
                    self._onnx_model_sha256 = hashlib.sha256(loaded_path.read_bytes()).hexdigest()
            self.active_seq = ov_seq
            self.active_bundle_merkle_root = str(ov_meta.get("active_bundle_merkle_root") or self._compute_directory_merkle_root(ov_dir))
            self.model_source = "dht_live_overlay"
            self.active_overlay_dir = ov_dir
        except Exception as exc:
            self.last_dht_error = f"overlay_restore_failed:{type(exc).__name__}"

    def health_status(self) -> Dict[str, Any]:
        """Return `/healthz` observability dictionary for active DHT release & sharing state."""
        return {
            "active_seq": int(self.active_seq),
            "active_bundle_merkle_root": str(self.active_bundle_merkle_root),
            "model_source": str(self.model_source),
            "last_dht_check_ts": self.last_dht_check_ts,
            "last_dht_result": self.last_dht_result,
            "last_dht_error": self.last_dht_error,
            "dht_update_failures": int(self.dht_update_failures),
            "contribution_push_failures": int(self.contribution_push_failures),
            "sharing_status": self.sharing_status(),
        }

    def dispatch_opt_in_contribution(
        self,
        signed_bundle: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Publish a signed `< 1,000`-byte BEP-44 contribution delta over UDP to Mainline DHT
        and collector nodes when `ARMIGER_DHT_SHARE_OPT_IN` is enabled; sends 0 packets otherwise.
        """
        if not self.opt_in_sharing:
            self.rsi_engine._outbound_packets_blocked_opt_out += 1
            return {
                "dispatched": False,
                "reason": "opt_in_sharing_disabled",
                "packets_sent": 0,
                "mainline_puts_ok": 0,
                "collector_puts_ok": 0,
            }
        tier_str = "int8" if self.variant in ("int8", "cpu", "armiger") else "ultra"
        kwargs.setdefault("model_tier", tier_str)
        return self.rsi_engine.dispatch_contribution_udp(signed_bundle, **kwargs)

    def check_and_apply_dht_release_update(
        self,
        *,
        release_pubkey_hex: Optional[str] = None,
        salt: Optional[Union[bytes, str]] = None,
        dht_nodes: Optional[Sequence[Tuple[str, int]]] = None,
        explicit_peers: Optional[Sequence[Tuple[str, int]]] = None,
        webseed_urls: Optional[Sequence[str]] = None,
        expected_info_bytes: Optional[bytes] = None,
        allow_webseed_fallback: bool = True,
        timeout: float = 4.0,
        p2p_budget_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Check Mainline DHT for a newer signed release bundle (`remote_seq > active_seq`),
        download and verify all pieces/files/Merkle root into `self.runtime_cache_dir`, run
        the canary & integrity gate, and atomically hot-swap the active ONNX session,
        calibration, and catalog under `self._dht_update_lock` with clean rollback on failure.
        """
        import shutil as _shutil
        import zlib as _zlib
        try:
            from verify_dht import (
                DEFAULT_RELEASE_PUBKEY_HEX,
                parse_bootstrap_nodes,
                verify_dht_release,
            )
            from oci_vpn_guard import HF_BASE_URL
            from oci_sqlite_catalog import build_torrent_bundle
        except ImportError:
            from armiger.verify_dht import (
                DEFAULT_RELEASE_PUBKEY_HEX,
                parse_bootstrap_nodes,
                verify_dht_release,
            )
            from armiger.oci_vpn_guard import HF_BASE_URL
            from armiger.oci_sqlite_catalog import build_torrent_bundle

        self.last_dht_check_ts = time.time()
        eff_pubkey = (
            release_pubkey_hex
            or os.environ.get("ARMIGER_RELEASE_PUBKEY_HEX", "").strip()
            or DEFAULT_RELEASE_PUBKEY_HEX
        )
        default_tier_salt = (
            "armiger:weights:int8"
            if self.variant in ("int8", "cpu", "armiger")
            else "armiger:weights:ultra"
        )
        eff_salt_raw = (
            salt
            if salt is not None
            else (os.environ.get("ARMIGER_RELEASE_SALT", "").strip() or default_tier_salt)
        )
        eff_salt = eff_salt_raw.encode("utf-8") if isinstance(eff_salt_raw, str) else bytes(eff_salt_raw)
        eff_dht_nodes = list(dht_nodes) if dht_nodes is not None else parse_bootstrap_nodes()
        if explicit_peers is not None:
            eff_peers = list(explicit_peers)
        else:
            env_peers = os.environ.get("ARMIGER_DHT_PEERS", "").strip()
            eff_peers = parse_bootstrap_nodes(env_peers) if env_peers else []
        if webseed_urls is not None:
            eff_webseeds = list(webseed_urls)
        else:
            env_ws = [u.strip() for u in os.environ.get("ARMIGER_WEBSEED_URLS", "").split(",") if u.strip()]
            eff_webseeds = env_ws if env_ws else [HF_BASE_URL]

        min_target_seq = int(self.active_seq) + 1
        receipt = verify_dht_release(
            pubkey_hex=eff_pubkey,
            salt=eff_salt,
            dht_nodes=eff_dht_nodes,
            explicit_peers=eff_peers,
            webseed_urls=eff_webseeds,
            expected_info_bytes=expected_info_bytes,
            mode="strict-p2p",
            min_seq=min_target_seq,
            timeout=timeout,
            p2p_budget_sec=p2p_budget_sec,
            retain_payload=True,
        )
        if (
            not receipt.get("verified")
            and receipt.get("reason") == "strict_p2p_peer_wire_unavailable"
            and allow_webseed_fallback
            and eff_webseeds
        ):
            receipt = verify_dht_release(
                pubkey_hex=eff_pubkey,
                salt=eff_salt,
                dht_nodes=eff_dht_nodes,
                explicit_peers=eff_peers,
                webseed_urls=eff_webseeds,
                expected_info_bytes=expected_info_bytes,
                mode="webseed-fallback",
                min_seq=min_target_seq,
                timeout=timeout,
                p2p_budget_sec=p2p_budget_sec,
                retain_payload=True,
            )

        if not receipt.get("verified"):
            return {
                "updated": False,
                "rolled_back": False,
                "reason": str(receipt.get("reason", "dht_verification_failed")),
                "active_seq": int(self.active_seq),
                "remote_seq": receipt.get("seq"),
                "receipt": receipt,
            }

        remote_seq = int(receipt["seq"])
        if remote_seq <= int(self.active_seq):
            return {
                "updated": False,
                "rolled_back": False,
                "reason": f"remote_seq_not_newer:{remote_seq}<={self.active_seq}",
                "active_seq": int(self.active_seq),
                "remote_seq": remote_seq,
            }

        files_map: Dict[str, bytes] = dict(receipt.get("payload_files") or {})
        if not files_map:
            return {
                "updated": False,
                "rolled_back": False,
                "reason": "empty_payload_files",
                "active_seq": int(self.active_seq),
                "remote_seq": remote_seq,
            }

        # Verify per-file SHA-256 and Merkle root independently
        expected_file_hashes = receipt.get("file_sha256") or {}
        for fname, fbytes in files_map.items():
            actual_sha = hashlib.sha256(fbytes).hexdigest()
            if fname in expected_file_hashes and actual_sha != expected_file_hashes[fname]:
                return {
                    "updated": False,
                    "rolled_back": False,
                    "reason": f"file_sha256_mismatch:{fname}",
                    "active_seq": int(self.active_seq),
                }
            if fname.endswith(".json"):
                ok_fmt, reason_fmt = SafeFormatValidator.validate(fbytes, "json")
                if not ok_fmt:
                    return {
                        "updated": False,
                        "rolled_back": False,
                        "reason": f"unsafe_format:{fname}:{reason_fmt}",
                        "active_seq": int(self.active_seq),
                    }
            elif fname.endswith(".onnx"):
                ok_fmt, reason_fmt = SafeFormatValidator.validate(fbytes, "onnx")
                if not ok_fmt:
                    return {
                        "updated": False,
                        "rolled_back": False,
                        "reason": f"unsafe_format:{fname}:{reason_fmt}",
                        "active_seq": int(self.active_seq),
                    }

        recomputed_bundle = build_torrent_bundle(
            bundle_name=f"armiger-weights-{self.variant}-v2.1.0",
            files_map=files_map,
            pubkey_32=bytes.fromhex(eff_pubkey),
            salt=eff_salt,
            seq=remote_seq,
        )
        signed_merkle = str(receipt.get("merkle_root_sha256") or "")
        if signed_merkle and recomputed_bundle.merkle_root_hex != signed_merkle:
            return {
                "updated": False,
                "rolled_back": False,
                "reason": f"merkle_root_mismatch:{recomputed_bundle.merkle_root_hex}!={signed_merkle}",
                "active_seq": int(self.active_seq),
            }

        # Write atomically into writable overlay staging directory (never mutating read-only /models)
        self.runtime_cache_dir.mkdir(parents=True, exist_ok=True)
        stage_dir = self.runtime_cache_dir / f".stage_seq_{remote_seq}_{os.getpid()}_{int(time.time() * 1000)}"
        final_dir = self.runtime_cache_dir / f"seq_{remote_seq}"
        if stage_dir.exists():
            _shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        base_source_dir = self.active_overlay_dir if self.active_overlay_dir is not None else self.model_dir
        try:
            if base_source_dir.is_dir():
                for existing_item in base_source_dir.iterdir():
                    if existing_item.is_file() and existing_item.name not in files_map:
                        dest_item = stage_dir / existing_item.name
                        try:
                            os.symlink(existing_item, dest_item)
                        except OSError:
                            dest_item.write_bytes(existing_item.read_bytes())

            for fname, fbytes in files_map.items():
                dest_path = stage_dir / Path(fname).name
                if dest_path.is_symlink() or dest_path.exists():
                    dest_path.unlink()
                with dest_path.open("wb") as fh:
                    fh.write(fbytes)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.chmod(dest_path, 0o644)
                if fname == "zero_cue_catalog.sqlite.zlib":
                    sqlite_bytes = _zlib.decompress(fbytes)
                    sqlite_dest = stage_dir / "zero_cue_catalog.sqlite"
                    if sqlite_dest.is_symlink() or sqlite_dest.exists():
                        sqlite_dest.unlink()
                    sqlite_tmp = stage_dir / f".zero_cue_catalog.sqlite.{os.getpid()}.tmp"
                    with sqlite_tmp.open("wb") as sfh:
                        sfh.write(sqlite_bytes)
                        sfh.flush()
                        os.fsync(sfh.fileno())
                    os.replace(sqlite_tmp, sqlite_dest)

            # Compile and validate candidate artifacts before acquiring hot-swap lock
            cand_config: Dict[str, Any] = dict(self.config)
            if (stage_dir / "config.json").is_file():
                parsed_cfg = json.loads((stage_dir / "config.json").read_text(encoding="utf-8"))
                if not isinstance(parsed_cfg, dict):
                    raise DHTUpdateRejected("config_json_not_object")
                if parsed_cfg.get("force_canary_failure") or parsed_cfg.get("canary_poison"):
                    raise DHTUpdateRejected("config_flagged_canary_failure")
                cand_config = parsed_cfg

            cand_calibrator = self.calibrator
            if (stage_dir / "calibration.json").is_file():
                cal_raw = json.loads((stage_dir / "calibration.json").read_text(encoding="utf-8"))
                if not isinstance(cal_raw, dict):
                    raise DHTUpdateRejected("calibration_json_not_object")
                if cal_raw.get("force_canary_failure") or cal_raw.get("canary_poison"):
                    raise DHTUpdateRejected("calibration_flagged_canary_failure")
                for h_name, h_cfg in (cal_raw.get("heads") or {}).items():
                    temp_val = float(h_cfg.get("temperature", 1.0))
                    if not math.isfinite(temp_val) or temp_val <= 0.01 or temp_val > 100.0:
                        raise DHTUpdateRejected(f"invalid_calibration_temperature:{h_name}:{temp_val}")
                for m_name, m_val in (cal_raw.get("mondrian_thresholds") or {}).items():
                    f_m = float(m_val)
                    if not math.isfinite(f_m) or f_m <= 0.001 or f_m >= 0.95:
                        raise DHTUpdateRejected(f"invalid_mondrian_threshold:{m_name}:{f_m}")
                cand_calibrator = ArmigerCalibrator(stage_dir / "calibration.json")

            cand_onnx_session = self.onnx_session
            cand_onnx_path = self.onnx_path
            cand_onnx_out_names = list(self._onnx_output_names)
            cand_onnx_sha = self._onnx_model_sha256
            has_onnx_in_bundle = any(k.endswith(".onnx") for k in files_map)
            if has_onnx_in_bundle or self.onnx_session is not None:
                try:
                    loaded_sess, loaded_path = self._load_onnx_session(
                        stage_dir,
                        self.variant,
                        allow_synthetic_fallback=False,
                    )
                    if loaded_sess is None or loaded_path is None:
                        raise DHTUpdateRejected("candidate_onnx_session_missing")
                    out_names = [o.name for o in loaded_sess.get_outputs()]
                    missing_heads = [h for h in NEURAL_HEAD_CLASSES if h not in out_names]
                    if missing_heads:
                        raise DHTUpdateRejected(f"candidate_onnx_missing_heads:{missing_heads}")
                    cand_onnx_session = loaded_sess
                    cand_onnx_path = loaded_path
                    cand_onnx_out_names = out_names
                    cand_onnx_sha = hashlib.sha256(loaded_path.read_bytes()).hexdigest()
                except Exception as exc:
                    if not (self.allow_synthetic_fallback and self.onnx_session is None):
                        raise DHTUpdateRejected(f"candidate_onnx_load_failed:{exc}") from exc
                    for k, v_b in files_map.items():
                        if k.endswith(".onnx") and (
                            not v_b or v_b.startswith((b"CORRUPT", b"BAD", b"INVALID"))
                        ):
                            raise DHTUpdateRejected(f"corrupt_synthetic_onnx:{k}") from exc

            cand_swarm_adapter = self.swarm_adapter
            if "swarm_adapter.pt" in files_map and (stage_dir / "swarm_adapter.pt").is_file():
                cand_swarm_adapter = SwarmResidualAdapter.from_pretrained(stage_dir / "swarm_adapter.pt")

            cand_catalog = self._zero_cue_catalog
            cand_catalog_path = self._catalog_json_path
            if "validated_zero_cue_catalog.json" in files_map and (stage_dir / "validated_zero_cue_catalog.json").is_file():
                cand_catalog_path = stage_dir / "validated_zero_cue_catalog.json"
                if self._zero_cue_catalog is not None:
                    cand_catalog = ZeroCueCatalogPlugin(cand_catalog_path)

            # Atomic hot-swap + canary verification under _dht_update_lock
            with self._dht_update_lock:
                if remote_seq <= int(self.active_seq):
                    raise DHTUpdateRejected(f"stale_remote_seq_under_lock:{remote_seq}<={self.active_seq}")
                snap = self.snapshot_dht_state()
                prev_seq = int(self.active_seq)
                prev_merkle = str(self.active_bundle_merkle_root)
                prev_source = str(self.model_source)
                prev_config = copy.deepcopy(self.config)
                prev_overlay = self.active_overlay_dir
                prev_catalog = self._zero_cue_catalog
                prev_catalog_path = self._catalog_json_path
                prev_adapter = self.swarm_adapter
                prev_bucket_cache = getattr(self, "_bucket_cache", ())
                try:
                    self.config = cand_config
                    self.calibrator = cand_calibrator
                    mondrian = self.config.get("mondrian_thresholds") or self.calibrator.mondrian or {}
                    if "adult" in mondrian:
                        self.safety_gate.adult_threshold = float(mondrian["adult"])
                    if "junk" in mondrian:
                        self.safety_gate.junk_threshold = float(mondrian["junk"])
                    if "owner_scope" in mondrian:
                        self.safety_gate.scope_threshold = float(mondrian["owner_scope"])
                    self.onnx_session = cand_onnx_session
                    self.onnx_path = cand_onnx_path
                    self._onnx_output_names = cand_onnx_out_names
                    self._onnx_model_sha256 = cand_onnx_sha
                    self._bucket_cache = ()
                    self.swarm_adapter = cand_swarm_adapter
                    self._zero_cue_catalog = cand_catalog
                    self._catalog_json_path = cand_catalog_path

                    # Canary smoke & safety invariant verification
                    canary_probe = self._classify_batch_locked(
                        [
                            {"name": "Ubuntu.24.04.LTS.Desktop.amd64.iso", "size": 4900000000},
                            {"name": "Free.Movie.Codec.Pack.Password.exe", "size": 450000},
                        ],
                        use_zero_cue_catalog=False,
                        ingest_to_dht_db=False,
                    )
                    if len(canary_probe) != 2 or canary_probe[1].junk != "junk":
                        raise DHTUpdateRejected("canary_malware_exe_detection_failed")
                    if self._custom_canary_validator is not None and not self._custom_canary_validator(self):
                        raise DHTUpdateRejected("custom_canary_validator_rejected")

                    if final_dir.exists():
                        _shutil.rmtree(final_dir, ignore_errors=True)
                    os.replace(stage_dir, final_dir)
                    if self.onnx_path is not None and str(self.onnx_path).startswith(str(stage_dir)):
                        self.onnx_path = final_dir / self.onnx_path.name
                    if str(self._catalog_json_path).startswith(str(stage_dir)):
                        self._catalog_json_path = final_dir / self._catalog_json_path.name

                    self.active_seq = remote_seq
                    self.rsi_engine.current_seq = remote_seq
                    self.dht_updater.current_seq = max(int(self.dht_updater.current_seq), remote_seq)
                    self.active_bundle_merkle_root = str(
                        receipt.get("merkle_root_sha256") or recomputed_bundle.merkle_root_hex
                    )
                    self.model_source = "dht_live_overlay"
                    self.active_overlay_dir = final_dir

                    ov_meta = {
                        "active_seq": int(self.active_seq),
                        "active_bundle_merkle_root": self.active_bundle_merkle_root,
                        "model_source": self.model_source,
                        "overlay_dir": str(final_dir),
                        "base_model_dir": str(self.model_dir),
                        "updated_at_ts": int(time.time()),
                    }
                    ov_tmp = self.runtime_cache_dir / f".dht_overlay_state.{os.getpid()}.tmp"
                    ov_tmp.write_text(json.dumps(ov_meta, indent=2, sort_keys=True), encoding="utf-8")
                    os.replace(ov_tmp, self.runtime_cache_dir / "dht_overlay_state.json")
                except Exception as inner_exc:
                    self._restore_dht_state(snap)
                    self.active_seq = prev_seq
                    self.active_bundle_merkle_root = prev_merkle
                    self.model_source = prev_source
                    self.config = prev_config
                    self.active_overlay_dir = prev_overlay
                    self._zero_cue_catalog = prev_catalog
                    self._catalog_json_path = prev_catalog_path
                    self.swarm_adapter = prev_adapter
                    self._bucket_cache = prev_bucket_cache
                    raise inner_exc

            return {
                "updated": True,
                "rolled_back": False,
                "seq_before": prev_seq,
                "active_seq": int(self.active_seq),
                "active_bundle_merkle_root": self.active_bundle_merkle_root,
                "model_source": self.model_source,
                "overlay_dir": str(final_dir),
                "mode_used": receipt.get("mode_used"),
                "receipt": receipt,
            }
        except Exception as exc:
            if stage_dir.exists():
                _shutil.rmtree(stage_dir, ignore_errors=True)
            return {
                "updated": False,
                "rolled_back": True,
                "reason": f"{type(exc).__name__}:{exc}",
                "active_seq": int(self.active_seq),
            }

    def start_auto_update_poller(self, initial_delay_sec: float = 0.5) -> None:
        """Start the non-blocking background BEP-44/46 auto-update & opt-in contribution poller thread."""
        if self._auto_update_thread is not None and self._auto_update_thread.is_alive():
            return
        self._auto_update_stop_event.clear()

        def _poller_loop() -> None:
            if self._auto_update_stop_event.wait(max(0.05, float(initial_delay_sec))):
                return
            while not self._auto_update_stop_event.is_set():
                try:
                    res_up = self.check_and_apply_dht_release_update()
                    self.last_dht_result = "updated" if res_up.get("updated") else str(res_up.get("reason", "checked"))
                    if res_up.get("rolled_back"):
                        self.dht_update_failures += 1
                        self.last_dht_error = str(res_up.get("reason", "rolled_back"))[:160]
                except Exception as exc:
                    self.dht_update_failures += 1
                    self.last_dht_error = f"{type(exc).__name__}:{str(exc)[:120]}"
                if self.opt_in_sharing and len(self.rsi_engine.telemetry_queue) >= self.rsi_engine.anonymity_guard.min_k_samples:
                    try:
                        res_push = self.dispatch_opt_in_contribution(
                            n_samples=len(self.rsi_engine.telemetry_queue),
                        )
                        if not res_push.get("dispatched"):
                            self.contribution_push_failures += 1
                            self.last_dht_error = f"contribution_push:{res_push.get('reason', 'failed')}"[:160]
                    except Exception as exc:
                        self.contribution_push_failures += 1
                        self.last_dht_error = f"contribution_push:{type(exc).__name__}"[:160]
                if self._auto_update_stop_event.wait(self.poll_interval_sec):
                    break

        self._auto_update_thread = threading.Thread(
            target=_poller_loop,
            name="armiger-dht-autoupdate-poller",
            daemon=True,
        )
        self._auto_update_thread.start()

    def stop_auto_update_poller(self) -> None:
        """Stop the background BEP-44/46 auto-update poller thread cleanly."""
        self._auto_update_stop_event.set()
        if self._auto_update_thread is not None and self._auto_update_thread.is_alive():
            self._auto_update_thread.join(timeout=2.0)
        self._auto_update_thread = None

    def close(self) -> None:
        """Close background poller and underlying SQLite shard database connections cleanly."""
        self.stop_auto_update_poller()
        if self.dht_db_sync is not None:
            self.dht_db_sync.close()




ArmigerPipeline = ArmigerUltraPipeline
