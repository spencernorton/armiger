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
    from armiger_ultra_pipeline import ArmigerUltraPipeline

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

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
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
    SeriesWebhookAdapter,
    TelemetryAnonymizer,
    ThresholdDHTManifest,
    WOTSPlusSHAKE256,
    X25519ChaCha20Poly1305Engine,
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


from armiger_inference import (
    ArmigerStandaloneONNX,
    _get_vocab_ext_map,
    _hash_ext,
    _hash_word,
    _log2_size_bucket,
    build_onnx_batch_inputs,
    encode_torrent_features,
)


_EPISODE_FILE_RE = re.compile(r"(?i)(?:\bS\d{1,2}E\d{1,3}\b|\bS\d{1,2}\b|\bSeason\s*\d+\b)")


def _file_tree_structural_override(row: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    """Check physical file-tree extensions (.m4b audiobooks, .cbz/.cbr comics, multi-episode packs) before title lookup."""
    files = row.get("files") or []
    ep_files = 0
    for f in files:
        p = (f.get("path") if isinstance(f, Mapping) else str(f or "")) or ""
        low = p.lower().strip()
        if low.endswith(".m4b"):
            return "audiobook", "structural_m4b"
        if low.endswith((".cbz", ".cbr", ".cb7")):
            return "comic", "structural_comic_archive"
        if low.endswith((".mkv", ".mp4", ".avi", ".ts")) and _EPISODE_FILE_RE.search(p):
            ep_files += 1
    if ep_files >= 2:
        return "tv_show", "structural_multi_episode_pack"
    return None, "none"


# =====================================================================
# Plugin #4 Object Wrapper: AudioLanguageGuard
# =====================================================================

class AudioLanguageGuard:
    """Object-oriented wrapper around `audio_language_guard.py` for SDK pipeline composition."""

    _DUAL_MULTI_RE = re.compile(
        r"(?i)(?:\bdual[\s._-]*audio\b|\bmulti[\s._-]*audio\b|\bmulti\s*\([^\)]*\b(?:en|eng|english)\b|"
        r"\[\s*(?:rus|jpn|jap|ita|fre|ger|deu|spa|lat)[^\]]*\+\s*(?:en|eng|english)\b)"
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

        # Explicit dual/multi-audio release tags
        if self._DUAL_MULTI_RE.search(raw_name):
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
        )
        if not report["is_anime"]:
            dub_status = "non_anime"
        elif report["is_anime_dub_or_dual"] or self._DUAL_MULTI_RE.search(raw_name):
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
        model_dir: Union[str, Path],
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
        rsi_mode: str = "anonymous_contribute",
        dht_sync_subscription: Optional[SubscriptionConfig] = None,
        trusted_root_pubkeys_hex: Optional[Sequence[str]] = None,
        dht_master_secret: Optional[bytes] = None,
    ) -> None:
        self.model_dir = Path(model_dir).resolve()
        self.variant = variant.lower().strip()
        self.config_path = self.model_dir / "config.json"
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        # 1. Load ONNX Runtime session (fp32 ultra or int8 cpu)
        self.onnx_session = onnx_Session
        self.onnx_path: Optional[Path] = None
        if self.onnx_session is None:
            self.onnx_session, self.onnx_path = self._load_onnx_session(self.model_dir, self.variant)
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
        self.rsi_engine = rsi_engine or ArmigerDHTRSIEngine(mode=rsi_mode)

        # 7. Plugin #6: ArmigerDHTDBSync + CSAMGuard (`armiger_dht_db_sync.py` + `csam_guard.py`)
        mondrian = self.config.get("mondrian_thresholds", {})
        self.safety_gate = safety_gate or ArmigerSafetyGate(
            adult_threshold=float(mondrian.get("adult", 0.5451)),
            junk_threshold=float(mondrian.get("junk", 0.1010)),
            scope_threshold=float(mondrian.get("owner_scope", 0.4770)),
        )
        self.dht_db_sync = dht_db_sync or ArmigerDHTDatabaseSync(
            subscription=dht_sync_subscription or SubscriptionConfig(sync_mode=SyncMode.SUBSCRIBE_AND_SEED),
            master_secret=dht_master_secret,
            trusted_root_pubkeys_hex=trusted_root_pubkeys_hex,
        )
        self.csam_guard = csam_guard or self.dht_db_sync.csam_guard

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
        rsi_mode: str = "anonymous_contribute",
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
        rsi_mode: str = "anonymous_contribute",
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
        """Hugging Face style loader supporting local directory paths or `huggingface_hub.snapshot_download`."""
        cand_path = Path(model_name_or_path)
        if cand_path.exists() and cand_path.is_dir():
            return cls.from_Directory(cand_path, **kwargs)
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=str(model_name_or_path))
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
    def _load_onnx_session(model_dir: Path, variant: str) -> Tuple[Any, Optional[Path]]:
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
        except Exception:
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
            anime_probs = np.asarray(self.calibrator.calibrate_head("anime", raw_logits["anime"][idx].tolist()), dtype=np.float64)
            eng_subs_probs = np.asarray(self.calibrator.calibrate_head("english_subs", raw_logits["english_subs"][idx].tolist()), dtype=np.float64)
            lang_probs = np.asarray(self.calibrator.calibrate_head("lang", raw_logits["lang"][idx].tolist()), dtype=np.float64)
            scope_probs = np.asarray(self.calibrator.calibrate_head("owner_scope", raw_logits["owner_scope"][idx].tolist()), dtype=np.float64)
            title_lang_probs = np.asarray(self.calibrator.calibrate_head("title_language", raw_logits["title_language"][idx].tolist()), dtype=np.float64)

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
                )
                pred_category = cat_classes[c_idx]
                if catalog_source != "neural_backbone":
                    cat_probs[c_idx] = max(float(cat_probs[c_idx]), 0.92)
                    cat_probs /= cat_probs.sum()

            # Step 4: Plugin #4 (`AudioLanguageGuard`) — Cross-Head Audio Language & Anime Dub Guard
            pred_lang, lang_probs, _ = self.audio_guard.apply(row_dict, lang_probs, title_lang_probs)
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

            way3_max_anom = float(np.max(way3_vec[:6])) if len(way3_vec) >= 6 else 0.0
            trap_prob = float(max(swarm_eval["trap_score"], way3_max_anom * 0.90))
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
            p_excluded = float(scope_probs[1])

            if not csam_result.is_safe and csam_result.violation_type == CSAMViolationType.MALWARE_PAYLOAD:
                p_junk = max(p_junk, 0.99)
                junk_probs = np.array([1.0 - p_junk, p_junk], dtype=np.float64)

            gate_passed, gate_reason, gate_violation = self.safety_gate.evaluate(
                p_adult=p_adult,
                p_junk=p_junk,
                p_excluded_scope=p_excluded,
                target_category=pred_category,
            )

            pred_adult = "adult" if p_adult >= self.safety_gate.adult_threshold else "non_adult"
            pred_junk = "junk" if (p_junk >= self.safety_gate.junk_threshold or not csam_result.is_safe) else "clean"
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
        if isinstance(observation_or_webhook, CandidateObservation):
            cand = observation_or_webhook
        else:
            payload = dict(observation_or_webhook)
            ad_lower = adapter.lower().strip()
            if ad_lower == "series_webhook":
                cand = SeriesWebhookAdapter.parse_webhook(payload)
            elif ad_lower == "movie_webhook":
                cand = MovieWebhookAdapter.parse_webhook(payload)
            elif ad_lower == "audio_webhook":
                cand = AudioWebhookAdapter.parse_webhook(payload)
            elif ad_lower == "publication_webhook":
                cand = PublicationWebhookAdapter.parse_webhook(payload)
            elif ad_lower in ("qbittorrent", "qb"):
                cand = QBittorrentAdapter.parse_event(payload)
            else:
                cand = CandidateObservation(**payload)

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

    def close(self) -> None:
        """Close underlying SQLite shard database connections cleanly."""
        if self.dht_db_sync is not None:
            self.dht_db_sync.close()
