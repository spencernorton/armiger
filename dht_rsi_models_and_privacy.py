"""Armiger BEP-44 / BEP-46 Distributed Recursive Self-Improvement (RSI) & Distributed Data Plugin.

Standalone decentralized RSI engine for Armiger and Armiger Ultra.
Supports:
  1. Universal Adapters: Series, Movie, Audio, and Publication webhooks, ffprobe/MediaInfo stream
     verification, and qBittorrent/Transmission completion telemetry (client-agnostic across media automation stacks).
  2. Foolproof Physical Ground-Truth Verification: Validates physical download completion, stream
     decodability, and canonical catalog matches. Rejects raw unverified model predictions.
  3. Complete Anonymity & Privacy Engine: PII, IP, and private tracker passkey stripping, salted
     blinded title stems, differential privacy (Laplace perturbation + logarithmic size binning),
     and Poisson timing jitter against timing correlation attacks.
  4. Post-Quantum End-to-End Encryption: Hybrid X25519 + NIST FIPS 203 ML-KEM-768 key encapsulation
     combined via SHAKE-256 / HKDF-SHA3-256 with 256-bit ChaCha20-Poly1305 AEAD authentication.
  5. Post-Quantum Spoof-Proof Protocol: BEP-44 strictly monotonic sequence numbers + Hybrid Ed25519
     and WOTS+-SHAKE256 (RFC 8391 / NIST FIPS 205 hash-based PQ) threshold multi-signatures ($M$-of-$N$ quorum).
  6. Poison-Proof Guards: $K$-of-$N$ distinct-ASN Sybil quorum verification, strict format
     whitelisting (SafeTensors, ONNX, JSON only; rejects arbitrary pickle/executable bytecode), and
     a mandatory 500-item golden canary regression gate verifying Mondrian conformal safety bounds.
  7. Opt-In / Opt-Out Modes: 'off', 'pull_only', and 'anonymous_contribute'.
"""
from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import hmac
import ipaddress
import json
import math
import os
import pathlib
import random
import secrets
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# =====================================================================
# 1. Enums and Configuration Types
# =====================================================================

class DHTRSIMode(str, enum.Enum):
    """Operational mode controlling network participation and privacy."""
    OFF = "off"
    PULL_ONLY = "pull_only"
    ANONYMOUS_CONTRIBUTE = "anonymous_contribute"


class MediaCategory(str, enum.Enum):
    """Armiger 11-class classification categories."""
    MOVIE = "movie"
    TV_SHOW = "tv_show"
    XXX = "xxx"
    MUSIC = "music"
    COMIC = "comic"
    GAME = "game"
    AUDIOBOOK = "audiobook"
    SOFTWARE = "software"
    EBOOK = "ebook"
    COURSE = "course"
    UNKNOWN = "unknown"


# v3 owner-scope policy: these categories are always `excluded`; the canary gate and the BEP-46
# Mondrian drift guard measure "excluded rejection accuracy" over exactly this set.
EXCLUDED_SCOPE_CATEGORIES: frozenset = frozenset({"course", "ebook", "audiobook", "software", "game", "music", "comic"})

# Deterministic episode-structure cue (SxxEyy / 1x02 / Ep.03 / " - 02 "): structural, not neural, evidence that
# a release is episodic. Used by the aggregator to block single-file Movie<->TV misclassifications.
EPISODE_MARKER_RE = re.compile(
    r"(?<![a-z0-9])s\d{1,2}[ ._-]?e\d{1,3}(?![a-z0-9])"
    r"|(?<![a-z0-9])\d{1,2}x\d{2,3}(?![a-z0-9])"
    r"|(?<![a-z0-9])(?:ep|episode)[ ._-]?\d{1,3}(?![a-z0-9])"
    r"|\s-\s\d{2,3}(?=\s|\[|\(|$)",
    re.IGNORECASE,
)


class BundleFormat(str, enum.Enum):
    """Permitted safe serialization formats (rejects pickle, pth, bin)."""
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    JSON = "json"


# =====================================================================
# 2. Data Structures
# =====================================================================

@dataclasses.dataclass
class CandidateObservation:
    """Raw ingested observation from download clients, media inspection, or media library webhooks."""
    source_adapter: str  # 'series_webhook', 'movie_webhook', 'audio_webhook', 'publication_webhook', 'qbittorrent', 'transmission', 'ffprobe'
    raw_title: str
    claimed_category: str
    download_completed: bool
    bytes_downloaded: int
    total_bytes: int
    file_path: Optional[str] = None
    catalog_id_type: Optional[str] = None  # 'series_catalog', 'movie_catalog', 'audio_catalog', 'isbn', 'anime_catalog'
    catalog_id_value: Optional[Union[str, int]] = None
    stream_metadata: Optional[Dict[str, Any]] = None
    client_state: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


@dataclasses.dataclass(frozen=True)
class VerifiedObservation:
    """Foolproof physical ground-truth observation verified by download and stream inspection."""
    category: str
    adult: bool
    anime: bool
    lang: str  # 'english', 'non_english', 'multi_or_unknown'
    junk: bool
    owner_scope: str  # 'allowed_type', 'excluded'
    english_subs: bool
    title_language: str  # 'english_title', 'non_english_title'
    source_adapter: str
    catalog_id_type: str
    catalog_id_value: str
    size_bytes: int
    raw_title: str
    stream_info: Dict[str, Any]
    verified_at: int
    evidence_masks: Dict[str, bool] = dataclasses.field(default_factory=dict)
    evidence_sources: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class PhysicalVerificationResult:
    """Result of physical ground-truth verification."""
    is_verified: bool
    rejection_reason: Optional[str] = None
    observation: Optional[VerifiedObservation] = None


@dataclasses.dataclass(frozen=True)
class BlindedTitleStem:
    """Blinded syntactic title stem preserving feature distribution without leaking title."""
    blinded_hash: str
    token_features: List[str]
    length_bucket: str


@dataclasses.dataclass(frozen=True)
class AnonymizedTelemetryRecord:
    """Fully sanitized, anonymized, and differential-privacy perturbed telemetry record."""
    blinded_stem: BlindedTitleStem
    category: str
    adult: bool
    anime: bool
    lang: str
    junk: bool
    owner_scope: str
    english_subs: bool
    title_language: str
    dp_size_mb: float
    dp_duration_mins: Optional[float]
    verified_features: Dict[str, Any]
    timestamp: int
    evidence_masks: Dict[str, bool] = dataclasses.field(default_factory=dict)

    def canonical_json_bytes(self) -> bytes:
        data: Dict[str, Any] = {
            "adult": self.adult,
            "anime": self.anime,
            "blinded_hash": self.blinded_stem.blinded_hash,
            "category": self.category,
            "dp_duration_mins": self.dp_duration_mins,
            "dp_size_mb": round(self.dp_size_mb, 2),
            "english_subs": self.english_subs,
            "junk": self.junk,
            "lang": self.lang,
            "length_bucket": self.blinded_stem.length_bucket,
            "owner_scope": self.owner_scope,
            "timestamp": self.timestamp,
            "title_language": self.title_language,
            "token_features": sorted(self.blinded_stem.token_features),
            "verified_features": {k: self.verified_features[k] for k in sorted(self.verified_features.keys())},
        }
        if self.evidence_masks:
            data["evidence_masks"] = {k: bool(self.evidence_masks[k]) for k in sorted(self.evidence_masks.keys())}
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclasses.dataclass
class QueuedTelemetryItem:
    """Buffered telemetry item scheduled for Poisson timing jitter dispatch."""
    scheduled_dispatch_time: float
    record: AnonymizedTelemetryRecord
    retry_count: int = 0


@dataclasses.dataclass(frozen=True)
class EncryptedTelemetryBundle:
    """Hybrid Post-Quantum (X25519 + ML-KEM-768) + ChaCha20-Poly1305 AEAD encrypted payload."""
    ephemeral_pubkey_hex: str  # 32B X25519 ephem_pub || 1088B ML-KEM-768 ciphertext (1120B hybrid KEM token)
    nonce_hex: str
    ciphertext_hex: str
    aad: str
    seq: int
    cipher_suite: str = "X25519-MLKEM768-ChaCha20Poly1305-SHA3"

    @property
    def is_post_quantum(self) -> bool:
        return len(self.ephemeral_pubkey_hex) == 2240  # 1120 bytes = 32B X25519 + 1088B ML-KEM-768

    @property
    def classical_ephemeral_pubkey_hex(self) -> str:
        return self.ephemeral_pubkey_hex[:64]

    @property
    def pq_kem_ciphertext_hex(self) -> str:
        return self.ephemeral_pubkey_hex[64:] if len(self.ephemeral_pubkey_hex) > 64 else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ephemeral_pubkey_hex": self.ephemeral_pubkey_hex,
            "nonce_hex": self.nonce_hex,
            "ciphertext_hex": self.ciphertext_hex,
            "aad": self.aad,
            "seq": self.seq,
            "cipher_suite": self.cipher_suite,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EncryptedTelemetryBundle:
        return cls(
            ephemeral_pubkey_hex=str(data["ephemeral_pubkey_hex"]),
            nonce_hex=str(data["nonce_hex"]),
            ciphertext_hex=str(data["ciphertext_hex"]),
            aad=str(data["aad"]),
            seq=int(data["seq"]),
            cipher_suite=str(data.get("cipher_suite", "X25519-MLKEM768-ChaCha20Poly1305-SHA3")),
        )


@dataclasses.dataclass(frozen=True)
class PeerAttestation:
    """Peer attestation containing distinct-ASN information and hybrid Ed25519 + WOTS+-SHAKE256 PQ signatures."""
    peer_id: str
    asn: int
    peer_pubkey_hex: str
    timestamp: int
    signature_hex: str
    pq_pubkey_hex: str = ""
    pq_signature_hex: str = ""
    ip_address: Optional[str] = None

    def canonical_bytes(self, target_hash: str) -> bytes:
        body: Dict[str, Any] = {
            "asn": int(self.asn),
            "peer_id": self.peer_id,
            "target_hash": target_hash.lower(),
            "timestamp": int(self.timestamp),
        }
        if self.ip_address is not None and str(self.ip_address).strip() != "":
            body["ip_address"] = str(self.ip_address).strip()
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclasses.dataclass(frozen=True)
class ThresholdDHTManifest:
    """BEP-44 / BEP-46 update manifest with Hybrid Ed25519 + WOTS+-SHAKE256 PQ threshold multi-signatures."""
    seq: int
    version: str
    infohash: str
    bundle_sha256: str
    bundle_format: str
    min_asn_quorum: int
    timestamp: int
    signatures: Union[Dict[str, str], Sequence[Tuple[str, str]]]  # pubkey_hex -> ed25519_signature_hex
    pq_signatures: Union[Dict[str, str], Sequence[Tuple[str, str]]] = dataclasses.field(default_factory=dict)  # pq_pubkey_hex -> wots_shake256_sig_hex
    signer_pubkeys: Optional[List[str]] = None
    signer_attestations: Optional[List[PeerAttestation]] = None

    def canonical_payload(self) -> bytes:
        body = {
            "bundle_format": self.bundle_format.lower(),
            "bundle_sha256": self.bundle_sha256.lower(),
            "infohash": self.infohash.lower(),
            "min_asn_quorum": int(self.min_asn_quorum),
            "seq": int(self.seq),
            "timestamp": int(self.timestamp),
            "version": str(self.version),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _normalize_pubkey_id(raw_pk: str) -> str:
        cleaned = str(raw_pk).strip().lower()
        if cleaned.startswith("0x"):
            cleaned = cleaned[2:]
        return re.split(r"[:#/\s]", cleaned)[0]

    def iter_ed25519_signatures(self) -> List[Tuple[str, str]]:
        if isinstance(self.signatures, dict):
            return [(str(k), str(v)) for k, v in self.signatures.items()]
        return [(str(k), str(v)) for k, v in self.signatures]

    def iter_pq_signatures(self) -> List[Tuple[str, str]]:
        if isinstance(self.pq_signatures, dict):
            return [(str(k), str(v)) for k, v in self.pq_signatures.items()]
        return [(str(k), str(v)) for k, v in self.pq_signatures]

    def check_signer_uniqueness(self) -> Tuple[bool, str]:
        """Verify that no public key or signature is reused more than once in the manifest."""
        seen_ed_pks: Set[str] = set()
        seen_ed_sigs: Set[str] = set()
        for raw_pk, raw_sig in self.iter_ed25519_signatures():
            norm_pk = self._normalize_pubkey_id(raw_pk)
            norm_sig = str(raw_sig).strip().lower()
            if norm_pk in seen_ed_pks:
                return False, f"duplicate_manifest_signer_pubkey:{norm_pk[:16]}"
            seen_ed_pks.add(norm_pk)
            if norm_sig:
                if norm_sig in seen_ed_sigs:
                    return False, "duplicate_manifest_signature_reused"
                seen_ed_sigs.add(norm_sig)

        seen_pq_pks: Set[str] = set()
        seen_pq_sigs: Set[str] = set()
        for raw_pq_pk, raw_pq_sig in self.iter_pq_signatures():
            norm_pq_pk = self._normalize_pubkey_id(raw_pq_pk)
            norm_pq_sig = str(raw_pq_sig).strip().lower()
            if norm_pq_pk in seen_pq_pks:
                return False, f"duplicate_manifest_pq_signer_pubkey:{norm_pq_pk[:16]}"
            seen_pq_pks.add(norm_pq_pk)
            if norm_pq_sig:
                if norm_pq_sig in seen_pq_sigs:
                    return False, "duplicate_manifest_pq_signature_reused"
                seen_pq_sigs.add(norm_pq_sig)

        if self.signer_pubkeys is not None:
            seen_list_pks: Set[str] = set()
            for raw_pk in self.signer_pubkeys:
                norm_pk = self._normalize_pubkey_id(raw_pk)
                if norm_pk in seen_list_pks:
                    return False, f"duplicate_manifest_signer_pubkey_in_list:{norm_pk[:16]}"
                seen_list_pks.add(norm_pk)

        return True, "unique_signers"


@dataclasses.dataclass(frozen=True)
class TelemetrySubmission:
    """Encrypted telemetry bundle submission from a DHT peer with ASN and network binding metadata."""
    encrypted_bundle: EncryptedTelemetryBundle
    asn: int
    peer_id: str = ""
    peer_pubkey_hex: str = ""
    ip_address: Optional[str] = None
    attestation: Optional[PeerAttestation] = None
    canonical_title: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ZeroCueCatalogDelta:
    """Verified ZeroCueCatalog or release-group prior delta emitted after multi-ASN k-anonymity consensus."""
    blinded_hash: str
    canonical_title: Optional[str]
    category: str
    adult: bool
    anime: bool
    lang: str
    junk: bool
    owner_scope: str
    english_subs: bool
    title_language: str
    stream_signature: Dict[str, Any]
    distinct_asn_count: int
    verified_asns: List[int]
    observation_count: int
    agreement_ratio: float
    release_group_prior_delta: Dict[str, float]
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adult": self.adult,
            "agreement_ratio": round(self.agreement_ratio, 6),
            "anime": self.anime,
            "blinded_hash": self.blinded_hash,
            "canonical_title": self.canonical_title,
            "category": self.category,
            "distinct_asn_count": self.distinct_asn_count,
            "english_subs": self.english_subs,
            "junk": self.junk,
            "lang": self.lang,
            "observation_count": self.observation_count,
            "owner_scope": self.owner_scope,
            "release_group_prior_delta": dict(self.release_group_prior_delta),
            "stream_signature": dict(self.stream_signature),
            "timestamp": self.timestamp,
            "title_language": self.title_language,
            "verified_asns": sorted(self.verified_asns),
        }

    def canonical_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclasses.dataclass
class CatalogConsensusResult:
    """Result of multi-ASN k-anonymity catalog observation aggregation."""
    consensus_reached: bool
    reason: str
    blinded_hash: str
    delta: Optional[ZeroCueCatalogDelta] = None
    distinct_asns: int = 0
    total_valid_observations: int = 0
    quarantined_byzantine_count: int = 0
    quarantined_reasons: List[str] = dataclasses.field(default_factory=list)
    agreement_ratio: float = 0.0
    manifest: Optional[ThresholdDHTManifest] = None
    signed_bundle_bytes: Optional[bytes] = None

    @property
    def aggregated_deltas(self) -> Dict[str, float]:
        """Return verified release-group prior deltas from the consensus catalog delta (empty dict if no consensus)."""
        if self.delta is None:
            return {}
        return dict(self.delta.release_group_prior_delta)


@dataclasses.dataclass(frozen=True)
class ContinuousDeltaSubmission:
    """Continuous calibration temperature or adapter weight delta submission from a DHT peer."""
    peer_id: str
    asn: int
    peer_pubkey_hex: str
    deltas: Dict[str, Union[float, List[float]]]
    ip_address: Optional[str] = None
    encrypted_bundle: Optional[EncryptedTelemetryBundle] = None
    attestation: Optional[PeerAttestation] = None
    timestamp: int = 0


@dataclasses.dataclass
class ContinuousAggregationResult:
    """Result of per-ASN coordinate-wise trimmed mean and median continuous delta aggregation."""
    success: bool
    reason: str
    distinct_asn_count: int
    verified_asns: List[int]
    total_submissions: int
    rejected_submissions: int
    trimmed_mean_deltas: Dict[str, Union[float, List[float]]]
    median_deltas: Dict[str, Union[float, List[float]]]
    trim_fraction: float
    manifest: Optional[ThresholdDHTManifest] = None
    signed_bundle_bytes: Optional[bytes] = None


@dataclasses.dataclass(frozen=True)
class CanaryItem:
    """Benchmark reference sample for the mandatory 500-item golden canary regression gate."""
    item_id: str
    title: str
    ground_truth: Dict[str, Any]
    is_malware_trap: bool = False
    is_adult_boundary: bool = False


@dataclasses.dataclass
class CanaryRegressionReport:
    """Detailed results from the 500-item golden canary regression gate."""
    passed: bool
    reason: str
    total_items: int
    junk_balanced_accuracy: float
    junk_false_negatives: int
    adult_accuracy: float
    category_accuracy: float
    owner_scope_accuracy: float
    regressions: List[Dict[str, Any]]
    excluded_rejection_accuracy: float = 1.0  # owner_scope=="excluded" recall over EXCLUDED_SCOPE_CATEGORIES items
    stratum_worst_group_accuracy: float = 1.0  # min over (category, packaging stratum) of category+scope accuracy
    worst_stratum: str = ""
    stratum_accuracy: Dict[str, float] = dataclasses.field(default_factory=dict)
    nonconformity_quantiles: Dict[str, float] = dataclasses.field(default_factory=dict)


# =====================================================================
# 3. Privacy & Differential Privacy Helpers
# =====================================================================

_CSPRNG = secrets.SystemRandom()


def pure_laplace_noise(scale: float) -> float:
    """CSPRNG-backed exact Laplace noise generator via inverse CDF: F^-1(u) = -b * sgn(u - 0.5) * ln(1 - 2|u - 0.5|)."""
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    u = _CSPRNG.random() - 0.5
    sgn = 1.0 if u >= 0 else -1.0
    return -scale * sgn * math.log(max(1.0 - 2.0 * abs(u), 1e-15))


class TelemetryAnonymizer:
    """Anonymizes telemetry records, strips PII/passkeys, blinds titles, and applies DP noise."""

    RE_PASSKEY = re.compile(
        r'((?:[?&]|%26)(?:passkey|authkey|torrent_pass|auth|token|uk|key|api_key|apikey|pk)(?:=|%3[dD]))[a-zA-Z0-9_.\-%]+',
        re.IGNORECASE,
    )
    RE_PASSKEY_INLINE = re.compile(
        r'(?:passkey|authkey|torrent_pass|api_key|apikey)\s*(?:[:=]|%3[dD])\s*[a-zA-Z0-9_.\-%]+',
        re.IGNORECASE,
    )
    RE_PATH_PASSKEY = re.compile(
        r'(https?://[^/\s]+/)[a-zA-Z0-9_.\-]{8,}(/announce\b)',
        re.IGNORECASE,
    )
    RE_EMAIL = re.compile(
        r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    )
    RE_IPV4 = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )
    RE_IPV6 = re.compile(
        r'(?:\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b|'
        r'\b[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*::[0-9a-fA-F]{0,4}(?::[0-9a-fA-F]{1,4})*|'
        r'::(?:ffff:(?:\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*))',
        re.IGNORECASE,
    )
    RE_UNC_PATH = re.compile(r'\\\\[^\s\\]+(?:\\[^\s\\]+)+')
    RE_UNIX_PATH = re.compile(r"/(?:Users|home|mnt|media|var|srv|opt|private)/[^\s\"'<>]+")
    RE_WIN_PATH = re.compile(r'[A-Za-z]:[/\\]Users[/\\][^/\\\r\n]+(?:[/\\][^/\\\r\n]+)*')

    TOKEN_SYNTAX_CATEGORIES = {
        "res": {"2160p", "4k", "1080p", "720p", "480p", "uhd"},
        "codec": {"hevc", "x265", "h265", "x264", "h264", "av1", "xvid"},
        "audio": {"truehd", "atmos", "dts", "dtshd", "flac", "aac", "eac3", "ac3", "mp3"},
        "channels": {"7.1", "5.1", "2.0"},
        "source": {"bluray", "remux", "webrip", "webdl", "hdtv", "bdrip"},
        "tags": {"hdr", "hdr10", "dv", "multisubs", "dual-audio", "proper", "repack"},
    }

    def __init__(
        self,
        blinding_salt: Optional[bytes] = None,
        dp_epsilon: float = 1.0,
        poisson_mean_seconds: float = 1800.0,
        *,
        max_releases: int = 1024,
    ) -> None:
        self.blinding_salt = blinding_salt or os.urandom(32)
        self.dp_epsilon = max(float(dp_epsilon), 0.01)
        self.poisson_mean_seconds = max(float(poisson_mean_seconds), 1.0)
        self.max_releases = max(int(max_releases), 1)
        self._release_count: int = 0
        self._dp_memo: Dict[str, Tuple[float, Optional[float]]] = {}

    def strip_pii(self, text: str) -> str:
        """Strip private tracker passkeys, emails, IPv4/IPv6 addresses, UNC paths, and local filesystem paths."""
        if not text:
            return ""
        s = str(text)
        s = self.RE_PASSKEY.sub(r'\1[REDACTED_PASSKEY]', s)
        s = self.RE_PASSKEY_INLINE.sub('passkey=[REDACTED_PASSKEY]', s)
        s = self.RE_PATH_PASSKEY.sub(r'\1[REDACTED_PASSKEY]\2', s)
        s = self.RE_EMAIL.sub('[EMAIL_REDACTED]', s)
        s = self.RE_UNC_PATH.sub(r'\\\\[UNC_REDACTED]', s)
        s = self.RE_WIN_PATH.sub('C:/Users/[USER_REDACTED]', s)
        s = self.RE_UNIX_PATH.sub('/home/[USER_REDACTED]', s)
        s = self.RE_IPV6.sub('[IPV6_REDACTED]', s)
        s = self.RE_IPV4.sub('[IP_REDACTED]', s)
        # Secondary token-level IPv4/IPv6 literal check via ipaddress
        out_tokens = []
        for tok in s.split(" "):
            stripped_tok = tok.strip("[](),;'\"")
            if ":" in stripped_tok or "." in stripped_tok:
                try:
                    addr = ipaddress.ip_address(stripped_tok)
                    tok = "[IPV6_REDACTED]" if isinstance(addr, ipaddress.IPv6Address) else "[IP_REDACTED]"
                except ValueError:
                    pass
            out_tokens.append(tok)
        return " ".join(out_tokens)

    def blind_title_stem(self, raw_title: str) -> BlindedTitleStem:
        """Blinds the title string into an HMAC hash and coarse syntactic tokens without leaking name."""
        clean = self.strip_pii(raw_title).lower()
        # Remove release group suffixes e.g. -OPEN, -OPEN, [OPEN]
        clean = re.sub(r'-[a-zA-Z0-9]+(\[[^\]]+\])?$', '', clean)
        raw_tokens = re.findall(r'[a-zA-Z0-9]+', clean)

        matched_tokens: Set[str] = set()
        for t in raw_tokens:
            for cat_tokens in self.TOKEN_SYNTAX_CATEGORIES.values():
                if t in cat_tokens:
                    matched_tokens.add(t)

        token_count = len(raw_tokens)
        if token_count <= 4:
            length_bucket = "short"
        elif token_count <= 9:
            length_bucket = "medium"
        else:
            length_bucket = "long"

        # Compute Post-Quantum keyed HMAC-SHA256 + SHAKE-256 blinded hash (256-bit state, 128-bit Grover PQ security)
        normalized_str = " ".join(raw_tokens) + "|" + str(len(clean))
        h_sha2 = hmac.new(self.blinding_salt, normalized_str.encode("utf-8"), hashlib.sha256).digest()
        h_shake = hashlib.shake_256(b"armiger-pq-blind-v1:" + self.blinding_salt + h_sha2).digest(32)
        blinded_hash = h_shake.hex()

        return BlindedTitleStem(
            blinded_hash=blinded_hash,
            token_features=sorted(list(matched_tokens)),
            length_bucket=length_bucket,
        )

    # Bounded logarithmic local DP domain & global sensitivity constants:
    #   - Neighboring-input model: any two valid media observations (s, d), (s', d') in the
    #     bounded domain [MIN_SIZE_MB, MAX_SIZE_MB] x [MIN_DURATION_MINS, MAX_DURATION_MINS]
    #     are neighboring inputs under full replacement local differential privacy.
    #   - Size domain: [16.0, 65536.0] MB (16 MiB .. 64 GiB) -> log2 span = log2(65536 / 16) = 12.0 bits.
    #   - Duration domain: [5.0, 1280.0] mins (5 min .. 21.33 h) -> log2 span = log2(1280 / 5) = 8.0 bits.
    MIN_SIZE_MB: float = 16.0
    MAX_SIZE_MB: float = 65536.0
    SIZE_QUANT_STEP_MB: float = 16.0
    SIZE_LOG2_SENSITIVITY: float = 12.0  # math.log2(MAX_SIZE_MB / MIN_SIZE_MB)

    MIN_DURATION_MINS: float = 5.0
    MAX_DURATION_MINS: float = 1280.0
    DURATION_QUANT_STEP_MINS: float = 5.0
    DURATION_LOG2_SENSITIVITY: float = 8.0  # math.log2(MAX_DURATION_MINS / MIN_DURATION_MINS)

    def compute_dp_parameters(
        self,
        size_bytes: int,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute bounded-domain clipped values, log2 sensitivities, split privacy budgets, and Laplace scales."""
        if (
            not isinstance(size_bytes, bool)
            and isinstance(size_bytes, (int, float))
            and math.isfinite(size_bytes)
            and size_bytes > 0
        ):
            raw_size_mb = float(size_bytes) / (1024.0 * 1024.0)
        else:
            raw_size_mb = 0.0
        clipped_size_mb = min(max(raw_size_mb, self.MIN_SIZE_MB), self.MAX_SIZE_MB)
        quantized_mb = min(
            max(round(clipped_size_mb / self.SIZE_QUANT_STEP_MB) * self.SIZE_QUANT_STEP_MB, self.MIN_SIZE_MB),
            self.MAX_SIZE_MB,
        )
        log2_size = math.log2(quantized_mb)

        has_duration = (
            duration_seconds is not None
            and not isinstance(duration_seconds, bool)
            and isinstance(duration_seconds, (int, float))
            and math.isfinite(duration_seconds)
            and duration_seconds >= 0.0
        )
        clipped_dur_mins: Optional[float] = None
        quantized_mins: Optional[float] = None
        log2_dur: Optional[float] = None
        if has_duration:
            raw_dur_mins = float(duration_seconds) / 60.0  # type: ignore[arg-type]
            clipped_dur_mins = min(max(raw_dur_mins, self.MIN_DURATION_MINS), self.MAX_DURATION_MINS)
            quantized_mins = min(
                max(
                    round(clipped_dur_mins / self.DURATION_QUANT_STEP_MINS) * self.DURATION_QUANT_STEP_MINS,
                    self.MIN_DURATION_MINS,
                ),
                self.MAX_DURATION_MINS,
            )
            log2_dur = math.log2(quantized_mins)

        # Sequential composition across released fields: split total budget dp_epsilon across released fields
        released_fields = 2 if has_duration else 1
        epsilon_per_field = self.dp_epsilon / float(released_fields)
        size_laplace_scale = self.SIZE_LOG2_SENSITIVITY / epsilon_per_field
        dur_laplace_scale = (self.DURATION_LOG2_SENSITIVITY / epsilon_per_field) if has_duration else None

        return {
            "clipped_size_mb": clipped_size_mb,
            "quantized_size_mb": quantized_mb,
            "log2_size": log2_size,
            "size_log2_sensitivity": self.SIZE_LOG2_SENSITIVITY,
            "epsilon_size": epsilon_per_field,
            "size_laplace_scale": size_laplace_scale,
            "clipped_duration_mins": clipped_dur_mins,
            "quantized_duration_mins": quantized_mins,
            "log2_duration": log2_dur,
            "duration_log2_sensitivity": self.DURATION_LOG2_SENSITIVITY if has_duration else None,
            "epsilon_duration": epsilon_per_field if has_duration else None,
            "duration_laplace_scale": dur_laplace_scale,
            "released_fields": released_fields,
        }

    def apply_differential_privacy(
        self,
        size_bytes: int,
        duration_seconds: Optional[float] = None,
        *,
        contribution_id: Optional[str] = None,
    ) -> Tuple[float, Optional[float]]:
        """Apply bounded logarithmic local differential privacy with sequential budget splitting.

        Clips size to `[16.0, 65536.0]` MB (`Delta_log2 = 12.0`) and duration to `[5.0, 1280.0]` mins
        (`Delta_log2 = 8.0`), splits `dp_epsilon` across all released fields (`eps / 2` when both
        size and duration are released), perturbs in `log2` space with `Laplace(Delta_log2 / eps_field)`,
        and applies deterministic post-processing back to bounded physical units.

        Distinct releases with identical numeric values always draw fresh Laplace noise. Cached
        noise is reused ONLY when an explicit non-empty `contribution_id` is supplied to retransmit
        the exact same contribution.
        """
        cid = str(contribution_id).strip() if contribution_id is not None else ""
        if cid and cid in self._dp_memo:
            return self._dp_memo[cid]

        if self._release_count >= self.max_releases:
            raise AnonymityViolationError(
                f"dp_privacy_budget_exhausted:{self._release_count}>={self.max_releases}"
            )

        params = self.compute_dp_parameters(size_bytes, duration_seconds)
        quantized_mins: Optional[float] = params["quantized_duration_mins"]

        log2_min_size = math.log2(self.MIN_SIZE_MB)
        log2_max_size = math.log2(self.MAX_SIZE_MB)
        noisy_log2_size = float(params["log2_size"]) + pure_laplace_noise(scale=float(params["size_laplace_scale"]))
        clamped_log2_size = min(max(noisy_log2_size, log2_min_size), log2_max_size)
        dp_size = round(
            min(
                max(
                    round((2.0 ** clamped_log2_size) / self.SIZE_QUANT_STEP_MB) * self.SIZE_QUANT_STEP_MB,
                    self.MIN_SIZE_MB,
                ),
                self.MAX_SIZE_MB,
            ),
            4,
        )

        dp_duration: Optional[float] = None
        if quantized_mins is not None and params["log2_duration"] is not None and params["duration_laplace_scale"] is not None:
            log2_min_dur = math.log2(self.MIN_DURATION_MINS)
            log2_max_dur = math.log2(self.MAX_DURATION_MINS)
            noisy_log2_dur = float(params["log2_duration"]) + pure_laplace_noise(
                scale=float(params["duration_laplace_scale"])
            )
            clamped_log2_dur = min(max(noisy_log2_dur, log2_min_dur), log2_max_dur)
            dp_duration = round(
                min(
                    max(
                        round((2.0 ** clamped_log2_dur) / self.DURATION_QUANT_STEP_MINS) * self.DURATION_QUANT_STEP_MINS,
                        self.MIN_DURATION_MINS,
                    ),
                    self.MAX_DURATION_MINS,
                ),
                4,
            )

        self._release_count += 1
        if cid:
            if len(self._dp_memo) >= self.max_releases:
                oldest_cid = next(iter(self._dp_memo))
                self._dp_memo.pop(oldest_cid, None)
            self._dp_memo[cid] = (dp_size, dp_duration)
        return dp_size, dp_duration

    def calculate_poisson_jitter(self) -> float:
        """Sample an arrival delay from a CSPRNG Poisson process (exponential inter-arrival distribution)."""
        delay = _CSPRNG.expovariate(1.0 / self.poisson_mean_seconds)
        if self.poisson_mean_seconds >= 5.0:
            return max(2.0, delay)
        return delay

    ALLOWED_VIDEO_CODECS = {
        "hevc", "h265", "x265", "h264", "x264", "av1", "vp9", "mpeg2video", "vc1", "prores", "xvid", "none",
    }
    ALLOWED_AUDIO_CODECS = {
        "truehd", "eac3", "ac3", "dts", "dtshd", "flac", "aac", "mp3", "m4b", "opus", "vorbis", "alac", "pcm", "wav", "none",
    }
    ALLOWED_CATALOG_TYPES = {
        "series_catalog", "movie_catalog", "audio_catalog", "isbn", "anime_catalog", "none",
    }

    @classmethod
    def _sanitize_allowlisted_token(cls, raw_val: Any, allowed: Set[str]) -> str:
        if not raw_val:
            return "none"
        tokens = re.findall(r"[a-zA-Z0-9_]+", str(raw_val).lower())
        for tok in tokens:
            if tok in allowed:
                return tok
        return "none"

    def anonymize(self, observation: VerifiedObservation) -> AnonymizedTelemetryRecord:
        """Transform a verified physical observation into an anonymous telemetry record."""
        blinded = self.blind_title_stem(observation.raw_title)
        dur = observation.stream_info.get("duration_seconds")
        dp_size, dp_dur = self.apply_differential_privacy(observation.size_bytes, dur)

        vcodec = self._sanitize_allowlisted_token(observation.stream_info.get("video_codec"), self.ALLOWED_VIDEO_CODECS)
        acodec = self._sanitize_allowlisted_token(observation.stream_info.get("audio_codec"), self.ALLOWED_AUDIO_CODECS)
        cat_type = self._sanitize_allowlisted_token(observation.catalog_id_type, self.ALLOWED_CATALOG_TYPES)
        raw_ch = observation.stream_info.get("audio_channels", 0)
        achannels = float(raw_ch) if isinstance(raw_ch, (int, float)) and math.isfinite(raw_ch) and 0 <= raw_ch <= 32 else 0
        features = {
            "has_video": vcodec != "none",
            "has_audio": acodec != "none",
            "video_codec": vcodec,
            "audio_codec": acodec,
            "audio_channels": achannels,
            "catalog_type": cat_type,
            "episode_marker": bool(EPISODE_MARKER_RE.search(str(observation.raw_title or ""))),
        }

        return AnonymizedTelemetryRecord(
            blinded_stem=blinded,
            category=observation.category,
            adult=observation.adult,
            anime=observation.anime,
            lang=observation.lang,
            junk=observation.junk,
            owner_scope=observation.owner_scope,
            english_subs=observation.english_subs,
            title_language=observation.title_language,
            dp_size_mb=dp_size,
            dp_duration_mins=dp_dur,
            verified_features=features,
            timestamp=((int(getattr(observation, "verified_at", 0) or time.time())) // 3600) * 3600,
            evidence_masks=dict(observation.evidence_masks),
        )


# =====================================================================
# 3B. Strict Opt-In-By-Default & Cryptographic/DP AnonymityGuard
# =====================================================================

class AnonymityViolationError(ValueError):
    """Raised when any wire payload, catalog record, weight delta, or timestamp violates anonymity invariants."""


class InvalidOptInSharingError(TypeError, ValueError):
    """Raised when `opt_in_sharing` is passed a non-boolean, non-None value."""


class _DerivedCatalogKey(bytes):
    """Marker bytes subclass representing an already-derived 32-byte catalog BLAKE2b key."""


def resolve_opt_in_sharing(opt_in_sharing: Optional[bool] = None) -> bool:
    """Resolve whether outbound anonymous DHT sharing is explicitly enabled.

    Strictly defaults to False (`ARMIGER_DHT_SHARE_OPT_IN="0"`).
    Accepts ONLY actual `bool` when `opt_in_sharing is not None`; rejects non-boolean non-None
    types (such as `"0"`, `"false"`, `"no"`, integers, or arbitrary objects) instead of coercing
    via Python truthiness.
    """
    if isinstance(opt_in_sharing, bool):
        return opt_in_sharing
    if opt_in_sharing is not None:
        raise InvalidOptInSharingError(
            f"opt_in_sharing must be bool or None, got {type(opt_in_sharing).__name__}: {opt_in_sharing!r}"
        )
    env_val = os.environ.get("ARMIGER_DHT_SHARE_OPT_IN", "0").strip().lower()
    return env_val in ("1", "true", "yes")


def _bep44_canonical_sign_buffer(v: bytes, seq: int, salt: bytes = b"") -> bytes:
    """Canonical BEP-44 signing buffer: [4:salt<len>:<salt>]3:seqi<seq>e1:v<len>:<v>."""
    out = bytearray()
    if salt:
        out.extend(b"4:salt" + f"{len(salt)}:".encode("ascii") + salt)
    out.extend(f"3:seqi{int(seq)}e".encode("ascii"))
    out.extend(f"1:v{len(v)}:".encode("ascii") + v)
    return bytes(out)


class AnonymityGuard:
    """Unconditional Anonymity, Differential-Privacy & Deep Wire-Payload Canary Firewall.

    Enforces 5 mathematical and cryptographic guarantees whenever `opt_in_sharing=True`:
      1. Ephemeral Unlinkable Key Rotation (`ephemeral_identity=True` by default):
         Rotates ephemeral in-memory Ed25519 signing keys per contribution session or epoch with zero host metadata.
      2. Coarse Epoch Timestamp Bucketing:
         Quantizes all wire timestamps (`ts_utc`, `updated_utc`, `epoch`, `t`) to coarse UTC hour boundaries
         (`floor(ts / 3600) * 3600`).
      3. Clipped + Differentially Private Weight & Calibration Aggregation:
         Enforces L2-norm clipping (`||Delta||_2 <= clip_norm`), calibrated Gaussian DP noise, and minimum
         sample-count k-anonymity (`min_k_samples >= 25`) before any weight delta or reliability bin can be exported.
      4. One-Way Domain-Separated Catalog Hashing Only:
         Enforces that catalog keys (`h`) are strictly 16-hex-char truncated
         `BLAKE2b-64(domain="armiger-catalog-v2" || normalized_key)` hashes, stripping/rejecting any raw titles,
         filenames, release names, IMDb/TMDB/TVDB IDs, infohashes, URLs, passkeys, paths, or hostnames.
      5. Deep Wire-Payload PII / Canary Firewall (`AnonymityViolationError`):
         Inspects every serialized JSON/bencode wire payload (`v`) before signing and before UDP transmission,
         rejecting any IPv4/IPv6/CIDR, hostname, filesystem path/extension, URL/magnet, passkey/secret,
         40-hex infohash, unbucketed timestamp, `n < min_k_samples`, or unexpected schema key outside the
         strict mathematical whitelist.
    """

    DOMAIN_CATALOG_V2: bytes = b"armiger-catalog-v2"
    _DEFAULT_PROCESS_CATALOG_SECRET: bytes = secrets.token_bytes(32)
    DEFAULT_MIN_K_SAMPLES: int = 25
    DEFAULT_BUCKET_SECONDS: int = 3600
    DEFAULT_CLIP_NORM: float = 1.0
    DEFAULT_DP_SIGMA: float = 0.05

    ALLOWED_WIRE_KEYS: Set[str] = {
        "v",
        "epoch",
        "seq",
        "model_tier",
        "delta_w_q8",
        "delta_b_q8",
        "scale",
        "reliability_bins",
        "n_bucket",
        "records",
        "ts_utc",
        "updated_utc",
    }
    ALLOWED_RECORD_KEYS: Tuple[str, ...] = ("h", "c", "r", "s", "a", "v", "n", "t")
    ALLOWED_RECORD_KEY_SET: Set[str] = set(ALLOWED_RECORD_KEYS)
    ALLOWED_MODEL_TIERS: Set[Union[str, int]] = {"base", "ultra", "int8", "fp32", 0, 1, 2, 3}

    RE_HEX16 = re.compile(r"^[0-9a-f]{16}$")
    RE_INFOHASH_40 = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")
    RE_HEX_24_PLUS = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{24,128}(?![0-9a-fA-F])")
    RE_IPV4_OR_CIDR = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:/\d{1,2})?\b"
    )
    RE_CIDR_BLOCK = re.compile(
        r"\b(?:\d{1,3}\.){1,3}\d{1,3}/\d{1,2}\b|[0-9a-fA-F:]{2,}/(?:[1-9]\d?|1[01]\d|12[0-8])\b"
    )
    RE_IPV6 = TelemetryAnonymizer.RE_IPV6
    RE_HOSTNAME = re.compile(
        r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
        r"(?:local|lan|internal|home|arpa|corp|tailnet|ts[\.]net|invalid|test|example|"
        r"com|net|org|io|dev|me|tv|cc|ru|de|uk|fr|nl|se|ch|ca|au|jp|cn|br|in|xyz|info|biz|top|club|online|site|tech)\b",
        re.IGNORECASE,
    )
    RE_FS_PATH = re.compile(
        r"(?:/(?:Users|home|srv|var|mnt|media|opt|private|tmp|etc|root|Volumes|Library|data|downloads|torrents|share|nas)\b|"
        r"[A-Za-z]:[/\\]|\\\\[^\s\\]+|"
        r"\.(?:mkv|mp4|avi|mov|wmv|flv|webm|m4v|torrent|nzb|exe|scr|bat|ps1|sh|srt|ass|sub|nfo|iso|rar|zip|7z|epub|mobi|pdf|flac|mp3|m4b)\b)",
        re.IGNORECASE,
    )
    RE_URL_OR_MAGNET = re.compile(
        r"(?:(?:https?|udp|ws|wss|ftp|dht|bep44)://|magnet:\?|xt=urn:btih:|/announce\b|announce\?)",
        re.IGNORECASE,
    )
    RE_SECRET_OR_PASSKEY = re.compile(
        r"(?:passkey|authkey|torrent_pass|api_key|apikey|secret|token|password|passwd|"
        r"bearer\s+[a-zA-Z0-9_.\-]+|ghp_[a-zA-Z0-9]+|glpat-[a-zA-Z0-9_\-]+|sk-[a-zA-Z0-9_\-]+)",
        re.IGNORECASE,
    )
    RE_EXTERNAL_ID = re.compile(
        r"(?:\btt\d{6,10}\b|\b(?:imdb|tmdb|tvdb|mal|anilist|musicbrainz|isbn)[_:\-\s=]*[a-zA-Z0-9\-]+\b)",
        re.IGNORECASE,
    )
    RE_RELEASE_TITLE = re.compile(
        r"(?:\b(?:1080p|2160p|720p|480p|bluray|web-dl|webrip|remux|x264|x265|hevc|h264|h265|truehd|atmos|ddp5|dts-hd)\b|"
        r"\bS\d{1,2}E\d{1,2}\b)",
        re.IGNORECASE,
    )
    RE_EMAIL = TelemetryAnonymizer.RE_EMAIL

    def __init__(
        self,
        *,
        ephemeral_identity: bool = True,
        min_k_samples: int = DEFAULT_MIN_K_SAMPLES,
        bucket_seconds: int = DEFAULT_BUCKET_SECONDS,
        clip_norm: float = DEFAULT_CLIP_NORM,
        dp_sigma: float = DEFAULT_DP_SIGMA,
        catalog_secret: Optional[Union[bytes, str]] = None,
    ) -> None:
        self.ephemeral_identity: bool = bool(ephemeral_identity)
        self.min_k_samples: int = max(int(min_k_samples), self.DEFAULT_MIN_K_SAMPLES)
        self.bucket_seconds: int = max(int(bucket_seconds), 3600)
        self.clip_norm: float = float(clip_norm) if clip_norm > 0 else self.DEFAULT_CLIP_NORM
        self.dp_sigma: float = max(float(dp_sigma), 0.0)
        self.catalog_secret: bytes = self._derive_catalog_hmac_key(catalog_secret)
        self.rotation_count: int = 0
        self._current_epoch_tag: Optional[Union[str, int]] = None
        self._ephem_priv: ed25519.Ed25519PrivateKey = ed25519.Ed25519PrivateKey.generate()
        self.rotation_count = 1
        self.blocked_violations_count: int = 0

    @classmethod
    def _derive_catalog_hmac_key(cls, secret: Optional[Union[bytes, str]] = None) -> _DerivedCatalogKey:
        """Derive a 32-byte keyed BLAKE2b secret from an explicit secret, `ARMIGER_CATALOG_HMAC_KEY`, or process secret."""
        if isinstance(secret, _DerivedCatalogKey):
            return secret
        raw: Optional[Union[bytes, str]] = secret
        if raw is None:
            env_secret = os.environ.get("ARMIGER_CATALOG_HMAC_KEY", "").strip()
            raw = env_secret if env_secret else cls._DEFAULT_PROCESS_CATALOG_SECRET
        if isinstance(raw, _DerivedCatalogKey):
            return raw
        raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if not raw_bytes:
            raise AnonymityViolationError("empty_catalog_hmac_secret")
        derived = hashlib.blake2b(
            cls.DOMAIN_CATALOG_V2 + b":kdf:" + raw_bytes,
            digest_size=32,
            person=b"armiger-cat-kdf",
        ).digest()
        return _DerivedCatalogKey(derived)

    @classmethod
    def rotate_default_catalog_secret(cls, new_secret: Optional[Union[bytes, str]] = None) -> None:
        """Rotate the default process-level catalog HMAC key for epoch unlinkability."""
        if new_secret is None:
            cls._DEFAULT_PROCESS_CATALOG_SECRET = secrets.token_bytes(32)
        else:
            cls._DEFAULT_PROCESS_CATALOG_SECRET = cls._derive_catalog_hmac_key(new_secret)

    def rotate_catalog_secret(self, new_secret: Optional[Union[bytes, str]] = None) -> None:
        """Rotate this guard's keyed catalog HMAC secret."""
        if new_secret is None:
            self.catalog_secret = self._derive_catalog_hmac_key(secrets.token_bytes(32))
        else:
            self.catalog_secret = self._derive_catalog_hmac_key(new_secret)

    # -----------------------------------------------------------------
    # 1. Ephemeral Unlinkable Ed25519 Key Rotation
    # -----------------------------------------------------------------

    def rotate_ephemeral_identity(self, epoch: Optional[Union[str, int]] = None) -> str:
        """Rotate to a brand-new unlinkable in-memory Ed25519 signing keypair with zero host metadata."""
        self._ephem_priv = ed25519.Ed25519PrivateKey.generate()
        self.rotation_count += 1
        if epoch is not None:
            self._current_epoch_tag = epoch
        return self.current_pubkey_hex()

    def current_pubkey_bytes(self) -> bytes:
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        return self._ephem_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def current_pubkey_hex(self) -> str:
        return self.current_pubkey_bytes().hex()

    def current_privkey_bytes(self) -> bytes:
        from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
        return self._ephem_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())

    # -----------------------------------------------------------------
    # 2. Coarse Epoch Timestamp Bucketing
    # -----------------------------------------------------------------

    def bucket_timestamp(self, ts: Optional[Union[int, float]] = None, bucket_seconds: Optional[int] = None) -> int:
        """Quantize timestamp to coarse UTC hour/day epoch boundary: floor(ts / 3600) * 3600."""
        b_sec = int(bucket_seconds or self.bucket_seconds)
        raw_ts = float(ts if ts is not None else time.time())
        if not math.isfinite(raw_ts) or raw_ts < 0:
            raise AnonymityViolationError(f"invalid_timestamp:{ts}")
        return int(math.floor(raw_ts / b_sec) * b_sec)

    # -----------------------------------------------------------------
    # 3. Clipped + Differentially Private Weight & Calibration Aggregation
    # -----------------------------------------------------------------

    @staticmethod
    def _gaussian_noise(sigma: float) -> float:
        if sigma <= 0.0:
            return 0.0
        return _CSPRNG.gauss(0.0, sigma)

    def clip_and_privatize_weights(
        self,
        delta_w: Sequence[float],
        delta_b: Sequence[float],
        *,
        n_samples: int,
        clip_norm: Optional[float] = None,
        dp_sigma: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Enforce k-anonymity (`n_samples >= min_k_samples`), L2-norm clipping (`||Delta||_2 <= C`),
        calibrated Gaussian DP noise, and int8 quantization (`delta_w_q8`, `delta_b_q8`, `scale`).
        """
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < self.min_k_samples:
            self.blocked_violations_count += 1
            raise AnonymityViolationError(
                f"k_anonymity_threshold_violation:n_samples={n_samples}<{self.min_k_samples}"
            )

        w_list = [float(x) for x in delta_w]
        b_list = [float(x) for x in delta_b]
        if not all(math.isfinite(x) for x in w_list + b_list):
            self.blocked_violations_count += 1
            raise AnonymityViolationError("non_finite_weight_delta")

        c_bound = float(clip_norm if clip_norm is not None else self.clip_norm)
        sigma = float(dp_sigma if dp_sigma is not None else self.dp_sigma)

        sq_sum = sum(x * x for x in w_list) + sum(x * x for x in b_list)
        pre_l2 = math.sqrt(sq_sum)
        clip_factor = min(1.0, c_bound / max(pre_l2, 1e-12))

        w_clipped = [x * clip_factor for x in w_list]
        b_clipped = [x * clip_factor for x in b_list]
        clipped_l2 = math.sqrt(sum(x * x for x in w_clipped) + sum(x * x for x in b_clipped))

        noise_std = sigma * c_bound
        w_noised = [x + self._gaussian_noise(noise_std) for x in w_clipped]
        b_noised = [x + self._gaussian_noise(noise_std) for x in b_clipped]

        # Post-noise norm clip so exported vector is strictly bounded by c_bound
        post_l2 = math.sqrt(sum(x * x for x in w_noised) + sum(x * x for x in b_noised))
        if post_l2 > c_bound:
            post_factor = c_bound / max(post_l2, 1e-12)
            w_noised = [x * post_factor for x in w_noised]
            b_noised = [x * post_factor for x in b_noised]

        max_abs = max([abs(x) for x in (w_noised + b_noised)] + [1e-8])
        scale = max(round(float(max_abs / 127.0), 8), 1e-8)
        w_q8 = [max(-127, min(127, int(round(x / scale)))) for x in w_noised]
        b_q8 = [max(-127, min(127, int(round(x / scale)))) for x in b_noised]
        n_bucket = (int(n_samples) // self.min_k_samples) * self.min_k_samples

        return {
            "delta_w_q8": w_q8,
            "delta_b_q8": b_q8,
            "scale": scale,
            "n_bucket": n_bucket,
            "pre_dp_l2_norm": pre_l2,
            "clipped_l2_norm": clipped_l2,
            "delta_w_dp": [q * scale for q in w_q8],
            "delta_b_dp": [q * scale for q in b_q8],
        }

    def sanitize_reliability_bins(
        self,
        bins: Sequence[Any],
        *,
        dp_sigma: Optional[float] = None,
    ) -> List[List[Union[int, float]]]:
        """Validate and DP-perturb reliability bins, enforcing `count >= min_k_samples` per exported bin."""
        sigma = float(dp_sigma if dp_sigma is not None else self.dp_sigma)
        sanitized: List[List[Union[int, float]]] = []
        for idx, b in enumerate(bins):
            if isinstance(b, Mapping):
                b_idx = int(b.get("bin", b.get("b", idx)))
                conf = float(b.get("conf", b.get("confidence", 0.5)))
                acc = float(b.get("acc", b.get("accuracy", 0.5)))
                count = int(b.get("n", b.get("count", 0)))
            elif isinstance(b, (list, tuple)) and len(b) >= 4:
                b_idx, conf, acc, count = int(b[0]), float(b[1]), float(b[2]), int(b[3])
            elif isinstance(b, (list, tuple)) and len(b) == 3:
                b_idx, conf, acc, count = idx, float(b[0]), float(b[1]), int(b[2])
            else:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"malformed_reliability_bin:{b!r}")

            if count < self.min_k_samples:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(
                    f"reliability_bin_k_anonymity_violation:count={count}<{self.min_k_samples}"
                )
            noisy_conf = round(max(0.0, min(1.0, conf + self._gaussian_noise(sigma * 0.25))), 4)
            noisy_acc = round(max(0.0, min(1.0, acc + self._gaussian_noise(sigma * 0.25))), 4)
            n_bucket = (int(count) // 5) * 5
            sanitized.append([b_idx, noisy_conf, noisy_acc, n_bucket])
        return sanitized

    # -----------------------------------------------------------------
    # 4. One-Way Domain-Separated Catalog Hashing (`armiger-catalog-v2`)
    # -----------------------------------------------------------------

    @classmethod
    def contains_identifying_or_canary_pattern(cls, text: str) -> Tuple[bool, str]:
        """Check whether a string contains any forbidden PII, IP, path, URL, passkey, infohash, or raw title cue."""
        if not text:
            return False, ""
        s = str(text)
        if cls.RE_IPV4_OR_CIDR.search(s):
            return True, "ipv4_or_cidr_detected"
        if cls.RE_CIDR_BLOCK.search(s):
            return True, "cidr_block_detected"
        if cls.RE_IPV6.search(s) or "::1" in s or "fe80:" in s.lower() or "::ffff:" in s.lower():
            return True, "ipv6_detected"
        if cls.RE_URL_OR_MAGNET.search(s):
            return True, "url_or_magnet_detected"
        if cls.RE_FS_PATH.search(s):
            return True, "filesystem_path_or_extension_detected"
        if cls.RE_SECRET_OR_PASSKEY.search(s):
            return True, "secret_or_passkey_detected"
        if cls.RE_EXTERNAL_ID.search(s):
            return True, "external_catalog_id_detected"
        if cls.RE_INFOHASH_40.search(s):
            return True, "raw_40hex_infohash_detected"
        if cls.RE_EMAIL.search(s):
            return True, "email_detected"
        if cls.RE_HOSTNAME.search(s):
            return True, "hostname_or_fqdn_detected"
        # Secondary token-level IPv4/IPv6 check
        for tok in re.split(r"[\s,;|\"'<>()\[\]{}]+", s):
            if not tok:
                continue
            cand = tok.split("/")[0] if "/" in tok else tok
            if ":" in cand or cand.count(".") == 3:
                try:
                    ipaddress.ip_address(cand)
                    return True, "ip_literal_detected"
                except ValueError:
                    pass
        return False, ""

    @classmethod
    def _hash_catalog_key_with_derived_key(
        cls,
        normalized_key: str,
        derived_hmac_key: bytes,
        *,
        reject_raw_pii: bool = True,
    ) -> str:
        """Hash `normalized_key` using an already-derived 32-byte catalog BLAKE2b key."""
        if not isinstance(normalized_key, str) or not normalized_key.strip():
            raise AnonymityViolationError("empty_catalog_key")
        if reject_raw_pii:
            has_pii, reason = cls.contains_identifying_or_canary_pattern(normalized_key)
            if has_pii:
                raise AnonymityViolationError(f"catalog_key_pii_rejected:{reason}")
            if cls.RE_RELEASE_TITLE.search(normalized_key):
                raise AnonymityViolationError("raw_release_name_not_normalized")

        clean = re.sub(r"[^a-z0-9| ]+", "", normalized_key.strip().lower())
        clean = re.sub(r"\s*\|\s*", "|", re.sub(r"\s+", " ", clean)).strip()
        if not clean:
            raise AnonymityViolationError("catalog_key_empty_after_normalization")

        key_bytes = bytes(derived_hmac_key)
        if len(key_bytes) != 32:
            raise AnonymityViolationError("invalid_derived_catalog_hmac_key_length")

        digest = hashlib.blake2b(
            cls.DOMAIN_CATALOG_V2 + b":" + clean.encode("utf-8"),
            key=key_bytes,
            digest_size=8,
            person=b"armiger-cat-v2",
        ).hexdigest()
        assert len(digest) == 16
        return digest

    @classmethod
    def hash_catalog_key(
        cls,
        normalized_key: str,
        *,
        reject_raw_pii: bool = True,
        catalog_secret: Optional[Union[bytes, str]] = None,
    ) -> str:
        """Compute strictly 16-hex-char keyed, domain-separated BLAKE2b-64(key, domain='armiger-catalog-v2' || normalized_key).

        Uses a 32-byte secret key derived once from `catalog_secret`, `ARMIGER_CATALOG_HMAC_KEY`, or the rotated
        process catalog secret so knowledge of a candidate title and public constants alone cannot reproduce `h`.
        Strips/rejects any raw paths, URLs, IPs, passkeys, infohashes, or external IDs.
        """
        hmac_key = cls._derive_catalog_hmac_key(catalog_secret)
        return cls._hash_catalog_key_with_derived_key(
            normalized_key,
            hmac_key,
            reject_raw_pii=reject_raw_pii,
        )

    def build_anonymous_catalog_record(
        self,
        *,
        key_or_hash: str,
        category_code: int,
        res_bucket: int = 0,
        size_bucket: int = 0,
        adult_flag: int = 0,
        viability_bucket: Union[int, float] = 1,
        sample_count: int = 25,
        timestamp: Optional[Union[int, float]] = None,
        is_prehashed: bool = False,
        as_dict: bool = True,
    ) -> Union[Dict[str, Any], List[Any]]:
        """Construct a strictly whitelisted anonymous catalog record `[h, c, r, s, a, v, n, t]`."""
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < self.min_k_samples:
            self.blocked_violations_count += 1
            raise AnonymityViolationError(
                f"catalog_record_k_anonymity_violation:n={sample_count}<{self.min_k_samples}"
            )

        if is_prehashed:
            if not isinstance(key_or_hash, str) or not self.RE_HEX16.match(key_or_hash):
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"invalid_prehashed_catalog_key:{key_or_hash!r}")
            h_val = key_or_hash
        else:
            try:
                h_val = self._hash_catalog_key_with_derived_key(
                    key_or_hash,
                    self.catalog_secret,
                    reject_raw_pii=True,
                )
            except AnonymityViolationError:
                self.blocked_violations_count += 1
                raise

        t_bucket = self.bucket_timestamp(timestamp)
        n_bucket = (int(sample_count) // 5) * 5
        c_val = int(category_code)
        r_val = int(res_bucket)
        s_val = int(size_bucket)
        a_val = 1 if adult_flag else 0
        v_val = round(float(viability_bucket), 4) if isinstance(viability_bucket, float) else int(viability_bucket)

        if as_dict:
            return {
                "h": h_val,
                "c": c_val,
                "r": r_val,
                "s": s_val,
                "a": a_val,
                "v": v_val,
                "n": n_bucket,
                "t": t_bucket,
            }
        return [h_val, c_val, r_val, s_val, a_val, v_val, n_bucket, t_bucket]

    # -----------------------------------------------------------------
    # 5. Deep Wire-Payload PII / Canary Firewall (`inspect_wire_payload`)
    # -----------------------------------------------------------------

    def _parse_json_no_duplicate_keys(self, pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        """Reject duplicate JSON object keys during wire payload decoding to prevent parser differentials."""
        obj: Dict[str, Any] = {}
        for k, v in pairs:
            if k in obj:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"duplicate_json_key:{k}")
            obj[k] = v
        return obj

    def _inspect_decoded_strings(self, node: Any) -> None:
        """Recursively inspect all decoded string keys and values inside a parsed JSON structure."""
        if isinstance(node, str):
            has_pii, pii_reason = self.contains_identifying_or_canary_pattern(node)
            if has_pii:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"wire_canary_firewall_blocked:{pii_reason}")
            if self.RE_RELEASE_TITLE.search(node):
                self.blocked_violations_count += 1
                raise AnonymityViolationError("wire_canary_firewall_blocked:raw_release_title_detected")
            if self.RE_HEX_24_PLUS.search(node):
                self.blocked_violations_count += 1
                raise AnonymityViolationError("wire_canary_firewall_blocked:long_hex_hash_or_passkey_detected")
        elif isinstance(node, Mapping):
            for k, v in node.items():
                if isinstance(k, str):
                    self._inspect_decoded_strings(k)
                self._inspect_decoded_strings(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                self._inspect_decoded_strings(item)

    def inspect_wire_payload(self, payload: Union[bytes, str, Mapping[str, Any]]) -> Dict[str, Any]:
        """Inspect serialized JSON/bencode wire payload (`v`) before signing and before UDP transmission.

        Raises `AnonymityViolationError` if any PII, canary, raw string, unbucketed timestamp,
        `n < min_k_samples`, or non-whitelisted schema key is present.
        """
        raw_text: str
        parsed_obj: Any = None

        if isinstance(payload, (bytes, bytearray)):
            try:
                raw_text = bytes(payload).decode("utf-8")
            except UnicodeDecodeError as exc:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"non_utf8_wire_payload:{exc}") from exc
        elif isinstance(payload, str):
            raw_text = payload
        elif isinstance(payload, Mapping):
            parsed_obj = dict(payload)
            raw_text = json.dumps(parsed_obj, sort_keys=True, separators=(",", ":"))
        else:
            self.blocked_violations_count += 1
            raise AnonymityViolationError(f"unsupported_wire_payload_type:{type(payload).__name__}")

        # 1. Raw string firewall scan across the entire serialized payload
        has_pii, pii_reason = self.contains_identifying_or_canary_pattern(raw_text)
        if has_pii:
            self.blocked_violations_count += 1
            raise AnonymityViolationError(f"wire_canary_firewall_blocked:{pii_reason}")
        if self.RE_RELEASE_TITLE.search(raw_text):
            self.blocked_violations_count += 1
            raise AnonymityViolationError("wire_canary_firewall_blocked:raw_release_title_detected")
        if self.RE_HEX_24_PLUS.search(raw_text):
            self.blocked_violations_count += 1
            raise AnonymityViolationError("wire_canary_firewall_blocked:long_hex_hash_or_passkey_detected")

        # 2. Parse JSON structure (rejecting any duplicate keys via object_pairs_hook)
        if parsed_obj is None:
            stripped = raw_text.strip()
            if stripped.startswith("{"):
                try:
                    parsed_obj = json.loads(
                        stripped,
                        object_pairs_hook=self._parse_json_no_duplicate_keys,
                    )
                except AnonymityViolationError:
                    raise
                except Exception as exc:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"malformed_json_wire_payload:{exc}") from exc
            else:
                self.blocked_violations_count += 1
                raise AnonymityViolationError("wire_payload_must_be_canonical_json_dict")

        if not isinstance(parsed_obj, dict) or not parsed_obj:
            self.blocked_violations_count += 1
            raise AnonymityViolationError("wire_payload_must_be_non_empty_dict")

        # 2b. Inspect all decoded string values after JSON unescaping (`\uXXXX`, etc.)
        self._inspect_decoded_strings(parsed_obj)

        # 3. Enforce strict mathematical top-level key whitelist
        unexpected_keys = set(parsed_obj.keys()) - self.ALLOWED_WIRE_KEYS
        if unexpected_keys:
            self.blocked_violations_count += 1
            raise AnonymityViolationError(f"unexpected_schema_keys:{sorted(unexpected_keys)}")

        # 4. Validate version, seq, model_tier, and coarse timestamp bucketing
        if "v" in parsed_obj:
            v_ver = parsed_obj["v"]
            if isinstance(v_ver, bool) or v_ver not in (1, 2, "1", "2"):
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"invalid_protocol_version:{v_ver!r}")

        if "seq" in parsed_obj:
            seq_val = parsed_obj["seq"]
            if isinstance(seq_val, bool) or not isinstance(seq_val, int) or seq_val < 1:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"invalid_seq:{seq_val!r}")

        if "model_tier" in parsed_obj:
            mt = parsed_obj["model_tier"]
            if isinstance(mt, bool) or mt not in self.ALLOWED_MODEL_TIERS:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"disallowed_model_tier:{mt!r}")

        for ts_key in ("epoch", "ts_utc", "updated_utc"):
            if ts_key in parsed_obj:
                ts_val = parsed_obj[ts_key]
                if isinstance(ts_val, bool) or not isinstance(ts_val, int) or ts_val <= 0:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"invalid_timestamp_type:{ts_key}={ts_val!r}")
                if ts_val % self.bucket_seconds != 0:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(
                        f"unbucketed_high_precision_timestamp:{ts_key}={ts_val}"
                    )

        # 5. Validate k-anonymity sample count `n_bucket`
        if "n_bucket" in parsed_obj:
            nb = parsed_obj["n_bucket"]
            if isinstance(nb, bool) or not isinstance(nb, int) or nb < self.min_k_samples:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(
                    f"k_anonymity_violation:n_bucket={nb}<{self.min_k_samples}"
                )

        # 6. Validate `scale` whenever present, and enforce complete weight-delta schema combinations
        has_weight_delta = ("delta_w_q8" in parsed_obj) or ("delta_b_q8" in parsed_obj)
        has_reliability_bins = "reliability_bins" in parsed_obj
        has_records = "records" in parsed_obj

        if "scale" in parsed_obj:
            sc = parsed_obj["scale"]
            if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not math.isfinite(sc) or sc <= 0 or sc > 10.0:
                self.blocked_violations_count += 1
                raise AnonymityViolationError(f"invalid_quantization_scale:{sc!r}")
            if not has_weight_delta:
                self.blocked_violations_count += 1
                raise AnonymityViolationError("scale_without_weight_delta_schema")

        if has_weight_delta:
            if "scale" not in parsed_obj:
                self.blocked_violations_count += 1
                raise AnonymityViolationError("weight_delta_missing_scale")
            if "n_bucket" not in parsed_obj or int(parsed_obj["n_bucket"]) < self.min_k_samples:
                self.blocked_violations_count += 1
                raise AnonymityViolationError("weight_delta_missing_k_anonymous_n_bucket")
            for arr_key in ("delta_w_q8", "delta_b_q8"):
                if arr_key in parsed_obj:
                    arr = parsed_obj[arr_key]
                    if not isinstance(arr, list) or not arr:
                        self.blocked_violations_count += 1
                        raise AnonymityViolationError(f"invalid_{arr_key}_array")
                    for q in arr:
                        if isinstance(q, bool) or not isinstance(q, int) or q < -128 or q > 127:
                            self.blocked_violations_count += 1
                            raise AnonymityViolationError(f"out_of_range_int8_in_{arr_key}:{q!r}")

        if not (has_weight_delta or has_reliability_bins or has_records):
            self.blocked_violations_count += 1
            raise AnonymityViolationError("incomplete_wire_schema:missing_payload_body")

        if has_records and (has_weight_delta or "scale" in parsed_obj or has_reliability_bins):
            self.blocked_violations_count += 1
            raise AnonymityViolationError("conflicting_wire_schema_combination:records_with_delta_or_bins")

        # 7. Validate `reliability_bins`
        if "reliability_bins" in parsed_obj:
            rbins = parsed_obj["reliability_bins"]
            if not isinstance(rbins, list):
                self.blocked_violations_count += 1
                raise AnonymityViolationError("reliability_bins_must_be_list")
            for b in rbins:
                if not isinstance(b, list) or len(b) != 4:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"invalid_reliability_bin_structure:{b!r}")
                b_idx, conf, acc, cnt = b
                if isinstance(b_idx, bool) or not isinstance(b_idx, int) or b_idx < 0 or b_idx > 100:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"invalid_bin_index:{b_idx!r}")
                if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"invalid_bin_conf:{conf!r}")
                if isinstance(acc, bool) or not isinstance(acc, (int, float)) or not (0.0 <= float(acc) <= 1.0):
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"invalid_bin_acc:{acc!r}")
                if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < self.min_k_samples:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"bin_k_anonymity_violation:{cnt!r}<{self.min_k_samples}")

        # 8. Validate `records` (strict `[h, c, r, s, a, v, n, t]`)
        if "records" in parsed_obj:
            recs = parsed_obj["records"]
            if not isinstance(recs, list) or not recs:
                self.blocked_violations_count += 1
                raise AnonymityViolationError("records_must_be_non_empty_list")
            for rec in recs:
                if isinstance(rec, dict):
                    if set(rec.keys()) != self.ALLOWED_RECORD_KEY_SET:
                        self.blocked_violations_count += 1
                        raise AnonymityViolationError(
                            f"record_schema_violation:keys={sorted(rec.keys())}"
                        )
                    vals = [rec[k] for k in self.ALLOWED_RECORD_KEYS]
                elif isinstance(rec, list) and len(rec) == 8:
                    vals = rec
                else:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"record_must_have_8_fields:{rec!r}")

                h_val, c_val, r_val, s_val, a_val, v_val, n_val, t_val = vals
                if not isinstance(h_val, str) or not self.RE_HEX16.match(h_val):
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"invalid_record_blake2b_hash_h:{h_val!r}")
                for int_field_name, int_field_val in (("c", c_val), ("r", r_val), ("s", s_val), ("a", a_val), ("n", n_val), ("t", t_val)):
                    if isinstance(int_field_val, bool) or not isinstance(int_field_val, int):
                        self.blocked_violations_count += 1
                        raise AnonymityViolationError(f"record_field_not_int:{int_field_name}={int_field_val!r}")
                if isinstance(v_val, bool) or not isinstance(v_val, (int, float)) or not math.isfinite(v_val):
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"record_field_v_invalid:{v_val!r}")
                if n_val < self.min_k_samples:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(
                        f"record_k_anonymity_violation:n={n_val}<{self.min_k_samples}"
                    )
                if t_val <= 0 or t_val % self.bucket_seconds != 0:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"record_unbucketed_timestamp:t={t_val}")

        return parsed_obj

    def sign_wire_payload(
        self,
        v_bytes: bytes,
        *,
        seq: int,
        salt: bytes = b"",
        rotate_key: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Inspect wire payload via Canary Firewall, optionally rotate ephemeral Ed25519 key, and sign."""
        parsed = self.inspect_wire_payload(v_bytes)
        if len(v_bytes) > 1000:
            self.blocked_violations_count += 1
            raise AnonymityViolationError(f"bep44_value_exceeds_1000_bytes:{len(v_bytes)}")

        do_rotate = self.ephemeral_identity if rotate_key is None else bool(rotate_key)
        if do_rotate:
            self.rotate_ephemeral_identity(epoch=parsed.get("epoch"))

        sig_buf = _bep44_canonical_sign_buffer(v=v_bytes, seq=seq, salt=salt)
        sig_bytes = self._ephem_priv.sign(sig_buf)
        pub_bytes = self.current_pubkey_bytes()
        return {
            "v": v_bytes,
            "v_dict": parsed,
            "seq": int(seq),
            "salt": bytes(salt),
            "k": pub_bytes,
            "pubkey_hex": pub_bytes.hex(),
            "privkey_bytes": self.current_privkey_bytes(),
            "sig": sig_bytes,
            "sig_hex": sig_bytes.hex(),
        }

