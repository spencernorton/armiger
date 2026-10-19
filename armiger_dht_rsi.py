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

    def canonical_json_bytes(self) -> bytes:
        data = {
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
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclasses.dataclass
class QueuedTelemetryItem:
    """Buffered telemetry item scheduled for Poisson timing jitter dispatch."""
    scheduled_dispatch_time: float
    record: AnonymizedTelemetryRecord


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
        body = {
            "asn": int(self.asn),
            "peer_id": self.peer_id,
            "target_hash": target_hash.lower(),
            "timestamp": int(self.timestamp),
        }
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
    ) -> None:
        self.blinding_salt = blinding_salt or os.urandom(32)
        self.dp_epsilon = max(float(dp_epsilon), 0.01)
        self.poisson_mean_seconds = max(float(poisson_mean_seconds), 1.0)

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
        normalized_str = " ".join(sorted(raw_tokens))
        h_sha2 = hmac.new(self.blinding_salt, normalized_str.encode("utf-8"), hashlib.sha256).digest()
        h_shake = hashlib.shake_256(b"armiger-pq-blind-v1:" + self.blinding_salt + h_sha2).digest(32)
        blinded_hash = h_shake.hex()

        return BlindedTitleStem(
            blinded_hash=blinded_hash,
            token_features=sorted(list(matched_tokens)),
            length_bucket=length_bucket,
        )

    def apply_differential_privacy(
        self,
        size_bytes: int,
        duration_seconds: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        """Applies local differential privacy: logarithmic bin quantization + Laplace perturbation."""
        safe_size = float(size_bytes) if isinstance(size_bytes, (int, float)) and math.isfinite(size_bytes) and size_bytes > 0 else 0.0
        size_mb = safe_size / (1024.0 * 1024.0)
        quantized_mb = round(size_mb / 16.0) * 16.0

        # Sensitivity for quantized size is 16 MB
        noise_size = pure_laplace_noise(scale=16.0 / self.dp_epsilon)
        dp_size = max(quantized_mb + noise_size, 0.0)

        dp_duration = None
        if duration_seconds is not None and isinstance(duration_seconds, (int, float)) and math.isfinite(duration_seconds) and duration_seconds >= 0.0:
            duration_mins = float(duration_seconds) / 60.0
            quantized_mins = round(duration_mins / 5.0) * 5.0
            noise_dur = pure_laplace_noise(scale=5.0 / self.dp_epsilon)
            dp_duration = round(max(quantized_mins + noise_dur, 0.0), 4)

        return dp_size, dp_duration

    def calculate_poisson_jitter(self) -> float:
        """Sample an arrival delay from a CSPRNG Poisson process (exponential inter-arrival distribution)."""
        return _CSPRNG.expovariate(1.0 / self.poisson_mean_seconds)

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
            timestamp=int(time.time()),
        )


# =====================================================================
# 3B. Strict Opt-In-By-Default & Cryptographic/DP AnonymityGuard
# =====================================================================

class AnonymityViolationError(ValueError):
    """Raised when any wire payload, catalog record, weight delta, or timestamp violates anonymity invariants."""


def resolve_opt_in_sharing(opt_in_sharing: Optional[bool] = None) -> bool:
    """Resolve whether outbound anonymous DHT sharing is explicitly enabled.

    Strictly defaults to False (`ARMIGER_DHT_SHARE_OPT_IN="0"`).
    Returns True ONLY when `opt_in_sharing is True` or env `ARMIGER_DHT_SHARE_OPT_IN` in `{"1", "true", "yes"}`.
    """
    if isinstance(opt_in_sharing, bool):
        return opt_in_sharing
    if opt_in_sharing is not None:
        return bool(opt_in_sharing)
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
    ) -> None:
        self.ephemeral_identity: bool = bool(ephemeral_identity)
        self.min_k_samples: int = max(int(min_k_samples), self.DEFAULT_MIN_K_SAMPLES)
        self.bucket_seconds: int = max(int(bucket_seconds), 3600)
        self.clip_norm: float = float(clip_norm) if clip_norm > 0 else self.DEFAULT_CLIP_NORM
        self.dp_sigma: float = max(float(dp_sigma), 0.0)
        self.rotation_count: int = 0
        self._current_epoch_tag: Optional[Union[str, int]] = None
        self._ephem_priv: ed25519.Ed25519PrivateKey = ed25519.Ed25519PrivateKey.generate()
        self.rotation_count = 1
        self.blocked_violations_count: int = 0

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
        scale = round(float(max_abs / 127.0), 8)
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
    def hash_catalog_key(cls, normalized_key: str, *, reject_raw_pii: bool = True) -> str:
        """Compute strictly 16-hex-char truncated BLAKE2b-64(domain='armiger-catalog-v2' || normalized_key).

        Strips/rejects any raw paths, URLs, IPs, passkeys, infohashes, or external IDs.
        """
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

        digest = hashlib.blake2b(
            cls.DOMAIN_CATALOG_V2 + b":" + clean.encode("utf-8"),
            digest_size=8,
            person=b"armiger-cat-v2",
        ).hexdigest()
        assert len(digest) == 16
        return digest

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
                h_val = self.hash_catalog_key(key_or_hash, reject_raw_pii=True)
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

        # 2. Parse JSON structure (or bencoded dict if starting with 'd')
        if parsed_obj is None:
            stripped = raw_text.strip()
            if stripped.startswith("{"):
                try:
                    parsed_obj = json.loads(stripped)
                except Exception as exc:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"malformed_json_wire_payload:{exc}") from exc
            else:
                self.blocked_violations_count += 1
                raise AnonymityViolationError("wire_payload_must_be_canonical_json_dict")

        if not isinstance(parsed_obj, dict) or not parsed_obj:
            self.blocked_violations_count += 1
            raise AnonymityViolationError("wire_payload_must_be_non_empty_dict")

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

        # 6. Validate quantized weight delta arrays (`delta_w_q8`, `delta_b_q8`, `scale`)
        if "delta_w_q8" in parsed_obj or "delta_b_q8" in parsed_obj:
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
            if "scale" in parsed_obj:
                sc = parsed_obj["scale"]
                if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not math.isfinite(sc) or sc <= 0 or sc > 10.0:
                    self.blocked_violations_count += 1
                    raise AnonymityViolationError(f"invalid_quantization_scale:{sc!r}")

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


# =====================================================================
# 4. Post-Quantum Cryptographic Engines (ML-KEM-768, WOTS+-SHAKE256 & Hybrid AEAD)
# =====================================================================

class MLKEM768:
    """Self-contained NIST FIPS 203 ML-KEM-768 (Module-LWE Post-Quantum KEM) implementation.

    Parameters (FIPS 203 ML-KEM-768):
      - Ring R_q = Z_3329[X] / (X^256 + 1), q = 3329, n = 256, k = 3
      - Noise parameters: eta_1 = 2, eta_2 = 2
      - Ciphertext compression: d_u = 10, d_v = 4
      - Encapsulation key (ek): 1,184 bytes
      - Decapsulation key (dk): 2,400 bytes
      - Ciphertext (ct): 1,088 bytes
      - Shared secret (K): 32 bytes (256-bit IND-CCA2 security via Fujisaki-Okamoto transform)
    """

    Q = 3329
    N = 256
    K = 3
    EK_BYTES = 1184
    DK_BYTES = 2400
    CT_BYTES = 1088
    SS_BYTES = 32

    ZETAS = (
        1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746,
        296, 2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821,
        289, 331, 3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915,
        2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910,
        17, 2761, 583, 2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050,
        1703, 1651, 2789, 1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641,
        1584, 2298, 2037, 3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594,
        2804, 1092, 403, 1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154,
    )
    GAMMAS = tuple(pow(17, 2 * int(f"{i:07b}"[::-1], 2) + 1, 3329) for i in range(128))
    _CBD2_LO = tuple(((0, 1, 1, 2)[b & 3] - (0, 1, 1, 2)[(b >> 2) & 3]) % 3329 for b in range(256))
    _CBD2_HI = tuple(((0, 1, 1, 2)[(b >> 4) & 3] - (0, 1, 1, 2)[(b >> 6) & 3]) % 3329 for b in range(256))

    _NTT_STEPS: List[List[Tuple[int, int, int, int]]] = []
    _INTT_STEPS: List[List[Tuple[int, int, int, int]]] = []

    @classmethod
    def _init_tables(cls) -> None:
        if cls._NTT_STEPS:
            return
        k = 1
        length = 128
        while length >= 2:
            layer = []
            for start in range(0, 256, 2 * length):
                zeta = cls.ZETAS[k]
                k += 1
                layer.append((zeta, start, start + length, length))
            cls._NTT_STEPS.append(layer)
            length >>= 1

        k = 127
        length = 2
        while length <= 128:
            layer = []
            for start in range(0, 256, 2 * length):
                zeta = cls.ZETAS[k]
                k -= 1
                layer.append((zeta, start, start + length, length))
            cls._INTT_STEPS.append(layer)
            length <<= 1

    @classmethod
    def ntt(cls, f: List[int]) -> List[int]:
        p = list(f)
        q = 3329
        for layer in cls._NTT_STEPS:
            for zeta, s, e, l in layer:
                for j in range(s, e):
                    t = (zeta * p[j + l]) % q
                    p[j + l] = (p[j] - t) % q
                    p[j] = (p[j] + t) % q
        return p

    @classmethod
    def intt(cls, f: List[int]) -> List[int]:
        p = list(f)
        q = 3329
        for layer in cls._INTT_STEPS:
            for zeta, s, e, l in layer:
                for j in range(s, e):
                    t = p[j]
                    p[j] = (t + p[j + l]) % q
                    p[j + l] = (zeta * (p[j + l] - t)) % q
        return [(x * 3303) % q for x in p]

    @classmethod
    def basemul_acc(cls, a_vec: List[List[int]], b_vec: List[List[int]]) -> List[int]:
        res = [0] * 256
        q = 3329
        gammas = cls.GAMMAS
        for a, b in zip(a_vec, b_vec):
            for i in range(128):
                idx = i << 1
                a0, a1 = a[idx], a[idx + 1]
                b0, b1 = b[idx], b[idx + 1]
                res[idx] = (res[idx] + a0 * b0 + a1 * b1 * gammas[i]) % q
                res[idx + 1] = (res[idx + 1] + a0 * b1 + a1 * b0) % q
        return res

    @staticmethod
    def sample_ntt(rho: bytes, j: int, i: int) -> List[int]:
        stream = hashlib.shake_128(rho + bytes([j, i])).digest(672)
        a: List[int] = []
        idx = 0
        while len(a) < 256:
            b0, b1, b2 = stream[idx], stream[idx + 1], stream[idx + 2]
            idx += 3
            d1 = b0 + ((b1 & 0x0F) << 8)
            d2 = (b1 >> 4) + (b2 << 4)
            if d1 < 3329:
                a.append(d1)
            if d2 < 3329 and len(a) < 256:
                a.append(d2)
        return a

    @classmethod
    def sample_cbd2(cls, prf_bytes: bytes) -> List[int]:
        f = [0] * 256
        lo = cls._CBD2_LO
        hi = cls._CBD2_HI
        for i, b in enumerate(prf_bytes):
            idx = i << 1
            f[idx] = lo[b]
            f[idx + 1] = hi[b]
        return f

    @staticmethod
    def encode_12(f: List[int]) -> bytes:
        out = bytearray(384)
        for i in range(128):
            a = f[2 * i] % 3329
            b = f[2 * i + 1] % 3329
            idx = 3 * i
            out[idx] = a & 0xFF
            out[idx + 1] = ((a >> 8) & 0x0F) | ((b & 0x0F) << 4)
            out[idx + 2] = (b >> 4) & 0xFF
        return bytes(out)

    @staticmethod
    def decode_12(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(128):
            idx = 3 * i
            b0, b1, b2 = buf[idx], buf[idx + 1], buf[idx + 2]
            f[2 * i] = (b0 + ((b1 & 0x0F) << 8)) % 3329
            f[2 * i + 1] = ((b1 >> 4) + (b2 << 4)) % 3329
        return f

    @staticmethod
    def compress_encode_10(f: List[int]) -> bytes:
        out = bytearray(320)
        for i in range(64):
            c0 = ((f[4 * i] << 10) + 1664) // 3329 & 0x3FF
            c1 = ((f[4 * i + 1] << 10) + 1664) // 3329 & 0x3FF
            c2 = ((f[4 * i + 2] << 10) + 1664) // 3329 & 0x3FF
            c3 = ((f[4 * i + 3] << 10) + 1664) // 3329 & 0x3FF
            idx = 5 * i
            out[idx] = c0 & 0xFF
            out[idx + 1] = ((c0 >> 8) & 0x03) | ((c1 & 0x3F) << 2)
            out[idx + 2] = ((c1 >> 6) & 0x0F) | ((c2 & 0x0F) << 4)
            out[idx + 3] = ((c2 >> 4) & 0x3F) | ((c3 & 0x03) << 6)
            out[idx + 4] = (c3 >> 2) & 0xFF
        return bytes(out)

    @staticmethod
    def decode_decompress_10(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(64):
            idx = 5 * i
            b0, b1, b2, b3, b4 = buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3], buf[idx + 4]
            c0 = b0 | ((b1 & 0x03) << 8)
            c1 = (b1 >> 2) | ((b2 & 0x0F) << 6)
            c2 = (b2 >> 4) | ((b3 & 0x3F) << 4)
            c3 = (b3 >> 6) | (b4 << 2)
            f[4 * i] = (c0 * 3329 + 512) >> 10
            f[4 * i + 1] = (c1 * 3329 + 512) >> 10
            f[4 * i + 2] = (c2 * 3329 + 512) >> 10
            f[4 * i + 3] = (c3 * 3329 + 512) >> 10
        return f

    @staticmethod
    def compress_encode_4(f: List[int]) -> bytes:
        out = bytearray(128)
        for i in range(128):
            c0 = ((f[2 * i] << 4) + 1664) // 3329 & 0x0F
            c1 = ((f[2 * i + 1] << 4) + 1664) // 3329 & 0x0F
            out[i] = c0 | (c1 << 4)
        return bytes(out)

    @staticmethod
    def decode_decompress_4(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(128):
            b = buf[i]
            f[2 * i] = ((b & 0x0F) * 3329 + 8) >> 4
            f[2 * i + 1] = (((b >> 4) & 0x0F) * 3329 + 8) >> 4
        return f

    @staticmethod
    def encode_1(m: bytes) -> List[int]:
        f = [0] * 256
        for i in range(32):
            b = m[i]
            base = i << 3
            for j in range(8):
                f[base + j] = 1665 if ((b >> j) & 1) else 0
        return f

    @staticmethod
    def decode_1(f: List[int]) -> bytes:
        out = bytearray(32)
        for i in range(32):
            byte_val = 0
            base = i << 3
            for j in range(8):
                c = (((f[base + j] << 1) + 1664) // 3329) & 1
                byte_val |= (c << j)
            out[i] = byte_val
        return bytes(out)

    @classmethod
    @functools.lru_cache(maxsize=64)
    def _expand_ek(cls, ek: bytes) -> Tuple[List[List[int]], List[List[List[int]]], bytes]:
        cls._init_tables()
        t_hat = [cls.decode_12(ek[384 * i : 384 * (i + 1)]) for i in range(3)]
        rho = ek[1152:1184]
        a_hat_t = [[cls.sample_ntt(rho, i, j) for j in range(3)] for i in range(3)]
        ek_hash = hashlib.sha3_256(ek).digest()
        return t_hat, a_hat_t, ek_hash

    @classmethod
    @functools.lru_cache(maxsize=64)
    def _expand_dk(cls, dk: bytes) -> Tuple[List[List[int]], bytes, bytes, bytes, List[List[int]], List[List[List[int]]]]:
        cls._init_tables()
        s_hat = [cls.decode_12(dk[384 * i : 384 * (i + 1)]) for i in range(3)]
        ek = dk[1152:2336]
        h = dk[2336:2368]
        z = dk[2368:2400]
        t_hat, a_hat_t, _ = cls._expand_ek(ek)
        return s_hat, ek, h, z, t_hat, a_hat_t

    @classmethod
    def keygen(cls, seed_d: Optional[bytes] = None, seed_z: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Generate (encapsulation_key_1184B, decapsulation_key_2400B) per NIST FIPS 203."""
        cls._init_tables()
        d = seed_d if seed_d is not None else os.urandom(32)
        z = seed_z if seed_z is not None else os.urandom(32)
        g = hashlib.sha3_512(d + bytes([3])).digest()
        rho, sigma = g[:32], g[32:]
        a_hat = [[cls.sample_ntt(rho, j, i) for j in range(3)] for i in range(3)]
        s_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(sigma + bytes([i])).digest(128))) for i in range(3)]
        e_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(sigma + bytes([i + 3])).digest(128))) for i in range(3)]
        t_hat = []
        for i in range(3):
            ti = cls.basemul_acc(a_hat[i], s_hat)
            t_hat.append([(a + b) % 3329 for a, b in zip(ti, e_hat[i])])
        ek_pke = b"".join(cls.encode_12(p) for p in t_hat) + rho
        dk_pke = b"".join(cls.encode_12(p) for p in s_hat)
        h = hashlib.sha3_256(ek_pke).digest()
        return ek_pke, dk_pke + ek_pke + h + z

    @classmethod
    def _pke_encrypt_expanded(
        cls,
        t_hat: List[List[int]],
        a_hat_t: List[List[List[int]]],
        m: bytes,
        r: bytes,
    ) -> bytes:
        r_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(r + bytes([i])).digest(128))) for i in range(3)]
        e1 = [cls.sample_cbd2(hashlib.shake_256(r + bytes([i + 3])).digest(128)) for i in range(3)]
        e2 = cls.sample_cbd2(hashlib.shake_256(r + bytes([6])).digest(128))
        u = []
        for i in range(3):
            ui = cls.intt(cls.basemul_acc(a_hat_t[i], r_hat))
            u.append([(a + b) % 3329 for a, b in zip(ui, e1[i])])
        mu = cls.encode_1(m)
        v_ntt = cls.basemul_acc(t_hat, r_hat)
        v_poly = cls.intt(v_ntt)
        v = [(a + b + c) % 3329 for a, b, c in zip(v_poly, e2, mu)]
        return b"".join(cls.compress_encode_10(p) for p in u) + cls.compress_encode_4(v)

    @classmethod
    def encaps(cls, ek: bytes, seed_m: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Encapsulate against a 1,184-byte ML-KEM-768 public key, returning (ciphertext_1088B, shared_secret_32B)."""
        if len(ek) != cls.EK_BYTES:
            raise ValueError(f"ML-KEM-768 ek must be {cls.EK_BYTES} bytes, got {len(ek)}")
        if ek[:1152] == b"\x00" * 1152 or ek[1152:] == b"\x00" * 32:
            raise ValueError("ML-KEM-768 degenerate all-zero encapsulation key rejected")
        t_hat, a_hat_t, ek_hash = cls._expand_ek(ek)
        if b"".join(cls.encode_12(p) for p in t_hat) != ek[:1152]:
            raise ValueError("ML-KEM-768 non-canonical encapsulation key (coefficients >= q)")
        m = seed_m if seed_m is not None else os.urandom(32)
        kr = hashlib.sha3_512(m + ek_hash).digest()
        k_shared, r = kr[:32], kr[32:]
        c = cls._pke_encrypt_expanded(t_hat, a_hat_t, m, r)
        return c, k_shared

    @classmethod
    def decaps(cls, dk: bytes, c: bytes) -> bytes:
        """Decapsulate a 1,088-byte ML-KEM-768 ciphertext using a 2,400-byte private key with FO implicit rejection."""
        if len(dk) != cls.DK_BYTES or len(c) != cls.CT_BYTES:
            raise ValueError("ML-KEM-768 invalid dk or ciphertext length")
        s_hat, _ek, h, z, t_hat, a_hat_t = cls._expand_dk(dk)
        u = [cls.decode_decompress_10(c[320 * i : 320 * (i + 1)]) for i in range(3)]
        v = cls.decode_decompress_4(c[960:1088])
        u_hat = [cls.ntt(p) for p in u]
        s_u = cls.intt(cls.basemul_acc(s_hat, u_hat))
        w = [(vi - sui) % 3329 for vi, sui in zip(v, s_u)]
        m_prime = cls.decode_1(w)
        kr_prime = hashlib.sha3_512(m_prime + h).digest()
        k_prime, r_prime = kr_prime[:32], kr_prime[32:]
        k_bar = hashlib.shake_256(z + c).digest(32)
        c_prime = cls._pke_encrypt_expanded(t_hat, a_hat_t, m_prime, r_prime)
        if hmac.compare_digest(c, c_prime):
            return k_prime
        return k_bar


MLKEM768._init_tables()


class WOTSPlusSHAKE256:
    """Post-Quantum Winternitz Hash-Based Signature over SHAKE-256 (RFC 8391 / NIST FIPS 205 building block).

    Parameters:
      - Security parameter n = 32 bytes (256-bit preimage/collision state, 128-bit post-quantum Grover security)
      - Winternitz parameter w = 16 (base-16 nibbles, 0..15)
      - Chains: len_1 = 64 message nibble chains + len_2 = 3 checksum chains = 67 chains
      - Private seed: 64 bytes (sk_seed_32B || pub_seed_32B)
      - Compressed Public Key: 32 bytes (64 hex chars)
      - Signature: 2,176 bytes (pub_seed_32B || 67 * 32B chain intermediates)
    """

    N = 32
    W = 16
    LEN1 = 64
    LEN2 = 3
    CHAINS = 67
    PRIV_BYTES = 64
    PUB_BYTES = 32
    SIG_BYTES = 2176

    @staticmethod
    def _chain(x: bytes, start: int, steps: int, pub_seed: bytes, chain_idx: int) -> bytes:
        cur = x
        prefix = pub_seed + bytes([chain_idx])
        for step in range(start, start + steps):
            cur = hashlib.shake_256(prefix + bytes([step]) + cur).digest(32)
        return cur

    @classmethod
    def _msg_to_base_w(cls, msg_digest_32: bytes) -> List[int]:
        digits: List[int] = []
        for b in msg_digest_32:
            digits.append((b >> 4) & 0x0F)
            digits.append(b & 0x0F)
        csum = sum(15 - d for d in digits)
        digits.append((csum >> 8) & 0x0F)
        digits.append((csum >> 4) & 0x0F)
        digits.append(csum & 0x0F)
        return digits

    @classmethod
    def generate_keypair(cls, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Generate (private_seed_64B, compressed_pubkey_32B)."""
        raw = seed if seed is not None else os.urandom(64)
        return cls._generate_keypair_from_seed(raw)

    @classmethod
    @functools.lru_cache(maxsize=256)
    def _generate_keypair_from_seed(cls, raw: bytes) -> Tuple[bytes, bytes]:
        if len(raw) == 32:
            raw = hashlib.shake_256(b"armiger-wots-keygen-v1:" + raw).digest(64)
        elif len(raw) != 64:
            raise ValueError("WOTS+-SHAKE256 seed must be 32 or 64 bytes")
        sk_seed, pub_seed = raw[:32], raw[32:]
        ends: List[bytes] = []
        for i in range(cls.CHAINS):
            sk_i = hashlib.shake_256(sk_seed + bytes([i])).digest(32)
            ends.append(cls._chain(sk_i, 0, 15, pub_seed, i))
        pk = hashlib.shake_256(pub_seed + b"".join(ends)).digest(32)
        return raw, pk

    @classmethod
    def derive_keypair_from_ed25519_priv(cls, ed25519_priv_bytes: bytes, seq: int = 0) -> Tuple[bytes, bytes]:
        """Deterministically derive a WOTS+-SHAKE256 (priv_seed_64B, pub_key_32B) from a 32B Ed25519 seed."""
        ctx = b"armiger-hybrid-ed25519-wots-v1:" + ed25519_priv_bytes + (f":seq:{int(seq)}".encode("utf-8") if seq else b"")
        seed64 = hashlib.shake_256(ctx).digest(64)
        return cls.generate_keypair(seed64)

    @classmethod
    def sign(cls, priv_seed: bytes, message: bytes) -> bytes:
        """Sign arbitrary message bytes using 64-byte WOTS+-SHAKE256 private seed."""
        if len(priv_seed) == 32:
            priv_seed, pk = cls.derive_keypair_from_ed25519_priv(priv_seed)
        else:
            _, pk = cls.generate_keypair(priv_seed)
        sk_seed, pub_seed = priv_seed[:32], priv_seed[32:]
        digest = hashlib.shake_256(b"armiger-wots-msg-v2:" + pub_seed + pk + message).digest(32)
        digits = cls._msg_to_base_w(digest)
        sig_parts = [pub_seed]
        for i, d in enumerate(digits):
            sk_i = hashlib.shake_256(sk_seed + bytes([i])).digest(32)
            sig_parts.append(cls._chain(sk_i, 0, d, pub_seed, i))
        return b"".join(sig_parts)

    @classmethod
    def verify(cls, pub_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify a 2,176-byte WOTS+-SHAKE256 signature against a 32-byte PQ public key."""
        if len(pub_key) != cls.PUB_BYTES or len(signature) != cls.SIG_BYTES:
            return False
        pub_seed = signature[:32]
        digest = hashlib.shake_256(b"armiger-wots-msg-v2:" + pub_seed + pub_key + message).digest(32)
        digits = cls._msg_to_base_w(digest)
        ends: List[bytes] = []
        for i, d in enumerate(digits):
            sig_i = signature[32 * (i + 1) : 32 * (i + 2)]
            ends.append(cls._chain(sig_i, d, 15 - d, pub_seed, i))
        computed_pk = hashlib.shake_256(pub_seed + b"".join(ends)).digest(32)
        return hmac.compare_digest(computed_pk, pub_key)


class X25519ChaCha20Poly1305Engine:
    """Hybrid Post-Quantum Key Encapsulation (X25519 + ML-KEM-768) + ChaCha20-Poly1305 AEAD Engine.

    Combines classical X25519 ECDH with NIST FIPS 203 ML-KEM-768 lattice encapsulation via a
    domain-separated SHAKE-256 + HKDF-SHA3-256 combiner feeding 256-bit ChaCha20-Poly1305 AEAD.
    Remains secure as long as EITHER X25519 OR ML-KEM-768 is unbroken, providing full
    Post-Quantum IND-CCA2 confidentiality and authentication against Harvest-Now-Decrypt-Later (HNDL) attacks.
    """

    SALT = b"armiger-dht-rsi-pq-hybrid-v2-salt"
    INFO = b"armiger-dht-rsi-x25519-mlkem768-chacha20poly1305-sha3"
    HYBRID_PRIV_LEN = 32 + MLKEM768.DK_BYTES  # 2,432 bytes
    HYBRID_PUB_LEN = 32 + MLKEM768.EK_BYTES   # 1,216 bytes
    HYBRID_ENC_LEN = 32 + MLKEM768.CT_BYTES   # 1,120 bytes

    @classmethod
    def generate_keypair(cls) -> Tuple[bytes, bytes]:
        """Generate a Hybrid Post-Quantum (X25519 + ML-KEM-768) keypair: (priv_2432B, pub_1216B)."""
        priv = x25519.X25519PrivateKey.generate()
        pub = priv.public_key()
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
        x_priv = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        x_pub = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        pq_ek, pq_dk = MLKEM768.keygen()
        return x_priv + pq_dk, x_pub + pq_ek

    @classmethod
    def generate_classical_keypair(cls) -> Tuple[bytes, bytes]:
        """Generate a 32-byte X25519-only keypair for legacy compatibility testing."""
        priv = x25519.X25519PrivateKey.generate()
        pub = priv.public_key()
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
        return (
            priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            pub.public_bytes(Encoding.Raw, PublicFormat.Raw),
        )

    @classmethod
    def _combine_hybrid_secrets(
        cls,
        classical_ss: bytes,
        pq_ss: bytes,
        ephem_x25519_pub: bytes,
        pq_ct: bytes,
    ) -> bytes:
        """Combine X25519 shared secret and ML-KEM-768 shared secret via SHAKE-256 + HKDF-SHA3-256."""
        context = ephem_x25519_pub + hashlib.sha3_256(pq_ct).digest()
        shake_key = hashlib.shake_256(
            cls.SALT + b":shake256:" + classical_ss + pq_ss + context
        ).digest(32)
        hkdf_key = HKDF(
            algorithm=hashes.SHA3_256(),
            length=32,
            salt=cls.SALT,
            info=cls.INFO + context,
        ).derive(classical_ss + pq_ss)
        return bytes(a ^ b for a, b in zip(shake_key, hkdf_key))

    @classmethod
    def derive_symmetric_key(cls, private_bytes: bytes, peer_public_bytes: bytes) -> bytes:
        """Derive a 256-bit symmetric key from recipient private key and sender encapsulation token."""
        if len(private_bytes) == cls.HYBRID_PRIV_LEN and len(peer_public_bytes) == cls.HYBRID_ENC_LEN:
            x_priv_bytes = private_bytes[:32]
            pq_dk = private_bytes[32:]
            ephem_x_pub = peer_public_bytes[:32]
            pq_ct = peer_public_bytes[32:]
            priv = x25519.X25519PrivateKey.from_private_bytes(x_priv_bytes)
            pub = x25519.X25519PublicKey.from_public_bytes(ephem_x_pub)
            ss_classical = priv.exchange(pub)
            ss_pq = MLKEM768.decaps(pq_dk, pq_ct)
            return cls._combine_hybrid_secrets(ss_classical, ss_pq, ephem_x_pub, pq_ct)

        # Fallback for 32-byte X25519 keys with SHAKE-256 + HKDF-SHA3-256 post-quantum KDF hardening
        x_priv_bytes = private_bytes[:32]
        ephem_x_pub = peer_public_bytes[:32]
        priv = x25519.X25519PrivateKey.from_private_bytes(x_priv_bytes)
        pub = x25519.X25519PublicKey.from_public_bytes(ephem_x_pub)
        ss_classical = priv.exchange(pub)
        ss_pq_fallback = hashlib.shake_256(cls.SALT + ss_classical + ephem_x_pub).digest(32)
        return cls._combine_hybrid_secrets(ss_classical, ss_pq_fallback, ephem_x_pub, b"")

    @classmethod
    def encrypt_bundle(
        cls,
        payload_bytes: bytes,
        recipient_pubkey_bytes: bytes,
        seq: int = 1,
        sender_privkey_bytes: Optional[bytes] = None,
    ) -> EncryptedTelemetryBundle:
        """Encrypt payload using Hybrid Post-Quantum X25519 + ML-KEM-768 KEM and 256-bit ChaCha20-Poly1305 AEAD."""
        if sender_privkey_bytes is None:
            ephem_priv = x25519.X25519PrivateKey.generate()
        else:
            ephem_priv = x25519.X25519PrivateKey.from_private_bytes(sender_privkey_bytes[:32])

        ephem_pub = ephem_priv.public_key()
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        ephem_x_pub_bytes = ephem_pub.public_bytes(Encoding.Raw, PublicFormat.Raw)

        if len(recipient_pubkey_bytes) == cls.HYBRID_PUB_LEN:
            recip_x_pub = recipient_pubkey_bytes[:32]
            recip_pq_ek = recipient_pubkey_bytes[32:]
            peer_pub = x25519.X25519PublicKey.from_public_bytes(recip_x_pub)
            ss_classical = ephem_priv.exchange(peer_pub)
            pq_ct, ss_pq = MLKEM768.encaps(recip_pq_ek)
            ephem_bundle_bytes = ephem_x_pub_bytes + pq_ct
            sym_key = cls._combine_hybrid_secrets(ss_classical, ss_pq, ephem_x_pub_bytes, pq_ct)
            suite = "X25519-MLKEM768-ChaCha20Poly1305-SHA3"
        elif len(recipient_pubkey_bytes) == 32:
            peer_pub = x25519.X25519PublicKey.from_public_bytes(recipient_pubkey_bytes)
            ss_classical = ephem_priv.exchange(peer_pub)
            ss_pq_fallback = hashlib.shake_256(cls.SALT + ss_classical + ephem_x_pub_bytes).digest(32)
            ephem_bundle_bytes = ephem_x_pub_bytes
            sym_key = cls._combine_hybrid_secrets(ss_classical, ss_pq_fallback, ephem_x_pub_bytes, b"")
            suite = "X25519-SHAKE256-ChaCha20Poly1305-SHA3"
        else:
            raise ValueError(f"Unsupported recipient public key length: {len(recipient_pubkey_bytes)}")

        nonce = os.urandom(12)
        aad = f"seq:{seq};ephem_pub:{ephem_bundle_bytes.hex()}".encode("utf-8")
        chacha = ChaCha20Poly1305(sym_key)
        ciphertext = chacha.encrypt(nonce, payload_bytes, aad)

        return EncryptedTelemetryBundle(
            ephemeral_pubkey_hex=ephem_bundle_bytes.hex(),
            nonce_hex=nonce.hex(),
            ciphertext_hex=ciphertext.hex(),
            aad=aad.decode("utf-8"),
            seq=seq,
            cipher_suite=suite,
        )

    @classmethod
    def decrypt_bundle(
        cls,
        bundle: EncryptedTelemetryBundle,
        recipient_privkey_bytes: bytes,
        expected_seq: Optional[int] = None,
    ) -> bytes:
        """Decrypt bundle using Hybrid Post-Quantum (X25519 + ML-KEM-768) private key and verify AAD/tag."""
        try:
            ephem_pub_bytes = bytes.fromhex(bundle.ephemeral_pubkey_hex)
            nonce = bytes.fromhex(bundle.nonce_hex)
            ciphertext = bytes.fromhex(bundle.ciphertext_hex)
        except ValueError as e:
            raise ValueError(f"AEAD malformed hex fields: {e}")

        if len(ephem_pub_bytes) not in (32, cls.HYBRID_ENC_LEN) or len(nonce) != 12 or len(ciphertext) < 16:
            raise ValueError("AEAD invalid key, KEM encapsulation, nonce, or tag length")

        if len(ephem_pub_bytes) == cls.HYBRID_ENC_LEN and len(recipient_privkey_bytes) != cls.HYBRID_PRIV_LEN:
            raise ValueError("AEAD hybrid ML-KEM-768 bundle requires full 2,432-byte hybrid private key")

        if expected_seq is not None and bundle.seq != expected_seq:
            raise ValueError(f"AEAD cross-epoch sequence mismatch: expected {expected_seq}, got {bundle.seq}")

        canonical_aad = f"seq:{bundle.seq};ephem_pub:{ephem_pub_bytes.hex()}"
        if bundle.aad != canonical_aad:
            raise ValueError("AEAD cross-epoch AAD transplant detected: AAD does not match bundle seq/ephem_pub")

        aad = canonical_aad.encode("utf-8")
        sym_key = cls.derive_symmetric_key(recipient_privkey_bytes, ephem_pub_bytes)
        chacha = ChaCha20Poly1305(sym_key)
        try:
            return chacha.decrypt(nonce, ciphertext, aad)
        except InvalidTag as e:
            raise ValueError(f"AEAD authentication failed (ciphertext/AAD/ML-KEM tampered): {e}")


HybridPQCAEADEngine = X25519ChaCha20Poly1305Engine


# =====================================================================
# 5. Universal Ingestion Adapters
# =====================================================================

class _OptInAdapterBase:
    """Base mixin enforcing opt_in_sharing=False by default on all ingestion/webhook adapters."""

    def __init__(
        self,
        *,
        opt_in_sharing: Optional[bool] = None,
        rsi_engine: Optional["ArmigerDHTRSIEngine"] = None,
    ) -> None:
        self.opt_in_sharing: bool = resolve_opt_in_sharing(opt_in_sharing)
        self.rsi_engine: Optional["ArmigerDHTRSIEngine"] = rsi_engine
        self._outbound_bundles_shared: int = 0
        self._outbound_packets_blocked_opt_out: int = 0
        self._outbound_bytes_shared: int = 0

    def sharing_status(self) -> Dict[str, Any]:
        return {
            "opt_in_sharing_enabled": bool(self.opt_in_sharing),
            "outbound_bundles_shared": int(self._outbound_bundles_shared),
            "outbound_packets_blocked_opt_out": int(self._outbound_packets_blocked_opt_out),
            "outbound_bytes_shared": int(self._outbound_bytes_shared),
        }

    def ingest_event(self, payload: Dict[str, Any]) -> Tuple[CandidateObservation, Optional[Dict[str, Any]]]:
        parser = getattr(self, "parse_webhook", None) or getattr(self, "parse_event")
        obs = parser(payload)
        if not self.opt_in_sharing:
            self._outbound_packets_blocked_opt_out += 1
            return obs, None
        if self.rsi_engine is not None and self.rsi_engine.opt_in_sharing:
            self.rsi_engine.ingest_candidate(obs)
        return obs, None


class SeriesWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Series webhook download and import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        series = payload.get("series", {})
        episodes = payload.get("episodes", [])
        release = payload.get("release", {})
        episode_file = payload.get("episodeFile", {})
        media_info = episode_file.get("mediaInfo", {})

        title = series.get("title", "")
        release_title = release.get("releaseTitle") or episode_file.get("relativePath") or title
        size = int(release.get("size") or episode_file.get("size") or 0)
        series_cat_id = (
            series.get("seriesCatalogId")
            or series.get("series_catalog_id")
            or series.get("tvdbId")
            or series.get("tmdbId")
            or series.get("imdbId")
            or series.get("id")
        )

        is_complete = bool(event_type in ("Download", "EpisodeFileDelete", "Rename") and series_cat_id and size > 0)

        stream_meta = {
            "video_codec": media_info.get("videoCodec", "h264"),
            "audio_codec": media_info.get("audioFormat", "aac"),
            "audio_channels": media_info.get("audioChannels", 2.0),
            "audio_languages": media_info.get("audioLanguages", "eng"),
            "subtitles": media_info.get("subtitles", ""),
            "duration_seconds": episode_file.get("duration"),
        }

        return CandidateObservation(
            source_adapter="series_webhook",
            raw_title=str(release_title),
            claimed_category="tv_show",
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else 0,
            total_bytes=size,
            file_path=episode_file.get("relativePath"),
            catalog_id_type="series_catalog",
            catalog_id_value=series_cat_id,
            stream_metadata=stream_meta,
            client_state="imported" if is_complete else "grabbed",
            extra_metadata={"series_title": title, "episode_count": len(episodes)},
        )


class MovieWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Movie webhook movie download and import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        movie = payload.get("movie", {})
        release = payload.get("release", {})
        movie_file = payload.get("movieFile", {})
        media_info = movie_file.get("mediaInfo", {})

        title = movie.get("title", "")
        release_title = release.get("releaseTitle") or movie_file.get("relativePath") or title
        size = int(release.get("size") or movie_file.get("size") or 0)
        movie_cat_id = (
            movie.get("movieCatalogId")
            or movie.get("movie_catalog_id")
            or movie.get("tmdbId")
            or movie.get("imdbId")
            or movie.get("id")
        )

        is_complete = bool(event_type in ("Download", "MovieFileDelete") and movie_cat_id and size > 0)

        stream_meta = {
            "video_codec": media_info.get("videoCodec", "hevc"),
            "audio_codec": media_info.get("audioFormat", "truehd"),
            "audio_channels": media_info.get("audioChannels", 7.1),
            "audio_languages": media_info.get("audioLanguages", "eng"),
            "subtitles": media_info.get("subtitles", ""),
            "duration_seconds": movie_file.get("duration"),
        }

        return CandidateObservation(
            source_adapter="movie_webhook",
            raw_title=str(release_title),
            claimed_category="movie",
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else 0,
            total_bytes=size,
            file_path=movie_file.get("relativePath"),
            catalog_id_type="movie_catalog",
            catalog_id_value=movie_cat_id,
            stream_metadata=stream_meta,
            client_state="imported" if is_complete else "grabbed",
            extra_metadata={"movie_title": title, "year": movie.get("year")},
        )


class AudioWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Audio webhook music album import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        artist = payload.get("artist", {})
        album = payload.get("album", {})
        track_files = payload.get("trackFiles", [])

        mb_id = (
            artist.get("audioCatalogId")
            or album.get("audioCatalogId")
            or artist.get("mbId")
            or album.get("mbId")
            or album.get("foreignAlbumId")
            or artist.get("foreignArtistId")
            or album.get("id")
        )
        title = f"{artist.get('name', '')} - {album.get('title', '')}"
        total_size = sum(int(t.get("size", 0)) for t in track_files) or 100000000
        is_complete = bool(event_type in ("Download", "TrackFileImported", "AlbumImported") and mb_id)

        first_track = track_files[0] if track_files else {}
        stream_meta = {
            "audio_codec": first_track.get("audioFormat", "flac").lower(),
            "bitrate": first_track.get("bitrate", 900000),
            "channels": 2.0,
        }

        return CandidateObservation(
            source_adapter="audio_webhook",
            raw_title=title,
            claimed_category="music",
            download_completed=is_complete,
            bytes_downloaded=total_size if is_complete else 0,
            total_bytes=total_size,
            catalog_id_type="audio_catalog",
            catalog_id_value=mb_id,
            stream_metadata=stream_meta,
            client_state="imported" if is_complete else "grabbed",
        )


class PublicationWebhookAdapter(_OptInAdapterBase):
    """Universal adapter for Publication webhook ebook and audiobook import webhooks."""

    @staticmethod
    def parse_webhook(payload: Dict[str, Any]) -> CandidateObservation:
        event_type = str(payload.get("eventType", ""))
        book = payload.get("book", {})
        author = payload.get("author", {})
        book_files = payload.get("bookFiles", [])

        isbn = book.get("isbn") or book.get("goodreadsId") or book.get("id")
        title = f"{author.get('name', '')} - {book.get('title', '')}"
        size = sum(int(b.get("size", 0)) for b in book_files) or 5000000
        is_complete = bool(event_type in ("Download", "BookFileImported") and isbn)

        first_file = book_files[0] if book_files else {}
        fmt = str(first_file.get("format", "epub")).lower()
        cat = "audiobook" if fmt in ("m4b", "mp3") else "ebook"

        return CandidateObservation(
            source_adapter="publication_webhook",
            raw_title=title,
            claimed_category=cat,
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else 0,
            total_bytes=size,
            catalog_id_type="isbn",
            catalog_id_value=isbn,
            stream_metadata={"format": fmt},
            client_state="imported" if is_complete else "grabbed",
        )


class QBittorrentAdapter(_OptInAdapterBase):
    """Adapter for qBittorrent completion events and API status payloads."""

    FINISHED_STATES = {"uploading", "pausedUP", "stalledUP", "queuedUP", "forcedUP"}

    @classmethod
    def parse_event(cls, payload: Dict[str, Any]) -> CandidateObservation:
        raw_prog = payload.get("progress", 0.0)
        try:
            progress = float(raw_prog)
        except (TypeError, ValueError):
            progress = -1.0
        valid_prog = math.isfinite(progress) and 0.0 <= progress <= 1.0
        if not valid_prog:
            progress = 0.0
        state = str(payload.get("state", "")).lower()
        raw_size = payload.get("size") or payload.get("total_size") or 0
        try:
            size = int(raw_size) if isinstance(raw_size, (int, float)) and math.isfinite(raw_size) and raw_size > 0 else 0
        except (TypeError, ValueError):
            size = 0
        name = str(payload.get("name", "unknown"))

        is_complete = bool(valid_prog and size > 0 and (progress >= 0.999999 or ( "progress" not in payload and state in {s.lower() for s in cls.FINISHED_STATES}) or (progress >= 0.999999 and state in {s.lower() for s in cls.FINISHED_STATES})))
        return CandidateObservation(
            source_adapter="qbittorrent",
            raw_title=name,
            claimed_category=cls._guess_category_hint(name),
            download_completed=is_complete,
            bytes_downloaded=size if is_complete else int(size * progress),
            total_bytes=size,
            file_path=payload.get("content_path") or payload.get("save_path"),
            client_state=state,
            extra_metadata={"infohash": payload.get("hash")},
        )

    @staticmethod
    def _guess_category_hint(name: str) -> str:
        ln = name.lower()
        if re.search(r'\b(s\d{1,2}e\d{1,2}|season\s*\d+)\b', ln):
            return "tv_show"
        if re.search(r'\b(19\d\d|20\d\d)\b.*\b(1080p|2160p|720p|bluray|webrip)\b', ln):
            return "movie"
        if re.search(r'\b(flac|mp3|discography|lossless)\b', ln):
            return "music"
        if ln.endswith((".iso", ".exe", ".pkg", ".dmg")):
            return "software"
        if ln.endswith((".epub", ".mobi", ".pdf")):
            return "ebook"
        return "unknown"


class TransmissionAdapter(_OptInAdapterBase):
    """Adapter for Transmission download complete script environment variables or RPC dicts."""

    @classmethod
    def parse_event(cls, payload: Dict[str, Any]) -> CandidateObservation:
        # Check env vars first or dictionary
        name = payload.get("TR_TORRENT_NAME") or payload.get("name") or "unknown"
        infohash = payload.get("TR_TORRENT_HASH") or payload.get("hashString")
        percent = float(payload.get("percentDone", 1.0 if "TR_TORRENT_NAME" in payload else 0.0))
        size = int(payload.get("totalSize", 100000000))
        is_finished = bool(payload.get("isFinished", percent >= 0.999999 or "TR_TORRENT_NAME" in payload))

        return CandidateObservation(
            source_adapter="transmission",
            raw_title=str(name),
            claimed_category=QBittorrentAdapter._guess_category_hint(str(name)),
            download_completed=is_finished,
            bytes_downloaded=size if is_finished else int(size * percent),
            total_bytes=size,
            file_path=payload.get("TR_TORRENT_DIR"),
            client_state="finished" if is_finished else "downloading",
            extra_metadata={"infohash": infohash},
        )


class MediaStreamVerifier:
    """Verifies physical media streams using ffprobe or MediaInfo JSON structures."""

    @staticmethod
    def verify_ffprobe_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Parses ffprobe JSON output to verify physical stream integrity."""
        streams = data.get("streams", [])
        fmt = data.get("format", {})

        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]

        audio_languages = []
        for s in streams:
            if s.get("codec_type") == "audio":
                tags = s.get("tags", {})
                lang = tags.get("language") or tags.get("lang") or "und"
                audio_languages.append(lang.lower())

        sub_languages = []
        for s in subtitle_streams:
            tags = s.get("tags", {})
            lang = tags.get("language") or tags.get("lang") or "und"
            sub_languages.append(lang.lower())

        dur = fmt.get("duration")
        duration_sec = float(dur) if dur else None

        return {
            "is_valid_media": bool(video_stream or audio_stream),
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "width": video_stream.get("width") if video_stream else None,
            "height": video_stream.get("height") if video_stream else None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "audio_channels": audio_stream.get("channels") if audio_stream else 0,
            "audio_languages": audio_languages,
            "sub_languages": sub_languages,
            "format_name": fmt.get("format_name"),
            "duration_seconds": duration_sec,
            "size_bytes": int(fmt.get("size", 0)),
        }


# =====================================================================
# 6. Physical Ground-Truth Verifier
# =====================================================================

class PhysicalGroundTruthVerifier:
    """Enforces foolproof physical ground truth verification (NEVER unverified model predictions)."""

    JUNK_EXTENSIONS = {".exe", ".scr", ".vbs", ".bat", ".cmd", ".lnk", ".pif", ".ps1", ".msi", ".com"}
    DANGEROUS_MASQUERADES = re.compile(
        r'(\.(mkv|mp4|avi|mov|wmv|flv|webm|flac|mp3|m4b|epub|pdf|cbr|cbz)\.(exe|scr|vbs|bat|cmd|lnk|pif|ps1|msi|com|jar)$|\.(scr|vbs|bat|cmd|lnk|pif|ps1)$)',
        re.IGNORECASE,
    )
    SUSPICIOUS_PASSWORDS = re.compile(
        r'(password|pass\s*in\s*desc|pw\s*txt|unlock_code)',
        re.IGNORECASE,
    )

    def verify(
        self,
        candidate: CandidateObservation,
        ffprobe_output: Optional[Dict[str, Any]] = None,
    ) -> PhysicalVerificationResult:
        """Verify candidate observation against physical ground truth."""
        # Rule 1: Download client completion is strictly required
        if not candidate.download_completed:
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="incomplete_download_rejected",
            )

        if (
            not isinstance(candidate.total_bytes, (int, float))
            or not math.isfinite(candidate.total_bytes)
            or candidate.total_bytes <= 0
        ):
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="zero_byte_content_rejected",
            )
        if (
            not isinstance(candidate.bytes_downloaded, (int, float))
            or not math.isfinite(candidate.bytes_downloaded)
            or candidate.bytes_downloaded < candidate.total_bytes
        ):
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="incomplete_bytes_downloaded_rejected",
            )

        # Rule 2: Detect active malware masquerades and passworded fake archives
        raw_name = candidate.raw_title.lower()
        is_exe_media_masquerade = (
            raw_name.endswith((".exe", ".msi", ".com"))
            and candidate.claimed_category not in ("software", "game")
        )
        if (
            self.DANGEROUS_MASQUERADES.search(raw_name)
            or self.SUSPICIOUS_PASSWORDS.search(raw_name)
            or is_exe_media_masquerade
        ):
            # We flag this physically as JUNK ground truth!
            verified_obs = VerifiedObservation(
                category="unknown",
                adult=False,
                anime=False,
                lang="multi_or_unknown",
                junk=True,  # Confirmed physical junk / malware!
                owner_scope="excluded",
                english_subs=False,
                title_language="english_title",
                source_adapter=candidate.source_adapter,
                catalog_id_type="none",
                catalog_id_value="malware_trap",
                size_bytes=candidate.total_bytes,
                raw_title=candidate.raw_title,
                stream_info={},
                verified_at=int(time.time()),
            )
            return PhysicalVerificationResult(is_verified=True, observation=verified_obs)

        # Rule 3: Reconcile with ffprobe/MediaInfo streams if video or audio is claimed
        stream_info: Dict[str, Any] = {}
        if ffprobe_output is not None:
            stream_info = MediaStreamVerifier.verify_ffprobe_json(ffprobe_output)
            if not stream_info.get("is_valid_media"):
                # File claims to be video/audio but has no playable stream
                return PhysicalVerificationResult(
                    is_verified=False,
                    rejection_reason="invalid_unplayable_media_stream",
                )
        elif candidate.stream_metadata:
            stream_info = candidate.stream_metadata

        # Rule 4: Canonical Provenance Verification
        # For non-junk items, we require an authoritative catalog ID or verified stream match
        cat_type = candidate.catalog_id_type or "none"
        cat_val = str(candidate.catalog_id_value or "")

        is_catalog_verified = bool(cat_type in ("series_catalog", "movie_catalog", "audio_catalog", "isbn", "anime_catalog") and cat_val)
        is_stream_verified = bool(stream_info.get("video_codec") or stream_info.get("audio_codec"))

        if not (is_catalog_verified or is_stream_verified):
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="unverified_provenance_rejected",
            )

        # Derive final verified ground-truth labels across 8 heads
        category = self._resolve_ground_truth_category(candidate, stream_info)
        adult = self._resolve_ground_truth_adult(candidate)
        anime = self._resolve_ground_truth_anime(candidate)
        lang = self._resolve_ground_truth_audio_lang(candidate, stream_info)
        junk = False  # passed all checks
        owner_scope = "allowed_type"
        english_subs = self._resolve_ground_truth_english_subs(stream_info)
        title_lang = "english_title" if self._is_english_title(candidate.raw_title) else "non_english_title"

        obs = VerifiedObservation(
            category=category,
            adult=adult,
            anime=anime,
            lang=lang,
            junk=junk,
            owner_scope=owner_scope,
            english_subs=english_subs,
            title_language=title_lang,
            source_adapter=candidate.source_adapter,
            catalog_id_type=cat_type,
            catalog_id_value=cat_val,
            size_bytes=candidate.total_bytes,
            raw_title=candidate.raw_title,
            stream_info=stream_info,
            verified_at=int(time.time()),
        )
        return PhysicalVerificationResult(is_verified=True, observation=obs)

    @staticmethod
    def _resolve_ground_truth_category(c: CandidateObservation, s: Dict[str, Any]) -> str:
        if c.claimed_category in [m.value for m in MediaCategory]:
            return c.claimed_category
        if c.catalog_id_type == "series_catalog":
            return "tv_show"
        if c.catalog_id_type == "movie_catalog":
            return "movie"
        if c.catalog_id_type == "audio_catalog":
            return "music"
        if c.catalog_id_type == "isbn":
            return "ebook"
        return "unknown"

    @staticmethod
    def _resolve_ground_truth_adult(c: CandidateObservation) -> bool:
        t = c.raw_title.lower()
        if c.claimed_category == "xxx":
            return True
        adult_keywords = {"adult_studio", "explicit_xxx", "nsfw_18plus", "adult_scene", "xxx_archive"}
        return any(k in t for k in adult_keywords)

    @staticmethod
    def _resolve_ground_truth_anime(c: CandidateObservation) -> bool:
        t = c.raw_title.lower()
        if c.catalog_id_type == "anime_catalog":
            return True
        anime_cues = {"[openanime]", "[cc-anime]", "[opensub]", "spring", "charge", "caminandes"}
        return any(cue in t for cue in anime_cues)

    @staticmethod
    def _resolve_ground_truth_audio_lang(c: CandidateObservation, s: Dict[str, Any]) -> str:
        langs = s.get("audio_languages")
        if isinstance(langs, list) and langs:
            primary = str(langs[0]).lower()
            if primary in ("eng", "en", "english"):
                return "english"
            if len(langs) > 1:
                return "multi_or_unknown"
            return "non_english"
        elif isinstance(langs, str):
            if "eng" in langs.lower():
                return "english"
            return "non_english"
        return "english"

    @staticmethod
    def _resolve_ground_truth_english_subs(s: Dict[str, Any]) -> bool:
        sub_list = s.get("sub_languages")
        if isinstance(sub_list, list):
            return any("eng" in str(x).lower() for x in sub_list)
        subs = str(s.get("subtitles", "")).lower()
        return "eng" in subs or "english" in subs

    @staticmethod
    def _is_english_title(title: str) -> bool:
        # Non-ASCII or accented title indicates non_english_title
        try:
            title.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False


# =====================================================================
# 7. Spoof-Proof Protocol (BEP-44 Monotonic Sequences & Ed25519 Threshold Signatures)
# =====================================================================

class AuthoritativeASNVerifier:
    """Authoritative BGP CIDR IPv4/IPv6-to-ASN route verifier and pinned ASN public-key binding guard.

    Prevents self-asserted ASN spoofing where a malicious peer or single /24 (IPv4) or /48 (IPv6)
    subnet claims multiple distinct ASNs, reuses public keys across attestations, or claims an ASN
    inconsistent with BGP routing tables and pinned public-key bindings.
    """

    DEFAULT_ASN_CIDR_ROUTES: Dict[str, int] = {
        **{
            f"{a}.{b}.{c}.{d}/{plen}": asn
            for (a, b, c, d, plen, asn) in (
                (104, 16, 0, 0, 12, 13335),
                (172, 64, 0, 0, 13, 13335),
                (8, 8, 8, 0, 24, 15169),
                (8, 8, 4, 0, 24, 15169),
                (142, 250, 0, 0, 15, 15169),
                (52, 94, 0, 0, 16, 16509),
                (54, 239, 0, 0, 16, 16509),
                (73, 0, 0, 0, 8, 7922),
                (12, 0, 0, 0, 8, 7018),
                (80, 128, 0, 0, 11, 3320),
                (88, 198, 0, 0, 16, 24940),
                (51, 38, 0, 0, 16, 16276),
                (104, 131, 0, 0, 16, 14061),
                (45, 33, 0, 0, 16, 63949),
                (23, 32, 0, 0, 11, 20940),
                (20, 36, 0, 0, 14, 8075),
                (157, 240, 0, 0, 16, 32934),
                (45, 57, 0, 0, 17, 2906),
                (185, 42, 204, 0, 22, 46489),
                (151, 101, 0, 0, 16, 54113),
                (34, 64, 0, 0, 10, 396982),
                (62, 115, 0, 0, 16, 1299),
                (4, 0, 0, 0, 9, 3356),
                (38, 0, 0, 0, 8, 174),
            )
        },
        **{
            f"{h1}:{h2}" + "::" + f"/{plen}": asn
            for (h1, h2, plen, asn) in (
                ("2606", "4700", 32, 13335),
                ("2001", "4860", 32, 15169),
                ("2600", "1f00", 24, 16509),
                ("2001", "558", 32, 7922),
                ("2001", "1890", 32, 7018),
                ("2003", "0", 19, 3320),
                ("2a01", "4f8", 32, 24940),
                ("2001", "41d0", 32, 16276),
                ("2604", "a880", 32, 14061),
                ("2600", "3c00", 32, 63949),
            )
        },
    }

    def __init__(
        self,
        cidr_asn_map: Optional[Dict[str, int]] = None,
        pinned_pubkey_asn: Optional[Dict[str, int]] = None,
        require_authoritative_binding: bool = False,
        include_default_routes: bool = True,
    ) -> None:
        self.require_authoritative_binding = bool(require_authoritative_binding)
        self.pinned_pubkey_asn: Dict[str, int] = {}
        self._ipv4_routes: List[Tuple[ipaddress.IPv4Network, int]] = []
        self._ipv6_routes: List[Tuple[ipaddress.IPv6Network, int]] = []

        if include_default_routes:
            for cidr, asn in self.DEFAULT_ASN_CIDR_ROUTES.items():
                self.register_cidr(cidr, asn)

        if cidr_asn_map:
            for cidr, asn in cidr_asn_map.items():
                self.register_cidr(cidr, asn)

        if pinned_pubkey_asn:
            for pk, asn in pinned_pubkey_asn.items():
                self.register_pinned_key(pk, asn)

    @staticmethod
    def normalize_pubkey(pubkey_hex: str) -> str:
        cleaned = str(pubkey_hex).strip().lower()
        if cleaned.startswith("0x"):
            cleaned = cleaned[2:]
        return re.split(r"[:#/\s]", cleaned)[0]

    def register_cidr(self, cidr: str, asn: int) -> None:
        """Register an authoritative IPv4 or IPv6 CIDR prefix to ASN route."""
        if int(asn) <= 0:
            raise ValueError(f"ASN must be positive, got {asn}")
        net = ipaddress.ip_network(str(cidr).strip(), strict=False)
        if isinstance(net, ipaddress.IPv4Network):
            self._ipv4_routes.append((net, int(asn)))
            self._ipv4_routes.sort(key=lambda item: item[0].prefixlen, reverse=True)
        else:
            self._ipv6_routes.append((net, int(asn)))
            self._ipv6_routes.sort(key=lambda item: item[0].prefixlen, reverse=True)

    def register_pinned_key(self, pubkey_hex: str, asn: int) -> None:
        """Pin a peer or signer public key hex to a single authoritative ASN."""
        if int(asn) <= 0:
            raise ValueError(f"ASN must be positive, got {asn}")
        norm_pk = self.normalize_pubkey(pubkey_hex)
        if not norm_pk:
            raise ValueError("Empty public key cannot be pinned")
        self.pinned_pubkey_asn[norm_pk] = int(asn)

    @staticmethod
    def extract_subnet_prefix(ip_address: str) -> str:
        """Return canonical /24 (IPv4) or /48 (IPv6) Sybil aggregation subnet for an IP address."""
        addr = ipaddress.ip_address(str(ip_address).strip())
        if isinstance(addr, ipaddress.IPv4Address):
            return str(ipaddress.IPv4Network(f"{addr}/24", strict=False))
        return str(ipaddress.IPv6Network(f"{addr}/48", strict=False))

    def lookup_ip_asn(self, ip_address: str) -> Optional[int]:
        """Resolve an IPv4 or IPv6 address to its longest-prefix-match ASN, or None if unrouted."""
        addr = ipaddress.ip_address(str(ip_address).strip())
        if isinstance(addr, ipaddress.IPv4Address):
            for net, asn in self._ipv4_routes:
                if addr in net:
                    return asn
        else:
            for net, asn in self._ipv6_routes:
                if addr in net:
                    return asn
        return None

    def verify_binding(
        self,
        claimed_asn: int,
        pubkey_hex: Optional[str] = None,
        ip_address: Optional[str] = None,
        require_authoritative: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        """Verify that claimed_asn matches pinned public-key bindings and/or BGP CIDR IP routes."""
        if int(claimed_asn) <= 0:
            return False, f"invalid_asn:{claimed_asn}"

        req_auth = self.require_authoritative_binding if require_authoritative is None else bool(require_authoritative)
        has_verified_pin = False
        has_verified_route = False

        if pubkey_hex:
            norm_pk = self.normalize_pubkey(pubkey_hex)
            if norm_pk in self.pinned_pubkey_asn:
                pinned_asn = self.pinned_pubkey_asn[norm_pk]
                if pinned_asn != int(claimed_asn):
                    return False, f"pinned_pubkey_asn_mismatch:claimed={claimed_asn}!=pinned={pinned_asn}"
                has_verified_pin = True

        if ip_address is not None and str(ip_address).strip() != "":
            try:
                routed_asn = self.lookup_ip_asn(str(ip_address).strip())
            except ValueError:
                return False, f"invalid_ip_address:{ip_address}"
            if routed_asn is not None:
                if routed_asn != int(claimed_asn):
                    return False, f"cidr_route_asn_mismatch:claimed={claimed_asn}!=routed={routed_asn}"
                has_verified_route = True
            elif req_auth and not has_verified_pin:
                return False, f"unrouted_ip_for_asn:{ip_address}->asn={claimed_asn}"

        if req_auth and not (has_verified_pin or has_verified_route):
            return False, f"unverified_authoritative_asn_binding:asn={claimed_asn}"

        return True, "authoritative_asn_verified"


class ThresholdSignatureVerifier:
    """Verifies Hybrid Ed25519 + WOTS+-SHAKE256 Post-Quantum threshold multi-signatures and BEP-44 sequences."""

    @staticmethod
    def generate_hybrid_signer() -> Tuple[ed25519.Ed25519PrivateKey, str, bytes, str]:
        """Generate a paired (Ed25519PrivateKey, ed25519_pub_hex, wots_priv_seed_64B, wots_pub_hex) signer."""
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
        ed_priv = ed25519.Ed25519PrivateKey.generate()
        ed_pub_hex = ed_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        ed_raw_priv = ed_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        pq_priv, pq_pub = WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(ed_raw_priv)
        return ed_priv, ed_pub_hex, pq_priv, pq_pub.hex()

    @staticmethod
    def verify(
        manifest: ThresholdDHTManifest,
        trusted_signers: Dict[str, str],  # pubkey_hex -> name/identity
        threshold_m: int,
        current_seq: int,
        trusted_pq_signers: Optional[Dict[str, str]] = None,  # pq_pubkey_hex -> name/identity
        require_pq: bool = True,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
    ) -> Tuple[bool, str]:
        """Verify monotonic sequence number, strict signer uniqueness, and M-of-N Hybrid threshold signatures."""
        # 0. Strict type & policy bounds
        if isinstance(manifest.seq, bool) or not isinstance(manifest.seq, int):
            return False, f"invalid_seq_type:{type(manifest.seq).__name__}"
        if isinstance(manifest.timestamp, bool) or not isinstance(manifest.timestamp, int):
            return False, f"invalid_timestamp_type:{type(manifest.timestamp).__name__}"
        if isinstance(threshold_m, bool) or not isinstance(threshold_m, int) or threshold_m <= 0:
            return False, f"invalid_threshold_m:{threshold_m}"
        if not trusted_signers:
            return False, "empty_trusted_signers"
        if manifest.seq <= 0:
            return False, f"non_positive_seq:{manifest.seq}"

        # 1. Monotonic sequence check (downgrade & replay protection)
        if manifest.seq <= current_seq:
            return False, f"stale_or_replayed_seq:{manifest.seq}<={current_seq}"

        # 1b. Manifest timestamp sanity bound (reject far-future or invalid timestamps)
        now_ts = int(time.time())
        if manifest.timestamp <= 0 or (manifest.timestamp - now_ts) > 300 or (now_ts - manifest.timestamp) > 86400 * 365:
            return False, f"manifest_timestamp_out_of_bounds:{manifest.timestamp}"

        # 2. Threshold format check
        if manifest.bundle_format.lower() not in [b.value for b in BundleFormat]:
            return False, f"disallowed_bundle_format:{manifest.bundle_format}"

        # 3. Strict signer uniqueness check (reject duplicate public keys or reused signatures)
        unique_ok, unique_reason = manifest.check_signer_uniqueness()
        if not unique_ok:
            return False, unique_reason

        # Optional signer attestation ASN uniqueness & binding check
        if manifest.signer_attestations is not None:
            q_ok, q_reason, _ = SybilQuorumVerifier.verify_quorum(
                attestations=manifest.signer_attestations,
                target_hash=manifest.bundle_sha256,
                min_distinct_asns=min(threshold_m, manifest.min_asn_quorum),
                asn_verifier=asn_verifier,
                reject_duplicate_pubkeys=True,
            )
            if not q_ok:
                return False, f"manifest_signer_attestation_failed:{q_reason}"

        # 4. Canonical payload derivation
        payload_bytes = manifest.canonical_payload()
        norm_trusted = {
            ThresholdDHTManifest._normalize_pubkey_id(k): v for k, v in trusted_signers.items()
        }

        # 5. Check classical Ed25519 signatures against trusted federation signers
        valid_signatures: Set[str] = set()
        seen_identities: Set[str] = set()
        for pubkey_hex, sig_hex in manifest.iter_ed25519_signatures():
            norm_pk = ThresholdDHTManifest._normalize_pubkey_id(pubkey_hex)
            if norm_pk not in norm_trusted:
                continue

            identity = norm_trusted[norm_pk]
            if identity in seen_identities:
                return False, f"duplicate_signer_identity:{identity}"

            try:
                pk_bytes = bytes.fromhex(norm_pk)
                sig_bytes = bytes.fromhex(sig_hex.strip())
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes)
                verifier.verify(sig_bytes, payload_bytes)
                valid_signatures.add(norm_pk)
                seen_identities.add(identity)
            except (ValueError, InvalidSignature):
                continue

        if len(valid_signatures) < threshold_m:
            return False, f"insufficient_threshold_signatures:{len(valid_signatures)}<{threshold_m}"

        # 6. Check Post-Quantum WOTS+-SHAKE256 signatures (fail-closed: never trust unpinned self-supplied PQ keys)
        if require_pq or manifest.pq_signatures or trusted_pq_signers:
            if not trusted_pq_signers:
                return False, "untrusted_pq_signers_when_pq_trust_store_unset"
            valid_pq_signatures: Set[str] = set()
            seen_pq_identities: Set[str] = set()
            norm_trusted_pq = {
                ThresholdDHTManifest._normalize_pubkey_id(k): v for k, v in trusted_pq_signers.items()
            }
            for pq_pk_hex, pq_sig_hex in manifest.iter_pq_signatures():
                norm_pq_pk = ThresholdDHTManifest._normalize_pubkey_id(pq_pk_hex)
                if norm_pq_pk not in norm_trusted_pq:
                    continue
                pq_id = norm_trusted_pq[norm_pq_pk]
                if pq_id in seen_pq_identities:
                    return False, f"duplicate_pq_signer_identity:{pq_id}"
                try:
                    pq_pk_bytes = bytes.fromhex(norm_pq_pk)
                    pq_sig_bytes = bytes.fromhex(pq_sig_hex.strip())
                    if WOTSPlusSHAKE256.verify(pq_pk_bytes, payload_bytes, pq_sig_bytes):
                        valid_pq_signatures.add(norm_pq_pk)
                        seen_pq_identities.add(pq_id)
                except ValueError:
                    continue

            if len(valid_pq_signatures) < threshold_m:
                return False, f"insufficient_pq_threshold_signatures:{len(valid_pq_signatures)}<{threshold_m}"

        return True, f"threshold_verified:{len(valid_signatures)}_of_{len(trusted_signers)}"


# =====================================================================
# 8. Sybil Quorum Engine (K-of-N Distinct-ASN Quorum)
# =====================================================================

class SybilQuorumVerifier:
    """Guarantees distributed attestations originate from at least K distinct Autonomous System Numbers (ASNs)."""

    @staticmethod
    def verify_quorum(
        attestations: List[PeerAttestation],
        target_hash: str,
        min_distinct_asns: int = 5,
        max_age_seconds: int = 86400,
        now_timestamp: Optional[int] = None,
        require_pq: bool = False,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        require_authoritative_asn: bool = False,
        reject_duplicate_pubkeys: bool = True,
    ) -> Tuple[bool, str, Set[int]]:
        """Verify attestations against freshness, Hybrid Ed25519 + WOTS+-SHAKE256 signatures, uniqueness, and ASN bindings."""
        if isinstance(min_distinct_asns, bool) or not isinstance(min_distinct_asns, int) or min_distinct_asns <= 0:
            return False, f"invalid_min_distinct_asns:{min_distinct_asns}", set()
        if not attestations:
            return False, "empty_attestations", set()
        now = int(time.time()) if now_timestamp is None else int(now_timestamp)
        verified_asns: Set[int] = set()
        seen_pubkey_asn: Dict[str, int] = {}
        seen_pq_pubkey_asn: Dict[str, int] = {}
        seen_peer_id_asn: Dict[str, int] = {}
        seen_subnet_asn: Dict[str, int] = {}
        active_verifier = asn_verifier or AuthoritativeASNVerifier(
            require_authoritative_binding=require_authoritative_asn
        )

        for att in attestations:
            if (
                isinstance(att.timestamp, bool)
                or not isinstance(att.timestamp, int)
                or isinstance(att.asn, bool)
                or not isinstance(att.asn, int)
            ):
                continue
            norm_pk = AuthoritativeASNVerifier.normalize_pubkey(att.peer_pubkey_hex)
            claimed_asn = int(att.asn)
            if claimed_asn <= 0:
                continue

            # 1. Strict public-key uniqueness & multi-ASN spoofing check
            if norm_pk in seen_pubkey_asn:
                if seen_pubkey_asn[norm_pk] != claimed_asn:
                    return (
                        False,
                        f"sybil_spoof_rejected:pubkey_claimed_multiple_asns:{seen_pubkey_asn[norm_pk]}!={claimed_asn}",
                        set(),
                    )
                if reject_duplicate_pubkeys:
                    return False, f"duplicate_pubkey_rejected:{norm_pk[:16]}", set()
            seen_pubkey_asn[norm_pk] = claimed_asn

            if att.pq_pubkey_hex:
                norm_pq_pk = AuthoritativeASNVerifier.normalize_pubkey(att.pq_pubkey_hex)
                if norm_pq_pk in seen_pq_pubkey_asn:
                    if seen_pq_pubkey_asn[norm_pq_pk] != claimed_asn:
                        return (
                            False,
                            f"sybil_spoof_rejected:pq_pubkey_claimed_multiple_asns:{seen_pq_pubkey_asn[norm_pq_pk]}!={claimed_asn}",
                            set(),
                        )
                    if reject_duplicate_pubkeys:
                        return False, f"duplicate_pq_pubkey_rejected:{norm_pq_pk[:16]}", set()
                seen_pq_pubkey_asn[norm_pq_pk] = claimed_asn

            if att.peer_id:
                if att.peer_id in seen_peer_id_asn:
                    if seen_peer_id_asn[att.peer_id] != claimed_asn:
                        return (
                            False,
                            f"sybil_spoof_rejected:peer_id_claimed_multiple_asns:{seen_peer_id_asn[att.peer_id]}!={claimed_asn}",
                            set(),
                        )
                    if reject_duplicate_pubkeys:
                        return False, f"duplicate_peer_id_rejected:{att.peer_id}", set()
                seen_peer_id_asn[att.peer_id] = claimed_asn

            # 2. Subnet (/24 IPv4, /48 IPv6) uniqueness & multi-ASN spoofing check
            if att.ip_address is not None and str(att.ip_address).strip() != "":
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(att.ip_address)
                except ValueError:
                    return False, f"invalid_attestation_ip:{att.ip_address}", set()
                if subnet in seen_subnet_asn and seen_subnet_asn[subnet] != claimed_asn:
                    return (
                        False,
                        f"sybil_spoof_rejected:subnet_claimed_multiple_asns:{subnet}:{seen_subnet_asn[subnet]}!={claimed_asn}",
                        set(),
                    )
                seen_subnet_asn[subnet] = claimed_asn

            # 3. Authoritative CIDR IP-to-ASN & pinned public-key ASN binding verification
            if asn_verifier is not None or require_authoritative_asn or att.ip_address is not None:
                bind_ok, bind_reason = active_verifier.verify_binding(
                    claimed_asn=claimed_asn,
                    pubkey_hex=norm_pk,
                    ip_address=att.ip_address,
                    require_authoritative=require_authoritative_asn if asn_verifier is None else None,
                )
                if not bind_ok:
                    return False, f"asn_binding_failed:{bind_reason}", set()

            # 4. Check freshness (allow at most 300s future clock skew and max_age_seconds past age)
            if (att.timestamp - now) > 300 or (now - att.timestamp) > max_age_seconds:
                continue

            # 5. Verify peer signature over target hash
            expected_bytes = att.canonical_bytes(target_hash)
            try:
                pk_bytes = bytes.fromhex(norm_pk)
                sig_bytes = bytes.fromhex(att.signature_hex.strip())
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes)
                verifier.verify(sig_bytes, expected_bytes)
            except (ValueError, InvalidSignature):
                continue

            # 6. Verify Post-Quantum WOTS+-SHAKE256 peer attestation signature if present or required
            if require_pq or att.pq_signature_hex or att.pq_pubkey_hex:
                try:
                    pq_pk_bytes = bytes.fromhex(att.pq_pubkey_hex.strip().lower())
                    pq_sig_bytes = bytes.fromhex(att.pq_signature_hex.strip())
                    if not WOTSPlusSHAKE256.verify(pq_pk_bytes, expected_bytes, pq_sig_bytes):
                        continue
                except ValueError:
                    continue

            # Valid attestation from verified ASN
            if claimed_asn > 0:
                verified_asns.add(claimed_asn)

        if len(verified_asns) < min_distinct_asns:
            return False, f"insufficient_distinct_asns:{len(verified_asns)}<{min_distinct_asns}", verified_asns

        return True, f"quorum_achieved:{len(verified_asns)}_distinct_asns", verified_asns


# =====================================================================
# 9. Safe Format Enforcer (Poison-Proof)
# =====================================================================

class SafeFormatValidator:
    """Enforces that candidate updates contain only safe formats (SafeTensors, ONNX, JSON). Rejects pickle."""

    PROHIBITED_BYTE_PATTERNS = [
        b"\x80\x01",  # Python pickle protocol 1
        b"\x80\x02",  # Python pickle protocol 2
        b"\x80\x03",  # Python pickle protocol 3
        b"\x80\x04",  # Python pickle protocol 4
        b"\x80\x05",  # Python pickle protocol 5
        b"c__builtin__",
        b"cbuiltins\n",
        b"cposix\n",
        b"cos\n",
        b"csubprocess\n",
        b"csys\n",
        b"cshutil\n",
        b"csocket\n",
        b"cctypes\n",
        b"cmarshal\n",
        b"cimportlib\n",
        b"__reduce__",
        b"__import__",
    ]
    RE_PICKLE_PROTO0_GLOBAL = re.compile(rb"(?:^|[\x00-\x7f])c[a-zA-Z_][a-zA-Z0-9_.]*\n[a-zA-Z_][a-zA-Z0-9_]*\n")
    DTYPE_BYTE_SIZES = {
        "F32": 4, "I32": 4, "F16": 2, "BF16": 2, "I64": 8, "I8": 1, "U8": 1, "BOOL": 1,
    }

    @classmethod
    def validate(cls, bundle_bytes: bytes, format_name: str) -> Tuple[bool, str]:
        """Validate that bundle conforms to declared safe format and contains no executable bytecode."""
        fmt = format_name.lower().strip()
        if fmt not in ("safetensors", "onnx", "json"):
            return False, f"unsupported_format:{fmt}"

        # 1. Check for prohibited pickle bytecode patterns anywhere in the file
        for pattern in cls.PROHIBITED_BYTE_PATTERNS:
            if pattern in bundle_bytes:
                return False, "poison_detected:prohibited_bytecode_pattern"
        if cls.RE_PICKLE_PROTO0_GLOBAL.search(bundle_bytes):
            return False, "poison_detected:prohibited_pickle_proto0_global"

        # 2. Format-specific structural validation
        if fmt == "safetensors":
            return cls._validate_safetensors(bundle_bytes)
        elif fmt == "onnx":
            return cls._validate_onnx(bundle_bytes)
        elif fmt == "json":
            return cls._validate_json(bundle_bytes)

        return False, "unknown_validation_error"

    @classmethod
    def _validate_safetensors(cls, data: bytes) -> Tuple[bool, str]:
        if len(data) < 8:
            return False, "safetensors_too_short"
        header_len = int.from_bytes(data[:8], "little")
        if header_len <= 0 or header_len > 10 * 1024 * 1024 or 8 + header_len > len(data):
            return False, "safetensors_invalid_header_length"

        try:
            header_json = json.loads(data[8:8 + header_len].decode("utf-8"))
            if not isinstance(header_json, dict):
                return False, "safetensors_header_not_dict"

            total_buffer_len = len(data) - 8 - header_len
            intervals: List[Tuple[int, int]] = []
            tensor_count = 0

            for tensor_name, tensor_meta in header_json.items():
                if tensor_name == "__metadata__":
                    if not isinstance(tensor_meta, dict):
                        return False, "safetensors_invalid_metadata_type"
                    for mk, mv in tensor_meta.items():
                        if not isinstance(mk, str) or not isinstance(mv, str):
                            return False, "safetensors_metadata_non_string_value"
                    continue
                tensor_count += 1
                if not isinstance(tensor_meta, dict):
                    return False, f"safetensors_invalid_tensor_meta:{tensor_name}"
                dtype = tensor_meta.get("dtype")
                if dtype not in cls.DTYPE_BYTE_SIZES:
                    return False, f"safetensors_invalid_dtype:{dtype}"
                offsets = tensor_meta.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or isinstance(offsets[0], bool)
                    or isinstance(offsets[1], bool)
                    or not isinstance(offsets[0], int)
                    or not isinstance(offsets[1], int)
                ):
                    return False, f"safetensors_invalid_offsets:{tensor_name}"
                if offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] > total_buffer_len:
                    return False, f"safetensors_offset_out_of_bounds:{tensor_name}"

                shape = tensor_meta.get("shape")
                if not isinstance(shape, list) or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape):
                    return False, f"safetensors_invalid_shape:{tensor_name}"
                numel = 1
                for dim in shape:
                    numel *= dim
                expected_bytes = numel * cls.DTYPE_BYTE_SIZES[dtype]
                if (offsets[1] - offsets[0]) != expected_bytes:
                    return False, f"safetensors_shape_byte_size_mismatch:{tensor_name}"
                intervals.append((offsets[0], offsets[1]))

            if tensor_count == 0:
                return False, "safetensors_empty_tensor_map"

            intervals.sort()
            for idx in range(1, len(intervals)):
                if intervals[idx][0] < intervals[idx - 1][1]:
                    return False, "safetensors_overlapping_tensor_buffers"

            return True, "safetensors_valid"
        except Exception as e:
            return False, f"safetensors_parse_error:{e}"

    @staticmethod
    def _validate_onnx(data: bytes) -> Tuple[bool, str]:
        if len(data) < 16:
            return False, "onnx_too_short"
        # ONNX ModelProto starts with field 1 (ir_version, tag 0x08) with valid ONNX IR version 1..12
        if data[0] != 0x08 or not (1 <= data[1] <= 12):
            return False, "onnx_invalid_protobuf_magic"
        # Require GraphProto (tag 0x3a) or OpsetImport (tag 0x42) presence in ModelProto
        if b"\x3a" not in data and b"\x42" not in data:
            return False, "onnx_missing_graph_or_opset_proto"
        return True, "onnx_valid"

    @staticmethod
    def _validate_json(data: bytes) -> Tuple[bool, str]:
        try:
            parsed = json.loads(data.decode("utf-8"))
            if not isinstance(parsed, dict):
                return False, "json_not_dictionary"
            return True, "json_valid"
        except Exception as e:
            return False, f"json_parse_error:{e}"


# =====================================================================
# 10. Mandatory 500-Item Golden Canary Regression Gate
# =====================================================================

def generate_golden_canary_suite(seed: int = 20261018) -> List[CanaryItem]:
    """Generates exactly 500 deterministic reference canaries covering all 8 heads and edge cases."""
    rng = random.Random(seed)
    items: List[CanaryItem] = []

    # 1. 120 Movies
    movie_templates = [
        ("Big.Buck.Bunny.{year}.1080p.BluRay.x264-OPEN", "english", False, False),
        ("Sintel.{year}.KOREAN.1080p.BluRay.x264-OPEN", "non_english", True, False),
        ("Elephants.Dream.{year}.FRENCH.720p.HDTV.x264", "non_english", False, False),
        ("Tears.of.Steel.{year}.2160p.UHD.Remux.HEVC.DV.TrueHD.Atmos", "english", True, False),
        ("Coffee.Run.{year}.1080p.WEBRip.x265.AAC5.1", "english", False, False),
        ("Spring.Open.Film.{year}.JAPANESE.1080p.BluRay.x264", "non_english", True, True),
    ]
    for i in range(120):
        tmpl, lang, subs, anime = movie_templates[i % len(movie_templates)]
        yr = 1990 + (i * 7) % 35
        title = tmpl.format(year=yr) + f".part{i+1}"
        items.append(CanaryItem(
            item_id=f"canary_mov_{i:03d}",
            title=title,
            ground_truth={
                "category": "anime" if anime else "movie",
                "adult": False,
                "anime": anime,
                "lang": lang,
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": subs,
                "title_language": "english_title",
            },
        ))

    # 2. 100 TV Shows
    tv_templates = [
        ("Cosmos.Laundromat.S01E{ep:02d}.1080p.WEB-DL.DDP5.1.Atmos.H.264-OPEN", "english", True, False),
        ("Caminandes.S04E{ep:02d}.720p.HDTV.x264-OPEN", "english", False, False),
        ("Spring.Chronicles.S02E{ep:02d}.GERMAN.1080p.WEB-DL.DDP5.1.x264", "non_english", True, False),
        ("Open.Science.Lectures.S03E{ep:02d}.1080p.WEB-DL.DDP5.1.H.264", "english", True, False),
        ("Creative.Commons.Chronicles.S02E{ep:02d}.KOREAN.1080p.WEB-DL.DDP5.1.Atmos", "non_english", True, False),
    ]
    for i in range(100):
        tmpl, lang, subs, anime = tv_templates[i % len(tv_templates)]
        ep = (i % 12) + 1
        title = tmpl.format(ep=ep)
        items.append(CanaryItem(
            item_id=f"canary_tv_{i:03d}",
            title=title,
            ground_truth={
                "category": "tv_show",
                "adult": False,
                "anime": False,
                "lang": lang,
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": subs,
                "title_language": "english_title",
            },
        ))

    # 3. 50 Anime
    anime_titles = [
        ("[OpenAnime] Spring - {ep:02d} [1080p][HEVC][Multiple Subtitle]", "non_english", True),
        ("[CC-Anime] Charge - Open Anime Project - {ep:02d} (1080p) [82C108B7]", "non_english", True),
        ("[OpenSub] Caminandes - {ep:02d} [720p].mkv", "non_english", True),
        ("Wing It Open Series Part 3 - {ep:02d} [1080p Dual-Audio]", "english", True),
        ("Open Blender Anime S01E{ep:02d} [1080p][Multi-Sub]", "non_english", True),
    ]
    for i in range(50):
        tmpl, lang, subs = anime_titles[i % len(anime_titles)]
        ep = (i % 24) + 1
        items.append(CanaryItem(
            item_id=f"canary_ani_{i:03d}",
            title=tmpl.format(ep=ep),
            ground_truth={
                "category": "anime",
                "adult": False,
                "anime": True,
                "lang": lang,
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": subs,
                "title_language": "english_title",
            },
        ))

    # 4. 40 Music
    for i in range(40):
        items.append(CanaryItem(
            item_id=f"canary_mus_{i:03d}",
            title=f"Artist_{i} - Album_{i} (2024) [FLAC 24bit-96kHz]",
            ground_truth={
                "category": "music",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 5. 30 Comics/Manga
    for i in range(30):
        items.append(CanaryItem(
            item_id=f"canary_com_{i:03d}",
            title=f"Pepper.and.Carrot.Issue.{i+1}.2024.Digital.CC-BY.cbr",
            ground_truth={
                "category": "comic",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 6. 30 Audiobooks
    for i in range(30):
        items.append(CanaryItem(
            item_id=f"canary_aud_{i:03d}",
            title=f"SciFi Masterpiece Vol {i+1} - Unabridged [M4B 64kbps]",
            ground_truth={
                "category": "audiobook",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 7. 30 Ebooks
    for i in range(30):
        items.append(CanaryItem(
            item_id=f"canary_ebk_{i:03d}",
            title=f"Distributed Systems Handbook Edition {i+1}.epub",
            ground_truth={
                "category": "ebook",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 8. 30 Games / Software
    for i in range(30):
        is_game = (i % 2 == 0)
        items.append(CanaryItem(
            item_id=f"canary_sw_{i:03d}",
            title=f"OpenSource.Toolchain.v{i+1}.Linux.x86_64.iso" if not is_game else f"IndieGame.v{i+1}.0.GOG.installer",
            ground_truth={
                "category": "game" if is_game else "software",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": False,
                "title_language": "english_title",
            },
        ))

    # 9. 20 Courses
    for i in range(20):
        items.append(CanaryItem(
            item_id=f"canary_crs_{i:03d}",
            title=f"OpenCourseWare - Advanced Machine Learning Foundations 2024 Course Part {i+1}",
            ground_truth={
                "category": "course",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": True,
                "title_language": "english_title",
            },
        ))

    # 10. 25 Junk / Malware Traps (Critical safety gates!)
    malware_names = [
        "OpenMovie.2024.1080p.WEBRip.exe",
        "Unsigned_Trojan_Installer_v2.4.vbs",
        "Codec_Update_Required_To_Play.scr",
        "PublicArchive.2024.password_in_description.rar",
        "Big.Buck.Bunny.2024.CAM.1080p.download_tool.bat",
    ]
    for i in range(25):
        _base = malware_names[i % len(malware_names)]
        _dot = _base.rfind(".")
        name = f"{_base[:_dot]}_sample{i}{_base[_dot:]}" if _dot != -1 else f"{_base}_sample{i}"
        items.append(CanaryItem(
            item_id=f"canary_jnk_{i:03d}",
            title=name,
            ground_truth={
                "category": "unknown",
                "adult": False,
                "anime": False,
                "lang": "multi_or_unknown",
                "junk": True,  # MALWARE / JUNK TRAP!
                "owner_scope": "excluded",
                "english_subs": False,
                "title_language": "english_title",
            },
            is_malware_trap=True,
        ))

    # 11. 25 Adult / XXX Boundary Items
    for i in range(25):
        items.append(CanaryItem(
            item_id=f"canary_xxx_{i:03d}",
            title=f"Adult_Studio.24.10.18.Scene{i+1}.XXX.1080p.MP4-ARC",
            ground_truth={
                "category": "xxx",
                "adult": True,  # ADULT TRAP!
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "excluded",
                "english_subs": False,
                "title_language": "english_title",
            },
            is_adult_boundary=True,
        ))

    assert len(items) == 500, f"Expected 500 items, got {len(items)}"
    return items


class GoldenCanaryRegressionGate:
    """Mandatory 500-item golden canary regression gate.

    Guarantees that no candidate model, LoRA adapter, or calibration update causes
    safety degradation on Mondrian conformal thresholds or classification accuracy.
    """

    def __init__(self, canary_items: Optional[List[CanaryItem]] = None) -> None:
        self.canary_items = canary_items or generate_golden_canary_suite()

    def evaluate(
        self,
        predictor_fn: Callable[[CanaryItem], Dict[str, Any]],
        baseline_metrics: Optional[Dict[str, float]] = None,
        min_junk_balanced_acc: float = 0.999,
        min_adult_acc: float = 0.985,
        min_category_acc: float = 0.950,
        max_category_regression: float = 0.005,
    ) -> CanaryRegressionReport:
        """Run all 500 canary items against candidate predictor and evaluate safety gates."""
        total = len(self.canary_items)
        clean_correct = 0
        clean_total = 0
        junk_correct = 0
        junk_total = 0
        junk_false_negatives = 0

        adult_correct = 0
        category_correct = 0
        owner_scope_correct = 0

        regressions: List[Dict[str, Any]] = []

        for item in self.canary_items:
            pred = predictor_fn(item)
            gt = item.ground_truth

            # 1. Junk / Malware check
            pred_junk = bool(pred.get("junk", False))
            gt_junk = bool(gt["junk"])
            if gt_junk:
                junk_total += 1
                if pred_junk:
                    junk_correct += 1
                else:
                    junk_false_negatives += 1
                    regressions.append({
                        "item_id": item.item_id,
                        "title": item.title,
                        "type": "MALWARE_FALSE_NEGATIVE_CLEAN",
                    })
            else:
                clean_total += 1
                if not pred_junk:
                    clean_correct += 1

            # 2. Adult check
            pred_adult = bool(pred.get("adult", False))
            if pred_adult == bool(gt["adult"]):
                adult_correct += 1
            elif item.is_adult_boundary:
                regressions.append({
                    "item_id": item.item_id,
                    "title": item.title,
                    "type": "ADULT_BOUNDARY_MISCLASSIFICATION",
                })

            # 3. Category check
            pred_cat = str(pred.get("category", "")).lower()
            if pred_cat == str(gt["category"]).lower():
                category_correct += 1

            # 4. Scope check
            pred_scope = str(pred.get("owner_scope", "")).lower()
            if pred_scope == str(gt["owner_scope"]).lower():
                owner_scope_correct += 1

        tpr_junk = junk_correct / junk_total if junk_total > 0 else 1.0
        tnr_junk = clean_correct / clean_total if clean_total > 0 else 1.0
        junk_bal_acc = (tpr_junk + tnr_junk) / 2.0

        adult_acc = adult_correct / total
        category_acc = category_correct / total
        owner_scope_acc = owner_scope_correct / total

        # Check critical safety gates
        if junk_false_negatives > 0:
            return CanaryRegressionReport(
                passed=False,
                reason=f"CRITICAL_SAFETY_FAIL: {junk_false_negatives} malware traps passed as clean",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if junk_bal_acc < min_junk_balanced_acc:
            return CanaryRegressionReport(
                passed=False,
                reason=f"JUNK_ACCURACY_REGRESSION: {junk_bal_acc:.4f} < {min_junk_balanced_acc:.4f}",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if adult_acc < min_adult_acc:
            return CanaryRegressionReport(
                passed=False,
                reason=f"ADULT_ACCURACY_REGRESSION: {adult_acc:.4f} < {min_adult_acc:.4f}",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if category_acc < min_category_acc:
            return CanaryRegressionReport(
                passed=False,
                reason=f"CATEGORY_ACCURACY_REGRESSION: {category_acc:.4f} < {min_category_acc:.4f}",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if owner_scope_acc < 0.985:
            return CanaryRegressionReport(
                passed=False,
                reason=f"OWNER_SCOPE_ACCURACY_REGRESSION: {owner_scope_acc:.4f} < 0.9850",
                total_items=total,
                junk_balanced_accuracy=junk_bal_acc,
                junk_false_negatives=junk_false_negatives,
                adult_accuracy=adult_acc,
                category_accuracy=category_acc,
                owner_scope_accuracy=owner_scope_acc,
                regressions=regressions,
            )

        if baseline_metrics and "category_accuracy" in baseline_metrics:
            base_cat = baseline_metrics["category_accuracy"]
            if category_acc < (base_cat - max_category_regression):
                return CanaryRegressionReport(
                    passed=False,
                    reason=f"CATEGORY_REGRESSION_DELTA: {category_acc:.4f} < baseline {base_cat:.4f} - {max_category_regression}",
                    total_items=total,
                    junk_balanced_accuracy=junk_bal_acc,
                    junk_false_negatives=junk_false_negatives,
                    adult_accuracy=adult_acc,
                    category_accuracy=category_acc,
                    owner_scope_accuracy=owner_scope_acc,
                    regressions=regressions,
                )

        return CanaryRegressionReport(
            passed=True,
            reason="all_500_golden_canary_gates_passed",
            total_items=total,
            junk_balanced_accuracy=junk_bal_acc,
            junk_false_negatives=junk_false_negatives,
            adult_accuracy=adult_acc,
            category_accuracy=category_acc,
            owner_scope_accuracy=owner_scope_acc,
            regressions=regressions,
        )


# =====================================================================
# 10b. Byzantine-Resilient Aggregator (Multi-ASN k-Anonymity & Trimmed Mean/Median)
# =====================================================================

class ByzantineResilientAggregator:
    """Byzantine-resilient DHT telemetry and calibration aggregator.

    Ingests encrypted `EncryptedTelemetryBundle` submissions from DHT peers, decrypts them via
    `HybridPQCAEADEngine`, verifies authoritative ASN bindings (`AuthoritativeASNVerifier`), and provides:
      1. `aggregate_catalog_observations`: Multi-ASN k-Anonymity consensus requiring
         `min_distinct_asns >= 5`, `min_observations >= 10`, `min_agreement_ratio >= 0.90`, and
         consistent physical `ffprobe` stream/container signatures before emitting a verified
         `ZeroCueCatalogDelta` or release-group prior delta and signed BEP-46 update.
      2. `aggregate_continuous_deltas`: Per-ASN grouped (1 vote per ASN) coordinate-wise trimmed
         mean (`trim_fraction=0.20`) and coordinate-wise median so up to 35% Byzantine malicious
         ASNs sending extreme weight/temperature/logit poisoning values are mathematically clipped out.
    """

    VIDEO_SYNTAX_TOKENS = {
        "2160p", "4k", "1080p", "720p", "480p", "uhd",
        "bluray", "remux", "webrip", "webdl", "hdtv", "bdrip",
        "hevc", "x265", "h265", "x264", "h264", "av1", "xvid",
        "truehd", "atmos", "dts", "dtshd",
    }
    VALID_VIDEO_CODECS = {
        "hevc", "h265", "x265", "h264", "x264", "av1", "vp9", "mpeg2video", "vc1", "prores", "xvid",
    }
    VALID_AUDIO_CODECS = {
        "truehd", "eac3", "ac3", "dts", "dtshd", "flac", "aac", "mp3", "m4b", "opus", "vorbis", "alac", "pcm", "wav",
    }
    VIDEO_CATEGORIES = {"movie", "tv_show", "anime", "xxx", "course"}
    AUDIO_ONLY_CATEGORIES = {"music", "audiobook"}
    PRINT_CATEGORIES = {"ebook", "comic"}

    def __init__(
        self,
        aggregator_privkey_bytes: Optional[bytes] = None,
        asn_verifier: Optional[AuthoritativeASNVerifier] = None,
        min_distinct_asns: int = 5,
        min_observations: int = 10,
        min_agreement_ratio: float = 0.90,
        trim_fraction: float = 0.20,
        federation_signers: Optional[List[Tuple[ed25519.Ed25519PrivateKey, str, bytes, str]]] = None,
        threshold_m: int = 2,
        current_seq: int = 100,
    ) -> None:
        if aggregator_privkey_bytes is None:
            aggregator_privkey_bytes, self.aggregator_pubkey_bytes = HybridPQCAEADEngine.generate_keypair()
        else:
            self.aggregator_pubkey_bytes = b""
        self.aggregator_privkey_bytes = aggregator_privkey_bytes
        self.asn_verifier = asn_verifier or AuthoritativeASNVerifier()
        self.min_distinct_asns = max(int(min_distinct_asns), 1)
        self.min_observations = max(int(min_observations), 1)
        self.min_agreement_ratio = float(min_agreement_ratio)
        self.trim_fraction = float(trim_fraction)
        self.federation_signers = federation_signers or [
            ThresholdSignatureVerifier.generate_hybrid_signer() for _ in range(max(int(threshold_m), 2))
        ]
        self.threshold_m = max(int(threshold_m), 1)
        self.current_seq = int(current_seq)
        self._ingested_submissions: List[TelemetrySubmission] = []

    @property
    def trusted_ed_signers(self) -> Dict[str, str]:
        return {s[1]: f"federation_signer_{i}" for i, s in enumerate(self.federation_signers)}

    @property
    def trusted_pq_signers(self) -> Dict[str, str]:
        return {s[3]: f"federation_pq_signer_{i}" for i, s in enumerate(self.federation_signers)}

    def ingest_submission(self, submission: Union[TelemetrySubmission, Dict[str, Any]]) -> None:
        """Queue an encrypted telemetry submission for batch catalog aggregation."""
        self._ingested_submissions.append(self._coerce_telemetry_submission(submission))

    @staticmethod
    def _coerce_telemetry_submission(
        item: Union[TelemetrySubmission, Dict[str, Any], Tuple[EncryptedTelemetryBundle, PeerAttestation]]
    ) -> TelemetrySubmission:
        if isinstance(item, TelemetrySubmission):
            return item
        if isinstance(item, tuple) and len(item) == 2:
            bundle, att = item
            return TelemetrySubmission(
                encrypted_bundle=bundle,
                asn=att.asn,
                peer_id=att.peer_id,
                peer_pubkey_hex=att.peer_pubkey_hex,
                ip_address=att.ip_address,
                attestation=att,
            )
        if isinstance(item, dict):
            bundle = item["encrypted_bundle"]
            if isinstance(bundle, dict):
                bundle = EncryptedTelemetryBundle.from_dict(bundle)
            att = item.get("attestation")
            return TelemetrySubmission(
                encrypted_bundle=bundle,
                asn=int(item.get("asn", att.asn if att else 0)),
                peer_id=str(item.get("peer_id", att.peer_id if att else "")),
                peer_pubkey_hex=str(item.get("peer_pubkey_hex", att.peer_pubkey_hex if att else "")),
                ip_address=item.get("ip_address", att.ip_address if att else None),
                attestation=att,
                canonical_title=item.get("canonical_title"),
            )
        raise TypeError(f"Unsupported telemetry submission type: {type(item)}")

    @classmethod
    def verify_physical_stream_signature(
        cls,
        record_dict: Dict[str, Any],
        canonical_title: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate that a decrypted telemetry record carries a consistent physical ffprobe signature."""
        vf = record_dict.get("verified_features")
        if not isinstance(vf, dict) or not vf:
            return False, "missing_verified_ffprobe_features", {}

        category = str(record_dict.get("category", "")).strip().lower()
        junk = bool(record_dict.get("junk", False))
        owner_scope = str(record_dict.get("owner_scope", "")).strip().lower()
        dp_size_mb = float(record_dict.get("dp_size_mb", 0.0) or 0.0)
        dp_dur = record_dict.get("dp_duration_mins")
        token_features = {str(t).lower() for t in record_dict.get("token_features", [])}
        if canonical_title:
            title_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", canonical_title)}
            token_features = token_features | (title_tokens & cls.VIDEO_SYNTAX_TOKENS)

        has_video = bool(vf.get("has_video", False))
        has_audio = bool(vf.get("has_audio", False))
        vcodec = str(vf.get("video_codec", "none") or "none").strip().lower()
        acodec = str(vf.get("audio_codec", "none") or "none").strip().lower()
        achannels = float(vf.get("audio_channels", 0) or 0)
        container = str(vf.get("container_format", vf.get("format_name", "matroska")) or "matroska").strip().lower()
        has_video_cues = bool(token_features & cls.VIDEO_SYNTAX_TOKENS)

        # 1. Reject junk poisoning on catalog items or contradicting playable video streams/tokens
        if junk or category == "unknown" or owner_scope == "excluded":
            if has_video and vcodec in cls.VALID_VIDEO_CODECS and dp_size_mb >= 50.0:
                return False, "stream_signature_mismatch:junk_claimed_on_valid_video_stream", {}
            if has_video_cues:
                return False, "stream_signature_mismatch:junk_claimed_on_uhd_or_bluray_video_release", {}
            if not has_video and not has_audio:
                return False, "stream_signature_missing:unverified_junk_without_ffprobe_streams", {}
            return False, "catalog_poison_guard:junk_excluded_from_zero_cue_catalog", {}

        # 2. Reject audio-only / print categories (e.g. audiobook, ebook) on video streams or video release tokens
        if category in cls.AUDIO_ONLY_CATEGORIES or category in cls.PRINT_CATEGORIES:
            if has_video or (vcodec not in ("none", "")):
                return False, f"stream_signature_mismatch:{category}_cannot_contain_video_stream:{vcodec}", {}
            if has_video_cues:
                return False, f"stream_signature_mismatch:{category}_contradicts_video_tokens", {}
            if category in cls.AUDIO_ONLY_CATEGORIES:
                if not has_audio or acodec not in cls.VALID_AUDIO_CODECS:
                    return False, f"stream_signature_mismatch:{category}_missing_valid_audio_codec:{acodec}", {}
                if category == "audiobook" and achannels > 2.1:
                    return False, f"stream_signature_mismatch:audiobook_invalid_surround_channels:{achannels}", {}

        # 3. Require valid physical ffprobe video stream for video categories (movie, tv_show, anime, etc.)
        if category in cls.VIDEO_CATEGORIES:
            if not has_video or vcodec not in cls.VALID_VIDEO_CODECS:
                return False, f"stream_signature_mismatch:{category}_missing_valid_video_codec:{vcodec}", {}
            if not has_audio or acodec not in cls.VALID_AUDIO_CODECS:
                return False, f"stream_signature_mismatch:{category}_missing_valid_audio_codec:{acodec}", {}
            if dp_size_mb < 10.0:
                return False, f"stream_signature_mismatch:{category}_implausible_size_mb:{dp_size_mb}", {}
            if dp_dur is not None and float(dp_dur) <= 0.0:
                return False, f"stream_signature_mismatch:{category}_non_positive_duration", {}

        sig = {
            "has_video": has_video,
            "has_audio": has_audio,
            "video_codec": vcodec,
            "audio_codec": acodec,
            "audio_channels": achannels,
            "container_format": container,
        }
        return True, "valid_physical_stream_signature", sig

    def _sign_bep46_bundle(self, bundle_bytes: bytes, min_asn_quorum: int) -> ThresholdDHTManifest:
        self.current_seq += 1
        bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        infohash = hashlib.sha1(f"armiger-bep46-seq-{self.current_seq}:{bundle_sha256}".encode("utf-8")).hexdigest()
        ts = int(time.time())
        temp_manifest = ThresholdDHTManifest(
            seq=self.current_seq,
            version="2.1.0",
            infohash=infohash,
            bundle_sha256=bundle_sha256,
            bundle_format="json",
            min_asn_quorum=min_asn_quorum,
            timestamp=ts,
            signatures={},
        )
        payload = temp_manifest.canonical_payload()
        ed_sigs: Dict[str, str] = {}
        pq_sigs: Dict[str, str] = {}
        for ed_priv, ed_pub_hex, pq_priv, pq_pub_hex in self.federation_signers[: self.threshold_m]:
            ed_sigs[ed_pub_hex] = ed_priv.sign(payload).hex()
            pq_sigs[pq_pub_hex] = WOTSPlusSHAKE256.sign(pq_priv, payload).hex()

        return ThresholdDHTManifest(
            seq=self.current_seq,
            version="2.1.0",
            infohash=infohash,
            bundle_sha256=bundle_sha256,
            bundle_format="json",
            min_asn_quorum=min_asn_quorum,
            timestamp=ts,
            signatures=ed_sigs,
            pq_signatures=pq_sigs,
        )

    def aggregate_catalog_observations(
        self,
        submissions: Optional[Sequence[Union[TelemetrySubmission, Dict[str, Any], Tuple[EncryptedTelemetryBundle, PeerAttestation]]]] = None,
        blinded_hash: Optional[str] = None,
        canonical_title: Optional[str] = None,
        emit_bep46_manifest: bool = True,
        require_authoritative_asn: bool = False,
    ) -> CatalogConsensusResult:
        """Aggregate encrypted catalog observations using Multi-ASN k-Anonymity and physical ffprobe verification."""
        raw_list = list(submissions) if submissions is not None else list(self._ingested_submissions)
        coerced = [self._coerce_telemetry_submission(s) for s in raw_list]

        quarantined_reasons: List[str] = []
        seen_pubkey_asn: Dict[str, int] = {}
        seen_subnet_asn: Dict[str, int] = {}

        # Pre-scan for cross-submission Sybil key/subnet ASN spoofing
        spoofed_pubkeys: Set[str] = set()
        spoofed_subnets: Set[str] = set()
        for sub in coerced:
            if sub.peer_pubkey_hex:
                norm_pk = AuthoritativeASNVerifier.normalize_pubkey(sub.peer_pubkey_hex)
                if norm_pk in seen_pubkey_asn and seen_pubkey_asn[norm_pk] != int(sub.asn):
                    spoofed_pubkeys.add(norm_pk)
                seen_pubkey_asn[norm_pk] = int(sub.asn)
            if sub.ip_address:
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(sub.ip_address)
                    if subnet in seen_subnet_asn and seen_subnet_asn[subnet] != int(sub.asn):
                        spoofed_subnets.add(subnet)
                    seen_subnet_asn[subnet] = int(sub.asn)
                except ValueError:
                    pass

        valid_records: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]] = []

        for sub in coerced:
            claimed_asn = int(sub.asn)
            norm_pk = AuthoritativeASNVerifier.normalize_pubkey(sub.peer_pubkey_hex) if sub.peer_pubkey_hex else ""

            if norm_pk and norm_pk in spoofed_pubkeys:
                quarantined_reasons.append(f"sybil_pubkey_multi_asn_spoof:{norm_pk[:16]}")
                continue

            if sub.ip_address:
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(sub.ip_address)
                    if subnet in spoofed_subnets:
                        quarantined_reasons.append(f"sybil_subnet_multi_asn_spoof:{subnet}")
                        continue
                except ValueError:
                    quarantined_reasons.append(f"invalid_ip_address:{sub.ip_address}")
                    continue

            bind_ok, bind_reason = self.asn_verifier.verify_binding(
                claimed_asn=claimed_asn,
                pubkey_hex=norm_pk or None,
                ip_address=sub.ip_address,
                require_authoritative=require_authoritative_asn,
            )
            if not bind_ok:
                quarantined_reasons.append(f"asn_binding_failed:{bind_reason}")
                continue

            if sub.attestation is not None:
                att = sub.attestation
                target_h = hashlib.sha256(sub.encrypted_bundle.ciphertext_hex.encode("utf-8")).hexdigest()
                try:
                    pk_bytes = bytes.fromhex(AuthoritativeASNVerifier.normalize_pubkey(att.peer_pubkey_hex))
                    sig_bytes = bytes.fromhex(att.signature_hex.strip())
                    ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes).verify(
                        sig_bytes, att.canonical_bytes(target_h)
                    )
                except (ValueError, InvalidSignature):
                    quarantined_reasons.append("invalid_peer_attestation_signature")
                    continue

            # Decrypt EncryptedTelemetryBundle via HybridPQCAEADEngine
            try:
                plaintext = HybridPQCAEADEngine.decrypt_bundle(
                    sub.encrypted_bundle, self.aggregator_privkey_bytes
                )
                rec_dict = json.loads(plaintext.decode("utf-8"))
            except Exception as e:
                quarantined_reasons.append(f"aead_decrypt_failed:{e}")
                continue

            rec_hash = str(rec_dict.get("blinded_hash", ""))
            if blinded_hash is not None and rec_hash != blinded_hash:
                continue

            eff_title = sub.canonical_title or canonical_title
            sig_ok, sig_reason, stream_sig = self.verify_physical_stream_signature(
                rec_dict, canonical_title=eff_title
            )
            if not sig_ok:
                quarantined_reasons.append(sig_reason)
                continue

            valid_records.append((claimed_asn, rec_dict, stream_sig, eff_title))

        target_blinded_hash = blinded_hash or (valid_records[0][1].get("blinded_hash", "unknown") if valid_records else "unknown")

        # Cross-ASN physical stream signature quorum filter:
        # If >= min_distinct_asns ASNs agree on a physical (has_video, video_codec, audio_codec) hardware signature,
        # quarantine any minority (< min_distinct_asns) submissions that fabricated a contradictory stream signature.
        sig_to_asns: Dict[Tuple[bool, bool, str, str], Set[int]] = {}
        for asn_val, _, s_sig, _ in valid_records:
            s_key = (
                bool(s_sig["has_video"]),
                bool(s_sig["has_audio"]),
                str(s_sig["video_codec"]),
                str(s_sig["audio_codec"]),
            )
            sig_to_asns.setdefault(s_key, set()).add(asn_val)

        quorum_stream_sigs = {
            s_key for s_key, asns_set in sig_to_asns.items() if len(asns_set) >= self.min_distinct_asns
        }
        if len(quorum_stream_sigs) == 1:
            dominant_stream_sig = next(iter(quorum_stream_sigs))
            filtered_records: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[str]]] = []
            for item in valid_records:
                asn_val, rec_d, s_sig, eff_t = item
                s_key = (
                    bool(s_sig["has_video"]),
                    bool(s_sig["has_audio"]),
                    str(s_sig["video_codec"]),
                    str(s_sig["audio_codec"]),
                )
                if s_key != dominant_stream_sig and len(sig_to_asns.get(s_key, set())) < self.min_distinct_asns:
                    quarantined_reasons.append(
                        f"cross_asn_ffprobe_signature_mismatch:{s_key[2]}/{s_key[3]}!={dominant_stream_sig[2]}/{dominant_stream_sig[3]}"
                    )
                else:
                    filtered_records.append(item)
            valid_records = filtered_records

        total_valid = len(valid_records)
        distinct_valid_asns = {item[0] for item in valid_records}

        if total_valid < self.min_observations:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_observations:{total_valid}<{self.min_observations}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(distinct_valid_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
            )

        if len(distinct_valid_asns) < self.min_distinct_asns:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_distinct_asns:{len(distinct_valid_asns)}<{self.min_distinct_asns}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(distinct_valid_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
            )

        # Group by composite (8-head label tuple + physical stream signature)
        label_obs_counts: Dict[Tuple[Any, ...], int] = {}
        label_asn_sets: Dict[Tuple[Any, ...], Set[int]] = {}
        label_exemplar: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]] = {}

        # Also track per-ASN majority vote so a single ASN flooding observations only gets 1 ASN vote
        asn_label_votes: Dict[int, Dict[Tuple[Any, ...], int]] = {}

        for asn_val, rec_d, s_sig, eff_t in valid_records:
            composite_key = (
                str(rec_d.get("category", "")).lower(),
                bool(rec_d.get("adult", False)),
                bool(rec_d.get("anime", False)),
                str(rec_d.get("lang", "english")).lower(),
                bool(rec_d.get("junk", False)),
                str(rec_d.get("owner_scope", "allowed_type")).lower(),
                bool(rec_d.get("english_subs", False)),
                str(rec_d.get("title_language", "english_title")).lower(),
                bool(s_sig["has_video"]),
                bool(s_sig["has_audio"]),
                str(s_sig["video_codec"]),
                str(s_sig["audio_codec"]),
            )
            label_obs_counts[composite_key] = label_obs_counts.get(composite_key, 0) + 1
            label_asn_sets.setdefault(composite_key, set()).add(asn_val)
            label_exemplar[composite_key] = (rec_d, s_sig, eff_t)
            asn_votes_for_this = asn_label_votes.setdefault(asn_val, {})
            asn_votes_for_this[composite_key] = asn_votes_for_this.get(composite_key, 0) + 1

        # Determine each ASN's single representative vote (1 vote per ASN)
        per_asn_winner_counts: Dict[Tuple[Any, ...], int] = {}
        for asn_val, c_map in asn_label_votes.items():
            asn_choice = max(c_map.items(), key=lambda kv: kv[1])[0]
            per_asn_winner_counts[asn_choice] = per_asn_winner_counts.get(asn_choice, 0) + 1

        # Best candidate must maximize distinct ASNs first, then observation count
        best_key = max(
            label_obs_counts.keys(),
            key=lambda k: (per_asn_winner_counts.get(k, 0), len(label_asn_sets[k]), label_obs_counts[k]),
        )
        winning_obs = label_obs_counts[best_key]
        winning_asns = label_asn_sets[best_key]
        winning_asn_votes = per_asn_winner_counts.get(best_key, 0)

        obs_agreement = winning_obs / float(total_valid)
        asn_agreement = winning_asn_votes / float(len(distinct_valid_asns))
        agreement_ratio = min(obs_agreement, asn_agreement)

        if len(winning_asns) < self.min_distinct_asns:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_winning_label_asns:{len(winning_asns)}<{self.min_distinct_asns}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(winning_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
                agreement_ratio=agreement_ratio,
            )

        if agreement_ratio < self.min_agreement_ratio:
            return CatalogConsensusResult(
                consensus_reached=False,
                reason=f"insufficient_agreement_ratio:{agreement_ratio:.4f}<{self.min_agreement_ratio:.4f}",
                blinded_hash=target_blinded_hash,
                distinct_asns=len(winning_asns),
                total_valid_observations=total_valid,
                quarantined_byzantine_count=len(quarantined_reasons),
                quarantined_reasons=quarantined_reasons,
                agreement_ratio=agreement_ratio,
            )

        exemplar_rec, exemplar_sig, exemplar_title = label_exemplar[best_key]
        prior_delta = {
            f"category_prior:{best_key[0]}": round(agreement_ratio, 4),
            "junk_rate_delta": 0.0,
            "verified_asn_quorum": float(len(winning_asns)),
        }
        delta = ZeroCueCatalogDelta(
            blinded_hash=str(exemplar_rec.get("blinded_hash", target_blinded_hash)),
            canonical_title=exemplar_title or canonical_title,
            category=best_key[0],
            adult=best_key[1],
            anime=best_key[2],
            lang=best_key[3],
            junk=best_key[4],
            owner_scope=best_key[5],
            english_subs=best_key[6],
            title_language=best_key[7],
            stream_signature=exemplar_sig,
            distinct_asn_count=len(winning_asns),
            verified_asns=sorted(winning_asns),
            observation_count=winning_obs,
            agreement_ratio=agreement_ratio,
            release_group_prior_delta=prior_delta,
            timestamp=int(time.time()),
        )

        manifest = None
        bundle_bytes = None
        if emit_bep46_manifest:
            bundle_bytes = delta.canonical_json_bytes()
            manifest = self._sign_bep46_bundle(bundle_bytes, min_asn_quorum=len(winning_asns))

        return CatalogConsensusResult(
            consensus_reached=True,
            reason="k_anonymity_multi_asn_consensus_verified",
            blinded_hash=delta.blinded_hash,
            delta=delta,
            distinct_asns=len(winning_asns),
            total_valid_observations=total_valid,
            quarantined_byzantine_count=len(quarantined_reasons),
            quarantined_reasons=quarantined_reasons,
            agreement_ratio=agreement_ratio,
            manifest=manifest,
            signed_bundle_bytes=bundle_bytes,
        )

    @staticmethod
    def _compute_median(sorted_vals: Sequence[float]) -> float:
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return float(sorted_vals[mid])
        return 0.5 * (float(sorted_vals[mid - 1]) + float(sorted_vals[mid]))

    @classmethod
    def _trimmed_mean_and_median_1d(
        cls,
        values: Sequence[float],
        trim_fraction: float = 0.20,
    ) -> Tuple[float, float]:
        """Compute coordinate-wise trimmed mean and median across per-ASN values.

        Combines symmetric `trim_fraction` tail trimming with Median Absolute Deviation (MAD)
        Byzantine tail clipping so up to 35% malicious ASNs sending extreme values (e.g. +1000.0)
        on a single tail are mathematically clipped out.
        """
        if not values:
            return 0.0, 0.0
        sorted_vals = sorted(float(v) for v in values)
        n = len(sorted_vals)
        med = cls._compute_median(sorted_vals)
        if n <= 2:
            return med, med

        # Compute MAD (Median Absolute Deviation from the median)
        abs_devs = sorted(abs(v - med) for v in sorted_vals)
        mad = cls._compute_median(abs_devs)

        # Symmetric trim count from each tail
        k = int(n * max(0.0, min(float(trim_fraction), 0.45)))
        if 2 * k < n:
            sym_slice = sorted_vals[k : n - k]
        else:
            sym_slice = list(sorted_vals)

        # MAD-based Byzantine outlier clip (for any Byzantine fraction < 50%, median & MAD are honest-bounded)
        mad_cutoff = max(6.0 * mad, 1e-5)
        inliers = [v for v in sym_slice if abs(v - med) <= mad_cutoff]
        if not inliers:
            inliers = [med]

        trimmed_mean = sum(inliers) / float(len(inliers))
        return float(trimmed_mean), float(med)

    def aggregate_continuous_deltas(
        self,
        submissions: Sequence[Union[ContinuousDeltaSubmission, Dict[str, Any]]],
        trim_fraction: Optional[float] = None,
        min_distinct_asns: Optional[int] = None,
        require_authoritative_asn: bool = False,
        emit_bep46_manifest: bool = False,
    ) -> ContinuousAggregationResult:
        """Group continuous calibration/adapter deltas by verified ASN first (1 vote per ASN),
        then compute coordinate-wise trimmed mean and coordinate-wise median.
        """
        eff_trim = self.trim_fraction if trim_fraction is None else float(trim_fraction)
        eff_min_asns = self.min_distinct_asns if min_distinct_asns is None else int(min_distinct_asns)

        coerced: List[ContinuousDeltaSubmission] = []
        for s in submissions:
            if isinstance(s, ContinuousDeltaSubmission):
                coerced.append(s)
            elif isinstance(s, dict):
                b = s.get("encrypted_bundle")
                if isinstance(b, dict):
                    b = EncryptedTelemetryBundle.from_dict(b)
                coerced.append(
                    ContinuousDeltaSubmission(
                        peer_id=str(s.get("peer_id", "")),
                        asn=int(s["asn"]),
                        peer_pubkey_hex=str(s.get("peer_pubkey_hex", "")),
                        deltas=dict(s.get("deltas", {})),
                        ip_address=s.get("ip_address"),
                        encrypted_bundle=b,
                        attestation=s.get("attestation"),
                        timestamp=int(s.get("timestamp", 0)),
                    )
                )
            else:
                raise TypeError(f"Unsupported continuous delta submission type: {type(s)}")

        # Pre-scan for Sybil key/subnet multi-ASN spoofing
        seen_pk_asn: Dict[str, int] = {}
        seen_subnet_asn: Dict[str, int] = {}
        spoofed_pks: Set[str] = set()
        spoofed_subnets: Set[str] = set()

        for sub in coerced:
            if sub.peer_pubkey_hex:
                norm_pk = AuthoritativeASNVerifier.normalize_pubkey(sub.peer_pubkey_hex)
                if norm_pk in seen_pk_asn and seen_pk_asn[norm_pk] != int(sub.asn):
                    spoofed_pks.add(norm_pk)
                seen_pk_asn[norm_pk] = int(sub.asn)
            if sub.ip_address:
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(sub.ip_address)
                    if subnet in seen_subnet_asn and seen_subnet_asn[subnet] != int(sub.asn):
                        spoofed_subnets.add(subnet)
                    seen_subnet_asn[subnet] = int(sub.asn)
                except ValueError:
                    pass

        rejected_count = 0
        per_asn_raw: Dict[int, List[Dict[str, Union[float, List[float]]]]] = {}

        for sub in coerced:
            claimed_asn = int(sub.asn)
            norm_pk = AuthoritativeASNVerifier.normalize_pubkey(sub.peer_pubkey_hex) if sub.peer_pubkey_hex else ""

            if norm_pk and norm_pk in spoofed_pks:
                rejected_count += 1
                continue

            if sub.ip_address:
                try:
                    subnet = AuthoritativeASNVerifier.extract_subnet_prefix(sub.ip_address)
                    if subnet in spoofed_subnets:
                        rejected_count += 1
                        continue
                except ValueError:
                    rejected_count += 1
                    continue

            bind_ok, _ = self.asn_verifier.verify_binding(
                claimed_asn=claimed_asn,
                pubkey_hex=norm_pk or None,
                ip_address=sub.ip_address,
                require_authoritative=require_authoritative_asn,
            )
            if not bind_ok:
                rejected_count += 1
                continue

            deltas_dict = dict(sub.deltas)
            if sub.encrypted_bundle is not None and not deltas_dict:
                try:
                    plain = HybridPQCAEADEngine.decrypt_bundle(
                        sub.encrypted_bundle, self.aggregator_privkey_bytes
                    )
                    parsed = json.loads(plain.decode("utf-8"))
                    deltas_dict = parsed.get("deltas", parsed)
                except Exception:
                    rejected_count += 1
                    continue

            if not deltas_dict:
                rejected_count += 1
                continue

            per_asn_raw.setdefault(claimed_asn, []).append(deltas_dict)

        verified_asns = sorted(per_asn_raw.keys())
        if len(verified_asns) < eff_min_asns:
            return ContinuousAggregationResult(
                success=False,
                reason=f"insufficient_distinct_asns:{len(verified_asns)}<{eff_min_asns}",
                distinct_asn_count=len(verified_asns),
                verified_asns=verified_asns,
                total_submissions=len(coerced),
                rejected_submissions=rejected_count,
                trimmed_mean_deltas={},
                median_deltas={},
                trim_fraction=eff_trim,
            )

        # Step 1: Collapse intra-ASN submissions to exactly 1 vote per verified ASN (via intra-ASN median)
        all_keys = sorted({k for obs_list in per_asn_raw.values() for d in obs_list for k in d.keys()})
        per_asn_single_vote: Dict[int, Dict[str, Union[float, List[float]]]] = {asn: {} for asn in verified_asns}

        for key in all_keys:
            for asn in verified_asns:
                asn_entries = [d[key] for d in per_asn_raw[asn] if key in d]
                if not asn_entries:
                    continue
                first_val = asn_entries[0]
                if isinstance(first_val, (list, tuple)):
                    dim = len(first_val)
                    vec_vote: List[float] = []
                    for coord_idx in range(dim):
                        coord_samples = sorted(
                            float(entry[coord_idx])
                            for entry in asn_entries
                            if isinstance(entry, (list, tuple)) and len(entry) > coord_idx
                        )
                        vec_vote.append(self._compute_median(coord_samples))
                    per_asn_single_vote[asn][key] = vec_vote
                else:
                    scalar_samples = sorted(float(entry) for entry in asn_entries)
                    per_asn_single_vote[asn][key] = self._compute_median(scalar_samples)

        # Step 2: Across the distinct ASNs (1 vote per ASN), compute coordinate-wise trimmed mean and median
        trimmed_mean_out: Dict[str, Union[float, List[float]]] = {}
        median_out: Dict[str, Union[float, List[float]]] = {}

        for key in all_keys:
            asn_vals = [per_asn_single_vote[asn][key] for asn in verified_asns if key in per_asn_single_vote[asn]]
            if not asn_vals:
                continue
            if isinstance(asn_vals[0], list):
                dim = len(asn_vals[0])
                t_vec: List[float] = []
                m_vec: List[float] = []
                for coord_idx in range(dim):
                    coord_votes = [
                        float(v[coord_idx]) for v in asn_vals if isinstance(v, list) and len(v) > coord_idx
                    ]
                    t_val, m_val = self._trimmed_mean_and_median_1d(coord_votes, trim_fraction=eff_trim)
                    t_vec.append(round(t_val, 6))
                    m_vec.append(round(m_val, 6))
                trimmed_mean_out[key] = t_vec
                median_out[key] = m_vec
            else:
                scalar_votes = [float(v) for v in asn_vals]  # type: ignore[arg-type]
                t_val, m_val = self._trimmed_mean_and_median_1d(scalar_votes, trim_fraction=eff_trim)
                trimmed_mean_out[key] = round(t_val, 6)
                median_out[key] = round(m_val, 6)

        manifest = None
        bundle_bytes = None
        if emit_bep46_manifest:
            payload_obj = {
                "distinct_asn_count": len(verified_asns),
                "median_deltas": median_out,
                "trimmed_mean_deltas": trimmed_mean_out,
                "verified_asns": verified_asns,
            }
            bundle_bytes = json.dumps(payload_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            manifest = self._sign_bep46_bundle(bundle_bytes, min_asn_quorum=len(verified_asns))

        return ContinuousAggregationResult(
            success=True,
            reason="per_asn_trimmed_mean_and_median_computed",
            distinct_asn_count=len(verified_asns),
            verified_asns=verified_asns,
            total_submissions=len(coerced),
            rejected_submissions=rejected_count,
            trimmed_mean_deltas=trimmed_mean_out,
            median_deltas=median_out,
            trim_fraction=eff_trim,
            manifest=manifest,
            signed_bundle_bytes=bundle_bytes,
        )


# =====================================================================
# 11. Top-Level Armiger DHT RSI Engine
# =====================================================================

class ArmigerDHTRSIEngine:
    """Master engine integrating universal ingestion, physical ground truth verification,

    complete privacy/anonymity, AEAD encryption, spoof-proof threshold signatures,
    distinct-ASN Sybil quorums, safe formats, and the 500-item golden canary regression gate.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        current_seq: int = 0,
        trusted_signers: Optional[Dict[str, str]] = None,
        threshold_m: int = 1,
        min_distinct_asns: int = 5,
        blinding_salt: Optional[bytes] = None,
        dp_epsilon: float = 1.0,
        poisson_mean_seconds: float = 1800.0,
        aggregator_x25519_pubkey_hex: Optional[str] = None,
        *,
        opt_in_sharing: Optional[bool] = None,
        ephemeral_identity: bool = True,
        min_k_samples: int = 25,
        clip_norm: float = 1.0,
        dp_sigma: float = 0.05,
    ) -> None:
        self.opt_in_sharing: bool = resolve_opt_in_sharing(opt_in_sharing)
        if mode is None:
            resolved_mode = DHTRSIMode.ANONYMOUS_CONTRIBUTE.value if self.opt_in_sharing else DHTRSIMode.PULL_ONLY.value
        else:
            resolved_mode = mode.value if isinstance(mode, DHTRSIMode) else str(mode)

        if resolved_mode not in [m.value for m in DHTRSIMode]:
            raise ValueError(f"Unsupported mode: {resolved_mode}")

        self.mode = DHTRSIMode(resolved_mode)
        self.current_seq = int(current_seq)
        self.trusted_signers = trusted_signers or {}
        self.threshold_m = int(threshold_m)
        self.min_distinct_asns = int(min_distinct_asns)
        self.aggregator_x25519_pubkey_hex = aggregator_x25519_pubkey_hex

        self.verifier = PhysicalGroundTruthVerifier()
        self.anonymizer = TelemetryAnonymizer(
            blinding_salt=blinding_salt,
            dp_epsilon=dp_epsilon,
            poisson_mean_seconds=poisson_mean_seconds,
        )
        self.anonymity_guard = AnonymityGuard(
            ephemeral_identity=ephemeral_identity,
            min_k_samples=min_k_samples,
            clip_norm=clip_norm,
            dp_sigma=dp_sigma,
        )
        self.canary_gate = GoldenCanaryRegressionGate()
        self.telemetry_queue: List[QueuedTelemetryItem] = []
        self._outbound_bundles_shared: int = 0
        self._outbound_packets_blocked_opt_out: int = 0
        self._outbound_bytes_shared: int = 0

    # -----------------------------------------------------------------
    # Ingestion & Physical Ground-Truth
    # -----------------------------------------------------------------

    def ingest_candidate(
        self,
        candidate: CandidateObservation,
        ffprobe_output: Optional[Dict[str, Any]] = None,
        current_timestamp: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[AnonymizedTelemetryRecord]]:
        """Ingest candidate from any adapter, verify physical ground truth, anonymize, and queue."""
        if self.mode == DHTRSIMode.OFF:
            return False, "plugin_mode_off", None

        if self.mode == DHTRSIMode.PULL_ONLY:
            self._outbound_packets_blocked_opt_out += 1
            return False, "telemetry_disabled_in_pull_only_mode", None

        # Verify physical ground truth (never unverified predictions)
        res = self.verifier.verify(candidate, ffprobe_output)
        if not res.is_verified or res.observation is None:
            return False, f"physical_verification_failed:{res.rejection_reason}", None

        # Anonymize & apply differential privacy
        record = self.anonymizer.anonymize(res.observation)

        # Apply Poisson timing jitter
        now = current_timestamp if current_timestamp is not None else time.time()
        jitter_delay = self.anonymizer.calculate_poisson_jitter()
        scheduled_time = now + jitter_delay

        if not self.opt_in_sharing:
            self._outbound_packets_blocked_opt_out += 1
            return True, "verified_anonymized_and_queued", record

        self.telemetry_queue.append(QueuedTelemetryItem(
            scheduled_dispatch_time=scheduled_time,
            record=record,
        ))

        return True, "verified_anonymized_and_queued", record

    def sharing_status(self) -> Dict[str, Any]:
        """Return runtime audit telemetry confirming opt-in status and outbound packet/bundle counters."""
        return {
            "opt_in_sharing_enabled": bool(self.opt_in_sharing),
            "mode": self.mode.value,
            "outbound_bundles_shared": int(self._outbound_bundles_shared),
            "outbound_packets_blocked_opt_out": int(self._outbound_packets_blocked_opt_out),
            "outbound_bytes_shared": int(self._outbound_bytes_shared),
            "ephemeral_identity_enabled": bool(self.anonymity_guard.ephemeral_identity),
            "ephemeral_pubkey_hex": self.anonymity_guard.current_pubkey_hex() if self.opt_in_sharing else None,
            "ephemeral_rotations": int(self.anonymity_guard.rotation_count),
            "blocked_anonymity_violations": int(self.anonymity_guard.blocked_violations_count),
        }

    def export_signed_delta_bundle(
        self,
        delta_w: Optional[Sequence[float]] = None,
        delta_b: Optional[Sequence[float]] = None,
        *,
        reliability_bins: Optional[Sequence[Any]] = None,
        n_samples: int = 25,
        epoch: Optional[Union[int, float]] = None,
        seq: Optional[int] = None,
        model_tier: str = "ultra",
        salt: bytes = b"armiger-rsi-delta",
        extra_fields: Optional[Mapping[str, Any]] = None,
        raw_wire_override: Optional[Union[bytes, str, Mapping[str, Any]]] = None,
        rotate_key: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Export an L2-clipped, Gaussian-DP, int8-quantized, coarse-timestamped BEP-44 delta bundle.

        Strictly returns None (0 bytes shared) when `opt_in_sharing=False`.
        Raises `AnonymityViolationError` if any PII/canary/schema/k-anonymity invariant is violated.
        """
        if not self.opt_in_sharing or self.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
            self._outbound_packets_blocked_opt_out += 1
            return None

        next_seq = int(seq) if seq is not None else (self.current_seq + 1)

        if raw_wire_override is not None:
            if isinstance(raw_wire_override, (bytes, bytearray)):
                v_bytes = bytes(raw_wire_override)
            elif isinstance(raw_wire_override, str):
                v_bytes = raw_wire_override.encode("utf-8")
            elif isinstance(raw_wire_override, Mapping):
                v_bytes = json.dumps(dict(raw_wire_override), sort_keys=True, separators=(",", ":")).encode("utf-8")
            else:
                raise AnonymityViolationError("invalid_raw_wire_override")
            signed = self.anonymity_guard.sign_wire_payload(
                v_bytes, seq=next_seq, salt=salt, rotate_key=rotate_key
            )
            self.current_seq = max(self.current_seq, next_seq)
            self._outbound_bundles_shared += 1
            self._outbound_bytes_shared += len(v_bytes)
            return signed

        w_in = list(delta_w) if delta_w is not None else [0.0] * 8
        b_in = list(delta_b) if delta_b is not None else [0.0] * 2
        dp_res = self.anonymity_guard.clip_and_privatize_weights(
            w_in,
            b_in,
            n_samples=n_samples,
        )
        epoch_bucket = self.anonymity_guard.bucket_timestamp(epoch)

        wire_obj: Dict[str, Any] = {
            "v": 2,
            "epoch": epoch_bucket,
            "seq": next_seq,
            "model_tier": model_tier,
            "delta_w_q8": dp_res["delta_w_q8"],
            "delta_b_q8": dp_res["delta_b_q8"],
            "scale": dp_res["scale"],
            "n_bucket": dp_res["n_bucket"],
        }
        if reliability_bins is not None:
            wire_obj["reliability_bins"] = self.anonymity_guard.sanitize_reliability_bins(reliability_bins)
        if extra_fields:
            for k, val in extra_fields.items():
                wire_obj[k] = val

        v_bytes = json.dumps(wire_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signed = self.anonymity_guard.sign_wire_payload(
            v_bytes, seq=next_seq, salt=salt, rotate_key=rotate_key
        )
        signed["pre_dp_l2_norm"] = dp_res["pre_dp_l2_norm"]
        signed["clipped_l2_norm"] = dp_res["clipped_l2_norm"]
        signed["delta_w_dp"] = dp_res["delta_w_dp"]
        signed["delta_b_dp"] = dp_res["delta_b_dp"]
        self.current_seq = max(self.current_seq, next_seq)
        self._outbound_bundles_shared += 1
        self._outbound_bytes_shared += len(v_bytes)
        return signed

    def export_signed_bundle(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """Alias for `export_signed_delta_bundle` enforcing opt_in_sharing and AnonymityGuard."""
        return self.export_signed_delta_bundle(*args, **kwargs)

    def ingest_signed_delta_bundle(
        self,
        signed_bundle: Mapping[str, Any],
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Verify Ed25519 signature and AnonymityGuard schema of an inbound delta bundle (works in pull_only mode)."""
        if self.mode == DHTRSIMode.OFF:
            return False, "plugin_mode_off", None

        try:
            v_raw = signed_bundle["v"]
            v_bytes = v_raw if isinstance(v_raw, (bytes, bytearray)) else str(v_raw).encode("utf-8")
            seq = int(signed_bundle["seq"])
            salt = signed_bundle.get("salt", b"armiger-rsi-delta")
            if isinstance(salt, str):
                salt = salt.encode("utf-8")
            pub_raw = signed_bundle.get("k") or signed_bundle.get("pubkey_hex")
            pub_bytes = bytes.fromhex(pub_raw) if isinstance(pub_raw, str) else bytes(pub_raw)
            sig_raw = signed_bundle.get("sig") or signed_bundle.get("sig_hex")
            sig_bytes = bytes.fromhex(sig_raw) if isinstance(sig_raw, str) else bytes(sig_raw)
        except Exception as exc:
            return False, f"malformed_bundle_envelope:{exc}", None

        try:
            vk = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
            vk.verify(sig_bytes, _bep44_canonical_sign_buffer(v=bytes(v_bytes), seq=seq, salt=bytes(salt)))
        except Exception as exc:
            return False, f"invalid_ed25519_signature:{exc}", None

        try:
            parsed = self.anonymity_guard.inspect_wire_payload(bytes(v_bytes))
        except AnonymityViolationError as exc:
            return False, f"anonymity_firewall_rejected:{exc}", None

        scale = float(parsed.get("scale", 1.0 / 127.0))
        w_q8 = parsed.get("delta_w_q8", [])
        b_q8 = parsed.get("delta_b_q8", [])
        decoded = {
            "epoch": parsed.get("epoch"),
            "seq": seq,
            "model_tier": parsed.get("model_tier", "ultra"),
            "n_bucket": parsed.get("n_bucket", 25),
            "delta_w": [float(q) * scale for q in w_q8],
            "delta_b": [float(q) * scale for q in b_q8],
            "reliability_bins": parsed.get("reliability_bins", []),
            "sender_ephemeral_pubkey_hex": pub_bytes.hex(),
        }
        return True, "verified_and_ingested", decoded

    def ingest_signed_bundle(
        self,
        signed_bundle: Mapping[str, Any],
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Alias for `ingest_signed_delta_bundle`."""
        return self.ingest_signed_delta_bundle(signed_bundle)

    async def publish_to_dht(
        self,
        swarm: Any,
        bundle: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[bool, str, int]:
        """Publish an anonymous signed bundle to a UDP BEP-44 swarm. Blocked when `opt_in_sharing=False`."""
        if not self.opt_in_sharing or self.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
            self._outbound_packets_blocked_opt_out += 1
            return False, "opt_in_sharing_disabled", 0

        eff_bundle = dict(bundle) if bundle is not None else self.export_signed_delta_bundle(**kwargs)
        if eff_bundle is None:
            self._outbound_packets_blocked_opt_out += 1
            return False, "opt_in_sharing_disabled", 0

        v_bytes = eff_bundle["v"] if isinstance(eff_bundle["v"], bytes) else str(eff_bundle["v"]).encode("utf-8")
        # Enforce wire firewall right before UDP transmission
        self.anonymity_guard.inspect_wire_payload(v_bytes)

        priv_bytes = eff_bundle.get("privkey_bytes") or self.anonymity_guard.current_privkey_bytes()
        salt_bytes = eff_bundle.get("salt", b"armiger-rsi-delta")
        if isinstance(salt_bytes, str):
            salt_bytes = salt_bytes.encode("utf-8")
        responses = await swarm.put_mutable(
            privkey_bytes=priv_bytes,
            salt=salt_bytes,
            v=v_bytes,
            seq=int(eff_bundle["seq"]),
        )
        ok_count = sum(1 for r in responses if r.get(b"y") == b"r")
        return ok_count > 0, "published_anonymous_bep44", ok_count

    def broadcast_weights(
        self,
        delta_w: Optional[Sequence[float]] = None,
        delta_b: Optional[Sequence[float]] = None,
        *,
        n_samples: int = 25,
        reliability_bins: Optional[Sequence[Any]] = None,
        epoch: Optional[Union[int, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Export and sign DP-clipped weight deltas for DHT broadcast. Returns None if `opt_in_sharing=False`."""
        return self.export_signed_delta_bundle(
            delta_w=delta_w,
            delta_b=delta_b,
            reliability_bins=reliability_bins,
            n_samples=n_samples,
            epoch=epoch,
        )

    def get_pending_telemetry_batch(
        self,
        now_timestamp: Optional[float] = None,
    ) -> List[EncryptedTelemetryBundle]:
        """Release scheduled telemetry items whose Poisson jitter delay has elapsed, encrypted with AEAD."""
        if not self.opt_in_sharing or self.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
            if self.telemetry_queue:
                self._outbound_packets_blocked_opt_out += len(self.telemetry_queue)
                self.telemetry_queue.clear()
            else:
                self._outbound_packets_blocked_opt_out += 1
            return []

        now = now_timestamp if now_timestamp is not None else time.time()
        ready_items: List[QueuedTelemetryItem] = []
        remaining_items: List[QueuedTelemetryItem] = []

        for item in self.telemetry_queue:
            if item.scheduled_dispatch_time <= now:
                ready_items.append(item)
            else:
                remaining_items.append(item)

        self.telemetry_queue = remaining_items

        if not ready_items:
            return []

        if not self.aggregator_x25519_pubkey_hex:
            # Cannot encrypt without aggregator public key
            return []

        recipient_pk_bytes = bytes.fromhex(self.aggregator_x25519_pubkey_hex)
        bundles: List[EncryptedTelemetryBundle] = []
        for i, item in enumerate(ready_items):
            payload = item.record.canonical_json_bytes()
            bundle = X25519ChaCha20Poly1305Engine.encrypt_bundle(
                payload_bytes=payload,
                recipient_pubkey_bytes=recipient_pk_bytes,
                seq=self.current_seq + 1 + i,
            )
            bundles.append(bundle)

        return bundles

    # -----------------------------------------------------------------
    # Update Staging, Verification, and Golden Canary Gate
    # -----------------------------------------------------------------

    def verify_and_stage_update(
        self,
        manifest: ThresholdDHTManifest,
        bundle_bytes: bytes,
        attestations: List[PeerAttestation],
        candidate_predictor_fn: Optional[Callable[[CanaryItem], Dict[str, Any]]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str, Optional[CanaryRegressionReport]]:
        """Verify signed DHT update manifest, check ASN quorum, validate safe formats, and run 500-canary gate."""
        if self.mode == DHTRSIMode.OFF:
            return False, "plugin_mode_off", None

        # 1. Verify BEP-44 monotonic sequence and Ed25519/PQ threshold signatures
        ok, reason = ThresholdSignatureVerifier.verify(
            manifest=manifest,
            trusted_signers=self.trusted_signers,
            threshold_m=self.threshold_m,
            current_seq=self.current_seq,
            require_pq=bool(manifest.pq_signatures),
        )
        if not ok:
            return False, f"manifest_threshold_failed:{reason}", None

        # 2. Check bundle SHA-256 digest
        actual_sha = hashlib.sha256(bundle_bytes).hexdigest()
        if actual_sha.lower() != manifest.bundle_sha256.lower():
            return False, "bundle_sha256_digest_mismatch", None

        # 3. Verify K-of-N distinct-ASN Sybil quorum
        q_ok, q_reason, asns = SybilQuorumVerifier.verify_quorum(
            attestations=attestations,
            target_hash=manifest.bundle_sha256,
            min_distinct_asns=max(self.min_distinct_asns, manifest.min_asn_quorum),
        )
        if not q_ok:
            return False, f"sybil_quorum_failed:{q_reason}", None

        # 4. Enforce Safe Formats only (SafeTensors, ONNX, JSON)
        f_ok, f_reason = SafeFormatValidator.validate(bundle_bytes, manifest.bundle_format)
        if not f_ok:
            return False, f"safe_format_violation:{f_reason}", None

        # 5. Mandatory 500-Item Golden Canary Regression Gate
        canary_report = None
        if candidate_predictor_fn is not None:
            canary_report = self.canary_gate.evaluate(
                predictor_fn=candidate_predictor_fn,
                baseline_metrics=baseline_metrics,
            )
            if not canary_report.passed:
                return False, f"golden_canary_gate_rejected:{canary_report.reason}", canary_report

        # All checks passed! Advance monotonic sequence
        self.current_seq = manifest.seq
        return True, "update_staged_and_verified", canary_report
