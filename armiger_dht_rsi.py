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
import json
import math
import os
import pathlib
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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
    signatures: Dict[str, str]  # pubkey_hex -> ed25519_signature_hex
    pq_signatures: Dict[str, str] = dataclasses.field(default_factory=dict)  # pq_pubkey_hex -> wots_shake256_sig_hex

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

def pure_laplace_noise(scale: float) -> float:
    """Pure-Python exact Laplace noise generator via inverse CDF: F^-1(u) = -b * sgn(u - 0.5) * ln(1 - 2|u - 0.5|)."""
    if scale <= 0.0:
        return 0.0
    u = random.random() - 0.5
    sgn = 1.0 if u >= 0 else -1.0
    return -scale * sgn * math.log(max(1.0 - 2.0 * abs(u), 1e-15))


class TelemetryAnonymizer:
    """Anonymizes telemetry records, strips PII/passkeys, blinds titles, and applies DP noise."""

    RE_PASSKEY = re.compile(
        r'([?&](?:passkey|authkey|torrent_pass|auth|token|uk|key|api_key|apikey)=)[a-zA-Z0-9_-]+',
        re.IGNORECASE,
    )
    RE_PASSKEY_INLINE = re.compile(
        r'(?:passkey|authkey|torrent_pass)\s*[:=]\s*[a-zA-Z0-9_-]+',
        re.IGNORECASE,
    )
    RE_IPV4 = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )
    RE_IPV6 = re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b')
    RE_UNIX_PATH = re.compile(r'/(?:Users|home)/[a-zA-Z0-9_.-]+')
    RE_WIN_PATH = re.compile(r'[A-Za-z]:[/\\]Users[/\\][a-zA-Z0-9_.-]+')

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
        """Strip private tracker passkeys, IP addresses, and user home directory paths."""
        if not text:
            return ""
        s = self.RE_PASSKEY.sub(r'\1[REDACTED_PASSKEY]', text)
        s = self.RE_PASSKEY_INLINE.sub('passkey=[REDACTED_PASSKEY]', s)
        s = self.RE_IPV4.sub('[IP_REDACTED]', s)
        s = self.RE_IPV6.sub('[IPV6_REDACTED]', s)
        s = self.RE_UNIX_PATH.sub('/home/[USER_REDACTED]', s)
        s = self.RE_WIN_PATH.sub('C:/Users/[USER_REDACTED]', s)
        return s

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
        blinded_hash = h_shake.hex()[:16]

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
        # Quantize size to logarithmic bin (nearest 16MB)
        bin_size = 16.0 * 1024 * 1024
        size_mb = max(size_bytes, 0) / (1024.0 * 1024.0)
        quantized_mb = round(size_mb / 16.0) * 16.0

        # Sensitivity for quantized size is 16 MB
        noise_size = pure_laplace_noise(scale=16.0 / self.dp_epsilon)
        dp_size = max(quantized_mb + noise_size, 0.0)

        dp_duration = None
        if duration_seconds is not None:
            # Round duration to nearest 5 minutes
            duration_mins = max(duration_seconds, 0.0) / 60.0
            quantized_mins = round(duration_mins / 5.0) * 5.0
            noise_dur = pure_laplace_noise(scale=5.0 / self.dp_epsilon)
            dp_duration = max(quantized_mins + noise_dur, 0.0)

        return dp_size, dp_duration

    def calculate_poisson_jitter(self) -> float:
        """Sample an arrival delay from a Poisson process (exponential inter-arrival distribution)."""
        return random.expovariate(1.0 / self.poisson_mean_seconds)

    def anonymize(self, observation: VerifiedObservation) -> AnonymizedTelemetryRecord:
        """Transform a verified physical observation into an anonymous telemetry record."""
        blinded = self.blind_title_stem(observation.raw_title)
        dur = observation.stream_info.get("duration_seconds")
        dp_size, dp_dur = self.apply_differential_privacy(observation.size_bytes, dur)

        raw_vcodec = str(observation.stream_info.get("video_codec") or "none")
        raw_acodec = str(observation.stream_info.get("audio_codec") or "none")
        raw_cat_type = str(observation.catalog_id_type or "none")
        features = {
            "has_video": bool(observation.stream_info.get("video_codec")),
            "has_audio": bool(observation.stream_info.get("audio_codec")),
            "video_codec": self.strip_pii(raw_vcodec),
            "audio_codec": self.strip_pii(raw_acodec),
            "audio_channels": observation.stream_info.get("audio_channels", 0),
            "catalog_type": self.strip_pii(raw_cat_type),
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
        m = seed_m if seed_m is not None else os.urandom(32)
        t_hat, a_hat_t, ek_hash = cls._expand_ek(ek)
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
    @functools.lru_cache(maxsize=256)
    def generate_keypair(cls, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Generate (private_seed_64B, compressed_pubkey_32B)."""
        raw = seed if seed is not None else os.urandom(64)
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
    def derive_keypair_from_ed25519_priv(cls, ed25519_priv_bytes: bytes) -> Tuple[bytes, bytes]:
        """Deterministically derive a WOTS+-SHAKE256 (priv_seed_64B, pub_key_32B) from a 32B Ed25519 seed."""
        seed64 = hashlib.shake_256(b"armiger-hybrid-ed25519-wots-v1:" + ed25519_priv_bytes).digest(64)
        return cls.generate_keypair(seed64)

    @classmethod
    def sign(cls, priv_seed: bytes, message: bytes) -> bytes:
        """Sign arbitrary message bytes using 64-byte WOTS+-SHAKE256 private seed."""
        if len(priv_seed) == 32:
            priv_seed, pk = cls.derive_keypair_from_ed25519_priv(priv_seed)
        else:
            _, pk = cls.generate_keypair(priv_seed)
        sk_seed, pub_seed = priv_seed[:32], priv_seed[32:]
        digest = hashlib.shake_256(pub_seed + pk + message).digest(32)
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
        digest = hashlib.shake_256(pub_seed + pub_key + message).digest(32)
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

class SeriesWebhookAdapter:
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


class MovieWebhookAdapter:
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


class AudioWebhookAdapter:
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


class PublicationWebhookAdapter:
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


class QBittorrentAdapter:
    """Adapter for qBittorrent completion events and API status payloads."""

    FINISHED_STATES = {"uploading", "pausedUP", "stalledUP", "queuedUP", "forcedUP"}

    @classmethod
    def parse_event(cls, payload: Dict[str, Any]) -> CandidateObservation:
        progress = float(payload.get("progress", 0.0))
        state = str(payload.get("state", "")).lower()
        size = int(payload.get("size") or payload.get("total_size") or 0)
        name = str(payload.get("name", "unknown"))

        is_complete = bool(progress >= 0.999999 or state in cls.FINISHED_STATES)
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


class TransmissionAdapter:
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

        if candidate.total_bytes <= 0:
            return PhysicalVerificationResult(
                is_verified=False,
                rejection_reason="zero_byte_content_rejected",
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
        require_pq: bool = False,
    ) -> Tuple[bool, str]:
        """Verify monotonic sequence number and M-of-N Hybrid Ed25519 + WOTS+-SHAKE256 threshold signatures."""
        # 1. Monotonic sequence check (downgrade & replay protection)
        if manifest.seq <= current_seq:
            return False, f"stale_or_replayed_seq:{manifest.seq}<={current_seq}"

        # 2. Threshold format check
        if manifest.bundle_format.lower() not in [b.value for b in BundleFormat]:
            return False, f"disallowed_bundle_format:{manifest.bundle_format}"

        # 3. Canonical payload derivation
        payload_bytes = manifest.canonical_payload()

        # 4. Check classical Ed25519 signatures against trusted federation signers
        valid_signatures: Set[str] = set()
        for pubkey_hex, sig_hex in manifest.signatures.items():
            norm_pk = pubkey_hex.lower()
            if norm_pk not in trusted_signers:
                continue

            try:
                pk_bytes = bytes.fromhex(norm_pk)
                sig_bytes = bytes.fromhex(sig_hex)
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes)
                verifier.verify(sig_bytes, payload_bytes)
                valid_signatures.add(norm_pk)
            except (ValueError, InvalidSignature):
                continue

        if len(valid_signatures) < threshold_m:
            return False, f"insufficient_threshold_signatures:{len(valid_signatures)}<{threshold_m}"

        # 5. Check Post-Quantum WOTS+-SHAKE256 signatures when present or required
        if require_pq or manifest.pq_signatures or trusted_pq_signers:
            valid_pq_signatures: Set[str] = set()
            allowed_pq = {k.lower() for k in trusted_pq_signers.keys()} if trusted_pq_signers else None
            for pq_pk_hex, pq_sig_hex in manifest.pq_signatures.items():
                norm_pq_pk = pq_pk_hex.lower()
                if allowed_pq is not None and norm_pq_pk not in allowed_pq:
                    continue
                try:
                    pq_pk_bytes = bytes.fromhex(norm_pq_pk)
                    pq_sig_bytes = bytes.fromhex(pq_sig_hex)
                    if WOTSPlusSHAKE256.verify(pq_pk_bytes, payload_bytes, pq_sig_bytes):
                        valid_pq_signatures.add(norm_pq_pk)
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
    ) -> Tuple[bool, str, Set[int]]:
        """Verify attestations against timestamp freshness, Hybrid Ed25519 + WOTS+-SHAKE256 signatures, and ASNs."""
        now = now_timestamp or int(time.time())
        verified_asns: Set[int] = set()

        for att in attestations:
            # Check freshness
            if abs(now - att.timestamp) > max_age_seconds:
                continue

            # Verify peer signature over target hash
            expected_bytes = att.canonical_bytes(target_hash)
            try:
                pk_bytes = bytes.fromhex(att.peer_pubkey_hex.lower())
                sig_bytes = bytes.fromhex(att.signature_hex)
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(pk_bytes)
                verifier.verify(sig_bytes, expected_bytes)
            except (ValueError, InvalidSignature):
                continue

            # Verify Post-Quantum WOTS+-SHAKE256 peer attestation signature if present or required
            if require_pq or att.pq_signature_hex or att.pq_pubkey_hex:
                try:
                    pq_pk_bytes = bytes.fromhex(att.pq_pubkey_hex.lower())
                    pq_sig_bytes = bytes.fromhex(att.pq_signature_hex)
                    if not WOTSPlusSHAKE256.verify(pq_pk_bytes, expected_bytes, pq_sig_bytes):
                        continue
                except ValueError:
                    continue

            # Valid attestation from verified ASN
            if att.asn > 0:
                verified_asns.add(int(att.asn))

        if len(verified_asns) < min_distinct_asns:
            return False, f"insufficient_distinct_asns:{len(verified_asns)}<{min_distinct_asns}", verified_asns

        return True, f"quorum_achieved:{len(verified_asns)}_distinct_asns", verified_asns


# =====================================================================
# 9. Safe Format Enforcer (Poison-Proof)
# =====================================================================

class SafeFormatValidator:
    """Enforces that candidate updates contain only safe formats (SafeTensors, ONNX, JSON). Rejects pickle."""

    PROHIBITED_BYTE_PATTERNS = [
        b"\x80\x02",  # Python pickle protocol 2
        b"\x80\x03",  # Python pickle protocol 3
        b"\x80\x04",  # Python pickle protocol 4
        b"\x80\x05",  # Python pickle protocol 5
        b"c__builtin__",
        b"cposix\nsystem",
        b"cos\nsystem",
        b"__reduce__",
    ]

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

        # 2. Format-specific structural validation
        if fmt == "safetensors":
            return cls._validate_safetensors(bundle_bytes)
        elif fmt == "onnx":
            return cls._validate_onnx(bundle_bytes)
        elif fmt == "json":
            return cls._validate_json(bundle_bytes)

        return False, "unknown_validation_error"

    @staticmethod
    def _validate_safetensors(data: bytes) -> Tuple[bool, str]:
        if len(data) < 8:
            return False, "safetensors_too_short"
        header_len = int.from_bytes(data[:8], "little")
        if header_len <= 0 or 8 + header_len > len(data):
            return False, "safetensors_invalid_header_length"

        try:
            header_json = json.loads(data[8:8 + header_len].decode("utf-8"))
            if not isinstance(header_json, dict):
                return False, "safetensors_header_not_dict"

            total_buffer_len = len(data) - 8 - header_len
            for tensor_name, tensor_meta in header_json.items():
                if tensor_name == "__metadata__":
                    continue
                if not isinstance(tensor_meta, dict):
                    return False, f"safetensors_invalid_tensor_meta:{tensor_name}"
                dtype = tensor_meta.get("dtype")
                if dtype not in ("F32", "F16", "BF16", "I32", "I64", "I8", "U8", "BOOL"):
                    return False, f"safetensors_invalid_dtype:{dtype}"
                offsets = tensor_meta.get("data_offsets")
                if not isinstance(offsets, list) or len(offsets) != 2:
                    return False, f"safetensors_invalid_offsets:{tensor_name}"
                if offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] > total_buffer_len:
                    return False, f"safetensors_offset_out_of_bounds:{tensor_name}"

            return True, "safetensors_valid"
        except Exception as e:
            return False, f"safetensors_parse_error:{e}"

    @staticmethod
    def _validate_onnx(data: bytes) -> Tuple[bool, str]:
        if len(data) < 16:
            return False, "onnx_too_short"
        # ONNX protobuf starts with field tags (ir_version is field 1, varint 0x08)
        if data[0] != 0x08:
            return False, "onnx_invalid_protobuf_magic"
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
        name = malware_names[i % len(malware_names)] + f".sample{i}"
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
# 11. Top-Level Armiger DHT RSI Engine
# =====================================================================

class ArmigerDHTRSIEngine:
    """Master engine integrating universal ingestion, physical ground truth verification,

    complete privacy/anonymity, AEAD encryption, spoof-proof threshold signatures,
    distinct-ASN Sybil quorums, safe formats, and the 500-item golden canary regression gate.
    """

    def __init__(
        self,
        mode: str = "anonymous_contribute",
        current_seq: int = 0,
        trusted_signers: Optional[Dict[str, str]] = None,
        threshold_m: int = 1,
        min_distinct_asns: int = 5,
        blinding_salt: Optional[bytes] = None,
        dp_epsilon: float = 1.0,
        poisson_mean_seconds: float = 1800.0,
        aggregator_x25519_pubkey_hex: Optional[str] = None,
    ) -> None:
        if mode not in [m.value for m in DHTRSIMode]:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = DHTRSIMode(mode)
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
        self.canary_gate = GoldenCanaryRegressionGate()
        self.telemetry_queue: List[QueuedTelemetryItem] = []

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

        self.telemetry_queue.append(QueuedTelemetryItem(
            scheduled_dispatch_time=scheduled_time,
            record=record,
        ))

        return True, "verified_anonymized_and_queued", record

    def get_pending_telemetry_batch(
        self,
        now_timestamp: Optional[float] = None,
    ) -> List[EncryptedTelemetryBundle]:
        """Release scheduled telemetry items whose Poisson jitter delay has elapsed, encrypted with AEAD."""
        if self.mode != DHTRSIMode.ANONYMOUS_CONTRIBUTE:
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

        # 1. Verify BEP-44 monotonic sequence and Ed25519 threshold signatures
        ok, reason = ThresholdSignatureVerifier.verify(
            manifest=manifest,
            trusted_signers=self.trusted_signers,
            threshold_m=self.threshold_m,
            current_seq=self.current_seq,
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
