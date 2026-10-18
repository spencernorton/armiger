"""Modular, CSAM-Protected, Encrypted, Poison/Spoof-Proof BEP-44 / BEP-46 DHT DB Sync Plugin.

Enforces:
  1. Multi-Layer CSAM & Abuse Protection:
     - Multi-stage ContentFilter normalization parity (normalization, leetspeak, zero-width stripping)
     - Deterministic & structural pre-ingest CSAM/abuse blocker
     - Ed25519-signed blinded SHA-256 tombstone purge engine
     - Armiger Mondrian conformal safety gates (junk/malware, out-of-scope, adult containment)
  2. Modular Sharded Subscription Architecture:
     - Partitioned by Category (10 modules, xxx strictly isolated & opt-in) x
       Language (14 modules) x Quality Tier (7 tiers) x Time Epoch
     - Fine-grained subscription filters: nodes only fetch and store subscribed shards
  3. X25519 + ChaCha20-Poly1305 AEAD Encryption:
     - Ephemeral / topic-derived key exchange via HKDF-SHA256
     - 12-byte random nonces and AAD binding shard ID, epoch, sequence, and Merkle root
     - Tamper-proof ciphertext with replay and cross-shard transplant defense
  4. BEP-44 / BEP-46 Root-of-Trust + K-of-N Distinct-ASN Quorum:
     - Strictly monotonic sequence counters (anti-replay/anti-downgrade)
     - Offline Ed25519 root signatures over canonical BEP-44 mutable manifests
     - Sybil-resistant K-of-N attestation quorum requiring independent distinct ASNs
     - Binary Merkle tree root verification and individual record inclusion proofs
  5. Anonymity & Per-Module Opt-In/Opt-Out:
     - Zero PII, peer IPs, or tracker credentials in payloads
     - Blinded infohash lookups and sanitized titles
     - Full sync modes: OFF, PULL_ONLY, SUBSCRIBE_AND_SEED, PUBLISHER
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import pathlib
import sqlite3
import time
import zlib
from typing import Any, Iterable, Sequence

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from csam_guard import (
    ArmigerSafetyGate,
    CSAMGuard,
    CSAMViolationError,
    CSAMViolationType,
    ContentFilterResult,
    MLKEM768,
    SignedTombstoneList,
    TombstoneStore,
    WOTSPlusSHAKE256,
    blind_infohash,
    generate_ed25519_keypair,
    normalize_text,
    sanitize_title,
    sign_tombstone_list,
)

# Optional zstandard import with graceful zlib fallback
try:
    import zstandard as zstd  # type: ignore

    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


# ---------------------------------------------------------------------------
# Constants & Partition Taxonomies
# ---------------------------------------------------------------------------

CATEGORIES_MAINSTREAM: tuple[str, ...] = (
    "movie",
    "tv_show",
    "music",
    "game",
    "software",
    "anime",
    "book",
    "comic",
    "course",
)

CATEGORY_ADULT = "xxx"

ALL_CATEGORIES: tuple[str, ...] = CATEGORIES_MAINSTREAM + (CATEGORY_ADULT,)

LANGUAGES: tuple[str, ...] = (
    "en",
    "ja",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ru",
    "zh",
    "ko",
    "hi",
    "ar",
    "multi",
    "other",
)

QUALITY_TIERS: tuple[str, ...] = (
    "uhd_4k",
    "hd_1080p",
    "hd_720p",
    "sd",
    "lossless_audio",
    "standard_audio",
    "any",
)

DEFAULT_EPOCH = "2026_q4"


class SyncMode(str, enum.Enum):
    """Operational mode for decentralized DHT sync."""

    OFF = "off"
    PULL_ONLY = "pull_only"
    SUBSCRIBE_AND_SEED = "subscribe_and_seed"
    PUBLISHER = "publisher"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TorrentRecord:
    """Canonical representation of an indexed torrent entry."""

    infohash: str  # 40-char hex
    title: str
    category: str
    language: str
    quality_tier: str
    epoch: str
    size_bytes: int
    file_count: int
    seeders: int
    leechers: int
    content_sha256: str
    file_list: list[str]
    safety_gate_passed: bool
    created_at: int

    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic dictionary for serialization and hashing."""
        return {
            "category": self.category.lower(),
            "content_sha256": self.content_sha256.lower(),
            "created_at": int(self.created_at),
            "epoch": self.epoch.lower(),
            "file_count": int(self.file_count),
            "file_list": sorted(self.file_list),
            "infohash": self.infohash.lower(),
            "language": self.language.lower(),
            "leechers": int(self.leechers),
            "quality_tier": self.quality_tier.lower(),
            "safety_gate_passed": bool(self.safety_gate_passed),
            "seeders": int(self.seeders),
            "size_bytes": int(self.size_bytes),
            "title": self.title,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def leaf_hash(self) -> bytes:
        return hashlib.sha256(self.canonical_bytes()).digest()


@dataclasses.dataclass(frozen=True)
class QuorumAttestation:
    """Individual node attestation with Hybrid Ed25519 + WOTS+-SHAKE256 PQ signatures."""

    node_id: str  # 40-char hex DHT node ID
    asn: int  # Autonomous System Number (e.g. AS13335)
    ip_digest: str  # Blinded SHA256 of IP/subnet
    pubkey_hex: str  # Ed25519 public key (64 hex chars)
    signature_hex: str  # Ed25519 signature over manifest canonical hash
    timestamp: int
    pq_pubkey_hex: str = ""  # WOTS+-SHAKE256 PQ public key (64 hex chars)
    pq_signature_hex: str = ""  # WOTS+-SHAKE256 PQ signature (4352 hex chars)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "asn": int(self.asn),
            "ip_digest": self.ip_digest.lower(),
            "node_id": self.node_id.lower(),
            "pubkey_hex": self.pubkey_hex.lower(),
            "timestamp": int(self.timestamp),
        }


@dataclasses.dataclass(frozen=True)
class DHTShardManifest:
    """BEP-44 / BEP-46 Hybrid Post-Quantum (Ed25519 + WOTS+-SHAKE256) signed manifest for an encrypted shard."""

    seq: int
    shard_id: str
    epoch: str
    record_count: int
    compressed_size_bytes: int
    uncompressed_size_bytes: int
    merkle_root: str  # 64-char hex
    ciphertext_sha256: str  # 64-char hex
    bep46_infohash: str  # 40-char hex
    timestamp: int
    root_pubkey_hex: str
    ed25519_signature_hex: str
    attestations: list[QuorumAttestation] = dataclasses.field(default_factory=list)
    root_pq_pubkey_hex: str = ""
    pq_signature_hex: str = ""

    def with_attestations(self, attestations: Sequence[QuorumAttestation]) -> DHTShardManifest:
        """Return a copy of the manifest with attached quorum attestations."""
        return dataclasses.replace(self, attestations=list(attestations))

    def canonical_dict_without_sig(self) -> dict[str, Any]:
        return {
            "bep46_infohash": self.bep46_infohash.lower(),
            "ciphertext_sha256": self.ciphertext_sha256.lower(),
            "compressed_size_bytes": int(self.compressed_size_bytes),
            "epoch": self.epoch.lower(),
            "merkle_root": self.merkle_root.lower(),
            "record_count": int(self.record_count),
            "root_pubkey_hex": self.root_pubkey_hex.lower(),
            "seq": int(self.seq),
            "shard_id": self.shard_id.lower(),
            "timestamp": int(self.timestamp),
            "uncompressed_size_bytes": int(self.uncompressed_size_bytes),
        }

    def canonical_payload(self) -> bytes:
        return json.dumps(self.canonical_dict_without_sig(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def canonical_digest(self) -> bytes:
        return hashlib.sha256(self.canonical_payload()).digest()

    def bep44_target(self, salt: str | None = None) -> str:
        """Compute BEP-44 DHT target lookup key: SHA1(pubkey || salt)."""
        pub_bytes = bytes.fromhex(self.root_pubkey_hex)
        salt_bytes = (salt or f"armiger-shard:{self.shard_id}").encode("utf-8")
        return hashlib.sha1(pub_bytes + salt_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Merkle Tree Implementation
# ---------------------------------------------------------------------------

class MerkleTree:
    """Binary Merkle tree over canonical record leaf hashes."""

    def __init__(self, leaves: Sequence[bytes]) -> None:
        self.leaves = list(leaves)
        self.levels: list[list[bytes]] = []
        self._build()

    def _build(self) -> None:
        if not self.leaves:
            self.root = b"\x00" * 32
            self.levels = [[]]
            return

        current = self.leaves[:]
        self.levels.append(current)

        while len(current) > 1:
            next_level: list[bytes] = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if i + 1 < len(current) else left
                parent = hashlib.sha256(left + right).digest()
                next_level.append(parent)
            self.levels.append(next_level)
            current = next_level

        self.root = self.levels[-1][0]

    @property
    def root_hex(self) -> str:
        return self.root.hex()

    def get_proof(self, index: int) -> list[tuple[str, bool]]:
        """Return list of (sibling_hash_hex, is_right) proof pairs for leaf at index."""
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range (total {len(self.leaves)})")

        proof: list[tuple[str, bool]] = []
        cur_idx = index
        for level in self.levels[:-1]:
            is_right = (cur_idx % 2 == 0)
            sib_idx = cur_idx + 1 if is_right else cur_idx - 1
            if sib_idx >= len(level):
                sib_idx = cur_idx  # Odd leaf duplicated
            proof.append((level[sib_idx].hex(), is_right))
            cur_idx //= 2

        return proof

    @staticmethod
    def verify_proof(leaf_hash: bytes, proof: Sequence[tuple[str, bool]], root_hex: str) -> bool:
        """Verify an inclusion proof against expected Merkle root."""
        current = leaf_hash
        for sib_hex, is_right in proof:
            sib = bytes.fromhex(sib_hex)
            if is_right:
                current = hashlib.sha256(current + sib).digest()
            else:
                current = hashlib.sha256(sib + current).digest()
        return current.hex().lower() == root_hex.lower()


# ---------------------------------------------------------------------------
# X25519 & ChaCha20-Poly1305 AEAD Cryptographic Engine
# ---------------------------------------------------------------------------

class ShardCryptoEngine:
    """Post-Quantum Hybrid (X25519 + ML-KEM-768 + SHAKE-256 / HKDF-SHA3-256) & ChaCha20-Poly1305 AEAD Engine."""

    HYBRID_PRIV_LEN = 32 + MLKEM768.DK_BYTES  # 2,432 bytes
    HYBRID_PUB_LEN = 32 + MLKEM768.EK_BYTES   # 1,216 bytes
    HYBRID_ENC_LEN = 32 + MLKEM768.CT_BYTES   # 1,120 bytes

    @staticmethod
    def derive_shard_key(
        master_secret: bytes,
        shard_id: str,
        epoch: str,
        pq_shared_secret: bytes = b"",
    ) -> bytes:
        """Derive 256-bit ChaCha20-Poly1305 key via Dual-PRF (HKDF-SHA256 XOR SHAKE-256/SHA3-256)."""
        salt_bytes = shard_id.lower().encode("utf-8")
        info_bytes = f"armiger-shard-chacha20-poly1305-pq-v2:{epoch.lower()}".encode("utf-8")
        ikm = master_secret + pq_shared_secret
        k_hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt_bytes,
            info=info_bytes,
        ).derive(ikm)
        k_shake = hashlib.shake_256(
            b"armiger-pq-shard-kdf-v2:" + salt_bytes + b":" + info_bytes + b":" + ikm
        ).digest(32)
        return bytes(a ^ b for a, b in zip(k_hkdf, k_shake))

    @classmethod
    def generate_pq_keypair(cls) -> tuple[bytes, bytes]:
        """Generate a Hybrid Post-Quantum (X25519 + ML-KEM-768) keypair: (priv_2432B, pub_1216B)."""
        priv = X25519PrivateKey.generate()
        pub = priv.public_key()
        x_priv = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        x_pub = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        pq_ek, pq_dk = MLKEM768.keygen()
        return x_priv + pq_dk, x_pub + pq_ek

    @classmethod
    def encapsulate_shard_key(
        cls,
        recipient_hybrid_pubkey: bytes,
        shard_id: str,
        epoch: str,
    ) -> tuple[bytes, bytes]:
        """Encapsulate a 256-bit shard key using Hybrid X25519 + ML-KEM-768, returning (kem_token_1120B, key_32B)."""
        if len(recipient_hybrid_pubkey) != cls.HYBRID_PUB_LEN:
            raise ValueError(f"Expected {cls.HYBRID_PUB_LEN}-byte hybrid public key")
        recip_x_pub = recipient_hybrid_pubkey[:32]
        recip_pq_ek = recipient_hybrid_pubkey[32:]
        ephem_priv = X25519PrivateKey.generate()
        ephem_x_pub = ephem_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        ss_classical = ephem_priv.exchange(X25519PublicKey.from_public_bytes(recip_x_pub))
        pq_ct, ss_pq = MLKEM768.encaps(recip_pq_ek)
        kem_token = ephem_x_pub + pq_ct
        ctx = hashlib.sha3_256(kem_token).digest()
        key = cls.derive_shard_key(ss_classical, shard_id, epoch, pq_shared_secret=ss_pq + ctx)
        return kem_token, key

    @classmethod
    def decapsulate_shard_key(
        cls,
        recipient_hybrid_privkey: bytes,
        kem_token: bytes,
        shard_id: str,
        epoch: str,
    ) -> bytes:
        """Decapsulate a 1,120-byte X25519 + ML-KEM-768 token to recover the 256-bit shard key."""
        if len(recipient_hybrid_privkey) != cls.HYBRID_PRIV_LEN or len(kem_token) != cls.HYBRID_ENC_LEN:
            raise ValueError("Invalid hybrid private key or KEM encapsulation token length")
        x_priv = X25519PrivateKey.from_private_bytes(recipient_hybrid_privkey[:32])
        pq_dk = recipient_hybrid_privkey[32:]
        ephem_x_pub = kem_token[:32]
        pq_ct = kem_token[32:]
        ss_classical = x_priv.exchange(X25519PublicKey.from_public_bytes(ephem_x_pub))
        ss_pq = MLKEM768.decaps(pq_dk, pq_ct)
        ctx = hashlib.sha3_256(kem_token).digest()
        return cls.derive_shard_key(ss_classical, shard_id, epoch, pq_shared_secret=ss_pq + ctx)

    @staticmethod
    def construct_aad(shard_id: str, epoch: str, seq: int, merkle_root: str) -> bytes:
        """Deterministic Authenticated Associated Data (AAD) binding ciphertext to shard metadata."""
        return f"armiger-shard-aad-v1:{shard_id.lower()}:{epoch.lower()}:{seq}:{merkle_root.lower()}".encode("utf-8")

    @classmethod
    def encrypt_shard(
        cls,
        plaintext: bytes,
        key: bytes,
        shard_id: str,
        epoch: str,
        seq: int,
        merkle_root: str,
    ) -> bytes:
        """Encrypt shard payload with 256-bit ChaCha20-Poly1305 AEAD (128-bit Grover PQ security).

        Format: nonce (12 bytes) + ciphertext_with_poly1305_tag
        """
        if len(key) != 32:
            raise ValueError(f"ChaCha20 key must be 32 bytes, got {len(key)}")
        nonce = os.urandom(12)
        aad = cls.construct_aad(shard_id, epoch, seq, merkle_root)
        aead = ChaCha20Poly1305(key)
        ciphertext = aead.encrypt(nonce, plaintext, aad)
        return nonce + ciphertext

    @classmethod
    def decrypt_shard(
        cls,
        bundle: bytes,
        key: bytes,
        shard_id: str,
        epoch: str,
        seq: int,
        merkle_root: str,
    ) -> bytes:
        """Decrypt shard payload and verify Poly1305 tag against AAD.

        Raises InvalidTag if tampered or transplanted across different shard_id/epoch/seq.
        """
        if len(bundle) < 12 + 16:
            raise ValueError("Encrypted bundle is too short to contain nonce and tag")
        nonce = bundle[:12]
        ciphertext = bundle[12:]
        aad = cls.construct_aad(shard_id, epoch, seq, merkle_root)
        aead = ChaCha20Poly1305(key)
        return aead.decrypt(nonce, ciphertext, aad)

    @classmethod
    def encrypt_shard_pq(
        cls,
        plaintext: bytes,
        recipient_hybrid_pubkey: bytes,
        shard_id: str,
        epoch: str,
        seq: int,
        merkle_root: str,
    ) -> bytes:
        """Encrypt shard with ephemeral X25519 + ML-KEM-768 encapsulation + 256-bit ChaCha20-Poly1305 AEAD.

        Format: kem_token (1,120 bytes) + nonce (12 bytes) + ciphertext_with_poly1305_tag
        """
        kem_token, key = cls.encapsulate_shard_key(recipient_hybrid_pubkey, shard_id, epoch)
        nonce = os.urandom(12)
        aad = cls.construct_aad(shard_id, epoch, seq, merkle_root) + b":kem:" + hashlib.sha3_256(kem_token).digest()
        aead = ChaCha20Poly1305(key)
        ciphertext = aead.encrypt(nonce, plaintext, aad)
        return kem_token + nonce + ciphertext

    @classmethod
    def decrypt_shard_pq(
        cls,
        pq_bundle: bytes,
        recipient_hybrid_privkey: bytes,
        shard_id: str,
        epoch: str,
        seq: int,
        merkle_root: str,
    ) -> bytes:
        """Decrypt an X25519 + ML-KEM-768 + ChaCha20-Poly1305 shard bundle."""
        min_len = cls.HYBRID_ENC_LEN + 12 + 16
        if len(pq_bundle) < min_len:
            raise ValueError("PQ encrypted shard bundle too short")
        kem_token = pq_bundle[: cls.HYBRID_ENC_LEN]
        nonce = pq_bundle[cls.HYBRID_ENC_LEN : cls.HYBRID_ENC_LEN + 12]
        ciphertext = pq_bundle[cls.HYBRID_ENC_LEN + 12 :]
        key = cls.decapsulate_shard_key(recipient_hybrid_privkey, kem_token, shard_id, epoch)
        aad = cls.construct_aad(shard_id, epoch, seq, merkle_root) + b":kem:" + hashlib.sha3_256(kem_token).digest()
        aead = ChaCha20Poly1305(key)
        return aead.decrypt(nonce, ciphertext, aad)


# ---------------------------------------------------------------------------
# Compression Utilities (Zstd with Zlib fallback)
# ---------------------------------------------------------------------------

def compress_payload(data: bytes) -> tuple[bytes, str]:
    """Compress payload using Zstd (level 7) or fallback to Zlib (level 9)."""
    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=7)
        return cctx.compress(data), "zstd"
    return zlib.compress(data, level=9), "zlib"


def decompress_payload(compressed: bytes, codec: str = "zstd") -> bytes:
    """Decompress payload using specified codec."""
    if codec == "zstd":
        if _HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(compressed)
        # If zstd not present, attempt zlib decompression as transparent fallback
        return zlib.decompress(compressed)
    elif codec == "zlib":
        return zlib.decompress(compressed)
    raise ValueError(f"Unsupported compression codec: {codec}")


# ---------------------------------------------------------------------------
# Modular Sharded Subscription Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SubscriptionConfig:
    """User-selected subscription preferences for modular DHT sharding."""

    categories: set[str] = dataclasses.field(default_factory=lambda: {"movie", "tv_show"})
    languages: set[str] = dataclasses.field(default_factory=lambda: {"en", "ja"})
    quality_tiers: set[str] = dataclasses.field(default_factory=lambda: {"uhd_4k", "hd_1080p"})
    epochs: set[str] = dataclasses.field(default_factory=lambda: {DEFAULT_EPOCH})
    enable_xxx: bool = False  # Strictly isolated, default OFF
    min_quorum: int = 5
    min_distinct_asns: int = 5
    sync_mode: SyncMode = SyncMode.PULL_ONLY

    @classmethod
    def parse_shard_id(cls, shard_id: str) -> tuple[str, str, str, str]:
        """Deconstruct 'category/language/quality_tier/epoch' shard identifier."""
        parts = shard_id.strip().lower().split("/")
        if len(parts) != 4:
            raise ValueError(f"Malformed shard_id: '{shard_id}'. Expected 'category/lang/quality/epoch'")
        return parts[0], parts[1], parts[2], parts[3]

    @classmethod
    def format_shard_id(cls, category: str, language: str, quality_tier: str, epoch: str) -> str:
        return f"{category.lower()}/{language.lower()}/{quality_tier.lower()}/{epoch.lower()}"

    def is_subscribed(self, shard_id: str) -> bool:
        """Check whether a given shard matches user subscription filters.

        Invariant: If category == 'xxx' and enable_xxx is False, unconditionally REJECT.
        """
        try:
            cat, lang, quality, ep = self.parse_shard_id(shard_id)
        except ValueError:
            return False

        # Strict isolation gate for adult content
        if cat == CATEGORY_ADULT and not self.enable_xxx:
            return False

        return (
            cat in self.categories
            and lang in self.languages
            and quality in self.quality_tiers
            and ep in self.epochs
        )


# ---------------------------------------------------------------------------
# SQLite Columnar Shard Database Engine
# ---------------------------------------------------------------------------

class ShardDatabase:
    """Local SQLite database storing partitioned torrent metadata."""

    def __init__(self, db_path: str | pathlib.Path = ":memory:", enable_xxx: bool = False) -> None:
        self.db_path = str(db_path)
        self.enable_xxx = enable_xxx
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self.conn:
            # Mainstream torrents table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS mainstream_torrents (
                    infohash TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    language TEXT NOT NULL,
                    quality_tier TEXT NOT NULL,
                    epoch TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    file_count INTEGER NOT NULL,
                    seeders INTEGER NOT NULL,
                    leechers INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    file_list_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_main_cat_lang ON mainstream_torrents(category, language)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_main_epoch ON mainstream_torrents(epoch)")

            # Isolated adult table
            if self.enable_xxx:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS adult_torrents (
                        infohash TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        category TEXT NOT NULL,
                        language TEXT NOT NULL,
                        quality_tier TEXT NOT NULL,
                        epoch TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        file_count INTEGER NOT NULL,
                        seeders INTEGER NOT NULL,
                        leechers INTEGER NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        file_list_json TEXT NOT NULL,
                        created_at INTEGER NOT NULL
                    )
                """)
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_adult_lang ON adult_torrents(language)")

    def insert_record(self, record: TorrentRecord) -> bool:
        """Insert a torrent record into the appropriate partitioned table."""
        is_adult = (record.category.lower() == CATEGORY_ADULT)
        if is_adult and not self.enable_xxx:
            return False  # Drop un-opted adult record

        table = "adult_torrents" if is_adult else "mainstream_torrents"
        with self.conn:
            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO {table} (
                    infohash, title, category, language, quality_tier, epoch,
                    size_bytes, file_count, seeders, leechers, content_sha256,
                    file_list_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.infohash.lower(),
                    record.title,
                    record.category.lower(),
                    record.language.lower(),
                    record.quality_tier.lower(),
                    record.epoch.lower(),
                    record.size_bytes,
                    record.file_count,
                    record.seeders,
                    record.leechers,
                    record.content_sha256.lower(),
                    json.dumps(record.file_list),
                    record.created_at,
                ),
            )
        return True

    def insert_records_batch(self, records: Sequence[TorrentRecord]) -> int:
        """Batch-insert validated torrent records within a single SQLite transaction."""
        if not records:
            return 0
        main_rows: list[tuple[Any, ...]] = []
        adult_rows: list[tuple[Any, ...]] = []
        for record in records:
            is_adult = (record.category.lower() == CATEGORY_ADULT)
            if is_adult and not self.enable_xxx:
                continue
            row = (
                record.infohash.lower(),
                record.title,
                record.category.lower(),
                record.language.lower(),
                record.quality_tier.lower(),
                record.epoch.lower(),
                record.size_bytes,
                record.file_count,
                record.seeders,
                record.leechers,
                record.content_sha256.lower(),
                json.dumps(record.file_list),
                record.created_at,
            )
            if is_adult:
                adult_rows.append(row)
            else:
                main_rows.append(row)

        with self.conn:
            if main_rows:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO mainstream_torrents (
                        infohash, title, category, language, quality_tier, epoch,
                        size_bytes, file_count, seeders, leechers, content_sha256,
                        file_list_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    main_rows,
                )
            if adult_rows and self.enable_xxx:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO adult_torrents (
                        infohash, title, category, language, quality_tier, epoch,
                        size_bytes, file_count, seeders, leechers, content_sha256,
                        file_list_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    adult_rows,
                )
        return len(main_rows) + len(adult_rows)

    def get_record(self, infohash: str) -> TorrentRecord | None:
        """Retrieve record by infohash across accessible tables."""
        clean = infohash.lower().strip()
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM mainstream_torrents WHERE infohash = ?", (clean,))
        row = cur.fetchone()
        if row is None and self.enable_xxx:
            cur.execute("SELECT * FROM adult_torrents WHERE infohash = ?", (clean,))
            row = cur.fetchone()

        if row is None:
            return None

        return TorrentRecord(
            infohash=row["infohash"],
            title=row["title"],
            category=row["category"],
            language=row["language"],
            quality_tier=row["quality_tier"],
            epoch=row["epoch"],
            size_bytes=row["size_bytes"],
            file_count=row["file_count"],
            seeders=row["seeders"],
            leechers=row["leechers"],
            content_sha256=row["content_sha256"],
            file_list=json.loads(row["file_list_json"]),
            safety_gate_passed=True,
            created_at=row["created_at"],
        )

    def purge_blinded_hashes(self, tombstone_store: TombstoneStore) -> int:
        """Purge any records whose blinded infohashes are registered on tombstone list."""
        purged = 0
        cur = self.conn.cursor()

        # Check mainstream
        cur.execute("SELECT infohash FROM mainstream_torrents")
        main_hashes = [r[0] for r in cur.fetchall()]
        to_delete_main = [h for h in main_hashes if tombstone_store.is_tombstoned(h)]
        if to_delete_main:
            with self.conn:
                self.conn.executemany("DELETE FROM mainstream_torrents WHERE infohash = ?", [(h,) for h in to_delete_main])
            purged += len(to_delete_main)

        # Check adult if enabled
        if self.enable_xxx:
            cur.execute("SELECT infohash FROM adult_torrents")
            adult_hashes = [r[0] for r in cur.fetchall()]
            to_delete_adult = [h for h in adult_hashes if tombstone_store.is_tombstoned(h)]
            if to_delete_adult:
                with self.conn:
                    self.conn.executemany("DELETE FROM adult_torrents WHERE infohash = ?", [(h,) for h in to_delete_adult])
                purged += len(to_delete_adult)

        return purged

    def count_records(self, category: str | None = None) -> int:
        cur = self.conn.cursor()
        if category == CATEGORY_ADULT:
            if not self.enable_xxx:
                return 0
            cur.execute("SELECT COUNT(*) FROM adult_torrents")
            return int(cur.fetchone()[0])
        elif category is not None:
            cur.execute("SELECT COUNT(*) FROM mainstream_torrents WHERE category = ?", (category.lower(),))
            return int(cur.fetchone()[0])
        else:
            cur.execute("SELECT COUNT(*) FROM mainstream_torrents")
            tot = int(cur.fetchone()[0])
            if self.enable_xxx:
                cur.execute("SELECT COUNT(*) FROM adult_torrents")
                tot += int(cur.fetchone()[0])
            return tot

    def search(self, query: str = "", category: str | None = None, language: str | None = None) -> list[TorrentRecord]:
        """Search records by title query, category, and language."""
        cur = self.conn.cursor()
        is_adult_query = (category == CATEGORY_ADULT)
        if is_adult_query and not self.enable_xxx:
            return []

        table = "adult_torrents" if is_adult_query else "mainstream_torrents"
        sql = f"SELECT * FROM {table} WHERE 1=1"
        params: list[Any] = []

        if query:
            sql += " AND title LIKE ?"
            params.append(f"%{query}%")
        if category and not is_adult_query:
            sql += " AND category = ?"
            params.append(category.lower())
        if language:
            sql += " AND language = ?"
            params.append(language.lower())

        cur.execute(sql, params)
        results: list[TorrentRecord] = []
        for row in cur.fetchall():
            results.append(TorrentRecord(
                infohash=row["infohash"],
                title=row["title"],
                category=row["category"],
                language=row["language"],
                quality_tier=row["quality_tier"],
                epoch=row["epoch"],
                size_bytes=row["size_bytes"],
                file_count=row["file_count"],
                seeders=row["seeders"],
                leechers=row["leechers"],
                content_sha256=row["content_sha256"],
                file_list=json.loads(row["file_list_json"]),
                safety_gate_passed=True,
                created_at=row["created_at"],
            ))
        return results

    def close(self) -> None:
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shard Serialization & Deserialization
# ---------------------------------------------------------------------------

def serialize_shard_records(records: Sequence[TorrentRecord]) -> tuple[bytes, MerkleTree]:
    """Serialize a list of TorrentRecords into a deterministic compressed micro-bundle."""
    # Deterministic sorting by infohash
    sorted_records = sorted(records, key=lambda r: r.infohash.lower())
    leaves = [r.leaf_hash() for r in sorted_records]
    tree = MerkleTree(leaves)

    payload_dict = {
        "format": "armiger-shard-v1",
        "record_count": len(sorted_records),
        "records": [r.canonical_dict() for r in sorted_records],
    }
    raw_json = json.dumps(payload_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    compressed, _codec = compress_payload(raw_json)
    return compressed, tree


def deserialize_shard_records(compressed: bytes) -> list[TorrentRecord]:
    """Decompress and parse shard records."""
    raw_json = decompress_payload(compressed)
    payload_dict = json.loads(raw_json.decode("utf-8"))
    records: list[TorrentRecord] = []
    for d in payload_dict.get("records", []):
        records.append(TorrentRecord(
            infohash=d["infohash"],
            title=d["title"],
            category=d["category"],
            language=d["language"],
            quality_tier=d["quality_tier"],
            epoch=d["epoch"],
            size_bytes=d["size_bytes"],
            file_count=d["file_count"],
            seeders=d["seeders"],
            leechers=d["leechers"],
            content_sha256=d["content_sha256"],
            file_list=d.get("file_list", []),
            safety_gate_passed=bool(d.get("safety_gate_passed", True)),
            created_at=d.get("created_at", int(time.time())),
        ))
    return records


# ---------------------------------------------------------------------------
# Quorum & Attestation Verification Engine
# ---------------------------------------------------------------------------

class QuorumVerifier:
    """Validates K-of-N distinct-ASN quorum attestations and Hybrid Ed25519 + WOTS+-SHAKE256 root signatures."""

    def __init__(
        self,
        trusted_root_pubkeys_hex: Sequence[str],
        min_quorum: int = 5,
        min_distinct_asns: int = 5,
        trusted_root_pq_pubkeys_hex: Sequence[str] | None = None,
    ) -> None:
        self.trusted_roots = [
            bytes.fromhex(k.strip().lower())
            for k in trusted_root_pubkeys_hex
            if len(k.strip()) == 64
        ]
        self.trusted_pq_roots = [
            bytes.fromhex(k.strip().lower())
            for k in (trusted_root_pq_pubkeys_hex or [])
            if len(k.strip()) == 64
        ]
        self.min_quorum = min_quorum
        self.min_distinct_asns = min_distinct_asns

    def verify_manifest(
        self,
        manifest: DHTShardManifest,
        current_seq: int = -1,
    ) -> tuple[bool, str]:
        """Verify:
          1. Monotonic sequence: seq > current_seq
          2. Root Hybrid Ed25519 + WOTS+-SHAKE256 Post-Quantum signature
          3. K-of-N attestation count >= min_quorum
          4. Distinct ASNs >= min_distinct_asns
          5. Each attestation Hybrid Ed25519 + WOTS+-SHAKE256 signature
        """
        # 1. Monotonic sequence check
        if manifest.seq <= current_seq:
            return False, f"stale_or_downgrade_sequence:{manifest.seq}<={current_seq}"

        # 2. Root Ed25519 signature check
        root_pub_bytes = bytes.fromhex(manifest.root_pubkey_hex)
        if self.trusted_roots and root_pub_bytes not in self.trusted_roots:
            return False, f"untrusted_root_pubkey:{manifest.root_pubkey_hex}"

        payload = manifest.canonical_payload()
        try:
            vk = Ed25519PublicKey.from_public_bytes(root_pub_bytes)
            vk.verify(bytes.fromhex(manifest.ed25519_signature_hex), payload)
        except (InvalidSignature, ValueError) as exc:
            return False, f"invalid_root_ed25519_signature:{exc}"

        # 2b. Root WOTS+-SHAKE256 Post-Quantum signature check
        if manifest.pq_signature_hex or manifest.root_pq_pubkey_hex or self.trusted_pq_roots:
            try:
                pq_root_bytes = bytes.fromhex(manifest.root_pq_pubkey_hex.strip().lower())
                pq_sig_bytes = bytes.fromhex(manifest.pq_signature_hex.strip())
            except ValueError as exc:
                return False, f"invalid_root_pq_wots_signature:{exc}"
            if self.trusted_pq_roots and pq_root_bytes not in self.trusted_pq_roots:
                return False, f"untrusted_root_pq_pubkey:{manifest.root_pq_pubkey_hex}"
            if not WOTSPlusSHAKE256.verify(pq_root_bytes, payload, pq_sig_bytes):
                return False, "invalid_root_pq_wots_signature:verification_failed"

        # 3. Quorum threshold check
        if len(manifest.attestations) < self.min_quorum:
            return False, f"insufficient_attestations:{len(manifest.attestations)}<{self.min_quorum}"

        # 4. Validate individual attestations & count distinct ASNs
        manifest_digest = manifest.canonical_digest()
        verified_asns: set[int] = set()
        seen_node_ids: set[str] = set()

        for att in manifest.attestations:
            # Prevent node duplicate Sybil
            if att.node_id.lower() in seen_node_ids:
                continue
            seen_node_ids.add(att.node_id.lower())

            try:
                node_vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(att.pubkey_hex))
                node_vk.verify(bytes.fromhex(att.signature_hex), manifest_digest)
            except (InvalidSignature, ValueError):
                continue

            if att.pq_signature_hex or att.pq_pubkey_hex:
                try:
                    node_pq_pk = bytes.fromhex(att.pq_pubkey_hex.strip().lower())
                    node_pq_sig = bytes.fromhex(att.pq_signature_hex.strip())
                    if not WOTSPlusSHAKE256.verify(node_pq_pk, manifest_digest, node_pq_sig):
                        continue
                except ValueError:
                    continue

            verified_asns.add(int(att.asn))

        if len(verified_asns) < self.min_distinct_asns:
            return False, f"insufficient_distinct_asns:{len(verified_asns)}<{self.min_distinct_asns}"

        return True, "quorum_verified"


# ---------------------------------------------------------------------------
# Main Plugin Interface: ArmigerDHTDatabaseSync
# ---------------------------------------------------------------------------

class ArmigerDHTDatabaseSync:
    """End-to-end decentralized, CSAM-safe, encrypted, modular DHT database sync plugin."""

    def __init__(
        self,
        subscription: SubscriptionConfig | None = None,
        master_secret: bytes | None = None,
        trusted_root_pubkeys_hex: Sequence[str] | None = None,
        db_path: str | pathlib.Path = ":memory:",
        tombstone_store: TombstoneStore | None = None,
    ) -> None:
        self.subscription = subscription or SubscriptionConfig()
        self.master_secret = master_secret or os.urandom(32)
        self.tombstone_store = tombstone_store or TombstoneStore(trusted_root_pubkeys_hex)
        self.csam_guard = CSAMGuard(self.tombstone_store)
        self.safety_gate = ArmigerSafetyGate()
        self.quorum_verifier = QuorumVerifier(
            trusted_root_pubkeys_hex=trusted_root_pubkeys_hex or [],
            min_quorum=self.subscription.min_quorum,
            min_distinct_asns=self.subscription.min_distinct_asns,
        )
        self.db = ShardDatabase(db_path=db_path, enable_xxx=self.subscription.enable_xxx)
        self.shard_sequences: dict[str, int] = {}

    def set_sync_mode(self, mode: SyncMode) -> None:
        self.subscription.sync_mode = mode

    def ingest_candidate_record(
        self,
        infohash: str,
        title: str,
        category: str,
        language: str,
        quality_tier: str,
        epoch: str = DEFAULT_EPOCH,
        size_bytes: int = 0,
        file_count: int = 1,
        seeders: int = 0,
        leechers: int = 0,
        content_sha256: str = "",
        file_list: list[str] | None = None,
        p_adult: float = 0.0,
        p_junk: float = 0.0,
        p_excluded_scope: float = 0.0,
    ) -> tuple[bool, str, TorrentRecord | None]:
        """Validate candidate record through CSAM guard and safety gates, then insert locally."""
        if self.subscription.sync_mode == SyncMode.OFF:
            return False, "sync_mode_off", None

        # 1. CSAM & structural harmful content blocker
        files = file_list or [title]
        res = self.csam_guard.inspect_torrent(
            infohash=infohash,
            title=title,
            file_list=files,
            category=category,
        )
        if not res.is_safe:
            return False, f"csam_guard_rejection:{res.violation_type.value}:{res.reason}", None

        # 2. Armiger Conformal Safety Gate
        gate_ok, gate_reason, gate_violation = self.safety_gate.evaluate(
            p_adult=p_adult,
            p_junk=p_junk,
            p_excluded_scope=p_excluded_scope,
            target_category=category,
        )
        if not gate_ok:
            return False, f"safety_gate_rejection:{gate_reason}", None

        # 3. Construct canonical TorrentRecord
        clean_sha = content_sha256 or hashlib.sha256(title.encode()).hexdigest()
        record = TorrentRecord(
            infohash=infohash.lower().strip(),
            title=res.sanitized_title,
            category=category.lower().strip(),
            language=language.lower().strip(),
            quality_tier=quality_tier.lower().strip(),
            epoch=epoch.lower().strip(),
            size_bytes=size_bytes,
            file_count=file_count,
            seeders=seeders,
            leechers=leechers,
            content_sha256=clean_sha.lower(),
            file_list=files,
            safety_gate_passed=True,
            created_at=int(time.time()),
        )

        # 4. Insert into local partitioned database
        inserted = self.db.insert_record(record)
        if not inserted:
            return False, "record_dropped_by_subscription_policy", None

        return True, "ingested", record

    def build_and_sign_shard(
        self,
        shard_id: str,
        records: Sequence[TorrentRecord],
        root_privkey: bytes,
        seq: int,
        attestations: Sequence[QuorumAttestation] | None = None,
    ) -> tuple[DHTShardManifest, bytes]:
        """Package a list of records into an encrypted micro-shard and generate signed manifest."""
        cat, lang, qual, epoch = SubscriptionConfig.parse_shard_id(shard_id)

        # 1. Serialize records and compute Merkle tree
        compressed_records, merkle_tree = serialize_shard_records(records)
        uncompressed_size = sum(len(r.canonical_bytes()) for r in records)

        # 2. Derive ChaCha20-Poly1305 key & encrypt
        key = ShardCryptoEngine.derive_shard_key(self.master_secret, shard_id, epoch)
        ciphertext_bundle = ShardCryptoEngine.encrypt_shard(
            plaintext=compressed_records,
            key=key,
            shard_id=shard_id,
            epoch=epoch,
            seq=seq,
            merkle_root=merkle_tree.root_hex,
        )

        ciphertext_sha = hashlib.sha256(ciphertext_bundle).hexdigest()
        bep46_infohash = hashlib.sha1(ciphertext_bundle).hexdigest()

        # 3. Create manifest and sign with Ed25519 root private key
        sk = Ed25519PrivateKey.from_private_bytes(root_privkey)
        pk = sk.public_key()
        root_pub_bytes = pk.public_bytes(Encoding.Raw, PublicFormat.Raw)

        manifest_dummy = DHTShardManifest(
            seq=seq,
            shard_id=shard_id,
            epoch=epoch,
            record_count=len(records),
            compressed_size_bytes=len(ciphertext_bundle),
            uncompressed_size_bytes=uncompressed_size,
            merkle_root=merkle_tree.root_hex,
            ciphertext_sha256=ciphertext_sha,
            bep46_infohash=bep46_infohash,
            timestamp=int(time.time()),
            root_pubkey_hex=root_pub_bytes.hex(),
            ed25519_signature_hex="",
            attestations=list(attestations or []),
        )

        payload_bytes = manifest_dummy.canonical_payload()
        sig = sk.sign(payload_bytes)
        pq_priv, pq_pub = WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(root_privkey)
        pq_sig = WOTSPlusSHAKE256.sign(pq_priv, payload_bytes)

        manifest = DHTShardManifest(
            seq=seq,
            shard_id=shard_id,
            epoch=epoch,
            record_count=len(records),
            compressed_size_bytes=len(ciphertext_bundle),
            uncompressed_size_bytes=uncompressed_size,
            merkle_root=merkle_tree.root_hex,
            ciphertext_sha256=ciphertext_sha,
            bep46_infohash=bep46_infohash,
            timestamp=manifest_dummy.timestamp,
            root_pubkey_hex=root_pub_bytes.hex(),
            ed25519_signature_hex=sig.hex(),
            attestations=list(attestations or []),
            root_pq_pubkey_hex=pq_pub.hex(),
            pq_signature_hex=pq_sig.hex(),
        )

        return manifest, ciphertext_bundle

    def receive_and_apply_shard(
        self,
        manifest: DHTShardManifest,
        ciphertext_bundle: bytes,
    ) -> tuple[bool, str, int]:
        """Validate, decrypt, Merkle-verify, and ingest an incoming decentralized shard."""
        if self.subscription.sync_mode == SyncMode.OFF:
            return False, "sync_mode_off", 0

        # 1. Check user subscription filter (zero bandwidth / zero storage for unselected shards)
        if not self.subscription.is_subscribed(manifest.shard_id):
            return False, f"unsubscribed_shard_filtered:{manifest.shard_id}", 0

        # 2. Verify monotonic sequence & quorum
        cur_seq = self.shard_sequences.get(manifest.shard_id, -1)
        q_ok, q_msg = self.quorum_verifier.verify_manifest(manifest, current_seq=cur_seq)
        if not q_ok:
            return False, f"quorum_verification_failed:{q_msg}", 0

        # 3. Content-addressed verification of ciphertext
        actual_sha = hashlib.sha256(ciphertext_bundle).hexdigest()
        if actual_sha.lower() != manifest.ciphertext_sha256.lower():
            return False, "ciphertext_sha256_mismatch", 0

        # 4. Decrypt shard with AEAD
        key = ShardCryptoEngine.derive_shard_key(self.master_secret, manifest.shard_id, manifest.epoch)
        try:
            plaintext = ShardCryptoEngine.decrypt_shard(
                bundle=ciphertext_bundle,
                key=key,
                shard_id=manifest.shard_id,
                epoch=manifest.epoch,
                seq=manifest.seq,
                merkle_root=manifest.merkle_root,
            )
        except (InvalidTag, ValueError) as exc:
            return False, f"aead_decryption_failed:{exc}", 0

        # 5. Decompress records & verify Merkle root
        records = deserialize_shard_records(plaintext)
        reconstructed_leaves = [r.leaf_hash() for r in sorted(records, key=lambda r: r.infohash.lower())]
        reconstructed_tree = MerkleTree(reconstructed_leaves)

        if reconstructed_tree.root_hex.lower() != manifest.merkle_root.lower():
            return False, "merkle_root_mismatch", 0

        # 6. Pre-ingest safety re-inspection of decrypted records
        accepted_records: list[TorrentRecord] = []
        for rec in records:
            # Check against tombstone store
            if self.tombstone_store.is_tombstoned(rec.infohash):
                continue

            # CSAM inspection
            c_res = self.csam_guard.inspect_torrent(
                infohash=rec.infohash,
                title=rec.title,
                file_list=rec.file_list,
                category=rec.category,
            )
            if not c_res.is_safe:
                continue

            accepted_records.append(rec)

        ingested_count = self.db.insert_records_batch(accepted_records)

        # 7. Update sequence number
        self.shard_sequences[manifest.shard_id] = manifest.seq
        return True, "shard_applied_successfully", ingested_count

    def apply_signed_tombstones(self, manifest: SignedTombstoneList) -> tuple[bool, str, int, int]:
        """Apply a signed tombstone list and purge local records.

        Returns (ok, msg, new_tombstones_count, purged_records_count).
        """
        ok, msg, count = self.tombstone_store.apply_signed_tombstones(manifest)
        if not ok:
            return False, msg, 0, 0

        purged = self.db.purge_blinded_hashes(self.tombstone_store)
        return True, msg, count, purged

    def close(self) -> None:
        """Close local SQLite database connection."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Quorum Helper Factory
# ---------------------------------------------------------------------------

def create_mock_quorum_attestations(
    manifest_digest: bytes,
    distinct_asns: Sequence[int],
    timestamp: int | None = None,
) -> list[QuorumAttestation]:
    """Generate cryptographically valid Hybrid Ed25519 + WOTS+-SHAKE256 QuorumAttestations from distinct ASNs."""
    ts = timestamp or int(time.time())
    attestations: list[QuorumAttestation] = []
    for asn in distinct_asns:
        priv, pub = generate_ed25519_keypair()
        sk = Ed25519PrivateKey.from_private_bytes(priv)
        sig = sk.sign(manifest_digest)
        pq_priv, pq_pub = WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(priv)
        pq_sig = WOTSPlusSHAKE256.sign(pq_priv, manifest_digest)
        node_id = hashlib.sha1(pub + f"asn{asn}".encode()).hexdigest()
        ip_digest = hashlib.sha256(f"192.0.2.{asn % 250}".encode()).hexdigest()

        attestations.append(QuorumAttestation(
            node_id=node_id,
            asn=int(asn),
            ip_digest=ip_digest,
            pubkey_hex=pub.hex(),
            signature_hex=sig.hex(),
            timestamp=ts,
            pq_pubkey_hex=pq_pub.hex(),
            pq_signature_hex=pq_sig.hex(),
        ))
    return attestations
