"""Comprehensive Test Suite for Armiger Modular CSAM-Safe Encrypted DHT DB Sync Plugin.

Tests:
  1. Multi-Layer CSAM & Harmful-Content Protection:
     - Multi-stage ContentFilter normalization parity (normalization, leetspeak, zero-width chars, diacritics)
     - Private tracker passkey stripping for complete anonymity
     - Direct CSAM & underage exploitation blocking
     - Non-consensual abuse & extreme violence blocking
     - Disguised malware dropper (.mp4.exe, .scr, .bat) blocking
     - Armiger Mondrian conformal safety gates (junk, out-of-scope, adult leakage)
     - Ed25519-signed blinded SHA-256 tombstone purge engine (verification, anti-replay, active purging)
  2. Modular Sharded Subscription Architecture:
     - Partitioning by Category (10 modules) x Language (14 modules) x Quality Tier x Epoch
     - Strict isolation & opt-in for 'xxx' (default OFF, segregated table, no leakage)
     - Selective subscription filtering (only subscribed shards are processed)
  3. X25519 + ChaCha20-Poly1305 AEAD Encryption:
     - HKDF key derivation
     - Round-trip encryption and decryption
     - AAD integrity protection (tampering, cross-shard transplant defense)
  4. BEP-44 / BEP-46 Root-of-Trust + K-of-N Distinct-ASN Quorum + Merkle Shards:
     - BEP-44 DHT target derivation
     - Monotonic sequence verification
     - K-of-N quorum threshold & Sybil distinct-ASN enforcement
     - Merkle root verification and individual inclusion proofs
  5. Anonymity & Per-Module Opt-In / Opt-Out:
     - Zero PII in canonical payloads
     - SyncMode (OFF, PULL_ONLY, SUBSCRIBE_AND_SEED, PUBLISHER)
     - Dynamic subscription modifications
"""
from __future__ import annotations

import hashlib
import os
import sys
import unittest

from csam_guard import (
    ArmigerSafetyGate,
    CSAMGuard,
    CSAMViolationType,
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
    SubscriptionConfig,
    SyncMode,
    TorrentRecord,
    create_mock_quorum_attestations,
    deserialize_shard_records,
    serialize_shard_records,
)
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class TestCSAMGuardAndSafetyGates(unittest.TestCase):
    """Test Suite 1: Multi-layer CSAM, harmful content, and safety gate protections."""

    def setUp(self) -> None:
        self.root_priv, self.root_pub = generate_ed25519_keypair()
        self.tombstone_store = TombstoneStore(trusted_root_pubkeys_hex=[self.root_pub.hex()])
        self.guard = CSAMGuard(tombstone_store=self.tombstone_store)
        self.safety_gate = ArmigerSafetyGate()

    def test_text_normalization_and_leetspeak(self) -> None:
        # Leetspeak deobfuscation
        raw = "P3d0ph1l1a"
        norm = normalize_text(raw)
        self.assertIn("pedophilia", norm)

        # Zero-width stripping and diacritics
        raw2 = "T\u200Be\u200De\u200Bn"
        self.assertEqual(normalize_text(raw2), "teen")

        # Repetitive runs
        raw3 = "teeeen porn"
        self.assertEqual(normalize_text(raw3), "teen porn")

    def test_private_tracker_passkey_sanitization(self) -> None:
        raw_title = "Big.Buck.Bunny.2008.1080p.BluRay[passkey=abc123secretkey]?passkey=xyz789"
        sanitized = sanitize_title(raw_title)
        self.assertNotIn("passkey=", sanitized)
        self.assertNotIn("abc123secretkey", sanitized)
        self.assertEqual(sanitized, "Big.Buck.Bunny.2008.1080p.BluRay")

    def test_direct_csam_detection(self) -> None:
        forbidden_samples = [
            "Archive_Collection_CSAM_vids",
            "Underage porn leaked collection",
            "Pre-teen sex video archive",
            "Kindergarten erotic private camera",
            "Ch1ld p0rn direct link 2026",
            "Lolita hardcore uncensored stream",
        ]
        for sample in forbidden_samples:
            res = self.guard.inspect_torrent(
                infohash="a" * 40,
                title=sample,
                category="movie",
            )
            self.assertFalse(res.is_safe, f"Failed to block CSAM sample: {sample}")
            self.assertEqual(res.violation_type, CSAMViolationType.CSAM_EXPLOITATION)

    def test_nonconsensual_and_extreme_violence_detection(self) -> None:
        # Non-consensual exploitation
        res1 = self.guard.inspect_torrent("b" * 40, "Secret Bathroom Spycam Upskirt Footage")
        self.assertFalse(res1.is_safe)
        self.assertEqual(res1.violation_type, CSAMViolationType.NONCONSENSUAL_ABUSE)

        # Extreme violence / snuff
        res2 = self.guard.inspect_torrent("c" * 40, "Real Snuff Film Execution Video 2026")
        self.assertFalse(res2.is_safe)
        self.assertEqual(res2.violation_type, CSAMViolationType.EXTREME_VIOLENCE_GORE)

    def test_disguised_malware_dropper_in_media_torrents(self) -> None:
        # Dangerous executable disguised as media in a movie torrent
        malicious_files = ["Big.Buck.Bunny.2008.1080p.mkv", "codec_update.mp4.exe", "instructions.txt"]
        res = self.guard.inspect_torrent(
            infohash="d" * 40,
            title="Big.Buck.Bunny.2008.1080p.BluRay.x264",
            file_list=malicious_files,
            category="movie",
        )
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_type, CSAMViolationType.MALWARE_PAYLOAD)
        self.assertIn(".exe", res.reason)

    def test_armiger_mondrian_safety_gate(self) -> None:
        # 1. High junk probability (malware/phishing)
        ok1, reason1, vtype1 = self.safety_gate.evaluate(p_adult=0.01, p_junk=0.15, p_excluded_scope=0.02, target_category="movie")
        self.assertFalse(ok1)
        self.assertEqual(vtype1, CSAMViolationType.SAFETY_GATE_JUNK)

        # 2. High out-of-scope probability
        ok2, reason2, vtype2 = self.safety_gate.evaluate(p_adult=0.01, p_junk=0.005, p_excluded_scope=0.55, target_category="movie")
        self.assertFalse(ok2)
        self.assertEqual(vtype2, CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE)

        # 3. Adult leakage into mainstream category
        ok3, reason3, vtype3 = self.safety_gate.evaluate(p_adult=0.88, p_junk=0.001, p_excluded_scope=0.01, target_category="movie")
        self.assertFalse(ok3)
        self.assertEqual(vtype3, CSAMViolationType.UNISOLATED_ADULT)

        # 4. Adult permitted when targeted at isolated 'xxx' category
        ok4, reason4, vtype4 = self.safety_gate.evaluate(p_adult=0.88, p_junk=0.001, p_excluded_scope=0.01, target_category="xxx")
        self.assertTrue(ok4)
        self.assertIsNone(vtype4)

    def test_signed_tombstone_purge_lifecycle(self) -> None:
        bad_hash_1 = "1111111111222222222233333333334444444444"
        bad_hash_2 = "aaaaaaaaaabbbbbbbbbbccccccccccdddddddddd"

        # Initially, hash is not tombstoned
        self.assertFalse(self.tombstone_store.is_tombstoned(bad_hash_1))

        # Sign tombstone manifest with root key
        manifest = sign_tombstone_list(
            seq=1,
            infohashes_or_blinded=[bad_hash_1, bad_hash_2],
            private_key_bytes=self.root_priv,
            reasons={blind_infohash(bad_hash_1): "CSAM_ABUSE"},
        )

        ok, msg, count = self.tombstone_store.apply_signed_tombstones(manifest)
        self.assertTrue(ok)
        self.assertEqual(count, 2)
        self.assertTrue(self.tombstone_store.is_tombstoned(bad_hash_1))
        self.assertTrue(self.tombstone_store.is_tombstoned(bad_hash_2))

        # Anti-replay / downgrade check: seq=1 re-submitted must be rejected
        ok_dup, msg_dup, _ = self.tombstone_store.apply_signed_tombstones(manifest)
        self.assertFalse(ok_dup)
        self.assertIn("stale_or_replayed_tombstone_seq", msg_dup)

        # Guard immediately blocks ingestion of tombstoned hash
        res = self.guard.inspect_torrent(infohash=bad_hash_1, title="Harmless Looking Title")
        self.assertFalse(res.is_safe)
        self.assertEqual(res.violation_type, CSAMViolationType.TOMBSTONED_HASH)


class TestModularShardedSubscription(unittest.TestCase):
    """Test Suite 2: Modular subscription partitioned by Category, Language, Quality, Epoch."""

    def test_taxonomy_definitions(self) -> None:
        self.assertEqual(len(CATEGORIES_MAINSTREAM), 9)
        self.assertEqual(len(ALL_CATEGORIES), 10)
        self.assertEqual(CATEGORY_ADULT, "xxx")
        self.assertGreaterEqual(len(LANGUAGES), 14)
        self.assertGreaterEqual(len(QUALITY_TIERS), 7)

    def test_strict_xxx_isolation_in_subscription(self) -> None:
        # Default config: enable_xxx=False
        cfg = SubscriptionConfig(
            categories={"movie", "xxx"},  # Even if user includes "xxx" in category set
            languages={"en"},
            quality_tiers={"hd_1080p"},
            epochs={"2026_q4"},
            enable_xxx=False,
        )

        # Mainstream shard is subscribed
        self.assertTrue(cfg.is_subscribed("movie/en/hd_1080p/2026_q4"))

        # Adult shard is strictly blocked because enable_xxx is False
        self.assertFalse(cfg.is_subscribed("xxx/en/hd_1080p/2026_q4"))

        # Opt-in to adult:
        cfg.enable_xxx = True
        self.assertTrue(cfg.is_subscribed("xxx/en/hd_1080p/2026_q4"))

    def test_selective_shard_filtering(self) -> None:
        # Node subscribes only to English & Japanese movies in 1080p/4K
        cfg = SubscriptionConfig(
            categories={"movie"},
            languages={"en", "ja"},
            quality_tiers={"uhd_4k", "hd_1080p"},
            epochs={"2026_q4"},
            enable_xxx=False,
        )

        # Matches
        self.assertTrue(cfg.is_subscribed("movie/en/hd_1080p/2026_q4"))
        self.assertTrue(cfg.is_subscribed("movie/ja/uhd_4k/2026_q4"))

        # Mismatches (ignored without bandwidth overhead)
        self.assertFalse(cfg.is_subscribed("movie/fr/hd_1080p/2026_q4"))  # Wrong language
        self.assertFalse(cfg.is_subscribed("tv_show/en/hd_1080p/2026_q4"))  # Wrong category
        self.assertFalse(cfg.is_subscribed("movie/en/sd/2026_q4"))  # Wrong quality
        self.assertFalse(cfg.is_subscribed("movie/en/hd_1080p/2025_q1"))  # Wrong epoch

    def test_database_table_segregation(self) -> None:
        sync_engine = ArmigerDHTDatabaseSync(
            subscription=SubscriptionConfig(enable_xxx=False),
        )

        # Ingest mainstream record
        rec_main = TorrentRecord(
            infohash="1111222233334444555566667777888899990000",
            title="Tears.of.Steel.2012.1080p",
            category="movie",
            language="en",
            quality_tier="hd_1080p",
            epoch="2026_q4",
            size_bytes=10000000,
            file_count=1,
            seeders=150,
            leechers=10,
            content_sha256="abcd" * 16,
            file_list=["Tears.of.Steel.2012.1080p.mkv"],
            safety_gate_passed=True,
            created_at=100,
        )
        self.assertTrue(sync_engine.db.insert_record(rec_main))

        # Attempt to insert adult record into non-opted database
        rec_adult = TorrentRecord(
            infohash="9999888877776666555544443333222211110000",
            title="Adult.Feature.Film.XXX",
            category="xxx",
            language="en",
            quality_tier="hd_1080p",
            epoch="2026_q4",
            size_bytes=5000000,
            file_count=1,
            seeders=50,
            leechers=5,
            content_sha256="ef01" * 16,
            file_list=["Adult.Feature.Film.XXX.mp4"],
            safety_gate_passed=True,
            created_at=100,
        )
        self.assertFalse(sync_engine.db.insert_record(rec_adult))

        # Verify search only sees mainstream
        results = sync_engine.db.search("Tears.of.Steel")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].category, "movie")

        # Adult search returns empty
        adult_results = sync_engine.db.search(category="xxx")
        self.assertEqual(len(adult_results), 0)
        sync_engine.close()


class TestAEADEncryptionAndKeyDerivation(unittest.TestCase):
    """Test Suite 3: X25519 & ChaCha20-Poly1305 AEAD payload security."""

    def setUp(self) -> None:
        self.master_secret = os.urandom(32)
        self.shard_id = "movie/en/hd_1080p/2026_q4"
        self.epoch = "2026_q4"
        self.seq = 42
        self.merkle_root = "f" * 64
        self.key = ShardCryptoEngine.derive_shard_key(self.master_secret, self.shard_id, self.epoch)
        self.sample_payload = b"Armiger Columnar SQLite Shard Binary Micro-Bundle"

    def test_key_derivation_deterministic_and_unique(self) -> None:
        # Same parameters -> same key
        key1 = ShardCryptoEngine.derive_shard_key(self.master_secret, self.shard_id, self.epoch)
        key2 = ShardCryptoEngine.derive_shard_key(self.master_secret, self.shard_id, self.epoch)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 32)

        # Different shard_id -> different key
        key_other = ShardCryptoEngine.derive_shard_key(self.master_secret, "tv_show/en/hd_1080p/2026_q4", self.epoch)
        self.assertNotEqual(key1, key_other)

    def test_round_trip_encryption_decryption(self) -> None:
        bundle = ShardCryptoEngine.encrypt_shard(
            plaintext=self.sample_payload,
            key=self.key,
            shard_id=self.shard_id,
            epoch=self.epoch,
            seq=self.seq,
            merkle_root=self.merkle_root,
        )
        self.assertGreater(len(bundle), len(self.sample_payload))

        decrypted = ShardCryptoEngine.decrypt_shard(
            bundle=bundle,
            key=self.key,
            shard_id=self.shard_id,
            epoch=self.epoch,
            seq=self.seq,
            merkle_root=self.merkle_root,
        )
        self.assertEqual(decrypted, self.sample_payload)

    def test_ciphertext_tampering_defense(self) -> None:
        bundle = bytearray(ShardCryptoEngine.encrypt_shard(
            plaintext=self.sample_payload,
            key=self.key,
            shard_id=self.shard_id,
            epoch=self.epoch,
            seq=self.seq,
            merkle_root=self.merkle_root,
        ))
        # Tamper single byte in ciphertext
        bundle[-1] ^= 0x01

        with self.assertRaises(InvalidTag):
            ShardCryptoEngine.decrypt_shard(
                bundle=bytes(bundle),
                key=self.key,
                shard_id=self.shard_id,
                epoch=self.epoch,
                seq=self.seq,
                merkle_root=self.merkle_root,
            )

    def test_cross_shard_transplant_defense(self) -> None:
        # An attacker attempts to reuse valid ciphertext from 'movie' in 'game' shard
        bundle = ShardCryptoEngine.encrypt_shard(
            plaintext=self.sample_payload,
            key=self.key,
            shard_id=self.shard_id,
            epoch=self.epoch,
            seq=self.seq,
            merkle_root=self.merkle_root,
        )

        with self.assertRaises(InvalidTag):
            # Attempting to decrypt with different shard_id in AAD fails
            ShardCryptoEngine.decrypt_shard(
                bundle=bundle,
                key=self.key,
                shard_id="game/en/hd_1080p/2026_q4",  # Transplanted shard_id
                epoch=self.epoch,
                seq=self.seq,
                merkle_root=self.merkle_root,
            )

    def test_post_quantum_mlkem768_hybrid_shard_encryption_and_fo_rejection(self) -> None:
        """Verify Hybrid X25519 + NIST FIPS 203 ML-KEM-768 shard encryption and FO implicit rejection."""
        priv_pq, pub_pq = ShardCryptoEngine.generate_pq_keypair()
        self.assertEqual(len(priv_pq), 2432)  # 32B X25519 + 2400B ML-KEM-768 dk
        self.assertEqual(len(pub_pq), 1216)   # 32B X25519 + 1184B ML-KEM-768 ek

        pq_bundle = ShardCryptoEngine.encrypt_shard_pq(
            plaintext=self.sample_payload,
            recipient_hybrid_pubkey=pub_pq,
            shard_id=self.shard_id,
            epoch=self.epoch,
            seq=self.seq,
            merkle_root=self.merkle_root,
        )
        decrypted = ShardCryptoEngine.decrypt_shard_pq(
            pq_bundle=pq_bundle,
            recipient_hybrid_privkey=priv_pq,
            shard_id=self.shard_id,
            epoch=self.epoch,
            seq=self.seq,
            merkle_root=self.merkle_root,
        )
        self.assertEqual(decrypted, self.sample_payload)

        # Flip 1 bit inside the 1,088-byte ML-KEM-768 lattice ciphertext -> FO implicit rejection -> InvalidTag
        tampered = bytearray(pq_bundle)
        tampered[64] ^= 0x01
        with self.assertRaises(InvalidTag):
            ShardCryptoEngine.decrypt_shard_pq(
                pq_bundle=bytes(tampered),
                recipient_hybrid_privkey=priv_pq,
                shard_id=self.shard_id,
                epoch=self.epoch,
                seq=self.seq,
                merkle_root=self.merkle_root,
            )

    def test_post_quantum_wots_shake256_manifest_and_tombstone_tamper_rejection(self) -> None:
        """Verify WOTS+-SHAKE256 Post-Quantum dual-signatures block forged shard manifests and tombstones."""
        import dataclasses
        root_priv, root_pub = generate_ed25519_keypair()
        tomb_store = TombstoneStore(trusted_root_pubkeys_hex=[root_pub.hex()])
        tomb = sign_tombstone_list(seq=1, infohashes_or_blinded=["a" * 40], private_key_bytes=root_priv)
        self.assertEqual(len(bytes.fromhex(tomb.pq_pubkey_hex)), 32)
        self.assertEqual(len(bytes.fromhex(tomb.pq_signature_hex)), 2176)

        # Tamper 1 byte of the WOTS+-SHAKE256 PQ signature while keeping valid Ed25519 signature
        bad_pq_sig = bytearray(bytes.fromhex(tomb.pq_signature_hex))
        bad_pq_sig[100] ^= 0xFF
        tampered_tomb = dataclasses.replace(tomb, pq_signature_hex=bad_pq_sig.hex())
        ok, msg, _ = tomb_store.apply_signed_tombstones(tampered_tomb)
        self.assertFalse(ok)
        self.assertEqual(msg, "pq_wots_signature_verification_failed")


class TestBEP44QuorumAndMerkleTree(unittest.TestCase):
    """Test Suite 4: BEP-44/46 Ed25519 root, K-of-N distinct-ASN quorum, and Merkle proofs."""

    def setUp(self) -> None:
        self.root_priv, self.root_pub = generate_ed25519_keypair()
        self.verifier = QuorumVerifier(
            trusted_root_pubkeys_hex=[self.root_pub.hex()],
            min_quorum=5,
            min_distinct_asns=5,
        )

    def test_merkle_tree_construction_and_inclusion_proofs(self) -> None:
        records = [
            TorrentRecord(
                infohash=f"{i:040x}",
                title=f"Movie Title {i}",
                category="movie",
                language="en",
                quality_tier="hd_1080p",
                epoch="2026_q4",
                size_bytes=1000 * i,
                file_count=1,
                seeders=10 * i,
                leechers=i,
                content_sha256=f"{i:064x}",
                file_list=[f"movie_{i}.mkv"],
                safety_gate_passed=True,
                created_at=100 + i,
            )
            for i in range(1, 8)  # 7 records (tests odd-leaf handling)
        ]
        compressed, tree = serialize_shard_records(records)
        self.assertIsNotNone(tree.root_hex)

        # Verify inclusion proof for each record
        for idx in range(len(records)):
            leaf_hash = records[idx].leaf_hash()
            proof = tree.get_proof(idx)
            self.assertTrue(MerkleTree.verify_proof(leaf_hash, proof, tree.root_hex))

        # Tampered leaf hash must fail proof verification
        fake_leaf = hashlib.sha256(b"tampered_record").digest()
        self.assertFalse(MerkleTree.verify_proof(fake_leaf, tree.get_proof(0), tree.root_hex))

    def test_quorum_verification_success(self) -> None:
        # Create manifest
        manifest_dummy = DHTShardManifest(
            seq=1,
            shard_id="movie/en/hd_1080p/2026_q4",
            epoch="2026_q4",
            record_count=10,
            compressed_size_bytes=500,
            uncompressed_size_bytes=1200,
            merkle_root="a" * 64,
            ciphertext_sha256="b" * 64,
            bep46_infohash="c" * 40,
            timestamp=1000,
            root_pubkey_hex=self.root_pub.hex(),
            ed25519_signature_hex="",
        )
        sk = Ed25519PrivateKey.from_private_bytes(self.root_priv)
        sig = sk.sign(manifest_dummy.canonical_payload())

        # Generate 5 valid attestations from 5 distinct ASNs (Google, Cloudflare, Amazon, Comcast, Lumen)
        distinct_asns = [15169, 13335, 16509, 7922, 3356]
        attestations = create_mock_quorum_attestations(manifest_dummy.canonical_digest(), distinct_asns)

        manifest = DHTShardManifest(
            seq=manifest_dummy.seq,
            shard_id=manifest_dummy.shard_id,
            epoch=manifest_dummy.epoch,
            record_count=manifest_dummy.record_count,
            compressed_size_bytes=manifest_dummy.compressed_size_bytes,
            uncompressed_size_bytes=manifest_dummy.uncompressed_size_bytes,
            merkle_root=manifest_dummy.merkle_root,
            ciphertext_sha256=manifest_dummy.ciphertext_sha256,
            bep46_infohash=manifest_dummy.bep46_infohash,
            timestamp=manifest_dummy.timestamp,
            root_pubkey_hex=self.root_pub.hex(),
            ed25519_signature_hex=sig.hex(),
            attestations=attestations,
        )

        ok, msg = self.verifier.verify_manifest(manifest, current_seq=0)
        self.assertTrue(ok)
        self.assertEqual(msg, "quorum_verified")

    def test_sybil_attack_rejected_duplicate_asns(self) -> None:
        manifest_dummy = DHTShardManifest(
            seq=1,
            shard_id="movie/en/hd_1080p/2026_q4",
            epoch="2026_q4",
            record_count=10,
            compressed_size_bytes=500,
            uncompressed_size_bytes=1200,
            merkle_root="a" * 64,
            ciphertext_sha256="b" * 64,
            bep46_infohash="c" * 40,
            timestamp=1000,
            root_pubkey_hex=self.root_pub.hex(),
            ed25519_signature_hex="",
        )
        sk = Ed25519PrivateKey.from_private_bytes(self.root_priv)
        sig = sk.sign(manifest_dummy.canonical_payload())

        # Sybil attacker creates 5 nodes but all hosted in the same ASN (e.g. AS99999)
        sybil_asns = [99999, 99999, 99999, 99999, 99999]
        attestations = create_mock_quorum_attestations(manifest_dummy.canonical_digest(), sybil_asns)

        manifest = DHTShardManifest(
            seq=manifest_dummy.seq,
            shard_id=manifest_dummy.shard_id,
            epoch=manifest_dummy.epoch,
            record_count=manifest_dummy.record_count,
            compressed_size_bytes=manifest_dummy.compressed_size_bytes,
            uncompressed_size_bytes=manifest_dummy.uncompressed_size_bytes,
            merkle_root=manifest_dummy.merkle_root,
            ciphertext_sha256=manifest_dummy.ciphertext_sha256,
            bep46_infohash=manifest_dummy.bep46_infohash,
            timestamp=manifest_dummy.timestamp,
            root_pubkey_hex=self.root_pub.hex(),
            ed25519_signature_hex=sig.hex(),
            attestations=attestations,
        )

        ok, msg = self.verifier.verify_manifest(manifest, current_seq=0)
        self.assertFalse(ok)
        self.assertIn("insufficient_distinct_asns", msg)


class TestEndToEndDHTSyncEngine(unittest.TestCase):
    """Test Suite 5: End-to-end sync, publisher-subscriber pipeline, purge, and modes."""

    def setUp(self) -> None:
        self.root_priv, self.root_pub = generate_ed25519_keypair()
        self.master_secret = os.urandom(32)

        # Publisher instance
        self.publisher = ArmigerDHTDatabaseSync(
            subscription=SubscriptionConfig(
                categories={"movie", "tv_show", "anime"},
                languages={"en", "ja"},
                quality_tiers={"uhd_4k", "hd_1080p"},
                epochs={"2026_q4"},
                enable_xxx=False,
                sync_mode=SyncMode.PUBLISHER,
            ),
            master_secret=self.master_secret,
            trusted_root_pubkeys_hex=[self.root_pub.hex()],
        )

        # Subscriber instance (Only subscribes to anime in ja)
        self.subscriber = ArmigerDHTDatabaseSync(
            subscription=SubscriptionConfig(
                categories={"anime"},
                languages={"ja"},
                quality_tiers={"hd_1080p"},
                epochs={"2026_q4"},
                enable_xxx=False,
                sync_mode=SyncMode.PULL_ONLY,
                min_quorum=5,
                min_distinct_asns=5,
            ),
            master_secret=self.master_secret,
            trusted_root_pubkeys_hex=[self.root_pub.hex()],
        )

    def tearDown(self) -> None:
        if hasattr(self, "publisher"):
            self.publisher.close()
        if hasattr(self, "subscriber"):
            self.subscriber.close()

    def test_full_publish_subscribe_and_shard_application(self) -> None:
        # Ingest 3 anime records on publisher
        records_to_shard = []
        for i in range(1, 4):
            ok, msg, rec = self.publisher.ingest_candidate_record(
                infohash=f"abcdef{i:034x}",
                title=f"Wing It Open Series S04E{i:02d} 1080p",
                category="anime",
                language="ja",
                quality_tier="hd_1080p",
                epoch="2026_q4",
                size_bytes=1500000000,
                file_count=1,
                seeders=200,
                leechers=15,
                p_adult=0.01,
                p_junk=0.001,
                p_excluded_scope=0.01,
            )
            self.assertTrue(ok)
            self.assertIsNotNone(rec)
            records_to_shard.append(rec)

        # Build and sign shard on publisher
        shard_id = "anime/ja/hd_1080p/2026_q4"
        manifest, bundle = self.publisher.build_and_sign_shard(
            shard_id=shard_id,
            records=records_to_shard,
            root_privkey=self.root_priv,
            seq=1,
        )

        # Add 5 distinct ASN attestations over the canonical manifest digest
        distinct_asns = [15169, 13335, 16509, 7922, 3356]
        attestations = create_mock_quorum_attestations(manifest.canonical_digest(), distinct_asns)
        manifest_with_quorum = manifest.with_attestations(attestations)

        # Subscriber applies shard
        ok, msg, count = self.subscriber.receive_and_apply_shard(manifest_with_quorum, bundle)
        self.assertTrue(ok, f"Subscriber failed to apply shard: {msg}")
        self.assertEqual(count, 3)

        # Verify records exist in subscriber database
        sub_records = self.subscriber.db.search("Wing It Open Series")
        self.assertEqual(len(sub_records), 3)

        # Test tombstone purge on subscriber
        tombstone_manifest = sign_tombstone_list(
            seq=1,
            infohashes_or_blinded=[records_to_shard[0].infohash],
            private_key_bytes=self.root_priv,
        )
        ok_tomb, msg_tomb, new_tomb, purged = self.subscriber.apply_signed_tombstones(tombstone_manifest)
        self.assertTrue(ok_tomb)
        self.assertEqual(new_tomb, 1)
        self.assertEqual(purged, 1)

        # Database should now have only 2 records
        remaining = self.subscriber.db.search("Wing It Open Series")
        self.assertEqual(len(remaining), 2)
        self.assertNotIn(records_to_shard[0].infohash, [r.infohash for r in remaining])

    def test_sync_mode_off_disables_all_operations(self) -> None:
        self.subscriber.set_sync_mode(SyncMode.OFF)

        # Candidate ingestion blocked
        ok1, msg1, _ = self.subscriber.ingest_candidate_record(
            infohash="1" * 40,
            title="Some Movie",
            category="movie",
            language="en",
            quality_tier="hd_1080p",
        )
        self.assertFalse(ok1)
        self.assertEqual(msg1, "sync_mode_off")

        # Shard application blocked
        dummy_manifest = DHTShardManifest(
            seq=1,
            shard_id="anime/ja/hd_1080p/2026_q4",
            epoch="2026_q4",
            record_count=1,
            compressed_size_bytes=10,
            uncompressed_size_bytes=10,
            merkle_root="0" * 64,
            ciphertext_sha256="0" * 64,
            bep46_infohash="0" * 40,
            timestamp=100,
            root_pubkey_hex=self.root_pub.hex(),
            ed25519_signature_hex="0" * 128,
        )
        ok2, msg2, count2 = self.subscriber.receive_and_apply_shard(dummy_manifest, b"")
        self.assertFalse(ok2)
        self.assertEqual(msg2, "sync_mode_off")

    def test_untrusted_root_key_rejected(self) -> None:
        untrusted_priv, untrusted_pub = generate_ed25519_keypair()
        manifest, bundle = self.publisher.build_and_sign_shard(
            shard_id="anime/ja/hd_1080p/2026_q4",
            records=[],
            root_privkey=untrusted_priv,
            seq=1,
        )
        distinct_asns = [15169, 13335, 16509, 7922, 3356]
        attestations = create_mock_quorum_attestations(manifest.canonical_digest(), distinct_asns)
        manifest_with_quorum = manifest.with_attestations(attestations)

        ok, msg, count = self.subscriber.receive_and_apply_shard(manifest_with_quorum, bundle)
        self.assertFalse(ok)
        self.assertIn("untrusted_root_pubkey", msg)

    def test_stale_or_downgrade_sequence_rejected(self) -> None:
        shard_id = "anime/ja/hd_1080p/2026_q4"
        manifest1, bundle1 = self.publisher.build_and_sign_shard(
            shard_id=shard_id,
            records=[],
            root_privkey=self.root_priv,
            seq=5,
        )
        distinct_asns = [15169, 13335, 16509, 7922, 3356]
        attestations1 = create_mock_quorum_attestations(manifest1.canonical_digest(), distinct_asns)
        manifest1 = manifest1.with_attestations(attestations1)

        ok1, _, _ = self.subscriber.receive_and_apply_shard(manifest1, bundle1)
        self.assertTrue(ok1)

        # Re-send seq=5 (replay) or seq=4 (downgrade)
        manifest_stale, bundle_stale = self.publisher.build_and_sign_shard(
            shard_id=shard_id,
            records=[],
            root_privkey=self.root_priv,
            seq=5,
        )
        attestations_stale = create_mock_quorum_attestations(manifest_stale.canonical_digest(), distinct_asns)
        manifest_stale = manifest_stale.with_attestations(attestations_stale)

        ok2, msg2, _ = self.subscriber.receive_and_apply_shard(manifest_stale, bundle_stale)
        self.assertFalse(ok2)
        self.assertIn("stale_or_downgrade_sequence", msg2)

    def test_ciphertext_sha256_mismatch_rejected(self) -> None:
        shard_id = "anime/ja/hd_1080p/2026_q4"
        manifest, bundle = self.publisher.build_and_sign_shard(
            shard_id=shard_id,
            records=[],
            root_privkey=self.root_priv,
            seq=10,
        )
        distinct_asns = [15169, 13335, 16509, 7922, 3356]
        manifest = manifest.with_attestations(create_mock_quorum_attestations(manifest.canonical_digest(), distinct_asns))

        # Corrupted bundle bytes in transit
        corrupted_bundle = bundle[:-1] + bytes([bundle[-1] ^ 0xFF])
        ok, msg, _ = self.subscriber.receive_and_apply_shard(manifest, corrupted_bundle)
        self.assertFalse(ok)
        self.assertEqual(msg, "ciphertext_sha256_mismatch")

    def test_merkle_root_mismatch_rejected(self) -> None:
        shard_id = "anime/ja/hd_1080p/2026_q4"
        # Generate shard with 1 record
        rec = TorrentRecord(
            infohash="1" * 40,
            title="Anime Episode 1",
            category="anime",
            language="ja",
            quality_tier="hd_1080p",
            epoch="2026_q4",
            size_bytes=100,
            file_count=1,
            seeders=10,
            leechers=1,
            content_sha256="2" * 64,
            file_list=["ep1.mkv"],
            safety_gate_passed=True,
            created_at=100,
        )
        manifest, bundle = self.publisher.build_and_sign_shard(
            shard_id=shard_id,
            records=[rec],
            root_privkey=self.root_priv,
            seq=20,
        )
        # Forge manifest with incorrect merkle_root
        import dataclasses
        tampered_manifest = dataclasses.replace(manifest, merkle_root="9" * 64)
        sk = Ed25519PrivateKey.from_private_bytes(self.root_priv)
        new_sig = sk.sign(tampered_manifest.canonical_payload())
        pq_priv, _ = WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(self.root_priv)
        new_pq_sig = WOTSPlusSHAKE256.sign(pq_priv, tampered_manifest.canonical_payload())
        tampered_manifest = dataclasses.replace(
            tampered_manifest,
            ed25519_signature_hex=new_sig.hex(),
            pq_signature_hex=new_pq_sig.hex(),
        )

        distinct_asns = [15169, 13335, 16509, 7922, 3356]
        tampered_manifest = tampered_manifest.with_attestations(
            create_mock_quorum_attestations(tampered_manifest.canonical_digest(), distinct_asns)
        )

        ok, msg, _ = self.subscriber.receive_and_apply_shard(tampered_manifest, bundle)
        self.assertFalse(ok)
        self.assertIn("aead_decryption_failed", msg)  # Because AAD checks merkle_root!

    def test_pre_ingest_reinspection_drops_bad_records_in_shard(self) -> None:
        shard_id = "anime/ja/hd_1080p/2026_q4"
        # A shard contains 1 valid record and 1 disguised malware dropper
        rec_clean = TorrentRecord(
            infohash="clean" + "0" * 35,
            title="Clean Anime Movie",
            category="anime",
            language="ja",
            quality_tier="hd_1080p",
            epoch="2026_q4",
            size_bytes=100,
            file_count=1,
            seeders=10,
            leechers=1,
            content_sha256="1" * 64,
            file_list=["movie.mkv"],
            safety_gate_passed=True,
            created_at=100,
        )
        rec_malware = TorrentRecord(
            infohash="malware" + "0" * 33,
            title="Clean Anime Movie Part 2",
            category="anime",
            language="ja",
            quality_tier="hd_1080p",
            epoch="2026_q4",
            size_bytes=100,
            file_count=1,
            seeders=10,
            leechers=1,
            content_sha256="2" * 64,
            file_list=["movie.mkv.exe"],  # Disguised malware dropper
            safety_gate_passed=True,
            created_at=100,
        )
        manifest, bundle = self.publisher.build_and_sign_shard(
            shard_id=shard_id,
            records=[rec_clean, rec_malware],
            root_privkey=self.root_priv,
            seq=30,
        )
        distinct_asns = [15169, 13335, 16509, 7922, 3356]
        manifest = manifest.with_attestations(
            create_mock_quorum_attestations(manifest.canonical_digest(), distinct_asns)
        )

        ok, msg, count = self.subscriber.receive_and_apply_shard(manifest, bundle)
        self.assertTrue(ok)
        # Only clean record is ingested, malware dropped!
        self.assertEqual(count, 1)
        self.assertIsNotNone(self.subscriber.db.get_record(rec_clean.infohash))
        self.assertIsNone(self.subscriber.db.get_record(rec_malware.infohash))

    def test_x25519_ecdh_key_exchange(self) -> None:
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        alice_priv = X25519PrivateKey.generate()
        bob_priv = X25519PrivateKey.generate()

        alice_shared = alice_priv.exchange(bob_priv.public_key())
        bob_shared = bob_priv.exchange(alice_priv.public_key())
        self.assertEqual(alice_shared, bob_shared)

        key_alice = ShardCryptoEngine.derive_shard_key(alice_shared, "movie/en/hd_1080p/2026_q4", "2026_q4")
        key_bob = ShardCryptoEngine.derive_shard_key(bob_shared, "movie/en/hd_1080p/2026_q4", "2026_q4")
        self.assertEqual(key_alice, key_bob)

    def test_bep44_target_derivation(self) -> None:
        manifest_dummy = DHTShardManifest(
            seq=1,
            shard_id="movie/en/hd_1080p/2026_q4",
            epoch="2026_q4",
            record_count=0,
            compressed_size_bytes=0,
            uncompressed_size_bytes=0,
            merkle_root="0" * 64,
            ciphertext_sha256="0" * 64,
            bep46_infohash="0" * 40,
            timestamp=100,
            root_pubkey_hex=self.root_pub.hex(),
            ed25519_signature_hex="0" * 128,
        )
        target = manifest_dummy.bep44_target()
        self.assertEqual(len(target), 40)
        # Deterministic
        self.assertEqual(target, manifest_dummy.bep44_target())


class TestPhase2CatalogRedTeamAndBenchmarks(unittest.TestCase):
    """Test Suite 6 (Phase 2): 10,000-Item Catalog Sharding, 120+ Adversarial Red-Team Traps & Benchmarks."""

    def setUp(self) -> None:
        self.root_priv, self.root_pub = generate_ed25519_keypair()
        self.master_secret = os.urandom(32)
        self.tombstone_store = TombstoneStore(trusted_root_pubkeys_hex=[self.root_pub.hex()])
        self.guard = CSAMGuard(tombstone_store=self.tombstone_store)
        self.safety_gate = ArmigerSafetyGate()

    def test_10000_item_catalog_sharding_selective_sync_and_merkle_proofs(self) -> None:
        """Verify 10,000-item multi-category/multi-language sharding, selective sync, 0% xxx leakage, and 10k Merkle proofs."""
        categories = list(ALL_CATEGORIES)  # 10 categories (9 mainstream + 'xxx')
        languages = ["en", "ja", "es", "fr", "de"]  # 5 languages
        qualities = ["uhd_4k", "hd_1080p"]  # 2 quality tiers
        epoch = "2026_q4"

        publisher = ArmigerDHTDatabaseSync(
            subscription=SubscriptionConfig(
                categories=set(categories),
                languages=set(languages),
                quality_tiers=set(qualities),
                epochs={epoch},
                enable_xxx=True,
                sync_mode=SyncMode.PUBLISHER,
            ),
            master_secret=self.master_secret,
            trusted_root_pubkeys_hex=[self.root_pub.hex()],
        )

        subscriber = ArmigerDHTDatabaseSync(
            subscription=SubscriptionConfig(
                categories={"movie", "tv_show", "anime"},
                languages={"en", "ja"},
                quality_tiers={"uhd_4k", "hd_1080p"},
                epochs={epoch},
                enable_xxx=False,
                sync_mode=SyncMode.PULL_ONLY,
                min_quorum=5,
                min_distinct_asns=5,
            ),
            master_secret=self.master_secret,
            trusted_root_pubkeys_hex=[self.root_pub.hex()],
        )

        try:
            # 1. Generate 10,000 catalog items partitioned into 100 shards (100 items per shard)
            shards_map: dict[str, list[TorrentRecord]] = {}
            total_generated = 0

            for c_idx, cat in enumerate(categories):
                for l_idx, lang in enumerate(languages):
                    for q_idx, qual in enumerate(qualities):
                        shard_id = SubscriptionConfig.format_shard_id(cat, lang, qual, epoch)
                        shard_records: list[TorrentRecord] = []
                        for item_idx in range(100):
                            global_id = (c_idx * 1000) + (l_idx * 200) + (q_idx * 100) + item_idx + 1
                            infohash = f"{global_id:040x}"
                            rec = TorrentRecord(
                                infohash=infohash,
                                title=f"Catalog_{cat}_{lang}_{qual}_Item_{global_id:05d}",
                                category=cat,
                                language=lang,
                                quality_tier=qual,
                                epoch=epoch,
                                size_bytes=1_000_000 * (item_idx + 1),
                                file_count=1,
                                seeders=50 + item_idx,
                                leechers=5,
                                content_sha256=f"{global_id:064x}",
                                file_list=[f"item_{global_id:05d}.mkv"],
                                safety_gate_passed=True,
                                created_at=1700000000 + global_id,
                            )
                            shard_records.append(rec)
                            total_generated += 1
                        shards_map[shard_id] = shard_records

            self.assertEqual(total_generated, 10_000)
            self.assertEqual(len(shards_map), 100)

            # 2. Build & sign all 100 shards, verify 10,000 Merkle inclusion proofs, and feed to subscriber
            distinct_asns = [15169, 13335, 16509, 7922, 3356]
            reusable_node_keys = [generate_ed25519_keypair() for _ in distinct_asns]

            merkle_proofs_verified = 0
            subscribed_shards_applied = 0
            unsubscribed_shards_skipped = 0
            records_ingested = 0

            for shard_id, recs in shards_map.items():
                # Verify binary Merkle tree inclusion proof for every record in the shard
                sorted_recs = sorted(recs, key=lambda r: r.infohash.lower())
                leaves = [r.leaf_hash() for r in sorted_recs]
                tree = MerkleTree(leaves)
                for idx, r in enumerate(sorted_recs):
                    proof = tree.get_proof(idx)
                    self.assertTrue(MerkleTree.verify_proof(r.leaf_hash(), proof, tree.root_hex))
                    merkle_proofs_verified += 1

                manifest, bundle = publisher.build_and_sign_shard(
                    shard_id=shard_id,
                    records=recs,
                    root_privkey=self.root_priv,
                    seq=1,
                )
                digest = manifest.canonical_digest()
                attestations = []
                for asn, (n_priv, n_pub) in zip(distinct_asns, reusable_node_keys):
                    sk = Ed25519PrivateKey.from_private_bytes(n_priv)
                    sig = sk.sign(digest)
                    attestations.append(
                        QuorumAttestation(
                            node_id=hashlib.sha1(n_pub + f"asn{asn}".encode()).hexdigest(),
                            asn=asn,
                            ip_digest=hashlib.sha256(f"198.51.100.{asn % 250}".encode()).hexdigest(),
                            pubkey_hex=n_pub.hex(),
                            signature_hex=sig.hex(),
                            timestamp=manifest.timestamp,
                        )
                    )
                manifest_quorum = manifest.with_attestations(attestations)

                ok, msg, count = subscriber.receive_and_apply_shard(manifest_quorum, bundle)
                cat, lang, qual, _ = SubscriptionConfig.parse_shard_id(shard_id)
                expected_match = (cat in {"movie", "tv_show", "anime"}) and (lang in {"en", "ja"})
                if expected_match:
                    self.assertTrue(ok, f"Expected shard {shard_id} to be applied, got: {msg}")
                    self.assertEqual(count, 100)
                    subscribed_shards_applied += 1
                    records_ingested += count
                else:
                    self.assertFalse(ok)
                    self.assertIn("unsubscribed_shard_filtered", msg)
                    self.assertEqual(count, 0)
                    unsubscribed_shards_skipped += 1

            self.assertEqual(merkle_proofs_verified, 10_000)
            self.assertEqual(subscribed_shards_applied, 12)  # 3 categories * 2 languages * 2 qualities
            self.assertEqual(unsubscribed_shards_skipped, 88)
            self.assertEqual(records_ingested, 1_200)

            # 3. Verify SQLite state & 0% 'xxx' leakage
            self.assertEqual(subscriber.db.count_records(), 1_200)
            self.assertEqual(subscriber.db.count_records(category="movie"), 400)
            self.assertEqual(subscriber.db.count_records(category="tv_show"), 400)
            self.assertEqual(subscriber.db.count_records(category="anime"), 400)
            self.assertEqual(subscriber.db.count_records(category="xxx"), 0)

            cur = subscriber.db.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM mainstream_torrents WHERE category = 'xxx'")
            self.assertEqual(cur.fetchone()[0], 0)
            cur.execute("SELECT DISTINCT category, language FROM mainstream_torrents")
            pairs = {(r[0], r[1]) for r in cur.fetchall()}
            self.assertEqual(
                pairs,
                {
                    ("movie", "en"), ("movie", "ja"),
                    ("tv_show", "en"), ("tv_show", "ja"),
                    ("anime", "en"), ("anime", "ja"),
                },
            )
        finally:
            publisher.close()
            subscriber.close()

    def test_adversarial_csam_nonconsensual_malware_redteam_100_plus_traps(self) -> None:
        """Red-team 120+ obfuscated CSAM, non-consensual, extreme violence, malware, conformal, and tombstone traps."""
        total_traps_verified = 0

        # Surface A: 25 Unicode NFKD, Fullwidth, Mathematical & Cyrillic/Greek Homoglyph Traps
        homoglyph_traps = [
            "\uff23\uff33\uff21\uff2d Archive 2026",                         # Fullwidth CSAM
            "\uff23\uff33\uff21\uff25 Collection",                            # Fullwidth CSAE
            "\uff30\uff45\uff44\uff4f\uff50\uff48\uff49\uff4c\uff45",    # Fullwidth Pedophile
            "\U0001d429\U0001d41e\U0001d41d\U0001d428\U0001d429\U0001d421\U0001d422\U0001d425\U0001d41e",  # Math Bold
            "\U0001d450\U0001d455\U0001d456\U0001d459\U0001d451 \U0001d45d\U0001d45c\U0001d45f\U0001d45b",  # Math Italic child porn
            "\U0001d52d\U0001d52f\U0001d522\U0001d531\U0001d522\U0001d522\U0001d52b \U0001d530\U0001d522\U0001d535",  # Math Fraktur preteen sex
            "p\u00e9d\u00f6ph\u00efl\u00eb hidden pack",                      # Diacritic storm
            "pr\u00ebt\u00e9\u00e8n s\u00eax leak",                           # Diacritics preteen sex
            "l\u00f6l\u00eft\u00e4 h\u00e4rdc\u00f6r\u00eb",                # Diacritics lolita hardcore
            "\u0440\u0435d\u043e\u0440hil\u0435 underground",                # Cyrillic р, е, о
            "\u0441hild \u0440\u043ern video",                                 # Cyrillic с, р, о
            "l\u043elit\u0430 h\u0430rd\u0441\u043er\u0435",                # Cyrillic о, а, с, е
            "\u0455h\u043et\u0430\u0441\u043en collection",                  # Cyrillic ѕ, о, а, с
            "t\u043eddler \u0440\u043ern archive",                             # Cyrillic о, р
            "kinderg\u0430rten \u0455\u0435\u0445",                           # Cyrillic а, ѕ, е, х
            "inf\u0430nt \u0435r\u043eti\u0441",                              # Cyrillic а, е, о, с
            "und\u0435r\u0430ge \u0440\u043ern",                              # Cyrillic е, а, р, о
            "j\u0430ilb\u0430it pack",                                          # Cyrillic а
            "in\u0441\u0435\u0455t \u0441hild",                               # Cyrillic с, е, ѕ
            "r\u0430\u0440\u0435 vid\u0435\u043e",                           # Cyrillic а, р, е, о
            "r\u0435v\u0435ng\u0435 \u0440\u043ern",                         # Cyrillic е, р, о
            "\u0455\u0440\u0443\u0441\u0430m u\u0440\u0455kirt",           # Cyrillic ѕ, р, у, с, а
            "\u0455nuff film uncensored",                                        # Cyrillic ѕ
            "r\u0435\u0430l b\u0435h\u0435\u0430ding",                       # Cyrillic е, а
            "\u03c1\u03b5d\u03bf\u03c1hil\u03b5 archive",                    # Greek ρ, ε, ο
        ]
        self.assertEqual(len(homoglyph_traps), 25)
        for sample in homoglyph_traps:
            res = self.guard.inspect_torrent("a" * 40, sample, category="movie")
            self.assertFalse(res.is_safe, f"Homoglyph trap bypassed guard: {sample!r}")
            total_traps_verified += 1

        # Surface B: 25 Zero-Width Joiner, Soft Hyphen & Bidirectional Override Traps
        zw_bidi_traps = [
            "c\u200bs\u200ca\u200dm leak",
            "c\u2060s\u2066a\u2069e dump",
            "p\u200be\u200cd\u200do\u200ep\u200fh\u202ai\u202el\u202ce",
            "ch\u200bild p\u200dorn",
            "cp\u200b v\u200cids",
            "cp\u2060 l\u2060inks",
            "cp\ufeff m\ufeffegas",
            "ch\u00adild er\u00adotic",
            "under\u200bage p\u200dorn",
            "pre\u200bteen s\u200dex",
            "pre\u202eteen n\u202cude",
            "loli\u200bta hard\u200ccore",
            "loli\u200dta hen\u200btai",
            "sho\u200btacon archive",
            "todd\u200bler p\u200dorn",
            "kinder\u200bgarten s\u200cex",
            "in\u200bfant n\u200dude",
            "elemen\u200btary er\u200cotic",
            "jail\u200bbait collection",
            "in\u200bcest mi\u200dnor",
            "non\u200bconsensual leak",
            "rape\u200b v\u200dideo",
            "revenge\u200b p\u200corn",
            "snuff\u200b f\u200dilm",
            "real\u200b ex\u202eecution",
        ]
        self.assertEqual(len(zw_bidi_traps), 25)
        for sample in zw_bidi_traps:
            res = self.guard.inspect_torrent("b" * 40, sample, category="movie")
            self.assertFalse(res.is_safe, f"Zero-width/Bidi trap bypassed guard: {sample!r}")
            total_traps_verified += 1

        # Surface C: 25 Heavy Leetspeak, Character Repetition & Delimiter Spacing Traps
        leet_traps = [
            "P3d0ph1l14_v1ds",
            "c_h_1_l_d__p_0_r_n",
            "pr3333t333n s333x",
            "l0l1t@ h@rdc0r3",
            "j@1lb@1t_2026",
            "r3v3ng3 p0rn",
            "5nuff f1lm",
            "b1@ckm@1l s3x",
            "5pyc@m uP5k1rt",
            "g@ng r@p3",
            "f0rc3d s3x",
            "c.s.a.m.pack",
            "c-s-a-e-collection",
            "k1nd3rg@rt3n p0rn",
            "t0ddl3r s3x",
            "1nf@nt p0rn",
            "3l3m3nt@ry 3r0t1c",
            "53xt0rt10n_leaks",
            "r3@l b3h3@d1ng",
            "r3@l 3x3cut10n",
            "l1v3 5u1c1d3 v1d30",
            "m@55@cr3 l1v35tr3@m",
            "r3@l t0rtur3 f1lm",
            "cru5h f3t15h k1ll",
            "1nc35t ch1ld",
        ]
        self.assertEqual(len(leet_traps), 25)
        for sample in leet_traps:
            res = self.guard.inspect_torrent("c" * 40, sample, category="movie")
            self.assertFalse(res.is_safe, f"Leetspeak trap bypassed guard: {sample!r}")
            total_traps_verified += 1

        # Surface D: 25 Nested Archive & Double-Extension Disguised Executable Malware Traps
        malware_file_traps = [
            ("movie", ["Movie.2026.1080p.mp4.exe"]),
            ("tv_show", ["Series.S01E01.mkv.scr"]),
            ("anime", ["Episode_01.avi.bat"]),
            ("book", ["Complete_Guide.pdf.vbs"]),
            ("music", ["01_Lossless_Track.flac.ps1"]),
            ("music", ["Album_Track_02.mp3.cmd"]),
            ("book", ["Novel_Edition.epub.hta"]),
            ("comic", ["Issue_001.cbz.jar"]),
            ("movie", ["Trailer_4K.mov.msi"]),
            ("comic", ["Cover_Art.jpg.pif"]),
            ("tv_show", ["Subtitles_EN.srt.wsf"]),
            ("music", ["Studio_Master.wav.vbe"]),
            ("music", ["Podcast_Ep1.m4a.jse"]),
            ("movie", ["WebRip_Stream.webm.com"]),
            ("tv_show", ["Remux_2160p.mkv.cpl"]),
            ("anime", ["OVA_Special.avi.wsh"]),
            ("movie", ["Feature_Film.mp4.psm1"]),
            ("book", ["Reference_Manual.pdf.application"]),
            ("music", ["Bonus_Track.flac.gadget"]),
            ("tv_show", ["Finale_1080p.mkv.msp"]),
            ("Archive.zip/Nested/Video.mp4.exe", ["Archive.zip/Nested/Video.mp4.exe"]),
            ("Season1.rar/S01E02.mkv.scr", ["Season1.rar/S01E02.mkv.scr"]),
            ("Pack.7z/Clip.avi.bat", ["Pack.7z/Clip.avi.bat"]),
            ("Library.tar.gz/Book.pdf.vbs", ["Library.tar.gz/Book.pdf.vbs"]),
            ("Disc.iso/Audio/01.flac.ps1", ["Disc.iso/Audio/01.flac.ps1"]),
        ]
        self.assertEqual(len(malware_file_traps), 25)
        for cat_or_path, flist in malware_file_traps:
            cat = cat_or_path if cat_or_path in ALL_CATEGORIES else None
            res = self.guard.inspect_torrent(
                infohash="d" * 40,
                title="Clean_Release_Title_2026",
                file_list=flist,
                category=cat,
            )
            self.assertFalse(res.is_safe, f"Malware dropper bypassed guard: {flist}")
            self.assertEqual(res.violation_type, CSAMViolationType.MALWARE_PAYLOAD)
            total_traps_verified += 1

        # Surface E: 20 Mondrian Conformal Safety Gate Boundary Traps
        # E1: Safe just below thresholds
        ok_safe, _, _ = self.safety_gate.evaluate(p_adult=0.5450, p_junk=0.1009, p_excluded_scope=0.4769, target_category="movie")
        self.assertTrue(ok_safe)

        conformal_traps = [
            # 5 junk boundary & high-risk cases (threshold = 0.1010)
            (0.01, 0.1010, 0.01, "movie", CSAMViolationType.SAFETY_GATE_JUNK),
            (0.01, 0.1011, 0.01, "tv_show", CSAMViolationType.SAFETY_GATE_JUNK),
            (0.01, 0.1500, 0.01, "anime", CSAMViolationType.SAFETY_GATE_JUNK),
            (0.01, 0.5000, 0.01, "software", CSAMViolationType.SAFETY_GATE_JUNK),
            (0.01, 0.9900, 0.01, "game", CSAMViolationType.SAFETY_GATE_JUNK),
            # 6 scope boundary & high-risk cases (threshold = 0.4770)
            (0.01, 0.01, 0.4770, "movie", CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE),
            (0.01, 0.01, 0.4771, "tv_show", CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE),
            (0.01, 0.01, 0.5500, "music", CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE),
            (0.01, 0.01, 0.7000, "book", CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE),
            (0.01, 0.01, 0.8500, "comic", CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE),
            (0.01, 0.01, 0.9900, "course", CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE),
            # 9 unisolated adult boundary cases across all 9 mainstream categories (threshold = 0.5451)
            (0.5451, 0.01, 0.01, "movie", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "tv_show", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "music", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "game", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "software", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "anime", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "book", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "comic", CSAMViolationType.UNISOLATED_ADULT),
            (0.5451, 0.01, 0.01, "course", CSAMViolationType.UNISOLATED_ADULT),
        ]
        self.assertEqual(len(conformal_traps), 20)
        for p_a, p_j, p_s, cat, expected_vtype in conformal_traps:
            ok, reason, vtype = self.safety_gate.evaluate(p_a, p_j, p_s, cat)
            self.assertFalse(ok, f"Conformal trap bypassed gate: {(p_a, p_j, p_s, cat)}")
            self.assertEqual(vtype, expected_vtype)
            total_traps_verified += 1

        # Verify total adversarial traps >= 120
        self.assertGreaterEqual(total_traps_verified, 120)

        # Surface F: Live Ed25519-Signed Blinded SHA-256 Tombstone Purge Propagation & Permanent Block
        engine = ArmigerDHTDatabaseSync(
            subscription=SubscriptionConfig(
                categories={"movie"},
                languages={"en"},
                quality_tiers={"hd_1080p"},
                epochs={"2026_q4"},
                enable_xxx=False,
                sync_mode=SyncMode.PUBLISHER,
            ),
            master_secret=self.master_secret,
            trusted_root_pubkeys_hex=[self.root_pub.hex()],
        )
        try:
            all_hashes = [f"{i:040x}" for i in range(500, 550)]
            for idx, h in enumerate(all_hashes):
                ok, _, _ = engine.ingest_candidate_record(
                    infohash=h,
                    title=f"Verified_Mainstream_Film_{idx}",
                    category="movie",
                    language="en",
                    quality_tier="hd_1080p",
                    epoch="2026_q4",
                )
                self.assertTrue(ok)
            self.assertEqual(engine.db.count_records(), 50)

            revoked_hashes = all_hashes[:15]
            tomb_manifest = sign_tombstone_list(
                seq=10,
                infohashes_or_blinded=revoked_hashes,
                private_key_bytes=self.root_priv,
            )
            ok_t, msg_t, new_t, purged_t = engine.apply_signed_tombstones(tomb_manifest)
            self.assertTrue(ok_t)
            self.assertEqual(new_t, 15)
            self.assertEqual(purged_t, 15)
            self.assertEqual(engine.db.count_records(), 35)

            # Confirm permanent re-ingest block on all 15 tombstoned infohashes
            for rh in revoked_hashes:
                ok_re, msg_re, _ = engine.ingest_candidate_record(
                    infohash=rh,
                    title="Innocent_Renamed_Film",
                    category="movie",
                    language="en",
                    quality_tier="hd_1080p",
                    epoch="2026_q4",
                )
                self.assertFalse(ok_re)
                self.assertIn("tombstoned_hash", msg_re)
        finally:
            engine.close()

    def test_throughput_and_latency_benchmarks(self) -> None:
        """Measure CSAM Guard items/sec, AEAD encryption/decryption MB/s, and SQLite batch ingest speed."""
        import time

        # 1. CSAM Guard inspection throughput (5,000 items)
        n_inspect = 5_000
        sample_titles = [
            ("Clean.Movie.2026.2160p.WEB-DL.DDP5.1.Atmos.H.265", ["Movie.2026.2160p.mkv", "subs.srt"], True),
            ("Ch1ld_p0rn_hidden_archive_2026", ["archive.zip"], False),
            ("Disguised.Release.1080p", ["Release.1080p.mp4.exe"], False),
            ("Anime.Series.S02E12.1080p.HEVC", ["Anime.S02E12.mkv"], True),
            ("Secret_Bathroom_Spycam_Upskirt", ["clip.mp4"], False),
        ]
        t0 = time.perf_counter()
        for i in range(n_inspect):
            title, flist, expected_safe = sample_titles[i % len(sample_titles)]
            res = self.guard.inspect_torrent(f"{i + 10000:040x}", title, file_list=flist, category="movie")
            self.assertEqual(res.is_safe, expected_safe)
        t_inspect = time.perf_counter() - t0
        inspect_rate = n_inspect / max(t_inspect, 1e-9)

        # 2. AEAD Encryption & Decryption Throughput (4 MB payload x 5 iterations = 20 MB)
        shard_id = "movie/en/uhd_4k/2026_q4"
        epoch = "2026_q4"
        merkle_root = "a" * 64
        key = ShardCryptoEngine.derive_shard_key(self.master_secret, shard_id, epoch)
        payload_4mb = os.urandom(4 * 1024 * 1024)
        iters = 5
        total_mb = (len(payload_4mb) * iters) / (1024 * 1024)

        t_enc0 = time.perf_counter()
        bundles = [
            ShardCryptoEngine.encrypt_shard(payload_4mb, key, shard_id, epoch, seq=i + 1, merkle_root=merkle_root)
            for i in range(iters)
        ]
        t_enc = time.perf_counter() - t_enc0
        enc_mbps = total_mb / max(t_enc, 1e-9)

        t_dec0 = time.perf_counter()
        for i, b in enumerate(bundles):
            pt = ShardCryptoEngine.decrypt_shard(b, key, shard_id, epoch, seq=i + 1, merkle_root=merkle_root)
            self.assertEqual(len(pt), len(payload_4mb))
        t_dec = time.perf_counter() - t_dec0
        dec_mbps = total_mb / max(t_dec, 1e-9)

        # 3. SQLite Batch Ingest Throughput (10,000 records)
        n_sqlite = 10_000
        batch_records = [
            TorrentRecord(
                infohash=f"{i + 100000:040x}",
                title=f"Benchmark_Record_{i:05d}",
                category="movie",
                language="en",
                quality_tier="hd_1080p",
                epoch="2026_q4",
                size_bytes=2_000_000_000,
                file_count=2,
                seeders=100,
                leechers=10,
                content_sha256=f"{i + 100000:064x}",
                file_list=[f"movie_{i:05d}.mkv", f"movie_{i:05d}.srt"],
                safety_gate_passed=True,
                created_at=1700000000 + i,
            )
            for i in range(n_sqlite)
        ]
        engine = ArmigerDHTDatabaseSync(
            subscription=SubscriptionConfig(enable_xxx=False),
            master_secret=self.master_secret,
            trusted_root_pubkeys_hex=[self.root_pub.hex()],
        )
        try:
            t_db0 = time.perf_counter()
            inserted = engine.db.insert_records_batch(batch_records)
            t_db = time.perf_counter() - t_db0
            sqlite_rate = inserted / max(t_db, 1e-9)
            self.assertEqual(inserted, n_sqlite)
            self.assertEqual(engine.db.count_records(), n_sqlite)
        finally:
            engine.close()

        # Assert conservative floors to catch performance regressions across hosts
        self.assertGreater(inspect_rate, 5_000.0)
        self.assertGreater(enc_mbps, 50.0)
        self.assertGreater(dec_mbps, 50.0)
        self.assertGreater(sqlite_rate, 20_000.0)
        print(
            f"\n[BENCHMARK] CSAMGuard={inspect_rate:,.0f} items/s ({1e6/inspect_rate:.1f} us/item) | "
            f"AEAD_Enc={enc_mbps:,.1f} MB/s | AEAD_Dec={dec_mbps:,.1f} MB/s | "
            f"SQLite_Batch={sqlite_rate:,.0f} records/s"
        )


if __name__ == "__main__":
    unittest.main()
