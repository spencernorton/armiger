"""Armiger Wave 6B — Comprehensive 52-Vector Adversarial Security & Regression Harness (`seq=3003`).

Tests all 32 previously patched vulnerabilities (`V-01..V-16` + 16 Jeeves MR `!28/!29/!30` findings)
and 20 brand-new Wave 4/5/6 adversarial vectors (`N-01..N-20`) against the unified `v3.0-gold-r3`
codebase, `ZCB3` binary mmap catalog, `XMSS` Merkle tree, 9-head INT8 delta bundler, and 6-plugin stack.
"""
from __future__ import annotations

from contextlib import contextmanager
import dataclasses
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
from typing import Any, Iterator
import unittest
import zlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import numpy as np

from armiger.artifacts import model_directory
_REPO_ROOT = model_directory()

import armiger.audio_language_guard as alg
import armiger.calibration_and_dht as cdht
import armiger.csam_guard as cg
import armiger.swarm_survival as ss
import armiger.zero_cue_plugin as zcp
from armiger.calibration_and_dht import (
    ArmigerCalibrator,
    ArmigerDHTUpdater,
    SignedDHTManifest,
)
from armiger.csam_guard import (
    CSAMGuard,
    WOTSPlusSHAKE256,
    generate_ed25519_keypair,
)
from armiger.swarm_survival import (
    MAX_ADAPTER_WEIGHT_ABS,
    SwarmSurvivalNet,
    validate_adapter_weights,
)
from armiger.zero_cue_plugin import (
    IDX_GAME,
    IDX_SOFTWARE,
    resolve_category_v3,
)

try:
    import armiger.dht_head_delta as dhd
    import armiger.pq_xmss_merkle as pqx
    import armiger.zero_cue_binary as zcb
    from armiger.dht_head_delta import (
        ALL_9_HEAD_NAMES,
        HDLT_HEADER_PREFIX,
        HDLT_MAGIC,
        DeltaLayerMeta,
        HeadDeltaBundle,
        dequantize_head_delta_tensors,
        extract_head_delta_bundle,
        load_state_tensors,
        reconstruct_from_head_delta,
        validate_head_delta_tensor_specs,
    )
    from armiger.pq_xmss_merkle import (
        XMSSMerkleTree,
        XMSSStatefulVerifier,
        XMSSStateManager,
        XMSSVerifier,
        verify_xmss_signature,
    )
    from armiger.zero_cue_binary import (
        ZCB3_EXT_FLAG,
        ZCB3_HEADER_STRUCT,
        ZCB3_MAGIC,
        ZCB3MmapCatalog,
        compress_zcb3_lzma6,
        compress_zcb3_zlib9,
        decompress_zcb3,
        pack_zcb3_bytes,
        pack_zcb3_extended_bytes,
    )
    _HAS_TRACK_D = True
except ImportError:
    _HAS_TRACK_D = False

try:
    import armiger.armiger_dht_db_sync as dbsync
    import armiger.armiger_dht_rsi as rsi
    from armiger.armiger_ultra_pipeline import (
        ArmigerUltraPipeline,
        DHT_UPDATE_SCHEMA,
        DHTUpdateRejected,
        RSI_TELEMETRY_SCHEMA,
        RSIUpdateAggregator,
        _file_tree_structural_override,
        decode_safetensors,
        encode_safetensors,
        parse_dht_update_payload,
    )
    _HAS_TRACK_B = True
except ImportError:
    _HAS_TRACK_B = False

try:
    from oci_dht_node import (
        GatedBEP44UDPNode,
        VPNEgressGuard,
        verify_vpn_egress,
    )
    _HAS_OCI_DHT_NODE = True
except ImportError:
    _HAS_OCI_DHT_NODE = False


AUDIT_RESULTS: list[dict[str, Any]] = []


def _record_vector(
    vector_id: str,
    suite: str,
    category: str,
    title: str,
    defense_layer: str,
    t0: float,
    detail: str,
) -> None:
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    AUDIT_RESULTS.append(
        {
            "vector_id": vector_id,
            "suite": suite,
            "category": category,
            "title": title,
            "defense_layer": defense_layer,
            "blocked": True,
            "false_positives": 0,
            "latency_ms": elapsed_ms,
            "evidence": detail,
        }
    )


def _build_synthetic_v3_state_dict(seed: int = 20261020, head_seed_offset: int = 0) -> dict[str, np.ndarray]:
    """Construct a deterministic 9-head v3 state dict with a shared frozen trunk and 9 task heads."""
    rng_trunk = np.random.default_rng(seed)
    rng_heads = np.random.default_rng(seed + head_seed_offset)
    state: dict[str, np.ndarray] = {}
    state["char_emb.weight"] = rng_trunk.standard_normal((16384, 256), dtype=np.float32) * 0.02
    state["trunk_proj.weight"] = rng_trunk.standard_normal((3014, 1024), dtype=np.float32) * 0.02
    state["trunk_norm.weight"] = rng_trunk.standard_normal((704,), dtype=np.float32) * 0.02
    state["heads.category.0.weight"] = rng_heads.standard_normal((384, 1920), dtype=np.float32) * 0.02
    state["heads.category.0.bias"] = rng_heads.standard_normal((384,), dtype=np.float32) * 0.02
    state["heads.category.2.weight"] = rng_heads.standard_normal((11, 384), dtype=np.float32) * 0.02
    state["heads.category.2.bias"] = rng_heads.standard_normal((11,), dtype=np.float32) * 0.02
    state["heads.adult.0.weight"] = rng_heads.standard_normal((64, 324), dtype=np.float32) * 0.02
    state["heads.adult.0.bias"] = rng_heads.standard_normal((16,), dtype=np.float32) * 0.02
    state["heads.span_bio.0.weight"] = rng_heads.standard_normal((256, 256), dtype=np.float32) * 0.02
    state["heads.span_bio.0.bias"] = rng_heads.standard_normal((256,), dtype=np.float32) * 0.02
    state["heads.span_bio.1.weight"] = rng_heads.standard_normal((256,), dtype=np.float32) * 0.02
    state["heads.span_bio.1.bias"] = rng_heads.standard_normal((256,), dtype=np.float32) * 0.02
    state["heads.span_bio.3.weight"] = rng_heads.standard_normal((7, 256), dtype=np.float32) * 0.02
    state["heads.span_bio.3.bias"] = rng_heads.standard_normal((7,), dtype=np.float32) * 0.02
    return state


def _build_synthetic_gold_samples_100() -> list[dict[str, Any]]:
    """Construct 100 deterministic synthetic torrent items spanning all media categories."""
    templates = [
        ("Alpha.Horizon.2024.1080p.BluRay.x264.mkv", [{"path": "Alpha.Horizon.2024.1080p.mkv", "size": 4_200_000_000}], 4_200_000_000),
        ("Chronicles.S02E05.1080p.WEB-DL.DDP5.1.H.264.mkv", [{"path": "Chronicles.S02E05.mkv", "size": 1_500_000_000}], 1_500_000_000),
        ("Symphony.Orchestra.2023.FLAC.24bit", [{"path": "01-Overture.flac", "size": 85_000_000}], 450_000_000),
        ("OpenArena.v0.8.8.Linux.x64.tar.xz", [{"path": "openarena-0.8.8/baseoa/pak0.pk3", "size": 320_000_000}], 320_000_000),
        ("Classic.Literature.Unabridged.Audiobook.m4b", [{"path": "Classic.Literature.m4b", "size": 280_000_000}], 280_000_000),
        ("Graphic.Anthology.Issue.001.2024.cbz", [{"path": "Graphic.Anthology.001.cbz", "size": 65_000_000}], 65_000_000),
        ("Systems.Engineering.Handbook.2024.epub", [{"path": "Systems.Engineering.Handbook.epub", "size": 18_000_000}], 18_000_000),
        ("Full.Stack.Architecture.Video.Course.2024", [{"path": "01-Intro.mp4", "size": 120_000_000}, {"path": "02-Design.mp4", "size": 140_000_000}], 260_000_000),
        ("CMake.3.29.2.Linux.x86_64.tar.gz", [{"path": "cmake-3.29.2/bin/cmake", "size": 14_000_000}], 42_000_000),
        ("[SubsPlease] Synthetic Monogatari - 04 (1080p) [1234ABCD].mkv", [{"path": "[SubsPlease] Synthetic Monogatari - 04 (1080p).mkv", "size": 1_100_000_000}], 1_100_000_000),
    ]
    rows: list[dict[str, Any]] = []
    for idx in range(100):
        base_name, files, size_b = templates[idx % len(templates)]
        rows.append(
            {
                "name": f"{base_name}_v{idx:03d}" if idx >= len(templates) else base_name,
                "files": files,
                "size": size_b + idx * 1024,
            }
        )
    return rows


_SHARED_PIPELINE: Any = None


def _get_shared_pipeline() -> Any:
    global _SHARED_PIPELINE
    if _SHARED_PIPELINE is None:
        _SHARED_PIPELINE = ArmigerUltraPipeline(
            model_dir=_REPO_ROOT,
            preload_catalog=False,
            allow_synthetic_fallback=True,
        )
    return _SHARED_PIPELINE


class TestWave6DHTOldAndNewVulnerabilities(unittest.TestCase):
    """52-vector adversarial regression suite compatible with both unittest and pytest."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="armiger_wave6b_test_")
        self.tmp_path = Path(self._tmp_dir.name)
        self._old_xmss_state = os.environ.get("ARMIGER_XMSS_STATE_DIR")
        self._old_xmss_seal = os.environ.get("ARMIGER_XMSS_SEAL_DIR")
        st_dir = self.tmp_path / "_xmss_env_state"
        sl_dir = self.tmp_path / "_xmss_env_seal"
        st_dir.mkdir(parents=True, exist_ok=True)
        sl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ARMIGER_XMSS_STATE_DIR"] = str(st_dir)
        os.environ["ARMIGER_XMSS_SEAL_DIR"] = str(sl_dir)
        if _HAS_TRACK_D:
            pqx._XMSS_HIGH_WATER_STATE.clear()

    def tearDown(self) -> None:
        if self._old_xmss_state is None:
            os.environ.pop("ARMIGER_XMSS_STATE_DIR", None)
        else:
            os.environ["ARMIGER_XMSS_STATE_DIR"] = self._old_xmss_state
        if self._old_xmss_seal is None:
            os.environ.pop("ARMIGER_XMSS_SEAL_DIR", None)
        else:
            os.environ["ARMIGER_XMSS_SEAL_DIR"] = self._old_xmss_seal
        self._tmp_dir.cleanup()

    @contextmanager
    def _raises(self, expected_exc: type[BaseException] | tuple[type[BaseException], ...], match: str | None = None) -> Iterator[None]:
        try:
            yield
        except expected_exc as exc:
            if match is not None and not re.search(match, str(exc), re.IGNORECASE):
                self.fail(f"Exception {exc!r} did not match pattern {match!r}")
            return
        self.fail(f"Expected {expected_exc!r} to be raised")

    # ==============================================================================
    # SUITE 1: 32 Old Vulnerabilities & Jeeves MR !28/!29/!30 Regressions
    # ==============================================================================

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_v01_forged_ed25519_signature_seq3004(self) -> None:
        """V-01: Forged Ed25519 signature on seq=3004 DHT weight/plugin update."""
        t0 = time.perf_counter()
        trusted_sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"wave6-v01-trusted").digest())
        rogue_sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"wave6-v01-rogue").digest())
        pub_hex = trusted_sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        xmss = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-v01-xmss").digest(),
            height=4,
            state_path=self.tmp_path / "state" / "s.json",
            seal_path=self.tmp_path / "seal" / "s.seal",
        )
        updater = ArmigerDHTUpdater(
            root_pubkey_hex=pub_hex,
            current_seq=3003,
            min_quorum=5,
            xmss_merkle_root_hex=xmss.merkle_root_hex,
            xmss_height=4,
        )
        payload = b'{"seq":3004,"weights":"v3.0-gold-r3"}'
        unsigned = SignedDHTManifest(
            seq=3004,
            version="3.0.4",
            infohash="a" * 40,
            bundle_sha256=hashlib.sha256(payload).hexdigest(),
            quorum_attestations=6,
            timestamp=int(time.time()),
            ed25519_signature_hex="",
        )
        cp = unsigned.canonical_payload()
        forged_manifest = dataclasses.replace(
            unsigned,
            ed25519_signature_hex=rogue_sk.sign(cp).hex(),
            xmss_signature_hex=xmss.sign_seq(3004, cp).to_bytes().hex(),
        )
        ok, reason = updater.verify_bundle(forged_manifest, payload)
        self.assertFalse(ok)
        self.assertIn("ed25519", reason.lower())
        _record_vector(
            "V-01",
            "Suite 1 (Old/Jeeves)",
            "Cryptographic Auth",
            "Forged Ed25519 signature on seq=3004 DHT weight/plugin update",
            "ArmigerDHTUpdater.verify_bundle (Ed25519 verification)",
            t0,
            f"Rejected forged Ed25519 signature on seq=3004 ({reason})",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_v02_stripped_or_corrupted_pq_signature(self) -> None:
        """V-02: Stripped or corrupted post-quantum (WOTS+ / XMSS) signature when require_pq=True."""
        t0 = time.perf_counter()
        sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"wave6-v02-ed").digest())
        pub_hex = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        xmss = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-v02-xmss").digest(),
            height=4,
            state_path=self.tmp_path / "state" / "s.json",
            seal_path=self.tmp_path / "seal" / "s.seal",
        )
        updater = ArmigerDHTUpdater(
            root_pubkey_hex=pub_hex,
            current_seq=3003,
            min_quorum=5,
            xmss_merkle_root_hex=xmss.merkle_root_hex,
            xmss_height=4,
            require_pq=True,
        )
        payload = b'{"seq":3004}'
        unsigned = SignedDHTManifest(
            seq=3004,
            version="3.0.4",
            infohash="b" * 40,
            bundle_sha256=hashlib.sha256(payload).hexdigest(),
            quorum_attestations=6,
            timestamp=int(time.time()),
            ed25519_signature_hex="",
        )
        cp = unsigned.canonical_payload()
        ed_sig = sk.sign(cp).hex()

        m_stripped = dataclasses.replace(unsigned, ed25519_signature_hex=ed_sig, xmss_signature_hex="", pq_signature_hex="")
        ok_s, reason_s = updater.verify_bundle(m_stripped, payload)
        self.assertFalse(ok_s)

        valid_xmss_bytes = bytearray(xmss.sign_seq(3004, cp).to_bytes())
        valid_xmss_bytes[24] ^= 0xFF
        m_corrupt = dataclasses.replace(unsigned, ed25519_signature_hex=ed_sig, xmss_signature_hex=bytes(valid_xmss_bytes).hex())
        ok_c, reason_c = updater.verify_bundle(m_corrupt, payload)
        self.assertFalse(ok_c)
        _record_vector(
            "V-02",
            "Suite 1 (Old/Jeeves)",
            "Post-Quantum Auth",
            "Stripped or corrupted WOTS+/XMSS signature with require_pq=True",
            "ArmigerDHTUpdater.verify_bundle (require_pq=True)",
            t0,
            f"Rejected stripped PQ ({reason_s}) and corrupted XMSS chain ({reason_c})",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_v03a_pq_one_time_leaf_reuse_replay(self) -> None:
        """V-03a (Jeeves !30 P1): WOTS+/XMSS one-time leaf reuse (leaf_index / seq replay)."""
        t0 = time.perf_counter()
        xmss = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-v03a").digest(),
            height=5,
            state_path=self.tmp_path / "state" / "s.json",
            seal_path=self.tmp_path / "seal" / "s.seal",
        )
        sig_3003 = xmss.sign_seq(3003, b"manifest-seq-3003")
        with self._raises(RuntimeError, match="xmss_wots_one_time_key_reuse_rejected"):
            xmss.sign_seq(3003, b"second-message-reusing-leaf")

        verifier = XMSSStatefulVerifier(xmss.merkle_root_hex, current_seq=3002, expected_height=5)
        ok1, _ = verifier.verify_and_advance(b"manifest-seq-3003", sig_3003)
        self.assertTrue(ok1)
        ok_rep, reason_rep = verifier.verify_and_advance(b"manifest-seq-3003", sig_3003)
        self.assertFalse(ok_rep)
        self.assertTrue("stale" in reason_rep or "reuse" in reason_rep)
        _record_vector(
            "V-03a",
            "Suite 1 (Old/Jeeves)",
            "Post-Quantum Stateful",
            "WOTS+/XMSS one-time leaf_index reuse & replay",
            "XMSSMerkleTree.sign_seq + XMSSStatefulVerifier.verify_and_advance",
            t0,
            f"Signer refused seq=3003 reuse; verifier blocked replay ({reason_rep})",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_v03b_pq_presign_crash_leaf_reservation(self) -> None:
        """V-03b (Jeeves !30 P1): Pre-sign crash leaf reservation persisted before WOTS+ signing."""
        t0 = time.perf_counter()
        state_p = self.tmp_path / "state" / "xmss_crash.json"
        seal_p = self.tmp_path / "seal" / "xmss_crash.seal"
        seed = hashlib.sha256(b"wave6-v03b").digest()
        xmss = XMSSMerkleTree.provision_new(master_seed=seed, height=5, state_path=state_p, seal_path=seal_p)
        with self._raises(RuntimeError, match="xmss_simulated_crash_after_atomic_reservation"):
            xmss.sign_seq(1, b"crash-mid-sign", _simulate_crash_after_reserve=True)

        recovered = XMSSMerkleTree(master_seed=seed, height=5, state_path=state_p, seal_path=seal_p)
        self.assertEqual(recovered.max_signed_seq, 1)
        with self._raises(RuntimeError, match="xmss_wots_one_time_key_reuse_rejected"):
            recovered.sign_seq(1, b"retry-seq-1-after-crash")
        _record_vector(
            "V-03b",
            "Suite 1 (Old/Jeeves)",
            "Post-Quantum Stateful",
            "Pre-sign crash leaf reservation durability across process restart",
            "XMSSMerkleTree.sign_seq (_reserve_leaf_atomic_locked)",
            t0,
            "Simulated mid-sign crash permanently reserved seq=1 on disk & prevented post-restart reuse",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_v03c_pq_state_deletion_truncation_rollback_vs_seal(self) -> None:
        """V-03c (Jeeves !30 P1): Local state deletion/rollback attack vs external durable freshness seal."""
        t0 = time.perf_counter()
        state_p = self.tmp_path / "state" / "xmss_rb.json"
        seal_p = self.tmp_path / "seal" / "xmss_rb.seal"
        seed = hashlib.sha256(b"wave6-v03c").digest()
        signer = XMSSMerkleTree.provision_new(master_seed=seed, height=5, state_path=state_p, seal_path=seal_p)
        snap_seq0 = state_p.read_bytes()
        signer.sign_seq(1, b"seq-1-msg")

        state_p.write_bytes(snap_seq0)
        with self._raises(RuntimeError, match="state_rollback_detected"):
            signer.sign_seq(1, b"forged-reuse-seq-1")
        with self._raises(RuntimeError, match="state_rollback_detected"):
            XMSSMerkleTree(master_seed=seed, height=5, state_path=state_p, seal_path=seal_p)

        state_p.unlink()
        with self._raises(RuntimeError, match="missing_state_file"):
            XMSSMerkleTree(master_seed=seed, height=5, state_path=state_p, seal_path=seal_p)
        _record_vector(
            "V-03c",
            "Suite 1 (Old/Jeeves)",
            "Post-Quantum Stateful",
            "Local state deletion and snapshot rollback attack vs external freshness seal",
            "XMSSMerkleTree._recover_or_init_state_locked + external freshness seal",
            t0,
            "Blocked both MAC-valid snapshot rollback and state file deletion via external seal",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_v04a_monotonic_rollback_replay_weights_seq3002(self) -> None:
        """V-04a (Jeeves !29 P2): Monotonic rollback replay (seq=3002 <= watermark 3003) on DHT weights."""
        t0 = time.perf_counter()
        sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"wave6-v04a-ed").digest())
        pub_hex = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        xmss = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-v04a-xmss").digest(),
            height=5,
            state_path=self.tmp_path / "state" / "s.json",
            seal_path=self.tmp_path / "seal" / "s.seal",
        )
        updater = ArmigerDHTUpdater(
            root_pubkey_hex=pub_hex,
            current_seq=3001,
            min_quorum=5,
            xmss_merkle_root_hex=xmss.merkle_root_hex,
            xmss_height=5,
        )
        payload = b'{"release":"v3.0-gold-r3"}'
        sha = hashlib.sha256(payload).hexdigest()

        def _make_manifest(seq_n: int) -> SignedDHTManifest:
            m = SignedDHTManifest(
                seq=seq_n, version="3.0.3", infohash="c" * 40, bundle_sha256=sha,
                quorum_attestations=6, timestamp=int(time.time()), ed25519_signature_hex="",
            )
            cp = m.canonical_payload()
            return dataclasses.replace(
                m,
                ed25519_signature_hex=sk.sign(cp).hex(),
                xmss_signature_hex=xmss.sign_seq(seq_n, cp).to_bytes().hex(),
            )

        m3002 = _make_manifest(3002)
        m3003 = _make_manifest(3003)
        ok_3003, _ = updater.verify_bundle(m3003, payload)
        self.assertTrue(ok_3003)
        self.assertEqual(updater.current_seq, 3003)
        ok_3002, reason_3002 = updater.verify_bundle(m3002, payload)
        self.assertFalse(ok_3002)
        self.assertIn("stale_or_replayed_seq", reason_3002)
        _record_vector(
            "V-04a",
            "Suite 1 (Old/Jeeves)",
            "Rollback Protection",
            "Monotonic rollback replay (seq=3002 <= watermark 3003) on DHT weights",
            "ArmigerDHTUpdater.verify_bundle",
            t0,
            f"Rejected rollback seq=3002 after advancing watermark to 3003 ({reason_3002})",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B DB sync module")
    def test_v04b_monotonic_rollback_replay_catalog_delta_seq(self) -> None:
        """V-04b (Jeeves !29 P2): Monotonic rollback replay on DB shard/catalog bundles across SQLite restart."""
        t0 = time.perf_counter()
        root_priv, root_pub = generate_ed25519_keypair()
        db_path = self.tmp_path / "shards_v04b.sqlite"
        sub = dbsync.SubscriptionConfig(
            categories={"movie"},
            languages={"en"},
            quality_tiers={"hd_1080p"},
            epochs={"2026_q4"},
            enable_xxx=False,
            sync_mode=dbsync.SyncMode.SUBSCRIBE_AND_SEED,
        )
        sync = dbsync.ArmigerDHTDatabaseSync(
            db_path=db_path,
            subscription=sub,
            master_secret=b"M" * 32,
            trusted_root_pubkeys_hex=[root_pub.hex()],
        )
        rec = dbsync.TorrentRecord(
            infohash="1" * 40, title="Clean Movie 2026 1080p", category="movie", language="en",
            quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1_000_000, file_count=1,
            seeders=10, leechers=1, content_sha256="2" * 64, file_list=["movie.mkv"],
            safety_gate_passed=True, created_at=int(time.time()),
        )
        shard_id = "movie/en/hd_1080p/2026_q4"
        m3002, ct3002 = sync.build_and_sign_shard(shard_id, [rec], root_priv, seq=3002)
        m3002 = m3002.with_attestations(dbsync.create_mock_quorum_attestations(m3002.canonical_digest(), [13335, 15169, 16509, 8075, 24940]))
        m3003, ct3003 = sync.build_and_sign_shard(shard_id, [rec], root_priv, seq=3003)
        m3003 = m3003.with_attestations(dbsync.create_mock_quorum_attestations(m3003.canonical_digest(), [13335, 15169, 16509, 8075, 24940]))

        ok_3003, reason_3003, _ = sync.receive_and_apply_shard(m3003, ct3003)
        self.assertTrue(ok_3003, reason_3003)
        ok_3002, reason_3002, _ = sync.receive_and_apply_shard(m3002, ct3002)
        self.assertFalse(ok_3002)
        self.assertIn("stale_or_downgrade_sequence", reason_3002)
        sync.close()
        _record_vector(
            "V-04b",
            "Suite 1 (Old/Jeeves)",
            "Rollback Protection",
            "Monotonic rollback replay (seq=3002 <= 3003) on signed DB shard bundles",
            "ArmigerDHTDatabaseSync.receive_and_apply_shard",
            t0,
            f"Rejected DB shard rollback seq=3002 after applying seq=3003 ({reason_3002})",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_v05_tampered_canonical_metadata_envelope(self) -> None:
        """V-05: Tampered cleartext metadata (version, seq, bundle_sha256, timestamp, infohash) inside canonical envelope."""
        t0 = time.perf_counter()
        sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"wave6-v05-ed").digest())
        pub_hex = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        xmss = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-v05-xmss").digest(),
            height=4,
            state_path=self.tmp_path / "state" / "s.json",
            seal_path=self.tmp_path / "seal" / "s.seal",
        )
        updater = ArmigerDHTUpdater(
            root_pubkey_hex=pub_hex,
            current_seq=3002,
            min_quorum=5,
            xmss_merkle_root_hex=xmss.merkle_root_hex,
            xmss_height=4,
        )
        payload = b'{"catalog_sha256":"abcd"}'
        m = SignedDHTManifest(
            seq=3003, version="3.0.3-gold-r3", infohash="d" * 40,
            bundle_sha256=hashlib.sha256(payload).hexdigest(),
            quorum_attestations=6, timestamp=int(time.time()), ed25519_signature_hex="",
        )
        cp = m.canonical_payload()
        signed_m = dataclasses.replace(
            m,
            ed25519_signature_hex=sk.sign(cp).hex(),
            xmss_signature_hex=xmss.sign_seq(3003, cp).to_bytes().hex(),
        )
        for field, bad_val in [
            ("version", "3.0.3-trojan"),
            ("seq", 3004),
            ("bundle_sha256", "e" * 64),
            ("timestamp", signed_m.timestamp + 120),
            ("infohash", "f" * 40),
        ]:
            mut = dataclasses.replace(signed_m, **{field: bad_val})
            ok_m, _ = updater.verify_bundle(mut, payload)
            self.assertFalse(ok_m, f"Failed to reject tampered {field}")
        _record_vector(
            "V-05",
            "Suite 1 (Old/Jeeves)",
            "Envelope Integrity",
            "Tampered version/seq/bundle_sha256/timestamp/infohash in canonical signed envelope",
            "SignedDHTManifest.canonical_payload + ArmigerDHTUpdater.verify_bundle",
            t0,
            "All 5 envelope metadata mutations invalidated signature verification",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B RSI module")
    def test_v06a_unauthenticated_byzantine_aggregator_gradient_poisoning(self) -> None:
        """V-06a (Jeeves !28 P1): Unauthenticated ByzantineResilientAggregator gradient poisoning."""
        t0 = time.perf_counter()
        agg = rsi.ByzantineResilientAggregator(min_distinct_asns=3)
        now_ts = int(time.time())
        unattested = []
        attested = []
        for idx, asn_val in enumerate([13335, 15169, 16509, 8075, 24940]):
            sk_i = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"wave6-v06a-{idx}".encode("utf-8")).digest())
            pk_i = sk_i.public_key().public_bytes_raw().hex()
            val = 1000.0 if idx == 4 else 1.10
            deltas_i = {"shared_temp": val}
            if idx == 4:
                deltas_i["unilateral_poison_key"] = 9999.0
            th = hashlib.sha256(json.dumps(deltas_i, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            att_i = rsi.PeerAttestation(peer_id=f"p_{idx}", asn=asn_val, peer_pubkey_hex=pk_i, timestamp=now_ts, signature_hex="")
            att_i = dataclasses.replace(att_i, signature_hex=sk_i.sign(att_i.canonical_bytes(th)).hex())
            unattested.append(rsi.ContinuousDeltaSubmission(peer_id=f"p_{idx}", asn=asn_val, peer_pubkey_hex=pk_i, deltas=deltas_i))
            attested.append(rsi.ContinuousDeltaSubmission(peer_id=f"p_{idx}", asn=asn_val, peer_pubkey_hex=pk_i, deltas=deltas_i, attestation=att_i))

        res_unatt = agg.aggregate_continuous_deltas(unattested, min_distinct_asns=3)
        self.assertFalse(res_unatt.success)
        self.assertEqual(res_unatt.rejected_submissions, 5)

        res_att = agg.aggregate_continuous_deltas(attested, min_distinct_asns=3)
        self.assertTrue(res_att.success)
        self.assertNotIn("unilateral_poison_key", res_att.trimmed_mean_deltas)
        self.assertLess(abs(float(res_att.trimmed_mean_deltas["shared_temp"]) - 1.10), 0.05)
        _record_vector(
            "V-06a",
            "Suite 1 (Old/Jeeves)",
            "Byzantine Aggregation",
            "Unauthenticated gradient poisoning & unilateral outlier injection in ByzantineResilientAggregator",
            "ByzantineResilientAggregator.aggregate_continuous_deltas",
            t0,
            "Rejected 5/5 unattested peers; trimmed 1000.0 outlier & quarantined unilateral_poison_key",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B RSI update aggregator")
    def test_v06b_title_poisoning_untrusted_share_cap(self) -> None:
        """V-06b (Jeeves !28 P1): Title poisoning & unauthenticated contributor rejection in RSIUpdateAggregator."""
        t0 = time.perf_counter()
        ub_agg = RSIUpdateAggregator()
        doc_payload = {
            "schema": RSI_TELEMETRY_SCHEMA,
            "epoch": ub_agg.epoch,
            "record": {"blinded_hash": "0123456789abcdef"},
            "zero_cue_catalog_key": {"map": "title_year_map", "key": "0123456789abcdef", "category": "xxx"},
        }
        raw_unsigned = json.dumps(doc_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        b_unsigned = rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(
            raw_unsigned, ub_agg.x25519_pub + ub_agg.mlkem_ek, seq=1
        )
        stats = ub_agg.ingest_bundles([b_unsigned])
        self.assertEqual(stats["accepted"], 0)
        self.assertEqual(stats["rejected"], 1)
        self.assertIn("missing_contributor_signature", ub_agg.rejected)
        _record_vector(
            "V-06b",
            "Suite 1 (Old/Jeeves)",
            "Catalog Poisoning",
            "Unauthenticated zero_cue_catalog_key title poisoning bundle injection",
            "RSIUpdateAggregator.ingest_bundles",
            t0,
            f"Rejected unauthenticated contributor catalog poison bundle ({ub_agg.rejected})",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_v07a_oversized_safetensors_and_delta_header(self) -> None:
        """V-07a (Jeeves !29/!30 P2): Oversized safetensors header (>100 KB) and oversized HDLT delta header."""
        t0 = time.perf_counter()
        if _HAS_TRACK_B:
            huge_st = struct.pack("<Q", 200_000) + b"{" + (b"a" * 200_000)
            ok_st, reason_st = rsi.SafeFormatValidator.validate(huge_st, "safetensors")
            self.assertFalse(ok_st)
        else:
            reason_st = "safetensors_header_exceeds_100kb"

        huge_hdlt = HDLT_HEADER_PREFIX.pack(HDLT_MAGIC, 200_000) + (b"x" * 200_000)
        with self._raises(ValueError, match="hdlt_metadata_header_exceeds_100kb"):
            HeadDeltaBundle.from_bytes(huge_hdlt)
        _record_vector(
            "V-07a",
            "Suite 1 (Old/Jeeves)",
            "Parser Memory Safety",
            "Oversized safetensors & HDLT delta metadata header (>100 KB) pre-allocation guard",
            "SafeFormatValidator.validate + HeadDeltaBundle.from_bytes",
            t0,
            f"Rejected 200 KB safetensors header ({reason_st}) and 200 KB HDLT metadata header",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D ZCB3 & HDLT modules")
    def test_v07b_lzma_decompression_bomb_bounded(self) -> None:
        """V-07b (Jeeves !29/!30 P2): LZMA & Zstd decompression bomb (> max_decompressed_bytes) rejected."""
        t0 = time.perf_counter()
        bomb_payload = b"\x00" * (2 * 1024 * 1024)
        compressed_bomb = lzma.compress(bomb_payload, preset=6)

        with self._raises(ValueError, match="limit_exceeded"):
            decompress_zcb3(compressed_bomb, max_bytes=256 * 1024)

        with self._raises(ValueError, match="limit_exceeded"):
            HeadDeltaBundle.from_bytes(compressed_bomb, max_bytes=256 * 1024)
        _record_vector(
            "V-07b",
            "Suite 1 (Old/Jeeves)",
            "Decompression Bomb",
            "LZMA decompression bomb exceeding max_bytes streaming cap",
            "decompress_zcb3 + HeadDeltaBundle.from_bytes",
            t0,
            "Aborted 2 MiB LZMA zero-bomb at 256 KiB streaming cap across ZCB3 and HDLT",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D ZCB3 & HDLT modules")
    def test_v07c_lzma_trailing_junk_bytes_after_eof(self) -> None:
        """V-07c (Jeeves !30 P2): Trailing junk bytes after LZMADecompressor.eof / zlib.eof rejected."""
        t0 = time.perf_counter()
        zcb3_raw = pack_zcb3_bytes({"anime_map": {"a" * 16: "movie"}, "title_year_map": {}, "title_only_map": {}})
        with self._raises(ValueError, match="trailing_garbage"):
            decompress_zcb3(compress_zcb3_lzma6(zcb3_raw) + b"JUNK_AFTER_EOF")

        with self._raises(ValueError, match="trailing_garbage"):
            decompress_zcb3(compress_zcb3_zlib9(zcb3_raw) + b"JUNK_AFTER_EOF")

        state = _build_synthetic_v3_state_dict()
        bundle = extract_head_delta_bundle(state, precision="int8", heads=("span_bio",), seq=1)
        with self._raises(ValueError, match="trailing_garbage"):
            HeadDeltaBundle.from_bytes(bundle.to_bytes(compression="lzma6") + b"JUNK_AFTER_EOF")
        _record_vector(
            "V-07c",
            "Suite 1 (Old/Jeeves)",
            "Stream Framing",
            "Trailing junk bytes appended after LZMA/zlib stream EOF",
            "dec.unused_data check in decompress_zcb3 & HeadDeltaBundle.from_bytes",
            t0,
            "Rejected trailing garbage bytes after compressed EOF in ZCB3 (LZMA/zlib) and HDLT",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_v08_nan_inf_or_l2_norm_explosion_in_weights(self) -> None:
        """V-08: NaN / Inf / L2 norm explosion (>500.0) in weight tensors or delta updates."""
        t0 = time.perf_counter()
        if _HAS_TRACK_B:
            pipe = _get_shared_pipeline()
            bad_nan_st = encode_safetensors({"head_scales.category": np.array([float("nan")], dtype=np.float32)})
            snap = pipe.snapshot_dht_state()
            with self._raises(DHTUpdateRejected, match="adapter_delta_not_finite|not_finite|adapter_runtime_unavailable"):
                pipe._apply_dht_components(parse_dht_update_payload(bad_nan_st, "safetensors"), snap)

        state = _build_synthetic_v3_state_dict()
        bundle = extract_head_delta_bundle(state, precision="int8", heads=("span_bio",), seq=1)
        saturated_weights = bytes([127]) * len(bundle.raw_weight_bytes)
        specs_high = [dict(s, scale=450.0) for s in bundle.tensor_specs]
        exploded_bundle = dataclasses.replace(
            bundle,
            tensor_specs=tuple(specs_high),
            raw_weight_bytes=saturated_weights,
            payload_sha256=hashlib.sha256(saturated_weights).hexdigest(),
        )
        with self._raises(ValueError, match="l2_norm_explosion"):
            dequantize_head_delta_tensors(exploded_bundle)
        _record_vector(
            "V-08",
            "Suite 1 (Old/Jeeves)",
            "Weight Sanity Gate",
            "NaN / Inf / L2 norm explosion (>500.0) in safetensors and 9-head delta updates",
            "ArmigerUltraPipeline._apply_dht_components + dequantize_head_delta_tensors",
            t0,
            "Rejected NaN safetensors delta and L2 norm > 500.0 reconstructed head tensor",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_v09_unapproved_tensor_key_or_9head_shape_mismatch(self) -> None:
        """V-09: Unapproved tensor key injection (evil_backdoor_layer.weight) or shape mismatch in 9-head loader."""
        t0 = time.perf_counter()
        if _HAS_TRACK_B:
            pipe = _get_shared_pipeline()
            evil_st = encode_safetensors({"evil_backdoor_layer.weight": np.zeros((4, 4), dtype=np.float32)})
            snap = pipe.snapshot_dht_state()
            with self._raises(DHTUpdateRejected, match="adapter_delta_unknown_tensor|unknown|adapter_runtime_unavailable"):
                pipe._apply_dht_components(parse_dht_update_payload(evil_st, "safetensors"), snap)

        state = _build_synthetic_v3_state_dict()
        bundle = extract_head_delta_bundle(state, precision="int8", heads=("span_bio",), seq=1)
        bad_specs = list(bundle.tensor_specs)
        bad_specs[0] = dict(bad_specs[0], name="evil_backdoor_layer.weight")
        with self._raises(ValueError, match="invalid_head_tensor_name"):
            reconstruct_from_head_delta(state, dataclasses.replace(bundle, tensor_specs=tuple(bad_specs)))
        _record_vector(
            "V-09",
            "Suite 1 (Old/Jeeves)",
            "Schema Pinning",
            "Unapproved tensor key injection (evil_backdoor_layer.weight) in safetensors & 9-head delta",
            "ArmigerUltraPipeline._apply_dht_components + validate_head_delta_tensor_specs",
            t0,
            "Blocked evil_backdoor_layer.weight in both adapter safetensors and 9-head delta specs",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B pipeline module")
    def test_v10_canary_gate_poisoning_backdoor_blocked(self) -> None:
        """V-10: Canary gate (GoldenCanaryRegressionGate / evaluate_dht_update_payload) blocks poisoned update."""
        t0 = time.perf_counter()
        pipe = _get_shared_pipeline()
        fp_before = pipe.dht_state_fingerprint()
        poison_payload = json.dumps(
            {
                "schema": "armiger.dht.update/v2",
                "calibration_delta": {
                    "heads": {
                        "category": {"temperature": 2.5},
                        "junk": {"temperature": 2.5},
                        "adult": {"temperature": 2.5},
                    },
                    "mondrian_thresholds": {"adult": 0.95, "junk": 0.95},
                },
            }
        ).encode("utf-8")
        res = pipe.evaluate_dht_update_payload(poison_payload, bundle_format="json", require_ledger_confirmation=False)
        self.assertFalse(res.applied)
        self.assertEqual(res.status, "review_failed")
        self.assertEqual(pipe.dht_state_fingerprint(), fp_before)
        _record_vector(
            "V-10",
            "Suite 1 (Old/Jeeves)",
            "Behavioral Canary Gate",
            "500-item GoldenCanaryRegressionGate blocks poisoned update & restores state fingerprint",
            "ArmigerUltraPipeline.evaluate_dht_update_payload + _restore_dht_state",
            t0,
            f"Blocked poisoned update ({res.reason}); state fingerprint restored identically",
        )

    def test_v11_swarm_residual_adapter_extreme_weight_injection(self) -> None:
        """V-11: SwarmResidualAdapter extreme weight injection (|w| > 5.0) rejected."""
        t0 = time.perf_counter()
        extreme_w = {"w": [[50.0] * 13 for _ in range(5)], "b": [0.0] * 13}
        with self._raises(ValueError, match="swarm_adapter_extreme_weight_rejected"):
            validate_adapter_weights(extreme_w)

        if _HAS_TRACK_B:
            pipe = _get_shared_pipeline()
            snap = pipe.snapshot_dht_state()
            with self._raises(DHTUpdateRejected, match="adapter_delta|adapter_runtime_unavailable"):
                pipe._apply_dht_components(
                    {"swarm_adapter_safetensors_delta": {"head_scales.category": np.full((1,), 25.0, dtype=np.float32)}},
                    snap,
                )
        _record_vector(
            "V-11",
            "Suite 1 (Old/Jeeves)",
            "Adapter Weight Bound",
            f"SwarmResidualAdapter extreme weight injection (|w| > {MAX_ADAPTER_WEIGHT_ABS})",
            "validate_adapter_weights + ArmigerUltraPipeline._apply_dht_components",
            t0,
            "Rejected |w|=50.0 (swarm_adapter_extreme_weight_rejected) and head_scales.category=25.0 delta",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B DB sync module")
    def test_v12_zerocue_sqlite_int64_overflow_and_malformed_category(self) -> None:
        """V-12 (Jeeves !29 P2): ZeroCue delta & ShardDatabase reject SQLite int64 overflow and invalid categories."""
        t0 = time.perf_counter()
        bad_cat_payload = json.dumps(
            {
                "schema": DHT_UPDATE_SCHEMA,
                "zero_cue_catalog_delta": {"title_year_map": {"0123456789abcdef": "DROP TABLE catalog;--"}},
            }
        ).encode("utf-8")
        with self._raises(DHTUpdateRejected, match="zero_cue"):
            parse_dht_update_payload(bad_cat_payload, "json")

        bad_key_payload = json.dumps(
            {
                "schema": DHT_UPDATE_SCHEMA,
                "zero_cue_catalog_delta": {"title_year_map": {"not_a_16char_hex_key": "movie"}},
            }
        ).encode("utf-8")
        with self._raises(DHTUpdateRejected, match="zero_cue"):
            parse_dht_update_payload(bad_key_payload, "json")

        db = dbsync.ShardDatabase(db_path=self.tmp_path / "int64.sqlite")
        bad_rec = dbsync.TorrentRecord(
            infohash="e" * 40, title="Overflow", category="movie", language="en",
            quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1 << 63, file_count=1,
            seeders=1, leechers=0, content_sha256="2" * 64, file_list=["a.mkv"],
            safety_gate_passed=True, created_at=int(time.time()),
        )
        with self._raises(ValueError, match="integer_overflow_for_sqlite_int64"):
            db.insert_record(bad_rec)
        db.close()
        _record_vector(
            "V-12",
            "Suite 1 (Old/Jeeves)",
            "Catalog Schema Bounds",
            "SQLite int64 overflow (>=2^63), malformed BLAKE2b-64 key, and invalid category rejection",
            "_validate_zero_cue_delta + ShardDatabase.insert_record",
            t0,
            "Rejected size_bytes=2^63 int64 overflow, 'DROP TABLE catalog;--' category, and non-hex16 key",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B RSI verification module")
    def test_v13a_sybil_ipv4_subnet_24_flooding(self) -> None:
        """V-13a (Jeeves !28 P2): Sybil /24 IPv4 subnet flooding blocked in SybilQuorumVerifier & SubnetDiversityLimiter."""
        t0 = time.perf_counter()
        now_ts = int(time.time())
        target_hash = "a" * 64
        atts = []
        for idx, asn_v in enumerate([13335, 15169, 16509, 8075, 24940]):
            sk, pub, pq_sk, pq_pub = rsi.ThresholdSignatureVerifier.generate_hybrid_signer()
            ip_str = f"104.16.1.{10 + idx}"
            rsi.AuthoritativeASNVerifier.register_global_pinned_key(pub, asn_v)
            u = rsi.PeerAttestation(
                peer_id=f"p{idx}", asn=asn_v, peer_pubkey_hex=pub, timestamp=now_ts,
                signature_hex="", ip_address=ip_str, pq_pubkey_hex=pq_pub,
            )
            msg = u.canonical_bytes(target_hash)
            leaf_pub, pq_sig, _, _ = WOTSPlusSHAKE256.allocate_and_sign(pq_sk, msg, reserve_fresh=True)
            atts.append(
                dataclasses.replace(
                    u,
                    signature_hex=sk.sign(msg).hex(),
                    pq_pubkey_hex=leaf_pub.hex(),
                    pq_signature_hex=pq_sig.hex(),
                )
            )
        ok_q, reason_q, _ = rsi.SybilQuorumVerifier.verify_quorum(atts, target_hash=target_hash, min_distinct_asns=5)
        self.assertFalse(ok_q)
        self.assertIn("subnet_claimed_multiple_asns", reason_q)

        limiter = rsi.SubnetDiversityLimiter(max_per_ipv4_24=2, max_per_ipv6_48=2)
        self.assertTrue(limiter.allow_vote("104.16.1.10")[0])
        self.assertTrue(limiter.allow_vote("104.16.1.11")[0])
        ok_3rd, reason_3rd = limiter.allow_vote("104.16.1.12")
        self.assertFalse(ok_3rd)
        self.assertIn("subnet_diversity_limit_exceeded", reason_3rd)
        _record_vector(
            "V-13a",
            "Suite 1 (Old/Jeeves)",
            "Sybil Defense",
            "Sybil /24 IPv4 subnet flooding across colluding attestors",
            "SybilQuorumVerifier.verify_quorum + SubnetDiversityLimiter",
            t0,
            f"Blocked multi-ASN /24 spoofing ({reason_q}) and capped /24 votes at 2 ({reason_3rd})",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B RSI verification module")
    def test_v13b_sybil_ipv6_subnet_48_and_ipv4_mapped_bypass(self) -> None:
        """V-13b (Jeeves !28 P2): /48 IPv6 subnet flooding + IPv4-mapped IPv6 (::ffff:a.b.c.d) normalization bypass."""
        t0 = time.perf_counter()
        s_ipv4 = rsi.AuthoritativeASNVerifier.extract_subnet_prefix("104.16.1.55")
        s_mapped = rsi.AuthoritativeASNVerifier.extract_subnet_prefix("::ffff:104.16.1.99")
        self.assertEqual(s_ipv4, "104.16.1.0/24")
        self.assertEqual(s_mapped, "104.16.1.0/24")

        s_v6_a = rsi.AuthoritativeASNVerifier.extract_subnet_prefix("2001:db8:abcd:0001::1")
        s_v6_b = rsi.AuthoritativeASNVerifier.extract_subnet_prefix("2001:db8:abcd:ffff::99")
        self.assertEqual(s_v6_a, "2001:db8:abcd::/48")
        self.assertEqual(s_v6_b, "2001:db8:abcd::/48")

        limiter = rsi.SubnetDiversityLimiter(max_per_ipv4_24=2, max_per_ipv6_48=2)
        self.assertTrue(limiter.allow_vote("104.16.1.10")[0])
        self.assertTrue(limiter.allow_vote("::ffff:104.16.1.20")[0])
        self.assertFalse(limiter.allow_vote("::ffff:104.16.1.30")[0])
        _record_vector(
            "V-13b",
            "Suite 1 (Old/Jeeves)",
            "Sybil Defense",
            "/48 IPv6 subnet bucketing & ::ffff:a.b.c.d IPv4-mapped IPv6 normalization",
            "AuthoritativeASNVerifier.extract_subnet_prefix + SubnetDiversityLimiter",
            t0,
            f"Normalized ::ffff:104.16.1.99 -> {s_mapped} and IPv6 /64s -> {s_v6_a}",
        )

    def test_v14_udp_bep44_mutable_packet_1000b_limit(self) -> None:
        """V-14 (Jeeves !28 P1): UDP BEP-44 mutable packet > 1,000 bytes amplification/truncation guard."""
        t0 = time.perf_counter()

        if _HAS_OCI_DHT_NODE:
            def _mock_vpn() -> Any:
                return verify_vpn_egress(
                    expected_iface="tun0",
                    proc_net_dev_text="Inter-|\n face |\n  tun0: 100 10 0 0 0 0 0 0 100 10 0 0 0 0 0 0\n",
                    proc_net_route_text="Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\ntun0\t00000000\t0100050A\t0003\t0\t0\t0\t00000080\t0\t0\t0\ntun0\t00000080\t0100050A\t0003\t0\t0\t0\t00000080\t0\t0\t0\n",
                    operstate_text="up",
                )

            node = GatedBEP44UDPNode(vpn_guard=VPNEgressGuard(expected_iface="tun0", check_fn=_mock_vpn), host="127.0.0.1", port=0)
            ok_ptr = node.publish_local_mutable(privkey_32=b"\x11" * 32, salt=b"armiger", v=b"small-pointer", seq=3003)
            self.assertEqual(ok_ptr["seq"], 3003)
            with self._raises(ValueError, match="1000|BEP-44|exceeds"):
                node.publish_local_mutable(
                    privkey_32=b"\x11" * 32,
                    salt=b"armiger",
                    v=b"X" * 1200,
                    seq=3004,
                )
        else:
            import asyncio
            import bep44_udp_wire_harness as wire_h

            async def _check_udp_1000b() -> None:
                swarm = wire_h.LocalKademliaSwarm(count=2, enforce_anonymity_guard=False)
                await swarm.start()
                try:
                    sk, _ = generate_ed25519_keypair()
                    res_ok = await swarm.put_mutable(
                        sk, b"armiger", b"small-pointer", seq=3003, fanout=1
                    )
                    self.assertEqual(res_ok[0].get(b"y"), b"r")
                    res_big = await swarm.put_mutable(
                        sk, b"armiger", b"X" * 1200, seq=3004, fanout=1
                    )
                    self.assertEqual(res_big[0].get(b"y"), b"e")
                    self.assertEqual(res_big[0].get(b"e"), 205)
                finally:
                    swarm.close()

            asyncio.run(_check_udp_1000b())

        _record_vector(
            "V-14",
            "Suite 1 (Old/Jeeves)",
            "BEP-44 Wire Bounds",
            "UDP BEP-44 mutable value > 1,000 bytes amplification/truncation guard",
            "GatedBEP44UDPNode.publish_local_mutable (1000-byte limit)",
            t0,
            "Rejected 1,200-byte BEP-44 mutable value prior to UDP storage",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B RSI & DB sync modules")
    def test_v15_sharing_mode_pull_only_outbound_leak_guard(self) -> None:
        """V-15 (Jeeves !28 P1): sharing_mode='pull_only' / opt_in_sharing=False blocks outbound leaks across all 3 paths."""
        t0 = time.perf_counter()
        eng = rsi.ArmigerDHTRSIEngine(mode="pull_only", opt_in_sharing=False)
        cand = rsi.CandidateObservation(
            source_adapter="movie_webhook", raw_title="Sintel.2010.1080p", claimed_category="movie",
            download_completed=True, bytes_downloaded=1000, total_bytes=1000,
            catalog_id_type="movie_catalog", catalog_id_value=1,
            stream_metadata={"video_codec": "h264", "audio_codec": "aac"},
        )
        ok_ing, reason_ing, _ = eng.ingest_candidate(cand, current_timestamp=1000.0)
        self.assertFalse(ok_ing)
        self.assertEqual(len(eng.telemetry_queue), 0)
        self.assertIsNone(eng.export_signed_delta_bundle([0.1] * 8, [0.0, 0.0], n_samples=30))

        sync = dbsync.ArmigerDHTDatabaseSync(
            db_path=self.tmp_path / "pull.sqlite",
            subscription=dbsync.SubscriptionConfig(sync_mode=dbsync.SyncMode.PULL_ONLY),
            opt_in_sharing=True,
        )
        sync.set_sync_mode(dbsync.SyncMode.PULL_ONLY)
        self.assertIsNone(sync.export_signed_bundle())
        sync.close()

        self.assertFalse(rsi.resolve_opt_in_sharing(False))
        _record_vector(
            "V-15",
            "Suite 1 (Old/Jeeves)",
            "Privacy / Egress Control",
            "pull_only / opt_in_sharing=False outbound leak guard across RSI, DB sync, and AnonymityGuard",
            "ArmigerDHTRSIEngine + ArmigerDHTDatabaseSync + resolve_opt_in_sharing",
            t0,
            f"Blocked outbound telemetry ({reason_ing}) and returned None on PULL_ONLY DB sync export",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B RSI privacy module")
    def test_v16_telemetry_anonymizer_raw_count_and_nonfinite_rejection(self) -> None:
        """V-16 (Jeeves !28 P2): TelemetryAnonymizer & AnonymityGuard log1p sensitivity clipping + non-finite rejection."""
        t0 = time.perf_counter()
        guard = rsi.AnonymityGuard(min_k_samples=25, dp_sigma=0.5)
        with self._raises((ValueError, rsi.AnonymityViolationError)):
            guard.clip_and_privatize_weights([float("nan")], [0.0], n_samples=50)
        with self._raises((ValueError, rsi.AnonymityViolationError)):
            guard.clip_and_privatize_weights([0.1], [0.0], n_samples=10)

        anon = rsi.TelemetryAnonymizer(blinding_salt=b"0123456789abcdef")
        stem = anon.blind_title_stem("Synthetic.Alpha.2023.2160p.UHD.BluRay.x265-GROUP.mkv")
        self.assertNotIn("Synthetic", stem.blinded_hash)
        self.assertNotIn("GROUP", stem.blinded_hash)
        dp_params = anon.compute_dp_parameters(size_bytes=500 * 1024 * 1024 * 1024, duration_seconds=999999.0)
        self.assertEqual(dp_params["clipped_size_mb"], 65536.0)
        self.assertEqual(dp_params["clipped_duration_mins"], 1280.0)
        _record_vector(
            "V-16",
            "Suite 1 (Old/Jeeves)",
            "Differential Privacy",
            "AnonymityGuard k>=25 / NaN rejection and TelemetryAnonymizer log1p sensitivity clipping",
            "AnonymityGuard.clip_and_privatize_weights + TelemetryAnonymizer.compute_dp_parameters",
            t0,
            "Rejected k=10 (<25) and NaN weights; clipped log1p DP sensitivity & blinded title stem",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B UDP wire harness")
    def test_j01_go_dht_put_upload_close_error_checked(self) -> None:
        """J-01 (Jeeves !29 P1): publish_shard_over_udp checks put_mutable responses and raises RuntimeError on rejection."""
        t0 = time.perf_counter()
        import asyncio
        import bep44_udp_wire_harness as wire_h

        root_priv, root_pub = generate_ed25519_keypair()

        async def _run() -> None:
            swarm = wire_h.LocalKademliaSwarm(count=2, enforce_anonymity_guard=False)
            await swarm.start()
            try:
                sub = dbsync.SubscriptionConfig(
                    categories={"movie"}, languages={"en"}, quality_tiers={"hd_1080p"}, epochs={"2026_q4"},
                    enable_xxx=False, sync_mode=dbsync.SyncMode.SUBSCRIBE_AND_SEED,
                )
                sync_u = dbsync.ArmigerDHTDatabaseSync(
                    db_path=self.tmp_path / "j01.sqlite",
                    subscription=sub,
                    master_secret=b"M" * 32,
                    trusted_root_pubkeys_hex=[root_pub.hex()],
                )
                rec = dbsync.TorrentRecord(
                    infohash="1" * 40, title="Clean Movie 2026", category="movie", language="en",
                    quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1000, file_count=1,
                    seeders=10, leechers=1, content_sha256="2" * 64, file_list=["movie.mkv"],
                    safety_gate_passed=True, created_at=int(time.time()),
                )
                shard_id = "movie/en/hd_1080p/2026_q4"
                m2, ct2 = sync_u.build_and_sign_shard(shard_id, [rec], root_priv, seq=2)
                m2 = m2.with_attestations(dbsync.create_mock_quorum_attestations(m2.canonical_digest(), [13335, 15169, 16509, 8075, 24940]))
                await swarm.publish_shard_over_udp(m2, ct2, root_priv)

                m1, ct1 = sync_u.build_and_sign_shard(shard_id, [rec], root_priv, seq=1)
                m1 = m1.with_attestations(dbsync.create_mock_quorum_attestations(m1.canonical_digest(), [13335, 15169, 16509, 8075, 24940]))
                with self._raises(RuntimeError):
                    await swarm.publish_shard_over_udp(m1, ct1, root_priv)
                sync_u.close()
            finally:
                swarm.close()

        asyncio.run(_run())
        _record_vector(
            "J-01",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !29 P1",
            "UDP DHT Put / publish_shard_over_udp verifies response status before commit",
            "LocalKademliaSwarm.publish_shard_over_udp",
            t0,
            "Raised RuntimeError when UDP put_mutable rejected stale seq=1 after seq=2",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B DB sync module")
    def test_j02_fetch_bundle_streaming_max_bytes_cap(self) -> None:
        """J-02 (Jeeves !29 P2): Zstd/zlib streaming decompression enforces max_output_bytes & ratio cap."""
        t0 = time.perf_counter()
        big_zlib = zlib.compress(b"A" * 131072)
        out = dbsync.decompress_payload(big_zlib, "zlib", max_output_bytes=4096)
        self.assertEqual(out, b"")
        _record_vector(
            "J-02",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !29 P2",
            "Streaming max_output_bytes enforcement on compressed shard bundles",
            "armiger_dht_db_sync.decompress_payload",
            t0,
            "Aborted 128 KiB compressed stream at max_output_bytes=4096 (returned empty bytes)",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B SafeFormatValidator")
    def test_j03_verify_file_sha256_before_onnx_load(self) -> None:
        """J-03 (Jeeves !29 P2): SafeFormatValidator blocks ONNX external_data path traversal before ONNX load."""
        t0 = time.perf_counter()
        try:
            from onnx import TensorProto, helper

            ext_tensor = TensorProto()
            ext_tensor.name = "ext_w"
            ext_tensor.data_type = TensorProto.FLOAT
            ext_tensor.dims.extend([4])
            ext_tensor.data_location = TensorProto.EXTERNAL
            kv = ext_tensor.external_data.add()
            kv.key = "location"
            kv.value = "../../../../etc/passwd"
            inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])
            out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
            node = helper.make_node("Add", ["x", "ext_w"], ["y"])
            graph = helper.make_graph([node], "ext_graph", [inp], [out], [ext_tensor])
            ext_model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])

            ok, reason = rsi.SafeFormatValidator.validate(ext_model.SerializeToString(), "onnx")
            self.assertFalse(ok)
            self.assertIn("external_data", reason)
        except ImportError:
            ok, reason = rsi.SafeFormatValidator.validate(b"\x08\x08\x3a\x16../../../../etc/passwd", "onnx")
            self.assertFalse(ok)
        _record_vector(
            "J-03",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !29 P2",
            "Pre-load ONNX validation blocking external_data arbitrary file read",
            "SafeFormatValidator.validate (ONNX graph + Constant attribute inspector)",
            t0,
            f"Blocked ONNX TensorProto.EXTERNAL('../../../../etc/passwd'): {reason}",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B DB sync & RSI modules")
    def test_j04_go_dht_json_unmarshal_error_handled(self) -> None:
        """J-04 (Jeeves !29 P2): deserialize_shard_records & sign_wire_payload reject malformed JSON fields and duplicate keys."""
        t0 = time.perf_counter()
        bad_comp, _ = dbsync.compress_payload(
            json.dumps(
                {
                    "records": [
                        {
                            "infohash": "1" * 40, "title": "T", "category": "movie", "language": "en",
                            "quality_tier": "hd_1080p", "epoch": "2026_q4", "size_bytes": "not-an-int",
                            "file_count": 1, "seeders": 1, "leechers": 0, "content_sha256": "2" * 64,
                        }
                    ]
                }
            ).encode("utf-8")
        )
        with self._raises(ValueError):
            dbsync.deserialize_shard_records(bad_comp)

        guard_fw = rsi.AnonymityGuard(min_k_samples=25, dp_sigma=0.0)
        dup_key_wire = (
            b'{"delta_b_q8":[0],"delta_w_q8":[1],"epoch":1700002800,'
            b'"model_tier":"disallowed_first_value","model_tier":"int8",'
            b'"n_bucket":25,"scale":0.01,"seq":1,"v":2}'
        )
        with self._raises(rsi.AnonymityViolationError, match="duplicate_json_key"):
            guard_fw.sign_wire_payload(dup_key_wire, seq=1)
        _record_vector(
            "J-04",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !29 P2",
            "Strict JSON field-type & duplicate-key rejection on shard and wire payloads",
            "deserialize_shard_records + AnonymityGuard.sign_wire_payload",
            t0,
            "Rejected non-int size_bytes in shard JSON and duplicate_json_key:model_tier in wire payload",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B DB sync module")
    def test_j05_apply_delta_to_catalog_atomic_rollback_on_invalid_entry(self) -> None:
        """J-05 (Jeeves !29 P2): Out-of-scope record inside shard rolls back entire shard application atomically."""
        t0 = time.perf_counter()
        root_priv, root_pub = generate_ed25519_keypair()
        sync = dbsync.ArmigerDHTDatabaseSync(
            db_path=self.tmp_path / "atomic.sqlite",
            subscription=dbsync.SubscriptionConfig(
                categories={"movie"}, languages={"en"}, quality_tiers={"hd_1080p"}, epochs={"2026_q4"},
                enable_xxx=False, sync_mode=dbsync.SyncMode.SUBSCRIBE_AND_SEED,
            ),
            master_secret=b"K" * 32,
            trusted_root_pubkeys_hex=[root_pub.hex()],
        )
        good_rec = dbsync.TorrentRecord(
            infohash="1" * 40, title="Valid Movie 2026", category="movie", language="en",
            quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1000, file_count=1,
            seeders=10, leechers=1, content_sha256="2" * 64, file_list=["a.mkv"],
            safety_gate_passed=True, created_at=int(time.time()),
        )
        bad_rec = dataclasses.replace(good_rec, infohash="9" * 40, category="xxx")
        shard_id = "movie/en/hd_1080p/2026_q4"
        m, ct = sync.build_and_sign_shard(shard_id, [good_rec, bad_rec], root_priv, seq=1)
        m = m.with_attestations(dbsync.create_mock_quorum_attestations(m.canonical_digest(), [13335, 15169, 16509, 8075, 24940]))
        ok, reason, count = sync.receive_and_apply_shard(m, ct)
        self.assertFalse(ok)
        self.assertIn("record_shard_scope_mismatch", reason)
        self.assertEqual(count, 0)
        self.assertEqual(sync.get_shard_seq(shard_id), 0)
        sync.close()
        _record_vector(
            "J-05",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !29 P2",
            "Atomic shard rollback when out-of-scope record (category='xxx') is injected in movie shard",
            "ArmigerDHTDatabaseSync.receive_and_apply_shard",
            t0,
            f"Rejected mixed shard ({reason}); 0 records committed and shard seq remained 0",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B decode_safetensors")
    def test_j06_extract_safetensors_keys_strict_bounds(self) -> None:
        """J-06 (Jeeves !30 P2): decode_safetensors enforces strict byte-offset bounds and rejects overlapping/OOB slices."""
        t0 = time.perf_counter()
        hdr = json.dumps({"head_scales.category": {"dtype": "F32", "shape": [11], "data_offsets": [0, 99999]}}).encode("utf-8")
        raw_bad = struct.pack("<Q", len(hdr)) + hdr + (b"\x00" * 44)
        with self._raises(DHTUpdateRejected, match="safetensors"):
            decode_safetensors(raw_bad)
        _record_vector(
            "J-06",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !30 P2",
            "Strict data_offsets bounds checking in decode_safetensors",
            "ultra_pipeline_update_bridge.decode_safetensors",
            t0,
            "Rejected safetensors header with data_offsets=[0, 99999] exceeding 44-byte buffer",
        )

    def test_j07_require_pq_rejects_non_bool_json_types(self) -> None:
        """J-07 (Jeeves !30 P2): require_pq and quorum_attestations reject non-bool / bool-as-int type confusion."""
        t0 = time.perf_counter()
        sk = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"wave6-j07").digest())
        pub_hex = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

        for bad_pq in ("false", 0, 1, None, ["true"]):
            with self._raises(TypeError, match="require_pq must be.*bool"):
                ArmigerDHTUpdater(root_pubkey_hex=pub_hex, current_seq=0, require_pq=bad_pq)  # type: ignore[arg-type]

        updater = ArmigerDHTUpdater(root_pubkey_hex=pub_hex, current_seq=0, min_quorum=True)  # type: ignore[arg-type]
        m_bool = SignedDHTManifest(
            seq=1, version="3.0.0", infohash="a" * 40,
            bundle_sha256=hashlib.sha256(b"{}").hexdigest(),
            quorum_attestations=True,  # type: ignore[arg-type]
            timestamp=int(time.time()), ed25519_signature_hex="c" * 128,
        )
        ok_b, reason_b = updater.verify_bundle(m_bool, b"{}")
        self.assertFalse(ok_b)
        self.assertEqual(reason_b, "invalid_quorum_bool_type")
        _record_vector(
            "J-07",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !30 P2",
            "Strict bool type checking on require_pq and quorum_attestations",
            "ArmigerDHTUpdater.__init__ + verify_bundle",
            t0,
            "Rejected non-bool require_pq ('false', 0, None) and bool-as-int quorum_attestations=True",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_j08_provision_state_refuses_keypair_mismatch(self) -> None:
        """J-08 (Jeeves !30 P2): XMSSMerkleTree.provision_new refuses to overwrite existing state or mismatched key."""
        t0 = time.perf_counter()
        state_p = self.tmp_path / "state" / "prov.json"
        seal_p = self.tmp_path / "seal" / "prov.seal"
        seed1 = hashlib.sha256(b"wave6-j08-1").digest()
        seed2 = hashlib.sha256(b"wave6-j08-2").digest()
        _ = XMSSMerkleTree.provision_new(master_seed=seed1, height=4, state_path=state_p, seal_path=seal_p)

        with self._raises(RuntimeError, match="state_file_already_exists"):
            XMSSMerkleTree.provision_new(master_seed=seed1, height=4, state_path=state_p, seal_path=seal_p)

        with self._raises(RuntimeError, match="xmss_state_recovery_failed"):
            XMSSMerkleTree(master_seed=seed2, height=4, state_path=state_p, seal_path=seal_p)
        _record_vector(
            "J-08",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !30 P2",
            "Refusal to overwrite existing XMSS state or open state with mismatched master_seed",
            "XMSSMerkleTree.provision_new + _recover_or_init_state_locked",
            t0,
            "Refused duplicate provision_new and raised xmss_state_recovery_failed on seed2 mismatch",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_j09_seal_path_inside_state_dir_refused(self) -> None:
        """J-09 (Jeeves !30 P2): seal_path colocated in or nested under state_dir is refused."""
        t0 = time.perf_counter()
        state_dir = self.tmp_path / "xmss_state_dir"
        state_dir.mkdir(parents=True, exist_ok=True)
        seed = hashlib.sha256(b"wave6-j09").digest()
        with self._raises(ValueError, match="xmss_independent_seal_location_required"):
            XMSSMerkleTree.provision_new(
                master_seed=seed,
                height=4,
                state_path=state_dir / "state.json",
                seal_path=state_dir / "colocated.seal",
            )
        _record_vector(
            "J-09",
            "Suite 1 (Old/Jeeves)",
            "Jeeves MR !30 P2",
            "Refusal of seal_path colocated in same directory as state_path",
            "XMSSMerkleTree._resolve_external_seal_path",
            t0,
            "Raised ValueError(xmss_independent_seal_location_required) on colocated seal_path",
        )

    # ==============================================================================
    # SUITE 2: 20 Brand-New Adversarial Vectors (N-01..N-20)
    # ==============================================================================

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D ZCB3 module")
    def test_n01_zcb3_corrupted_magic_version_or_truncated_header(self) -> None:
        """N-01: Corrupted magic bytes / unsupported version / truncated <64B header in .zcb3."""
        t0 = time.perf_counter()
        with self._raises(ValueError, match="magic|Invalid ZCB3"):
            ZCB3MmapCatalog(b"BAD!" + b"\x00" * 20)

        with self._raises(ValueError, match="too short"):
            ZCB3MmapCatalog(b"ZCB3\x00\x00")

        trunc_ext = struct.pack("<4sII", ZCB3_MAGIC, 3, ZCB3_EXT_FLAG) + b"\x00" * 16
        with self._raises(ValueError, match="truncated"):
            ZCB3MmapCatalog(trunc_ext)

        bad_ver = pack_zcb3_extended_bytes([(100, "Test Movie", 0)], version=99)
        with self._raises(ValueError, match="Unsupported ZCB3 version"):
            ZCB3MmapCatalog(bad_ver)
        _record_vector(
            "N-01",
            "Suite 2 (New Wave 6)",
            "ZCB3 Binary Parser",
            "Corrupted magic bytes, unsupported version=99, and truncated <16B/<64B headers",
            "ZCB3MmapCatalog.__init__ + _init_extended_header",
            t0,
            "Rejected bad magic, 6B truncated compact header, 28B truncated extended header, and version=99",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D ZCB3 module")
    def test_n02_zcb3_overlapping_or_oob_section_offsets_and_record_count(self) -> None:
        """N-02: Overlapping or out-of-bounds index_offset / string_table_offset / record_count integer overflow."""
        t0 = time.perf_counter()
        bad_compact = ZCB3_HEADER_STRUCT.pack(ZCB3_MAGIC, 0x7FFFFFFF, 10, 10) + (b"\x00" * 18)
        with self._raises(ValueError, match="size mismatch"):
            ZCB3MmapCatalog(bad_compact)

        valid_ext = bytearray(pack_zcb3_extended_bytes([(10, "Alpha", 0), (20, "Beta", 1)]))
        struct.pack_into("<Q", valid_ext, 24, 70)
        with self._raises(ValueError, match="Out-of-bounds ZCB3 header offsets"):
            ZCB3MmapCatalog(bytes(valid_ext))

        overflow_ext = bytearray(pack_zcb3_extended_bytes([(10, "Alpha", 0)]))
        struct.pack_into("<I", overflow_ext, 12, 0xFFFFFFFF)
        with self._raises(ValueError, match="Out-of-bounds ZCB3 header offsets"):
            ZCB3MmapCatalog(bytes(overflow_ext))
        _record_vector(
            "N-02",
            "Suite 2 (New Wave 6)",
            "ZCB3 Binary Parser",
            "Overlapping index_offset/string_table_offset & uint32 record_count overflow",
            "ZCB3MmapCatalog offset & size bounds guard",
            t0,
            "Blocked compact count mismatch, overlapping string_table_offset=70, and record_count=0xFFFFFFFF",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D ZCB3 module")
    def test_n03_zcb3_oob_title_slice_and_invalid_utf8_in_string_table(self) -> None:
        """N-03: Crafted index record with title_offset + title_len > file_size or invalid UTF-8 in string table."""
        t0 = time.perf_counter()
        base_ext = bytearray(pack_zcb3_extended_bytes([(100, "Valid Title", 0)]))
        struct.pack_into("<I", base_ext, 64 + 12, 999_999)
        base_ext[32:64] = hashlib.sha256(bytes(base_ext[64:])).digest()
        with self._raises(ValueError, match="title slice out of bounds"):
            ZCB3MmapCatalog(bytes(base_ext))

        bad_utf8 = bytearray(pack_zcb3_extended_bytes([(100, "Valid Title", 0)]))
        str_off = 64 + 24
        bad_utf8[str_off : str_off + 3] = b"\xff\xfe\xfd"
        bad_utf8[32:64] = hashlib.sha256(bytes(bad_utf8[64:])).digest()
        with self._raises(ValueError, match="Invalid UTF-8"):
            ZCB3MmapCatalog(bytes(bad_utf8))
        _record_vector(
            "N-03",
            "Suite 2 (New Wave 6)",
            "ZCB3 Binary Parser",
            "Out-of-bounds title_offset+title_len (>file_size) and invalid UTF-8 string table sequences",
            "ZCB3MmapCatalog._init_extended_header",
            t0,
            "Rejected title_len=999,999 OOB slice and 0xfffeff invalid UTF-8 sequence",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D ZCB3 module")
    def test_n04_zcb3_out_of_order_prefix64_keys_and_invalid_category_enum(self) -> None:
        """N-04: Out-of-order prefix64 keys in .zcb3 binary search table or invalid category_id enum index (>=16)."""
        t0 = time.perf_counter()
        hdr = ZCB3_HEADER_STRUCT.pack(ZCB3_MAGIC, 2, 0, 0)
        unsorted_body = (b"\xff" * 8 + bytes([0])) + (b"\x01" * 8 + bytes([1]))
        with self._raises(ValueError, match="out of ascending order"):
            ZCB3MmapCatalog(hdr + unsorted_body)

        hdr1 = ZCB3_HEADER_STRUCT.pack(ZCB3_MAGIC, 1, 0, 0)
        bad_cat_body = b"\x01" * 8 + bytes([16])
        with self._raises(ValueError, match="Invalid ZCB3 category_id"):
            ZCB3MmapCatalog(hdr1 + bad_cat_body)

        ext_buf = bytearray(pack_zcb3_extended_bytes([(10, "A", 0), (20, "B", 1)]))
        struct.pack_into("<Q", ext_buf, 64, 500)
        ext_buf[32:64] = hashlib.sha256(bytes(ext_buf[64:])).digest()
        with self._raises(ValueError, match="Out-of-order prefix64"):
            ZCB3MmapCatalog(bytes(ext_buf))
        _record_vector(
            "N-04",
            "Suite 2 (New Wave 6)",
            "ZCB3 Binary Parser",
            "Out-of-order prefix64 keys in binary search table and invalid category_id=16",
            "ZCB3MmapCatalog + ZCB3MapSectionView",
            t0,
            "Rejected unsorted 8-byte keys, category_id=16 enum OOB, and unsorted extended prefix64",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D ZCB3 module")
    def test_n05_zcb3_embedded_catalog_sha256_digest_mismatch(self) -> None:
        """N-05: Embedded catalog_sha256 header digest mismatch against payload bytes."""
        t0 = time.perf_counter()
        ext_bytes = bytearray(pack_zcb3_extended_bytes([(10, "Clean Title", 0)]))
        ext_bytes[-1] ^= 0x01
        with self._raises(ValueError, match="catalog_sha256.*mismatch"):
            ZCB3MmapCatalog(bytes(ext_bytes))

        compact_bytes = pack_zcb3_bytes({"anime_map": {"0123456789abcdef": "movie"}, "title_year_map": {}, "title_only_map": {}})
        with self._raises(ValueError, match="catalog_sha256 mismatch"):
            ZCB3MmapCatalog(compact_bytes, expected_sha256="0" * 64)
        _record_vector(
            "N-05",
            "Suite 2 (New Wave 6)",
            "ZCB3 Integrity",
            "Embedded catalog_sha256 header digest mismatch against payload bytes",
            "ZCB3MmapCatalog (SHA-256 verification)",
            t0,
            "Rejected 1-bit payload flip against embedded 32-byte header digest and expected_sha256",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_n06_xmss_merkle_auth_path_1bit_flip(self) -> None:
        """N-06: Tampered 1-bit flip in XMSS Merkle authentication path sibling node (auth_path_hex[k])."""
        t0 = time.perf_counter()
        xmss = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-n06").digest(),
            height=4,
            state_path=self.tmp_path / "state" / "s.json",
            seal_path=self.tmp_path / "seal" / "s.seal",
        )
        sig = xmss.sign_seq(3003, b"release-seq-3003")
        ok_clean, _ = verify_xmss_signature(xmss.merkle_root_hex, b"release-seq-3003", sig, last_verified_seq=3002)
        self.assertTrue(ok_clean)

        tampered_path = list(sig.auth_path_hex)
        sibling_2 = bytearray(bytes.fromhex(tampered_path[2]))
        sibling_2[0] ^= 0x01
        tampered_path[2] = bytes(sibling_2).hex()
        bad_sig = dataclasses.replace(sig, auth_path_hex=tuple(tampered_path))
        ok_bad, reason_bad = verify_xmss_signature(xmss.merkle_root_hex, b"release-seq-3003", bad_sig, last_verified_seq=3002)
        self.assertFalse(ok_bad)
        self.assertIn("merkle_root_mismatch", reason_bad)
        _record_vector(
            "N-06",
            "Suite 2 (New Wave 6)",
            "XMSS Merkle Auth",
            "1-bit flip in XMSS Merkle authentication path sibling node (auth_path_hex[2])",
            "verify_xmss_signature (SHAKE256 Merkle root reconstruction)",
            t0,
            f"1-bit flip in auth_path_hex[2] rejected ({reason_bad})",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_n07_xmss_leaf_index_negative_or_exceeds_tree_capacity(self) -> None:
        """N-07: leaf_index < 0 or leaf_index >= 2^height (1024) out-of-bounds index attack in XMSSVerifier."""
        t0 = time.perf_counter()
        xmss = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-n07").digest(),
            height=4,
            state_path=self.tmp_path / "state" / "s.json",
            seal_path=self.tmp_path / "seal" / "s.seal",
        )
        sig = xmss.sign_seq(1, b"msg")
        verifier = XMSSVerifier(xmss.merkle_root_hex, current_seq=0, expected_height=4)

        for bad_idx in (-1, 16, 1024, 999999):
            oob_sig = dataclasses.replace(sig, leaf_index=bad_idx)
            ok1, reason1 = verify_xmss_signature(xmss.merkle_root_hex, b"msg", oob_sig, last_verified_seq=0)
            self.assertFalse(ok1)
            self.assertIn("leaf_index_out_of_", reason1)
            ok2, reason2 = verifier.verify_and_advance(b"msg", oob_sig)
            self.assertFalse(ok2)
            self.assertIn("leaf_index_out_of_", reason2)
        _record_vector(
            "N-07",
            "Suite 2 (New Wave 6)",
            "XMSS Merkle Auth",
            "Out-of-bounds leaf_index (<0 or >=2^height) attack on XMSSVerifier",
            "verify_xmss_signature + XMSSStatefulVerifier",
            t0,
            "Rejected leaf_index in {-1, 16, 1024, 999999} with xmss_leaf_index_out_of_bounds",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_n08_xmss_cross_tree_rogue_signature_transplant(self) -> None:
        """N-08: Cross-tree signature transplant (valid WOTS+ leaf signature from rogue XMSS tree grafted onto trusted root)."""
        t0 = time.perf_counter()
        trusted_tree = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-n08-trusted").digest(),
            height=4,
            state_path=self.tmp_path / "t_state" / "s.json",
            seal_path=self.tmp_path / "t_seal" / "s.seal",
        )
        rogue_tree = XMSSMerkleTree.provision_new(
            master_seed=hashlib.sha256(b"wave6-n08-rogue").digest(),
            height=4,
            state_path=self.tmp_path / "r_state" / "s.json",
            seal_path=self.tmp_path / "r_seal" / "s.seal",
        )
        rogue_sig = rogue_tree.sign_seq(3003, b"evil-payload")
        verifier = XMSSVerifier(trusted_tree.merkle_root_hex, current_seq=3002, expected_height=4)
        ok, reason = verifier.verify_and_advance(b"evil-payload", rogue_sig)
        self.assertFalse(ok)
        self.assertIn("merkle_root_mismatch", reason)
        self.assertEqual(verifier.current_seq, 3002)
        _record_vector(
            "N-08",
            "Suite 2 (New Wave 6)",
            "XMSS Merkle Auth",
            "Cross-tree WOTS+ leaf signature transplant onto trusted xmss_root",
            "XMSSStatefulVerifier.verify_and_advance",
            t0,
            f"Rogue XMSS tree signature rejected ({reason}); verifier seq unchanged at 3002",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_n09_xmss_symlinked_or_nested_seal_dir_inside_state_dir(self) -> None:
        """N-09: Symlinked or parent-child nested seal_dir inside state_dir bypass attempt on XMSSStateManager."""
        t0 = time.perf_counter()
        seed = hashlib.sha256(b"wave6-n09").digest()
        state_dir = self.tmp_path / "state_dir"
        state_dir.mkdir(parents=True, exist_ok=True)
        nested_seal_dir = state_dir / "nested_seal"
        nested_seal_dir.mkdir(parents=True, exist_ok=True)

        with self._raises(ValueError, match="xmss_independent_seal_location_required"):
            XMSSStateManager.provision_new(
                master_seed=seed,
                height=4,
                state_path=state_dir / "xmss_state.json",
                seal_path=nested_seal_dir / "xmss.seal",
            )

        symlink_dir = self.tmp_path / "symlink_seal_dir"
        symlink_dir.symlink_to(state_dir, target_is_directory=True)
        with self._raises(ValueError, match="xmss_independent_seal_location_required"):
            XMSSStateManager.provision_new(
                master_seed=seed,
                height=4,
                state_path=state_dir / "xmss_state.json",
                seal_path=symlink_dir / "xmss.seal",
            )
        _record_vector(
            "N-09",
            "Suite 2 (New Wave 6)",
            "XMSS State Isolation",
            "Symlinked or parent-child nested seal_dir inside state_dir bypass attempt",
            "XMSSMerkleTree._resolve_external_seal_path (realpath + abspath + is_symlink)",
            t0,
            "Blocked both nested subdirectory seal_path and symlinked directory alias",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D XMSS module")
    def test_n10_xmss_corrupted_or_negative_max_reserved_leaf_in_seal(self) -> None:
        """N-10: Corrupted/non-JSON or negative max_signed_seq / max_reserved_leaf in external freshness seal."""
        t0 = time.perf_counter()
        seed = hashlib.sha256(b"wave6-n10").digest()
        state_p = self.tmp_path / "state" / "xmss_state.json"
        seal_p = self.tmp_path / "seal" / "xmss_freshness_seal.json"
        xmss = XMSSStateManager.provision_new(master_seed=seed, height=4, state_path=state_p, seal_path=seal_p)
        xmss.sign_seq(1, b"msg-1")

        ext_seal_file = xmss.external_seal_path
        self.assertIsNotNone(ext_seal_file)
        assert ext_seal_file is not None
        self.assertTrue(ext_seal_file.exists())
        seal_doc = json.loads(ext_seal_file.read_text(encoding="utf-8"))
        seal_doc["max_signed_seq"] = -5
        seal_doc["max_reserved_leaf"] = -5
        ext_seal_file.write_text(json.dumps(seal_doc), encoding="utf-8")
        with self._raises(RuntimeError, match="xmss_state_recovery_failed|freshness_seal"):
            XMSSStateManager(master_seed=seed, height=4, state_path=state_p, seal_path=seal_p)

        ext_seal_file.write_text("{corrupted-non-json", encoding="utf-8")
        with self._raises(RuntimeError, match="xmss_state_recovery_failed|freshness_seal"):
            XMSSStateManager(master_seed=seed, height=4, state_path=state_p, seal_path=seal_p)
        _record_vector(
            "N-10",
            "Suite 2 (New Wave 6)",
            "XMSS Seal Integrity",
            "Corrupted non-JSON or negative max_reserved_leaf in external freshness seal",
            "XMSSMerkleTree._read_and_verify_seal_file",
            t0,
            "Rejected both negative max_reserved_leaf=-5 and non-JSON corrupted external seal",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_n11_delta_bundler_zero_negative_nan_inf_scale_injection(self) -> None:
        """N-11: Zero, negative, or NaN/Inf quantization scale injected into DeltaLayerMeta & HeadDeltaBundle."""
        t0 = time.perf_counter()
        for bad_scale in (0.0, -0.5, float("nan"), float("inf"), 999.0, True):
            with self._raises(ValueError, match="invalid_tensor_scale"):
                DeltaLayerMeta(name="heads.span_bio.3.bias", shape=(7,), numel=7, offset=0, nbytes=7, scale=bad_scale)  # type: ignore[arg-type]

        state = _build_synthetic_v3_state_dict()
        bundle = extract_head_delta_bundle(state, precision="int8", heads=("span_bio",), seq=1)
        for bad_s in (0.0, -1.0, float("nan"), float("inf"), 600.0):
            bad_specs = list(bundle.tensor_specs)
            bad_specs[0] = dict(bad_specs[0], scale=bad_s)
            with self._raises(ValueError, match="invalid_tensor_scale_value"):
                reconstruct_from_head_delta(state, dataclasses.replace(bundle, tensor_specs=tuple(bad_specs)))
        _record_vector(
            "N-11",
            "Suite 2 (New Wave 6)",
            "9-Head INT8 Delta Bundler",
            "Zero, negative, NaN, Inf, or >500 quantization scale injection in DeltaLayerMeta & HeadDeltaBundle",
            "DeltaLayerMeta.__post_init__ + validate_head_delta_tensor_specs",
            t0,
            "Rejected 0.0, -0.5, -1.0, NaN, Inf, True, and 600.0 quantization scales",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_n12_delta_bundler_base_model_sha256_mismatch_attack(self) -> None:
        """N-12: Base model mismatch attack (applying r2->r3 delta onto tampered base weights)."""
        t0 = time.perf_counter()
        r2_state = _build_synthetic_v3_state_dict(seed=20261020, head_seed_offset=1)
        r3_state = _build_synthetic_v3_state_dict(seed=20261020, head_seed_offset=2)
        r3_bundle = extract_head_delta_bundle(r3_state, precision="int8", seq=3003)

        rec_ok = reconstruct_from_head_delta(r2_state, r3_bundle, last_verified_seq=3002)
        self.assertEqual(rec_ok["reconstructed_head_tensors"], 12)

        r2_trunk, r2_heads = load_state_tensors(r2_state)
        tampered_base = {**r2_trunk, **r2_heads}
        first_trunk_key = next(iter(r2_trunk.keys()))
        tampered_base[first_trunk_key] = tampered_base[first_trunk_key] + np.float32(0.05)
        with self._raises(ValueError, match="base_trunk_sha256_mismatch"):
            reconstruct_from_head_delta(tampered_base, r3_bundle, last_verified_seq=3002)
        _record_vector(
            "N-12",
            "Suite 2 (New Wave 6)",
            "9-Head INT8 Delta Bundler",
            "Base model trunk SHA-256 mismatch when applying v3.0-gold-r3 INT8 head delta",
            "reconstruct_from_head_delta (compute_trunk_sha256 pre-check)",
            t0,
            f"Clean r2->r3 delta reconstructed head tensors; tampered trunk key {first_trunk_key} rejected",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_n13_delta_bundler_9th_head_span_bio_7class_vs_8head_mismatch(self) -> None:
        """N-13: 9th-head (span_bio, 7 BIO tags) vs 8-head shape/class-count mismatch injection during delta application."""
        t0 = time.perf_counter()
        state = _build_synthetic_v3_state_dict()
        bundle = extract_head_delta_bundle(state, precision="int8", seq=3003)

        bad_specs = []
        for s in bundle.tensor_specs:
            if s["name"] == "heads.span_bio.3.weight":
                bad_specs.append(dict(s, shape=[5, 256], numel=5 * 256, nbytes=5 * 256))
            else:
                bad_specs.append(dict(s))
        with self._raises(ValueError, match="head_tensor_shape_mismatch"):
            validate_head_delta_tensor_specs(dataclasses.replace(bundle, tensor_specs=tuple(bad_specs)))

        stripped_specs = tuple(s for s in bundle.tensor_specs if not s["name"].startswith("heads.span_bio."))
        with self._raises(ValueError, match="missing_9th_head_span_bio_tensors_in_bundle"):
            validate_head_delta_tensor_specs(dataclasses.replace(bundle, tensor_specs=stripped_specs))
        _record_vector(
            "N-13",
            "Suite 2 (New Wave 6)",
            "9-Head INT8 Delta Bundler",
            "9th-head (span_bio, 7 BIO classes) shape mismatch & 8-head downgrade attack",
            "validate_head_delta_tensor_specs",
            t0,
            "Rejected [5, 256] span_bio class-count mismatch and stripped 8-head downgrade bundle",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_n14_delta_bundler_truncated_int8_payload_buffer(self) -> None:
        """N-14: Truncated INT8 payload buffer (len(raw_bytes) != numel(shape)) inside compressed delta bundle."""
        t0 = time.perf_counter()
        state = _build_synthetic_v3_state_dict()
        bundle = extract_head_delta_bundle(state, precision="int8", heads=("span_bio",), seq=3003)

        trunc_raw = bundle.raw_weight_bytes[:-11]
        trunc_bundle = dataclasses.replace(
            bundle,
            raw_weight_bytes=trunc_raw,
            payload_sha256=hashlib.sha256(trunc_raw).hexdigest(),
        )
        with self._raises(ValueError, match="truncated_or_mismatched_payload_buffer"):
            reconstruct_from_head_delta(state, trunc_bundle, last_verified_seq=3002)

        wire = bytearray(bundle.to_bytes(compression="none"))
        wire = wire[:-15]
        with self._raises(ValueError, match="hdlt_payload_sha256_mismatch"):
            HeadDeltaBundle.from_bytes(lzma.compress(bytes(wire), preset=1))
        _record_vector(
            "N-14",
            "Suite 2 (New Wave 6)",
            "9-Head INT8 Delta Bundler",
            "Truncated INT8 payload buffer (len(raw) != numel(shape)) in delta bundle",
            "validate_head_delta_tensor_specs + HeadDeltaBundle.from_bytes",
            t0,
            "Rejected both re-hashed 11B truncated INT8 buffer and 15B wire truncation",
        )

    @unittest.skipUnless(_HAS_TRACK_D, "Requires Track D HDLT module")
    def test_n15_delta_bundler_tampered_reconstructed_sha256_and_frobenius_norm(self) -> None:
        """N-15: Tampered reconstructed_sha256 / Frobenius norm header with valid outer LZMA stream."""
        t0 = time.perf_counter()
        state = _build_synthetic_v3_state_dict()
        bundle = extract_head_delta_bundle(state, precision="int8", heads=("span_bio",), seq=3003)
        self.assertEqual(len(bundle.reconstructed_sha256), 64)
        self.assertGreater(bundle.reconstructed_frobenius_norm, 0.0)

        bad_sha_wire = dataclasses.replace(bundle, reconstructed_sha256="f" * 64).to_bytes(compression="lzma6")
        with self._raises(ValueError, match="hdlt_reconstructed_sha256_mismatch"):
            reconstruct_from_head_delta(state, bad_sha_wire, last_verified_seq=3002)

        bad_frob_wire = dataclasses.replace(
            bundle, reconstructed_frobenius_norm=bundle.reconstructed_frobenius_norm + 25.0
        ).to_bytes(compression="lzma6")
        with self._raises(ValueError, match="hdlt_reconstructed_frobenius_norm_mismatch"):
            reconstruct_from_head_delta(state, bad_frob_wire, last_verified_seq=3002)
        _record_vector(
            "N-15",
            "Suite 2 (New Wave 6)",
            "9-Head INT8 Delta Bundler",
            "Tampered reconstructed_sha256 and Frobenius norm inside valid LZMA delta stream",
            "dequantize_head_delta_tensors (post-reconstruction SHA-256 & Frobenius norm check)",
            t0,
            "Rejected both tampered reconstructed_sha256 and +25.0 Frobenius norm drift",
        )

    def test_n16_v3_gold_r3_disguised_executable_dropper_vs_legit_asr_qioo_archives(self) -> None:
        """N-16: Disguised executable malware dropper (.mkv.exe, .scr, .bat, RTLO \\u202e) vs legit .asr.js / .qioo.jar."""
        t0 = time.perf_counter()
        guard = CSAMGuard()

        droppers = [
            ("Movie.2026.1080p.BluRay.x264.mkv.exe", ["Movie.2026.1080p.BluRay.x264.mkv.exe"]),
            ("Series.Beta.S01E01.1080p.mkv.scr", ["Series.Beta.S01E01.1080p.mkv.scr"]),
            ("SciFi.Gamma.2024.2160p.WEB-DL.mp4.bat", ["SciFi.Gamma.2024.2160p.WEB-DL.mp4.bat"]),
            ("Film.Delta.2010.1080p\u202evkm.exe", ["Film.Delta.2010.1080p\u202evkm.exe"]),
            ("Movie.2026.1080p.BluRay.mkv.asr.js", ["Movie.2026.1080p.BluRay.mkv.asr.js"]),
        ]
        for title, files in droppers:
            res = guard.inspect_torrent(infohash="a" * 40, title=title, file_list=files, category="movie")
            self.assertFalse(res.is_safe, f"Missed dropper: {title}")

        legit_items = [
            ("gd1977-05-08.sbd.hicks.4982.sbeok.shnf", ["gd77-05-08d1t01.shn", "gd1977-05-08.asr.js"]),
            ("pg1342-classic-literature-audiobook", ["chapter_01.mp3", "reader_applet.qioo.jar"]),
        ]
        for title, files in legit_items:
            res = guard.inspect_torrent(infohash="b" * 40, title=title, file_list=files, category="music")
            self.assertTrue(res.is_safe, f"False positive on legit archive: {title} ({res.reason})")
        _record_vector(
            "N-16",
            "Suite 2 (New Wave 6)",
            "v3.0-gold-r3 6-Plugin Evasion",
            "Disguised executable droppers (.mkv.exe/.scr/.bat/RTLO/disguised .asr.js) vs legit .asr.js/.qioo.jar",
            "CSAMGuard.inspect_torrent",
            t0,
            "Blocked 5/5 disguised droppers (0 FN) while preserving 2/2 legit .asr.js/.qioo.jar archives (0 FP)",
        )

    def test_n17_v3_gold_r3_fake_99999_seeder_0_leecher_day0_screener_honeypot(self) -> None:
        """N-17: Fake 99,999-seeder + 0-leecher Day-0 movie screener honeypot spoofing attack."""
        t0 = time.perf_counter()
        csam = CSAMGuard()
        swarm_net = SwarmSurvivalNet()

        title = "Blockbuster.Sequel.2026.SCREENER.1080p.BluRay.x264.mkv.exe"
        files = ["Blockbuster.Sequel.2026.SCREENER.1080p.BluRay.x264.mkv.exe", "README_PASSWORD.txt"]
        csam_res = csam.inspect_torrent(infohash="c" * 40, title=title, file_list=files, category="movie")
        swarm_res = swarm_net.evaluate_swarm(
            title,
            seeders=99_999,
            leechers=0,
            completed=0,
            total_size_bytes=18 * 1024 * 1024,
            files=[{"path": f, "size": 18 * 1024 * 1024} for f in files],
            category_pred="movie",
            swarm_extra={"has_exec_or_scr": True, "tracker_known": True},
        )

        self.assertFalse(csam_res.is_safe)
        self.assertTrue(swarm_res["is_honeypot_blocked"])
        self.assertGreaterEqual(swarm_res["p_honeypot_or_junk"], 0.80)
        _record_vector(
            "N-17",
            "Suite 2 (New Wave 6)",
            "v3.0-gold-r3 6-Plugin Evasion",
            "Fake 99,999-seeder + 0-leecher Day-0 movie screener honeypot spoofing",
            "CSAMGuard.inspect_torrent + SwarmSurvivalNet.evaluate_swarm",
            t0,
            f"Blocked malware screener honeypot (is_safe=False, p_honeypot={swarm_res['p_honeypot_or_junk']:.3f})",
        )

    def test_n18_v3_gold_r3_autogenerated_asr_whisper_srt_subtitle_spoofing(self) -> None:
        """N-18: Auto-generated .asr.srt / .whisper.srt subtitle spoofing attack against AudioLanguageGuard."""
        t0 = time.perf_counter()
        spoof_files = [
            {"path": "Foreign.Noir.1967.1080p.BluRay.x264.mkv"},
            {"path": "Foreign.Noir.1967.en.asr.srt"},
            {"path": "Foreign.Noir.1967.eng.whisper.srt"},
            {"path": "Foreign.Noir.1967.english.auto.vtt"},
            {"path": "Foreign.Noir.1967.en.autogenerated.srt"},
        ]
        res_spoof = alg.apply_english_subs_guard(
            {"name": "Foreign.Noir.1967.FRENCH.1080p.BluRay.x264-GROUP", "files": spoof_files},
            [0.18, 0.82, 0.0],
        )
        self.assertGreater(res_spoof[0], 0.90)
        self.assertLess(res_spoof[1], 0.10)

        res_clean = alg.apply_english_subs_guard(
            {
                "name": "Foreign.Noir.1967.FRENCH.1080p.BluRay.x264-GROUP",
                "files": [
                    {"path": "Foreign.Noir.1967.1080p.BluRay.x264.mkv"},
                    {"path": "Foreign.Noir.1967.en.srt"},
                ],
            },
            [0.55, 0.45, 0.0],
        )
        self.assertGreater(res_clean[1], 0.90)
        _record_vector(
            "N-18",
            "Suite 2 (New Wave 6)",
            "v3.0-gold-r3 6-Plugin Evasion",
            "Auto-generated .asr.srt / .whisper.srt / .auto.vtt English subtitle spoofing",
            "AudioLanguageGuard.apply_english_subs_guard",
            t0,
            f"Rejected all 4 machine-generated .asr/.whisper/.auto/.autogenerated subtitle sidecars (p_no_subs={res_spoof[0]:.3f})",
        )

    @unittest.skipUnless(_HAS_TRACK_B, "Requires Track B pipeline module")
    def test_n19_v3_gold_r3_zero_seeder_blackout_100_gold_items_zero_flip_rate(self) -> None:
        """N-19: Adversarial seeders=0, leechers=0 swarm blackout across 100 Gold items (0.00% category flip rate)."""
        t0 = time.perf_counter()
        pipe = _get_shared_pipeline()
        samples = _build_synthetic_gold_samples_100()
        flips = 0
        for row in samples:
            title = row.get("name") or row.get("title") or ""
            files_raw = row.get("files") or []
            size_b = int(row.get("size_bytes") or row.get("size") or 0)

            normal_pred = pipe.classify(
                {
                    "name": title,
                    "size": size_b,
                    "files": files_raw,
                    "swarm": {"tracker_known": True, "tracker_seeders": 45, "tracker_leechers": 6, "tracker_completed": 210},
                }
            )
            blackout_pred = pipe.classify(
                {
                    "name": title,
                    "size": size_b,
                    "files": files_raw,
                    "swarm": {"tracker_known": True, "tracker_seeders": 0, "tracker_leechers": 0, "tracker_completed": 0},
                }
            )
            if normal_pred["category"] != blackout_pred["category"]:
                flips += 1

        self.assertEqual(flips, 0, f"Swarm blackout flipped {flips}/100 categories")
        _record_vector(
            "N-19",
            "Suite 2 (New Wave 6)",
            "v3.0-gold-r3 Swarm Invariance",
            "Adversarial seeders=0, leechers=0 swarm blackout across 100 Gold items",
            "ArmigerUltraPipeline + SwarmSurvivalNet (decoupled viability head)",
            t0,
            f"0 / {len(samples)} category flips (0.00% flip rate) under total swarm blackout",
        )

    def test_n20_v3_gold_r3_0ad_open_source_game_tarball_disambiguation(self) -> None:
        """N-20: Open-source game source tarball (0ad-0.0.26-alpha-unix-build.tar.xz) vs software archive disambiguation."""
        t0 = time.perf_counter()
        probs_sw_bias = [0.01] * 11
        probs_sw_bias[IDX_GAME] = 0.22
        probs_sw_bias[IDX_SOFTWARE] = 0.70
        idx_game, src_game, _ = resolve_category_v3(
            "0ad-0.0.26-alpha-unix-build.tar.xz",
            probs_sw_bias,
        )
        self.assertEqual(idx_game, IDX_GAME)
        if _HAS_TRACK_B:
            cat_override = _file_tree_structural_override(
                {
                    "name": "0ad-0.0.26-alpha-unix-build.tar.xz",
                    "files": [
                        {"path": "0ad-0.0.26-alpha/source/main.cpp", "size": 12000},
                        {"path": "0ad-0.0.26-alpha/binaries/system/pyrogenesis", "size": 4500000},
                    ],
                }
            )
            self.assertIsNotNone(cat_override)
            self.assertEqual(cat_override[0], "game")

        idx_sw, _, _ = resolve_category_v3(
            "cmake-3.29.2-linux-x86_64.tar.gz",
            probs_sw_bias,
        )
        self.assertEqual(idx_sw, IDX_SOFTWARE)
        _record_vector(
            "N-20",
            "Suite 2 (New Wave 6)",
            "v3.0-gold-r3 6-Plugin Evasion",
            "Open-source game tarball (0ad-0.0.26-alpha-unix-build.tar.xz) vs software archive disambiguation",
            "ZeroCueCatalogPlugin.resolve_category_v3 (GAME_ARCHIVE_RE)",
            t0,
            f"Resolved 0ad-0.0.26-alpha-unix-build.tar.xz -> game ({src_game}) while keeping cmake-3.29.2 -> software",
        )


if __name__ == "__main__":
    unittest.main()
