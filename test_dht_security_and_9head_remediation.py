"""Regression unit tests for all 16 DHT security remediations and 9-head v3.0 loader support."""
from __future__ import annotations

import hashlib
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1] if len(_HERE.parents) >= 2 and (_HERE.parents[1] / "armiger").is_dir() else _HERE
for sub in (REPO_ROOT, REPO_ROOT / "armiger", REPO_ROOT / "deploy" / "oci-dht"):
    if sub.is_dir():
        s_sub = str(sub)
        if s_sub in sys.path:
            sys.path.remove(s_sub)
        sys.path.insert(0, s_sub)

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import armiger_dht_rsi as rsi
import armiger_ultra_pipeline as pipe
import calibration_and_dht as cdht
import csam_guard as cg
from audio_language_guard import classify_audio_and_anime_dub_v3
from armiger_inference import ArmigerStandaloneONNX
from zero_cue_plugin import hash_catalog_key


class TestDHTSecurityAnd9HeadRemediation(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self._wots_tmp = tempfile.TemporaryDirectory()
        base_p = Path(self._wots_tmp.name)
        cg.WOTSPlusSHAKE256.set_state_dir(base_p / "wots_state", seal_dir=base_p / "wots_seals")
        cg.WOTSPlusSHAKE256._IN_MEM_NEXT_LEAF.clear()
        cg.WOTSPlusSHAKE256._IN_MEM_USED_LEAVES.clear()
        cg.WOTSPlusSHAKE256._IN_MEM_MSG_TO_LEAF.clear()
        cg.WOTSPlusSHAKE256._IN_MEM_GENERATION.clear()
        cg.WOTSPlusSHAKE256._IN_MEM_NONCE.clear()
        cg.WOTSPlusSHAKE256._IN_MEM_BOUND_DIR.clear()

    def tearDown(self) -> None:
        cg.WOTSPlusSHAKE256.set_state_dir(None, seal_dir=None)
        self._wots_tmp.cleanup()

    def test_v01b_v01c_v01d_pq_mandatory_and_wots_binding(self) -> None:
        seed = bytes([0x31]) * 32
        priv = Ed25519PrivateKey.from_private_bytes(seed)
        pub_hex = priv.public_key().public_bytes_raw().hex()
        rsi.WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(seed)

        # V01C: ArmigerDHTUpdater rejects classical-only manifest when pq_signature_hex is stripped
        updater = cdht.ArmigerDHTUpdater(root_pubkey_hex=pub_hex, current_seq=0, min_quorum=5)
        payload_bytes = b"{\"test\":1}"
        sha_hex = hashlib.sha256(payload_bytes).hexdigest()
        now_ts = int(time.time())
        m_stripped = cdht.SignedDHTManifest(
            seq=1,
            version="3.0.0",
            infohash="a" * 40,
            bundle_sha256=sha_hex,
            quorum_attestations=5,
            timestamp=now_ts,
            ed25519_signature_hex="",
            pq_pubkey_hex="",
            pq_signature_hex="",
        )
        sig_ed = priv.sign(m_stripped.canonical_payload()).hex()
        m_stripped = cdht.SignedDHTManifest(
            seq=1,
            version="3.0.0",
            infohash="a" * 40,
            bundle_sha256=sha_hex,
            quorum_attestations=5,
            timestamp=now_ts,
            ed25519_signature_hex=sig_ed,
            pq_pubkey_hex="",
            pq_signature_hex="",
        )
        ok, _ = updater.verify_bundle(m_stripped, payload_bytes)
        self.assertFalse(ok)

        # V01D: TombstoneStore rejects rogue self-supplied WOTS+ key not bound to root Ed25519 key
        rogue_wots_sk, rogue_wots_pk = cg.WOTSPlusSHAKE256.generate_keypair(bytes([0x99]) * 64)
        store = cg.TombstoneStore(trusted_root_pubkeys_hex=[pub_hex])
        tm = cg.SignedTombstoneList(
            seq=1,
            timestamp=now_ts,
            tombstones=["b" * 64],
            reasons={"b" * 64: "csam"},
            signature_hex="",
            pq_pubkey_hex=rogue_wots_pk.hex(),
            pq_signature_hex="",
        )
        tm_payload = tm.canonical_payload()
        tm = cg.SignedTombstoneList(
            seq=1,
            timestamp=now_ts,
            tombstones=["b" * 64],
            reasons={"b" * 64: "csam"},
            signature_hex=priv.sign(tm_payload).hex(),
            pq_pubkey_hex=rogue_wots_pk.hex(),
            pq_signature_hex=cg.WOTSPlusSHAKE256.sign(rogue_wots_sk, tm_payload).hex(),
        )
        ok_tm, *_ = store.apply_signed_tombstones(tm)
        self.assertFalse(ok_tm)

    def test_v02_bool_type_confusion_and_seq_window(self) -> None:
        seed = bytes([0x32]) * 32
        priv = Ed25519PrivateKey.from_private_bytes(seed)
        pub_hex = priv.public_key().public_bytes_raw().hex()
        updater_bool = cdht.ArmigerDHTUpdater(root_pubkey_hex=pub_hex, current_seq=0, min_quorum=True)  # type: ignore[arg-type]
        m_bool = cdht.SignedDHTManifest(
            seq=1,
            version="3.0.0",
            infohash="a" * 40,
            bundle_sha256=hashlib.sha256(b"{}").hexdigest(),
            quorum_attestations=True,  # type: ignore[arg-type]
            timestamp=int(time.time()),
            ed25519_signature_hex="c" * 128,
        )
        ok_b, reason_b = updater_bool.verify_bundle(m_bool, b"{}")
        self.assertFalse(ok_b)
        self.assertEqual(reason_b, "invalid_quorum_bool_type")

        # V02D: Sequence jump > 1024 rejected
        updater = cdht.ArmigerDHTUpdater(root_pubkey_hex=pub_hex, current_seq=0, min_quorum=5)
        m_huge = cdht.SignedDHTManifest(
            seq=2**64,
            version="3.0.0",
            infohash="a" * 40,
            bundle_sha256=hashlib.sha256(b"{}").hexdigest(),
            quorum_attestations=5,
            timestamp=int(time.time()),
            ed25519_signature_hex="c" * 128,
        )
        ok, reason = updater.verify_bundle(m_huge, b"{}")
        self.assertFalse(ok)
        self.assertIn("seq_jump_exceeds_max_window", reason)

    def test_v04b_v04c_s1a_onnx_structural_validation_and_external_data_rejection(self) -> None:
        # S1-A: ONNX model containing raw 0x80 0x02 tensor weight bytes is accepted when structurally valid
        try:
            import onnx
            from onnx import TensorProto, helper
        except ImportError:
            ok_ext_raw, _ = rsi.SafeFormatValidator.validate(b"\x08\x08\x3a\x10../../../../etc/hostname", "onnx")
            self.assertFalse(ok_ext_raw)
            return

        weight_bytes = b"\x80\x02\x80\x03\x80\x04\x80\x05" + (b"\x01" * 24)
        tensor = helper.make_tensor("w", TensorProto.UINT8, [32], list(weight_bytes))
        inp = helper.make_tensor_value_info("x", TensorProto.UINT8, [32])
        out = helper.make_tensor_value_info("y", TensorProto.UINT8, [32])
        node = helper.make_node("Add", ["x", "w"], ["y"])
        graph = helper.make_graph([node], "test_graph", [inp], [out], [tensor])
        model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
        raw_onnx = model.SerializeToString()

        ok_valid, reason_valid = rsi.SafeFormatValidator.validate(raw_onnx, "onnx")
        self.assertTrue(ok_valid, f"Expected valid ONNX with 0x80 bytes in weights to pass, got {reason_valid}")

        # V04B: ONNX with external_data path traversal is strictly rejected
        ext_tensor = TensorProto()
        ext_tensor.name = "ext_w"
        ext_tensor.data_type = TensorProto.FLOAT
        ext_tensor.dims.extend([4])
        ext_tensor.data_location = TensorProto.EXTERNAL
        kv = ext_tensor.external_data.add()
        kv.key = "location"
        kv.value = "../../../../etc/hostname"
        ext_graph = helper.make_graph([node], "ext_graph", [inp], [out], [ext_tensor])
        ext_model = helper.make_model(ext_graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
        ok_ext, reason_ext = rsi.SafeFormatValidator.validate(ext_model.SerializeToString(), "onnx")
        self.assertFalse(ok_ext)
        self.assertIn("external_data", reason_ext)

    def test_v05b_unallocated_asn_without_ip_rejected(self) -> None:
        atts = []
        now_ts = int(time.time())
        target_hash = "d" * 64
        for idx in range(5):
            sk = Ed25519PrivateKey.from_private_bytes(bytes([0x50 + idx]) * 32)
            pk_hex = sk.public_key().public_bytes_raw().hex()
            att = rsi.PeerAttestation(
                peer_id=f"sybil_{idx}",
                asn=90001 + idx,
                peer_pubkey_hex=pk_hex,
                timestamp=now_ts,
                signature_hex="",
                ip_address=None,
            )
            att = rsi.PeerAttestation(
                peer_id=att.peer_id,
                asn=att.asn,
                peer_pubkey_hex=att.peer_pubkey_hex,
                timestamp=att.timestamp,
                signature_hex=sk.sign(att.canonical_bytes(target_hash)).hex(),
                ip_address=None,
            )
            atts.append(att)
        ok_q, _, _ = rsi.SybilQuorumVerifier.verify_quorum(atts, target_hash=target_hash, min_distinct_asns=5)
        self.assertFalse(ok_q)

    def test_v06b_calibrator_bounds_and_nan_rejection(self) -> None:
        cal_clamped = cdht.ArmigerCalibrator({"heads": {"junk": {"temperature": 100.0, "logit_bias": [10.0, -10.0]}}})
        probs = cal_clamped.calibrate_head("junk", [-2.0, 2.6])
        self.assertTrue(cal_clamped.evaluate_safety_gates(0.01, probs[1], 0.01)["junk_flagged"])
        nan_probs = cal_clamped.calibrate_head("junk", [float("nan"), 1.0])
        self.assertTrue(all(np.isfinite(nan_probs)))
        with self.assertRaises(ValueError):
            cdht.ArmigerCalibrator({"heads": {"junk": {"temperature": float("nan")}}})

    def test_v07b_keyed_catalog_hash_and_salted_blinding(self) -> None:
        h_default = hash_catalog_key("Oppenheimer|2023")
        h_keyed = hash_catalog_key("Oppenheimer|2023", hmac_key=b"secret-domain-key")
        self.assertNotEqual(h_default, h_keyed)
        self.assertEqual(len(h_keyed), 16)

        anonymizer = rsi.TelemetryAnonymizer(blinding_salt=b"shared_salt_12345678")
        s_rainbow = anonymizer.blind_title_stem("Oppenheimer.2023.2160p.WEB-DL.DDP5.1.Atmos.DV.HDR.H.265-FLUX.mkv")
        s_mystery = anonymizer.blind_title_stem("Oppenheimer.2023.1080p.BluRay.x264-SPARKS.mkv")
        self.assertNotEqual(s_rainbow.blinded_hash, s_mystery.blinded_hash)

    def test_v08a_s1b_9head_span_bio_and_audio_guard_dual_audio_multi_subs(self) -> None:
        mock_sess = MagicMock()
        head_names = [
            "category", "adult", "junk", "anime", "lang",
            "owner_scope", "title_language", "english_subs", "span_bio",
        ]
        outs = []
        for h in head_names:
            m_out = MagicMock()
            m_out.name = h
            outs.append(m_out)
        mock_sess.get_outputs.return_value = outs
        mock_sess.run.return_value = [
            np.zeros((1, 11), dtype=np.float32),
            np.array([[2.0, -2.0]], dtype=np.float32),
            np.array([[2.0, -2.0]], dtype=np.float32),
            np.array([[-2.0, 2.0]], dtype=np.float32),
            np.array([[2.0, -1.0, -1.0]], dtype=np.float32),
            np.array([[2.0, -2.0]], dtype=np.float32),
            np.array([[2.0, -2.0]], dtype=np.float32),
            np.array([[-1.0, 2.0]], dtype=np.float32),
            np.zeros((1, 256, 7), dtype=np.float32),
        ]
        clf = ArmigerStandaloneONNX.__new__(ArmigerStandaloneONNX)
        clf.session = mock_sess
        clf.head_names = head_names
        clf.mondrian_thresholds = {"adult": .5451, "junk": .101, "owner_scope": .477}
        clf.mondrian_strata = {}
        res = clf.predict({"name": "[Judas] Kusuriya no Hitorigoto (Season 1) [1080p][HEVC x265 10bit][Dual-Audio][Multi-Subs]"})
        self.assertIn("span_bio", res)
        self.assertEqual(len(res["heads"]["span_bio"]), 256)
        from armiger_inference import NEURAL_HEAD_CLASSES, SPAN_BIO_CLASSES
        from byte_span_extractor import SPAN_BIO_CLASSES as BSE_SPAN_BIO_CLASSES
        self.assertEqual(tuple(NEURAL_HEAD_CLASSES["span_bio"]), SPAN_BIO_CLASSES)
        self.assertEqual(SPAN_BIO_CLASSES, BSE_SPAN_BIO_CLASSES)
        self.assertEqual(SPAN_BIO_CLASSES[5:7], ("B-EP", "I-EP"))

        aud = classify_audio_and_anime_dub_v3(
            "[Judas] Kusuriya no Hitorigoto (Season 1) [1080p][HEVC x265 10bit][Dual-Audio][Multi-Subs]",
            p_en=0.95,
            p_anime=0.98,
        )
        self.assertTrue(aud["is_anime_dub_or_dual"])
        pipe._validate_calibration_delta({"heads": {"span_bio": {"temperature": 1.0}}})

    def test_jeeves_mr30_remediations_comprehensive(self) -> None:
        import dataclasses
        import json
        import tempfile
        import armiger_dht_db_sync as dbsync
        from byte_span_extractor import extract_byte_spans_v3

        # 1. Zstd unknown-frame-size and bomb streaming cap
        if getattr(dbsync, "_HAS_ZSTD", False):
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(write_content_size=False)
            streamed = cctx.compress(b"x" * 4096)
            self.assertEqual(dbsync.decompress_payload(streamed, "zstd", max_output_bytes=8192), b"x" * 4096)
            self.assertEqual(dbsync.decompress_payload(streamed, "zstd", max_output_bytes=1024), b"")

        # 2. Durable SQLite shard_sequences across restart + record scope enforcement
        root_priv, root_pub = cg.generate_ed25519_keypair()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "shards.sqlite"
            sub = dbsync.SubscriptionConfig(
                categories={"movie"},
                languages={"en"},
                quality_tiers={"hd_1080p"},
                epochs={"2026_q4"},
                enable_xxx=False,
                sync_mode=dbsync.SyncMode.SUBSCRIBE_AND_SEED,
            )
            sync1 = dbsync.ArmigerDHTDatabaseSync(
                db_path=db_path,
                subscription=sub,
                master_secret=b"M" * 32,
                trusted_root_pubkeys_hex=[root_pub.hex()],
            )
            good_rec = dbsync.TorrentRecord(
                infohash="1" * 40,
                title="Clean Movie 2026 1080p",
                category="movie",
                language="en",
                quality_tier="hd_1080p",
                epoch="2026_q4",
                size_bytes=1_000_000,
                file_count=1,
                seeders=10,
                leechers=1,
                content_sha256="2" * 64,
                file_list=["movie.mkv"],
                safety_gate_passed=True,
                created_at=int(time.time()),
            )
            shard_id = "movie/en/hd_1080p/2026_q4"
            m1, ct1 = sync1.build_and_sign_shard(shard_id, [good_rec], root_priv, seq=5)
            atts1 = dbsync.create_mock_quorum_attestations(m1.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
            m1 = m1.with_attestations(atts1)
            ok1, reason1, cnt1 = sync1.receive_and_apply_shard(m1, ct1)
            self.assertTrue(ok1, reason1)
            self.assertEqual(cnt1, 1)
            sync1.close()

            # Re-open from SQLite and verify seq=5 is persisted and replay is rejected
            sync2 = dbsync.ArmigerDHTDatabaseSync(
                db_path=db_path,
                subscription=sub,
                master_secret=b"M" * 32,
                trusted_root_pubkeys_hex=[root_pub.hex()],
            )
            self.assertEqual(sync2.get_shard_seq(shard_id), 5)
            ok_rep, reason_rep, _ = sync2.receive_and_apply_shard(m1, ct1)
            self.assertFalse(ok_rep)
            self.assertIn("stale_or_downgrade_sequence", reason_rep)

            # Out-of-scope record (category='xxx') packed inside movie/en/hd_1080p/2026_q4 shard is rejected
            bad_rec = dataclasses.replace(good_rec, infohash="3" * 40, category="xxx")
            m_bad, ct_bad = sync2.build_and_sign_shard(shard_id, [bad_rec], root_priv, seq=6)
            m_bad = m_bad.with_attestations(
                dbsync.create_mock_quorum_attestations(m_bad.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
            )
            ok_bad, reason_bad, _ = sync2.receive_and_apply_shard(m_bad, ct_bad)
            self.assertFalse(ok_bad)
            self.assertIn("record_shard_scope_mismatch", reason_bad)
            self.assertEqual(sync2.get_shard_seq(shard_id), 5)

            # WOTS+ one-time key reuse across different seqs on QuorumVerifier is rejected
            m_reuse, ct_reuse = sync2.build_and_sign_shard(shard_id, [good_rec], root_priv, seq=6)
            reused_wots_priv, reused_wots_pub = rsi.WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(root_priv, seq=5)
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey as _E
            sk_root = _E.from_private_bytes(root_priv)
            m_reuse_forged = dataclasses.replace(
                m_reuse,
                root_pq_pubkey_hex=reused_wots_pub.hex(),
                ed25519_signature_hex="",
                pq_signature_hex="",
            )
            payload_forged = m_reuse_forged.canonical_payload()
            m_reuse_forged = dataclasses.replace(
                m_reuse_forged,
                ed25519_signature_hex=sk_root.sign(payload_forged).hex(),
                pq_signature_hex=rsi.WOTSPlusSHAKE256.sign(reused_wots_priv, payload_forged).hex(),
            )
            m_reuse_forged = m_reuse_forged.with_attestations(
                dbsync.create_mock_quorum_attestations(
                    m_reuse_forged.canonical_digest(), [13335, 15169, 16509, 8075, 24940]
                )
            )
            ok_reuse, reason_reuse, _ = sync2.receive_and_apply_shard(m_reuse_forged, ct_reuse)
            self.assertFalse(ok_reuse)
            self.assertTrue(
                "wots_one_time_pubkey_reused" in reason_reuse
                or "untrusted_root_pq_pubkey" in reason_reuse,
                reason_reuse,
            )
            sync2.close()

        # 3. TombstoneStore mandatory PQ, canonical_payload binding, and per-sequence WOTS+ non-reuse
        tstore = cg.TombstoneStore(trusted_root_pubkeys_hex=[root_pub.hex()])
        t1 = cg.sign_tombstone_list(seq=1, infohashes_or_blinded=["a" * 40], private_key_bytes=root_priv)
        ok_t1, msg_t1, _ = tstore.apply_signed_tombstones(t1)
        self.assertTrue(ok_t1, msg_t1)
        # Strip PQ signature -> rejected under default require_pq=True
        t2_signed = cg.sign_tombstone_list(seq=2, infohashes_or_blinded=["b" * 40], private_key_bytes=root_priv)
        t2_stripped = dataclasses.replace(t2_signed, pq_signature_hex="")
        ok_t2, msg_t2, _ = tstore.apply_signed_tombstones(t2_stripped)
        self.assertFalse(ok_t2)
        self.assertEqual(msg_t2, "missing_pq_wots_signature")

        # 4. get_pending_telemetry_batch retains queue when aggregator_x25519_pubkey_hex is unset
        eng = rsi.ArmigerDHTRSIEngine(
            mode="anonymous_contribute",
            opt_in_sharing=True,
            aggregator_x25519_pubkey_hex=None,
        )
        cand = rsi.CandidateObservation(
            source_adapter="movie_webhook",
            raw_title="Sintel.2010.1080p.BluRay.x264",
            claimed_category="movie",
            download_completed=True,
            bytes_downloaded=1_000_000_000,
            total_bytes=1_000_000_000,
            catalog_id_type="movie_catalog",
            catalog_id_value=12345,
            stream_metadata={"video_codec": "h264", "audio_codec": "aac"},
        )
        ok_ing, _, _ = eng.ingest_candidate(cand, current_timestamp=1000.0)
        self.assertTrue(ok_ing)
        self.assertEqual(len(eng.telemetry_queue), 1)
        self.assertEqual(eng.get_pending_telemetry_batch(now_timestamp=999999.0), [])
        self.assertEqual(len(eng.telemetry_queue), 1)

        # 5. verify_and_stage_update requires candidate_predictor_fn
        s_ed, s_pub, s_pq_priv, s_pq_pub = rsi.ThresholdSignatureVerifier.generate_hybrid_signer()
        eng_upd = rsi.ArmigerDHTRSIEngine(
            mode="pull_only",
            current_seq=0,
            trusted_signers={s_pub: "root"},
            trusted_pq_signers={s_pq_pub: "root_pq"},
            threshold_m=1,
            min_distinct_asns=1,
        )
        b_bytes = b'{"heads":{"category":{"temperature":1.1}}}'
        b_sha = hashlib.sha256(b_bytes).hexdigest()
        now_ts = int(time.time())
        rsi.AuthoritativeASNVerifier.register_global_pinned_key(s_pub, 13335)
        m_upd = rsi.ThresholdDHTManifest(
            seq=1,
            version="3.0.0",
            infohash="a" * 40,
            bundle_sha256=b_sha,
            bundle_format="json",
            min_asn_quorum=1,
            timestamp=now_ts,
            signatures={},
        )
        p_upd = m_upd.canonical_payload()
        m_upd = dataclasses.replace(
            m_upd,
            signatures={s_pub: s_ed.sign(p_upd).hex()},
            pq_signatures={s_pq_pub: rsi.WOTSPlusSHAKE256.sign(s_pq_priv, p_upd).hex()},
        )
        att_u = rsi.PeerAttestation(
            peer_id="p1",
            asn=13335,
            peer_pubkey_hex=s_pub,
            timestamp=now_ts,
            signature_hex=s_ed.sign(
                rsi.PeerAttestation("p1", 13335, s_pub, now_ts, "", ip_address="104.16.1.1").canonical_bytes(b_sha)
            ).hex(),
            ip_address="104.16.1.1",
        )
        ok_u, reason_u, _ = eng_upd.verify_and_stage_update(
            m_upd, b_bytes, [att_u], candidate_predictor_fn=None
        )
        self.assertFalse(ok_u)
        self.assertEqual(reason_u, "missing_candidate_predictor_fn")

        # 6. Recursive ONNX external_data check in Constant node attribute tensor
        try:
            from onnx import TensorProto, helper
            ext_t = TensorProto()
            ext_t.name = "const_ext"
            ext_t.data_type = TensorProto.FLOAT
            ext_t.dims.extend([2])
            ext_t.data_location = TensorProto.EXTERNAL
            kv = ext_t.external_data.add()
            kv.key = "location"
            kv.value = "../../etc/passwd"
            c_node = helper.make_node("Constant", [], ["y"], value=ext_t)
            g = helper.make_graph([c_node], "g", [], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
            m_onnx = helper.make_model(g, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
            ok_ox, reason_ox = rsi.SafeFormatValidator.validate(m_onnx.SerializeToString(), "onnx")
            self.assertFalse(ok_ox)
            self.assertIn("external_data", reason_ox)
        except ImportError:
            pass

        # 7. clip_and_privatize_weights small-norm scale > 0 and inspect_wire_payload acceptance
        guard = rsi.AnonymityGuard(min_k_samples=25, dp_sigma=0.0)
        clipped = guard.clip_and_privatize_weights([1e-12, -1e-12], [0.0], n_samples=50, dp_sigma=0.0)
        self.assertGreater(clipped["scale"], 0.0)
        wire = {
            "v": 2,
            "seq": 1,
            "epoch": 1700002800,
            "model_tier": "ultra",
            "delta_w_q8": clipped["delta_w_q8"],
            "delta_b_q8": clipped["delta_b_q8"],
            "scale": clipped["scale"],
            "n_bucket": clipped["n_bucket"],
            "reliability_bins": [],
        }
        self.assertIsNotNone(guard.inspect_wire_payload(wire))

        # 8. aggregate_continuous_deltas per-key ASN quorum, default require_attestation=True, and dimension mismatch quarantine
        agg = rsi.ByzantineResilientAggregator(min_distinct_asns=3)
        subs_cont = []
        subs_unattested = []
        for idx, asn_val in enumerate([13335, 15169, 16509]):
            sk_i = Ed25519PrivateKey.from_private_bytes(bytes([0x70 + idx]) * 32)
            pk_i = sk_i.public_key().public_bytes_raw().hex()
            deltas_i = {"shared_temp": 1.15, "vec": [0.1, 0.2]}
            if idx == 0:
                deltas_i["unilateral_poison_key"] = 999.0
            target_h_i = hashlib.sha256(
                json.dumps(deltas_i, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            att_i = rsi.PeerAttestation(
                peer_id=f"p_{idx}",
                asn=asn_val,
                peer_pubkey_hex=pk_i,
                timestamp=now_ts,
                signature_hex="",
            )
            att_i = dataclasses.replace(att_i, signature_hex=sk_i.sign(att_i.canonical_bytes(target_h_i)).hex())
            subs_unattested.append(
                rsi.ContinuousDeltaSubmission(
                    peer_id=f"p_{idx}",
                    asn=asn_val,
                    peer_pubkey_hex=pk_i,
                    deltas=deltas_i,
                )
            )
            subs_cont.append(
                rsi.ContinuousDeltaSubmission(
                    peer_id=f"p_{idx}",
                    asn=asn_val,
                    peer_pubkey_hex=pk_i,
                    deltas=deltas_i,
                    attestation=att_i,
                )
            )
        # Default require_attestation=True rejects unattested submissions
        res_unatt = agg.aggregate_continuous_deltas(subs_unattested, min_distinct_asns=3)
        self.assertFalse(res_unatt.success)
        self.assertEqual(res_unatt.rejected_submissions, 3)
        # Signed PeerAttestation submissions succeed
        res_cont = agg.aggregate_continuous_deltas(subs_cont, min_distinct_asns=3)
        self.assertTrue(res_cont.success)
        self.assertIn("shared_temp", res_cont.trimmed_mean_deltas)
        self.assertNotIn("unilateral_poison_key", res_cont.trimmed_mean_deltas)

        # 9. Multi-season span extraction S01-S03 and trailing tech/junk byte span shrink
        spans = extract_byte_spans_v3("Foundation.S01-S03.2160p.WEB-DL.DDP5.1.Atmos.H.265-FLUX.mkv")
        self.assertEqual(spans["title"], "Foundation")
        self.assertEqual(spans["episode_spec"], "S01-S03")
        spans_aac = extract_byte_spans_v3("Movie.AAC.mkv")
        self.assertEqual(spans_aac["title"], "Movie")
        self.assertEqual(spans_aac["title_byte_span"], (0, 5))
        spans_yify = extract_byte_spans_v3("The.Matrix.[YIFY].mkv")
        self.assertEqual(spans_yify["title"], "The Matrix")
        self.assertEqual(spans_yify["title_byte_span"], (0, 10))

        # 10. Discussion 1: verify_and_stage_update enforces require_pq=True by default even when trusted_pq_signers={}
        eng_no_pq = rsi.ArmigerDHTRSIEngine(
            mode="pull_only",
            current_seq=0,
            trusted_signers={s_pub: "root"},
            trusted_pq_signers={},
            threshold_m=1,
            min_distinct_asns=1,
        )
        m_classical_only = dataclasses.replace(m_upd, pq_signatures={})
        ok_npq, reason_npq, _ = eng_no_pq.verify_and_stage_update(
            m_classical_only,
            b_bytes,
            [att_u],
            candidate_predictor_fn=lambda it: {
                "junk": bool(it.ground_truth["junk"]),
                "adult": bool(it.ground_truth["adult"]),
                "category": str(it.ground_truth["category"]),
                "owner_scope": str(it.ground_truth["owner_scope"]),
            },
        )
        self.assertFalse(ok_npq)
        self.assertIn("insufficient_pq_threshold_signatures", reason_npq)
        raw_ed = Ed25519PrivateKey.from_private_bytes(bytes([0x66]) * 32)
        raw_ed_pub = raw_ed.public_key().public_bytes_raw().hex()
        eng_raw_no_pq = rsi.ArmigerDHTRSIEngine(
            mode="pull_only",
            current_seq=0,
            trusted_signers={raw_ed_pub: "raw_root"},
            trusted_pq_signers={},
            threshold_m=1,
            min_distinct_asns=1,
        )
        m_raw_only = dataclasses.replace(
            m_upd,
            signatures={raw_ed_pub: raw_ed.sign(p_upd).hex()},
            pq_signatures={},
        )
        ok_raw_npq, reason_raw_npq, _ = eng_raw_no_pq.verify_and_stage_update(
            m_raw_only,
            b_bytes,
            [att_u],
            candidate_predictor_fn=lambda it: {
                "junk": bool(it.ground_truth["junk"]),
                "adult": bool(it.ground_truth["adult"]),
                "category": str(it.ground_truth["category"]),
                "owner_scope": str(it.ground_truth["owner_scope"]),
            },
        )
        self.assertFalse(ok_raw_npq)
        self.assertIn("untrusted_pq_signers_when_pq_trust_store_unset", reason_raw_npq)

        # 11. Discussion 2 & 3: Merkle-authenticated WOTS+ unique leaf per shard (even at seq=1 across shards)
        # and remote verifier without local signer state + durable state file persistence
        with tempfile.TemporaryDirectory() as tmpdir2:
            state_dir = Path(tmpdir2) / "wots_state"
            old_state_dir = cg.WOTSPlusSHAKE256._STATE_DIR
            cg.WOTSPlusSHAKE256.set_state_dir(state_dir)
            try:
                r_priv, r_pub = cg.generate_ed25519_keypair()
                r_pq_pub_hex = cg.WOTSPlusSHAKE256.derive_pq_root_pubkey_hex(r_priv)
                # Sign two tombstones on signer side
                t_a = cg.sign_tombstone_list(seq=1, infohashes_or_blinded=["a" * 40], private_key_bytes=r_priv)
                t_b = cg.sign_tombstone_list(seq=2, infohashes_or_blinded=["b" * 40], private_key_bytes=r_priv)
                self.assertNotEqual(t_a.pq_pubkey_hex, t_b.pq_pubkey_hex)
                # Clear in-memory map to simulate remote process and verify against pinned root_pq_pub_hex
                cg.WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.clear()
                cg.WOTSPlusSHAKE256._WOTS_TO_ED25519_PUB.clear()
                remote_tstore = cg.TombstoneStore(
                    trusted_root_pubkeys_hex=[r_pub.hex()],
                    trusted_root_pq_pubkeys_hex=[r_pq_pub_hex],
                )
                ok_ra, msg_ra, _ = remote_tstore.apply_signed_tombstones(t_a)
                self.assertTrue(ok_ra, msg_ra)
                ok_rb, msg_rb, _ = remote_tstore.apply_signed_tombstones(t_b)
                self.assertTrue(ok_rb, msg_rb)

                # Build two different shards both with seq=1 and verify distinct WOTS+ leaf pubkeys
                db_p = Path(tmpdir2) / "remote_shards.sqlite"
                sync_r = dbsync.ArmigerDHTDatabaseSync(
                    db_path=db_p,
                    subscription=dbsync.SubscriptionConfig(
                        categories={"movie", "tv_show"},
                        languages={"en"},
                        quality_tiers={"hd_1080p"},
                        epochs={"2026_q4"},
                        enable_xxx=False,
                        sync_mode=dbsync.SyncMode.SUBSCRIBE_AND_SEED,
                    ),
                    master_secret=b"S" * 32,
                    trusted_root_pubkeys_hex=[r_pub.hex()],
                    trusted_root_pq_pubkeys_hex=[r_pq_pub_hex],
                )
                rec_m = dbsync.TorrentRecord(
                    infohash="1" * 40, title="Movie A 2026", category="movie", language="en",
                    quality_tier="hd_1080p", epoch="2026_q4", size_bytes=1_000_000, file_count=1,
                    seeders=10, leechers=1, content_sha256="2" * 64, file_list=["a.mkv"],
                    safety_gate_passed=True, created_at=now_ts,
                )
                rec_tv = dataclasses.replace(rec_m, infohash="2" * 40, title="Show B S01E01", category="tv_show")
                ms1, cts1 = sync_r.build_and_sign_shard("movie/en/hd_1080p/2026_q4", [rec_m], r_priv, seq=1)
                ms2, cts2 = sync_r.build_and_sign_shard("tv_show/en/hd_1080p/2026_q4", [rec_tv], r_priv, seq=1)
                self.assertNotEqual(ms1.root_pq_pubkey_hex, ms2.root_pq_pubkey_hex)
                # Clear in-memory map again before remote verification
                cg.WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.clear()
                cg.WOTSPlusSHAKE256._WOTS_TO_ED25519_PUB.clear()
                ms1 = ms1.with_attestations(
                    dbsync.create_mock_quorum_attestations(ms1.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
                )
                ms2 = ms2.with_attestations(
                    dbsync.create_mock_quorum_attestations(ms2.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
                )
                ok_s1, msg_s1, _ = sync_r.receive_and_apply_shard(ms1, cts1)
                self.assertTrue(ok_s1, msg_s1)
                ok_s2, msg_s2, _ = sync_r.receive_and_apply_shard(ms2, cts2)
                self.assertTrue(ok_s2, msg_s2)
                sync_r.close()
            finally:
                cg.WOTSPlusSHAKE256.set_state_dir(old_state_dir)

        # 12. Discussion 6f: PeerAttestation binds ip_address and unpinned keys with routed IPs are rejected
        unpinned_atts = []
        for idx, (asn_v, ip_v) in enumerate([
            (13335, "104.16.1.1"),
            (15169, "8.8.8.8"),
            (16509, "54.239.28.1"),
            (8075, "20.190.159.1"),
            (24940, "88.99.1.1"),
        ]):
            sk_u = Ed25519PrivateKey.from_private_bytes(bytes([0x80 + idx]) * 32)
            pk_u = sk_u.public_key().public_bytes_raw().hex()
            u_att = rsi.PeerAttestation(
                peer_id=f"unpinned_{idx}", asn=asn_v, peer_pubkey_hex=pk_u,
                timestamp=now_ts, signature_hex="", ip_address=ip_v,
            )
            u_att = dataclasses.replace(u_att, signature_hex=sk_u.sign(u_att.canonical_bytes(b_sha)).hex())
            unpinned_atts.append(u_att)
        ok_unp, reason_unp, _ = rsi.SybilQuorumVerifier.verify_quorum(
            unpinned_atts, target_hash=b_sha, min_distinct_asns=5, require_authoritative_asn=True
        )
        self.assertFalse(ok_unp)
        self.assertIn("unverified_authoritative_asn_binding", reason_unp)
        # Mutating ip_address invalidates canonical_bytes signature
        mut_ip_att = dataclasses.replace(att_u, ip_address="104.16.2.2")
        ok_mut, _, _ = rsi.SybilQuorumVerifier.verify_quorum(
            [mut_ip_att], target_hash=b_sha, min_distinct_asns=1, require_authoritative_asn=True
        )
        self.assertFalse(ok_mut)

        # 13. Discussion 6b & 6e: aggregate_continuous_deltas ciphertext/plaintext consistency and malformed stream fields
        agg_b = rsi.ByzantineResilientAggregator(min_distinct_asns=1)
        mal_ok, mal_reason, _ = agg_b.verify_physical_stream_signature(
            {"verified_features": {"has_video": True, "audio_channels": "bad"}, "dp_size_mb": "not_a_float"}
        )
        self.assertFalse(mal_ok)
        self.assertIn("malformed_numeric_field", mal_reason)
        enc_bundle = rsi.HybridPQCAEADEngine.encrypt_bundle(
            b'{"deltas":{"shared_temp":1.05}}', agg_b.aggregator_pubkey_bytes, seq=1
        )
        sk_m = Ed25519PrivateKey.from_private_bytes(bytes([0x91]) * 32)
        pk_m = sk_m.public_key().public_bytes_raw().hex()
        att_m = rsi.PeerAttestation(
            peer_id="m1", asn=13335, peer_pubkey_hex=pk_m, timestamp=now_ts, signature_hex=""
        )
        att_m = dataclasses.replace(
            att_m,
            signature_hex=sk_m.sign(
                att_m.canonical_bytes(hashlib.sha256(enc_bundle.ciphertext_hex.encode("utf-8")).hexdigest())
            ).hex(),
        )
        mismatch_sub = rsi.ContinuousDeltaSubmission(
            peer_id="m1",
            asn=13335,
            peer_pubkey_hex=pk_m,
            deltas={"shared_temp": 999.0},
            encrypted_bundle=enc_bundle,
            attestation=att_m,
        )
        res_mis = agg_b.aggregate_continuous_deltas([mismatch_sub], min_distinct_asns=1)
        self.assertFalse(res_mis.success)
        self.assertEqual(res_mis.rejected_submissions, 1)

        # 14. Discussion 6g & 6h: RSIUpdateAggregator requires contributor auth, rejects re-encrypted duplicate observations, and sign_manifest allocates fresh WOTS+ leaves
        ub_agg = pipe.RSIUpdateAggregator()
        contrib_sk = Ed25519PrivateKey.from_private_bytes(bytes([0x92]) * 32)
        contrib_pk_hex = contrib_sk.public_key().public_bytes_raw().hex()
        ub_agg.register_contributor(contrib_pk_hex, 13335)
        doc_payload = {
            "schema": pipe.RSI_TELEMETRY_SCHEMA,
            "epoch": ub_agg.epoch,
            "record": {"blinded_hash": "abc123def4567890"},
            "zero_cue_catalog_key": {"map": "title_year_map", "key": "0123456789abcdef", "category": "movie"},
        }
        unsigned_doc = json.dumps(doc_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        # Unsigned anonymous bundle is rejected by default
        b_anon = rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(
            unsigned_doc, ub_agg.x25519_pub + ub_agg.mlkem_ek, seq=1
        )
        anon_stats = ub_agg.ingest_bundles([b_anon])
        self.assertEqual(anon_stats["accepted"], 0)
        self.assertEqual(anon_stats["rejected"], 1)
        self.assertIn("missing_contributor_signature", ub_agg.rejected)

        signed_doc_payload = dict(doc_payload)
        signed_doc_payload["contributor_pubkey_hex"] = contrib_pk_hex
        signed_doc_payload["contributor_signature_hex"] = contrib_sk.sign(unsigned_doc).hex()
        raw_doc = json.dumps(signed_doc_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        b_re1 = rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(
            raw_doc, ub_agg.x25519_pub + ub_agg.mlkem_ek, seq=2
        )
        b_re2 = rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(
            raw_doc, ub_agg.x25519_pub + ub_agg.mlkem_ek, seq=3
        )
        ing_stats = ub_agg.ingest_bundles([b_re1, b_re2])
        self.assertEqual(ing_stats["accepted"], 1)
        self.assertEqual(ing_stats["rejected"], 1)
        self.assertIn("duplicate_observation_content_replay", ub_agg.rejected)

        fed_signer = pipe.generate_federation_signer()
        m_sig1 = pipe.RSIUpdateAggregator.sign_manifest(b'{"a":1}', seq=1, signers=[fed_signer], version="3.0.0")
        m_sig2 = pipe.RSIUpdateAggregator.sign_manifest(b'{"a":2}', seq=2, signers=[fed_signer], version="3.0.0")
        self.assertNotEqual(list(m_sig1.pq_signatures.keys()), list(m_sig2.pq_signatures.keys()))

        # 15. Wave 3 Follow-Up Jeeves Discussions (Notes 46974-46979):
        # 15a. Crossed root Ed25519:PQ pairs (A:PQB) rejected in QuorumVerifier and TombstoneStore
        sk_a, pub_a = cg.generate_ed25519_keypair()
        sk_b, pub_b = cg.generate_ed25519_keypair()
        pq_root_a = cg.WOTSPlusSHAKE256.derive_pq_root_pubkey_hex(sk_a)
        pq_root_b = cg.WOTSPlusSHAKE256.derive_pq_root_pubkey_hex(sk_b)
        tstore_paired = cg.TombstoneStore(
            trusted_root_pubkeys_hex=[f"{pub_a.hex()}:{pq_root_a}", f"{pub_b.hex()}:{pq_root_b}"]
        )
        t_valid_a = cg.sign_tombstone_list(seq=1, infohashes_or_blinded=["1" * 40], private_key_bytes=sk_a)
        # Forge crossed tombstone signed by Ed25519 key A + WOTS+ key B
        pq_priv_b, _ = cg.WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(sk_b, seq=0)
        t_crossed_root = dataclasses.replace(t_valid_a, pq_pubkey_hex=pq_root_b, signature_hex="", pq_signature_hex="")
        cp_root = t_crossed_root.canonical_payload()
        _, sig_pq_b_root, _, _ = cg.WOTSPlusSHAKE256.allocate_and_sign(pq_priv_b, cp_root, reserve_fresh=True)
        t_crossed_root = dataclasses.replace(
            t_crossed_root,
            signature_hex=Ed25519PrivateKey.from_private_bytes(sk_a).sign(cp_root).hex(),
            pq_signature_hex=sig_pq_b_root.hex(),
        )
        ok_cross_t, reason_cross_t, _ = tstore_paired.apply_signed_tombstones(t_crossed_root)
        self.assertFalse(ok_cross_t)
        self.assertIn("untrusted_pq_root_pubkey", reason_cross_t)

        qv_paired = dbsync.QuorumVerifier(
            trusted_root_pubkeys_hex=[f"{pub_a.hex()}:{pq_root_a}", f"{pub_b.hex()}:{pq_root_b}"],
            min_quorum=1,
            min_distinct_asns=1,
        )
        m_cross = dbsync.DHTShardManifest(
            seq=1, shard_id="movie/en/hd_1080p/2026_q4", epoch="2026_q4",
            record_count=1, compressed_size_bytes=10, uncompressed_size_bytes=10,
            merkle_root="a" * 64, ciphertext_sha256="b" * 64, bep46_infohash="c" * 40,
            timestamp=now_ts, root_pubkey_hex=pub_a.hex(), ed25519_signature_hex="",
            root_pq_pubkey_hex=pq_root_b, pq_signature_hex="",
        )
        mp_cross = m_cross.canonical_payload()
        _, m_pq_sig_b, _, _ = cg.WOTSPlusSHAKE256.allocate_and_sign(pq_priv_b, mp_cross, reserve_fresh=True)
        m_cross = dataclasses.replace(
            m_cross,
            ed25519_signature_hex=Ed25519PrivateKey.from_private_bytes(sk_a).sign(mp_cross).hex(),
            pq_signature_hex=m_pq_sig_b.hex(),
        )
        ok_qv_cross, reason_qv_cross = qv_paired.verify_manifest(m_cross, current_seq=0)
        self.assertFalse(ok_qv_cross)
        self.assertIn("untrusted_root_pq_pubkey", reason_qv_cross)

        # 15b. get_pending_telemetry_batch advances outbound _telemetry_seq monotonically across calls
        _, agg_pub_1216 = rsi.X25519ChaCha20Poly1305Engine.generate_keypair()
        eng_seq = rsi.ArmigerDHTRSIEngine(
            mode="anonymous_contribute",
            opt_in_sharing=True,
            current_seq=10,
            aggregator_x25519_pubkey_hex=agg_pub_1216.hex(),
        )
        eng_seq.ingest_candidate(cand, current_timestamp=100.0)
        b_batch1 = eng_seq.get_pending_telemetry_batch(now_timestamp=999999.0)
        self.assertEqual(len(b_batch1), 1)
        self.assertEqual(b_batch1[0].seq, 11)
        eng_seq.ingest_candidate(cand, current_timestamp=200.0)
        b_batch2 = eng_seq.get_pending_telemetry_batch(now_timestamp=999999.0)
        self.assertEqual(len(b_batch2), 1)
        self.assertEqual(b_batch2[0].seq, 12)

        # 15c. deserialize_shard_records rejects malformed field types cleanly
        bad_comp, _ = dbsync.compress_payload(
            json.dumps({
                "records": [{
                    "infohash": "1" * 40, "title": "T", "category": "movie", "language": "en",
                    "quality_tier": "hd_1080p", "epoch": "2026_q4", "size_bytes": "not-an-int",
                    "file_count": 1, "seeders": 1, "leechers": 0, "content_sha256": "2" * 64,
                }]
            }).encode("utf-8")
        )
        with self.assertRaises(ValueError):
            dbsync.deserialize_shard_records(bad_comp)

        # 15d. WOTSPlusSHAKE256 leaf index 959 succeeds and 960 raises ValueError before state mutation
        with tempfile.TemporaryDirectory() as tmpdir_wots:
            w_priv, _ = cg.WOTSPlusSHAKE256.generate_keypair(bytes([0x44]) * 64)
            sig_959 = cg.WOTSPlusSHAKE256.sign(w_priv, b"msg-959", leaf_index=959, state_dir=tmpdir_wots)
            self.assertTrue(cg.WOTSPlusSHAKE256.verify(
                cg.WOTSPlusSHAKE256.generate_keypair(bytes([0x44]) * 64)[1], b"msg-959", sig_959
            ))
            with self.assertRaises(ValueError) as ctx_960:
                cg.WOTSPlusSHAKE256.sign(w_priv, b"msg-960", leaf_index=960, state_dir=tmpdir_wots)
            self.assertIn("wots_leaf_index_exhausted", str(ctx_960.exception))

        # 15e. Unauthenticated wrapper cannot poison spoofed_pubkeys for honest peer in aggregate_continuous_deltas
        sk_hon = Ed25519PrivateKey.from_private_bytes(bytes([0xA1]) * 32)
        pk_hon = sk_hon.public_key().public_bytes_raw().hex()
        d_hon = {"shared_temp": 1.2}
        th_hon = hashlib.sha256(json.dumps(d_hon, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        att_hon = rsi.PeerAttestation(peer_id="hon", asn=13335, peer_pubkey_hex=pk_hon, timestamp=now_ts, signature_hex="")
        att_hon = dataclasses.replace(att_hon, signature_hex=sk_hon.sign(att_hon.canonical_bytes(th_hon)).hex())
        forged_wrapper = rsi.ContinuousDeltaSubmission(
            peer_id="forged", asn=99999, peer_pubkey_hex=pk_hon, deltas=d_hon, attestation=None
        )
        honest_sub = rsi.ContinuousDeltaSubmission(
            peer_id="hon", asn=13335, peer_pubkey_hex=pk_hon, deltas=d_hon, attestation=att_hon
        )
        res_unauth_poison = agg_b.aggregate_continuous_deltas([forged_wrapper, honest_sub], min_distinct_asns=1)
        self.assertTrue(res_unauth_poison.success)
        self.assertEqual(res_unauth_poison.distinct_asn_count, 1)

        # 15f. Audio guard does not treat [JPN - ENG Sub] or [RUS / ENG_Sub] as multi-English audio
        from audio_language_guard import apply_audio_language_guard_v3
        p_jpn_sub = apply_audio_language_guard_v3("Show.2024.[JPN - ENG Sub].1080p.mkv", p_en=0.25)
        self.assertLess(p_jpn_sub, 0.50)
        p_rus_sub = apply_audio_language_guard_v3("Movie.2024.[RUS / ENG_Sub].1080p.mkv", p_en=0.25)
        self.assertLess(p_rus_sub, 0.50)

        # 16. Wave 3 Round-3 Jeeves Discussions (Notes 47016-47021):
        # 16a (Note 47016): ingest_signed_delta_bundle advances per-sender/salt inbound sequence watermark
        eng_export = rsi.ArmigerDHTRSIEngine(
            mode="anonymous_contribute", opt_in_sharing=True, current_seq=0, dp_sigma=0.0
        )
        eng_ingest = rsi.ArmigerDHTRSIEngine(mode="pull_only", current_seq=0)
        b_seq5 = eng_export.export_signed_delta_bundle([0.1] * 8, [0.0, 0.0], n_samples=30, seq=5, rotate_key=False)
        b_seq3 = eng_export.export_signed_delta_bundle([0.2] * 8, [0.0, 0.0], n_samples=30, seq=3, rotate_key=False)
        assert b_seq5 is not None and b_seq3 is not None
        ok_d5, _, _ = eng_ingest.ingest_signed_delta_bundle(b_seq5)
        self.assertTrue(ok_d5)
        ok_d3, reason_d3, _ = eng_ingest.ingest_signed_delta_bundle(b_seq3)
        self.assertFalse(ok_d3)
        self.assertIn("stale_or_replayed_delta_seq", reason_d3)

        # 16b (Note 47017): receive_and_apply_shard rejects records marked safety_gate_passed=False
        with tempfile.TemporaryDirectory() as tmpdir_sg:
            sync_sg = dbsync.ArmigerDHTDatabaseSync(
                db_path=Path(tmpdir_sg) / "sg.sqlite",
                subscription=sub,
                master_secret=b"M" * 32,
                trusted_root_pubkeys_hex=[root_pub.hex()],
            )
            unsafe_rec = dataclasses.replace(good_rec, infohash="9" * 40, safety_gate_passed=False)
            m_sg, ct_sg = sync_sg.build_and_sign_shard(shard_id, [unsafe_rec], root_priv, seq=1)
            m_sg = m_sg.with_attestations(
                dbsync.create_mock_quorum_attestations(m_sg.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
            )
            ok_sg, reason_sg, _ = sync_sg.receive_and_apply_shard(m_sg, ct_sg)
            self.assertFalse(ok_sg)
            self.assertIn("record_failed_safety_gate", reason_sg)
            self.assertIsNone(sync_sg.db.get_record("9" * 40))
            sync_sg.close()

        # 16c (Notes 47018 & 47019): publish_shard_over_udp checks put_mutable responses & get_bep46_bundle bounds total/bytes
        import asyncio
        import bep44_udp_wire_harness as wire_h

        async def _test_udp_bounds_and_publish() -> None:
            swarm = wire_h.LocalKademliaSwarm(count=2, enforce_anonymity_guard=False)
            await swarm.start()
            try:
                with tempfile.TemporaryDirectory() as tmpdir_udp:
                    sync_u = dbsync.ArmigerDHTDatabaseSync(
                        db_path=Path(tmpdir_udp) / "u.sqlite",
                        subscription=sub,
                        master_secret=b"M" * 32,
                        trusted_root_pubkeys_hex=[root_pub.hex()],
                    )
                    m_u2, ct_u2 = sync_u.build_and_sign_shard(shard_id, [good_rec], root_priv, seq=2)
                    m_u2 = m_u2.with_attestations(
                        dbsync.create_mock_quorum_attestations(m_u2.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
                    )
                    ih_u2 = await swarm.publish_shard_over_udp(m_u2, ct_u2, root_priv)
                    self.assertEqual(len(ih_u2), 40)
                    # Publishing with stale sequence seq=1 must raise RuntimeError when put_mutable is rejected
                    m_u1, ct_u1 = sync_u.build_and_sign_shard(shard_id, [good_rec], root_priv, seq=1)
                    m_u1 = m_u1.with_attestations(
                        dbsync.create_mock_quorum_attestations(m_u1.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
                    )
                    with self.assertRaises(RuntimeError):
                        await swarm.publish_shard_over_udp(m_u1, ct_u1, root_priv)
                    # get_bep46_bundle rejects bundle when max_pieces / max_bundle_bytes bound is exceeded
                    ih_big = await swarm.put_bep46_bundle(b"x" * 6000)
                    self.assertIsNone(await swarm.get_bep46_bundle(ih_big, max_pieces=1))
                    self.assertIsNone(await swarm.get_bep46_bundle(ih_big, max_bundle_bytes=2048))
                    self.assertEqual(await swarm.get_bep46_bundle(ih_big), b"x" * 6000)
                    sync_u.close()
            finally:
                swarm.close()

        asyncio.run(_test_udp_bounds_and_publish())

        # 16d (Note 47020 & Note 47021 items 1 & 6): Wrapper IP vs attestation IP binding, canonical_title quorum, and aggregated_deltas property
        att_ip_bound = rsi.PeerAttestation(
            peer_id="hon_ip", asn=13335, peer_pubkey_hex=pk_hon, timestamp=now_ts, signature_hex="", ip_address="104.16.1.10"
        )
        att_ip_bound = dataclasses.replace(att_ip_bound, signature_hex=sk_hon.sign(att_ip_bound.canonical_bytes(th_hon)).hex())
        sub_ip_changed = rsi.ContinuousDeltaSubmission(
            peer_id="hon_ip", asn=13335, peer_pubkey_hex=pk_hon, deltas=d_hon, ip_address="104.16.2.99", attestation=att_ip_bound
        )
        res_ip_changed = agg_b.aggregate_continuous_deltas([sub_ip_changed], min_distinct_asns=1)
        self.assertFalse(res_ip_changed.success)
        self.assertEqual(res_ip_changed.rejected_submissions, 1)

        cat_agg = rsi.ByzantineResilientAggregator(min_distinct_asns=3, min_observations=3)
        self.assertEqual(
            rsi.CatalogConsensusResult(consensus_reached=False, reason="no", blinded_hash="h").aggregated_deltas,
            {},
        )
        cat_subs = []
        for idx, (asn_v, ip_v) in enumerate([(13335, "104.16.1.1"), (15169, "8.8.8.8"), (16509, "54.239.28.1")]):
            sk_c = Ed25519PrivateKey.from_private_bytes(bytes([0xB0 + idx]) * 32)
            pk_c = sk_c.public_key().public_bytes_raw().hex()
            rec_c = {
                "blinded_hash": "deadbeefcafe1234",
                "category": "movie",
                "adult": False,
                "anime": False,
                "lang": "english",
                "junk": False,
                "owner_scope": "allowed_type",
                "english_subs": True,
                "title_language": "english_title",
                "dp_size_mb": 1500.0,
                "dp_duration_mins": 120.0,
                "verified_features": {"has_video": True, "has_audio": True, "video_codec": "h264", "audio_codec": "aac", "audio_channels": 2, "catalog_type": "movie_catalog"},
            }
            enc_c = rsi.HybridPQCAEADEngine.encrypt_bundle(
                json.dumps(rec_c).encode("utf-8"), cat_agg.aggregator_pubkey_bytes, seq=idx + 1
            )
            th_c = hashlib.sha256(enc_c.ciphertext_hex.encode("utf-8")).hexdigest()
            att_c = rsi.PeerAttestation(peer_id=f"c_{idx}", asn=asn_v, peer_pubkey_hex=pk_c, timestamp=now_ts, signature_hex="", ip_address=ip_v)
            att_c = dataclasses.replace(att_c, signature_hex=sk_c.sign(att_c.canonical_bytes(th_c)).hex())
            # One observer supplies an unverified unilateral canonical_title
            cat_subs.append(
                rsi.TelemetrySubmission(
                    encrypted_bundle=enc_c,
                    asn=asn_v,
                    peer_id=f"c_{idx}",
                    peer_pubkey_hex=pk_c,
                    ip_address=ip_v,
                    attestation=att_c,
                    canonical_title="Rogue Single Observer Title" if idx == 2 else None,
                )
            )
        cat_res = cat_agg.aggregate_catalog_observations(cat_subs)
        self.assertTrue(cat_res.consensus_reached, cat_res.quarantined_reasons)
        assert cat_res.delta is not None
        self.assertIsNone(cat_res.delta.canonical_title)
        self.assertIn("category_prior:movie", cat_res.aggregated_deltas)
        # Tampering wrapper IP on catalog submission is quarantined with attestation_ip_mismatch
        bad_ip_cat = dataclasses.replace(cat_subs[0], ip_address="104.16.9.9")
        cat_res_bad_ip = cat_agg.aggregate_catalog_observations([bad_ip_cat])
        self.assertTrue(any("attestation_ip_mismatch" in r for r in cat_res_bad_ip.quarantined_reasons))

        # 16e (Note 47021 item 2): WOTS+ failed os.replace does not commit in-memory caches; retry persists durably
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmpdir_fail:
            w_priv_f, w_pub_f = cg.WOTSPlusSHAKE256.generate_keypair(bytes([0x55]) * 64)
            with patch("os.replace", side_effect=OSError("simulated_disk_replace_failure")):
                with self.assertRaises(OSError):
                    cg.WOTSPlusSHAKE256.sign(w_priv_f, b"msg-atomic-1", state_dir=tmpdir_fail)
            self.assertNotIn(w_pub_f.hex().lower(), cg.WOTSPlusSHAKE256._IN_MEM_USED_LEAVES)
            # Retry succeeds and writes durably to disk
            sig_retry = cg.WOTSPlusSHAKE256.sign(w_priv_f, b"msg-atomic-1", state_dir=tmpdir_fail)
            self.assertTrue(cg.WOTSPlusSHAKE256.verify(w_pub_f, b"msg-atomic-1", sig_retry))
            # Clear process caches and verify leaf 0 cannot be reused for a different message
            cg.WOTSPlusSHAKE256._IN_MEM_USED_LEAVES.clear()
            cg.WOTSPlusSHAKE256._IN_MEM_NEXT_LEAF.clear()
            cg.WOTSPlusSHAKE256._IN_MEM_MSG_TO_LEAF.clear()
            with self.assertRaises(ValueError) as ctx_reuse_0:
                cg.WOTSPlusSHAKE256.sign(w_priv_f, b"msg-atomic-2", leaf_index=0, state_dir=tmpdir_fail)
            self.assertIn("wots_leaf_index_already_used", str(ctx_reuse_0.exception))

        # 16f (Note 47021 items 3, 4, 5): SafeTensors with 0x80..0x05 weight bytes, keyed catalog hash rotation, and paired hybrid threshold signers
        st_header = json.dumps({"w": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}}).encode("utf-8")
        st_bytes = len(st_header).to_bytes(8, "little") + st_header + b"\x80\x02\x80\x03\x80\x04\x80\x05"
        ok_st, reason_st = rsi.SafeFormatValidator.validate(st_bytes, "safetensors")
        self.assertTrue(ok_st, reason_st)

        guard_k = rsi.AnonymityGuard(min_k_samples=25, catalog_secret=b"federation-epoch-secret-1")
        rec_k1 = guard_k.build_anonymous_catalog_record(key_or_hash="sintel|2010", category_code=0, sample_count=25, timestamp=1700002800)
        unkeyed_public_guess = hashlib.blake2b(
            rsi.AnonymityGuard.DOMAIN_CATALOG_V2 + b":sintel|2010", digest_size=8, person=b"armiger-cat-v2"
        ).hexdigest()
        self.assertNotEqual(rec_k1["h"], unkeyed_public_guess)
        guard_k.rotate_catalog_secret(b"federation-epoch-secret-2")
        rec_k2 = guard_k.build_anonymous_catalog_record(key_or_hash="sintel|2010", category_code=0, sample_count=25, timestamp=1700002800)
        self.assertNotEqual(rec_k1["h"], rec_k2["h"])

        # Disjoint classical and PQ signer identities rejected in ThresholdSignatureVerifier.verify
        s1_ed, s1_pub, s1_pq_priv, s1_pq_pub = rsi.ThresholdSignatureVerifier.generate_hybrid_signer()
        s2_ed, s2_pub, s2_pq_priv, s2_pq_pub = rsi.ThresholdSignatureVerifier.generate_hybrid_signer()
        m_disjoint = dataclasses.replace(
            m_upd,
            seq=2,
            signatures={s1_pub: ""},
            pq_signatures={s2_pq_pub: ""},
        )
        p_dis = m_disjoint.canonical_payload()
        m_disjoint = dataclasses.replace(
            m_disjoint,
            signatures={s1_pub: s1_ed.sign(p_dis).hex()},
            pq_signatures={s2_pq_pub: rsi.WOTSPlusSHAKE256.sign(s2_pq_priv, p_dis).hex()},
        )
        ok_dis, reason_dis = rsi.ThresholdSignatureVerifier.verify(
            m_disjoint,
            trusted_signers={s1_pub: "signer_1", s2_pub: "signer_2"},
            threshold_m=1,
            current_seq=0,
            trusted_pq_signers={s1_pq_pub: "signer_1", s2_pq_pub: "signer_2"},
            require_pq=True,
        )
        self.assertFalse(ok_dis)
        self.assertIn("insufficient_paired_hybrid_threshold_signatures", reason_dis)

        # 16g (Note 47021 items 7 & 8): 3 unpinned self-generated keys cannot satisfy min_support, and malformed telemetry components are rejected before retention
        agg_strict = pipe.RSIUpdateAggregator()
        unpinned_bundles = []
        for idx in range(3):
            sk_u3 = Ed25519PrivateKey.from_private_bytes(bytes([0xC0 + idx]) * 32)
            pk_u3 = sk_u3.public_key().public_bytes_raw().hex()
            u_doc = {
                "schema": pipe.RSI_TELEMETRY_SCHEMA,
                "epoch": agg_strict.epoch,
                "record": {"blinded_hash": f"bh_{idx:016x}"},
                "zero_cue_catalog_key": {"map": "title_year_map", "key": "fedcba9876543210", "category": "tv_show"},
            }
            u_bytes = json.dumps(u_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
            u_doc["contributor_pubkey_hex"] = pk_u3
            u_doc["contributor_signature_hex"] = sk_u3.sign(u_bytes).hex()
            unpinned_bundles.append(
                rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(
                    json.dumps(u_doc).encode("utf-8"), agg_strict.x25519_pub + agg_strict.mlkem_ek, seq=idx + 1
                )
            )
        res_unp3 = agg_strict.ingest_bundles(unpinned_bundles)
        self.assertEqual(res_unp3["accepted"], 0)
        self.assertEqual(res_unp3["rejected"], 3)
        zc_delta_unp, _ = agg_strict.build_zero_cue_delta(min_support=3)
        self.assertEqual(zc_delta_unp, {})

        # Malformed calibration/swarm telemetry signed by a trusted contributor is rejected without poisoning later valid aggregation
        agg_Strict2 = pipe.RSIUpdateAggregator()
        agg_Strict2.register_contributor(contrib_pk_hex, 13335)
        bad_prof_doc = {
            "schema": pipe.RSI_TELEMETRY_SCHEMA,
            "epoch": agg_Strict2.epoch,
            "record": {"blinded_hash": "badprof000000001"},
            "calibration_nll_profile": {"grid": [0.8, 1.0, 1.2], "heads": {"category": [0.5, 0.4]}},  # length mismatch
        }
        bp_bytes = json.dumps(bad_prof_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        bad_prof_doc["contributor_pubkey_hex"] = contrib_pk_hex
        bad_prof_doc["contributor_signature_hex"] = contrib_sk.sign(bp_bytes).hex()
        b_bad_prof = rsi.X25519ChaCha20Poly1305Engine.encrypt_bundle(
            json.dumps(bad_prof_doc).encode("utf-8"), agg_Strict2.x25519_pub + agg_Strict2.mlkem_ek, seq=10
        )
        res_bp = agg_Strict2.ingest_bundles([b_bad_prof])
        self.assertEqual(res_bp["accepted"], 0)
        self.assertEqual(res_bp["rejected"], 1)
        self.assertEqual(len(agg_Strict2.observations), 0)

        # --- Follow-up Jeeves findings (Notes 47039–47044) ---
        # 17a (Note 47039): Two-connection SQLite BEGIN IMMEDIATE write transaction prevents stale writer from overwriting newer sequence records
        with tempfile.TemporaryDirectory() as tmp_db_dir:
            shared_db_file = os.path.join(tmp_db_dir, "shared_shard.sqlite3")
            db_conn1 = dbsync.ShardDatabase(db_path=shared_db_file)
            db_conn2 = dbsync.ShardDatabase(db_path=shared_db_file)
            try:
                rec_v5 = dbsync.TorrentRecord(
                    infohash="cc" * 20,
                    title="Sintel.2010.2160p.UHD.BluRay",
                    category="movie",
                    language="en",
                    quality_tier="uhd_4k",
                    epoch="2026-w42",
                    size_bytes=5_000_000_000,
                    file_count=1,
                    seeders=100,
                    leechers=2,
                    content_sha256="dd" * 32,
                    file_list=["Sintel.2010.2160p.mkv"],
                    safety_gate_passed=True,
                    created_at=now_ts,
                )
                rec_v3_stale = dataclasses.replace(rec_v5, title="Stale.Overwrite.Attempt.1080p", seeders=1)
                n_ins5 = db_conn1.insert_records_batch([rec_v5], shard_id="movie/en/uhd_4k/2026-w42", seq=5)
                self.assertEqual(n_ins5, 1)
                with self.assertRaises(ValueError) as ctx_stale_db:
                    db_conn2.insert_records_batch([rec_v3_stale], shard_id="movie/en/uhd_4k/2026-w42", seq=3)
                self.assertIn("stale_or_downgrade_sequence:3<=5", str(ctx_stale_db.exception))
                fetched = db_conn1.get_record("cc" * 20)
                self.assertIsNotNone(fetched)
                self.assertEqual(fetched.title, "Sintel.2010.2160p.UHD.BluRay")
                self.assertEqual(fetched.seeders, 100)
                self.assertEqual(db_conn1.get_shard_sequences()["movie/en/uhd_4k/2026-w42"], 5)
            finally:
                db_conn1.close()
                db_conn2.close()

        # 17b (Note 47040): ArmigerDHTDatabaseSync enforces both opt_in_sharing AND outbound sync_mode in export_signed_bundle and publish_to_dht
        sync_outbound = dbsync.ArmigerDHTDatabaseSync(opt_in_sharing=True)
        try:
            valid_pre_bundle = sync_outbound.export_signed_bundle()
            self.assertIsNotNone(valid_pre_bundle)

            class _SpySwarm:
                def __init__(self) -> None:
                    self.put_calls = 0

                async def put_mutable(self, **kwargs: Any) -> list[dict[bytes, bytes]]:
                    self.put_calls += 1
                    return [{b"y": b"r"}]

            spy_swarm = _SpySwarm()
            for blocked_mode in (dbsync.SyncMode.OFF, dbsync.SyncMode.PULL_ONLY):
                sync_outbound.set_sync_mode(blocked_mode)
                self.assertIsNone(sync_outbound.export_signed_bundle())
                pub_ok, pub_reason, pub_cnt = asyncio.run(
                    sync_outbound.publish_to_dht(spy_swarm, bundle=valid_pre_bundle)
                )
                self.assertFalse(pub_ok)
                self.assertIn("sync_mode_disallows_outbound", pub_reason)
                self.assertEqual(pub_cnt, 0)
                self.assertEqual(spy_swarm.put_calls, 0)
        finally:
            sync_outbound.close()

        # 17c (Note 47041): _apply_dht_components rejects unhandled component keys, rolls back partial mutations, and supports reversible onnx_model_bytes updates
        p_comp = pipe.ArmigerUltraPipeline()
        fp_initial = p_comp.dht_state_fingerprint()
        snap_initial = p_comp.snapshot_dht_state()
        with self.assertRaises(pipe.DHTUpdateRejected) as ctx_unhandled:
            p_comp._apply_dht_components(
                {
                    "calibration_delta": {"heads": {"category": {"temperature": 1.25}}},
                    "unhandled_future_component": {"foo": "bar"},
                },
                snap_initial,
            )
        self.assertIn("unsupported_or_unapplied_update_components", str(ctx_unhandled.exception))
        p_comp._restore_dht_state(snap_initial)
        self.assertEqual(p_comp.dht_state_fingerprint(), fp_initial)

        if p_comp.onnx_path is not None and p_comp.onnx_path.exists():
            applied_ox = p_comp._apply_dht_components(
                {"onnx_model_bytes": p_comp.onnx_path.read_bytes()}, snap_initial
            )
            self.assertEqual(applied_ox, ["onnx_model_bytes"])
            p_comp._restore_dht_state(snap_initial)
            self.assertEqual(p_comp.dht_state_fingerprint(), fp_initial)

        # 17d (Note 47042): LocalKademliaSwarm.get_mutable & get_bep46_bundle fail over past unresponsive/corrupt replicas and honor max_transfer_seconds
        async def _verify_swarm_failover() -> None:
            import bep44_udp_wire_harness as wire_h

            swarm_fo = wire_h.LocalKademliaSwarm(count=4, enforce_anonymity_guard=False)
            await swarm_fo.start()
            try:
                sk_fo = Ed25519PrivateKey.from_private_bytes(b"\x51" * 32)
                priv_fo = b"\x51" * 32
                pub_fo = sk_fo.public_key().public_bytes_raw()
                salt_fo = b"failover-slot"
                await swarm_fo.put_mutable(priv_fo, salt_fo, b'{"hello":"world"}', seq=1, fanout=4)
                infohash_fo = await swarm_fo.put_bep46_bundle(b"A" * 2048, fanout=4)

                target_20 = wire_h.compute_bep44_target(pub_fo, salt_fo)
                closest = swarm_fo.closest_nodes(target_20, k=4)
                first_host, first_port, _ = closest[0]
                ih_closest = swarm_fo.closest_nodes(bytes.fromhex(infohash_fo), k=4)
                ih_first_host, ih_first_port, _ = ih_closest[0]
                real_udp_query = wire_h.udp_query

                async def _flaky_udp_query(host: str, port: int, query_dict: dict[bytes, Any], timeout: float = 2.0) -> dict[bytes, Any]:
                    self.assertLessEqual(timeout, 1.5)
                    if (host, port) in {(first_host, first_port), (ih_first_host, ih_first_port)}:
                        raise asyncio.TimeoutError("simulated_dead_first_replica")
                    return await real_udp_query(host, port, query_dict, timeout=timeout)

                with unittest.mock.patch.object(wire_h, "udp_query", side_effect=_flaky_udp_query):
                    got_ptr = await swarm_fo.get_mutable(pub_fo, salt_fo, max_transfer_seconds=1.2)
                    self.assertIsNotNone(got_ptr)
                    got_bundle = await swarm_fo.get_bep46_bundle(
                        infohash_fo,
                        expected_sha256_hex=hashlib.sha256(b"A" * 2048).hexdigest(),
                        max_transfer_seconds=1.2,
                    )
                    self.assertEqual(got_bundle, b"A" * 2048)
                    self.assertEqual(len(infohash_fo), 40)
            finally:
                swarm_fo.close()

        asyncio.run(_verify_swarm_failover())

        # 17e (Note 47043): aggregate_catalog_observations ignores unauthenticated wrapper sub.canonical_title when rec_dict lacks authenticated title
        agg_cat_auth = rsi.ByzantineResilientAggregator(min_distinct_asns=3, min_observations=3)
        cat_subs_unauth_title = []
        base_rec = rsi.AnonymizedTelemetryRecord(
            blinded_stem=rsi.BlindedTitleStem(blinded_hash="1122334455667788" * 4, token_features=["1080p"], length_bucket="medium"),
            category="movie",
            adult=False,
            anime=False,
            lang="english",
            junk=False,
            owner_scope="allowed_type",
            english_subs=True,
            title_language="english_title",
            dp_size_mb=1024.0,
            dp_duration_mins=120.0,
            verified_features={"has_video": True, "has_audio": True, "video_codec": "h264", "audio_codec": "aac", "audio_channels": 2.0, "catalog_type": "movie_catalog"},
            timestamp=1700002800,
            evidence_masks={"category": True, "adult": True, "anime": True, "lang": True, "junk": True, "owner_scope": True, "english_subs": True, "title_language": True},
        )
        for idx, (asn_v, ip_v) in enumerate([(13335, "104.16.1.1"), (15169, "8.8.8.8"), (16509, "52.94.1.1")]):
            sk_c = Ed25519PrivateKey.from_private_bytes(bytes([0x61 + idx]) * 32)
            pk_c = sk_c.public_key().public_bytes_raw().hex()
            rsi.AuthoritativeASNVerifier.register_global_pinned_key(pk_c, asn_v)
            enc_b = rsi.HybridPQCAEADEngine.encrypt_bundle(
                base_rec.canonical_json_bytes(), agg_cat_auth.aggregator_pubkey_bytes, seq=idx + 1
            )
            th_c = hashlib.sha256(enc_b.ciphertext_hex.encode("utf-8")).hexdigest()
            att_c = rsi.PeerAttestation(
                peer_id=f"cat_peer_{idx}",
                asn=asn_v,
                peer_pubkey_hex=pk_c,
                timestamp=now_ts,
                signature_hex="",
                ip_address=ip_v,
            )
            att_c = dataclasses.replace(att_c, signature_hex=sk_c.sign(att_c.canonical_bytes(th_c)).hex())
            cat_subs_unauth_title.append(
                rsi.TelemetrySubmission(
                    encrypted_bundle=enc_b,
                    asn=asn_v,
                    peer_id=f"cat_peer_{idx}",
                    peer_pubkey_hex=pk_c,
                    ip_address=ip_v,
                    attestation=att_c,
                    canonical_title="Relay Injected Spoofed Title (2026)",
                )
            )
        cat_res_unauth = agg_cat_auth.aggregate_catalog_observations(
            cat_subs_unauth_title, emit_bep46_manifest=False
        )
        self.assertTrue(cat_res_unauth.consensus_reached, cat_res_unauth.reason)
        self.assertIsNone(cat_res_unauth.delta.canonical_title)

        # 17f (Note 47044 items 1 & 2): Continuous delta per-peer deduplication, freshness enforcement, and malformed-item quarantine
        agg_dedup = rsi.ByzantineResilientAggregator(min_distinct_asns=3)
        dedup_subs: list[Any] = [
            {"malformed": "missing_required_fields"},
            {"peer_id": "bad_asn", "asn": "not-an-int", "peer_pubkey_hex": "aa" * 32, "deltas": {"k": 1.0}},
            {"peer_id": "bad_att", "asn": 13335, "peer_pubkey_hex": "bb" * 32, "deltas": {"k": 1.0}, "attestation": "invalid_att"},
        ]
        # In ASN 13335, create 2 honest peers (value 1.0 and 1.2) and 1 peer whose signed submission (value 9.5) is replayed 5 times
        asn_13335_peers = [(0x81, 1.0), (0x82, 1.2), (0x83, 9.5)]
        for seed_b, val_v in asn_13335_peers:
            sk_p = Ed25519PrivateKey.from_private_bytes(bytes([seed_b]) * 32)
            pk_p = sk_p.public_key().public_bytes_raw().hex()
            rsi.AuthoritativeASNVerifier.register_global_pinned_key(pk_p, 13335)
            d_map = {"temp": val_v}
            th_p = hashlib.sha256(json.dumps(d_map, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            att_p = rsi.PeerAttestation(f"p_{seed_b}", 13335, pk_p, now_ts, "", ip_address=f"104.16.{seed_b - 0x80}.1")
            att_p = dataclasses.replace(att_p, signature_hex=sk_p.sign(att_p.canonical_bytes(th_p)).hex())
            sub_p = rsi.ContinuousDeltaSubmission(
                peer_id=f"p_{seed_b}",
                asn=13335,
                peer_pubkey_hex=pk_p,
                deltas=d_map,
                ip_address=f"104.16.{seed_b - 0x80}.1",
                attestation=att_p,
                timestamp=now_ts,
            )
            repeat_n = 5 if seed_b == 0x83 else 1
            for _ in range(repeat_n):
                dedup_subs.append(sub_p)

        for idx_o, (asn_o, ip_o, val_o) in enumerate([(15169, "8.8.8.8", 1.2), (16509, "52.94.2.1", 1.2)]):
            sk_o = Ed25519PrivateKey.from_private_bytes(bytes([0x88 + idx_o]) * 32)
            pk_o = sk_o.public_key().public_bytes_raw().hex()
            rsi.AuthoritativeASNVerifier.register_global_pinned_key(pk_o, asn_o)
            d_map_o = {"temp": val_o}
            th_o = hashlib.sha256(json.dumps(d_map_o, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            att_o = rsi.PeerAttestation(f"po_{idx_o}", asn_o, pk_o, now_ts, "", ip_address=ip_o)
            att_o = dataclasses.replace(att_o, signature_hex=sk_o.sign(att_o.canonical_bytes(th_o)).hex())
            dedup_subs.append(
                rsi.ContinuousDeltaSubmission(
                    peer_id=f"po_{idx_o}",
                    asn=asn_o,
                    peer_pubkey_hex=pk_o,
                    deltas=d_map_o,
                    ip_address=ip_o,
                    attestation=att_o,
                    timestamp=now_ts,
                )
            )

        res_dedup = agg_dedup.aggregate_continuous_deltas(dedup_subs)
        self.assertTrue(res_dedup.success, res_dedup.reason)
        # Because peer 0x83's 4 duplicate replays were deduplicated, ASN 13335's 3 unique peers [1.0, 1.2, 9.5] have median 1.2!
        self.assertAlmostEqual(float(res_dedup.median_deltas["temp"]), 1.2, places=5)
        self.assertGreaterEqual(res_dedup.rejected_submissions, 7)

        # 17g (Note 47044 items 3, 4, 5, 6, 7): strict bool opt_in_sharing, standalone scale & escaped PII firewall, single-derivation catalog hash equality, PQ signer_attestations propagation, and non-dict calibration containers
        for bad_opt_in in ("0", "false", "no", "", 0, 1, [], {}, object()):
            with self.assertRaises((TypeError, ValueError)):
                rsi.resolve_opt_in_sharing(bad_opt_in)  # type: ignore[arg-type]

        guard_fw = rsi.AnonymityGuard(min_k_samples=25, dp_sigma=0.0)
        for bad_wire_bytes in (
            b'{"epoch":1700002800,"scale":"standalone_unvalidated_string","seq":1,"v":2}',
            b'{"epoch":1700002800,"scale":0.05,"seq":1,"v":2}',
            b'{"delta_b_q8":[0],"delta_w_q8":[1],"epoch":1700002800,"model_tier":"\\u0070asskey=secret123","n_bucket":25,"scale":0.01,"seq":1,"v":2}',
            b'{"delta_b_q8":[0],"delta_w_q8":[1],"epoch":1700002800,"model_tier":"192\\u002e168.1.1","n_bucket":25,"scale":0.01,"seq":1,"v":2}',
        ):
            with self.assertRaises(rsi.AnonymityViolationError):
                guard_fw.sign_wire_payload(bad_wire_bytes, seq=1)

        explicit_sec = b"single-derivation-test-secret-2026"
        h_direct = rsi.AnonymityGuard.hash_catalog_key("sintel|2010", catalog_secret=explicit_sec)
        guard_eq = rsi.AnonymityGuard(min_k_samples=25, catalog_secret=explicit_sec)
        rec_from_key = guard_eq.build_anonymous_catalog_record(
            key_or_hash="sintel|2010", category_code=0, sample_count=25, timestamp=1700002800, is_prehashed=False
        )
        rec_from_pre = guard_eq.build_anonymous_catalog_record(
            key_or_hash=h_direct, category_code=0, sample_count=25, timestamp=1700002800, is_prehashed=True
        )
        self.assertEqual(rec_from_key["h"], h_direct)
        self.assertEqual(rec_from_key, rec_from_pre)

        # Classical-only signer_attestations on a hybrid manifest must be rejected when require_pq=True
        m_classical_att = dataclasses.replace(m_upd, seq=3, signer_attestations=[att_u])
        p_ca = m_classical_att.canonical_payload()
        m_classical_att = dataclasses.replace(
            m_classical_att,
            signatures={s_pub: s_ed.sign(p_ca).hex()},
            pq_signatures={s_pq_pub: rsi.WOTSPlusSHAKE256.sign(s_pq_priv, p_ca, leaf_index=2).hex()},
        )
        ok_ca, reason_ca = rsi.ThresholdSignatureVerifier.verify(
            m_classical_att,
            trusted_signers={s_pub: "root"},
            threshold_m=1,
            current_seq=0,
            trusted_pq_signers={s_pq_pub: "root_pq"},
            require_pq=True,
        )
        self.assertFalse(ok_ca)
        self.assertIn("manifest_signer_attestation_failed", reason_ca)

        for bad_container in (["not_a_dict"], "string_val", True, False, [], "", 0):
            with self.assertRaises(pipe.DHTUpdateRejected):
                pipe._validate_calibration_delta({"heads": bad_container})
            with self.assertRaises(pipe.DHTUpdateRejected):
                pipe._validate_calibration_delta({"mondrian_thresholds": bad_container})

        # --- Wave 3 Round-5 Jeeves Discussions (Notes 47069–47074) ---
        # 18a (Note 47069): ArmigerDHTDatabaseSync.ingest_signed_bundle enforces per-(pubkey, salt) monotonic sequence high-water mark
        sync_cat_seq = dbsync.ArmigerDHTDatabaseSync(opt_in_sharing=True)
        try:
            cat_b5 = sync_cat_seq.export_signed_bundle(seq=5, rotate_key=False)
            cat_b3 = sync_cat_seq.export_signed_bundle(seq=3, rotate_key=False)
            assert cat_b5 is not None and cat_b3 is not None
            ok_cb5, msg_cb5, _ = sync_cat_seq.ingest_signed_bundle(cat_b5)
            self.assertTrue(ok_cb5, msg_cb5)
            # Duplicate replay of seq=5 must be rejected
            ok_cb5_dup, msg_cb5_dup, _ = sync_cat_seq.ingest_signed_bundle(cat_b5)
            self.assertFalse(ok_cb5_dup)
            self.assertIn("stale_or_replayed_seq:5<=5", msg_cb5_dup)
            # Older seq=3 after seq=5 must be rejected
            ok_cb3, msg_cb3, _ = sync_cat_seq.ingest_signed_bundle(cat_b3)
            self.assertFalse(ok_cb3)
            self.assertIn("stale_or_replayed_seq:3<=5", msg_cb3)
        finally:
            sync_cat_seq.close()

        # 18b (Note 47070): deserialize_shard_records and receive_and_apply_shard reject integers > SQLITE_MAX_INT64 cleanly
        with tempfile.TemporaryDirectory() as tmp_i64:
            sync_i64 = dbsync.ArmigerDHTDatabaseSync(
                db_path=Path(tmp_i64) / "i64.sqlite",
                subscription=sub,
                master_secret=b"M" * 32,
                trusted_root_pubkeys_hex=[root_pub.hex()],
            )
            try:
                oversized_rec = dataclasses.replace(good_rec, infohash="e" * 40, size_bytes=1 << 63)
                m_i64, ct_i64 = sync_i64.build_and_sign_shard(shard_id, [oversized_rec], root_priv, seq=1)
                m_i64 = m_i64.with_attestations(
                    dbsync.create_mock_quorum_attestations(m_i64.canonical_digest(), [13335, 15169, 16509, 8075, 24940])
                )
                ok_i64, reason_i64, cnt_i64 = sync_i64.receive_and_apply_shard(m_i64, ct_i64)
                self.assertFalse(ok_i64)
                self.assertEqual(cnt_i64, 0)
                self.assertIn("integer_overflow_for_sqlite_int64:size_bytes", reason_i64)
                self.assertEqual(sync_i64.get_shard_seq(shard_id), 0)
                self.assertIsNone(sync_i64.db.get_record("e" * 40))
            finally:
                sync_i64.close()

        # 18c (Note 47071): udp_query, get_mutable, and get_bep46_bundle skip malformed bencoded list or non-dict r responses and fail over
        async def _verify_malformed_krpc_failover() -> None:
            import bep44_udp_wire_harness as wire_h

            swarm_krpc = wire_h.LocalKademliaSwarm(count=4, enforce_anonymity_guard=False)
            await swarm_krpc.start()
            try:
                sk_k = Ed25519PrivateKey.from_private_bytes(b"\x52" * 32)
                priv_k = b"\x52" * 32
                pub_k = sk_k.public_key().public_bytes_raw()
                salt_k = b"krpc-shape-slot"
                await swarm_krpc.put_mutable(priv_k, salt_k, b'{"status":"ok"}', seq=1, fanout=4)
                ih_k = await swarm_krpc.put_bep46_bundle(b"B" * 1500, fanout=4)

                target_k = wire_h.compute_bep44_target(pub_k, salt_k)
                closest_k = swarm_krpc.closest_nodes(target_k, k=4)
                bad_host_1, bad_port_1, _ = closest_k[0]
                bad_host_2, bad_port_2, _ = closest_k[1]
                ih_closest_k = swarm_krpc.closest_nodes(bytes.fromhex(ih_k), k=4)
                ih_bad_1, ih_port_1, _ = ih_closest_k[0]
                ih_bad_2, ih_port_2, _ = ih_closest_k[1]
                real_q = wire_h.udp_query

                async def _malformed_first_two_replicas(host: str, port: int, q_dict: dict[bytes, Any], timeout: float = 2.0) -> Any:
                    q_name = q_dict.get(b"q")
                    if q_name == b"get":
                        if (host, port) == (bad_host_1, bad_port_1):
                            raise ValueError("KRPC response must be a bencoded dict")
                        if (host, port) == (bad_host_2, bad_port_2):
                            return {b"t": b"x", b"y": b"r", b"r": b"malformed_non_dict_r_body"}
                    elif q_name == b"get_bep46_piece":
                        if (host, port) == (ih_bad_1, ih_port_1):
                            raise ValueError("KRPC response must be a bencoded dict")
                        if (host, port) == (ih_bad_2, ih_port_2):
                            return {b"t": b"x", b"y": b"r", b"r": b"malformed_non_dict_r_body"}
                    return await real_q(host, port, q_dict, timeout=timeout)

                with unittest.mock.patch.object(wire_h, "udp_query", side_effect=_malformed_first_two_replicas):
                    got_item = await swarm_krpc.get_mutable(pub_k, salt_k)
                    self.assertIsNotNone(got_item)
                    self.assertEqual(got_item.v, b'{"status":"ok"}')
                    got_blob = await swarm_krpc.get_bep46_bundle(ih_k)
                    self.assertEqual(got_blob, b"B" * 1500)
            finally:
                swarm_krpc.close()

        asyncio.run(_verify_malformed_krpc_failover())

        # 18d (Note 47072): WOTSPlusSHAKE256 does not default to tempfile.gettempdir() and fails closed when state file is deleted after signing
        with tempfile.TemporaryDirectory() as tmp_wots_prod:
            state_sub = Path(tmp_wots_prod) / "durable_state"
            seal_sub = Path(tmp_wots_prod) / "durable_seal"
            old_sd = cg.WOTSPlusSHAKE256.get_state_dir()
            old_seal = cg.WOTSPlusSHAKE256._SEAL_DIR
            try:
                cg.WOTSPlusSHAKE256.set_state_dir(None)
                self.assertNotEqual(
                    Path(cg.WOTSPlusSHAKE256.get_state_dir()).resolve(),
                    Path(tempfile.gettempdir()).resolve(),
                )
                cg.WOTSPlusSHAKE256.set_state_dir(state_sub)
                cg.WOTSPlusSHAKE256.set_seal_dir(seal_sub)
                w_priv_p, w_pub_p = cg.WOTSPlusSHAKE256.generate_keypair(bytes([0x66]) * 64)
                sig_p0 = cg.WOTSPlusSHAKE256.sign(w_priv_p, b"prod-msg-0", state_dir=state_sub)
                self.assertTrue(cg.WOTSPlusSHAKE256.verify(w_pub_p, b"prod-msg-0", sig_p0))
                # Simulate loss/wipe of state file while independent freshness seal survives
                for f_p in state_sub.glob("wots_state_*.json"):
                    f_p.unlink()
                cg.WOTSPlusSHAKE256._IN_MEM_USED_LEAVES.clear()
                cg.WOTSPlusSHAKE256._IN_MEM_NEXT_LEAF.clear()
                cg.WOTSPlusSHAKE256._IN_MEM_MSG_TO_LEAF.clear()
                with self.assertRaises(RuntimeError) as ctx_lost_state:
                    cg.WOTSPlusSHAKE256.sign(w_priv_p, b"prod-msg-1", state_dir=state_sub)
                self.assertIn("wots_durable_state_missing_after_establishment", str(ctx_lost_state.exception))
            finally:
                cg.WOTSPlusSHAKE256.set_state_dir(old_sd)
                cg.WOTSPlusSHAKE256.set_seal_dir(old_seal)

        # 18e (Note 47073 & Note 47074 item 1): aggregate_catalog_observations rejects stale/future/boolean attestation timestamps and validates title wrappers before consuming peer/IP quotas
        agg_fresh = rsi.ByzantineResilientAggregator(min_distinct_asns=1, min_observations=1)
        sk_f = Ed25519PrivateKey.from_private_bytes(bytes([0x71]) * 32)
        pk_f = sk_f.public_key().public_bytes_raw().hex()
        rsi.AuthoritativeASNVerifier.register_global_pinned_key(pk_f, 13335)
        enc_f = rsi.HybridPQCAEADEngine.encrypt_bundle(
            base_rec.canonical_json_bytes(), agg_fresh.aggregator_pubkey_bytes, seq=1
        )
        th_f = hashlib.sha256(enc_f.ciphertext_hex.encode("utf-8")).hexdigest()
        for bad_ts, expected_err in (
            (now_ts - 8 * 86400, "expired_attestation_timestamp"),
            (now_ts + 900, "future_attestation_timestamp"),
        ):
            att_bad_ts = rsi.PeerAttestation("pf", 13335, pk_f, bad_ts, "", ip_address="104.16.10.1")
            att_bad_ts = dataclasses.replace(att_bad_ts, signature_hex=sk_f.sign(att_bad_ts.canonical_bytes(th_f)).hex())
            res_bad_ts = agg_fresh.aggregate_catalog_observations([
                rsi.TelemetrySubmission(
                    encrypted_bundle=enc_f, asn=13335, peer_id="pf", peer_pubkey_hex=pk_f,
                    ip_address="104.16.10.1", attestation=att_bad_ts,
                )
            ], now_ts=now_ts, emit_bep46_manifest=False)
            self.assertFalse(res_bad_ts.consensus_reached)
            self.assertTrue(any(expected_err in r for r in res_bad_ts.quarantined_reasons), res_bad_ts.quarantined_reasons)
        res_bool_ts = agg_fresh.aggregate_catalog_observations([
            {
                "encrypted_bundle": enc_f,
                "asn": 13335,
                "attestation": {
                    "peer_id": "pf",
                    "asn": 13335,
                    "peer_pubkey_hex": pk_f,
                    "timestamp": True,
                    "signature_hex": "00" * 64,
                },
            }
        ], now_ts=now_ts, emit_bep46_manifest=False)
        self.assertFalse(res_bool_ts.consensus_reached)
        self.assertTrue(any("invalid_attestation_timestamp_type" in r for r in res_bool_ts.quarantined_reasons))

        # Unsigned wrapper with mismatched canonical_title placed BEFORE honest submission from same peer must not burn peer's deduplication slot
        rec_with_title_dict = json.loads(base_rec.canonical_json_bytes().decode("utf-8"))
        rec_with_title_dict["canonical_title"] = "Sintel (2010)"
        enc_t = rsi.HybridPQCAEADEngine.encrypt_bundle(
            json.dumps(rec_with_title_dict, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            agg_fresh.aggregator_pubkey_bytes,
            seq=2,
        )
        th_t = hashlib.sha256(enc_t.ciphertext_hex.encode("utf-8")).hexdigest()
        att_t = rsi.PeerAttestation("pf", 13335, pk_f, now_ts, "", ip_address="104.16.10.1")
        att_t = dataclasses.replace(att_t, signature_hex=sk_f.sign(att_t.canonical_bytes(th_t)).hex())
        spoofed_title_wrapper = rsi.TelemetrySubmission(
            encrypted_bundle=enc_t, asn=13335, peer_id="pf", peer_pubkey_hex=pk_f,
            ip_address="104.16.10.1", attestation=att_t, canonical_title="Attacker Injected Title",
        )
        honest_title_sub = rsi.TelemetrySubmission(
            encrypted_bundle=enc_t, asn=13335, peer_id="pf", peer_pubkey_hex=pk_f,
            ip_address="104.16.10.1", attestation=att_t, canonical_title="Sintel (2010)",
        )
        res_quota_safe = agg_fresh.aggregate_catalog_observations(
            [spoofed_title_wrapper, honest_title_sub], now_ts=now_ts, emit_bep46_manifest=False
        )
        self.assertTrue(res_quota_safe.consensus_reached, res_quota_safe.reason)
        self.assertEqual(res_quota_safe.delta.canonical_title, "Sintel (2010)")

        # 18f (Note 47074 items 2, 3, 4, 5): duplicate JSON key rejection, bounded logarithmic DP parameters, env-independent offline catalog hash, and span_bio B-EP/I-EP class labels
        dup_key_wire = (
            b'{"delta_b_q8":[0],"delta_w_q8":[1],"epoch":1700002800,'
            b'"model_tier":"disallowed_first_value","model_tier":"int8",'
            b'"n_bucket":25,"scale":0.01,"seq":1,"v":2}'
        )
        with self.assertRaises(rsi.AnonymityViolationError) as ctx_dup_json:
            guard_fw.sign_wire_payload(dup_key_wire, seq=1)
        self.assertIn("duplicate_json_key:model_tier", str(ctx_dup_json.exception))

        dp_anon = rsi.TelemetryAnonymizer(blinding_salt=b"dp_test_salt_1234", dp_epsilon=1.0)
        dp_params_both = dp_anon.compute_dp_parameters(size_bytes=500 * 1024 * 1024 * 1024, duration_seconds=999999.0)
        self.assertEqual(dp_params_both["clipped_size_mb"], 65536.0)
        self.assertEqual(dp_params_both["clipped_duration_mins"], 1280.0)
        self.assertEqual(dp_params_both["released_fields"], 2)
        self.assertAlmostEqual(dp_params_both["epsilon_size"], 0.5)
        self.assertAlmostEqual(dp_params_both["epsilon_duration"], 0.5)
        self.assertAlmostEqual(dp_params_both["size_laplace_scale"], 24.0)
        self.assertAlmostEqual(dp_params_both["duration_laplace_scale"], 16.0)
        dp_params_size_only = dp_anon.compute_dp_parameters(size_bytes=1024, duration_seconds=None)
        self.assertEqual(dp_params_size_only["clipped_size_mb"], 16.0)
        self.assertEqual(dp_params_size_only["released_fields"], 1)
        self.assertAlmostEqual(dp_params_size_only["size_laplace_scale"], 12.0)

        expected_offline_hash = hashlib.blake2b(b"sintel|2010", digest_size=8, person=b"armiger-zc-v2").hexdigest()
        with unittest.mock.patch.dict(os.environ, {"ARMIGER_CATALOG_HMAC_KEY": "env-secret-should-not-change-offline-id"}):
            self.assertEqual(hash_catalog_key("sintel|2010"), expected_offline_hash)
            try:
                from armiger.zero_cue_plugin import hash_catalog_key as pkg_hash_catalog_key
                self.assertEqual(pkg_hash_catalog_key("sintel|2010"), expected_offline_hash)
            except ImportError:
                pass

        from armiger_inference import NEURAL_HEAD_CLASSES, SPAN_BIO_CLASSES
        from byte_span_extractor import SPAN_BIO_CLASSES as BSE_SPAN_BIO_CLASSES
        self.assertEqual(tuple(NEURAL_HEAD_CLASSES["span_bio"]), SPAN_BIO_CLASSES)
        self.assertEqual(SPAN_BIO_CLASSES, BSE_SPAN_BIO_CLASSES)
        self.assertEqual(SPAN_BIO_CLASSES[5:7], ("B-EP", "I-EP"))
        try:
            from armiger.byte_span_extractor import SPAN_BIO_CLASSES as PKG_SPAN_BIO_CLASSES
            self.assertEqual(SPAN_BIO_CLASSES, PKG_SPAN_BIO_CLASSES)
        except ImportError:
            pass

        # --- Wave 3 Round-6 Jeeves Discussions (Notes 47148–47153 & Notes 47165–47170) ---
        # 19a (Notes 47148 & 47165): Bounded inbound delta replay state, 64-byte BEP-44 salt cap, and evicted sender sequence floor
        eng_bounded = rsi.ArmigerDHTRSIEngine(
            mode="anonymous_contribute",
            opt_in_sharing=True,
            max_seen_delta_digests=4,
            max_tracked_senders=2,
            delta_freshness_window_seconds=3600,
        )
        b_long_salt = eng_bounded.export_signed_delta_bundle([0.1] * 8, [0.0, 0.0], n_samples=25, seq=1, salt=b"s" * 65)
        assert b_long_salt is not None
        ok_ls, reason_ls, _ = eng_bounded.ingest_signed_delta_bundle(b_long_salt)
        self.assertFalse(ok_ls)
        self.assertIn("bep44_salt_exceeds_64_bytes", reason_ls)

        b_ev1 = eng_bounded.export_signed_delta_bundle([0.1] * 8, [0.0, 0.0], n_samples=25, seq=2, salt=b"s1", rotate_key=True)
        b_ev2 = eng_bounded.export_signed_delta_bundle([0.1] * 8, [0.0, 0.0], n_samples=25, seq=3, salt=b"s2", rotate_key=True)
        b_ev3 = eng_bounded.export_signed_delta_bundle([0.1] * 8, [0.0, 0.0], n_samples=25, seq=4, salt=b"s3", rotate_key=True)
        assert b_ev1 is not None and b_ev2 is not None and b_ev3 is not None
        self.assertTrue(eng_bounded.ingest_signed_delta_bundle(b_ev1)[0])
        self.assertTrue(eng_bounded.ingest_signed_delta_bundle(b_ev2)[0])
        self.assertTrue(eng_bounded.ingest_signed_delta_bundle(b_ev3)[0])
        self.assertLessEqual(len(eng_bounded._inbound_sender_seq), 2)
        self.assertLessEqual(len(eng_bounded._seen_delta_digests), 4)
        # Replaying evicted sender b_ev1 (seq=2) must still be rejected by _evicted_sender_seq_floor!
        ok_rep_ev1, reason_rep_ev1, _ = eng_bounded.ingest_signed_delta_bundle(b_ev1)
        self.assertFalse(ok_rep_ev1)
        self.assertIn("stale_or_replayed_delta_seq", reason_rep_ev1)

        # 19b (Note 47166): aggregator_x25519_pubkey_hex validation, bounded telemetry_queue, and bounded encryption retries
        with self.assertRaises(ValueError):
            rsi.ArmigerDHTRSIEngine(
                mode="anonymous_contribute",
                opt_in_sharing=True,
                aggregator_x25519_pubkey_hex="deadbeef",
            )
        eng_q = rsi.ArmigerDHTRSIEngine(
            mode="anonymous_contribute",
            opt_in_sharing=True,
            max_telemetry_queue_size=2,
            max_item_encryption_retries=2,
        )
        for idx_q in range(3):
            c_q = dataclasses.replace(cand, raw_title=f"Sintel.2010.1080p.Part{idx_q}", catalog_id_value=9000 + idx_q)
            eng_q.ingest_candidate(c_q, current_timestamp=100.0)
        self.assertEqual(len(eng_q.telemetry_queue), 2)
        self.assertEqual(eng_q.sharing_status()["telemetry_dropped_backlog"], 1)
        eng_q.aggregator_x25519_pubkey_hex = "bad_hex_after_init"
        self.assertEqual(eng_q.get_pending_telemetry_batch(now_timestamp=999999.0), [])
        self.assertEqual(len(eng_q.telemetry_queue), 2)
        self.assertEqual(eng_q.get_pending_telemetry_batch(now_timestamp=999999.0), [])
        self.assertEqual(len(eng_q.telemetry_queue), 0)
        self.assertGreaterEqual(eng_q.sharing_status()["telemetry_encryption_errors"], 2)
        self.assertEqual(eng_q.sharing_status()["telemetry_dropped_backlog"], 3)

        # 19c (Notes 47149 & 47150): Mandatory deterministic zlib wire codec and build-time shard decompression bounds
        comp_bytes, comp_codec = dbsync.compress_payload(b"armiger-deterministic-shard-payload")
        self.assertEqual(comp_codec, "zlib")
        self.assertEqual(dbsync.decompress_payload(comp_bytes), b"armiger-deterministic-shard-payload")
        rep_rec = dataclasses.replace(
            good_rec,
            infohash="f" * 40,
            file_list=["A" * 10000 for _ in range(15)],
        )
        with self.assertRaises(ValueError) as ctx_shard_bound:
            dbsync.serialize_shard_records([rep_rec], max_output_bytes=32768)
        self.assertIn("shard_exceeds_decompression_bounds", str(ctx_shard_bound.exception))
        batches = dbsync.split_records_into_bounded_shards(
            [dataclasses.replace(good_rec, infohash=f"{i:040x}") for i in range(1, 6)],
            max_output_bytes=600,
        )
        self.assertGreater(len(batches), 1)

        # 19d (Notes 47151, 47167, 47168): Pipeline skips ingest_to_dht_db when unsafe, '[JPN + ENG Subs]' is not dual-audio, and calibration_nll preserves logit_bias
        alg = pipe.AudioLanguageGuard()
        lbl_sub, probs_sub, meta_sub = alg.apply(
            {"name": "[SubsPlease] Anime Title - 01 [1080p][JPN + ENG Subs].mkv"},
            [0.10, 0.85, 0.05],
        )
        self.assertNotEqual(lbl_sub, "multi_or_unknown")
        self.assertNotEqual(meta_sub.get("anime_dub_status"), "dual_audio_or_dub")
        dub_rep = alg.classify_audio_and_dub(
            {"name": "[SubsPlease] Anime Title - 01 [1080p][JPN + ENG Subs].mkv"},
            [0.10, 0.85, 0.05],
            [0.05, 0.95],
            [0.05, 0.95],
        )
        self.assertEqual(dub_rep["anime_dub_status"], "subbed_only")

        p_synth = pipe.ArmigerUltraPipeline(allow_synthetic_fallback=True)
        with unittest.mock.patch.object(p_synth, "ingest_to_dht_db") as spy_ingest_db:
            unsafe_res = p_synth.classify(
                {
                    "infohash": "a" * 40,
                    "name": "Codec_Pack_Installer.scr",
                    "size": 1024,
                    "files": [{"path": "Codec_Pack_Installer.scr", "size": 1024}],
                },
                ingest_to_dht_db=True,
            )
            self.assertFalse(unsafe_res.mondrian_gates["passed_all"])
            self.assertTrue(unsafe_res.plugin_trace["plugin_6_dht_db_sync"]["dht_ingest_status"].startswith("rejected:"))
            self.assertEqual(spy_ingest_db.call_count, 0)

        p_synth.calibrator.heads["junk"] = {"temperature": 1.5, "logit_bias": [0.4, -0.4]}
        sample_rows = [{"name": "Sintel.2010.1080p.BluRay.x264.mkv", "size": 1_000_000_000}]
        nll_implicit = p_synth.calibration_nll(sample_rows, ["clean"], head="junk", temperature=None)
        nll_explicit = p_synth.calibration_nll(sample_rows, ["clean"], head="junk", temperature=1.5)
        self.assertAlmostEqual(nll_implicit, nll_explicit, places=6)

        # 19e (Note 47152): ArmigerDHTUpdater.verify_bundle is non-mutating by default; verify_and_apply_bundle commits seq only after canary passes
        u_seed = bytes([0x44]) * 32
        u_priv = Ed25519PrivateKey.from_private_bytes(u_seed)
        u_pub_hex = u_priv.public_key().public_bytes_raw().hex()
        u_pq_priv, _ = cg.WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(u_seed, seq=0)
        u_pq_root_hex = cg.WOTSPlusSHAKE256.derive_pq_root_pubkey_hex(u_seed)
        upd_stage = cdht.ArmigerDHTUpdater(
            root_pubkey_hex=u_pub_hex,
            current_seq=0,
            min_quorum=5,
            root_pq_pubkey_hex=u_pq_root_hex,
        )
        u_bundle = b'{"test":"staged_commit"}'
        u_m = cdht.SignedDHTManifest(
            seq=1,
            version="3.0.0",
            infohash="1" * 40,
            bundle_sha256=hashlib.sha256(u_bundle).hexdigest(),
            quorum_attestations=5,
            timestamp=now_ts,
            ed25519_signature_hex="",
            pq_pubkey_hex=u_pq_root_hex,
            pq_signature_hex="",
        )
        u_cp = u_m.canonical_payload()
        _, u_pq_sig, _, _ = cg.WOTSPlusSHAKE256.allocate_and_sign(u_pq_priv, u_cp, reserve_fresh=True)
        u_m = dataclasses.replace(
            u_m,
            ed25519_signature_hex=u_priv.sign(u_cp).hex(),
            pq_signature_hex=u_pq_sig.hex(),
        )
        ok_vb, _ = upd_stage.verify_bundle(u_m, u_bundle)
        self.assertTrue(ok_vb)
        self.assertEqual(upd_stage.current_seq, 0)
        rolled_back_flag = []
        ok_fail_canary, reason_fc = upd_stage.verify_and_apply_bundle(
            u_m,
            u_bundle,
            apply_fn=lambda b: True,
            canary_fn=lambda: False,
            rollback_fn=lambda: rolled_back_flag.append(True),
        )
        self.assertFalse(ok_fail_canary)
        self.assertEqual(reason_fc, "canary_regression_rejected")
        self.assertEqual(upd_stage.current_seq, 0)
        self.assertEqual(rolled_back_flag, [True])
        ok_pass_canary, _ = upd_stage.verify_and_apply_bundle(
            u_m,
            u_bundle,
            apply_fn=lambda b: True,
            canary_fn=lambda: True,
        )
        self.assertTrue(ok_pass_canary)
        self.assertEqual(upd_stage.current_seq, 1)

        # 19f (Notes 47153 & 47169): WOTS+ rejects generation-less state, blocks unmigrated directory switch,
        # rejects huge int overflow in aggregate_continuous_deltas, blocks classical KEM downgrade,
        # draws fresh DP noise per release, and prevents unauthenticated Sybil attestation from vetoing honest peer
        with tempfile.TemporaryDirectory() as tmp_wots_r6:
            sd_a = Path(tmp_wots_r6) / "dir_a"
            sd_b = Path(tmp_wots_r6) / "dir_b"
            seal_r6 = Path(tmp_wots_r6) / "seals"
            w_priv_r6, w_pub_r6 = cg.WOTSPlusSHAKE256.generate_keypair(bytes([0x77]) * 64)
            sig_r6_0 = cg.WOTSPlusSHAKE256.sign(w_priv_r6, b"r6-msg-0", state_dir=sd_a, seal_dir=seal_r6)
            self.assertTrue(cg.WOTSPlusSHAKE256.verify(w_pub_r6, b"r6-msg-0", sig_r6_0))
            # Switching state_dir to sd_b without migration must fail closed
            with self.assertRaises(RuntimeError):
                cg.WOTSPlusSHAKE256.sign(w_priv_r6, b"r6-msg-1", state_dir=sd_b, seal_dir=seal_r6)
            # Downgrading state file in sd_a to generation-less JSON must fail closed
            st_file_a = next(sd_a.glob("wots_state_*.json"))
            saved_a_bytes = st_file_a.read_bytes()
            st_file_a.write_text('{"next_leaf_index":0,"used_leaves":{}}', encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx_genless:
                cg.WOTSPlusSHAKE256.sign(w_priv_r6, b"r6-msg-1", state_dir=sd_a, seal_dir=seal_r6)
            self.assertIn("unauthenticated_legacy_wots_state_requires_migration", str(ctx_genless.exception))
            st_file_a.write_bytes(saved_a_bytes)
            # Explicit directory migration to sd_b succeeds and continues monotonically at leaf 1
            cg.WOTSPlusSHAKE256.migrate_state_directory(w_priv_r6, sd_a, sd_b, seal_dir=seal_r6)
            sig_r6_1 = cg.WOTSPlusSHAKE256.sign(w_priv_r6, b"r6-msg-1", state_dir=sd_b, seal_dir=seal_r6)
            self.assertTrue(cg.WOTSPlusSHAKE256.verify(w_pub_r6, b"r6-msg-1", sig_r6_1))
            meta_r6_1 = cg.WOTSPlusSHAKE256.extract_signature_metadata(w_pub_r6, sig_r6_1)
            assert meta_r6_1 is not None
            self.assertEqual(meta_r6_1["leaf_index"], 1)

        # Huge int (10**400) in aggregate_continuous_deltas is quarantined without OverflowError aborting honest batch
        bad_huge_int_sub = dataclasses.replace(
            dedup_subs[-1],
            deltas={"temp": 10**400},
        )
        res_huge_int = agg_dedup.aggregate_continuous_deltas(dedup_subs[3:] + [bad_huge_int_sub])
        self.assertTrue(res_huge_int.success)

        # Classical 32-byte KEM bundle refused when recipient holds 2,432-byte hybrid key unless allow_classical=True
        hyb_priv, hyb_pub = rsi.HybridPQCAEADEngine.generate_keypair()
        classical_bundle = rsi.HybridPQCAEADEngine.encrypt_bundle(b'{"x":1}', hyb_pub[:32], seq=1)
        with self.assertRaises(ValueError) as ctx_pq_downgrade:
            rsi.HybridPQCAEADEngine.decrypt_bundle(classical_bundle, hyb_priv)
        self.assertIn("post-quantum downgrade rejected", str(ctx_pq_downgrade.exception))
        self.assertEqual(
            rsi.HybridPQCAEADEngine.decrypt_bundle(classical_bundle, hyb_priv, allow_classical=True),
            b'{"x":1}',
        )

        # TelemetryAnonymizer draws fresh Laplace noise across repeated observations with equal size/duration
        anon_fresh = rsi.TelemetryAnonymizer(blinding_salt=b"fresh_dp_salt_1234", dp_epsilon=0.5, max_releases=3)
        draws = {anon_fresh.apply_differential_privacy(1024 * 1024 * 1024, 3600.0) for _ in range(3)}
        self.assertGreater(len(draws), 1)
        with self.assertRaises(rsi.AnonymityViolationError) as ctx_dp_budget:
            anon_fresh.apply_differential_privacy(1024 * 1024 * 1024, 3600.0)
        self.assertIn("dp_privacy_budget_exhausted", str(ctx_dp_budget.exception))

        # Forged unauthenticated attestation claiming a wrong ASN for an honest pubkey/subnet before the honest attestation cannot veto quorum
        forged_veto_att = rsi.PeerAttestation(
            peer_id=att_u.peer_id,
            asn=99999,
            peer_pubkey_hex=att_u.peer_pubkey_hex,
            timestamp=now_ts,
            signature_hex="00" * 64,
            ip_address=att_u.ip_address,
        )
        ok_veto, reason_veto, asns_veto = rsi.SybilQuorumVerifier.verify_quorum(
            [forged_veto_att, att_u],
            target_hash=b_sha,
            min_distinct_asns=1,
            require_pq=False,
            require_authoritative_asn=True,
        )
        self.assertTrue(ok_veto, reason_veto)
        self.assertEqual(asns_veto, {13335})

        # Valid ONNX tensor containing b"cos\n" weight bytes passes SafeFormatValidator
        try:
            from onnx import TensorProto, helper
            cos_weights = b"cos\ncsystem\n" + (b"\x01" * 20)
            t_cos = helper.make_tensor("w_cos", TensorProto.UINT8, [32], list(cos_weights))
            g_cos = helper.make_graph(
                [helper.make_node("Add", ["x", "w_cos"], ["y"])],
                "cos_graph",
                [helper.make_tensor_value_info("x", TensorProto.UINT8, [32])],
                [helper.make_tensor_value_info("y", TensorProto.UINT8, [32])],
                [t_cos],
            )
            m_cos = helper.make_model(g_cos, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
            ok_cos, reason_cos = rsi.SafeFormatValidator.validate(m_cos.SerializeToString(), "onnx")
            self.assertTrue(ok_cos, reason_cos)
        except ImportError:
            pass

        # RSIUpdateAggregator rejects out-of-bounds temperature grid (e.g. 0.05 < 0.25 or 20.0 > 10.0) and validates calibration_delta in build_update_payload
        ok_grid_oob, _ = pipe.RSIUpdateAggregator._validate_telemetry_doc_components({
            "calibration_nll_profile": {"grid": [0.05, 1.0, 20.0], "heads": {"category": [1.0, 0.8, 1.2]}},
        })
        self.assertFalse(ok_grid_oob)
        with self.assertRaises(pipe.DHTUpdateRejected):
            ub_agg.build_update_payload(calibration_delta={"heads": {"category": {"temperature": 0.05}}})


if __name__ == "__main__":
    unittest.main()
