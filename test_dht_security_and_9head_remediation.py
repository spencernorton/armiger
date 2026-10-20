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

ROOT = Path(__file__).resolve().parent
for anc in [ROOT, *ROOT.parents[:3]]:
    for sub in (anc, anc / "armiger", anc / "deploy" / "oci-dht"):
        if sub.is_dir() and str(sub) not in sys.path:
            sys.path.insert(0, str(sub))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import armiger_dht_rsi as rsi
import armiger_ultra_pipeline as pipe
import calibration_and_dht as cdht
import csam_guard as cg
from audio_language_guard import classify_audio_and_anime_dub_v3
from armiger_inference import ArmigerStandaloneONNX
from zero_cue_plugin import hash_catalog_key


class TestDHTSecurityAnd9HeadRemediation(unittest.TestCase):
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
        res = clf.predict({"name": "[Judas] Kusuriya no Hitorigoto (Season 1) [1080p][HEVC x265 10bit][Dual-Audio][Multi-Subs]"})
        self.assertIn("span_bio", res)
        self.assertEqual(len(res["heads"]["span_bio"]), 256)

        aud = classify_audio_and_anime_dub_v3(
            "[Judas] Kusuriya no Hitorigoto (Season 1) [1080p][HEVC x265 10bit][Dual-Audio][Multi-Subs]",
            p_en=0.95,
            p_anime=0.98,
        )
        self.assertTrue(aud["is_anime_dub_or_dual"])
        pipe._validate_calibration_delta({"heads": {"span_bio": {"temperature": 1.0}}})


if __name__ == "__main__":
    unittest.main()
