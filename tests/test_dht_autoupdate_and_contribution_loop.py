"""End-to-End Unit & Adversarial Fault Tests for Armiger BEP-44/46 Auto-Update (Pull)
and Opt-In Contribution (Push) Runtime Loop.

Covers:
  1. Read-only `/models` (`0444`) -> writable overlay (`ARMIGER_RUNTIME_CACHE_DIR`) hot-swap
     from `seq=3` to `seq=4` over live UDP DHT + TCP peer-wire while concurrent threads call `classify()`.
  2. Rollback & tamper resistance: forged Ed25519 signature, downgrade `seq <= current_seq`,
     corrupt piece SHA-1, mismatched signed Merkle root / file SHA-256, and failed canary gate rollback.
  3. Privacy & opt-out enforcement: `ARMIGER_DHT_SHARE_OPT_IN=false` sends 0 UDP packets;
     `ARMIGER_DHT_SHARE_OPT_IN=true` enforces `k >= 25`, PII/IP/title stripping, `< 1,000`-byte sizing,
     UDP collector ingestion into `inbound_contributions.jsonl`, and replay rejection.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from armiger.armiger_dht_rsi import AnonymityViolationError, ArmigerDHTRSIEngine
from armiger.armiger_ultra_pipeline import ArmigerUltraPipeline
from armiger.oci_dht_node import (
    DEFAULT_RELEASE_PUBKEY_HEX,
    DEFAULT_RELEASE_SALT,
    AnonymityGuard,
    BitTorrentPeerWireSeeder,
    GatedBEP44UDPNode,
    VPNEgressGuard,
    VPNEgressStatus,
    bencode,
    build_torrent_bundle,
)
from armiger.verify_dht import main as verify_dht_cli_main


def _always_verified_vpn_guard() -> VPNEgressGuard:
    return VPNEgressGuard(
        expected_iface="tun0",
        check_fn=lambda: VPNEgressStatus(
            verified=True,
            interface="tun0",
            interface_present=True,
            interface_up=True,
            has_tunnel_route=True,
            has_default_or_split_route=True,
            checked_at=time.time(),
            reason="ok",
        ),
    )


def _make_model_files(seq_val: int, *, cat_temp: float = 0.7854, force_canary_fail: bool = False) -> Dict[str, bytes]:
    cal_doc: Dict[str, Any] = {
        "schema": "armiger.calibration.v2",
        "heads": {
            "category": {"temperature": cat_temp},
            "adult": {"temperature": 0.91},
            "junk": {"temperature": 0.6251},
            "anime": {"temperature": 0.88},
            "english_subs": {"temperature": 0.95},
            "lang": {"temperature": 0.85},
            "owner_scope": {"temperature": 0.82},
            "title_language": {"temperature": 0.89},
        },
        "mondrian_thresholds": {
            "adult": 0.5451,
            "junk": 0.08 if seq_val >= 4 else 0.1010,
            "owner_scope": 0.4770,
        },
    }
    if force_canary_fail:
        cal_doc["force_canary_failure"] = True
    cfg_doc: Dict[str, Any] = {
        "version": f"v3.0-r13.{seq_val - 1}",
        "dht_seq": seq_val,
        "mondrian_thresholds": cal_doc["mondrian_thresholds"],
    }
    vocab_doc = {"<pad>": 0, "<unk>": 1, "ubuntu": 2, "1080p": 3, "mkv": 4}
    return {
        "calibration.json": json.dumps(cal_doc, sort_keys=True).encode("utf-8"),
        "config.json": json.dumps(cfg_doc, sort_keys=True).encode("utf-8"),
        "vocab.json": json.dumps(vocab_doc, sort_keys=True).encode("utf-8"),
    }


class _LiveSeederHarness:
    """Runs an in-process GatedBEP44UDPNode + BitTorrentPeerWireSeeder on an asyncio loop thread."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.vpn_guard = _always_verified_vpn_guard()
        self.privkey_32 = os.urandom(32)
        sk = Ed25519PrivateKey.from_private_bytes(self.privkey_32)
        self.pubkey_32 = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.pubkey_hex = self.pubkey_32.hex()
        self.dht_node = GatedBEP44UDPNode(
            vpn_guard=self.vpn_guard,
            host="127.0.0.1",
            port=0,
            anonymity_guard=AnonymityGuard(opt_in_sharing=True),
            contributions_jsonl_path=self.state_dir / "inbound_contributions.jsonl",
        )
        self.bt_seeder = BitTorrentPeerWireSeeder(
            vpn_guard=self.vpn_guard,
            bundles_by_infohash={},
            host="127.0.0.1",
            port=0,
        )
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self.dht_port = 0
        self.bt_port = 0

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> Tuple[int, int]:
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._start_coro(), self.loop)
        self.dht_port, self.bt_port = fut.result(timeout=5.0)
        return self.dht_port, self.bt_port

    async def _start_coro(self) -> Tuple[int, int]:
        _, dp = await self.dht_node.start()
        _, bp = await self.bt_seeder.start()
        return dp, bp

    def publish_bundle(
        self,
        files_map: Dict[str, bytes],
        *,
        seq: int,
        salt: bytes = b"armiger:weights:ultra",
        override_merkle_hex: str | None = None,
        forged_privkey_32: bytes | None = None,
        corrupt_first_piece: bool = False,
    ) -> Any:
        def _do() -> Any:
            bundle = build_torrent_bundle(
                bundle_name="armiger-weights-ultra-v3.10.0",
                files_map=files_map,
                pubkey_32=self.pubkey_32,
                salt=salt,
                seq=seq,
            )
            if corrupt_first_piece and bundle.payload_bytes:
                bad_payload = bytes([bundle.payload_bytes[0] ^ 0xFF]) + bundle.payload_bytes[1:]
                object.__setattr__(bundle, "payload_bytes", bad_payload)
            self.bt_seeder.bundles_by_infohash[bundle.infohash_20] = bundle
            self.dht_node._peers[bundle.infohash_20] = [("127.0.0.1", self.bt_port, time.time())]
            mr_hex = override_merkle_hex or bundle.merkle_root_hex
            v_payload = bencode(
                {
                    b"bundle": b"armiger-weights-ultra-v3.10.0",
                    b"ih": bundle.infohash_20,
                    b"merkle_root": mr_hex.encode("ascii"),
                    b"seq": int(seq),
                    b"tier": b"oci",
                }
            )
            if forged_privkey_32 is not None:
                from oci_vpn_guard import _sign_bep44, bep44_signing_buffer, compute_bep44_target
                sig_buf = bep44_signing_buffer(v=v_payload, seq=seq, salt=salt)
                _, bad_sig = _sign_bep44(forged_privkey_32, sig_buf)
                target = compute_bep44_target(self.pubkey_32, salt)
                self.dht_node._store[target] = {
                    "v": v_payload,
                    "seq": int(seq),
                    "sig": bad_sig,
                    "pubkey": self.pubkey_32,
                    "salt": salt,
                    "ts": time.time(),
                    "pinned": True,
                }
            else:
                self.dht_node.publish_local_mutable(
                    privkey_32=self.privkey_32,
                    salt=salt,
                    v=v_payload,
                    seq=seq,
                )
            return bundle

        fut = self.loop.call_soon_threadsafe
        res_q: List[Any] = []
        ev = threading.Event()

        def _wrap() -> None:
            try:
                res_q.append((True, _do()))
            except Exception as exc:
                res_q.append((False, exc))
            finally:
                ev.set()

        fut(_wrap)
        ev.wait(timeout=5.0)
        ok, val = res_q[0]
        if not ok:
            raise val
        return val

    def stop(self) -> None:
        async def _stop_coro() -> None:
            self.bt_seeder.close()
            self.dht_node.close()

        try:
            fut = asyncio.run_coroutine_threadsafe(_stop_coro(), self.loop)
            fut.result(timeout=3.0)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=2.0)


class TestDHTAutoUpdateAndContributionLoop(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.ro_models_dir = self.tmp_path / "ro_models"
        self.ro_models_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir = self.tmp_path / "writable_overlay"
        self.state_dir = self.tmp_path / "seeder_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        for fname, fbytes in _make_model_files(3, cat_temp=0.7854).items():
            p = self.ro_models_dir / fname
            p.write_bytes(fbytes)
            os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # 0444 read-only files
        os.chmod(self.ro_models_dir, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)  # 0550 read-only dir

    def tearDown(self) -> None:
        try:
            os.chmod(self.ro_models_dir, stat.S_IRWXU)
            for p in self.ro_models_dir.iterdir():
                os.chmod(p, stat.S_IRWXU)
        except OSError:
            pass
        self._tmp.cleanup()

    def test_canonical_release_key_and_verify_dht_defaults(self) -> None:
        self.assertEqual(
            DEFAULT_RELEASE_PUBKEY_HEX,
            "3599362bd52dad8b89398b904d3081c2cb483d13376f7ffb49f3748208731e76",
        )
        self.assertEqual(DEFAULT_RELEASE_SALT, "armiger:weights:ultra")
        # Verify CLI parser requires zero flags when invoked with --dht-node none
        rc = verify_dht_cli_main(["--dht-node", "none", "--timeout", "0.2"])
        self.assertEqual(rc, 1)

    def test_readonly_models_overlay_hotswap_under_concurrent_classify(self) -> None:
        harness = _LiveSeederHarness(self.state_dir)
        dht_port, _bt_port = harness.start()
        try:
            v4_files = _make_model_files(4, cat_temp=0.6915)
            bundle_v4 = harness.publish_bundle(v4_files, seq=4, salt=b"armiger:weights:ultra")

            pipeline = ArmigerUltraPipeline(
                model_dir=self.ro_models_dir,
                variant="ultra",
                allow_synthetic_fallback=True,
                runtime_cache_dir=self.overlay_dir,
                auto_update=False,
                initial_seq=3,
            )
            try:
                self.assertEqual(pipeline.active_seq, 3)
                self.assertEqual(pipeline.model_source, "baked_image")

                classify_errors: List[Exception] = []
                stop_workers = threading.Event()
                calls_completed = [0]

                def _worker() -> None:
                    while not stop_workers.is_set():
                        try:
                            pred = pipeline.classify(
                                {"name": "Ubuntu.24.04.Desktop.amd64.iso", "size": 4800000000},
                                use_zero_cue_catalog=False,
                            )
                            if not pred.category:
                                raise RuntimeError("empty_category")
                            calls_completed[0] += 1
                        except Exception as exc:
                            classify_errors.append(exc)

                threads = [threading.Thread(target=_worker, daemon=True) for _ in range(4)]
                for t in threads:
                    t.start()

                res = pipeline.check_and_apply_dht_release_update(
                    release_pubkey_hex=harness.pubkey_hex,
                    salt=b"armiger:weights:ultra",
                    dht_nodes=[("127.0.0.1", dht_port)],
                    allow_webseed_fallback=False,
                    timeout=2.0,
                )
                stop_workers.set()
                for t in threads:
                    t.join(timeout=2.0)

                self.assertEqual(classify_errors, [])
                self.assertGreater(calls_completed[0], 0)
                self.assertTrue(res["updated"], f"Expected update to succeed, got: {res}")
                self.assertEqual(res["seq_before"], 3)
                self.assertEqual(res["active_seq"], 4)
                self.assertEqual(pipeline.active_seq, 4)
                self.assertEqual(pipeline.model_source, "dht_live_overlay")
                self.assertEqual(pipeline.active_bundle_merkle_root, bundle_v4.merkle_root_hex)
                self.assertAlmostEqual(pipeline.calibrator.heads["category"]["temperature"], 0.6915, places=4)
                self.assertGreater(harness.bt_seeder.pieces_served, 0)

                hst = pipeline.health_status()
                self.assertEqual(hst["active_seq"], 4)
                self.assertEqual(hst["active_bundle_merkle_root"], bundle_v4.merkle_root_hex)
                self.assertEqual(hst["model_source"], "dht_live_overlay")
                self.assertIsNotNone(hst["last_dht_check_ts"])
                self.assertIn("sharing_status", hst)
            finally:
                pipeline.close()
        finally:
            harness.stop()

    def test_rollback_and_tamper_resistance(self) -> None:
        harness = _LiveSeederHarness(self.state_dir)
        dht_port, _ = harness.start()
        try:
            pipeline = ArmigerUltraPipeline(
                model_dir=self.ro_models_dir,
                variant="ultra",
                allow_synthetic_fallback=True,
                runtime_cache_dir=self.overlay_dir,
                auto_update=False,
                initial_seq=3,
            )
            try:
                fp_initial = pipeline.dht_state_fingerprint()

                # 1. Downgrade / same seq (seq=3 <= active_seq=3) -> rejected without updating
                harness.publish_bundle(_make_model_files(3), seq=3)
                res_downgrade = pipeline.check_and_apply_dht_release_update(
                    release_pubkey_hex=harness.pubkey_hex,
                    salt=b"armiger:weights:ultra",
                    dht_nodes=[("127.0.0.1", dht_port)],
                    allow_webseed_fallback=False,
                    timeout=1.5,
                )
                self.assertFalse(res_downgrade["updated"])
                self.assertIn("bep44_seq_below_minimum", res_downgrade["reason"])
                self.assertEqual(pipeline.active_seq, 3)

                # 2. Forged Ed25519 signature at seq=4 -> rejected
                harness.publish_bundle(
                    _make_model_files(4),
                    seq=4,
                    forged_privkey_32=os.urandom(32),
                )
                res_forged = pipeline.check_and_apply_dht_release_update(
                    release_pubkey_hex=harness.pubkey_hex,
                    salt=b"armiger:weights:ultra",
                    dht_nodes=[("127.0.0.1", dht_port)],
                    allow_webseed_fallback=False,
                    timeout=1.5,
                )
                self.assertFalse(res_forged["updated"])
                self.assertEqual(pipeline.active_seq, 3)

                # 3. Corrupt piece SHA-1 on TCP peer-wire at seq=5 -> rejected
                harness.publish_bundle(
                    _make_model_files(5),
                    seq=5,
                    corrupt_first_piece=True,
                )
                res_corrupt = pipeline.check_and_apply_dht_release_update(
                    release_pubkey_hex=harness.pubkey_hex,
                    salt=b"armiger:weights:ultra",
                    dht_nodes=[("127.0.0.1", dht_port)],
                    allow_webseed_fallback=False,
                    timeout=1.5,
                )
                self.assertFalse(res_corrupt["updated"])
                self.assertEqual(pipeline.active_seq, 3)

                # 4. Mismatched signed Merkle root at seq=6 -> rejected
                harness.publish_bundle(
                    _make_model_files(6),
                    seq=6,
                    override_merkle_hex="f" * 64,
                )
                res_bad_mr = pipeline.check_and_apply_dht_release_update(
                    release_pubkey_hex=harness.pubkey_hex,
                    salt=b"armiger:weights:ultra",
                    dht_nodes=[("127.0.0.1", dht_port)],
                    allow_webseed_fallback=False,
                    timeout=1.5,
                )
                self.assertFalse(res_bad_mr["updated"])
                self.assertEqual(pipeline.active_seq, 3)

                # 5. Failed canary check at seq=7 -> rolled back cleanly, fingerprint untouched
                harness.publish_bundle(
                    _make_model_files(7, cat_temp=0.55, force_canary_fail=True),
                    seq=7,
                )
                res_canary_fail = pipeline.check_and_apply_dht_release_update(
                    release_pubkey_hex=harness.pubkey_hex,
                    salt=b"armiger:weights:ultra",
                    dht_nodes=[("127.0.0.1", dht_port)],
                    allow_webseed_fallback=False,
                    timeout=1.5,
                )
                self.assertFalse(res_canary_fail["updated"])
                self.assertTrue(res_canary_fail["rolled_back"])
                self.assertEqual(pipeline.active_seq, 3)
                self.assertEqual(pipeline.model_source, "baked_image")
                self.assertEqual(pipeline.dht_state_fingerprint(), fp_initial)
            finally:
                pipeline.close()
        finally:
            harness.stop()

    def test_privacy_opt_out_and_opt_in_udp_contribution_loop(self) -> None:
        harness = _LiveSeederHarness(self.state_dir)
        dht_port, _ = harness.start()
        try:
            harness.publish_bundle(_make_model_files(4), seq=4, salt=b"armiger:weights:ultra")

            # 1. Opt-out by default (ARMIGER_DHT_SHARE_OPT_IN=false) -> 0 UDP packets sent
            pipeline_opt_out = ArmigerUltraPipeline(
                model_dir=self.ro_models_dir,
                variant="ultra",
                allow_synthetic_fallback=True,
                runtime_cache_dir=self.overlay_dir,
                auto_update=False,
                opt_in_sharing=False,
                rsi_mode=None,
            )
            try:
                self.assertFalse(pipeline_opt_out.opt_in_sharing)
                self.assertEqual(pipeline_opt_out.rsi_engine.mode.value, "pull_only")
                res_blocked = pipeline_opt_out.dispatch_opt_in_contribution(
                    dht_nodes=[("127.0.0.1", dht_port)],
                )
                self.assertFalse(res_blocked["dispatched"])
                self.assertEqual(res_blocked["packets_sent"], 0)
                self.assertEqual(pipeline_opt_out.sharing_status()["rsi_engine"]["outbound_udp_packets_sent"], 0)
                self.assertEqual(harness.dht_node.inbound_contributions_accepted, 0)
            finally:
                pipeline_opt_out.close()

            # 2. Opt-in enabled (opt_in_sharing=True) -> k>=25, PII firewall, <1000B, UDP collector ingestion, replay rejection
            pipeline_opt_in = ArmigerUltraPipeline(
                model_dir=self.ro_models_dir,
                variant="ultra",
                allow_synthetic_fallback=True,
                runtime_cache_dir=self.overlay_dir,
                auto_update=False,
                opt_in_sharing=True,
                rsi_mode=None,
            )
            try:
                self.assertTrue(pipeline_opt_in.opt_in_sharing)
                self.assertEqual(pipeline_opt_in.rsi_engine.mode.value, "anonymous_contribute")

                # Enforce k >= 25 anonymity floor
                with self.assertRaises(AnonymityViolationError):
                    pipeline_opt_in.export_signed_delta_bundle(n_samples=10)

                # Enforce PII / IP / raw title stripping firewall
                with self.assertRaises(AnonymityViolationError):
                    pipeline_opt_in.export_signed_delta_bundle(
                        raw_wire_override={"v": 2, "raw_title": "Secret.Movie.2026.mkv", "ip": "198.51.100.50"}
                    )

                signed_delta = pipeline_opt_in.export_signed_delta_bundle(
                    delta_w=[0.02, -0.01, 0.015, 0.0, -0.005, 0.01, -0.02, 0.005],
                    delta_b=[0.01, -0.01],
                    n_samples=32,
                    seq=10,
                )
                self.assertIsNotNone(signed_delta)
                assert signed_delta is not None
                self.assertLess(len(signed_delta["v"]), 1000)

                # Dispatch over UDP to DHT + collector node discovered via release pointer
                res_push = pipeline_opt_in.dispatch_opt_in_contribution(
                    signed_delta,
                    dht_nodes=[("127.0.0.1", dht_port)],
                    release_pubkey_hex=harness.pubkey_hex,
                    release_salt=b"armiger:weights:ultra",
                    discover_collectors=True,
                )
                self.assertTrue(res_push["dispatched"], f"Expected UDP dispatch to succeed: {res_push}")
                self.assertGreaterEqual(res_push["collector_puts_ok"] + res_push["mainline_puts_ok"], 1)
                self.assertEqual(harness.dht_node.inbound_contributions_accepted, 1)
                self.assertEqual(harness.dht_node.inbound_contributions_rejected, 0)

                jsonl_path = self.state_dir / "inbound_contributions.jsonl"
                self.assertTrue(jsonl_path.is_file())
                lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(lines), 1)
                self.assertEqual(lines[0]["seq"], 10)
                self.assertEqual(lines[0]["delta"]["n_bucket"], 25)
                self.assertNotIn("ip", lines[0])

                # Replay of identical seq=10 bundle must be rejected by collector
                res_replay = pipeline_opt_in.dispatch_opt_in_contribution(
                    signed_delta,
                    dht_nodes=[("127.0.0.1", dht_port)],
                    discover_collectors=False,
                )
                self.assertFalse(res_replay["dispatched"])
                self.assertEqual(harness.dht_node.inbound_contributions_accepted, 1)
                self.assertGreaterEqual(harness.dht_node.inbound_contributions_rejected, 1)

                # Transient journal persistence failure must not poison replay state on retry
                signed_delta_11 = pipeline_opt_in.export_signed_delta_bundle(
                    delta_w=[0.01, -0.02, 0.01, 0.0, -0.01, 0.02, -0.01, 0.01],
                    delta_b=[0.005, -0.005],
                    n_samples=30,
                    seq=11,
                )
                assert signed_delta_11 is not None
                orig_persist = harness.dht_node._persist_accepted_contribution
                harness.dht_node._persist_accepted_contribution = lambda *a, **kw: False
                try:
                    res_fail_persist = pipeline_opt_in.dispatch_opt_in_contribution(
                        signed_delta_11,
                        dht_nodes=[("127.0.0.1", dht_port)],
                        discover_collectors=False,
                    )
                    self.assertFalse(res_fail_persist["dispatched"])
                finally:
                    harness.dht_node._persist_accepted_contribution = orig_persist
                res_retry_persist = pipeline_opt_in.dispatch_opt_in_contribution(
                    signed_delta_11,
                    dht_nodes=[("127.0.0.1", dht_port)],
                    discover_collectors=False,
                )
                self.assertTrue(res_retry_persist["dispatched"])
                self.assertEqual(harness.dht_node.inbound_contributions_accepted, 2)
            finally:
                pipeline_opt_in.close()
        finally:
            harness.stop()


if __name__ == "__main__":
    unittest.main()
