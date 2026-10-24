"""Protocol/canonicalization regressions for untrusted DHT packets, BEP-9/10/19/42/44/46, and DSSE provenance."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.request

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OCI = ROOT / "deploy" / "oci-dht"
for p_dir in (SRC, OCI, ROOT):
    if p_dir.exists() and str(p_dir) not in sys.path:
        sys.path.insert(0, str(p_dir))

try:
    from armiger.oci_vpn_guard import (  # noqa: E402
        AnonymityGuard,
        _sign_bep44,
        bdecode,
        bdecode_krpc,
        bencode,
        bep44_signing_buffer,
        compute_bep44_target,
        generate_bep42_node_id,
        verify_bep42_node_id,
    )
    from armiger.oci_dht_seeder import BitTorrentPeerWireSeeder, GatedBEP44UDPNode  # noqa: E402
    from armiger.oci_sqlite_catalog import (  # noqa: E402
        BT_EXT_HANDSHAKE_ID,
        BT_MSG_EXTENDED,
        BT_PROTOCOL_HEADER,
        BT_RESERVED_BEP10,
        BT_UT_METADATA_ID,
        _CatalogJSONStream,
        build_sqlite_zero_cue_catalog,
        build_torrent_bundle,
        fetch_bundle_via_bep19_webseed,
        fetch_bundle_via_peer_wire,
        verify_info_and_payload,
        verify_torrent_payload,
    )
except ImportError:
    from oci_vpn_guard import (  # noqa: E402
        AnonymityGuard,
        _sign_bep44,
        bdecode,
        bdecode_krpc,
        bencode,
        bep44_signing_buffer,
        compute_bep44_target,
        generate_bep42_node_id,
        verify_bep42_node_id,
    )
    from oci_dht_seeder import BitTorrentPeerWireSeeder, GatedBEP44UDPNode  # noqa: E402
    from oci_sqlite_catalog import (  # noqa: E402
        BT_EXT_HANDSHAKE_ID,
        BT_MSG_EXTENDED,
        BT_PROTOCOL_HEADER,
        BT_RESERVED_BEP10,
        BT_UT_METADATA_ID,
        _CatalogJSONStream,
        build_sqlite_zero_cue_catalog,
        build_torrent_bundle,
        fetch_bundle_via_bep19_webseed,
        fetch_bundle_via_peer_wire,
        verify_info_and_payload,
        verify_torrent_payload,
    )
from armiger.armiger_dht_rsi import (  # noqa: E402
    AcyclicProvenanceVerifier,
    AcyclicReleaseEnvelope,
    AuthenticatedConformalBaseline,
    AuthoritativeASNVerifier,
    DSSE_TYPE_ATTESTATION_V2,
    ThresholdDHTManifest,
    ThresholdSignatureVerifier,
    WOTSPlusSHAKE256,
    dsse_pae,
)
from armiger.verify_dht import main as verify_dht_main, verify_dht_release  # noqa: E402


class LocalTestVPN:
    def register_on_drop(self, cb):
        pass

    def require_for_packet(self):
        return True

    def assert_egress_or_raise(self):
        return None


class TestSession9IngressAndBEP46Hardening(unittest.TestCase):
    def test_untrusted_bencode_is_canonical_and_bounded(self) -> None:
        bad_inputs = [
            b"01:x",
            b"d1:ai1e1:ai2ee",
            b"l" * 2000 + b"e" * 2000,
            b"i-0e",
            b"1:xtrailing",
        ]
        for bad in bad_inputs:
            with self.subTest(bad=bad[:24]):
                with self.assertRaises(ValueError):
                    bdecode(bad)
        with self.assertRaises(ValueError):
            bdecode_krpc(b"4000:" + b"a" * 4000, max_bytes=1024)
        with self.assertRaises(ValueError):
            bdecode_krpc(b"l" * 16 + b"e" * 16, max_depth=8)

    def test_invalid_catalog_preserves_existing_database(self) -> None:
        bad_payloads = [
            '{"title_only_map":{"0123456789abcdef":"movie"}',
            '{"title_only_map":{"0123456789abcdef":"movie\u0027);DROP TABLE zero_cue_entries;--"}}',
            '{"title_only_map":{"0123456789abcdef":"movie","0123456789abcdef":"course"}}',
        ]
        for bad in bad_payloads:
            with self.subTest(bad=bad[:32]), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                src, dst = tmp_path / "c.json", tmp_path / "c.sqlite"
                src.write_text(json.dumps({"title_only_map": {"0123456789abcdef": "movie"}}), encoding="utf-8")
                build_sqlite_zero_cue_catalog(src, dst)
                original = dst.read_bytes()
                src.write_text(bad, encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_sqlite_zero_cue_catalog(src, dst)
                self.assertEqual(dst.read_bytes(), original)

    def test_catalog_json_stream_split_numeric_token_across_chunk_boundaries(self) -> None:
        """Regression test for Jeeves !38 finding: numeric tokens split at chunk boundary must not truncate."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pad = " " * 46
            doc = f'{{"version":{pad}123456789,"title_only_map":{{"0123456789abcdef":"movie"}}}}'
            src = tmp_path / "split_num.json"
            dst = tmp_path / "split_num.sqlite"
            src.write_text(doc, encoding="utf-8")
            with src.open("rb") as fh:
                stream = _CatalogJSONStream(fh, chunk_size=64)
                stream.expect("{")
                k1 = stream.value()
                self.assertEqual(k1, "version")
                stream.expect(":")
                v1 = stream.value()
                self.assertEqual(v1, 123456789)
            built = build_sqlite_zero_cue_catalog(src, dst)
            self.assertTrue(dst.exists())
            self.assertTrue(isinstance(built, dict))

            # Fractional ("12." / "5") and exponent ("1e" / "+3") boundary splits
            for token_str, expected_val in (("12.5", 12.5), ("1e+3", 1000.0)):
                for split_idx in range(1, len(token_str)):
                    prefix = '{"version":'
                    pad_len = 64 - len(prefix) - split_idx
                    p_str = " " * max(1, pad_len)
                    cs = len(prefix) + len(p_str) + split_idx
                    f_doc = f'{prefix}{p_str}{token_str},"title_only_map":{{"0123456789abcdef":"movie"}}}}'
                    f_src = tmp_path / f"split_{token_str}_{split_idx}.json"
                    f_src.write_text(f_doc, encoding="utf-8")
                    with f_src.open("rb") as fh:
                        st = _CatalogJSONStream(fh, chunk_size=cs)
                        st.expect("{")
                        self.assertEqual(st.value(), "version")
                        st.expect(":")
                        self.assertEqual(st.value(), expected_val)

            bad_doc = f'{{"version":{pad}12abc,"title_only_map":{{"0123456789abcdef":"movie"}}}}'
            bad_src = tmp_path / "bad_split_num.json"
            bad_src.write_text(bad_doc, encoding="utf-8")
            with self.assertRaises(ValueError):
                build_sqlite_zero_cue_catalog(bad_src, tmp_path / "bad.sqlite")

    def test_payload_requires_original_piece_hashes(self) -> None:
        pk = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        bundle = build_torrent_bundle("synthetic", {"x.bin": b"abc" * 500}, pk, b"armiger:head:delta")
        self.assertEqual(verify_torrent_payload(bundle.torrent_bytes, bundle.payload_bytes)["bytes_verified"], 1500)
        self.assertIn("s=" + b"armiger:head:delta".hex(), bundle.magnet_uri)
        with self.assertRaisesRegex(ValueError, "piece_hash"):
            verify_torrent_payload(bundle.torrent_bytes, b"X" + bundle.payload_bytes[1:])

    def test_bep42_node_id_generation_and_verification(self) -> None:
        public_ip = "198.51.100.42"
        nid = generate_bep42_node_id(public_ip, rand_byte=5)
        self.assertEqual(len(nid), 20)
        self.assertTrue(verify_bep42_node_id(public_ip, nid, allow_local=False))
        forged = bytes([nid[0] ^ 0xFF]) + nid[1:]
        self.assertFalse(verify_bep42_node_id(public_ip, forged, allow_local=False))
        self.assertTrue(verify_bep42_node_id("127.0.0.1", forged, allow_local=True))
        self.assertFalse(verify_bep42_node_id("not-an-ip", b"X" * 20))
        self.assertFalse(verify_bep42_node_id("2001:db8::1", b"X" * 20))

    def test_mutable_put_rejects_out_of_range_signed_sequence_and_verifies_bep46_dict(self) -> None:
        for seq in (-1, 2**63):
            with self.subTest(seq=seq):
                async def run_bad(s: int) -> None:
                    node = GatedBEP44UDPNode(LocalTestVPN(), anonymity_guard=AnonymityGuard(True), require_write_token=False)
                    replies = []
                    node._send = lambda payload, addr: replies.append(payload)
                    sk = Ed25519PrivateKey.generate()
                    pk, salt, value = sk.public_key().public_bytes_raw(), b"armiger:head:delta", b"payload"
                    args = {
                        b"k": pk,
                        b"v": value,
                        b"salt": salt,
                        b"seq": s,
                        b"sig": sk.sign(bep44_signing_buffer(value, s, salt)),
                    }
                    await node._handle_put(b"s9", args, ("127.0.0.1", 12345))
                    self.assertEqual(replies[-1][b"y"], b"e")
                    self.assertFalse(node._store)

                asyncio.run(run_bad(seq))

        async def run_bep46() -> None:
            node = GatedBEP44UDPNode(LocalTestVPN(), anonymity_guard=AnonymityGuard(True), require_write_token=False)
            replies = []
            node._send = lambda payload, addr: replies.append(payload)
            sk = Ed25519PrivateKey.generate()
            pk = sk.public_key().public_bytes_raw()
            salt = b"armiger-oci-v1-int8"
            ih_20 = hashlib.sha1(b"test-info-dict").digest()
            v_dict = {b"ih": ih_20, b"seq": 10, b"tier": b"int8", b"ts": 1761264000, b"v": 2}
            _, sig = _sign_bep44(sk.private_bytes_raw(), bep44_signing_buffer(v_dict, 10, salt))
            args = {b"k": pk, b"v": v_dict, b"salt": salt, b"seq": 10, b"sig": sig}
            await node._handle_put(b"p1", args, ("127.0.0.1", 12345))
            self.assertEqual(replies[-1][b"y"], b"r")

            await node._handle_put(b"p2", args, ("127.0.0.1", 12345))
            self.assertEqual(replies[-1][b"y"], b"r")

            v_dict_conflict = {b"ih": hashlib.sha1(b"other").digest(), b"seq": 10, b"tier": b"int8", b"ts": 1761264000, b"v": 2}
            _, sig_conflict = _sign_bep44(sk.private_bytes_raw(), bep44_signing_buffer(v_dict_conflict, 10, salt))
            args_conflict = {b"k": pk, b"v": v_dict_conflict, b"salt": salt, b"seq": 10, b"sig": sig_conflict}
            await node._handle_put(b"p3", args_conflict, ("127.0.0.1", 12345))
            self.assertEqual(replies[-1][b"y"], b"e")
            self.assertEqual(replies[-1][b"e"], 302)

            bad_sig_args = {**args, b"seq": 11}
            await node._handle_put(b"p4", bad_sig_args, ("127.0.0.1", 12345))
            self.assertEqual(replies[-1][b"y"], b"e")
            self.assertEqual(replies[-1][b"e"], 206)

        asyncio.run(run_bep46())

    def test_bep9_bep10_ut_metadata_and_multi_hop_get_peers_verify_dht(self) -> None:
        """Test BEP-9/10 metadata bootstrap + 3-node multi-hop get_peers referral walk + oversized frame rejection."""
        sk = Ed25519PrivateKey.generate()
        priv_32 = sk.private_bytes_raw()
        pub_32 = sk.public_key().public_bytes_raw()
        salt = b"armiger-oci-v1-int8"
        files = {
            "weights.json": json.dumps({"tier": "int8", "weights": [1, 2, 3, 4]}).encode("utf-8"),
            "catalog.bin": b"0123456789abcdef" * 1200,
        }
        bundle = build_torrent_bundle("int8", files, pub_32, salt, seq=105)
        vpn = LocalTestVPN()

        async def _run_peer_wire_test() -> None:
            seeder = BitTorrentPeerWireSeeder(
                vpn,
                {bundle.infohash_20: bundle},
                host="127.0.0.1",
                port=0,
            )
            _, bt_port = await seeder.start()
            # 3 separate DHT nodes:
            # dht_mutable holds ONLY the BEP-44 mutable record and refers to dht_referral;
            # dht_referral holds no peers and refers to dht_swarm (closest to infohash_20);
            # dht_swarm holds the announced BitTorrent peer for infohash_20.
            dht_mutable = GatedBEP44UDPNode(vpn, anonymity_guard=AnonymityGuard(True), host="127.0.0.1", port=0)
            dht_referral = GatedBEP44UDPNode(vpn, anonymity_guard=AnonymityGuard(True), host="127.0.0.1", port=0)
            dht_swarm = GatedBEP44UDPNode(
                vpn,
                anonymity_guard=AnonymityGuard(True),
                host="127.0.0.1",
                port=0,
                node_id=bundle.infohash_20,
            )
            _, port_mut = await dht_mutable.start()
            _, port_ref = await dht_referral.start()
            _, port_swm = await dht_swarm.start()
            try:
                v_dict = {
                    b"ih": bundle.infohash_20,
                    b"mr": bytes.fromhex(bundle.merkle_root_hex),
                    b"seq": 105,
                    b"tier": b"int8",
                    b"ts": 1761264000,
                    b"v": 2,
                }
                dht_mutable.publish_local_mutable(
                    privkey_32=priv_32,
                    salt=salt,
                    v=v_dict,
                    seq=105,
                )
                dht_mutable.add_routing_node(dht_referral.node_id, "127.0.0.1", port_ref)
                dht_referral.add_routing_node(dht_swarm.node_id, "127.0.0.1", port_swm)
                dht_swarm._peers[bundle.infohash_20] = [("127.0.0.1", bt_port, time.time())]

                fetched = await fetch_bundle_via_peer_wire(
                    host="127.0.0.1",
                    port=bt_port,
                    infohash_20=bundle.infohash_20,
                    timeout=3.0,
                )
                self.assertEqual(fetched["files"], files)
                self.assertEqual(fetched["merkle_root_sha256"], bundle.merkle_root_hex)

                # Multi-hop verifier starting ONLY at dht_mutable must follow referrals -> dht_referral -> dht_swarm -> seeder
                receipt_p2p = await asyncio.to_thread(
                    verify_dht_release,
                    pubkey_hex=pub_32.hex(),
                    salt=salt,
                    dht_nodes=[("127.0.0.1", port_mut)],
                    mode="strict-p2p",
                    min_seq=100,
                    timeout=3.0,
                )
                self.assertTrue(receipt_p2p["verified"], msg=str(receipt_p2p))
                self.assertEqual(receipt_p2p["mode_used"], "strict-p2p")
                self.assertEqual(receipt_p2p["seq"], 105)

                small_bundle = build_torrent_bundle("int8", {"tiny.json": b"{}"}, pub_32, salt)

                # Oversized peer-wire piece frame must be rejected before reading body
                async def _malicious_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    try:
                        await reader.readexactly(68)
                        writer.write(BT_PROTOCOL_HEADER + BT_RESERVED_BEP10 + small_bundle.infohash_20 + b"-MALPEER-12345678901")
                        ext_hs = bencode({b"m": {b"ut_metadata": BT_UT_METADATA_ID}, b"metadata_size": len(small_bundle.info_bytes)})
                        writer.write(struct.pack("!IBB", 2 + len(ext_hs), BT_MSG_EXTENDED, BT_EXT_HANDSHAKE_ID) + ext_hs)
                        await writer.drain()
                        # Read client ext handshake + ut_metadata request
                        for _ in range(2):
                            h = await reader.readexactly(4)
                            (ml,) = struct.unpack("!I", h)
                            await reader.readexactly(ml)
                        # Send UNCHOKE + single metadata piece
                        writer.write(struct.pack("!IB", 1, 1))
                        meta_hdr = bencode({b"msg_type": 1, b"piece": 0, b"total_size": len(small_bundle.info_bytes)})
                        payload_m = meta_hdr + small_bundle.info_bytes
                        writer.write(struct.pack("!IBB", 2 + len(payload_m), BT_MSG_EXTENDED, BT_UT_METADATA_ID) + payload_m)
                        await writer.drain()
                        # Read interested + piece request, then send a 500 MiB frame length header
                        for _ in range(2):
                            h = await reader.readexactly(4)
                            (ml,) = struct.unpack("!I", h)
                            await reader.readexactly(ml)
                        writer.write(struct.pack("!I", 500 * 1024 * 1024))
                        await writer.drain()
                    except Exception:
                        pass
                    finally:
                        writer.close()
                mal_srv = await asyncio.start_server(_malicious_peer, "127.0.0.1", 0)
                mal_port = mal_srv.sockets[0].getsockname()[1]
                try:
                    with self.assertRaisesRegex(ValueError, "oversized_peer_wire_piece_message"):
                        await fetch_bundle_via_peer_wire("127.0.0.1", mal_port, small_bundle.infohash_20, timeout=2.0)
                finally:
                    mal_srv.close()
            finally:
                dht_mutable.close()
                dht_referral.close()
                dht_swarm.close()
                seeder.close()

        asyncio.run(_run_peer_wire_test())

    def test_bep19_webseed_fallback_and_oversized_response_rejection(self) -> None:
        sk = Ed25519PrivateKey.generate()
        priv_32 = sk.private_bytes_raw()
        pub_32 = sk.public_key().public_bytes_raw()
        salt = b"armiger-oci-v1-int8"
        files = {"model.json": b'{"ok":true}' * 100}
        bundle = build_torrent_bundle("int8", files, pub_32, salt, seq=210)
        serve_oversized = False

        class _WebSeedHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                fname = self.path.strip("/").split("/")[-1]
                if fname.endswith(".torrent"):
                    body = bundle.torrent_bytes
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif fname in files:
                    body = files[fname] + (b"X" * 5000 if serve_oversized else b"")
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, fmt, *args):
                return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _WebSeedHandler)
        ws_port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            def _local_fetcher(url: str, max_bytes: int = 1048576) -> bytes:
                with urllib.request.urlopen(url, timeout=3.0) as resp:
                    return resp.read(max_bytes + 1)

            res = fetch_bundle_via_bep19_webseed(
                info_bytes=bundle.info_bytes,
                webseed_base_url=f"http://127.0.0.1:{ws_port}/seed",
                expected_infohash_20=bundle.infohash_20,
                fetcher=_local_fetcher,
            )
            self.assertEqual(res["files"], files)
            self.assertEqual(res["merkle_root_sha256"], bundle.merkle_root_hex)

            # Oversized webseed body must fail closed immediately
            serve_oversized = True
            with self.assertRaisesRegex(ValueError, "bep19_response_exceeds_expected_length"):
                fetch_bundle_via_bep19_webseed(
                    info_bytes=bundle.info_bytes,
                    webseed_base_url=f"http://127.0.0.1:{ws_port}/seed",
                    expected_infohash_20=bundle.infohash_20,
                    fetcher=_local_fetcher,
                )
            serve_oversized = False

            # End-to-end CLI webseed-fallback test when P2P is unavailable
            async def _run_cli_ws_fallback() -> None:
                vpn = LocalTestVPN()
                dht = GatedBEP44UDPNode(vpn, anonymity_guard=AnonymityGuard(True), host="127.0.0.1", port=0)
                _, dht_port = await dht.start()
                try:
                    v_dict = {
                        b"ih": bundle.infohash_20,
                        b"mr": bytes.fromhex(bundle.merkle_root_hex),
                        b"seq": 210,
                        b"tier": b"int8",
                        b"ts": 1761264000,
                        b"v": 2,
                    }
                    dht.publish_local_mutable(privkey_32=priv_32, salt=salt, v=v_dict, seq=210)
                    # 1. Verify via byte-encoded bencoded BEP-46 pointer carrying `ws` and `mr` with no explicit webseed_urls
                    v_bencoded_bytes = bencode({
                        b"ih": bundle.infohash_20,
                        b"mr": bytes.fromhex(bundle.merkle_root_hex),
                        b"seq": 211,
                        b"tier": b"int8",
                        b"ts": 1761264000,
                        b"v": 2,
                        b"ws": f"http://127.0.0.1:{ws_port}/seed".encode("utf-8"),
                    })
                    dht.publish_local_mutable(privkey_32=priv_32, salt=salt, v=v_bencoded_bytes, seq=211)
                    # Also verify hostname resolution ("localhost") on active shared UDP transport
                    ping_resp = await dht.rpc_call(("localhost", dht_port), b"ping", {})
                    self.assertEqual(ping_resp.get(b"y"), b"r")

                    receipt_ws = await asyncio.to_thread(
                        verify_dht_release,
                        pubkey_hex=pub_32.hex(),
                        salt=salt,
                        dht_nodes=[("127.0.0.1", dht_port)],
                        webseed_urls=None,
                        webseed_fetcher=_local_fetcher,
                        mode="webseed-fallback",
                        min_seq=200,
                        timeout=2.0,
                    )
                    self.assertTrue(receipt_ws["verified"], msg=str(receipt_ws))
                    self.assertEqual(receipt_ws["mode_used"], "webseed-fallback")

                    # 2. Verify CLI main(--torrent-file) path
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tfile = Path(tmp_dir) / "bundle.torrent"
                        tfile.write_bytes(bundle.torrent_bytes)
                        # Unpatched HTTP webseed URL must fail closed (rc == 1) because default fetcher enforces HTTPS
                        rc_http = await asyncio.to_thread(
                            verify_dht_main,
                            [
                                "--pubkey", pub_32.hex(),
                                "--salt", salt.decode("utf-8"),
                                "--dht-node", f"127.0.0.1:{dht_port}",
                                "--webseed-url", f"http://127.0.0.1:{ws_port}/seed",
                                "--torrent-file", str(tfile),
                                "--mode", "webseed-fallback",
                            ],
                        )
                        self.assertEqual(rc_http, 1)

                        # With local fetcher injected into verify_dht_release, CLI main(--torrent-file) succeeds (rc == 0)
                        with unittest.mock.patch(
                            "armiger.verify_dht.verify_dht_release",
                            side_effect=lambda **kw: verify_dht_release(**{**kw, "webseed_fetcher": _local_fetcher}),
                        ):
                            rc_ok = await asyncio.to_thread(
                                verify_dht_main,
                                [
                                    "--pubkey", pub_32.hex(),
                                    "--salt", salt.decode("utf-8"),
                                    "--dht-node", f"127.0.0.1:{dht_port}",
                                    "--webseed-url", f"http://127.0.0.1:{ws_port}/seed",
                                    "--torrent-file", str(tfile),
                                    "--mode", "webseed-fallback",
                                ],
                            )
                        self.assertEqual(rc_ok, 0)
                finally:
                    dht.close()

            asyncio.run(_run_cli_ws_fallback())
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_dsse_pae_and_acyclic_provenance_envelope(self) -> None:
        pae = dsse_pae(DSSE_TYPE_ATTESTATION_V2, b'{"test":1}')
        self.assertTrue(pae.startswith(b"DSSEv1 "))

        with tempfile.TemporaryDirectory() as tmp_state, tempfile.TemporaryDirectory() as tmp_seal:
            WOTSPlusSHAKE256.set_state_dir(tmp_state, seal_dir=tmp_seal)
            try:
                contrib_ed, contrib_pub, _, _ = ThresholdSignatureVerifier.generate_hybrid_signer()
                eval_signers = [ThresholdSignatureVerifier.generate_hybrid_signer() for _ in range(3)]
                rel_signers = [ThresholdSignatureVerifier.generate_hybrid_signer() for _ in range(2)]

                parent_sha = hashlib.sha256(b"parent-bundle").hexdigest()
                cand_bytes = b'{"weights": [1.0, 2.0], "version": "2.12.3"}'
                cand_sha = hashlib.sha256(cand_bytes).hexdigest()
                suite_sha = hashlib.sha256(b"canary-suite-500").hexdigest()
                now_ts = int(time.time())

                baseline = AuthenticatedConformalBaseline.create_signed(
                    parent_bundle_sha256=parent_sha,
                    canary_suite_sha256=suite_sha,
                    accuracy=0.96,
                    ece=0.02,
                    macro_f1=0.95,
                    ed25519_priv=rel_signers[0][0],
                )

                eval_preds = []
                eval_asns = (13335, 15169, 16509)
                eval_ips = ("104.16.1.1", "8.8.8.8", "52.94.1.1")
                trusted_eval = {}
                asn_verifier = AuthoritativeASNVerifier(require_authoritative_binding=True)
                for idx, (e_priv, e_pub, e_wots_priv, e_wots_pub) in enumerate(eval_signers):
                    trusted_eval[e_pub] = f"evaluator-{idx}"
                    asn_verifier.register_pinned_key(e_pub, eval_asns[idx], pq_root_pubkey_hex=e_wots_pub)
                    pred = AcyclicProvenanceVerifier.sign_benchmark_attestation(
                        peer_id=f"eval-{idx}",
                        asn=eval_asns[idx],
                        parent_bundle_sha256=parent_sha,
                        candidate_bundle_sha256=cand_sha,
                        canary_suite_sha256=suite_sha,
                        canary_metrics={"accuracy": 0.97, "ece": 0.018, "macro_f1": 0.96, "mondrian_coverage": 0.96, "max_false_accept_rate": 0.0, "passed": True},
                        evaluator_version="2.12.2",
                        timestamp=now_ts,
                        ed25519_priv=e_priv,
                        wots_priv_seed=e_wots_priv,
                        ip_address=eval_ips[idx],
                    )
                    eval_preds.append(pred)

                manifest_unsigned = ThresholdDHTManifest(
                    seq=1001,
                    version="2.12.2",
                    bundle_sha256=cand_sha,
                    bundle_format="json",
                    infohash=hashlib.sha1(b"candidate").hexdigest(),
                    timestamp=now_ts,
                    min_asn_quorum=2,
                    signatures={},
                    pq_signatures={},
                )
                payload = manifest_unsigned.canonical_payload()
                sigs = {}
                pq_sigs = {}
                trusted_rel = {}
                rel_asns = (24940, 16276)
                for idx, (r_priv, r_pub, r_wots_priv, r_wots_pub) in enumerate(rel_signers):
                    sigs[r_pub] = r_priv.sign(payload).hex()
                    pq_sigs[r_wots_pub] = WOTSPlusSHAKE256.sign(r_wots_priv, payload).hex()
                    trusted_rel[r_pub] = f"release-signer-{idx}"
                    asn_verifier.register_pinned_key(r_pub, rel_asns[idx], pq_root_pubkey_hex=r_wots_pub)

                trusted_contrib = {contrib_pub: "contributor-0"}

                manifest = ThresholdDHTManifest(
                    seq=1001,
                    version="2.12.2",
                    bundle_sha256=cand_sha,
                    bundle_format="json",
                    infohash=hashlib.sha1(b"candidate").hexdigest(),
                    timestamp=now_ts,
                    min_asn_quorum=2,
                    signatures=sigs,
                    pq_signatures=pq_sigs,
                )

                unsigned_envelope = AcyclicReleaseEnvelope(
                    manifest=manifest,
                    parent_bundle_sha256=parent_sha,
                    canary_suite_sha256=suite_sha,
                    contributor_pubkeys=(contrib_pub,),
                    benchmark_attestations=tuple(eval_preds),
                    authenticated_baseline=baseline,
                )
                envelope = AcyclicProvenanceVerifier.sign_release_envelope(unsigned_envelope, rel_signers)
                ok, reason = AcyclicProvenanceVerifier.verify_envelope(
                    envelope,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertTrue(ok, msg=reason)

                # 1. Empty evaluator trust store must fail closed
                ok_empty, reason_empty = AcyclicProvenanceVerifier.verify_envelope(
                    envelope,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers={},
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_empty)
                self.assertEqual(reason_empty, "empty_trusted_evaluator_signers")

                # 2. Tampering with contributor_pubkeys or stripping baseline invalidates envelope signatures
                tampered_contrib = dataclasses.replace(envelope, contributor_pubkeys=(rel_signers[0][1],))
                ok_tc, reason_tc = AcyclicProvenanceVerifier.verify_envelope(
                    tampered_contrib,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_tc)

                stripped_contrib = dataclasses.replace(envelope, contributor_pubkeys=())
                ok_sc, reason_sc = AcyclicProvenanceVerifier.verify_envelope(
                    stripped_contrib,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_sc)
                self.assertEqual(reason_sc, "missing_contributor_pubkeys")

                # 3. Forged self-asserted ASN on an unpinned evaluator key must fail closed
                unpinned_asn_verifier = AuthoritativeASNVerifier(require_authoritative_binding=True)
                ok_unp, reason_unp = AcyclicProvenanceVerifier.verify_envelope(
                    envelope,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=unpinned_asn_verifier,
                )
                self.assertFalse(ok_unp)
                self.assertIn("evaluator_asn_binding_failed", reason_unp)

                # 4. Self-dealing contributor in evaluators must fail closed
                bad_envelope = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, contributor_pubkeys=(eval_signers[0][1],)),
                    rel_signers,
                )
                ok_bad, reason_bad = AcyclicProvenanceVerifier.verify_envelope(
                    bad_envelope,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers={eval_signers[0][1]: "contributor-0"},
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_bad)
                self.assertIn("cyclic_provenance_violation", reason_bad)

                # 5. Inner manifest disjoint, but envelope threshold includes an evaluator key -> must fail closed
                mixed_rel_trust = dict(trusted_rel)
                mixed_rel_trust[eval_signers[0][1]] = "colluding-eval-release"
                env_with_eval_sig = AcyclicProvenanceVerifier.sign_release_envelope(
                    unsigned_envelope,
                    [rel_signers[0], eval_signers[0]],
                )
                ok_mix, reason_mix = AcyclicProvenanceVerifier.verify_envelope(
                    env_with_eval_sig,
                    trusted_release_signers=mixed_rel_trust,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_mix)
                self.assertIn("cyclic_provenance_violation:evaluator_in_release_signers", reason_mix)

                # 5b. Manifest signed by a distinct key mapped to an evaluator identity (with independent envelope signers) must fail closed
                alt_rel_ed, alt_rel_pub, alt_rel_wots_priv, alt_rel_wots_pub = ThresholdSignatureVerifier.generate_hybrid_signer()
                asn_verifier.register_pinned_key(alt_rel_pub, 20940, pq_root_pubkey_hex=alt_rel_wots_pub)
                alias_rel_trust = dict(trusted_rel)
                alias_rel_trust[alt_rel_pub] = "evaluator-0"  # distinct key, same logical identity as eval_signers[0]
                alias_sigs = dict(sigs)
                alias_sigs[alt_rel_pub] = alt_rel_ed.sign(payload).hex()
                alias_pq_sigs = dict(pq_sigs)
                alias_pq_sigs[alt_rel_wots_pub] = WOTSPlusSHAKE256.sign(alt_rel_wots_priv, payload).hex()
                alias_manifest = dataclasses.replace(manifest, signatures=alias_sigs, pq_signatures=alias_pq_sigs)
                alias_env = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, manifest=alias_manifest),
                    rel_signers,
                )
                ok_alias, reason_alias = AcyclicProvenanceVerifier.verify_envelope(
                    alias_env,
                    trusted_release_signers=alias_rel_trust,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_alias)
                self.assertEqual(reason_alias, "cyclic_provenance_violation:evaluator_identity_in_release_signers:evaluator-0")

                # 6. ArmigerDHTRSIEngine.verify_and_stage_update rejects authenticated_baseline with mismatched parent or suite hash,
                #    and enforces AcyclicReleaseEnvelope verification before staging updates.
                from armiger.armiger_dht_rsi import ArmigerDHTRSIEngine, PeerAttestation
                eng = ArmigerDHTRSIEngine(
                    trusted_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_distinct_asns=2,
                )
                eng_ok_p, eng_reason_p, _ = eng.verify_and_stage_update(
                    manifest=manifest,
                    bundle_bytes=cand_bytes,
                    attestations=[],
                    candidate_predictor_fn=lambda item: dict(item.ground_truth),
                    authenticated_baseline=baseline,
                    expected_parent_bundle_sha256="f" * 64,
                )
                self.assertFalse(eng_ok_p)
                self.assertEqual(eng_reason_p, "authenticated_baseline_parent_bundle_mismatch")

                eng_ok_s, eng_reason_s, _ = eng.verify_and_stage_update(
                    manifest=manifest,
                    bundle_bytes=cand_bytes,
                    attestations=[],
                    candidate_predictor_fn=lambda item: dict(item.ground_truth),
                    authenticated_baseline=baseline,
                    expected_parent_bundle_sha256=parent_sha,
                )
                self.assertFalse(eng_ok_s)
                self.assertEqual(eng_reason_s, "authenticated_baseline_canary_suite_mismatch")

                peer_atts = []
                for idx, (e_priv, e_pub, e_wots_priv, e_wots_pub) in enumerate(eval_signers):
                    att_unsigned = PeerAttestation(
                        peer_id=f"eval-{idx}",
                        asn=eval_asns[idx],
                        peer_pubkey_hex=e_pub,
                        timestamp=now_ts,
                        signature_hex="",
                        pq_pubkey_hex=e_wots_pub,
                        pq_signature_hex="",
                        ip_address=eval_ips[idx],
                    )
                    att_msg = att_unsigned.canonical_bytes(cand_sha)
                    att_sig = e_priv.sign(att_msg).hex()
                    att_pq_sig = WOTSPlusSHAKE256.sign(e_wots_priv, att_msg).hex()
                    peer_atts.append(
                        dataclasses.replace(
                            att_unsigned,
                            signature_hex=att_sig,
                            pq_signature_hex=att_pq_sig,
                        )
                    )

                # 6a. Missing release_envelope is rejected by default
                eng_no_env, reason_no_env, _ = eng.verify_and_stage_update(
                    manifest=manifest,
                    bundle_bytes=cand_bytes,
                    attestations=peer_atts,
                    candidate_predictor_fn=lambda item: dict(item.ground_truth),
                    authenticated_baseline=baseline,
                    expected_parent_bundle_sha256=parent_sha,
                    expected_canary_suite_sha256=suite_sha,
                    asn_verifier=asn_verifier,
                    bootstrap_checkpoint_seq=1000,
                )
                self.assertFalse(eng_no_env)
                self.assertEqual(reason_no_env, "missing_acyclic_release_envelope")

                # 6b. Cyclic release_envelope is rejected at staging time
                eng_bad_env, reason_bad_env, _ = eng.verify_and_stage_update(
                    manifest=manifest,
                    bundle_bytes=cand_bytes,
                    attestations=peer_atts,
                    candidate_predictor_fn=lambda item: dict(item.ground_truth),
                    authenticated_baseline=baseline,
                    expected_parent_bundle_sha256=parent_sha,
                    expected_canary_suite_sha256=suite_sha,
                    release_envelope=bad_envelope,
                    asn_verifier=asn_verifier,
                    bootstrap_checkpoint_seq=1000,
                )
                self.assertFalse(eng_bad_env)
                self.assertIn("acyclic_provenance_failed:cyclic_provenance_violation", reason_bad_env)

                # 6c. Valid release_envelope passes acyclic provenance + 500-canary gate
                def _good_pred(item):
                    res = dict(item.ground_truth)
                    res["confidence"] = 0.96
                    return res

                eng_good_env, reason_good_env, report_good = eng.verify_and_stage_update(
                    manifest=manifest,
                    bundle_bytes=cand_bytes,
                    attestations=peer_atts,
                    candidate_predictor_fn=_good_pred,
                    authenticated_baseline=baseline,
                    expected_parent_bundle_sha256=parent_sha,
                    expected_canary_suite_sha256=suite_sha,
                    release_envelope=envelope,
                    asn_verifier=asn_verifier,
                    bootstrap_checkpoint_seq=1000,
                )
                self.assertTrue(eng_good_env, msg=reason_good_env)
                self.assertEqual(reason_good_env, "update_staged_and_verified")
                self.assertIsNotNone(report_good)

                # 6d. Fresh receiving process without process-local WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS registry
                #     succeeds using explicit pinned PQ trust from asn_verifier / trusted_release_pq_signers
                saved_ed_to_wots = {k: set(v) for k, v in WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.items()}
                saved_root_by_ed = dict(WOTSPlusSHAKE256._WOTS_TO_ED25519_PUB)
                try:
                    WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.clear()
                    WOTSPlusSHAKE256._WOTS_TO_ED25519_PUB.clear()
                    rel_pq_trust = {
                        r_wots_pub: f"release-signer-{idx}"
                        for idx, (_, _, _, r_wots_pub) in enumerate(rel_signers)
                    }
                    eval_pq_trust = {
                        e_wots_pub: f"evaluator-{idx}"
                        for idx, (_, _, _, e_wots_pub) in enumerate(eval_signers)
                    }
                    ok_fresh, reason_fresh = AcyclicProvenanceVerifier.verify_envelope(
                        envelope,
                        trusted_release_signers=trusted_rel,
                        trusted_evaluator_signers=trusted_eval,
                        trusted_contributor_signers=trusted_contrib,
                        trusted_release_pq_signers=rel_pq_trust,
                        trusted_evaluator_pq_signers=eval_pq_trust,
                        threshold_m=2,
                        min_evaluator_asns=3,
                        current_seq=0,
                        bootstrap_checkpoint_seq=1000,
                        asn_verifier=asn_verifier,
                    )
                    self.assertTrue(ok_fresh, msg=reason_fresh)
                finally:
                    WOTSPlusSHAKE256._ED25519_TO_WOTS_PUBS.update(saved_ed_to_wots)
                    WOTSPlusSHAKE256._WOTS_TO_ED25519_PUB.update(saved_root_by_ed)

                # 7. Incomplete or empty benchmark attestation metrics are rejected by sign_benchmark_attestation AND verify_envelope
                with self.assertRaisesRegex(ValueError, "missing_canary_metrics"):
                    AcyclicProvenanceVerifier.sign_benchmark_attestation(
                        peer_id="eval-0",
                        asn=eval_asns[0],
                        parent_bundle_sha256=parent_sha,
                        candidate_bundle_sha256=cand_sha,
                        canary_suite_sha256=suite_sha,
                        canary_metrics={},
                        evaluator_version="2.12.3",
                        timestamp=now_ts,
                        ed25519_priv=eval_signers[0][0],
                        wots_priv_seed=eval_signers[0][2],
                        ip_address=eval_ips[0],
                    )
                with self.assertRaisesRegex(ValueError, "missing_or_invalid_passed_flag|missing_canary_metric"):
                    AcyclicProvenanceVerifier.sign_benchmark_attestation(
                        peer_id="eval-0",
                        asn=eval_asns[0],
                        parent_bundle_sha256=parent_sha,
                        candidate_bundle_sha256=cand_sha,
                        canary_suite_sha256=suite_sha,
                        canary_metrics={"accuracy": 0.97, "ece": 0.018, "macro_f1": 0.96},
                        evaluator_version="2.12.3",
                        timestamp=now_ts,
                        ed25519_priv=eval_signers[0][0],
                        wots_priv_seed=eval_signers[0][2],
                        ip_address=eval_ips[0],
                    )
                empty_unsigned = dataclasses.replace(eval_preds[0], canary_metrics={}, signature_hex="", pq_signature_hex="")
                empty_pae = empty_unsigned.pae_bytes()
                empty_pred = dataclasses.replace(
                    empty_unsigned,
                    signature_hex=eval_signers[0][0].sign(empty_pae).hex(),
                    pq_signature_hex=WOTSPlusSHAKE256.sign(eval_signers[0][2], empty_pae).hex(),
                )
                env_empty_m = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, benchmark_attestations=(empty_pred, eval_preds[1], eval_preds[2])),
                    rel_signers,
                )
                ok_em, reason_em = AcyclicProvenanceVerifier.verify_envelope(
                    env_empty_m,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_em)
                self.assertEqual(reason_em, "attestation_missing_canary_metrics")

                failing_pred = AcyclicProvenanceVerifier.sign_benchmark_attestation(
                    peer_id="eval-0",
                    asn=eval_asns[0],
                    parent_bundle_sha256=parent_sha,
                    candidate_bundle_sha256=cand_sha,
                    canary_suite_sha256=suite_sha,
                    canary_metrics={"accuracy": 0.80, "ece": 0.02, "macro_f1": 0.96, "mondrian_coverage": 0.96, "max_false_accept_rate": 0.0, "passed": True},
                    evaluator_version="2.12.3",
                    timestamp=now_ts,
                    ed25519_priv=eval_signers[0][0],
                    wots_priv_seed=eval_signers[0][2],
                    ip_address=eval_ips[0],
                )
                env_fail_m = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, benchmark_attestations=(failing_pred, eval_preds[1], eval_preds[2])),
                    rel_signers,
                )
                ok_fm, reason_fm = AcyclicProvenanceVerifier.verify_envelope(
                    env_fail_m,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_fm)
                self.assertEqual(reason_fm, "attestation_accuracy_regression_exceeded")

                # 8. Attestation signed with passed=False is bound in PAE: rejected unmodified and rejected if mutated/stripped
                failed_pass_pred = AcyclicProvenanceVerifier.sign_benchmark_attestation(
                    peer_id="eval-0",
                    asn=eval_asns[0],
                    parent_bundle_sha256=parent_sha,
                    candidate_bundle_sha256=cand_sha,
                    canary_suite_sha256=suite_sha,
                    canary_metrics={"accuracy": 0.97, "ece": 0.018, "macro_f1": 0.96, "mondrian_coverage": 0.96, "max_false_accept_rate": 0.0, "passed": False},
                    evaluator_version="2.12.3",
                    timestamp=now_ts,
                    ed25519_priv=eval_signers[0][0],
                    wots_priv_seed=eval_signers[0][2],
                    ip_address=eval_ips[0],
                )
                self.assertIs(failed_pass_pred.canary_metrics["passed"], False)
                env_fp = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, benchmark_attestations=(failed_pass_pred, eval_preds[1], eval_preds[2])),
                    rel_signers,
                )
                ok_fp, reason_fp = AcyclicProvenanceVerifier.verify_envelope(
                    env_fp,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_fp)
                self.assertEqual(reason_fp, "attestation_canary_marked_failed")

                # Flipping "passed" to True post-signature invalidates the evaluator Ed25519 signature
                stripped_pass_pred = dataclasses.replace(
                    failed_pass_pred,
                    canary_metrics={
                        "accuracy": 0.97,
                        "ece": 0.018,
                        "macro_f1": 0.96,
                        "mondrian_coverage": 0.96,
                        "max_false_accept_rate": 0.0,
                        "passed": True,
                    },
                )
                env_sp = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, benchmark_attestations=(stripped_pass_pred, eval_preds[1], eval_preds[2])),
                    rel_signers,
                )
                ok_sp, reason_sp = AcyclicProvenanceVerifier.verify_envelope(
                    env_sp,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_sp)
                self.assertIn("invalid_evaluator_ed25519_signature", reason_sp)

                # 9. Omitting ece or macro_f1 from benchmark attestation must be rejected
                no_ece_unsigned = dataclasses.replace(
                    eval_preds[0],
                    canary_metrics={"accuracy": 0.97, "macro_f1": 0.96, "mondrian_coverage": 0.96, "max_false_accept_rate": 0.0, "passed": True},
                    signature_hex="",
                    pq_signature_hex="",
                )
                no_ece_pae = no_ece_unsigned.pae_bytes()
                missing_ece_pred = dataclasses.replace(
                    no_ece_unsigned,
                    signature_hex=eval_signers[0][0].sign(no_ece_pae).hex(),
                    pq_signature_hex=WOTSPlusSHAKE256.sign(eval_signers[0][2], no_ece_pae).hex(),
                )
                env_no_ece = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, benchmark_attestations=(missing_ece_pred, eval_preds[1], eval_preds[2])),
                    rel_signers,
                )
                ok_ne, reason_ne = AcyclicProvenanceVerifier.verify_envelope(
                    env_no_ece,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_ne)
                self.assertEqual(reason_ne, "attestation_missing_ece")

                # 10. Failing mondrian_coverage or max_false_accept_rate must be rejected
                low_cov_pred = AcyclicProvenanceVerifier.sign_benchmark_attestation(
                    peer_id="eval-0",
                    asn=eval_asns[0],
                    parent_bundle_sha256=parent_sha,
                    candidate_bundle_sha256=cand_sha,
                    canary_suite_sha256=suite_sha,
                    canary_metrics={"accuracy": 0.97, "ece": 0.018, "macro_f1": 0.96, "mondrian_coverage": 0.80, "max_false_accept_rate": 0.0, "passed": True},
                    evaluator_version="2.12.3",
                    timestamp=now_ts,
                    ed25519_priv=eval_signers[0][0],
                    wots_priv_seed=eval_signers[0][2],
                    ip_address=eval_ips[0],
                )
                env_low_cov = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, benchmark_attestations=(low_cov_pred, eval_preds[1], eval_preds[2])),
                    rel_signers,
                )
                ok_lc, reason_lc = AcyclicProvenanceVerifier.verify_envelope(
                    env_low_cov,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_lc)
                self.assertEqual(reason_lc, "attestation_mondrian_coverage_below_threshold")

                # 11. Distinct contributor and release/evaluator keys mapped to the same principal identity must fail,
                #     and unregistered contributor keys (not in trusted_contributor_signers) must fail closed.
                ok_no_c, reason_no_c = AcyclicProvenanceVerifier.verify_envelope(
                    envelope,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers={},
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_no_c)
                self.assertEqual(reason_no_c, "empty_trusted_contributor_signers")

                _, unreg_contrib_pub, _, _ = ThresholdSignatureVerifier.generate_hybrid_signer()
                unreg_env = AcyclicProvenanceVerifier.sign_release_envelope(
                    dataclasses.replace(unsigned_envelope, contributor_pubkeys=(unreg_contrib_pub,)),
                    rel_signers,
                )
                ok_unreg, reason_unreg = AcyclicProvenanceVerifier.verify_envelope(
                    unreg_env,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers=trusted_contrib,
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_unreg)
                self.assertTrue(reason_unreg.startswith("untrusted_contributor_pubkey:"))

                ok_cr, reason_cr = AcyclicProvenanceVerifier.verify_envelope(
                    envelope,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers={contrib_pub: "release-signer-0"},
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_cr)
                self.assertEqual(reason_cr, "cyclic_provenance_violation:contributor_identity_in_release_signers:release-signer-0")

                ok_ce, reason_ce = AcyclicProvenanceVerifier.verify_envelope(
                    envelope,
                    trusted_release_signers=trusted_rel,
                    trusted_evaluator_signers=trusted_eval,
                    trusted_contributor_signers={contrib_pub: "evaluator-0"},
                    threshold_m=2,
                    min_evaluator_asns=3,
                    current_seq=0,
                    bootstrap_checkpoint_seq=1000,
                    asn_verifier=asn_verifier,
                )
                self.assertFalse(ok_ce)
                self.assertEqual(reason_ce, "cyclic_provenance_violation:contributor_identity_in_evaluators:evaluator-0")
            finally:
                WOTSPlusSHAKE256.set_state_dir(None, seal_dir=None)

    def test_type_confusion_and_keepalive_flood_defenses(self) -> None:
        sk = Ed25519PrivateKey.generate()
        pub_32 = sk.public_key().public_bytes_raw()
        salt = b"armiger:v2:test"
        v_dict = {b"ih": b"C" * 20, b"seq": 5, b"v": 2}
        # Sign the byte-string bencode(v_dict) instead of native dict v_dict
        wrong_type_sig = sk.sign(bep44_signing_buffer(v=bencode(v_dict), seq=5, salt=salt))

        async def _run_checks() -> None:
            vpn = LocalTestVPN()
            dht = GatedBEP44UDPNode(vpn, anonymity_guard=AnonymityGuard(True), host="127.0.0.1", port=0)
            _, dht_port = await dht.start()
            try:
                target = compute_bep44_target(pub_32, salt)
                dht_client = GatedBEP44UDPNode(vpn, anonymity_guard=AnonymityGuard(True))
                g_resp = await dht_client.rpc_call(
                    ("127.0.0.1", dht_port),
                    b"get",
                    {b"id": b"B" * 20, b"target": target},
                )
                token = g_resp[b"r"][b"token"]
                # Supplying a dict with a byte-string signature must be rejected with 206
                p_resp = await dht_client.rpc_call(
                    ("127.0.0.1", dht_port),
                    b"put",
                    {
                        b"id": b"B" * 20,
                        b"k": pub_32,
                        b"seq": 5,
                        b"v": v_dict,
                        b"sig": wrong_type_sig,
                        b"salt": salt,
                        b"token": token,
                    },
                )
                self.assertEqual(p_resp[b"y"], b"e")
                self.assertEqual(p_resp[b"e"], 206)

                # Malicious peer flooding keepalives must fail closed via non-progress limit
                async def _keepalive_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    try:
                        await reader.readexactly(68)
                        writer.write(b"\x13BitTorrent protocol" + b"\x00\x00\x00\x00\x00\x10\x00\x00" + (b"C" * 20) + (b"P" * 20))
                        for _ in range(40):
                            writer.write(b"\x00\x00\x00\x00")
                        await writer.drain()
                    except Exception:
                        pass
                    finally:
                        writer.close()

                srv = await asyncio.start_server(_keepalive_peer, "127.0.0.1", 0)
                kp_port = srv.sockets[0].getsockname()[1]
                try:
                    with self.assertRaisesRegex(ValueError, "peer_wire_excessive_non_progress_messages"):
                        await fetch_bundle_via_peer_wire("127.0.0.1", kp_port, b"C" * 20, timeout=2.0)
                finally:
                    srv.close()
                    await srv.wait_closed()

                # Malicious UDP node returning integer k/sig (10^10) must be skipped without memory allocation
                class _BadUDPProto(asyncio.DatagramProtocol):
                    def connection_made(self, transport):
                        self.transport = transport
                    def datagram_received(self, data, addr):
                        self.transport.sendto(
                            bencode({
                                b"t": b"vf01",
                                b"y": b"r",
                                b"r": {
                                    b"id": b"Z" * 20,
                                    b"k": 10_000_000_000,
                                    b"sig": 10_000_000_000,
                                    b"seq": 999,
                                    b"v": {b"ih": b"C" * 20},
                                },
                            }),
                            addr,
                        )
                loop = asyncio.get_running_loop()
                bad_tr, _ = await loop.create_datagram_endpoint(_BadUDPProto, local_addr=("127.0.0.1", 0))
                bad_port = bad_tr.get_extra_info("sockname")[1]
                try:
                    bad_res = await asyncio.to_thread(
                        verify_dht_release,
                        pubkey_hex=pub_32.hex(),
                        salt=salt,
                        dht_nodes=[("127.0.0.1", bad_port)],
                        mode="strict-p2p",
                        timeout=1.0,
                    )
                    self.assertFalse(bad_res["verified"])
                    self.assertEqual(bad_res["reason"], "bep44_mutable_record_not_found_or_invalid_sig")
                finally:
                    bad_tr.close()

                # Strict BEP-3 peer that starts choked, waits for INTERESTED before UNCHOKE, and rejects >16KiB requests
                strict_files = {"big.bin": b"0123456789abcdef" * 2500}  # 40,000 bytes (> 2 x 16 KiB blocks)
                strict_bundle = build_torrent_bundle("int8", strict_files, pub_32, salt, seq=12)

                async def _strict_bep3_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    try:
                        await reader.readexactly(68)
                        writer.write(b"\x13BitTorrent protocol" + b"\x00\x00\x00\x00\x00\x10\x00\x00" + strict_bundle.infohash_20 + (b"-ST0001-" + b"1" * 12))
                        ext_hs = bencode({b"m": {b"ut_metadata": 2}, b"metadata_size": len(strict_bundle.info_bytes)})
                        writer.write(struct.pack("!IBB", 2 + len(ext_hs), 20, 0) + ext_hs)
                        await writer.drain()
                        un_choked = False
                        while True:
                            hdr = await reader.readexactly(4)
                            (mlen,) = struct.unpack("!I", hdr)
                            if mlen == 0:
                                continue
                            body = await reader.readexactly(mlen)
                            mid = body[0]
                            if mid == 20 and body[1] == 2:
                                mhdr = bencode({b"msg_type": 1, b"piece": 0, b"total_size": len(strict_bundle.info_bytes)})
                                writer.write(struct.pack("!IBB", 2 + len(mhdr) + len(strict_bundle.info_bytes), 20, 1) + mhdr + strict_bundle.info_bytes)
                                await writer.drain()
                            elif mid == 2:  # INTERESTED -> now send UNCHOKE
                                un_choked = True
                                writer.write(struct.pack("!IB", 1, 1))
                                await writer.drain()
                            elif mid == 6 and len(body) == 13:
                                if not un_choked:
                                    break
                                idx, beg, rlen = struct.unpack("!III", body[1:13])
                                if rlen > 16384:
                                    break
                                blk = strict_bundle.payload_bytes[idx * strict_bundle.piece_length + beg : idx * strict_bundle.piece_length + beg + rlen]
                                writer.write(struct.pack("!IBII", 9 + len(blk), 7, idx, beg) + blk)
                                await writer.drain()
                    except Exception:
                        pass
                    finally:
                        writer.close()

                st_srv = await asyncio.start_server(_strict_bep3_peer, "127.0.0.1", 0)
                st_port = st_srv.sockets[0].getsockname()[1]
                try:
                    st_res = await fetch_bundle_via_peer_wire(
                        "127.0.0.1",
                        st_port,
                        strict_bundle.infohash_20,
                        timeout=3.0,
                        pipeline_depth=8,
                    )
                    self.assertEqual(st_res["files"], strict_files)
                finally:
                    st_srv.close()
                    await st_srv.wait_closed()

                # Alternating CHOKE / UNCHOKE without delivering payload bytes must fail closed
                async def _choke_cycle_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    try:
                        await reader.readexactly(68)
                        writer.write(b"\x13BitTorrent protocol" + b"\x00\x00\x00\x00\x00\x10\x00\x00" + strict_bundle.infohash_20 + (b"-CC0001-" + b"1" * 12))
                        ext_hs = bencode({b"m": {b"ut_metadata": 2}, b"metadata_size": len(strict_bundle.info_bytes)})
                        writer.write(struct.pack("!IBB", 2 + len(ext_hs), 20, 0) + ext_hs)
                        await writer.drain()
                        while True:
                            hdr = await reader.readexactly(4)
                            (mlen,) = struct.unpack("!I", hdr)
                            if mlen == 0:
                                continue
                            body = await reader.readexactly(mlen)
                            mid = body[0]
                            if mid == 20 and body[1] == 2:
                                mhdr = bencode({b"msg_type": 1, b"piece": 0, b"total_size": len(strict_bundle.info_bytes)})
                                writer.write(struct.pack("!IBB", 2 + len(mhdr) + len(strict_bundle.info_bytes), 20, 1) + mhdr + strict_bundle.info_bytes)
                                await writer.drain()
                            elif mid == 2:
                                for _ in range(20):
                                    writer.write(struct.pack("!IB", 1, 1))  # UNCHOKE
                                    writer.write(struct.pack("!IB", 1, 0))  # CHOKE
                                await writer.drain()
                                await asyncio.sleep(0.5)
                                break
                    except Exception:
                        pass
                    finally:
                        writer.close()

                cc_srv = await asyncio.start_server(_choke_cycle_peer, "127.0.0.1", 0)
                cc_port = cc_srv.sockets[0].getsockname()[1]
                try:
                    with self.assertRaisesRegex(ValueError, "peer_wire_excessive_non_progress_messages"):
                        await fetch_bundle_via_peer_wire("127.0.0.1", cc_port, strict_bundle.infohash_20, timeout=2.0)
                finally:
                    cc_srv.close()
                    await cc_srv.wait_closed()

                # Oversized individual ut_metadata piece must fail closed before retention
                async def _oversized_meta_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    try:
                        await reader.readexactly(68)
                        writer.write(b"\x13BitTorrent protocol" + b"\x00\x00\x00\x00\x00\x10\x00\x00" + strict_bundle.infohash_20 + (b"-OM0001-" + b"1" * 12))
                        ext_hs = bencode({b"m": {b"ut_metadata": 2}, b"metadata_size": len(strict_bundle.info_bytes)})
                        writer.write(struct.pack("!IBB", 2 + len(ext_hs), 20, 0) + ext_hs)
                        await writer.drain()
                        while True:
                            hdr = await reader.readexactly(4)
                            (mlen,) = struct.unpack("!I", hdr)
                            if mlen == 0:
                                continue
                            body = await reader.readexactly(mlen)
                            if body[0] == 20 and body[1] == 2:
                                mhdr = bencode({b"msg_type": 1, b"piece": 0, b"total_size": len(strict_bundle.info_bytes)})
                                bad_chunk = strict_bundle.info_bytes + b"X" * 1024
                                writer.write(struct.pack("!IBB", 2 + len(mhdr) + len(bad_chunk), 20, 1) + mhdr + bad_chunk)
                                await writer.drain()
                                break
                    except Exception:
                        pass
                    finally:
                        writer.close()

                om_srv = await asyncio.start_server(_oversized_meta_peer, "127.0.0.1", 0)
                om_port = om_srv.sockets[0].getsockname()[1]
                try:
                    with self.assertRaisesRegex(ValueError, "ut_metadata_piece_length_mismatch"):
                        await fetch_bundle_via_peer_wire("127.0.0.1", om_port, strict_bundle.infohash_20, timeout=2.0)
                finally:
                    om_srv.close()
                    await om_srv.wait_closed()

                # Slow-drip peer exceeding max_total_seconds must fail closed with peer_wire_total_deadline_exceeded
                async def _slow_drip_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    try:
                        await reader.readexactly(68)
                        writer.write(b"\x13BitTorrent protocol" + b"\x00\x00\x00\x00\x00\x10\x00\x00" + strict_bundle.infohash_20 + (b"-SD0001-" + b"1" * 12))
                        await writer.drain()
                        await asyncio.sleep(0.5)
                    except Exception:
                        pass
                    finally:
                        writer.close()

                sd_srv = await asyncio.start_server(_slow_drip_peer, "127.0.0.1", 0)
                sd_port = sd_srv.sockets[0].getsockname()[1]
                try:
                    with self.assertRaisesRegex(asyncio.TimeoutError, "peer_wire_total_deadline_exceeded"):
                        await fetch_bundle_via_peer_wire(
                            "127.0.0.1",
                            sd_port,
                            strict_bundle.infohash_20,
                            timeout=2.0,
                            max_total_seconds=0.1,
                        )
                finally:
                    sd_srv.close()
                    await sd_srv.wait_closed()

                # Bounded peer attempts in verify_dht_release when >20 dead peers are returned
                many_dead_peers = [("127.0.0.1", 65000 + i) for i in range(20)]
                valid_sig = sk.sign(bep44_signing_buffer(v=v_dict, seq=5, salt=salt))
                tok_ok = g_resp[b"r"][b"token"]
                await dht_client.rpc_call(
                    ("127.0.0.1", dht_port),
                    b"put",
                    {
                        b"id": b"B" * 20,
                        b"k": pub_32,
                        b"seq": 5,
                        b"v": v_dict,
                        b"sig": valid_sig,
                        b"salt": salt,
                        b"token": tok_ok,
                    },
                )
                cap_res = await asyncio.to_thread(
                    verify_dht_release,
                    pubkey_hex=pub_32.hex(),
                    salt=salt,
                    dht_nodes=[("127.0.0.1", dht_port)],
                    explicit_peers=many_dead_peers,
                    mode="strict-p2p",
                    timeout=0.2,
                    max_peers=3,
                    p2p_budget_sec=2.0,
                )
                self.assertFalse(cap_res["verified"])
                self.assertEqual(len(cap_res["p2p_errors"]), 3)

                # Malformed metadata with piece length = 1 is rejected before allocating block list
                bad_geom_info = bencode({
                    b"files": [{b"length": 65536, b"path": [b"x.bin"]}],
                    b"name": b"bad-geom",
                    b"piece length": 1,
                    b"pieces": b"A" * 20,
                })
                bad_geom_ih = hashlib.sha1(bad_geom_info).digest()

                async def _bad_geom_peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    try:
                        await reader.readexactly(68)
                        writer.write(b"\x13BitTorrent protocol" + b"\x00\x00\x00\x00\x00\x10\x00\x00" + bad_geom_ih + (b"-BG0001-" + b"1" * 12))
                        ext_hs = bencode({b"m": {b"ut_metadata": 2}, b"metadata_size": len(bad_geom_info)})
                        writer.write(struct.pack("!IBB", 2 + len(ext_hs), 20, 0) + ext_hs)
                        await writer.drain()
                        while True:
                            hdr = await reader.readexactly(4)
                            (mlen,) = struct.unpack("!I", hdr)
                            if mlen == 0:
                                continue
                            body = await reader.readexactly(mlen)
                            if body[0] == 20 and body[1] == 2:
                                mhdr = bencode({b"msg_type": 1, b"piece": 0, b"total_size": len(bad_geom_info)})
                                writer.write(struct.pack("!IBB", 2 + len(mhdr) + len(bad_geom_info), 20, 1) + mhdr + bad_geom_info)
                                await writer.drain()
                                break
                    except Exception:
                        pass
                    finally:
                        writer.close()

                bg_srv = await asyncio.start_server(_bad_geom_peer, "127.0.0.1", 0)
                bg_port = bg_srv.sockets[0].getsockname()[1]
                try:
                    with self.assertRaisesRegex(ValueError, "invalid_piece_length"):
                        await fetch_bundle_via_peer_wire("127.0.0.1", bg_port, bad_geom_ih, timeout=2.0)
                finally:
                    bg_srv.close()
                    await bg_srv.wait_closed()

                # Trickling webseed exceeding max_total_seconds raises TimeoutError("webseed_total_deadline_exceeded")
                def _slow_fetcher(url: str, max_b: int = 0) -> bytes:
                    time.sleep(0.15)
                    return b"0123456789abcdef" * 2500

                with self.assertRaisesRegex(TimeoutError, "webseed_total_deadline_exceeded"):
                    fetch_bundle_via_bep19_webseed(
                        strict_bundle.info_bytes,
                        "https://example.org/seed",
                        expected_infohash_20=strict_bundle.infohash_20,
                        fetcher=_slow_fetcher,
                        timeout=1.0,
                        max_total_seconds=0.05,
                    )

                # URL-encoding of special characters (#, ?, space) in bundle and file names
                special_files = {"weights #1?.json": b'{"ok":true}'}
                special_bundle = build_torrent_bundle("int8", special_files, pub_32, salt, seq=15)
                seen_urls = []
                def _encoding_fetcher(url: str, max_b: int = 0) -> bytes:
                    seen_urls.append(url)
                    return b'{"ok":true}'
                sp_res = fetch_bundle_via_bep19_webseed(
                    special_bundle.info_bytes,
                    "https://example.org/seed",
                    expected_infohash_20=special_bundle.infohash_20,
                    fetcher=_encoding_fetcher,
                )
                self.assertEqual(sp_res["files"], special_files)
                self.assertIn("weights%20%231%3F.json", seen_urls[0])

                # Tiny-piece metadata on webseed and verify_info_and_payload rejected before fetching or chunk allocation
                tiny_ws_info = bencode({
                    b"files": [{b"length": 67108864, b"path": [b"big.bin"], b"sha256": b"S" * 32}],
                    b"merkle_root": b"M" * 32,
                    b"name": b"tiny-pieces",
                    b"piece length": 1,
                    b"pieces": b"",
                })
                tiny_ws_ih = hashlib.sha1(tiny_ws_info).digest()
                fetcher_called = False
                def _never_called_fetcher(url: str, max_b: int = 0) -> bytes:
                    nonlocal fetcher_called
                    fetcher_called = True
                    return b"x"
                with self.assertRaisesRegex(ValueError, "invalid_piece_length"):
                    fetch_bundle_via_bep19_webseed(
                        tiny_ws_info,
                        "https://example.org/seed",
                        expected_infohash_20=tiny_ws_ih,
                        fetcher=_never_called_fetcher,
                    )
                self.assertFalse(fetcher_called)
                with self.assertRaisesRegex(ValueError, "invalid_piece_length"):
                    verify_info_and_payload(tiny_ws_info, b"", expected_infohash_20=tiny_ws_ih)

                # _fetch_webseed_metadata enforces monotonic max_total_seconds across candidate URLs
                from armiger.verify_dht import _fetch_webseed_metadata
                import armiger.oci_dht_node as _shim_mod
                self.assertTrue(callable(getattr(_shim_mod, "parse_bootstrap_nodes", None)))
                meta_calls = 0
                def _slow_meta_fetcher(url: str) -> bytes:
                    nonlocal meta_calls
                    meta_calls += 1
                    time.sleep(0.08)
                    return strict_bundle.torrent_bytes
                self.assertIsNone(
                    _fetch_webseed_metadata(
                        "https://example.org/seed",
                        strict_bundle.infohash_20,
                        fetcher=_slow_meta_fetcher,
                        timeout=1.0,
                        max_total_seconds=0.04,
                    )
                )
                self.assertEqual(meta_calls, 1)

                # Delayed-header and delayed-redirect tests against the default transport (_default_get / _https_get_with_deadline)
                import http.client
                import socket as _sock_mod

                def _run_raw_http_server(handler_fn):
                    srv_sock = _sock_mod.socket(_sock_mod.AF_INET, _sock_mod.SOCK_STREAM)
                    srv_sock.setsockopt(_sock_mod.SOL_SOCKET, _sock_mod.SO_REUSEADDR, 1)
                    srv_sock.bind(("127.0.0.1", 0))
                    srv_sock.listen(8)
                    port_num = srv_sock.getsockname()[1]
                    stop_ev = threading.Event()
                    def _accept_loop():
                        while not stop_ev.is_set():
                            try:
                                srv_sock.settimeout(0.05)
                                conn_s, _ = srv_sock.accept()
                            except OSError:
                                continue
                            threading.Thread(target=handler_fn, args=(conn_s,), daemon=True).start()
                    t_srv = threading.Thread(target=_accept_loop, daemon=True)
                    t_srv.start()
                    return srv_sock, port_num, stop_ev

                # (a) Delayed response headers trickling across max_total_seconds
                def _trickle_headers_handler(conn_s):
                    try:
                        conn_s.recv(1024)
                        conn_s.sendall(b"HTTP/1.1 200 OK\r\n")
                        for i in range(10):
                            time.sleep(0.03)
                            conn_s.sendall(f"X-Trickle-{i}: val\r\n".encode("ascii"))
                    except Exception:
                        pass
                    finally:
                        try:
                            conn_s.close()
                        except Exception:
                            pass

                th_sock, th_port, th_stop = _run_raw_http_server(_trickle_headers_handler)
                try:
                    with self.assertRaisesRegex(TimeoutError, "webseed_total_deadline_exceeded"):
                        fetch_bundle_via_bep19_webseed(
                            strict_bundle.info_bytes,
                            f"https://127.0.0.1:{th_port}/seed",
                            expected_infohash_20=strict_bundle.infohash_20,
                            timeout=1.0,
                            max_total_seconds=0.07,
                            tls_wrap=False,
                        )
                finally:
                    th_stop.set()
                    th_sock.close()

                # (b) Successive delayed HTTPS redirects exceeding max_total_seconds
                def _delayed_redirect_handler(conn_s):
                    try:
                        conn_s.recv(1024)
                        time.sleep(0.035)
                        resp_bytes = (
                            f"HTTP/1.1 302 Found\r\n"
                            f"Location: https://127.0.0.1:{dr_port}/seed/next\r\n"
                            f"Content-Length: 0\r\n"
                            f"Connection: close\r\n\r\n"
                        ).encode("ascii")
                        conn_s.sendall(resp_bytes)
                    except Exception:
                        pass
                    finally:
                        try:
                            conn_s.close()
                        except Exception:
                            pass

                dr_sock, dr_port, dr_stop = _run_raw_http_server(_delayed_redirect_handler)
                try:
                    with self.assertRaisesRegex(TimeoutError, "webseed_total_deadline_exceeded"):
                        fetch_bundle_via_bep19_webseed(
                            strict_bundle.info_bytes,
                            f"https://127.0.0.1:{dr_port}/seed",
                            expected_infohash_20=strict_bundle.infohash_20,
                            timeout=1.0,
                            max_total_seconds=0.07,
                            tls_wrap=False,
                        )
                finally:
                    dr_stop.set()
                    dr_sock.close()

                # (c) Stalled DNS resolution in _DeadlineHTTPSConnection.connect() is bounded by max_total_seconds
                def _stalled_resolver(host: str, port: int):
                    time.sleep(0.25)
                    return [(_sock_mod.AF_INET, _sock_mod.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

                t0_dns = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, "webseed_total_deadline_exceeded"):
                    fetch_bundle_via_bep19_webseed(
                        strict_bundle.info_bytes,
                        "https://stalled.example.org/seed",
                        expected_infohash_20=strict_bundle.infohash_20,
                        timeout=1.0,
                        max_total_seconds=0.06,
                        resolver_fn=_stalled_resolver,
                    )
                self.assertLess(time.monotonic() - t0_dns, 0.22)

                # (d) Multiple failing address attempts in _DeadlineHTTPSConnection.connect() stop when max_total_seconds expires
                connect_attempts = 0
                class _SlowFailingSocket:
                    def __init__(self, af, socktype, proto):
                        self.timeout = None
                    def settimeout(self, t):
                        self.timeout = t
                    def bind(self, addr):
                        pass
                    def connect(self, sa):
                        nonlocal connect_attempts
                        connect_attempts += 1
                        time.sleep(0.03)
                        raise OSError("simulated_address_unreachable")
                    def shutdown(self, how):
                        pass
                    def close(self):
                        pass

                five_addrs = [
                    (_sock_mod.AF_INET, _sock_mod.SOCK_STREAM, 6, "", (f"192.0.2.{i}", 443))
                    for i in range(1, 6)
                ]
                with self.assertRaisesRegex(TimeoutError, "webseed_total_deadline_exceeded"):
                    fetch_bundle_via_bep19_webseed(
                        strict_bundle.info_bytes,
                        "https://multi.example.org/seed",
                        expected_infohash_20=strict_bundle.infohash_20,
                        timeout=1.0,
                        max_total_seconds=0.065,
                        resolver_fn=lambda h, p: five_addrs,
                        socket_factory=_SlowFailingSocket,
                        tls_wrap=False,
                    )
                self.assertLessEqual(connect_attempts, 3)
            finally:
                dht.close()

        asyncio.run(_run_checks())


if __name__ == "__main__":
    unittest.main()
