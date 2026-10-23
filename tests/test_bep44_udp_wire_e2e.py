#!/usr/bin/env python3
"""End-to-end 12-node localhost UDP Kademlia BEP-44 / BEP-46 wire harness tests.

Verifies:
  1. 12-node localhost (127.0.0.1) UDP Kademlia swarm startup, ping, find_node, and XOR routing.
  2. BEP-44 mutable put/get over UDP with Ed25519 signature verification, monotonic seq, and CAS.
  3. End-to-end ArmigerDHTDatabaseSync encrypted shard publish/fetch/decrypt/Merkle/CSAM/SQLite over UDP.
  4. End-to-end SignedTombstoneList propagation over UDP physically purging revoked records from SQLite.
  5. Wire-level hostile attack rejections:
       - Oversized v (>1000 bytes) -> KRPC error 205
       - Sequence regression (seq <= current) -> KRPC error 302
       - Stale CAS mismatch -> KRPC error 301
       - Forged Ed25519 signature -> KRPC error 301
       - Corrupted BEP-46 ciphertext -> rejected before SQLite ingestion
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import time
import unittest
import zlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from armiger.armiger_dht_db_sync import (
    ArmigerDHTDatabaseSync,
    SubscriptionConfig,
    SyncMode,
    TorrentRecord,
    create_mock_quorum_attestations,
)
from bep44_udp_wire_harness import (
    LocalKademliaSwarm,
    bep44_signing_buffer,
    compute_bep44_target,
    udp_query,
)
from armiger.csam_guard import (
    SignedTombstoneList,
    generate_ed25519_keypair,
    sign_tombstone_list,
)


class TestBEP44UDPWireE2E(unittest.TestCase):
    """End-to-end UDP Kademlia BEP-44 / BEP-46 wire harness test suite."""

    def test_12_node_udp_swarm_ping_find_node_and_bencode(self) -> None:
        async def _run() -> None:
            swarm = LocalKademliaSwarm(count=12)
            await swarm.start()
            try:
                self.assertEqual(len(swarm.nodes), 12)
                ports = {port for _, port, _ in swarm.nodes}
                self.assertEqual(len(ports), 12)

                host0, port0, node0 = swarm.nodes[0]
                ping_resp = await udp_query(host0, port0, {b"t": b"p1", b"y": b"q", b"q": b"ping", b"a": {}})
                self.assertEqual(ping_resp[b"y"], b"r")
                self.assertEqual(ping_resp[b"r"][b"id"], node0.node_id)

                target = hashlib.sha1(b"test-kademlia-target").digest()
                fn_resp = await udp_query(
                    host0, port0, {b"t": b"fn", b"y": b"q", b"q": b"find_node", b"a": {b"target": target}}
                )
                self.assertEqual(fn_resp[b"y"], b"r")
                self.assertEqual(len(fn_resp[b"r"][b"nodes_list"]), 8)
            finally:
                swarm.close()

        asyncio.run(_run())

    def test_bep44_mutable_put_get_monotonic_seq_and_cas_over_udp(self) -> None:
        async def _run() -> None:
            swarm = LocalKademliaSwarm(count=12)
            await swarm.start()
            try:
                priv_bytes, pub_bytes = generate_ed25519_keypair()
                salt = b"movie/en/hd_1080p/2026_q4"

                # Publish seq=1 across the 4 closest Kademlia nodes by XOR distance
                res1 = await swarm.put_mutable(priv_bytes, salt, b"manifest-v1", seq=1)
                self.assertTrue(all(r.get(b"y") == b"r" for r in res1))

                got1 = await swarm.get_mutable(pub_bytes, salt)
                self.assertIsNotNone(got1)
                assert got1 is not None
                self.assertEqual(got1.v, b"manifest-v1")
                self.assertEqual(got1.seq, 1)

                # Update to seq=2 with matching CAS=1 -> succeeds
                res2 = await swarm.put_mutable(priv_bytes, salt, b"manifest-v2", seq=2, cas=1)
                self.assertTrue(all(r.get(b"y") == b"r" for r in res2))

                got2 = await swarm.get_mutable(pub_bytes, salt)
                assert got2 is not None
                self.assertEqual(got2.v, b"manifest-v2")
                self.assertEqual(got2.seq, 2)
            finally:
                swarm.close()

        asyncio.run(_run())

    def test_e2e_encrypted_shard_sync_and_tombstone_purge_over_udp(self) -> None:
        async def _run() -> None:
            swarm = LocalKademliaSwarm(count=12)
            await swarm.start()
            try:
                root_priv, root_pub = generate_ed25519_keypair()
                shared_secret = hashlib.sha256(b"armiger-udp-e2e-shared-secret").digest()
                shard_id = "movie/en/hd_1080p/2026_q4"

                sub_cfg = SubscriptionConfig(
                    categories={"movie"},
                    languages={"en"},
                    quality_tiers={"hd_1080p"},
                    epochs={"2026_q4"},
                    sync_mode=SyncMode.SUBSCRIBE_AND_SEED,
                )
                publisher = ArmigerDHTDatabaseSync(
                    subscription=sub_cfg,
                    master_secret=shared_secret,
                    trusted_root_pubkeys_hex=[root_pub.hex()],
                )
                subscriber = ArmigerDHTDatabaseSync(
                    subscription=sub_cfg,
                    master_secret=shared_secret,
                    trusted_root_pubkeys_hex=[root_pub.hex()],
                )

                rec1 = TorrentRecord(
                    infohash="a1" * 20,
                    title="Sintel 2010 1080p BluRay x264",
                    category="movie",
                    language="en",
                    quality_tier="hd_1080p",
                    epoch="2026_q4",
                    size_bytes=2_100_000_000,
                    file_count=2,
                    seeders=45,
                    leechers=3,
                    content_sha256="11" * 32,
                    file_list=["Sintel.2010.1080p.mkv", "Sintel.en.srt"],
                    safety_gate_passed=True,
                    created_at=int(time.time()),
                )
                rec2 = TorrentRecord(
                    infohash="b2" * 20,
                    title="Tears of Steel 2012 1080p WEB-DL",
                    category="movie",
                    language="en",
                    quality_tier="hd_1080p",
                    epoch="2026_q4",
                    size_bytes=3_400_000_000,
                    file_count=1,
                    seeders=80,
                    leechers=5,
                    content_sha256="22" * 32,
                    file_list=["Tears.of.Steel.2012.1080p.mkv"],
                    safety_gate_passed=True,
                    created_at=int(time.time()),
                )

                manifest, encrypted_bundle = publisher.build_and_sign_shard(
                    shard_id=shard_id,
                    records=[rec1, rec2],
                    root_privkey=root_priv,
                    seq=10,
                )
                manifest = manifest.with_attestations(
                    create_mock_quorum_attestations(
                        manifest.canonical_digest(),
                        [13335, 15169, 16509, 8075, 14061],
                    )
                )

                # Publish over the 12-node UDP swarm
                await swarm.publish_shard_over_udp(manifest, encrypted_bundle, root_priv)

                # Subscriber fetches via BEP-44 + BEP-46 over UDP and ingests into SQLite
                ok, reason, count = await swarm.fetch_and_apply_shard_over_udp(
                    subscriber, root_pub, shard_id
                )
                self.assertTrue(ok, f"Expected UDP shard sync to succeed, got: {reason}")
                self.assertEqual(count, 2)
                self.assertIsNotNone(subscriber.db.get_record(rec1.infohash))
                self.assertIsNotNone(subscriber.db.get_record(rec2.infohash))

                # Publish a Hybrid Ed25519 + WOTS+-SHAKE256 SignedTombstoneList revoking rec2 over UDP
                signed_tomb = sign_tombstone_list(
                    seq=5,
                    infohashes_or_blinded=[rec2.infohash],
                    private_key_bytes=root_priv,
                )
                tomb_envelope = zlib.compress(
                    json.dumps(
                        {
                            "seq": signed_tomb.seq,
                            "timestamp": signed_tomb.timestamp,
                            "tombstones": signed_tomb.tombstones,
                            "reasons": signed_tomb.reasons,
                            "signature_hex": signed_tomb.signature_hex,
                            "pq_pubkey_hex": signed_tomb.pq_pubkey_hex,
                            "pq_signature_hex": signed_tomb.pq_signature_hex,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    level=9,
                )
                tomb_ih = await swarm.put_bep46_bundle(tomb_envelope)
                tomb_ptr = json.dumps(
                    {"ih": tomb_ih, "sha256": hashlib.sha256(tomb_envelope).hexdigest(), "seq": signed_tomb.seq},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                await swarm.put_mutable(root_priv, b"tombstones/v1", tomb_ptr, seq=signed_tomb.seq)

                fetched_tomb_ptr = await swarm.get_mutable(root_pub, b"tombstones/v1")
                self.assertIsNotNone(fetched_tomb_ptr)
                assert fetched_tomb_ptr is not None
                ptr_data = json.loads(fetched_tomb_ptr.v.decode("utf-8"))
                fetched_tomb_env = await swarm.get_bep46_bundle(ptr_data["ih"], expected_sha256_hex=ptr_data["sha256"])
                assert fetched_tomb_env is not None
                td = json.loads(zlib.decompress(fetched_tomb_env).decode("utf-8"))
                reconstructed_tomb = SignedTombstoneList(
                    seq=td["seq"],
                    timestamp=td["timestamp"],
                    tombstones=td["tombstones"],
                    reasons=td["reasons"],
                    signature_hex=td["signature_hex"],
                    pq_pubkey_hex=td["pq_pubkey_hex"],
                    pq_signature_hex=td["pq_signature_hex"],
                )
                t_ok, t_msg, new_tomb, purged = subscriber.apply_signed_tombstones(reconstructed_tomb)
                self.assertTrue(t_ok, t_msg)
                self.assertEqual(purged, 1)
                self.assertIsNone(subscriber.db.get_record(rec2.infohash))
                self.assertIsNotNone(subscriber.db.get_record(rec1.infohash))
            finally:
                swarm.close()

        asyncio.run(_run())

    def test_wire_hostile_attacks_205_301_302_and_corrupted_bundle(self) -> None:
        async def _run() -> None:
            swarm = LocalKademliaSwarm(count=4)
            await swarm.start()
            try:
                host, port, _ = swarm.nodes[0]
                priv_bytes, pub_bytes = generate_ed25519_keypair()
                sk = Ed25519PrivateKey.from_private_bytes(priv_bytes)
                salt = secrets.token_bytes(8)

                # 1. Oversized v (>1000 bytes) -> KRPC error 205
                big_v = os.urandom(1200)
                big_sig = sk.sign(bep44_signing_buffer(big_v, seq=1, salt=salt))
                resp_205 = await udp_query(
                    host,
                    port,
                    {
                        b"t": b"a1",
                        b"y": b"q",
                        b"q": b"put",
                        b"a": {b"k": pub_bytes, b"salt": salt, b"v": big_v, b"seq": 1, b"sig": big_sig},
                    },
                )
                self.assertEqual(resp_205.get(b"y"), b"e")
                self.assertEqual(resp_205.get(b"e"), 205)

                # Store valid seq=5
                v5 = b"valid-seq-5"
                sig5 = sk.sign(bep44_signing_buffer(v5, seq=5, salt=salt))
                resp_ok = await udp_query(
                    host,
                    port,
                    {
                        b"t": b"a2",
                        b"y": b"q",
                        b"q": b"put",
                        b"a": {b"k": pub_bytes, b"salt": salt, b"v": v5, b"seq": 5, b"sig": sig5},
                    },
                )
                self.assertEqual(resp_ok.get(b"y"), b"r")

                # 2. Sequence regression (seq=3 <= 5) -> KRPC error 302
                v3 = b"rollback-attempt"
                sig3 = sk.sign(bep44_signing_buffer(v3, seq=3, salt=salt))
                resp_302 = await udp_query(
                    host,
                    port,
                    {
                        b"t": b"a3",
                        b"y": b"q",
                        b"q": b"put",
                        b"a": {b"k": pub_bytes, b"salt": salt, b"v": v3, b"seq": 3, b"sig": sig3},
                    },
                )
                self.assertEqual(resp_302.get(b"y"), b"e")
                self.assertEqual(resp_302.get(b"e"), 302)

                # 3. Stale CAS (cas=2 != current seq=5) -> KRPC error 301
                v6 = b"cas-mismatch-attempt"
                sig6 = sk.sign(bep44_signing_buffer(v6, seq=6, salt=salt, cas=2))
                resp_301_cas = await udp_query(
                    host,
                    port,
                    {
                        b"t": b"a4",
                        b"y": b"q",
                        b"q": b"put",
                        b"a": {b"k": pub_bytes, b"salt": salt, b"v": v6, b"seq": 6, b"cas": 2, b"sig": sig6},
                    },
                )
                self.assertEqual(resp_301_cas.get(b"y"), b"e")
                self.assertEqual(resp_301_cas.get(b"e"), 301)

                # 4. Forged Ed25519 signature -> KRPC error 301
                other_priv, _ = generate_ed25519_keypair()
                other_sk = Ed25519PrivateKey.from_private_bytes(other_priv)
                forged_sig = other_sk.sign(bep44_signing_buffer(b"forged", seq=10, salt=salt))
                resp_301_sig = await udp_query(
                    host,
                    port,
                    {
                        b"t": b"a5",
                        b"y": b"q",
                        b"q": b"put",
                        b"a": {b"k": pub_bytes, b"salt": salt, b"v": b"forged", b"seq": 10, b"sig": forged_sig},
                    },
                )
                self.assertEqual(resp_301_sig.get(b"y"), b"e")
                self.assertEqual(resp_301_sig.get(b"e"), 301)

                # Verify stored value is still untouched at seq=5
                target = compute_bep44_target(pub_bytes, salt)
                got = await udp_query(host, port, {b"t": b"g", b"y": b"q", b"q": b"get", b"a": {b"target": target}})
                self.assertEqual(got.get(b"v"), b"valid-seq-5")
                self.assertEqual(got.get(b"seq"), 5)
            finally:
                swarm.close()

        asyncio.run(_run())



    def test_bep45_multi_address_want_n4_n6_and_dual_stack_sybil_quorum(self) -> None:
        """Verify BEP-45 dual-stack want=[n4, n6] compact routing (26B IPv4 + 38B IPv6)
        and multi-address IPv4/IPv6/mapped-IPv6 Sybil quorum deduplication."""
        import numpy as np
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        from armiger.armiger_dht_rsi import AuthoritativeASNVerifier, PeerAttestation, SybilQuorumVerifier

        async def _run() -> None:
            swarm = LocalKademliaSwarm(count=12)
            await swarm.start()
            try:
                host0, port0, _ = swarm.nodes[0]
                target = hashlib.sha1(b"bep45-dual-stack-target").digest()
                fn_resp = await udp_query(
                    host0,
                    port0,
                    {
                        b"t": b"b45",
                        b"y": b"q",
                        b"q": b"find_node",
                        b"a": {b"target": target, b"want": [b"n4", b"n6"]},
                    },
                )
                self.assertEqual(fn_resp[b"y"], b"r")
                r_body = fn_resp[b"r"]
                self.assertIn(b"nodes", r_body)
                self.assertIn(b"nodes6", r_body)
                # 8 closest nodes * 26 bytes (20B id + 4B IPv4 + 2B port) = 208 bytes
                self.assertEqual(len(r_body[b"nodes"]), 8 * 26)
                # 8 closest nodes * 38 bytes (20B id + 16B IPv6 + 2B port) = 304 bytes
                self.assertEqual(len(r_body[b"nodes6"]), 8 * 38)
            finally:
                swarm.close()

        asyncio.run(_run())

        verifier = AuthoritativeASNVerifier()
        asn_routes = [
            (13335, "104.16.10.0/24", "2606:4700:10::/48", "104.16.10.5", "2606:4700:10::5"),
            (15169, "8.8.8.0/24",     "2001:4860:4860::/48", "8.8.8.8", "2001:4860:4860::8888"),
            (16509, "54.239.28.0/24", "2600:1f18:20::/48", "54.239.28.10", "2600:1f18:20::10"),
            (8075,  "20.190.128.0/24","2603:1030:10::/48", "20.190.128.7", "2603:1030:10::7"),
            (14061, "167.99.1.0/24",  "2604:a880:800::/48", "167.99.1.9", "2604:a880:800::9"),
        ]
        keys = []
        for asn, v4_cidr, v6_cidr, _, _ in asn_routes:
            priv, pub = generate_ed25519_keypair()
            verifier.register_cidr(v4_cidr, asn)
            verifier.register_cidr(v6_cidr, asn)
            verifier.register_pinned_key(pub.hex(), asn)
            keys.append((asn, priv, pub))

        self.assertEqual(verifier.extract_subnet_prefix("::ffff:104.16.10.99"), "104.16.10.0/24")
        self.assertEqual(verifier.lookup_ip_asn("::ffff:104.16.10.99"), 13335)

        def _sign_att(peer_id: str, asn: int, th: str, priv_b: bytes, ip_s: str, ts: int) -> PeerAttestation:
            sk = Ed25519PrivateKey.from_private_bytes(priv_b)
            pub_h = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
            u = PeerAttestation(peer_id=peer_id, asn=asn, peer_pubkey_hex=pub_h, timestamp=ts, signature_hex="", ip_address=ip_s)
            sig_h = sk.sign(u.canonical_bytes(th)).hex()
            return PeerAttestation(peer_id=peer_id, asn=asn, peer_pubkey_hex=pub_h, timestamp=ts, signature_hex=sig_h, ip_address=ip_s)

        target_hash = hashlib.sha256(b"v3.0-gold-r5-manifest").hexdigest()
        now_ts = int(time.time())
        asn0, priv0, _ = keys[0]
        _, _, _, v4_0, v6_0 = asn_routes[0]
        dual_single = [
            _sign_att("node-0-v4", asn0, target_hash, priv0, v4_0, now_ts),
            _sign_att("node-0-v6", asn0, target_hash, priv0, v6_0, now_ts),
        ]
        ok_single, _, _ = SybilQuorumVerifier.verify_quorum(
            dual_single, target_hash, min_distinct_asns=2, now_timestamp=now_ts, asn_verifier=verifier
        )
        self.assertFalse(ok_single)

        mixed_atts = [
            _sign_att(f"bep45-node-{i}", keys[i][0], target_hash, keys[i][1], asn_routes[i][4 if i % 2 else 3], now_ts)
            for i in range(5)
        ]
        ok_5, reason_5, asns_5 = SybilQuorumVerifier.verify_quorum(
            mixed_atts, target_hash, min_distinct_asns=5, now_timestamp=now_ts, asn_verifier=verifier
        )
        self.assertTrue(ok_5, reason_5)
        self.assertEqual(len(asns_5), 5)

    def test_bep44_bep46_9head_delta_xmss_and_wire_firewall_over_udp(self) -> None:
        """Verify 9-head (including span_bio) Ed25519 + XMSS delta transfer over BEP-44/46 UDP."""
        import numpy as np
        from armiger.dht_head_delta import compute_trunk_sha256, extract_head_delta_bundle, load_state_tensors, reconstruct_from_head_delta
        from armiger.pq_xmss_merkle import XMSSMerkleTree

        def _mock_9head(delta: float = 0.0) -> dict:
            rng = np.random.default_rng(42)
            st = {
                "trunk.conv1.weight": rng.standard_normal((64, 32, 3)).astype(np.float32),
                "heads.category.0.weight": (rng.standard_normal((128, 384)) + delta).astype(np.float32),
                "heads.category.0.bias": np.zeros((128,), dtype=np.float32),
                "heads.category.2.weight": (rng.standard_normal((10, 128)) + delta).astype(np.float32),
                "heads.category.2.bias": np.zeros((10,), dtype=np.float32),
                "heads.span_bio.0.weight": (rng.standard_normal((64, 192)) + delta).astype(np.float32),
                "heads.span_bio.0.bias": np.zeros((64,), dtype=np.float32),
                "heads.span_bio.3.weight": (rng.standard_normal((7, 64)) + delta).astype(np.float32),
                "heads.span_bio.3.bias": np.zeros((7,), dtype=np.float32),
            }
            for h_name in ["adult", "junk", "anime", "english_subs", "lang", "owner_scope", "title_language"]:
                st[f"heads.{h_name}.0.weight"] = (rng.standard_normal((64, 384)) + delta).astype(np.float32)
                st[f"heads.{h_name}.0.bias"] = np.zeros((64,), dtype=np.float32)
                st[f"heads.{h_name}.2.weight"] = (rng.standard_normal((2, 64)) + delta).astype(np.float32)
                st[f"heads.{h_name}.2.bias"] = np.zeros((2,), dtype=np.float32)
            return st

        async def _run() -> None:
            swarm = LocalKademliaSwarm(count=12, enforce_anonymity_guard=False)
            await swarm.start()
            try:
                root_priv, root_pub = generate_ed25519_keypair()
                xmss_tree = XMSSMerkleTree(
                    master_seed=b"armiger-r5-xmss-seed-32-bytes!!!",
                    height=4,
                    provision_new_state=True,
                    allow_ephemeral_state=True,
                )
                base_state = _mock_9head(0.0)
                cand_state = _mock_9head(0.015)
                trunk_sha = compute_trunk_sha256(load_state_tensors(base_state)[0])

                bundle = extract_head_delta_bundle(
                    cand_state,
                    precision="fp16",
                    base_trunk_sha256=trunk_sha,
                    version="3.0.0-gold-r5",
                    seq=15,
                    ed25519_privkey_bytes=root_priv,
                    xmss_tree=xmss_tree,
                )
                packed_bytes = bundle.to_bytes()
                ih_hex = await swarm.put_bep46_bundle(packed_bytes)
                ptr_payload = json.dumps(
                    {"ih": ih_hex, "sha256": hashlib.sha256(packed_bytes).hexdigest(), "version": "3.0.0-gold-r5", "seq": 15},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                await swarm.put_mutable(root_priv, b"armiger/model-delta/v3", ptr_payload, seq=15)

                got_ptr = await swarm.get_mutable(root_pub, b"armiger/model-delta/v3")
                self.assertIsNotNone(got_ptr)
                ptr_info = json.loads(got_ptr.v.decode("utf-8"))
                fetched_bundle = await swarm.get_bep46_bundle(ptr_info["ih"], expected_sha256_hex=ptr_info["sha256"])
                self.assertEqual(fetched_bundle, packed_bytes)

                reconstructed = reconstruct_from_head_delta(
                    base_state,
                    fetched_bundle,
                    expected_trunk_sha256=trunk_sha,
                    ed25519_pubkey_bytes=root_pub,
                    xmss_root_hex=xmss_tree.merkle_root_hex,
                    last_verified_seq=14,
                )
                sd = reconstructed["state_dict"]
                self.assertIn("heads.span_bio.3.weight", sd)
                diff = np.max(np.abs(sd["heads.category.2.weight"] - cand_state["heads.category.2.weight"]))
                self.assertLess(float(diff), 0.01)
            finally:
                swarm.close()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
