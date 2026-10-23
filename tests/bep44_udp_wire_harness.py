#!/usr/bin/env python3
"""bep44_udp_wire_harness.py

Self-contained asyncio UDP Kademlia BEP-44 / BEP-46 wire harness on 127.0.0.1.
Implements:
  - Deterministic BEP-3 / BEP-44 bencode / bdecode without external dependencies.
  - Canonical BEP-44 signing buffer ([3:casi<cas>e][4:salt<len>:<salt>]3:seqi<seq>e1:v<len>:<v>).
  - SHA-1 target derivation: target = SHA1(pubkey_32 || salt).
  - Ed25519 signature verification via cryptography.hazmat.primitives.asymmetric.ed25519.
  - Strict 1000-byte BEP-44 value limit (KRPC error 205).
  - Strict CAS check (KRPC error 301) and monotonic seq enforcement (KRPC error 302).
  - Content-addressed BEP-46 encrypted bundle transfer over localhost UDP.
  - 12-node LocalKademliaSwarm with XOR-distance routing and ArmigerDHTDatabaseSync integration.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import zlib
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


# ---------------------------------------------------------------------------
# Deterministic BEP-3 / BEP-44 Bencode Codec
# ---------------------------------------------------------------------------

def bencode(value: Any) -> bytes:
    """Deterministically bencode int, bytes, str, list, or dict."""
    if isinstance(value, bool):
        return b"i1e" if value else b"i0e"
    if isinstance(value, int):
        return f"i{value}e".encode("ascii")
    if isinstance(value, bytes):
        return f"{len(value)}:".encode("ascii") + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return f"{len(raw)}:".encode("ascii") + raw
    if isinstance(value, (list, tuple)):
        return b"l" + b"".join(bencode(item) for item in value) + b"e"
    if isinstance(value, dict):
        items: List[Tuple[bytes, Any]] = []
        for k, v in value.items():
            kb = k if isinstance(k, bytes) else str(k).encode("utf-8")
            items.append((kb, v))
        items.sort(key=lambda pair: pair[0])
        out = bytearray(b"d")
        for kb, v in items:
            out.extend(bencode(kb))
            out.extend(bencode(v))
        out.extend(b"e")
        return bytes(out)
    raise TypeError(f"Unsupported bencode type: {type(value)!r}")


def _bdecode_at(data: bytes, idx: int) -> Tuple[Any, int]:
    if idx >= len(data):
        raise ValueError("Unexpected EOF in bdecode")
    ch = data[idx:idx + 1]
    if ch == b"i":
        end = data.index(b"e", idx + 1)
        num_bytes = data[idx + 1:end]
        if not num_bytes or (num_bytes.startswith(b"0") and len(num_bytes) > 1) or num_bytes == b"-0":
            raise ValueError("Invalid bencoded integer")
        return int(num_bytes.decode("ascii")), end + 1
    if ch == b"l":
        idx += 1
        arr: List[Any] = []
        while data[idx:idx + 1] != b"e":
            val, idx = _bdecode_at(data, idx)
            arr.append(val)
        return arr, idx + 1
    if ch == b"d":
        idx += 1
        out: Dict[bytes, Any] = {}
        prev_key: Optional[bytes] = None
        while data[idx:idx + 1] != b"e":
            key, idx = _bdecode_at(data, idx)
            if not isinstance(key, bytes):
                raise ValueError("Bencoded dictionary key must be bytes")
            if prev_key is not None and key < prev_key:
                raise ValueError("Non-canonical unsorted bencoded dictionary keys")
            prev_key = key
            val, idx = _bdecode_at(data, idx)
            out[key] = val
        return out, idx + 1
    if b"0" <= ch <= b"9":
        colon = data.index(b":", idx)
        length = int(data[idx:colon].decode("ascii"))
        if length < 0:
            raise ValueError("Negative byte string length")
        start = colon + 1
        end = start + length
        if end > len(data):
            raise ValueError("Truncated bencoded byte string")
        return data[start:end], end
    raise ValueError(f"Invalid bencode token at offset {idx}: {ch!r}")


def bdecode(data: bytes) -> Any:
    """Decode canonical bencoded bytes."""
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("Empty or non-bytes input to bdecode")
    val, idx = _bdecode_at(bytes(data), 0)
    if idx != len(data):
        raise ValueError("Trailing bytes after bencoded payload")
    return val


def bep44_signing_buffer(
    v: bytes,
    seq: int,
    salt: bytes = b"",
    cas: Optional[int] = None,
) -> bytes:
    """Construct the canonical BEP-44 signing buffer per BEP-44 specification:
    [4:salt<len>:<salt>]3:seqi<seq>e1:v<len>:<v>
    (Note: cas is a KRPC put precondition, never part of the signed buffer so get callers can verify sig.)
    """
    parts = bytearray()
    if salt:
        parts.extend(b"4:salt" + bencode(salt))
    parts.extend(b"3:seq" + bencode(int(seq)))
    parts.extend(b"1:v" + bencode(v))
    return bytes(parts)


def compute_bep44_target(pubkey_32: bytes, salt: bytes = b"") -> bytes:
    """BEP-44 target = SHA-1(pubkey_32 || salt)."""
    if len(pubkey_32) != 32:
        raise ValueError("Ed25519 public key must be 32 bytes")
    return hashlib.sha1(pubkey_32 + salt).digest()


# ---------------------------------------------------------------------------
# Stored BEP-44 Item & UDP Kademlia Node
# ---------------------------------------------------------------------------

@dataclass
class StoredItem:
    v: bytes
    seq: int
    sig: bytes
    pubkey: bytes
    salt: bytes


class BEP44UDPNode:
    """Asyncio UDP Kademlia node implementing BEP-44 mutable put/get and BEP-46 bundle storage."""

    MAX_V_BYTES = 1000

    def __init__(self, host: str = "127.0.0.1", port: int = 0, node_id: Optional[bytes] = None) -> None:
        self.host = host
        self.port = port
        self.node_id: bytes = node_id if (node_id and len(node_id) == 20) else os.urandom(20)
        self._store: Dict[bytes, StoredItem] = {}
        self._bundles: Dict[bytes, bytes] = {}
        self.routing_table: List[Tuple[bytes, str, int]] = []
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[asyncio.DatagramProtocol] = None

    async def start(self) -> Tuple[str, int]:
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self),
            local_addr=(self.host, self.port),
        )
        self.transport = transport
        self.protocol = protocol
        sockname = transport.get_extra_info("sockname")
        self.host, self.port = sockname[0], int(sockname[1])
        return self.host, self.port

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
            self.transport = None

    @staticmethod
    def _verify_sig(
        a_dict: Dict[bytes, Any],
        v: bytes,
        seq: int,
        salt: bytes,
        cas: Optional[int],
        sig: bytes,
        pubkey: bytes,
    ) -> bool:
        if len(pubkey) != 32 or len(sig) != 64:
            return False
        try:
            pub = Ed25519PublicKey.from_public_bytes(pubkey)
        except Exception:
            return False
        for candidate_buf in (
            bep44_signing_buffer(v=v, seq=seq, salt=salt, cas=cas),
            bep44_signing_buffer(v=v, seq=seq, salt=salt, cas=None),
            bencode({k: val for k, val in a_dict.items() if k != b"sig"}),
        ):
            try:
                pub.verify(sig, candidate_buf)
                return True
            except InvalidSignature:
                continue
        return False

    def _send(self, payload: Dict[bytes, Any], addr: Tuple[str, int]) -> None:
        if self.transport is not None:
            self.transport.sendto(bencode(payload), addr)

    async def handle_message(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            msg = bdecode(data)
            if not isinstance(msg, dict):
                raise ValueError("Top-level KRPC message must be dict")
        except Exception:
            self._send({b"t": b"0", b"y": b"e", b"e": 203, b"e_detail": [203, b"Protocol Error: malformed bencode"]}, addr)
            return

        t = msg.get(b"t", b"0")
        y = msg.get(b"y")
        if y != b"q":
            return
        q = msg.get(b"q")
        a = msg.get(b"a", {})
        if not isinstance(a, dict):
            self._send({b"t": t, b"y": b"e", b"e": 203}, addr)
            return

        if q == b"ping":
            self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id}}, addr)
        elif q == b"find_node":
            import ipaddress, struct
            target = a.get(b"target", b"")
            padded = target.ljust(20, b"\x00")[:20]
            closest = sorted(
                self.routing_table,
                key=lambda item: int.from_bytes(bytes(x ^ y_b for x, y_b in zip(item[0], padded)), "big"),
            )[:8]
            nodes_list = [[nid, h.encode("ascii"), p] for nid, h, p in closest]
            r_dict: Dict[bytes, Any] = {b"id": self.node_id, b"nodes_list": nodes_list}
            want_raw = a.get(b"want", [])
            want_set = set(want_raw) if isinstance(want_raw, list) else set()
            if not want_set or b"n4" in want_set:
                n4_buf = bytearray()
                for nid, h, p in closest:
                    try:
                        ip4 = ipaddress.IPv4Address(h)
                    except ValueError:
                        ip4 = ipaddress.IPv4Address("127.0.0.1")
                    n4_buf.extend(nid + ip4.packed + struct.pack(">H", int(p)))
                r_dict[b"nodes"] = bytes(n4_buf)
            if b"n6" in want_set:
                n6_buf = bytearray()
                for nid, h, p in closest:
                    v6_str = getattr(self, "ipv6_host", "::1")
                    try:
                        ip6 = ipaddress.IPv6Address(h if ":" in h else v6_str)
                    except ValueError:
                        ip6 = ipaddress.IPv6Address("::1")
                    n6_buf.extend(nid + ip6.packed + struct.pack(">H", int(p)))
                r_dict[b"nodes6"] = bytes(n6_buf)
            self._send({b"t": t, b"y": b"r", b"r": r_dict}, addr)
        elif q == b"get":
            await self._handle_get(t, a, addr)
        elif q == b"put":
            await self._handle_put(t, a, addr)
        elif q == b"put_bundle":
            await self._handle_put_bundle(t, a, addr)
        elif q == b"get_bundle":
            await self._handle_get_bundle(t, a, addr)
        else:
            self._send({b"t": t, b"y": b"e", b"e": 204}, addr)

    async def _handle_get(self, t: bytes, a: Dict[bytes, Any], addr: Tuple[str, int]) -> None:
        target = a.get(b"target")
        if not isinstance(target, bytes) or len(target) != 20 or target not in self._store:
            self._send({b"t": t, b"y": b"e", b"e": 201}, addr)
            return
        item = self._store[target]
        resp = {
            b"t": t,
            b"y": b"r",
            b"v": item.v,
            b"seq": item.seq,
            b"sig": item.sig,
            b"k": item.pubkey,
            b"salt": item.salt,
            b"r": {
                b"id": self.node_id,
                b"v": item.v,
                b"seq": item.seq,
                b"sig": item.sig,
                b"k": item.pubkey,
                b"salt": item.salt,
            },
        }
        self._send(resp, addr)

    async def _handle_put(self, t: bytes, a: Dict[bytes, Any], addr: Tuple[str, int]) -> None:
        required = {b"k", b"v", b"seq", b"sig"}
        if not required.issubset(a.keys()):
            self._send({b"t": t, b"y": b"e", b"e": 203}, addr)
            return
        pubkey = a[b"k"]
        salt = a.get(b"salt", b"")
        v = a[b"v"]
        seq = a[b"seq"]
        sig = a[b"sig"]
        cas = a.get(b"cas")
        if (
            not isinstance(pubkey, bytes)
            or not isinstance(salt, bytes)
            or not isinstance(v, bytes)
            or not isinstance(seq, int)
            or not isinstance(sig, bytes)
            or len(salt) > 64
        ):
            self._send({b"t": t, b"y": b"e", b"e": 203}, addr)
            return

        if len(v) > self.MAX_V_BYTES:
            self._send({b"t": t, b"y": b"e", b"e": 205, b"e_detail": [205, b"message (v field) too big"]}, addr)
            return

        cas_int = int(cas) if isinstance(cas, int) else None
        if not self._verify_sig(a, v=v, seq=seq, salt=salt, cas=cas_int, sig=sig, pubkey=pubkey):
            self._send({b"t": t, b"y": b"e", b"e": 301, b"e_detail": [301, b"Invalid Ed25519 signature"]}, addr)
            return

        target = compute_bep44_target(pubkey, salt)
        existing = self._store.get(target)
        if existing is not None:
            if cas_int is not None and cas_int != existing.seq:
                self._send({b"t": t, b"y": b"e", b"e": 301, b"e_detail": [301, b"the CAS hash mismatched"]}, addr)
                return
            if seq <= existing.seq:
                self._send({b"t": t, b"y": b"e", b"e": 302, b"e_detail": [302, b"sequence number less than current"]}, addr)
                return

        self._store[target] = StoredItem(v=v, seq=seq, sig=sig, pubkey=pubkey, salt=salt)
        self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id}}, addr)

    async def _handle_put_bundle(self, t: bytes, a: Dict[bytes, Any], addr: Tuple[str, int]) -> None:
        ih = a.get(b"infohash")
        idx = a.get(b"piece", 0)
        total = a.get(b"total", 1)
        chunk = a.get(b"blob")
        if not isinstance(ih, bytes) or len(ih) != 20 or not isinstance(chunk, bytes) or len(chunk) > 6000:
            self._send({b"t": t, b"y": b"e", b"e": 205}, addr)
            return
        key = ih + f":{int(idx)}:{int(total)}".encode("ascii")
        self._bundles[key] = chunk
        self._bundles_meta = getattr(self, "_bundles_meta", {})
        self._bundles_meta[ih] = int(total)
        self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id, b"infohash": ih, b"piece": int(idx)}}, addr)

    async def _handle_get_bundle(self, t: bytes, a: Dict[bytes, Any], addr: Tuple[str, int]) -> None:
        ih = a.get(b"infohash")
        idx = a.get(b"piece", 0)
        meta = getattr(self, "_bundles_meta", {})
        if not isinstance(ih, bytes) or ih not in meta:
            self._send({b"t": t, b"y": b"e", b"e": 201}, addr)
            return
        total = meta[ih]
        key = ih + f":{int(idx)}:{int(total)}".encode("ascii")
        if key not in self._bundles:
            self._send({b"t": t, b"y": b"e", b"e": 201}, addr)
            return
        self._send(
            {b"t": t, b"y": b"r", b"r": {b"id": self.node_id, b"blob": self._bundles[key], b"piece": int(idx), b"total": int(total)}},
            addr,
        )


class _UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, node: BEP44UDPNode) -> None:
        self.node = node

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        asyncio.create_task(self.node.handle_message(data, addr))


async def udp_query(host: str, port: int, query: Dict[bytes, Any], timeout: float = 2.0) -> Dict[bytes, Any]:
    """Send a single bencoded KRPC query over UDP and await the decoded response."""
    loop = asyncio.get_running_loop()
    on_response: asyncio.Future[Dict[bytes, Any]] = loop.create_future()

    class _ClientProtocol(asyncio.DatagramProtocol):
        def datagram_received(self, data: bytes, _: Tuple[str, int]) -> None:
            if not on_response.done():
                try:
                    decoded = bdecode(data)
                    if not isinstance(decoded, dict):
                        raise ValueError("KRPC response must be a bencoded dict")
                    on_response.set_result(decoded)
                except Exception as exc:
                    on_response.set_exception(exc)

    transport, _ = await loop.create_datagram_endpoint(
        lambda: _ClientProtocol(),
        remote_addr=(host, port),
    )
    transport.sendto(bencode(query))
    try:
        return await asyncio.wait_for(on_response, timeout)
    finally:
        transport.close()


# ---------------------------------------------------------------------------
# 12-Node Local Kademlia Swarm & BEP-44/46 Integration
# ---------------------------------------------------------------------------

class LocalKademliaSwarm:
    """Manages a multi-node UDP Kademlia swarm on 127.0.0.1 with XOR-distance routing."""

    def __init__(self, count: int = 12, *, enforce_anonymity_guard: bool = False) -> None:
        self.count = count
        self.nodes: List[Tuple[str, int, BEP44UDPNode]] = []
        self.enforce_anonymity_guard: bool = bool(enforce_anonymity_guard)
        self.wire_put_packets: List[bytes] = []
        self.blocked_wire_mutations: int = 0

    async def start(self) -> None:
        for i in range(self.count):
            nid = hashlib.sha1(f"armiger-kademlia-node-{i}".encode("ascii")).digest()
            node = BEP44UDPNode(node_id=nid)
            host, port = await node.start()
            self.nodes.append((host, port, node))
        all_entries = [(n.node_id, h, p) for h, p, n in self.nodes]
        for _, _, node in self.nodes:
            node.routing_table = [entry for entry in all_entries if entry[0] != node.node_id]

    def close(self) -> None:
        for _, _, node in self.nodes:
            node.close()

    def closest_nodes(self, target_20: bytes, k: int = 4) -> List[Tuple[str, int, BEP44UDPNode]]:
        """Return the k closest nodes in the swarm by Kademlia XOR metric."""
        def xor_dist(item: Tuple[str, int, BEP44UDPNode]) -> int:
            nid = item[2].node_id
            return int.from_bytes(bytes(a ^ b for a, b in zip(nid, target_20)), "big")
        return sorted(self.nodes, key=xor_dist)[:k]

    async def put_mutable(
        self,
        privkey_bytes: bytes,
        salt: bytes,
        v: bytes,
        seq: int,
        cas: Optional[int] = None,
        fanout: int = 4,
    ) -> List[Dict[bytes, Any]]:
        """Sign and publish a BEP-44 mutable item to the k closest UDP nodes."""
        if self.enforce_anonymity_guard:
            from armiger.armiger_dht_rsi import AnonymityGuard, AnonymityViolationError
            guard = AnonymityGuard()
            try:
                guard.inspect_wire_payload(v)
            except AnonymityViolationError:
                self.blocked_wire_mutations += 1
                raise
        sk = Ed25519PrivateKey.from_private_bytes(privkey_bytes)
        pubkey = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        sig_buf = bep44_signing_buffer(v=v, seq=seq, salt=salt, cas=cas)
        sig = sk.sign(sig_buf)
        a_dict: Dict[bytes, Any] = {
            b"k": pubkey,
            b"salt": salt,
            b"v": v,
            b"seq": int(seq),
            b"sig": sig,
        }
        if cas is not None:
            a_dict[b"cas"] = int(cas)
        target = compute_bep44_target(pubkey, salt)
        peers = self.closest_nodes(target, k=fanout)
        responses: List[Dict[bytes, Any]] = []
        for idx, (host, port, _) in enumerate(peers):
            q = {b"t": f"p{idx}".encode("ascii"), b"y": b"q", b"q": b"put", b"a": a_dict}
            self.wire_put_packets.append(bencode(q))
            resp = await udp_query(host, port, q)
            responses.append(resp)
        return responses

    async def get_mutable(
        self,
        pubkey_bytes: bytes,
        salt: bytes,
        fanout: int = 4,
        *,
        max_transfer_seconds: float = 30.0,
    ) -> Optional[StoredItem]:
        """Query the k closest UDP nodes for a BEP-44 mutable item, verifying Ed25519 signatures."""
        if (
            isinstance(max_transfer_seconds, bool)
            or not isinstance(max_transfer_seconds, (int, float))
            or float(max_transfer_seconds) <= 0.0
        ):
            return None
        deadline = time.monotonic() + float(max_transfer_seconds)
        target = compute_bep44_target(pubkey_bytes, salt)
        peers = self.closest_nodes(target, k=fanout)
        best: Optional[StoredItem] = None
        for idx, (host, port, _) in enumerate(peers):
            rem = deadline - time.monotonic()
            if rem <= 0.0:
                break
            q = {b"t": f"g{idx}".encode("ascii"), b"y": b"q", b"q": b"get", b"a": {b"target": target}}
            try:
                resp = await udp_query(host, port, q, timeout=min(2.0, rem))
            except (asyncio.TimeoutError, OSError, ValueError):
                continue
            if not isinstance(resp, dict) or resp.get(b"y") != b"r":
                continue
            r_body = resp.get(b"r", resp)
            if not isinstance(r_body, dict):
                continue
            v = r_body.get(b"v")
            seq = r_body.get(b"seq")
            sig = r_body.get(b"sig")
            k_bytes = r_body.get(b"k", pubkey_bytes)
            if not isinstance(v, bytes) or not isinstance(seq, int) or not isinstance(sig, bytes):
                continue
            if k_bytes != pubkey_bytes:
                continue
            if not BEP44UDPNode._verify_sig(
                {b"k": k_bytes, b"salt": salt, b"v": v, b"seq": seq},
                v=v,
                seq=seq,
                salt=salt,
                cas=None,
                sig=sig,
                pubkey=pubkey_bytes,
            ):
                continue
            if best is None or seq > best.seq:
                best = StoredItem(v=v, seq=seq, sig=sig, pubkey=pubkey_bytes, salt=salt)
        return best

    CHUNK_SIZE = 4096

    async def put_bep46_bundle(self, blob: bytes, fanout: int = 4) -> str:
        """Store a content-addressed BEP-46 bundle in 4 KiB UDP pieces across the k closest nodes."""
        ih = hashlib.sha1(blob).digest()
        chunks = [blob[i:i + self.CHUNK_SIZE] for i in range(0, len(blob), self.CHUNK_SIZE)] or [b""]
        total = len(chunks)
        peers = self.closest_nodes(ih, k=fanout)
        for idx, (host, port, _) in enumerate(peers):
            for p_idx, chunk in enumerate(chunks):
                q = {
                    b"t": f"b{idx}_{p_idx}".encode("ascii"),
                    b"y": b"q",
                    b"q": b"put_bundle",
                    b"a": {b"infohash": ih, b"piece": p_idx, b"total": total, b"blob": chunk},
                }
                resp = await udp_query(host, port, q)
                if resp.get(b"y") != b"r":
                    raise RuntimeError(f"put_bundle piece {p_idx}/{total} failed on {host}:{port}: {resp!r}")
        return ih.hex()

    MAX_BUNDLE_PIECES: int = 4096
    MAX_BUNDLE_BYTES: int = 16 * 1024 * 1024
    MAX_TRANSFER_SECONDS: float = 30.0

    async def get_bep46_bundle(
        self,
        infohash_hex: str,
        expected_sha256_hex: Optional[str] = None,
        fanout: int = 4,
        *,
        max_pieces: int = MAX_BUNDLE_PIECES,
        max_bundle_bytes: int = MAX_BUNDLE_BYTES,
        max_transfer_seconds: float = MAX_TRANSFER_SECONDS,
    ) -> Optional[bytes]:
        """Fetch a content-addressed BEP-46 bundle in 4 KiB UDP pieces and verify SHA-1/SHA-256."""
        if (
            isinstance(max_pieces, bool)
            or not isinstance(max_pieces, int)
            or max_pieces < 1
            or isinstance(max_bundle_bytes, bool)
            or not isinstance(max_bundle_bytes, int)
            or max_bundle_bytes < 1
            or isinstance(max_transfer_seconds, bool)
            or not isinstance(max_transfer_seconds, (int, float))
            or float(max_transfer_seconds) <= 0.0
        ):
            raise ValueError("Invalid get_bep46_bundle size or timeout bounds")
        ih = bytes.fromhex(infohash_hex)
        peers = self.closest_nodes(ih, k=fanout)
        for idx, (host, port, _) in enumerate(peers):
            peer_deadline = time.monotonic() + float(max_transfer_seconds)
            rem0 = peer_deadline - time.monotonic()
            if rem0 <= 0.0:
                continue
            q0 = {b"t": f"f{idx}_0".encode("ascii"), b"y": b"q", b"q": b"get_bundle", b"a": {b"infohash": ih, b"piece": 0}}
            try:
                resp0 = await udp_query(host, port, q0, timeout=min(2.0, rem0))
            except (asyncio.TimeoutError, OSError, ValueError):
                continue
            if not isinstance(resp0, dict) or resp0.get(b"y") != b"r":
                continue
            r0 = resp0.get(b"r", {})
            if not isinstance(r0, dict):
                continue
            c0 = r0.get(b"blob")
            raw_total = r0.get(b"total", 1)
            if (
                isinstance(raw_total, bool)
                or not isinstance(raw_total, int)
                or raw_total < 1
                or raw_total > max_pieces
            ):
                continue
            total = int(raw_total)
            if not isinstance(c0, bytes) or len(c0) > self.CHUNK_SIZE or (total > 1 and len(c0) == 0):
                continue
            cum_bytes = len(c0)
            if cum_bytes > max_bundle_bytes or (total - 1) * 1 + cum_bytes > max_bundle_bytes:
                continue
            pieces = [c0]
            ok_all = True
            for p_idx in range(1, total):
                rem_p = peer_deadline - time.monotonic()
                if rem_p <= 0.0:
                    ok_all = False
                    break
                qp = {b"t": f"f{idx}_{p_idx}".encode("ascii"), b"y": b"q", b"q": b"get_bundle", b"a": {b"infohash": ih, b"piece": p_idx}}
                try:
                    rp = await udp_query(host, port, qp, timeout=min(2.0, rem_p))
                except (asyncio.TimeoutError, OSError, ValueError):
                    ok_all = False
                    break
                if not isinstance(rp, dict) or rp.get(b"y") != b"r":
                    ok_all = False
                    break
                r_body = rp.get(b"r", {})
                if not isinstance(r_body, dict):
                    ok_all = False
                    break
                cp = r_body.get(b"blob")
                if not isinstance(cp, bytes) or len(cp) == 0 or len(cp) > self.CHUNK_SIZE:
                    ok_all = False
                    break
                cum_bytes += len(cp)
                if cum_bytes > max_bundle_bytes:
                    ok_all = False
                    break
                pieces.append(cp)
            if not ok_all:
                continue
            blob = b"".join(pieces)
            if hashlib.sha1(blob).digest() != ih:
                continue
            if expected_sha256_hex and hashlib.sha256(blob).hexdigest() != expected_sha256_hex.lower():
                continue
            return blob
        return None

    async def publish_shard_over_udp(
        self,
        manifest: Any,
        encrypted_bundle: bytes,
        root_privkey_bytes: bytes,
    ) -> str:
        """Publish an ArmigerDHTDatabaseSync shard + signed manifest over UDP BEP-44 + BEP-46."""
        envelope = zlib.compress(
            json.dumps(
                {
                    "manifest": {
                        "seq": manifest.seq,
                        "shard_id": manifest.shard_id,
                        "epoch": manifest.epoch,
                        "record_count": manifest.record_count,
                        "compressed_size_bytes": manifest.compressed_size_bytes,
                        "uncompressed_size_bytes": manifest.uncompressed_size_bytes,
                        "merkle_root": manifest.merkle_root,
                        "ciphertext_sha256": manifest.ciphertext_sha256,
                        "bep46_infohash": manifest.bep46_infohash,
                        "timestamp": manifest.timestamp,
                        "root_pubkey_hex": manifest.root_pubkey_hex,
                        "ed25519_signature_hex": manifest.ed25519_signature_hex,
                        "root_pq_pubkey_hex": manifest.root_pq_pubkey_hex,
                        "pq_signature_hex": manifest.pq_signature_hex,
                        "attestations": [
                            dataclasses.asdict(a)
                            for a in manifest.attestations
                        ],
                    },
                    "ciphertext_hex": encrypted_bundle.hex(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            level=9,
        )
        ih_hex = await self.put_bep46_bundle(envelope)
        pointer_v = json.dumps(
            {
                "ih": ih_hex,
                "sha256": hashlib.sha256(envelope).hexdigest(),
                "shard_id": manifest.shard_id.lower(),
                "seq": manifest.seq,
                "merkle_root": manifest.merkle_root,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert len(pointer_v) <= BEP44UDPNode.MAX_V_BYTES
        salt = manifest.shard_id.strip().lower().encode("utf-8")
        responses = await self.put_mutable(
            privkey_bytes=root_privkey_bytes,
            salt=salt,
            v=pointer_v,
            seq=manifest.seq,
        )
        ok_responses = [r for r in responses if isinstance(r, dict) and r.get(b"y") == b"r"]
        if not ok_responses:
            err_details = [r.get(b"e") for r in responses if isinstance(r, dict) and b"e" in r]
            raise RuntimeError(
                f"publish_shard_over_udp mutable pointer rejected by all peers (seq={manifest.seq}): {err_details or responses!r}"
            )
        return ih_hex

    async def fetch_and_apply_shard_over_udp(
        self,
        subscriber_sync: Any,
        root_pubkey_bytes: bytes,
        shard_id: str,
    ) -> Tuple[bool, str, int]:
        """Fetch a shard pointer via BEP-44 UDP, retrieve its BEP-46 envelope via UDP, and apply it."""
        from armiger.armiger_dht_db_sync import DHTShardManifest, QuorumAttestation, decompress_payload, decompress_payload

        salt = shard_id.strip().lower().encode("utf-8")
        item = await self.get_mutable(pubkey_bytes=root_pubkey_bytes, salt=salt)
        if item is None:
            return False, "bep44_pointer_not_found", 0
        ptr = json.loads(item.v.decode("utf-8"))
        envelope = await self.get_bep46_bundle(ptr["ih"], expected_sha256_hex=ptr["sha256"])
        if envelope is None:
            return False, "bep46_bundle_not_found_or_corrupted", 0
        unpacked = json.loads(decompress_payload(envelope).decode("utf-8"))
        m_dict = unpacked["manifest"]
        atts = tuple(QuorumAttestation(**a) for a in m_dict.pop("attestations", []))
        manifest = DHTShardManifest(**m_dict, attestations=atts)
        ciphertext = bytes.fromhex(unpacked["ciphertext_hex"])
        return subscriber_sync.receive_and_apply_shard(manifest, ciphertext)
