"""Gated BEP-44/46 UDP Kademlia Node and BitTorrent Peer-Wire TCP Seeder."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from armiger.oci_sqlite_catalog import (  # noqa: F401
    BT_EXT_HANDSHAKE_ID,
    BT_METADATA_PIECE_SIZE,
    BT_MSG_BITFIELD,
    BT_MSG_EXTENDED,
    BT_MSG_INTERESTED,
    BT_MSG_PIECE,
    BT_MSG_REQUEST,
    BT_MSG_UNCHOKE,
    BT_PROTOCOL_HEADER,
    BT_RESERVED_BEP10,
    BT_UT_METADATA_ID,
    BitTorrentPeerWireSeeder,
    SeededTorrentBundle,
    fetch_bundle_via_bep19_webseed,
    fetch_bundle_via_peer_wire,
)
from armiger.oci_vpn_guard import (
    AnonymityGuard,
    VPNEgressGuard,
    _sign_bep44,
    _verify_bep44,
    bdecode,
    bdecode_krpc,
    bencode,
    bep44_signing_buffer,
    compute_bep44_target,
    generate_bep42_node_id,
    verify_bep42_node_id,
)


DEFAULT_DHT_BOOTSTRAP_NODES: Tuple[Tuple[str, int], ...] = (
    ("router.bittorrent.com", 6881),
    ("dht.transmissionbt.com", 6881),
    ("router.utorrent.com", 6881),
    ("dht.libtorrent.org", 25401),
)


def parse_bootstrap_nodes(raw: Optional[str] = None) -> List[Tuple[str, int]]:
    """Parse comma-separated `host:port` DHT bootstrap nodes or return `DEFAULT_DHT_BOOTSTRAP_NODES`."""
    val = raw if raw is not None else os.environ.get("ARMIGER_DHT_BOOTSTRAP_NODES", "")
    val = str(val).strip()
    if not val:
        return list(DEFAULT_DHT_BOOTSTRAP_NODES)
    if val.lower() in ("none", "off", "0", "false"):
        return []
    out: List[Tuple[str, int]] = []
    for item in val.split(","):
        tok = item.strip()
        if not tok or ":" not in tok:
            continue
        host, port_s = tok.rsplit(":", 1)
        host = host.strip()
        try:
            port = int(port_s.strip())
        except ValueError:
            continue
        if host and 0 < port <= 65535 and (host, port) not in out:
            out.append((host, port))
    return out if out else list(DEFAULT_DHT_BOOTSTRAP_NODES)


def _parse_compact_nodes(compact_bytes: Any) -> List[Tuple[bytes, str, int]]:
    """Parse BEP-5 compact 26-byte node info (`20B node_id + 4B IPv4 + 2B port`)."""
    if not isinstance(compact_bytes, (bytes, bytearray)) or len(compact_bytes) < 26:
        return []
    raw = bytes(compact_bytes)
    usable = (len(raw) // 26) * 26
    nodes: List[Tuple[bytes, str, int]] = []
    for off in range(0, usable, 26):
        nid = raw[off:off + 20]
        try:
            ip_str = socket.inet_ntoa(raw[off + 20:off + 24])
            (port,) = struct.unpack("!H", raw[off + 24:off + 26])
        except OSError:
            continue
        if 0 < port <= 65535:
            nodes.append((nid, ip_str, port))
    return nodes


def _xor_distance(a: bytes, b: bytes) -> int:
    return int.from_bytes(a[:20].ljust(20, b"\x00"), "big") ^ int.from_bytes(b[:20].ljust(20, b"\x00"), "big")


_KRPC_DNS_SEMAPHORE = threading.BoundedSemaphore(4)


class GatedBEP44UDPNode:
    """UDP Kademlia DHT node implementing BEP-44 mutable put/get, BEP-46 bundle storage,
    iterative XOR-distance bootstrap/publication, and `get_peers`/`announce_peer`, strictly
    gated by `VPNEgressGuard` with bounded storage, TTL eviction, per-IP rate limits, and write tokens.
    """
    MAX_V_BYTES = 1000
    MAX_MUTABLE_ITEMS = 256
    MAX_BUNDLE_PIECES = 256
    MAX_INFOHASHES = 256
    MAX_PEERS_PER_IH = 64
    MAX_ROUTING_NODES = 128
    ITEM_TTL_SEC = 7200.0
    MAX_WRITES_PER_IP_PER_MIN = 120
    MAX_TRACKED_WRITE_IPS = 4096
    MAX_CONCURRENT_HANDLERS = 64

    def __init__(
        self,
        vpn_guard: VPNEgressGuard,
        host: str = "0.0.0.0",
        port: int = 6881,
        node_id: Optional[bytes] = None,
        require_write_token: bool = True,
        anonymity_guard: Optional[AnonymityGuard] = None,
        enforce_bep42_routing: bool = False,
    ) -> None:
        self.vpn_guard = vpn_guard
        self.anonymity_guard = anonymity_guard or AnonymityGuard()
        self.host = host
        self.port = int(port)
        self.node_id: bytes = node_id if (node_id and len(node_id) == 20) else hashlib.sha1(os.urandom(32)).digest()
        self.require_write_token = bool(require_write_token)
        self.enforce_bep42_routing = bool(enforce_bep42_routing)
        self._token_secret: bytes = os.urandom(32)
        self._store: Dict[bytes, Dict[str, Any]] = {}
        self._bundles: Dict[bytes, Tuple[bytes, float]] = {}
        self._bundles_meta: Dict[bytes, int] = {}
        self._peers: Dict[bytes, List[Tuple[str, int, float]]] = {}
        self.routing_table: List[Tuple[bytes, str, int]] = []
        self._write_Window: Dict[str, Tuple[int, int]] = {}
        self._pending_rpcs: Dict[Tuple[bytes, str, int], asyncio.Future[bytes]] = {}
        self._active_handlers: int = 0
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.queries_handled: int = 0
        self.vpn_guard.register_on_drop(lambda _st: self.close())

    def make_token(self, ip: str, bucket_offset: int = 0) -> bytes:
        bucket = int(time.time() // 300) + int(bucket_offset)
        msg = f"{ip}:{bucket}".encode("utf-8")
        return hmac.new(self._token_secret, msg, hashlib.sha256).digest()[:16]

    def verify_token(self, ip: str, token: Any) -> bool:
        if not isinstance(token, bytes) or len(token) != 16:
            return False
        for offset in (0, -1):
            if hmac.compare_digest(token, self.make_token(ip, offset)):
                return True
        return False

    def _allow_write_from_ip(self, ip: str) -> bool:
        minute_bucket = int(time.time() // 60)
        if ip not in self._write_Window and len(self._write_Window) >= self.MAX_TRACKED_WRITE_IPS:
            stale = [k for k, (b, _) in self._write_Window.items() if b != minute_bucket]
            for k in stale:
                self._write_Window.pop(k, None)
            if len(self._write_Window) >= self.MAX_TRACKED_WRITE_IPS:
                oldest_ip = next(iter(self._write_Window))
                self._write_Window.pop(oldest_ip, None)
        prev_bucket, count = self._write_Window.get(ip, (minute_bucket, 0))
        if prev_bucket != minute_bucket:
            count = 0
        if count >= self.MAX_WRITES_PER_IP_PER_MIN:
            return False
        self._write_Window[ip] = (minute_bucket, count + 1)
        return True

    def _evict_expired(self) -> None:
        now = time.time()
        expired_targets = [
            k for k, v in self._store.items()
            if not v.get("pinned") and (now - float(v.get("ts", now))) > self.ITEM_TTL_SEC
        ]
        for k in expired_targets:
            self._store.pop(k, None)
        expired_bundles = [
            k for k, (_, ts) in self._bundles.items()
            if (now - ts) > self.ITEM_TTL_SEC
        ]
        for k in expired_bundles:
            self._bundles.pop(k, None)
        for ih in list(self._peers.keys()):
            kept = [(ip, pt, ts) for ip, pt, ts in self._peers[ih] if (now - ts) <= self.ITEM_TTL_SEC]
            if kept:
                self._peers[ih] = kept
            else:
                self._peers.pop(ih, None)

    def add_routing_node(self, node_id: bytes, ip: str, port: int) -> None:
        if len(node_id) != 20 or node_id == self.node_id or port <= 0 or port > 65535:
            return
        if self.enforce_bep42_routing and not verify_bep42_node_id(ip, node_id, allow_local=True):
            return
        self.routing_table = [n for n in self.routing_table if n[0] != node_id and (n[1], n[2]) != (ip, port)]
        self.routing_table.append((node_id, ip, int(port)))
        if len(self.routing_table) > self.MAX_ROUTING_NODES:
            self.routing_table = self.routing_table[-self.MAX_ROUTING_NODES:]

    def _closest_nodes_payload(self, target: bytes) -> Tuple[bytes, List[List[Any]]]:
        t20 = target[:20].ljust(20, b"\x00")
        closest = sorted(
            self.routing_table,
            key=lambda item: _xor_distance(item[0], t20),
        )[:8]
        nodes_list = [[nid, h.encode("ascii"), p] for nid, h, p in closest]
        compact_nodes = bytearray()
        for nid, h, p in closest:
            try:
                compact_nodes.extend(nid + socket.inet_aton(h) + struct.pack("!H", int(p)))
            except OSError:
                continue
        return bytes(compact_nodes), nodes_list

    async def start(self) -> Tuple[str, int]:
        self.vpn_guard.assert_egress_or_raise()
        if self.transport is not None:
            return self.host, self.port
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.setblocking(False)
        sock.bind((self.host, self.port))
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _GatedUDPProtocol(self),
            sock=sock,
        )
        self.transport = transport
        sockname = transport.get_extra_info("sockname")
        self.host, self.port = str(sockname[0]), int(sockname[1])
        return self.host, self.port

    def close(self) -> None:
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        for fut in list(self._pending_rpcs.values()):
            if not fut.done():
                fut.cancel()
        self._pending_rpcs.clear()

    def _send(self, payload: Dict[bytes, Any], addr: Tuple[str, int]) -> None:
        if not self.vpn_guard.require_for_packet():
            self.close()
            return
        if self.transport is not None:
            self.transport.sendto(bencode(payload), addr)

    def _send_error(self, t: bytes, code: int, message: bytes, addr: Tuple[str, int]) -> None:
        self._send(
            {
                b"t": t if isinstance(t, bytes) else b"0",
                b"y": b"e",
                b"e": int(code),
                b"e_detail": [int(code), message],
            },
            addr,
        )

    def publish_local_mutable(
        self,
        privkey_32: bytes,
        salt: bytes,
        v: Union[bytes, Dict[bytes, Any]],
        seq: int,
        presigned_pubkey_32: Optional[bytes] = None,
        presigned_sig_64: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Sign (or accept pre-signed) and store a pinned BEP-44 mutable pointer in the local DHT table."""
        self.vpn_guard.assert_egress_or_raise()
        self.anonymity_guard.assert_valid_or_raise(v, salt)
        encoded_v = bencode(v)
        if len(encoded_v) > self.MAX_V_BYTES:
            raise ValueError(f"BEP-44 encoded value exceeds {self.MAX_V_BYTES} bytes: {len(encoded_v)}")
        sig_buf = bep44_signing_buffer(v=v, seq=seq, salt=salt)
        if presigned_pubkey_32 is not None and presigned_sig_64 is not None:
            if not _verify_bep44(presigned_pubkey_32, presigned_sig_64, (sig_buf,)):
                raise ValueError("Pre-signed BEP-44 record failed Ed25519 verification")
            pubkey_32, sig_64 = bytes(presigned_pubkey_32), bytes(presigned_sig_64)
        else:
            pubkey_32, sig_64 = _sign_bep44(privkey_32, sig_buf)
        target = compute_bep44_target(pubkey_32, salt)
        self._store[target] = {
            "v": v,
            "seq": int(seq),
            "sig": sig_64,
            "pubkey": pubkey_32,
            "salt": salt,
            "ts": time.time(),
            "pinned": True,
        }
        return {
            "target_hex": target.hex(),
            "pubkey_hex": pubkey_32.hex(),
            "salt": salt.decode("utf-8", errors="replace"),
            "seq": int(seq),
            "sig_hex": sig_64.hex(),
        }

    async def rpc_call(
        self,
        peer_addr: Tuple[str, int],
        q: bytes,
        a: Dict[bytes, Any],
        timeout: float = 2.0,
    ) -> Dict[bytes, Any]:
        """Execute a single KRPC query against `peer_addr`, gated by `vpn_guard`."""
        if not self.vpn_guard.require_for_packet():
            raise RuntimeError("VPN egress unverified; blocking outbound DHT RPC")
        loop = asyncio.get_running_loop()
        rpc_deadline = loop.time() + max(0.001, float(timeout))

        def _rem_rpc_timeout() -> float:
            rem = rpc_deadline - loop.time()
            if rem <= 0:
                raise asyncio.TimeoutError("krpc_timeout")
            return rem

        tx_id = os.urandom(4)
        args = dict(a)
        args.setdefault(b"id", self.node_id)
        req = {b"t": tx_id, b"y": b"q", b"q": q, b"a": args}

        host_str = str(peer_addr[0]).strip()
        port_int = int(peer_addr[1])
        try:
            socket.inet_aton(host_str)
            resolved_ip = host_str
        except OSError:
            resolver_cb = getattr(self, "_resolver_fn", None) or socket.gethostbyname
            if not _KRPC_DNS_SEMAPHORE.acquire(blocking=False):
                raise asyncio.TimeoutError("krpc_dns_pool_saturated")
            dns_fut: asyncio.Future[str] = loop.create_future()

            def _dns_worker() -> None:
                try:
                    ip_res = str(resolver_cb(host_str))
                    loop.call_soon_threadsafe(
                        lambda: None if dns_fut.done() else dns_fut.set_result(ip_res)
                    )
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        lambda e=exc: None if dns_fut.done() else dns_fut.set_exception(e)
                    )
                finally:
                    _KRPC_DNS_SEMAPHORE.release()

            try:
                t_dns = threading.Thread(target=_dns_worker, daemon=True)
                t_dns.start()
            except BaseException:
                _KRPC_DNS_SEMAPHORE.release()
                raise
            resolved_ip = await asyncio.wait_for(dns_fut, timeout=_rem_rpc_timeout())
        target_addr = (resolved_ip, port_int)

        if self.transport is not None and len(self._pending_rpcs) < 128:
            key = (tx_id, resolved_ip, port_int)
            fut: asyncio.Future[bytes] = loop.create_future()
            self._pending_rpcs[key] = fut
            try:
                self.transport.sendto(bencode(req), target_addr)
                raw = await asyncio.wait_for(fut, timeout=_rem_rpc_timeout())
                resp = bdecode_krpc(raw)
                if not isinstance(resp, dict):
                    raise RuntimeError("Invalid non-dict KRPC response")
                r_dict = resp.get(b"r")
                if isinstance(r_dict, dict) and isinstance(r_dict.get(b"id"), bytes) and len(r_dict[b"id"]) == 20:
                    self.add_routing_node(r_dict[b"id"], resolved_ip, port_int)
                return resp
            finally:
                self._pending_rpcs.pop(key, None)

        fut_eph: asyncio.Future[bytes] = loop.create_future()

        class _ClientProto(asyncio.DatagramProtocol):
            def datagram_received(self, data: bytes, _addr: Tuple[str, int]) -> None:
                if not fut_eph.done():
                    fut_eph.set_result(data)

            def error_received(self, exc: Exception) -> None:
                if not fut_eph.done():
                    fut_eph.set_exception(exc)

        transport, _ = await asyncio.wait_for(
            loop.create_datagram_endpoint(_ClientProto, remote_addr=target_addr),
            timeout=_rem_rpc_timeout(),
        )
        try:
            transport.sendto(bencode(req))
            raw = await asyncio.wait_for(fut_eph, timeout=_rem_rpc_timeout())
            resp = bdecode_krpc(raw)
            if not isinstance(resp, dict):
                raise RuntimeError("Invalid non-dict KRPC response")
            r_dict = resp.get(b"r")
            if isinstance(r_dict, dict) and isinstance(r_dict.get(b"id"), bytes) and len(r_dict[b"id"]) == 20:
                self.add_routing_node(r_dict[b"id"], resolved_ip, port_int)
            return resp
        finally:
            transport.close()

    async def iterative_find_closest(
        self,
        target_20: bytes,
        entry_addrs: Sequence[Tuple[str, int]],
        mode: str = "get",
        alpha: int = 4,
        k: int = 8,
        max_rounds: int = 4,
    ) -> List[Dict[str, Any]]:
        """Perform a bounded iterative Kademlia XOR-distance lookup for `target_20`.
        `mode="get"` sends BEP-44 `get(target=target_20)`; `mode="get_peers"` sends BEP-5 `get_peers(info_hash=target_20)`.
        Returns up to `k` closest responsive nodes with their operation-specific write `token` and `cas` sequence.
        """
        self.vpn_guard.assert_egress_or_raise()
        candidates: Dict[Tuple[str, int], bytes] = {}
        for nid, nip, npt in self.routing_table:
            candidates[(nip, int(npt))] = nid
        for e_host, e_port in entry_addrs:
            candidates.setdefault((str(e_host), int(e_port)), hashlib.sha1(f"{e_host}:{e_port}".encode("utf-8")).digest())

        visited: Set[Tuple[str, int]] = set()
        responders: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for _round in range(max_rounds):
            unvisited = [addr for addr in candidates if addr not in visited]
            if not unvisited:
                break
            unvisited.sort(key=lambda addr: _xor_distance(candidates[addr], target_20))
            batch = unvisited[:max(1, int(alpha))]

            async def _probe(addr: Tuple[str, int]) -> None:
                visited.add(addr)
                try:
                    if mode == "get_peers":
                        resp = await self.rpc_call(addr, b"get_peers", {b"info_hash": target_20})
                    else:
                        resp = await self.rpc_call(addr, b"get", {b"target": target_20})
                    if resp.get(b"y") != b"r" or not isinstance(resp.get(b"r"), dict):
                        return
                    r_dict = resp[b"r"]
                    r_id = r_dict.get(b"id")
                    if not (isinstance(r_id, bytes) and len(r_id) == 20):
                        return
                    candidates[addr] = r_id
                    self.add_routing_node(r_id, addr[0], addr[1])
                    for entry in r_dict.get(b"nodes_list", []):
                        if (
                            isinstance(entry, list)
                            and len(entry) == 3
                            and isinstance(entry[0], bytes)
                            and len(entry[0]) == 20
                            and isinstance(entry[1], bytes)
                        ):
                            c_addr = (entry[1].decode("ascii", errors="ignore"), int(entry[2]))
                            candidates.setdefault(c_addr, entry[0])
                            self.add_routing_node(entry[0], c_addr[0], c_addr[1])
                    for nid, nip, npt in _parse_compact_nodes(r_dict.get(b"nodes", b"")):
                        c_addr = (nip, int(npt))
                        candidates.setdefault(c_addr, nid)
                        self.add_routing_node(nid, nip, int(npt))
                    tok = r_dict.get(b"token", b"")
                    cur_seq = r_dict.get(b"seq", resp.get(b"seq"))
                    responders[addr] = {
                        "addr": addr,
                        "node_id": r_id,
                        "token": tok if isinstance(tok, bytes) else b"",
                        "current_seq": int(cur_seq) if type(cur_seq) is int else None,
                        "r": r_dict,
                        "resp": resp,
                    }
                except Exception:
                    return

            await asyncio.gather(*(_probe(a) for a in batch))

        ranked = sorted(
            responders.values(),
            key=lambda item: (0 if item["token"] else 1, _xor_distance(item["node_id"], target_20)),
        )
        return ranked[:max(1, int(k))]

    async def bootstrap_and_publish(
        self,
        bootstrap_addrs: Sequence[Tuple[str, int]],
        privkey_32: bytes,
        bundles: Mapping[str, SeededTorrentBundle],
        salts_by_bundle: Mapping[str, bytes],
        seq: int,
        bt_port: int,
        announce_peers: bool = True,
    ) -> Dict[str, int]:
        """Bootstrap routing table from `bootstrap_addrs`, perform separate iterative Kademlia XOR
        lookups for each BEP-44 `target_20` and each `info_hash`, and publish non-equivocating
        BEP-44 mutable pointers (`put`) + optional BitTorrent seeder endpoints (`announce_peer`).
        """
        self.vpn_guard.assert_egress_or_raise()
        if not self.anonymity_guard.opt_in_sharing:
            return {"nodes_contacted": 0, "puts_ok": 0, "announces_ok": 0}
        contacted: Set[Tuple[str, int]] = set()
        resolved_bootstrap_addrs: List[Tuple[str, int]] = []
        loop = asyncio.get_running_loop()
        for b_host, b_port in bootstrap_addrs:
            try:
                addr_infos = await asyncio.wait_for(
                    loop.getaddrinfo(b_host, int(b_port), family=socket.AF_INET, type=socket.SOCK_DGRAM),
                    timeout=2.0,
                )
            except Exception:
                continue
            if not addr_infos:
                continue
            resolved_ip = str(addr_infos[0][4][0])
            target_addr = (resolved_ip, int(b_port))
            resolved_bootstrap_addrs.append(target_addr)
            try:
                resp = await self.rpc_call(target_addr, b"find_node", {b"target": self.node_id})
                contacted.add(target_addr)
                if resp.get(b"y") != b"r" or not isinstance(resp.get(b"r"), dict):
                    continue
                r_dict = resp[b"r"]
                r_id = r_dict.get(b"id")
                if isinstance(r_id, bytes) and len(r_id) == 20:
                    self.add_routing_node(r_id, resolved_ip, int(b_port))
                for entry in r_dict.get(b"nodes_list", []):
                    if (
                        isinstance(entry, list)
                        and len(entry) == 3
                        and isinstance(entry[0], bytes)
                        and len(entry[0]) == 20
                        and isinstance(entry[1], bytes)
                    ):
                        self.add_routing_node(entry[0], entry[1].decode("ascii", errors="ignore"), int(entry[2]))
                for nid, nip, npt in _parse_compact_nodes(r_dict.get(b"nodes", b"")):
                    self.add_routing_node(nid, nip, npt)
            except Exception:
                continue

        puts_ok = 0
        announces_ok = 0
        for bname, bobj in bundles.items():
            salt = salts_by_bundle[bname]
            pubkey_probe, _ = _sign_bep44(privkey_32, b"probe")
            target_20 = compute_bep44_target(pubkey_probe, salt)
            local_item = self._store.get(target_20)
            if local_item is not None and int(local_item["seq"]) == int(seq):
                v_payload = local_item["v"]
                pubkey_32 = local_item["pubkey"]
                sig_64 = local_item["sig"]
            else:
                v_payload = bencode(
                    {
                        b"bundle": bname.encode("utf-8"),
                        b"ih": bobj.infohash_20,
                        b"merkle_root": bobj.merkle_root_hex.encode("ascii"),
                        b"seq": int(seq),
                        b"tier": b"oci",
                    }
                )
                sig_buf = bep44_signing_buffer(v=v_payload, seq=seq, salt=salt)
                pubkey_32, sig_64 = _sign_bep44(privkey_32, sig_buf)

            if announce_peers and int(bt_port) > 0:
                closest_peer_nodes = await self.iterative_find_closest(
                    bobj.infohash_20,
                    resolved_bootstrap_addrs,
                    mode="get_peers",
                )
                for node_info in closest_peer_nodes:
                    contacted.add(node_info["addr"])
                    gp_tok = node_info["token"]
                    if not gp_tok:
                        continue
                    try:
                        an_resp = await self.rpc_call(
                            node_info["addr"],
                            b"announce_peer",
                            {b"info_hash": bobj.infohash_20, b"port": int(bt_port), b"token": gp_tok},
                        )
                        if an_resp.get(b"y") == b"r":
                            announces_ok += 1
                    except Exception:
                        continue

            closest_mutable_nodes = await self.iterative_find_closest(
                target_20,
                resolved_bootstrap_addrs,
                mode="get",
            )
            for node_info in closest_mutable_nodes:
                contacted.add(node_info["addr"])
                put_tok = node_info["token"]
                if not put_tok:
                    continue
                put_args: Dict[bytes, Any] = {
                    b"k": pubkey_32,
                    b"salt": salt,
                    b"seq": int(seq),
                    b"sig": sig_64,
                    b"token": put_tok,
                    b"v": v_payload,
                }
                cur_seq = node_info.get("current_seq")
                if type(cur_seq) is int and 0 <= cur_seq < int(seq):
                    put_args[b"cas"] = cur_seq
                try:
                    put_resp = await self.rpc_call(node_info["addr"], b"put", put_args)
                    if put_resp.get(b"y") == b"r":
                        puts_ok += 1
                except Exception:
                    continue
        return {"nodes_contacted": len(contacted), "puts_ok": puts_ok, "announces_ok": announces_ok}

    async def discover_mutable_and_peers(
        self,
        entry_addrs: Sequence[Tuple[str, int]],
        pubkey_32: bytes,
        salt: bytes,
    ) -> Dict[str, Any]:
        """Discover a signed BEP-44 mutable pointer and its active BitTorrent seeder peers across DHT nodes."""
        self.vpn_guard.assert_egress_or_raise()
        target_20 = compute_bep44_target(pubkey_32, salt)
        queue: List[Tuple[str, int]] = list(entry_addrs)
        visited: Set[Tuple[str, int]] = set()
        best_record: Optional[Dict[str, Any]] = None

        while queue and len(visited) < 24:
            addr = queue.pop(0)
            if addr in visited:
                continue
            visited.add(addr)
            try:
                fn_resp = await self.rpc_call(addr, b"find_node", {b"target": target_20})
                fn_r = fn_resp.get(b"r", {}) if isinstance(fn_resp.get(b"r"), dict) else {}
                for entry in fn_r.get(b"nodes_list", []):
                    if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[1], bytes):
                        cand = (entry[1].decode("ascii", errors="ignore"), int(entry[2]))
                        if cand not in visited and cand not in queue:
                            queue.append(cand)
                for _nid, nip, npt in _parse_compact_nodes(fn_r.get(b"nodes", b"")):
                    cand = (nip, int(npt))
                    if cand not in visited and cand not in queue:
                        queue.append(cand)
                g_resp = await self.rpc_call(addr, b"get", {b"target": target_20})
                r_body = g_resp.get(b"r") if isinstance(g_resp.get(b"r"), dict) else g_resp
                if isinstance(r_body, dict):
                    for entry in r_body.get(b"nodes_list", []):
                        if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[1], bytes):
                            cand = (entry[1].decode("ascii", errors="ignore"), int(entry[2]))
                            if cand not in visited and cand not in queue:
                                queue.append(cand)
                    for _nid, nip, npt in _parse_compact_nodes(r_body.get(b"nodes", b"")):
                        cand = (nip, int(npt))
                        if cand not in visited and cand not in queue:
                            queue.append(cand)
                if isinstance(r_body, dict) and b"v" in r_body and b"sig" in r_body and b"seq" in r_body:
                    v_item = r_body[b"v"]
                    seq_raw = r_body[b"seq"]
                    sig_bytes = r_body[b"sig"]
                    k_bytes = r_body.get(b"k", pubkey_32)
                    if (
                        type(seq_raw) is not int
                        or not (0 <= seq_raw <= 0x7FFFFFFF_FFFFFFFF)
                        or not isinstance(sig_bytes, (bytes, bytearray))
                        or len(sig_bytes) != 64
                        or not isinstance(k_bytes, (bytes, bytearray))
                        or len(k_bytes) != 32
                        or not isinstance(v_item, (bytes, bytearray, dict))
                    ):
                        continue
                    seq_val = int(seq_raw)
                    cand_bufs = [bep44_signing_buffer(v=v_item, seq=seq_val, salt=salt)]
                    if bytes(k_bytes) == pubkey_32 and _verify_bep44(pubkey_32, bytes(sig_bytes), cand_bufs):
                        decoded_v = v_item if isinstance(v_item, dict) else bdecode(v_item)
                        raw_v = bencode(v_item) if isinstance(v_item, dict) else bytes(v_item)
                        if best_record is None or seq_val > best_record["seq"]:
                            best_record = {
                                "v": decoded_v,
                                "v_raw": raw_v,
                                "seq": seq_val,
                                "sig": sig_bytes,
                                "source_node": addr,
                            }
            except Exception:
                continue

        if best_record is None:
            raise LookupError("Verified BEP-44 mutable pointer not found on DHT")
        ih_20 = best_record["v"][b"ih"]
        peers: List[Tuple[str, int]] = []
        peer_nodes = await self.iterative_find_closest(ih_20, list(visited) or list(entry_addrs), mode="get_peers")
        for n_info in peer_nodes:
            vals = n_info.get("r", {}).get(b"values", [])
            for c_peer in vals:
                if isinstance(c_peer, bytes) and len(c_peer) == 6:
                    p_ip = socket.inet_ntoa(c_peer[:4])
                    (p_port,) = struct.unpack("!H", c_peer[4:6])
                    if (p_ip, p_port) not in peers:
                        peers.append((p_ip, p_port))
        best_record["infohash_20"] = ih_20
        best_record["peers"] = peers
        return best_record

    async def handle_message(self, data: bytes, addr: Tuple[str, int]) -> None:
        if not self.vpn_guard.require_for_packet():
            self.close()
            return
        self._evict_expired()
        try:
            msg = bdecode_krpc(data)
            if not isinstance(msg, dict):
                raise ValueError("Top-level KRPC message must be dict")
        except Exception:
            self._send_error(b"0", 203, b"Protocol Error", addr)
            return
        t = msg.get(b"t", b"0")
        y = msg.get(b"y")
        if y in (b"r", b"e") and isinstance(t, bytes):
            fut = self._pending_rpcs.get((t, str(addr[0]), int(addr[1])))
            if fut is not None and not fut.done():
                fut.set_result(data)
            return
        if y != b"q":
            return
        q = msg.get(b"q")
        a = msg.get(b"a", {})
        if not isinstance(a, dict):
            self._send_error(t, 203, b"Protocol Error", addr)
            return

        # Validate query types before routing, dict indexing, and integer conversion.
        valid = isinstance(t, bytes) and 0 < len(t) <= 16 and isinstance(q, bytes)
        hash_key = b"info_hash" if q in (b"get_peers", b"announce_peer") else b"target"
        if q in (b"find_node", b"get", b"get_peers", b"announce_peer"):
            valid = valid and isinstance(a.get(hash_key), bytes) and len(a[hash_key]) == 20
        if q in (b"put_bundle", b"get_bundle"):
            idx = a.get(b"piece", 0)
            valid = valid and isinstance(a.get(b"infohash"), bytes) and len(a[b"infohash"]) == 20
            valid = valid and type(idx) is int and 0 <= idx < 4096
            if q == b"put_bundle":
                total = a.get(b"total", 1)
                valid = valid and type(total) is int and 0 < total <= 4096 and idx < total
        if q == b"announce_peer":
            implied = a.get(b"implied_port", 0)
            if implied == 1:
                valid = valid and 0 < int(addr[1]) <= 65535
            else:
                valid = valid and type(a.get(b"port")) is int and 0 < a[b"port"] <= 65535
        if not valid:
            self._send_error(t if isinstance(t, bytes) else b"0", 203, b"Protocol Error", addr)
            return
        sender_id = a.get(b"id")
        if isinstance(sender_id, bytes) and len(sender_id) == 20:
            self.add_routing_node(sender_id, addr[0], addr[1])
        self.queries_handled += 1
        tok_out = self.make_token(addr[0])
        if q == b"ping":
            self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id, b"token": tok_out}}, addr)
        elif q == b"find_node":
            target = a.get(b"target", b"").ljust(20, b"\x00")[:20]
            compact_nodes, nodes_list = self._closest_nodes_payload(target)
            self._send(
                {
                    b"t": t,
                    b"y": b"r",
                    b"r": {
                        b"id": self.node_id,
                        b"nodes": compact_nodes,
                        b"nodes_list": nodes_list,
                        b"token": tok_out,
                    },
                },
                addr,
            )
        elif q == b"get_peers":
            ih = a.get(b"info_hash", b"")
            peers = self._peers.get(ih, [])
            compact_vals: List[bytes] = []
            for p_ip, p_port, _ts in peers:
                try:
                    compact_vals.append(socket.inet_aton(p_ip) + struct.pack("!H", int(p_port)))
                except OSError:
                    continue
            compact_nodes, nodes_list = self._closest_nodes_payload(ih)
            self._send(
                {
                    b"t": t,
                    b"y": b"r",
                    b"r": {
                        b"id": self.node_id,
                        b"nodes": compact_nodes,
                        b"nodes_list": nodes_list,
                        b"token": tok_out,
                        b"values": compact_vals,
                    },
                },
                addr,
            )
        elif q == b"announce_peer":
            if not self._allow_write_from_ip(addr[0]):
                self._send_error(t, 202, b"Rate limit exceeded", addr)
                return
            if self.require_write_token and not self.verify_token(addr[0], a.get(b"token")):
                self._send_error(t, 203, b"Invalid token", addr)
                return
            ih = a.get(b"info_hash", b"")
            p_port = int(addr[1]) if a.get(b"implied_port") == 1 else int(a.get(b"port", addr[1]))
            if not isinstance(ih, bytes) or len(ih) != 20 or p_port <= 0 or p_port > 65535:
                self._send_error(t, 203, b"Protocol Error", addr)
                return
            if ih not in self._peers and len(self._peers) >= self.MAX_INFOHASHES:
                self._send_error(t, 202, b"Storage full", addr)
                return
            cur = [(ip, pt, ts) for ip, pt, ts in self._peers.get(ih, []) if (ip, pt) != (addr[0], p_port)]
            if len(cur) >= self.MAX_PEERS_PER_IH:
                self._send_error(t, 202, b"Peer list full", addr)
                return
            cur.append((addr[0], p_port, time.time()))
            self._peers[ih] = cur
            self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id}}, addr)
        elif q == b"get":
            target = a.get(b"target")
            if not isinstance(target, bytes) or len(target) != 20:
                self._send_error(t, 203, b"Protocol Error", addr)
                return
            compact_nodes, nodes_list = self._closest_nodes_payload(target)
            if target not in self._store:
                self._send(
                    {
                        b"t": t,
                        b"y": b"r",
                        b"r": {
                            b"id": self.node_id,
                            b"nodes": compact_nodes,
                            b"nodes_list": nodes_list,
                            b"token": tok_out,
                        },
                    },
                    addr,
                )
                return
            item = self._store[target]
            self._send(
                {
                    b"t": t,
                    b"y": b"r",
                    b"v": item["v"],
                    b"seq": item["seq"],
                    b"sig": item["sig"],
                    b"k": item["pubkey"],
                    b"salt": item["salt"],
                    b"r": {
                        b"id": self.node_id,
                        b"nodes": compact_nodes,
                        b"nodes_list": nodes_list,
                        b"token": tok_out,
                        b"v": item["v"],
                        b"seq": item["seq"],
                        b"sig": item["sig"],
                        b"k": item["pubkey"],
                        b"salt": item["salt"],
                    },
                },
                addr,
            )
        elif q == b"put":
            await self._handle_put(t, a, addr)
        elif q == b"put_bundle":
            if not self._allow_write_from_ip(addr[0]):
                self._send_error(t, 202, b"Rate limit exceeded", addr)
                return
            if self.require_write_token and not self.verify_token(addr[0], a.get(b"token")):
                self._send_error(t, 203, b"Invalid token", addr)
                return
            ih = a.get(b"infohash")
            idx = int(a.get(b"piece", 0))
            total = int(a.get(b"total", 1))
            chunk = a.get(b"blob")
            if not isinstance(ih, bytes) or len(ih) != 20 or not isinstance(chunk, bytes) or len(chunk) > 6000 or total <= 0 or total > 64:
                self._send_error(t, 205, b"Message too big", addr)
                return
            key = ih + f":{idx}:{total}".encode("ascii")
            if key not in self._bundles and len(self._bundles) >= self.MAX_BUNDLE_PIECES:
                self._send_error(t, 202, b"Storage full", addr)
                return
            self._bundles[key] = (chunk, time.time())
            self._bundles_meta[ih] = total
            self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id, b"infohash": ih, b"piece": idx}}, addr)
        elif q == b"get_bundle":
            ih = a.get(b"infohash")
            idx = int(a.get(b"piece", 0))
            if not isinstance(ih, bytes) or ih not in self._bundles_meta:
                self._send_error(t, 201, b"Not found", addr)
                return
            total = self._bundles_meta[ih]
            key = ih + f":{idx}:{total}".encode("ascii")
            if key not in self._bundles:
                self._send_error(t, 201, b"Not found", addr)
                return
            self._send(
                {
                    b"t": t,
                    b"y": b"r",
                    b"r": {b"id": self.node_id, b"blob": self._bundles[key][0], b"piece": idx, b"total": total},
                },
                addr,
            )
        else:
            self._send_error(t, 204, b"Method Unknown", addr)

    async def _handle_put(self, t: bytes, a: Dict[bytes, Any], addr: Tuple[str, int]) -> None:
        if not self._allow_write_from_ip(addr[0]):
            self._send_error(t, 202, b"Rate limit exceeded", addr)
            return
        if self.require_write_token and not self.verify_token(addr[0], a.get(b"token")):
            self._send_error(t, 203, b"Invalid token", addr)
            return
        required = {b"k", b"v", b"seq", b"sig"}
        if not required.issubset(a.keys()):
            self._send_error(t, 203, b"Missing required BEP-44 fields", addr)
            return
        pubkey = a[b"k"]
        salt = a.get(b"salt", b"")
        v = a[b"v"]
        seq = a[b"seq"]
        sig = a[b"sig"]
        cas = a.get(b"cas")
        if (
            not isinstance(pubkey, bytes)
            or len(pubkey) != 32
            or not isinstance(salt, bytes)
            or not isinstance(v, (bytes, dict))
            or type(seq) is not int or not 0 <= seq <= 2**63 - 1
            or (cas is not None and (type(cas) is not int or not 0 <= cas <= 2**63 - 1))
            or not isinstance(sig, bytes)
            or len(sig) != 64
            or len(salt) > 64
        ):
            self._send_error(t, 203, b"Invalid BEP-44 argument types", addr)
            return
        try:
            encoded_v = bencode(v)
        except Exception:
            self._send_error(t, 203, b"Unencodable BEP-44 value", addr)
            return
        if len(encoded_v) > self.MAX_V_BYTES:
            self._send_error(t, 205, b"Message (v field) too big", addr)
            return
        ok_anon, _reason_anon = self.anonymity_guard.validate_bep44_payload(v, salt)
        if not ok_anon:
            self._send_error(t, 203, b"Payload rejected by AnonymityGuard", addr)
            return
        cas_int = int(cas) if isinstance(cas, int) else None
        candidates = [
            bep44_signing_buffer(v=v, seq=seq, salt=salt, cas=cas_int),
            bep44_signing_buffer(v=v, seq=seq, salt=salt, cas=None),
        ]
        if not _verify_bep44(pubkey, sig, candidates):
            self._send_error(t, 206, b"Invalid signature", addr)
            return
        target = compute_bep44_target(pubkey, salt)
        existing = self._store.get(target)
        if existing is None and len(self._store) >= self.MAX_MUTABLE_ITEMS:
            self._send_error(t, 202, b"Mutable item table full", addr)
            return
        if existing is not None:
            if cas_int is not None and cas_int != existing["seq"]:
                self._send_error(t, 301, b"CAS hash mismatched", addr)
                return
            if seq < existing["seq"]:
                self._send_error(t, 302, b"Sequence number less than current", addr)
                return
            if seq == existing["seq"]:
                if bencode(v) == bencode(existing["v"]) and sig == existing["sig"]:
                    existing["ts"] = time.time()
                    self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id}}, addr)
                    return
                self._send_error(t, 302, b"Conflicting value at identical sequence number", addr)
                return
        self._store[target] = {
            "v": v,
            "seq": int(seq),
            "sig": sig,
            "pubkey": pubkey,
            "salt": salt,
            "ts": time.time(),
            "pinned": bool(existing and existing.get("pinned")),
        }
        self._send({b"t": t, b"y": b"r", b"r": {b"id": self.node_id}}, addr)


class _GatedUDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, node: GatedBEP44UDPNode) -> None:
        self.node = node

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if len(data) > 8192 or self.node._active_handlers >= self.node.MAX_CONCURRENT_HANDLERS:
            return
        self.node._active_handlers += 1

        async def _run() -> None:
            try:
                await self.node.handle_message(data, addr)
            finally:
                self.node._active_handlers = max(0, self.node._active_handlers - 1)

        asyncio.create_task(_run())
