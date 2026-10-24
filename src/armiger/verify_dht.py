#!/usr/bin/env python3
"""Standalone Zero-Trust BEP-44 / BEP-46 / BEP-9 / BEP-10 / BEP-19 Verifier CLI.

Allows any third party to independently verify Armiger's DHT-validated update chain:
  1. Computes BEP-44 target = SHA-1(pubkey_32 || salt)
  2. Performs iterative Kademlia XOR walk via UDP `get(target)` and verifies Ed25519 signature + sequence
  3. Extracts the 20-byte BitTorrent infohash from the signed BEP-46 pointer (`ih`)
  4. Performs a separate iterative Kademlia XOR walk toward `infohash` via `get_peers(infohash)`,
     following compact `nodes` referrals across the infohash keyspace.
  5. In `strict-p2p` mode: connects to discovered peers over TCP, fetches `.torrent` `info_bytes`
     via BEP-9 / BEP-10 (`ut_metadata`) using only the 20-byte infohash, verifies
     `SHA-1(info_bytes) == infohash`, downloads all pieces over BEP-3 TCP peer-wire, and verifies
     every piece SHA-1, the SHA-256 Merkle root, and per-file SHA-256 digests.
  6. In `webseed-fallback` mode: attempts `strict-p2p` first and falls back to BEP-19 HTTP/HTTPS
     webseed download (using `--torrent-file`, `--info-file`, or `<webseed>/<infohash>.torrent`
     authenticated against `SHA-1(info_bytes) == infohash`) with identical cryptographic verification.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import socket
import struct
import time
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib.request

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import urllib.parse

from armiger.oci_dht_node import (  # noqa: E402
    bdecode,
    bdecode_krpc,
    bencode,
    compute_bep44_target,
    fetch_bundle_via_bep19_webseed,
    fetch_bundle_via_peer_wire,
)
import os
import queue
import threading

from armiger.oci_dht_seeder import (  # noqa: E402
    DEFAULT_DHT_BOOTSTRAP_NODES,
    _KRPC_DNS_SEMAPHORE,
    parse_bootstrap_nodes,
)
from armiger.oci_vpn_guard import (  # noqa: E402
    _verify_bep44,
    bep44_signing_buffer,
    generate_bep42_node_id,
)

MAX_TORRENT_META_BYTES = 1048576


def _extract_from_bep46_dict(d: Dict[Any, Any]) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    meta: Dict[str, Any] = {}
    if b"mr" in d and isinstance(d[b"mr"], (bytes, bytearray)):
        mr_b = bytes(d[b"mr"])
        meta["merkle_root_sha256"] = mr_b.hex() if len(mr_b) == 32 else mr_b.decode("ascii", errors="ignore")
    elif "mr" in d and isinstance(d["mr"], str):
        meta["merkle_root_sha256"] = d["mr"]
    if b"ws" in d and isinstance(d[b"ws"], (bytes, bytearray)):
        meta["webseed_url"] = bytes(d[b"ws"]).decode("utf-8", errors="ignore")
    elif "ws" in d and isinstance(d["ws"], str):
        meta["webseed_url"] = d["ws"]
    ih = d.get(b"ih") if b"ih" in d else d.get("ih")
    if isinstance(ih, (bytes, bytearray)):
        ih_b = bytes(ih)
        if len(ih_b) == 20:
            return ih_b, meta
        if len(ih_b) == 40:
            return bytes.fromhex(ih_b.decode("ascii")), meta
    elif isinstance(ih, str) and len(ih.strip()) == 40:
        return bytes.fromhex(ih.strip()), meta
    return None


def _extract_infohash_20(v_obj: Any) -> Tuple[bytes, Dict[str, Any]]:
    """Extract 20-byte infohash and metadata from native BEP-46 dict or JSON payload."""
    if isinstance(v_obj, dict):
        parsed = _extract_from_bep46_dict(v_obj)
        if parsed is not None:
            return parsed

    raw_bytes: Optional[bytes] = None
    if isinstance(v_obj, (bytes, bytearray)):
        raw_bytes = bytes(v_obj)
    elif isinstance(v_obj, str):
        raw_bytes = v_obj.encode("utf-8")

    meta: Dict[str, Any] = {}
    if raw_bytes is not None:
        if raw_bytes.startswith(b"d") and raw_bytes.endswith(b"e"):
            try:
                inner = bdecode(raw_bytes)
                if isinstance(inner, dict):
                    parsed = _extract_from_bep46_dict(inner)
                    if parsed is not None:
                        return parsed
            except Exception:
                pass
        try:
            doc = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(doc, dict):
                ih_str = str(doc.get("ih") or doc.get("infohash") or "").strip()
                if len(ih_str) == 40:
                    if "mr" in doc:
                        meta["merkle_root_sha256"] = str(doc["mr"])
                    if "ws" in doc:
                        meta["webseed_url"] = str(doc["ws"])
                    return bytes.fromhex(ih_str), meta
        except Exception:
            pass

    raise ValueError("Unable to extract 20-byte BitTorrent infohash from BEP-44 value")


def _resolve_ipv4_with_deadline(
    host: str,
    deadline: float,
    resolver_fn: Optional[Callable[[str], str]] = None,
) -> str:
    host_str = str(host).strip()
    if resolver_fn is None:
        try:
            socket.inet_aton(host_str)
            return host_str
        except OSError:
            pass
    rem = deadline - time.monotonic()
    if rem <= 0:
        raise TimeoutError("krpc_dns_timeout")
    if not _KRPC_DNS_SEMAPHORE.acquire(blocking=False):
        raise TimeoutError("krpc_dns_pool_saturated")
    eff_resolver = resolver_fn or socket.gethostbyname
    res_q: "queue.Queue[Tuple[bool, Any]]" = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            res_q.put((True, str(eff_resolver(host_str))))
        except Exception as exc:
            res_q.put((False, exc))
        finally:
            _KRPC_DNS_SEMAPHORE.release()

    try:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
    except BaseException:
        _KRPC_DNS_SEMAPHORE.release()
        raise
    try:
        ok, payload = res_q.get(timeout=rem)
    except queue.Empty as exc:
        raise TimeoutError("krpc_dns_timeout") from exc
    if not ok:
        raise payload
    return str(payload)


def _extract_bep42_observed_ip(msg: Dict[bytes, Any]) -> Optional[str]:
    raw_ip = msg.get(b"ip")
    if not isinstance(raw_ip, (bytes, bytearray)):
        r_sub = msg.get(b"r")
        if isinstance(r_sub, dict):
            raw_ip = r_sub.get(b"ip")
    if isinstance(raw_ip, (bytes, bytearray)) and len(raw_ip) in (4, 6):
        try:
            return socket.inet_ntoa(bytes(raw_ip[:4]))
        except OSError:
            return None
    return None


def _udp_krpc_query(
    addr: Tuple[str, int],
    query_dict: Dict[bytes, Any],
    timeout: float = 2.0,
    resolver_fn: Optional[Callable[[str], str]] = None,
    sender_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[bytes, Any]]:
    deadline = time.monotonic() + max(0.001, float(timeout))
    try:
        resolved_ip = _resolve_ipv4_with_deadline(addr[0], deadline, resolver_fn=resolver_fn)
    except Exception:
        return None
    rem = deadline - time.monotonic()
    if rem <= 0:
        return None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(rem)
        sock.connect((resolved_ip, int(addr[1])))
        eff_ip: Optional[str] = None
        if sender_state is not None and isinstance(query_dict.get(b"a"), dict):
            eff_ip = sender_state.get("public_ip") or str(sock.getsockname()[0])
            id_map = sender_state.setdefault("id_by_ip", {})
            if eff_ip not in id_map:
                id_map[eff_ip] = generate_bep42_node_id(eff_ip)
            query_dict[b"a"][b"id"] = id_map[eff_ip]
            sender_state["last_sender_id"] = id_map[eff_ip]
        rem = deadline - time.monotonic()
        if rem <= 0:
            return None
        sock.settimeout(rem)
        sock.send(bencode(query_dict))
        rem = deadline - time.monotonic()
        if rem <= 0:
            return None
        sock.settimeout(rem)
        data = sock.recv(8192)
        msg = bdecode_krpc(data)
        if not isinstance(msg, dict):
            return None
        if (
            sender_state is not None
            and not sender_state.get("explicit_public_ip")
            and isinstance(query_dict.get(b"a"), dict)
        ):
            obs_ip = _extract_bep42_observed_ip(msg)
            if obs_ip:
                id_map = sender_state.setdefault("id_by_ip", {})
                if obs_ip not in id_map:
                    id_map[obs_ip] = generate_bep42_node_id(obs_ip)
                votes = sender_state.setdefault("ip_votes", {}).setdefault(obs_ip, set())
                votes.add((resolved_ip, int(addr[1])))
                if len(votes) >= 2:
                    sender_state["public_ip"] = obs_ip
                    sender_state["last_sender_id"] = id_map[obs_ip]
                if obs_ip != eff_ip and msg.get(b"y") == b"e":
                    rem_retry = deadline - time.monotonic()
                    if rem_retry > 0:
                        query_dict[b"a"][b"id"] = id_map[obs_ip]
                        sock.settimeout(rem_retry)
                        sock.send(bencode(query_dict))
                        rem_retry = deadline - time.monotonic()
                        if rem_retry > 0:
                            sock.settimeout(rem_retry)
                            data2 = sock.recv(8192)
                            msg2 = bdecode_krpc(data2)
                            if isinstance(msg2, dict):
                                return msg2
        return msg
    except Exception:
        return None
    finally:
        sock.close()


def _decode_compact_peers(values: Sequence[Any]) -> List[Tuple[str, int]]:
    peers: List[Tuple[str, int]] = []
    for item in values:
        if isinstance(item, (bytes, bytearray)) and len(item) == 6:
            ip_str = socket.inet_ntoa(bytes(item[:4]))
            port = struct.unpack("!H", bytes(item[4:6]))[0]
            if 0 < port <= 65535 and (ip_str, port) not in peers:
                peers.append((ip_str, port))
    return peers


def _decode_compact_nodes_with_id(blob: Any, nodes_list: Any = None) -> List[Tuple[Optional[bytes], Tuple[str, int]]]:
    out: List[Tuple[Optional[bytes], Tuple[str, int]]] = []
    seen: set[Tuple[str, int]] = set()
    if isinstance(blob, (bytes, bytearray)) and len(blob) >= 26:
        usable = (len(blob) // 26) * 26
        for off in range(0, usable, 26):
            entry = bytes(blob[off : off + 26])
            nid = entry[:20]
            ip_str = socket.inet_ntoa(entry[20:24])
            port = struct.unpack("!H", entry[24:26])[0]
            if 0 < port <= 65535 and (ip_str, port) not in seen:
                seen.add((ip_str, port))
                out.append((nid, (ip_str, port)))
    if isinstance(nodes_list, list):
        for item in nodes_list:
            if isinstance(item, list) and len(item) >= 2:
                host_b, port_v = item[0], item[1]
                host_s = host_b.decode("utf-8", errors="ignore") if isinstance(host_b, (bytes, bytearray)) else str(host_b)
                if isinstance(port_v, int) and 0 < port_v <= 65535 and (host_s, port_v) not in seen:
                    seen.add((host_s, port_v))
                    out.append((None, (host_s, port_v)))
    return out


def _xor_distance(nid: Optional[bytes], target_20: bytes) -> int:
    if not isinstance(nid, (bytes, bytearray)) or len(nid) != 20:
        return (1 << 160) - 1
    return int.from_bytes(bytes(nid), "big") ^ int.from_bytes(target_20, "big")


def extract_info_bytes_from_file(data: bytes, expected_infohash_20: Optional[bytes] = None) -> bytes:
    """Extract canonical `info_bytes` from either a `.torrent` metainfo file or raw `info_bytes`."""
    if not data or len(data) > MAX_TORRENT_META_BYTES:
        raise ValueError("invalid_or_oversized_metadata_bytes")
    if expected_infohash_20 is not None and hashlib.sha1(data).digest() == expected_infohash_20:
        return bytes(data)
    decoded = bdecode(data, max_bytes=MAX_TORRENT_META_BYTES)
    if isinstance(decoded, dict) and isinstance(decoded.get(b"info"), dict):
        info_bytes = bencode(decoded[b"info"])
    elif isinstance(decoded, dict) and b"pieces" in decoded and b"piece length" in decoded:
        info_bytes = bencode(decoded)
    else:
        raise ValueError("unrecognized_torrent_or_info_structure")
    if expected_infohash_20 is not None and hashlib.sha1(info_bytes).digest() != expected_infohash_20:
        raise ValueError("metadata_infohash_mismatch")
    return info_bytes


def _fetch_webseed_metadata(
    ws_base_url: str,
    infohash_20: bytes,
    fetcher: Optional[Callable[..., bytes]] = None,
    timeout: float = 10.0,
    max_total_seconds: Optional[float] = None,
) -> Optional[bytes]:
    """Attempt to fetch `<ws_base_url>/<infohash_hex>.torrent` (or `/bundle.torrent`) with a 1 MiB cap
    and a strict monotonic total deadline across redirects and both candidate URLs.
    """
    base = ws_base_url.rstrip("/")
    ih_hex = infohash_20.hex()
    meta_budget = float(max_total_seconds) if max_total_seconds is not None else max(float(timeout) * 2.0, 10.0)
    meta_deadline = time.monotonic() + meta_budget

    def _rem_meta_timeout() -> float:
        rem = meta_deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError("webseed_metadata_total_deadline_exceeded")
        return min(float(timeout), rem)

    from armiger.oci_sqlite_catalog import _https_get_with_deadline

    for candidate_url in (f"{base}/{ih_hex}.torrent", f"{base}/bundle.torrent"):
        if time.monotonic() >= meta_deadline:
            break
        try:
            _rem_meta_timeout()
            if fetcher is not None:
                raw = fetcher(candidate_url)
                _rem_meta_timeout()
            else:
                if not candidate_url.lower().startswith("https://"):
                    continue
                raw = _https_get_with_deadline(
                    url=candidate_url,
                    max_bytes=MAX_TORRENT_META_BYTES,
                    timeout=timeout,
                    deadline=meta_deadline,
                )
            if len(raw) > MAX_TORRENT_META_BYTES:
                continue
            return extract_info_bytes_from_file(raw, expected_infohash_20=infohash_20)
        except TimeoutError:
            if time.monotonic() >= meta_deadline:
                break
            continue
        except Exception:
            if time.monotonic() >= meta_deadline:
                break
            continue
    return None


def verify_dht_release(
    *,
    pubkey_hex: str,
    salt: bytes,
    dht_nodes: Sequence[Tuple[str, int]],
    explicit_peers: Sequence[Tuple[str, int]] = (),
    webseed_urls: Sequence[str] = (),
    expected_info_bytes: Optional[bytes] = None,
    webseed_fetcher: Optional[Callable[..., bytes]] = None,
    mode: str = "strict-p2p",
    min_seq: int = 1,
    timeout: float = 3.0,
    max_peers: int = 8,
    p2p_budget_sec: Optional[float] = None,
    public_ip: Optional[str] = None,
    resolver_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end BEP-44/46 + BEP-9/10/19 verification and return a structured receipt."""
    if mode not in ("strict-p2p", "webseed-fallback"):
        raise ValueError(f"Unsupported verification mode: {mode}")

    pubkey_32 = bytes.fromhex(pubkey_hex.strip())
    if len(pubkey_32) != 32:
        raise ValueError("pubkey_hex must be 32-byte Ed25519 public key (64 hex chars)")

    target_20 = compute_bep44_target(pubkey_32, salt)
    init_pub_ip = (public_ip or os.environ.get("ARMIGER_PUBLIC_IPV4", "")).strip() or None
    init_ip_for_id = init_pub_ip or "127.0.0.1"
    init_sender_id = generate_bep42_node_id(init_ip_for_id)
    sender_state: Dict[str, Any] = {
        "public_ip": init_pub_ip,
        "explicit_public_ip": bool(init_pub_ip),
        "ip_votes": {},
        "id_by_ip": {init_ip_for_id: init_sender_id},
        "last_sender_id": init_sender_id,
    }

    best_record: Optional[Dict[bytes, Any]] = None
    best_seq = -1
    get_frontier: List[Tuple[int, Tuple[str, int]]] = [
        (_xor_distance(None, target_20), addr) for addr in dht_nodes
    ]
    queried_get: set[Tuple[str, int]] = set()
    all_known_dht_nodes: List[Tuple[Optional[bytes], Tuple[str, int]]] = [
        (None, addr) for addr in dht_nodes
    ]

    while get_frontier and len(queried_get) < 24:
        get_frontier.sort(key=lambda item: item[0])
        _, addr = get_frontier.pop(0)
        if addr in queried_get:
            continue
        queried_get.add(addr)
        q_args: Dict[bytes, Any] = {b"id": sender_state["last_sender_id"], b"target": target_20}
        if salt:
            q_args[b"salt"] = salt
        resp = _udp_krpc_query(
            addr,
            {b"t": b"vf01", b"y": b"q", b"q": b"get", b"a": q_args},
            timeout=timeout,
            resolver_fn=resolver_fn,
            sender_state=sender_state,
        )
        if not isinstance(resp, dict) or resp.get(b"y") != b"r":
            continue
        r_body = resp.get(b"r")
        if not isinstance(r_body, dict):
            continue
        known_self_ids = set(sender_state["id_by_ip"].values())
        for ref_nid, ref_addr in _decode_compact_nodes_with_id(r_body.get(b"nodes"), None):
            if ref_nid in known_self_ids:
                continue
            all_known_dht_nodes.append((ref_nid, ref_addr))
            if ref_addr not in queried_get:
                get_frontier.append((_xor_distance(ref_nid, target_20), ref_addr))

        if b"v" in r_body and b"sig" in r_body and b"seq" in r_body:
            k_raw = r_body.get(b"k", pubkey_32)
            seq_raw = r_body.get(b"seq")
            sig_raw = r_body.get(b"sig")
            v_val = r_body.get(b"v")
            if (
                not isinstance(k_raw, (bytes, bytearray))
                or len(k_raw) != 32
                or type(seq_raw) is not int
                or not (0 <= seq_raw <= 0x7FFFFFFF_FFFFFFFF)
                or not isinstance(sig_raw, (bytes, bytearray))
                or len(sig_raw) != 64
                or not isinstance(v_val, (bytes, bytearray, dict))
            ):
                continue
            k_bytes = bytes(k_raw)
            seq_val = int(seq_raw)
            sig_bytes = bytes(sig_raw)
            if k_bytes != pubkey_32:
                continue
            try:
                encoded_v = bencode(v_val)
            except Exception:
                continue
            if len(encoded_v) > 1000:
                continue
            cand_bufs = [bep44_signing_buffer(v=v_val, seq=seq_val, salt=salt)]
            if not _verify_bep44(pubkey_32, sig_bytes, cand_bufs):
                continue
            try:
                _extract_infohash_20(v_val)
            except Exception:
                continue
            if seq_val > best_seq:
                best_seq = seq_val
                best_record = r_body

    if best_record is None:
        return {
            "verified": False,
            "reason": "bep44_mutable_record_not_found_or_invalid_sig",
            "mode_requested": mode,
            "target_hex": target_20.hex(),
        }

    if best_seq < int(min_seq):
        return {
            "verified": False,
            "reason": f"bep44_seq_below_minimum:{best_seq}<{min_seq}",
            "mode_requested": mode,
            "target_hex": target_20.hex(),
            "seq": best_seq,
        }

    infohash_20, pointer_meta = _extract_infohash_20(best_record[b"v"])

    # Separate bounded iterative Kademlia lookup toward infohash_20 for get_peers
    eff_max_peers = max(1, min(int(max_peers), 32))
    candidate_peers: List[Tuple[str, int]] = list(explicit_peers)[:eff_max_peers]
    gp_frontier: List[Tuple[int, Tuple[str, int]]] = []
    seen_gp_queued: set[Tuple[str, int]] = set()
    for ref_nid, ref_addr in all_known_dht_nodes:
        if ref_addr not in seen_gp_queued:
            seen_gp_queued.add(ref_addr)
            gp_frontier.append((_xor_distance(ref_nid, infohash_20), ref_addr))
    queried_gp: set[Tuple[str, int]] = set()

    while gp_frontier and len(queried_gp) < 24 and len(candidate_peers) < eff_max_peers:
        gp_frontier.sort(key=lambda item: item[0])
        _, addr = gp_frontier.pop(0)
        if addr in queried_gp:
            continue
        queried_gp.add(addr)
        gp_resp = _udp_krpc_query(
            addr,
            {
                b"t": b"gp01",
                b"y": b"q",
                b"q": b"get_peers",
                b"a": {b"id": sender_state["last_sender_id"], b"info_hash": infohash_20},
            },
            timeout=timeout,
            resolver_fn=resolver_fn,
            sender_state=sender_state,
        )
        if not isinstance(gp_resp, dict) or gp_resp.get(b"y") != b"r":
            continue
        r_gp = gp_resp.get(b"r")
        if not isinstance(r_gp, dict):
            continue
        if isinstance(r_gp.get(b"values"), list):
            for p in _decode_compact_peers(r_gp[b"values"]):
                if p not in candidate_peers:
                    candidate_peers.append(p)
                    if len(candidate_peers) >= eff_max_peers:
                        break
        known_self_ids = set(sender_state["id_by_ip"].values())
        for ref_nid, ref_addr in _decode_compact_nodes_with_id(r_gp.get(b"nodes"), None):
            if ref_nid in known_self_ids:
                continue
            if ref_addr not in queried_gp:
                gp_frontier.append((_xor_distance(ref_nid, infohash_20), ref_addr))

    # Attempt strict P2P BEP-9/10 ut_metadata + BEP-3 piece download first (bounded peer count + total budget)
    eff_p2p_budget = float(p2p_budget_sec) if p2p_budget_sec is not None else max(float(timeout) * 10.0, 30.0)
    p2p_deadline = time.monotonic() + eff_p2p_budget
    p2p_errors: List[str] = []
    for peer_host, peer_port in candidate_peers[:eff_max_peers]:
        rem_budget = p2p_deadline - time.monotonic()
        if rem_budget <= 0:
            p2p_errors.append("p2p_verification_budget_exceeded")
            break
        try:
            coro = fetch_bundle_via_peer_wire(
                host=peer_host,
                port=peer_port,
                infohash_20=infohash_20,
                timeout=min(float(timeout), rem_budget),
                max_total_seconds=rem_budget,
            )
            res = asyncio.run(coro)
            return {
                "verified": True,
                "mode_requested": mode,
                "mode_used": "strict-p2p",
                "peer_used": f"{peer_host}:{peer_port}",
                "target_hex": target_20.hex(),
                "seq": best_seq,
                "infohash_hex": infohash_20.hex(),
                "merkle_root_sha256": res["merkle_root_sha256"],
                "verified_files": sorted(res["files"].keys()),
                "file_sha256": {
                    fname: hashlib.sha256(fbytes).hexdigest()
                    for fname, fbytes in sorted(res["files"].items())
                },
            }
        except Exception as exc:
            p2p_errors.append(f"{peer_host}:{peer_port}:{exc}")

    if mode == "strict-p2p":
        return {
            "verified": False,
            "reason": "strict_p2p_peer_wire_unavailable",
            "p2p_errors": p2p_errors,
            "mode_requested": mode,
            "target_hex": target_20.hex(),
            "seq": best_seq,
            "infohash_hex": infohash_20.hex(),
        }

    # Mode is webseed-fallback: try BEP-19 webseed URLs
    all_webseeds: List[str] = list(webseed_urls or ())
    if pointer_meta.get("webseed_url") and pointer_meta["webseed_url"] not in all_webseeds:
        all_webseeds.append(pointer_meta["webseed_url"])

    ws_errors: List[str] = []
    authenticated_info: Optional[bytes] = None
    if expected_info_bytes is not None:
        try:
            authenticated_info = extract_info_bytes_from_file(expected_info_bytes, expected_infohash_20=infohash_20)
        except Exception as exc:
            ws_errors.append(f"supplied_metadata_invalid:{exc}")

    for ws_url in all_webseeds:
        try:
            ws_budget = max(float(timeout) * 3.0, 15.0)
            ws_deadline = time.monotonic() + ws_budget
            eff_info = authenticated_info
            if eff_info is None:
                rem_meta = ws_deadline - time.monotonic()
                if rem_meta <= 0:
                    ws_errors.append(f"{ws_url}:webseed_total_deadline_exceeded")
                    continue
                eff_info = _fetch_webseed_metadata(
                    ws_url,
                    infohash_20,
                    fetcher=webseed_fetcher,
                    timeout=timeout,
                    max_total_seconds=rem_meta,
                )
            if eff_info is None:
                ws_errors.append(f"{ws_url}:missing_authenticated_info_bytes")
                continue
            rem_payload = ws_deadline - time.monotonic()
            if rem_payload <= 0:
                ws_errors.append(f"{ws_url}:webseed_total_deadline_exceeded")
                continue
            res = fetch_bundle_via_bep19_webseed(
                info_bytes=eff_info,
                webseed_base_url=ws_url,
                expected_infohash_20=infohash_20,
                fetcher=webseed_fetcher,
                timeout=timeout,
                max_total_seconds=rem_payload,
            )
            return {
                "verified": True,
                "mode_requested": mode,
                "mode_used": "webseed-fallback",
                "webseed_used": ws_url,
                "target_hex": target_20.hex(),
                "seq": best_seq,
                "infohash_hex": infohash_20.hex(),
                "merkle_root_sha256": res["merkle_root_sha256"],
                "verified_files": sorted(res["files"].keys()),
                "file_sha256": {
                    fname: hashlib.sha256(fbytes).hexdigest()
                    for fname, fbytes in sorted(res["files"].items())
                },
            }
        except Exception as exc:
            ws_errors.append(f"{ws_url}:{exc}")

    return {
        "verified": False,
        "reason": "webseed_fallback_exhausted",
        "p2p_errors": p2p_errors,
        "webseed_errors": ws_errors,
        "mode_requested": mode,
        "target_hex": target_20.hex(),
        "seq": best_seq,
        "infohash_hex": infohash_20.hex(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Armiger BEP-44/46 DHT + BitTorrent release chain.")
    parser.add_argument("--pubkey", required=True, help="32-byte Ed25519 publisher public key hex (64 hex chars)")
    parser.add_argument("--salt", default="armiger-oci-v1-int8", help="BEP-44 salt string")
    parser.add_argument(
        "--dht-node",
        action="append",
        dest="dht_nodes",
        default=[],
        help="DHT node host:port (repeatable; defaults to mainline bootstrap nodes)",
    )
    parser.add_argument(
        "--peer",
        action="append",
        dest="peers",
        default=[],
        help="Explicit BitTorrent TCP peer host:port (repeatable)",
    )
    parser.add_argument(
        "--webseed-url",
        action="append",
        dest="webseed_urls",
        default=[],
        help="Optional BEP-19 HTTP/HTTPS webseed base URL (repeatable)",
    )
    parser.add_argument(
        "--torrent-file",
        default="",
        help="Optional path to a .torrent or raw info dictionary file for webseed-fallback verification",
    )
    parser.add_argument(
        "--mode",
        choices=("strict-p2p", "webseed-fallback"),
        default="strict-p2p",
        help="Verification mode: strict-p2p requires live TCP peer-wire + BEP-9/10; webseed-fallback permits BEP-19 fallback",
    )
    parser.add_argument("--min-seq", type=int, default=1, help="Minimum acceptable BEP-44 sequence number")
    parser.add_argument("--timeout", type=float, default=3.0, help="Per-socket timeout in seconds")
    parser.add_argument("--max-peers", type=int, default=8, help="Maximum candidate TCP peers to attempt")
    parser.add_argument("--p2p-budget", type=float, default=None, help="Total time budget in seconds for P2P peer-wire phase")
    args = parser.parse_args(argv)

    dht_list = (
        parse_bootstrap_nodes(",".join(args.dht_nodes))
        if args.dht_nodes
        else list(DEFAULT_DHT_BOOTSTRAP_NODES)
    )
    peer_list = parse_bootstrap_nodes(",".join(args.peers)) if args.peers else []
    expected_info: Optional[bytes] = None
    if args.torrent_file:
        tpath = Path(args.torrent_file)
        raw_t = tpath.read_bytes()
        expected_info = extract_info_bytes_from_file(raw_t)

    receipt = verify_dht_release(
        pubkey_hex=args.pubkey,
        salt=args.salt.encode("utf-8"),
        dht_nodes=dht_list,
        explicit_peers=peer_list,
        webseed_urls=args.webseed_urls,
        expected_info_bytes=expected_info,
        mode=args.mode,
        min_seq=args.min_seq,
        timeout=args.timeout,
        max_peers=args.max_peers,
        p2p_budget_sec=args.p2p_budget,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
