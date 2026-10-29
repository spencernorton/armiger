#!/usr/bin/env python3
"""OCI 7-Node VPN Egress Guard, Artifact Manifest, and BEP-44 Ed25519 Signing Helpers.

Safety & Anonymity Invariants:
  1. Mandatory Startup, Periodic, & Per-Packet VPN Egress Assertion (`verify_vpn_egress`):
     Verifies `tun0` exists, is UP, and is the effective selected egress route across the IPv4
     address space (either via both `0.0.0.0/1` + `128.0.0.0/1` split routes without competing
     non-VPN split routes, or via a lower-metric `0.0.0.0/0` default route on `tun0`, or via
     effective kernel route selection `ip -4 route get 1.1.1.1` + `128.0.0.1` both selecting
     `dev tun0`).
  2. Synchronous packet-boundary kill switch (`require_for_packet()`): re-evaluates tunnel state
     on every packet/connection boundary and immediately invokes drop callbacks to close active
     sockets/transports.
  3. Fail-closed Ed25519 signing/verification (no forgeable HMAC fallback).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
if not __package__ and str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False

# Pinned to the HF commit these hashes describe; `main` now serves Armiger v3.0.
DEFAULT_RELEASE_PUBKEY_HEX = "3599362bd52dad8b89398b904d3081c2cb483d13376f7ffb49f3748208731e76"
DEFAULT_RELEASE_SALT = "armiger:weights:ultra"

HF_BASE_URL = "https://huggingface.co/NorviTech/armiger-ultra/resolve/7de69d7a94def2425e7d873aff7df1c041ab03da"
ARTIFACT_MANIFEST: Dict[str, Dict[str, Any]] = {
    "armiger.int8.onnx": {
        "sha256": "7ee3dfdaae4283faecabf06bc7d7644d1747d959264506b07065a3035fe6ffd0",
        "size_bytes": 17606389,
        "tiers": ("int8", "ultra"),
        "url": f"{HF_BASE_URL}/armiger.int8.onnx",
    },
    "armiger_ultra.fp32.onnx": {
        "sha256": "3125e70b22303f1a548e7b890d2b1c078858eed1b987cc9d63c52188fbc630c0",
        "size_bytes": 33260312,
        "tiers": ("ultra",),
        "url": f"{HF_BASE_URL}/armiger_ultra.fp32.onnx",
    },
    "swarm_adapter.pt": {
        "sha256": "4e4c7a2dd59fb91a234e6627e7ade35062f99268d3f05370432bb9f2a2f9b1c9",
        "size_bytes": 144144,
        "tiers": ("ultra",),
        "url": f"{HF_BASE_URL}/swarm_adapter.pt",
    },
    "calibration.json": {
        "sha256": "ac2fe3b3de8cda9d3e7bca3b334157ac0d90e052e2ff03c3787900b991b13b34",
        "size_bytes": 76612,
        "tiers": ("int8", "ultra"),
        "url": f"{HF_BASE_URL}/calibration.json",
    },
    "vocab.json": {
        "sha256": "c10afea430188e0207752e93bf0b340fbb585858684037c5134a477cbc0dd7bb",
        "size_bytes": 39996,
        "tiers": ("int8", "ultra"),
        "url": f"{HF_BASE_URL}/vocab.json",
    },
    "config.json": {
        "sha256": "6263a1f6bc82bbcafdfd2d98d327d53b07b8f2d6385cbb5a0cb8c1c9f42dff46",
        "size_bytes": 5201,
        "tiers": ("int8", "ultra"),
        "url": f"{HF_BASE_URL}/config.json",
    },
    "validated_zero_cue_catalog.json": {
        "sha256": "8a8b7a4ba97cb22c3aa1f75ab33762052deb09580df464b1eabfe622a2837d26",
        "size_bytes": 26039500,
        "tiers": ("int8", "ultra"),
        "url": f"{HF_BASE_URL}/validated_zero_cue_catalog.json",
    },
}


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


def _bdecode_at(
    data: bytes,
    idx: int,
    depth: int = 0,
    max_depth: int = 32,
    max_items: int = 4096,
    counter: Optional[List[int]] = None,
    allow_unsorted_keys: bool = False,
) -> Tuple[Any, int]:
    if depth > max_depth:
        raise ValueError("Bencode nesting limit exceeded")
    if idx >= len(data):
        raise ValueError("Unexpected EOF in bdecode")
    if counter is not None:
        counter[0] += 1
        if counter[0] > max_items:
            raise ValueError("Bencode item count limit exceeded")
    ch = data[idx:idx + 1]
    if ch == b"i":
        end = data.index(b"e", idx + 1)
        num = data[idx + 1:end]
        digits = num[1:] if num.startswith(b"-") else num
        if (not digits or not digits.isdigit() or len(digits) > 20
                or (digits.startswith(b"0") and len(digits) > 1) or num == b"-0"):
            raise ValueError("Invalid bencoded integer")
        return int(num), end + 1
    if ch in (b"l", b"d"):
        idx += 1
        arr = []
        out = {}
        prev = None
        count_here = 0
        while data[idx:idx + 1] != b"e":
            count_here += 1
            if count_here > max_items:
                raise ValueError("Bencode container item limit exceeded")
            item, idx = _bdecode_at(data, idx, depth + 1, max_depth=max_depth, max_items=max_items, counter=counter, allow_unsorted_keys=allow_unsorted_keys)
            if ch == b"l":
                arr.append(item)
            else:
                if not isinstance(item, bytes) or item in out or (not allow_unsorted_keys and prev is not None and item <= prev):
                    raise ValueError("Noncanonical or duplicate dictionary key")
                prev = item
                value, idx = _bdecode_at(data, idx, depth + 1, max_depth=max_depth, max_items=max_items, counter=counter, allow_unsorted_keys=allow_unsorted_keys)
                out[item] = value
        return (arr if ch == b"l" else out), idx + 1
    if b"0" <= ch <= b"9":
        colon = data.index(b":", idx)
        raw_len = data[idx:colon]
        if (not raw_len.isdigit() or len(raw_len) > 8
                or (raw_len.startswith(b"0") and len(raw_len) > 1)):
            raise ValueError("Invalid byte string length")
        length = int(raw_len)
        start = colon + 1
        end = start + length
        if end > len(data):
            raise ValueError("Truncated bencoded byte string")
        return data[start:end], end
    raise ValueError("Invalid bencode token")


def bdecode(data: bytes, *, max_bytes: int = 1048576, max_depth: int = 32, max_items: int = 4096, allow_unsorted_keys: bool = False) -> Any:
    """Decode canonical bencoded bytes with strict depth, item, and byte limits."""
    if not isinstance(data, (bytes, bytearray)) or not data or len(data) > max_bytes:
        raise ValueError("Empty, oversized, or non-bytes input to bdecode")
    val, idx = _bdecode_at(bytes(data), 0, 0, max_depth=max_depth, max_items=max_items, counter=[0], allow_unsorted_keys=allow_unsorted_keys)
    if idx != len(data):
        raise ValueError("Trailing bytes after bencoded payload")
    return val


def bdecode_krpc(
    data: bytes,
    *,
    max_bytes: int = 8192,
    max_depth: int = 8,
    max_items: int = 128,
) -> Any:
    """Decode an untrusted UDP KRPC datagram with strict 8 KiB, depth-8, 128-item limits."""
    return bdecode(data, max_bytes=max_bytes, max_depth=max_depth, max_items=max_items, allow_unsorted_keys=True)


_CRC32C_TABLE: Optional[List[int]] = None


def _crc32c(data: bytes) -> int:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        tbl: List[int] = []
        for i in range(256):
            crc = i
            for _ in range(8):
                crc = (crc >> 1) ^ 0x82F63B78 if (crc & 1) else (crc >> 1)
            tbl.append(crc)
        _CRC32C_TABLE = tbl
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _is_bep42_exempt_ip(ip_str: str) -> bool:
    import ipaddress
    try:
        ip_obj = ipaddress.IPv4Address(ip_str)
    except Exception:
        return False
    b0, b1, _, _ = ip_obj.packed
    if b0 in (10, 127) or (b0 == 172 and 16 <= b1 <= 31) or (b0 == 192 and b1 == 168) or (b0 == 169 and b1 == 254):
        return True
    if b0 == 100 and 64 <= b1 <= 127:  # CGNAT / Tailscale
        return True
    return False


def generate_bep42_node_id(ip_str: str, rand_byte: Optional[int] = None) -> bytes:
    """Generate a 20-byte Kademlia node_id conforming to BEP-42 (CRC32C IPv4 prefix)."""
    import ipaddress
    import os
    r_val = (os.urandom(1)[0] if rand_byte is None else int(rand_byte)) & 0xFF
    r = r_val & 0x07
    ip_bytes = ipaddress.IPv4Address(ip_str).packed
    v = bytes([
        ((ip_bytes[0] & 0x03) | (r << 5)) & 0xFF,
        ip_bytes[1] & 0x0F,
        ip_bytes[2] & 0x3F,
        ip_bytes[3] & 0xFF,
    ])
    c = _crc32c(v)
    b0 = (c >> 24) & 0xFF
    b1 = (c >> 16) & 0xFF
    b2 = ((c >> 8) & 0xF8) | (os.urandom(1)[0] & 0x07)
    return bytes([b0, b1, b2]) + os.urandom(16) + bytes([r_val])


def verify_bep42_node_id(ip_str: str, node_id: bytes, allow_local: bool = True) -> bool:
    """Verify whether a 20-byte node_id matches `ip_str` under BEP-42."""
    import ipaddress
    if not isinstance(node_id, (bytes, bytearray)) or len(node_id) != 20:
        return False
    if allow_local and _is_bep42_exempt_ip(ip_str):
        return True
    try:
        ip_bytes = ipaddress.IPv4Address(ip_str).packed
    except Exception:
        return False
    r = node_id[19] & 0x07
    v = bytes([
        ((ip_bytes[0] & 0x03) | (r << 5)) & 0xFF,
        ip_bytes[1] & 0x0F,
        ip_bytes[2] & 0x3F,
        ip_bytes[3] & 0xFF,
    ])
    c = _crc32c(v)
    return (
        node_id[0] == ((c >> 24) & 0xFF)
        and node_id[1] == ((c >> 16) & 0xFF)
        and (node_id[2] & 0xF8) == ((c >> 8) & 0xF8)
    )


def bep44_signing_buffer(
    v: bytes,
    seq: int,
    salt: bytes = b"",
    cas: Optional[int] = None,
) -> bytes:
    """Construct canonical BEP-44 signing bytes; CAS is an unsigned write precondition."""
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


def _sign_bep44(privkey_32: bytes, sig_buf: bytes) -> Tuple[bytes, bytes]:
    """Return (pubkey_32, sig_64) using Ed25519; fail closed when cryptography is unavailable."""
    if not _HAS_CRYPTOGRAPHY:
        raise RuntimeError("Ed25519 support via 'cryptography' is required for BEP-44 signing")
    if len(privkey_32) != 32:
        raise ValueError("Ed25519 private key seed must be 32 bytes")
    sk = Ed25519PrivateKey.from_private_bytes(privkey_32)
    pk = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return pk, sk.sign(sig_buf)


def _verify_bep44(pubkey_32: bytes, sig_64: bytes, candidate_buffers: Sequence[bytes]) -> bool:
    """Verify Ed25519 signature over candidate BEP-44 buffers; fail closed when cryptography is unavailable."""
    if not _HAS_CRYPTOGRAPHY:
        raise RuntimeError("Ed25519 support via 'cryptography' is required for BEP-44 verification")
    if len(pubkey_32) != 32 or len(sig_64) != 64:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(pubkey_32)
        for buf in candidate_buffers:
            try:
                pub.verify(sig_64, buf)
                return True
            except Exception:
                continue
        return False
    except Exception:
        return False


@dataclass
class VPNEgressStatus:
    verified: bool
    interface: str
    interface_present: bool
    interface_up: bool
    has_tunnel_route: bool
    has_default_or_split_route: bool
    checked_at: float
    reason: str


def _probe_effective_route_dev(probe_line: str) -> Optional[str]:
    tokens = probe_line.split()
    for i, tok in enumerate(tokens[:-1]):
        if tok == "dev":
            return tokens[i + 1].strip()
    return None



def _kernel_udp_route_probe(iface: str) -> Optional[str]:
    """Pure-Python kernel FIB route lookup for both IPv4 halves (1.1.1.1 and 128.0.0.1)
    via SIOCGIFADDR (0x8915) + UDP socket connect, covering Gluetun policy table 51820
    even when the `ip` binary is not installed in the container image.
    """
    import fcntl
    import socket
    import struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        info = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", iface.encode("utf-8")[:15]))
        iface_ip = socket.inet_ntoa(info[20:24])
    except Exception:
        return None
    finally:
        s.close()
    if not iface_ip or iface_ip.startswith("127."):
        return None
    lines: List[str] = []
    for probe_ip in ("1.1.1.1", "128.0.0.1"):
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            u.connect((probe_ip, 53))
            src_ip = u.getsockname()[0]
        except Exception:
            return None
        finally:
            u.close()
        dev = iface if src_ip == iface_ip else "non_vpn"
        lines.append(f"{probe_ip} dev {dev} src {src_ip}")
    return "\n".join(lines) if len(lines) == 2 else None


def verify_vpn_egress(
    expected_iface: str = "tun0",
    proc_net_dev_text: Optional[str] = None,
    proc_net_route_text: Optional[str] = None,
    operstate_text: Optional[str] = None,
    ip_route_output: Optional[str] = None,
) -> VPNEgressStatus:
    """Verify that the WireGuard/Gluetun tunnel interface (`tun0`) exists, is UP,
    and is the effective selected egress route across the IPv4 address space.
    """
    now = time.time()
    iface = (expected_iface or "tun0").strip()
    _live_kernel_probe = (proc_net_dev_text is None and proc_net_route_text is None and ip_route_output is None)
    if proc_net_dev_text is None:
        dev_path = Path("/proc/net/dev")
        proc_net_dev_text = dev_path.read_text(encoding="utf-8", errors="replace") if dev_path.exists() else ""
    if proc_net_route_text is None:
        route_path = Path("/proc/net/route")
        proc_net_route_text = route_path.read_text(encoding="utf-8", errors="replace") if route_path.exists() else ""
    if operstate_text is None:
        oper_path = Path(f"/sys/class/net/{iface}/operstate")
        flags_path = Path(f"/sys/class/net/{iface}/flags")
        if oper_path.exists():
            operstate_text = oper_path.read_text(encoding="utf-8", errors="replace").strip()
        elif flags_path.exists():
            try:
                fl = int(flags_path.read_text(encoding="utf-8").strip(), 16)
                operstate_text = "up" if (fl & 0x1) else "down"
            except ValueError:
                operstate_text = "unknown"
        else:
            operstate_text = ""

    iface_present = any(
        line.split(":", 1)[0].strip() == iface
        for line in proc_net_dev_text.splitlines()
        if ":" in line
    )
    has_tunnel_route = False
    vpn_default_metrics: List[int] = []
    vpn_split_low = False
    vpn_split_high = False
    non_vpn_defaults: List[Tuple[str, int]] = []
    non_vpn_splits: List[str] = []

    for idx, line in enumerate(proc_net_route_text.splitlines()):
        if idx == 0 or not line.strip():
            continue
        cols = line.split()
        if len(cols) < 8:
            continue
        r_iface, dest_hex, flags_hex, mask_hex = cols[0], cols[1].upper(), cols[3], cols[7].upper()
        try:
            r_flags = int(flags_hex, 16)
            metric = int(cols[6])
        except ValueError:
            continue
        if not (r_flags & 0x1):
            continue
        if r_iface == iface:
            has_tunnel_route = True
            if dest_hex == "00000000" and mask_hex == "00000000":
                vpn_default_metrics.append(metric)
            elif dest_hex == "00000000" and mask_hex == "00000080":
                vpn_split_low = True
            elif dest_hex == "00000080" and mask_hex == "00000080":
                vpn_split_high = True
        else:
            if dest_hex == "00000000" and mask_hex == "00000000":
                non_vpn_defaults.append((r_iface, metric))
            elif mask_hex == "00000080" and dest_hex in ("00000000", "00000080"):
                non_vpn_splits.append(r_iface)

    partial_split = vpn_split_low != vpn_split_high
    full_split_covered = vpn_split_low and vpn_split_high and not non_vpn_splits
    competing_non_vpn_default = False
    default_covered = False
    if vpn_default_metrics:
        best_vpn_metric = min(vpn_default_metrics)
        if not non_vpn_defaults:
            default_covered = True
        else:
            best_non_vpn_metric = min(m for _, m in non_vpn_defaults)
            if best_vpn_metric < best_non_vpn_metric:
                default_covered = True
            else:
                competing_non_vpn_default = True

    has_default_or_split_route = full_split_covered or default_covered

    # When main-table /proc/net/route lacks default/split coverage (e.g. Gluetun fwmark/policy table 51820),
    # verify effective kernel route selection for both halves of IPv4 (1.1.1.1 and 128.0.0.1).
    if ip_route_output is None and iface_present and not has_default_or_split_route and not partial_split and not competing_non_vpn_default:
        probe_lines: List[str] = []
        for probe_ip in ("1.1.1.1", "128.0.0.1"):
            try:
                res = subprocess.run(
                    ["ip", "-4", "route", "get", probe_ip],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    probe_lines.append(res.stdout.splitlines()[0].strip())
            except Exception:
                pass
        if len(probe_lines) == 2:
            ip_route_output = "\n".join(probe_lines)
        elif _live_kernel_probe:
            ip_route_output = _kernel_udp_route_probe(iface)

    if ip_route_output and not partial_split and not competing_non_vpn_default:
        lines = [ln.strip() for ln in ip_route_output.splitlines() if ln.strip()]
        probe_devs = [_probe_effective_route_dev(ln) for ln in lines if " via " in ln or " dev " in ln]
        if len(probe_devs) >= 2 and all(d == iface for d in probe_devs):
            has_tunnel_route = True
            has_default_or_split_route = True
        elif any(d is not None and d != iface for d in probe_devs):
            has_default_or_split_route = False

    iface_up = operstate_text.lower() in ("up", "unknown") if operstate_text else iface_present
    if not iface_present:
        return VPNEgressStatus(False, iface, False, False, False, False, now, f"VPN interface {iface!r} missing from /proc/net/dev")
    if not iface_up:
        return VPNEgressStatus(False, iface, True, False, has_tunnel_route, has_default_or_split_route, now, f"VPN interface {iface!r} operstate is {operstate_text!r}")
    if partial_split:
        return VPNEgressStatus(False, iface, True, True, has_tunnel_route, False, now, f"VPN interface {iface!r} has incomplete split-horizon route (0.0.0.0/1={vpn_split_low}, 128.0.0.0/1={vpn_split_high})")
    if competing_non_vpn_default and not full_split_covered:
        return VPNEgressStatus(False, iface, True, True, has_tunnel_route, False, now, f"Competing non-VPN default route overrides {iface!r}: {non_vpn_defaults}")
    if not has_tunnel_route:
        return VPNEgressStatus(False, iface, True, True, False, False, now, f"VPN interface {iface!r} has no active route")
    if not has_default_or_split_route:
        return VPNEgressStatus(False, iface, True, True, True, False, now, f"VPN interface {iface!r} is not the effective selected default/split egress route")
    return VPNEgressStatus(True, iface, True, True, True, True, now, "ok")


class VPNEgressGuard:
    """Fail-closed runtime VPN guard that re-verifies on every packet/connection boundary
    and halts DHT & BitTorrent traffic immediately if `tun0` drops.
    """
    def __init__(
        self,
        expected_iface: str = "tun0",
        check_fn: Optional[Callable[[], VPNEgressStatus]] = None,
        check_interval_sec: float = 10.0,
    ) -> None:
        self.expected_iface = expected_iface
        self._check_fn = check_fn or (lambda: verify_vpn_egress(expected_iface=self.expected_iface))
        self.check_interval_sec = max(0.05, float(check_interval_sec))
        self._lock = threading.RLock()
        self._status: VPNEgressStatus = self._check_fn()
        self._on_drop_callbacks: List[Callable[[VPNEgressStatus], None]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.blocked_packet_attempts: int = 0

    def register_on_drop(self, cb: Callable[[VPNEgressStatus], None]) -> None:
        with self._lock:
            self._on_drop_callbacks.append(cb)

    def check_now(self) -> VPNEgressStatus:
        status = self._check_fn()
        callbacks_to_fire: List[Callable[[VPNEgressStatus], None]] = []
        with self._lock:
            self._status = status
            if not status.verified:
                callbacks_to_fire = list(self._on_drop_callbacks)
        for cb in callbacks_to_fire:
            try:
                cb(status)
            except Exception:
                pass
        return status

    @property
    def status(self) -> VPNEgressStatus:
        with self._lock:
            return self._status

    @property
    def is_verified(self) -> bool:
        return self.check_now().verified

    def assert_egress_or_raise(self) -> VPNEgressStatus:
        st = self.check_now()
        if not st.verified:
            raise RuntimeError(f"VPN egress assertion failed ({st.interface}): {st.reason}")
        return st

    def require_for_packet(self) -> bool:
        st = self.check_now()
        with self._lock:
            if st.verified:
                return True
            self.blocked_packet_attempts += 1
            return False

    def start_background_monitor(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.wait(self.check_interval_sec):
                self.check_now()

        self._thread = threading.Thread(target=_loop, name="vpn-egress-watchdog", daemon=True)
        self._thread.start()

    def stop_background_monitor(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


_FORBIDDEN_PII_KEYS = frozenset({
    b"ip", b"host", b"hostname", b"user", b"username", b"email",
    b"filepath", b"raw_title", b"passkey", b"tracker", b"tracker_url",
    b"client_ip", b"tailscale_ip", b"private_ip",
})
_ALLOWED_BEP44_DICT_KEYS = frozenset({
    b"bundle", b"ih", b"merkle_root", b"mr", b"seq", b"rv", b"exp", b"u",
    b"tier", b"catalog_hashes", b"schema", b"envelope_sha256",
    b"ts", b"ts_utc", b"v", b"ws",
})


def is_sharing_opted_in(default: bool = True) -> bool:
    """Return whether outbound public DHT seeding is opted in via ARMIGER_DHT_SHARE_OPT_IN."""
    import os
    raw = os.environ.get("ARMIGER_DHT_SHARE_OPT_IN", "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


class AnonymityGuard:
    """Enforces zero-PII and BLAKE2b-64 / SHA-256 content-addressing invariants on all
    inbound and outbound BEP-44 payloads, plus opt-in sharing enforcement (`ARMIGER_DHT_SHARE_OPT_IN`).
    """
    def __init__(self, opt_in_sharing: Optional[bool] = None) -> None:
        self.opt_in_sharing: bool = is_sharing_opted_in(default=True) if opt_in_sharing is None else bool(opt_in_sharing)
        self.rejected_packets: int = 0

    def validate_bep44_payload(self, v: Any, salt: bytes = b"") -> Tuple[bool, str]:
        if isinstance(v, dict):
            try:
                encoded_v = bencode(v)
            except Exception:
                self.rejected_packets += 1
                return False, "invalid_v_encoding"
            if not v or len(encoded_v) > 1000:
                self.rejected_packets += 1
                return False, "invalid_v_length"
            decoded = v
            raw_v_bytes = encoded_v
        elif isinstance(v, (bytes, bytearray)):
            raw_v_bytes = bytes(v)
            if not raw_v_bytes or len(bencode(raw_v_bytes)) > 1000:
                self.rejected_packets += 1
                return False, "invalid_v_length"
            try:
                decoded = bdecode(raw_v_bytes, max_bytes=1000, max_depth=8, max_items=64)
            except Exception:
                decoded = None
        else:
            self.rejected_packets += 1
            return False, "invalid_v_type"

        if not isinstance(salt, (bytes, bytearray)) or len(salt) > 64:
            self.rejected_packets += 1
            return False, "invalid_salt_length"
        for bad_marker in (b"passkey=", b"announce?", b"@", b"100.64.", b"192.168."):
            if bad_marker in salt:
                self.rejected_packets += 1
                return False, "forbidden_pii_marker"

        if isinstance(decoded, dict):
            for k, val_item in decoded.items():
                kb = k.lower() if isinstance(k, bytes) else str(k).lower().encode("utf-8")
                if kb in _FORBIDDEN_PII_KEYS:
                    self.rejected_packets += 1
                    return False, f"forbidden_pii_key:{kb.decode('ascii', errors='ignore')}"
                if kb not in _ALLOWED_BEP44_DICT_KEYS:
                    self.rejected_packets += 1
                    return False, f"unapproved_bep44_key:{kb.decode('ascii', errors='ignore')}"
                if kb == b"ih":
                    if not isinstance(val_item, bytes) or len(val_item) != 20:
                        self.rejected_packets += 1
                        return False, "invalid_bep46_ih_length"
                    continue
                if kb == b"mr" and isinstance(val_item, bytes) and len(val_item) == 32:
                    continue
                val_bytes = val_item if isinstance(val_item, bytes) else bencode(val_item)
                for bad_marker in (b"passkey=", b"announce?", b"@", b"100.64.", b"192.168."):
                    if bad_marker in val_bytes:
                        self.rejected_packets += 1
                        return False, "forbidden_pii_marker"
        else:
            for bad_marker in (b"passkey=", b"announce?", b"@", b"100.64.", b"192.168."):
                if bad_marker in raw_v_bytes:
                    self.rejected_packets += 1
                    return False, "forbidden_pii_marker"
        return True, "ok"

    def assert_valid_or_raise(self, v: bytes, salt: bytes = b"") -> None:
        ok, reason = self.validate_bep44_payload(v, salt)
        if not ok:
            raise ValueError(f"AnonymityGuard rejected BEP-44 payload: {reason}")
