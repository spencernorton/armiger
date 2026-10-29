"""Low-Memory SQLite Zero-Cue Catalog and BitTorrent Bundle Builder for OCI Nodes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import codecs
import math
import tempfile
from pathlib import Path
import re
import sqlite3
import threading
import asyncio
import os
import struct
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from armiger.oci_vpn_guard import VPNEgressGuard, bencode, compute_bep44_target

# ---------------------------------------------------------------------------
# 2. Low-Memory SQLite Zero-Cue Catalog (`<=4 MiB` cache for Tier B 1-CPU nodes)
# ---------------------------------------------------------------------------
_MAP_KIND_CODES: Dict[str, int] = {
    "anime_map": 1,
    "title_only_map": 2,
    "title_year_map": 3,
}
_STREAM_ENTRY_RE = re.compile(r'"([0-9a-f]{16})"\s*:\s*"([^"]+)"')
_STREAM_SECTION_RE = re.compile(r'"(anime_map|title_only_map|title_year_map)"\s*:\s*\{')

class _CatalogJSONStream:
    """Bounded stdlib JSON reader; catalog map entries are parsed one at a time."""
    def __init__(self, fh, chunk_size):
        self.fh, self.chunk_size, self.buf, self.pos = fh, max(64, min(chunk_size, 262144)), "", 0
        self.hasher = hashlib.sha256()
        self.decoder = json.JSONDecoder(object_pairs_hook=self._object)
        self.eof = False
        self.utf8 = codecs.getincrementaldecoder("utf-8")()

    @staticmethod
    def _object(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate_catalog_key")
            out[key] = value
        return out

    def refill(self):
        self.buf = self.buf[self.pos:]
        self.pos = 0
        raw = self.fh.read(self.chunk_size)
        if not raw:
            self.eof = True
            self.buf += self.utf8.decode(b"", final=True)
            return False
        self.hasher.update(raw)
        # Input is ASCII JSON hashes/categories; decode strictly, never discard invalid bytes.
        self.buf += self.utf8.decode(raw)
        if len(self.buf) > 1048576:
            raise ValueError("catalog_json_token_exceeds_limit")
        return True

    def peek(self):
        while True:
            while self.pos < len(self.buf) and self.buf[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buf):
                return self.buf[self.pos]
            if not self.refill():
                return ""

    def expect(self, char):
        if self.peek() != char:
            raise ValueError("malformed_catalog_json")
        self.pos += 1

    def value(self):
        ch = self.peek()
        if ch in "-0123456789":
            while not self.eof and not any(c.isspace() or c in ",}]" for c in self.buf[self.pos:]):
                self.refill()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buf, self.pos)
                if not isinstance(value, bool) and isinstance(value, (int, float)):
                    if end == len(self.buf) and not self.eof:
                        if self.refill():
                            continue
                    if end < len(self.buf) and not (self.buf[end].isspace() or self.buf[end] in ",}]"):
                        if not self.eof and self.buf[end] in ".eE+-" and self.refill():
                            continue
                        raise ValueError("malformed_catalog_json")
                self.pos = end
                return value
            except json.JSONDecodeError:
                if self.eof or not self.refill():
                    raise ValueError("malformed_catalog_json")


def build_sqlite_zero_cue_catalog(json_path: str | Path, sqlite_path: str | Path,
                                  chunk_size: int = 262144) -> Dict[str, Any]:
    """Validate the entire catalog and atomically install a parameterized SQLite conversion.

    Parse one catalog entry at a time. On malformed JSON/schema the previous database survives.
    """
    src, dst = Path(json_path), Path(sqlite_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=dst.parent)
    os.close(fd)
    tmp_dst = Path(tmp)
    conn = sqlite3.connect(str(tmp_dst))
    count = 0
    categories = {"movie", "tv_show", "xxx", "music", "comic", "game", "audiobook", "software", "ebook", "course", "unknown", "books"}
    try:
        conn.execute("PRAGMA cache_size=-4000")
        conn.execute("PRAGMA mmap_size=0")
        conn.execute("CREATE TABLE zero_cue_entries (map_kind INTEGER NOT NULL,key_hash TEXT NOT NULL,category TEXT NOT NULL,PRIMARY KEY(map_kind,key_hash)) WITHOUT ROWID")
        conn.execute("CREATE TABLE catalog_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID")
        with src.open("rb") as fh:
            parser = _CatalogJSONStream(fh, chunk_size)
            parser.expect("{")
            top_keys = set()
            while parser.peek() != "}":
                name = parser.value()
                if not isinstance(name, str) or name in top_keys:
                    raise ValueError("duplicate_or_invalid_catalog_section")
                top_keys.add(name)
                parser.expect(":")
                if name in _MAP_KIND_CODES:
                    parser.expect("{")
                    while parser.peek() != "}":
                        key = parser.value()
                        parser.expect(":")
                        cat = parser.value()
                        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{16}", key) is None or not isinstance(cat, str) or cat not in categories:
                            raise ValueError("invalid_catalog_entry")
                        try:
                            conn.execute("INSERT INTO zero_cue_entries VALUES (?,?,?)", (_MAP_KIND_CODES[name], key, cat))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("duplicate_catalog_hash") from exc
                        count += 1
                        if parser.peek() != ",":
                            break
                        parser.expect(",")
                        if parser.peek() == "}":
                            raise ValueError("catalog_trailing_comma")
                    parser.expect("}")
                else:
                    parser.value()  # Bounded metadata, never interpreted as SQL.
                if parser.peek() != ",":
                    break
                parser.expect(",")
                if parser.peek() == "}":
                    raise ValueError("catalog_trailing_comma")
            parser.expect("}")
            if parser.peek():
                raise ValueError("catalog_trailing_data")
            if not top_keys.intersection(_MAP_KIND_CODES):
                raise ValueError("missing_catalog_maps")
            sha = parser.hasher.hexdigest()
        conn.executemany("INSERT INTO catalog_meta VALUES (?,?)", [("schema_version","2.1.0"),("source_sha256",sha),("total_entries",str(count))])
        conn.commit()
        conn.close()
        tmp_dst.replace(dst)
    except BaseException:
        conn.close()
        tmp_dst.unlink(missing_ok=True)
        raise
    return {"sqlite_path":str(dst),"source_sha256":sha,"total_entries":count}

class _SQLiteCatalogMapView:
    """Dict-compatible read-only view over a single `map_kind` in `zero_cue_catalog.sqlite`."""
    def __init__(self, db_getter: Callable[[], sqlite3.Connection], map_kind: int) -> None:
        self._db_getter = db_getter
        self._map_kind = int(map_kind)
    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        row = self._db_getter().execute(
            "SELECT 1 FROM zero_cue_entries WHERE map_kind = ? AND key_hash = ? LIMIT 1",
            (self._map_kind, key),
        ).fetchone()
        return row is not None
    def __getitem__(self, key: str) -> str:
        row = self._db_getter().execute(
            "SELECT category FROM zero_cue_entries WHERE map_kind = ? AND key_hash = ? LIMIT 1",
            (self._map_kind, key),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

class SQLiteZeroCueCatalog:
    """Drop-in low-memory replacement for `validated_zero_cue_catalog.json` dict."""
    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self._local = threading.local()
        self._views = {
            name: _SQLiteCatalogMapView(self._conn, code)
            for name, code in _MAP_KIND_CODES.items()
        }
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True, check_same_thread=False)
            conn.execute("PRAGMA mmap_size=0")
            conn.execute("PRAGMA cache_size=-4000")
            conn.execute("PRAGMA temp_store=FILE")
            self._local.conn = conn
        return conn
    def get(self, key: str, default: Any = None) -> Any:
        if key in self._views:
            return self._views[key]
        return default
    def __getitem__(self, key: str) -> Any:
        if key in self._views:
            return self._views[key]
        raise KeyError(key)
    def total_entries(self) -> int:
        row = self._conn().execute(
            "SELECT value FROM catalog_meta WHERE key = 'total_entries'"
        ).fetchone()
        if row is not None:
            return int(row[0])
        c_row = self._conn().execute("SELECT COUNT(*) FROM zero_cue_entries").fetchone()
        return int(c_row[0]) if c_row else 0
    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

# ---------------------------------------------------------------------------
# 3. BitTorrent `.torrent` Generator + Peer-Wire TCP Seeder + BEP-44/46 UDP Node
# ---------------------------------------------------------------------------
@dataclass
class SeededTorrentBundle:
    bundle_name: str
    infohash_20: bytes
    infohash_hex: str
    torrent_bytes: bytes
    info_bytes: bytes
    piece_length: int
    piece_count: int
    total_bytes: int
    merkle_root_hex: str
    seq: int
    bep44_target_hex: str
    magnet_uri: str
    payload_bytes: bytes

def _compute_piece_merkle_root(piece_sha256_digests: Sequence[bytes]) -> str:
    if not piece_sha256_digests:
        return hashlib.sha256(b"armiger-empty-merkle").hexdigest()
    layer = [ hashlib.sha256(b"\x00" + d).digest() for d in piece_sha256_digests ]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        next_layer: List[bytes] = []
        for i in range(0, len(layer), 2):
            next_layer.append(hashlib.sha256(b"\x01" + layer[i] + layer[i + 1]).digest())
        layer = next_layer
    return layer[0].hex()

def build_torrent_bundle(
    bundle_name: str,
    files_map: Mapping[str, bytes],
    pubkey_32: bytes,
    salt: bytes,
    seq: int = 1,
    piece_length: int = 262144,
    webseed_urls: Optional[Sequence[str]] = None,
) -> SeededTorrentBundle:
    """Build a canonical bencoded BEP-3/BEP-46 `.torrent` bundle with SHA-1 pieces and
    a SHA-256 Merkle root over all pieces.
    """
    sorted_names = sorted(files_map.keys())
    concat = bytearray()
    files_info_list: List[Dict[bytes, Any]] = []
    for fname in sorted_names:
        fbytes = files_map[fname]
        concat.extend(fbytes)
        files_info_list.append(
            {
                b"length": len(fbytes),
                b"path": [part.encode("utf-8") for part in fname.split("/") if part],
                b"sha256": hashlib.sha256(fbytes).digest(),
            }
        )
    payload_bytes = bytes(concat)
    piece_sha1_parts = bytearray()
    piece_sha256_list: List[bytes] = []
    for offset in range(0, max(1, len(payload_bytes)), piece_length):
        chunk = payload_bytes[offset:offset + piece_length]
        piece_sha1_parts.extend(hashlib.sha1(chunk).digest())
        piece_sha256_list.append(hashlib.sha256(chunk).digest())
    merkle_root_hex = _compute_piece_merkle_root(piece_sha256_list)
    info_dict: Dict[bytes, Any] = {
        b"files": files_info_list,
        b"merkle_root": bytes.fromhex(merkle_root_hex),
        b"name": bundle_name.encode("utf-8"),
        b"piece length": int(piece_length),
        b"pieces": bytes(piece_sha1_parts),
    }
    info_bytes = bencode(info_dict)
    infohash_20 = hashlib.sha1(info_bytes).digest()
    infohash_hex = infohash_20.hex()
    target_20 = compute_bep44_target(pubkey_32, salt)
    salt_param = f"&s={salt.hex()}" if salt else ""
    magnet_uri = (
        f"magnet:?xt=urn:btih:{infohash_hex}"
        f"&dn={bundle_name}"
        f"&xs=urn:btpk:{pubkey_32.hex()}"
        f"{salt_param}"
    )
    metainfo: Dict[bytes, Any] = {
        b"comment": f"Armiger v3.8.0 Verified Bundle ({bundle_name})".encode("utf-8"),
        b"created by": b"armiger-oci-dht/3.8.0",
        b"info": info_dict,
    }
    if webseed_urls:
        clean_urls = [u.strip().encode("utf-8") for u in webseed_urls if str(u).strip().startswith("https://")]
        if clean_urls:
            metainfo[b"url-list"] = clean_urls
    torrent_bytes = bencode(metainfo)
    return SeededTorrentBundle(
        bundle_name=bundle_name,
        infohash_20=infohash_20,
        infohash_hex=infohash_hex,
        torrent_bytes=torrent_bytes,
        info_bytes=info_bytes,
        piece_length=piece_length,
        piece_count=max(1, len(piece_sha256_list)),
        total_bytes=len(payload_bytes),
        merkle_root_hex=merkle_root_hex,
        seq=int(seq),
        bep44_target_hex=target_20.hex(),
        magnet_uri=magnet_uri,
        payload_bytes=payload_bytes,
    )


BT_PROTOCOL_HEADER = b"\x13BitTorrent protocol"
BT_RESERVED_BEP10 = b"\x00\x00\x00\x00\x00\x10\x00\x00"
BT_MSG_CHOKE = 0
BT_MSG_UNCHOKE = 1
BT_MSG_INTERESTED = 2
BT_MSG_BITFIELD = 5
BT_MSG_REQUEST = 6
BT_MSG_PIECE = 7
BT_MSG_EXTENDED = 20
BT_EXT_HANDSHAKE_ID = 0
BT_UT_METADATA_ID = 1
BT_METADATA_PIECE_SIZE = 16384


class BitTorrentPeerWireSeeder:
    """TCP BitTorrent Peer-Wire Protocol (BEP-3 + BEP-9/10 `ut_metadata`) seeder for Armiger bundles.
    Strictly gated by `VPNEgressGuard` so zero bytes are served and active client streams are
    closed immediately if `tun0` drops.
    """
    MAX_CONCURRENT_CLIENTS = 32

    def __init__(
        self,
        vpn_guard: VPNEgressGuard,
        bundles_by_infohash: Mapping[bytes, SeededTorrentBundle],
        host: str = "0.0.0.0",
        port: int = 6882,
        peer_id: Optional[bytes] = None,
    ) -> None:
        self.vpn_guard = vpn_guard
        self.bundles_by_infohash: Dict[bytes, SeededTorrentBundle] = dict(bundles_by_infohash)
        self.host = host
        self.port = int(port)
        self.peer_id: bytes = peer_id if (peer_id and len(peer_id) == 20) else b"-AM2100-" + os.urandom(12)
        self._server: Optional[asyncio.AbstractServer] = None
        self._active_writers: Set[asyncio.StreamWriter] = set()
        self.bytes_seeded: int = 0
        self.pieces_served: int = 0
        self.metadata_pieces_served: int = 0
        self.vpn_guard.register_on_drop(lambda _st: self.close())

    @property
    def is_serving(self) -> bool:
        return self._server is not None

    async def start(self) -> Tuple[str, int]:
        self.vpn_guard.assert_egress_or_raise()
        if self._server is not None:
            return self.host, self.port
        self._server = await asyncio.start_server(self._handle_client, self.host, self.port, reuse_address=True)
        sock = self._server.sockets[0]
        sockname = sock.getsockname()
        self.host, self.port = str(sockname[0]), int(sockname[1])
        return self.host, self.port

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        for w in list(self._active_writers):
            try:
                w.close()
            except Exception:
                pass
        self._active_writers.clear()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(self._active_writers) >= self.MAX_CONCURRENT_CLIENTS:
            writer.close()
            return
        self._active_writers.add(writer)
        try:
            if not self.vpn_guard.require_for_packet():
                writer.close()
                return
            hs = await asyncio.wait_for(reader.readexactly(68), timeout=5.0)
            if hs[:20] != BT_PROTOCOL_HEADER:
                writer.close()
                return
            peer_supports_bep10 = (hs[25] & 0x10) != 0
            info_hash = hs[28:48]
            bundle = self.bundles_by_infohash.get(info_hash)
            if bundle is None or not self.vpn_guard.require_for_packet():
                writer.close()
                return
            reply_reserved = BT_RESERVED_BEP10 if peer_supports_bep10 else (b"\x00" * 8)
            reply_hs = BT_PROTOCOL_HEADER + reply_reserved + info_hash + self.peer_id
            writer.write(reply_hs)
            if peer_supports_bep10:
                ext_hs_payload = bencode(
                    {
                        b"m": {b"ut_metadata": BT_UT_METADATA_ID},
                        b"metadata_size": len(bundle.info_bytes),
                    }
                )
                writer.write(
                    struct.pack("!IBB", 2 + len(ext_hs_payload), BT_MSG_EXTENDED, BT_EXT_HANDSHAKE_ID)
                    + ext_hs_payload
                )
            bf_len = (bundle.piece_count + 7) // 8
            bf_bytes = bytearray([0xFF] * bf_len)
            rem_bits = (bf_len * 8) - bundle.piece_count
            if rem_bits > 0 and bf_len > 0:
                bf_bytes[-1] &= (0xFF << rem_bits) & 0xFF
            writer.write(struct.pack("!IB", 1 + len(bf_bytes), BT_MSG_BITFIELD) + bytes(bf_bytes))
            writer.write(struct.pack("!IB", 1, BT_MSG_UNCHOKE))
            await writer.drain()
            peer_ut_metadata_id = BT_UT_METADATA_ID
            while True:
                hdr = await asyncio.wait_for(reader.readexactly(4), timeout=10.0)
                (msg_len,) = struct.unpack("!I", hdr)
                if msg_len == 0:
                    continue
                if msg_len > 131072:
                    break
                body = await asyncio.wait_for(reader.readexactly(msg_len), timeout=10.0)
                if not self.vpn_guard.require_for_packet():
                    break
                msg_id = body[0]
                if msg_id == BT_MSG_INTERESTED:
                    writer.write(struct.pack("!IB", 1, BT_MSG_UNCHOKE))
                    await writer.drain()
                elif msg_id == BT_MSG_EXTENDED and len(body) >= 2:
                    from armiger.oci_vpn_guard import _bdecode_at, bdecode_krpc
                    ext_id = body[1]
                    if ext_id == BT_EXT_HANDSHAKE_ID:
                        try:
                            ext_dict = bdecode_krpc(body[2:])
                            if isinstance(ext_dict, dict) and isinstance(ext_dict.get(b"m"), dict):
                                ut_id = ext_dict[b"m"].get(b"ut_metadata")
                                if type(ut_id) is int and 0 < ut_id <= 255:
                                    peer_ut_metadata_id = ut_id
                        except Exception:
                            break
                    elif ext_id == BT_UT_METADATA_ID:
                        try:
                            req_dict, end_idx = _bdecode_at(body, 2, max_depth=4, max_items=16, counter=[0])
                        except Exception:
                            break
                        if not isinstance(req_dict, dict) or end_idx != len(body):
                            break
                        msg_type = req_dict.get(b"msg_type")
                        piece_idx = req_dict.get(b"piece")
                        total_meta_pieces = max(
                            1,
                            (len(bundle.info_bytes) + BT_METADATA_PIECE_SIZE - 1) // BT_METADATA_PIECE_SIZE,
                        )
                        if msg_type == 0 and type(piece_idx) is int and 0 <= piece_idx < total_meta_pieces:
                            m_start = piece_idx * BT_METADATA_PIECE_SIZE
                            m_end = min(len(bundle.info_bytes), m_start + BT_METADATA_PIECE_SIZE)
                            m_chunk = bundle.info_bytes[m_start:m_end]
                            resp_hdr = bencode(
                                {
                                    b"msg_type": 1,
                                    b"piece": piece_idx,
                                    b"total_size": len(bundle.info_bytes),
                                }
                            )
                            pkt = (
                                struct.pack(
                                    "!IBB",
                                    2 + len(resp_hdr) + len(m_chunk),
                                    BT_MSG_EXTENDED,
                                    peer_ut_metadata_id,
                                )
                                + resp_hdr
                                + m_chunk
                            )
                            writer.write(pkt)
                            await writer.drain()
                            self.metadata_pieces_served += 1
                        else:
                            rej_hdr = bencode({b"msg_type": 2, b"piece": piece_idx if type(piece_idx) is int else 0})
                            writer.write(
                                struct.pack("!IBB", 2 + len(rej_hdr), BT_MSG_EXTENDED, peer_ut_metadata_id) + rej_hdr
                            )
                            await writer.drain()
                elif msg_id == BT_MSG_REQUEST and len(body) == 13:
                    idx, begin, length = struct.unpack("!III", body[1:13])
                    if length <= 0 or length > 65536:
                        break
                    start_off = idx * bundle.piece_length + begin
                    end_off = start_off + length
                    if start_off < 0 or end_off > len(bundle.payload_bytes):
                        break
                    block = bundle.payload_bytes[start_off:end_off]
                    pkt = struct.pack("!IBII", 9 + len(block), BT_MSG_PIECE, idx, begin) + block
                    writer.write(pkt)
                    await writer.drain()
                    self.bytes_seeded += len(block)
                    self.pieces_served += 1
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError, OSError):
            pass
        finally:
            self._active_writers.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


MIN_TORRENT_PIECE_LENGTH = 16384


def _validate_torrent_info_geometry(
    info: Any,
    max_payload_bytes: int = 67108864,
    min_piece_length: int = MIN_TORRENT_PIECE_LENGTH,
) -> Tuple[int, int, int, bytes, bytes]:
    """Validate untrusted torrent `info` dictionary piece geometry and hash-vector bounds
    before downloading webseed data or allocating per-piece buffers.
    """
    if not isinstance(info, dict) or not isinstance(info.get(b"files"), list) or not info[b"files"]:
        raise ValueError("invalid_info_dict")
    raw_plen = info.get(b"piece length")
    if type(raw_plen) is not int or raw_plen < min_piece_length or raw_plen > max_payload_bytes:
        raise ValueError("invalid_piece_length")
    piece_length = int(raw_plen)
    total_bytes = 0
    for f in info[b"files"]:
        flen = f.get(b"length") if isinstance(f, dict) else None
        if type(flen) is not int or flen < 0 or flen > max_payload_bytes:
            raise ValueError("invalid_file_length")
        total_bytes += int(flen)
    if total_bytes < 0 or total_bytes > max_payload_bytes:
        raise ValueError("payload_size_exceeds_limit")
    num_pieces = max(1, (total_bytes + piece_length - 1) // piece_length) if total_bytes > 0 else 0
    max_pieces = max(1, (max_payload_bytes + min_piece_length - 1) // min_piece_length)
    if num_pieces > max_pieces:
        raise ValueError("invalid_piece_length")
    raw_pieces = info.get(b"pieces")
    if not isinstance(raw_pieces, (bytes, bytearray)) or len(raw_pieces) != num_pieces * 20:
        raise ValueError("invalid_pieces_hash_vector")
    raw_merkle = info.get(b"merkle_root")
    if not isinstance(raw_merkle, (bytes, bytearray)) or len(raw_merkle) != 32:
        raise ValueError("invalid_merkle_root")
    return piece_length, total_bytes, num_pieces, bytes(raw_pieces), bytes(raw_merkle)


def verify_info_and_payload(
    info_bytes: bytes,
    payload: bytes,
    expected_infohash_20: Optional[bytes] = None,
    max_payload_bytes: int = 67108864,
) -> Dict[str, Any]:
    """Verify raw `info` dictionary bytes against `expected_infohash_20`, every SHA-1 piece,
    the SHA-256 Merkle root, and each file's SHA-256 digest without materializing chunk copies.
    """
    from armiger.oci_vpn_guard import bdecode
    computed_ih = hashlib.sha1(info_bytes).digest()
    if expected_infohash_20 is not None and computed_ih != expected_infohash_20:
        raise ValueError("infohash_mismatch")
    info = bdecode(info_bytes)
    piece_length, total_bytes, num_pieces, raw_pieces, raw_merkle = _validate_torrent_info_geometry(
        info, max_payload_bytes=max_payload_bytes
    )
    if len(payload) != total_bytes:
        raise ValueError("payload_length_mismatch")
    files = info[b"files"]
    mv = memoryview(payload)
    piece_sha256_digests: List[bytes] = []
    if num_pieces == 0:
        pass
    elif total_bytes == 0:
        if hashlib.sha1(b"").digest() != raw_pieces[:20]:
            raise ValueError("piece_hash_mismatch")
        piece_sha256_digests.append(hashlib.sha256(b"").digest())
    else:
        for p_idx in range(num_pieces):
            start_off = p_idx * piece_length
            end_off = min(total_bytes, start_off + piece_length)
            piece_view = mv[start_off:end_off]
            if hashlib.sha1(piece_view).digest() != raw_pieces[p_idx * 20 : (p_idx + 1) * 20]:
                raise ValueError("piece_hash_mismatch")
            piece_sha256_digests.append(hashlib.sha256(piece_view).digest())
    root = _compute_piece_merkle_root(piece_sha256_digests)
    if root != raw_merkle.hex():
        raise ValueError("merkle_root_mismatch")
    offset = 0
    extracted_files: Dict[str, bytes] = {}
    for f in files:
        end = offset + f[b"length"]
        f_slice = payload[offset:end]
        if hashlib.sha256(f_slice).digest() != f[b"sha256"]:
            raise ValueError("file_hash_mismatch")
        rel_path = "/".join(p.decode("utf-8") for p in f[b"path"])
        extracted_files[rel_path] = f_slice
        offset = end
    return {
        "infohash_hex": computed_ih.hex(),
        "bytes_verified": len(payload),
        "pieces_verified": num_pieces,
        "merkle_root": root,
        "merkle_root_sha256": root,
        "files": extracted_files,
    }


def verify_torrent_payload(
    torrent_bytes: bytes,
    payload: bytes,
    expected_infohash_20: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Verify metainfo length, every SHA-1 piece, SHA-256 Merkle root and file hash.

    The caller must pin metainfo to the authenticated pointer's infohash before using this check.
    """
    from armiger.oci_vpn_guard import bdecode
    meta = bdecode(torrent_bytes)
    info = meta[b"info"]
    raw_info = bencode(info)
    return verify_info_and_payload(raw_info, payload, expected_infohash_20=expected_infohash_20)


async def fetch_bundle_via_peer_wire(
    host: str,
    port: int,
    infohash_20: bytes,
    timeout: float = 10.0,
    max_metadata_bytes: int = 1048576,
    max_payload_bytes: int = 67108864,
    max_total_seconds: Optional[float] = None,
    pipeline_depth: int = 16,
) -> Dict[str, Any]:
    """Bootstrap a bundle from ONLY its 20-byte `infohash_20` over TCP Peer-Wire using
    BEP-10 + BEP-9 (`ut_metadata`) to fetch and authenticate `info_bytes`, followed by
    BEP-3 piece requests and full SHA-1 / SHA-256 Merkle verification.
    """
    from armiger.oci_vpn_guard import _bdecode_at, bdecode, bdecode_krpc
    if not isinstance(infohash_20, (bytes, bytearray)) or len(infohash_20) != 20:
        raise ValueError("infohash_20 must be 20 bytes")
    loop = asyncio.get_running_loop()
    total_budget = float(max_total_seconds) if max_total_seconds is not None else max(float(timeout) * 10.0, 30.0)
    total_deadline = loop.time() + total_budget
    progress_deadline = loop.time() + float(timeout)

    def _rem_timeout() -> float:
        now = loop.time()
        rem_total = total_deadline - now
        if rem_total <= 0:
            raise asyncio.TimeoutError("peer_wire_total_deadline_exceeded")
        rem = progress_deadline - now
        if rem <= 0:
            raise asyncio.TimeoutError("peer_wire_no_progress_timeout")
        return min(rem, rem_total)

    async def _wait_io(coro):
        try:
            return await asyncio.wait_for(coro, timeout=_rem_timeout())
        except asyncio.TimeoutError as exc:
            if str(exc) in ("peer_wire_total_deadline_exceeded", "peer_wire_no_progress_timeout"):
                raise
            if loop.time() >= total_deadline:
                raise asyncio.TimeoutError("peer_wire_total_deadline_exceeded") from exc
            raise asyncio.TimeoutError("peer_wire_no_progress_timeout") from exc

    reader, writer = await _wait_io(asyncio.open_connection(host, int(port)))
    try:
        hs = BT_PROTOCOL_HEADER + BT_RESERVED_BEP10 + bytes(infohash_20) + (b"-AM21CL-" + os.urandom(12))
        writer.write(hs)
        ext_hs = bencode({b"m": {b"ut_metadata": BT_UT_METADATA_ID}})
        writer.write(struct.pack("!IBB", 2 + len(ext_hs), BT_MSG_EXTENDED, BT_EXT_HANDSHAKE_ID) + ext_hs)
        await writer.drain()

        reply_hs = await _wait_io(reader.readexactly(68))
        if reply_hs[:20] != BT_PROTOCOL_HEADER or reply_hs[28:48] != bytes(infohash_20):
            raise ValueError("peer_wire_handshake_mismatch")
        if (reply_hs[25] & 0x10) == 0:
            raise ValueError("peer_does_not_support_bep10_extensions")

        progress_deadline = loop.time() + float(timeout)
        non_progress_msgs = 0
        max_non_progress_msgs = 32
        handshake_seen = False

        def _mark_progress() -> None:
            nonlocal progress_deadline, non_progress_msgs
            progress_deadline = loop.time() + float(timeout)
            non_progress_msgs = 0

        def _mark_non_progress() -> None:
            nonlocal non_progress_msgs
            non_progress_msgs += 1
            if non_progress_msgs > max_non_progress_msgs:
                raise ValueError("peer_wire_excessive_non_progress_messages")

        peer_ut_metadata_id: Optional[int] = None
        metadata_size: Optional[int] = None
        meta_pieces: Dict[int, bytes] = {}
        retained_meta_bytes = 0
        total_meta_pieces: Optional[int] = None
        peer_choking = True

        while total_meta_pieces is None or len(meta_pieces) < total_meta_pieces:
            hdr = await _wait_io(reader.readexactly(4))
            (msg_len,) = struct.unpack("!I", hdr)
            if msg_len == 0:
                _mark_non_progress()
                continue
            if msg_len > 131072:
                raise ValueError("oversized_peer_wire_message")
            body = await _wait_io(reader.readexactly(msg_len))
            msg_id = body[0]
            if msg_id == BT_MSG_UNCHOKE:
                peer_choking = False
                _mark_non_progress()
                continue
            if msg_id == BT_MSG_CHOKE:
                peer_choking = True
                _mark_non_progress()
                continue
            if msg_id == BT_MSG_EXTENDED and len(body) >= 2:
                ext_id = body[1]
                if ext_id == BT_EXT_HANDSHAKE_ID:
                    if handshake_seen:
                        raise ValueError("duplicate_bep10_handshake")
                    ext_dict = bdecode_krpc(body[2:])
                    if not isinstance(ext_dict, dict) or not isinstance(ext_dict.get(b"m"), dict):
                        raise ValueError("invalid_bep10_handshake")
                    ut_id = ext_dict[b"m"].get(b"ut_metadata")
                    m_size = ext_dict.get(b"metadata_size")
                    if type(ut_id) is not int or ut_id <= 0 or type(m_size) is not int or m_size <= 0 or m_size > max_metadata_bytes:
                        raise ValueError("invalid_ut_metadata_handshake")
                    handshake_seen = True
                    peer_ut_metadata_id = ut_id
                    metadata_size = m_size
                    total_meta_pieces = (metadata_size + BT_METADATA_PIECE_SIZE - 1) // BT_METADATA_PIECE_SIZE
                    _mark_progress()
                    for p_idx in range(total_meta_pieces):
                        req = bencode({b"msg_type": 0, b"piece": p_idx})
                        writer.write(struct.pack("!IBB", 2 + len(req), BT_MSG_EXTENDED, peer_ut_metadata_id) + req)
                    await writer.drain()
                elif ext_id == BT_UT_METADATA_ID:
                    meta_hdr, split_pos = _bdecode_at(body, 2, max_depth=4, max_items=16, counter=[0])
                    if not isinstance(meta_hdr, dict) or meta_hdr.get(b"msg_type") != 1:
                        raise ValueError("ut_metadata_piece_rejected")
                    p_idx = meta_hdr.get(b"piece")
                    if (
                        type(p_idx) is not int
                        or total_meta_pieces is None
                        or metadata_size is None
                        or not (0 <= p_idx < total_meta_pieces)
                        or p_idx in meta_pieces
                    ):
                        raise ValueError("invalid_or_duplicate_ut_metadata_piece")
                    hdr_total = meta_hdr.get(b"total_size")
                    if hdr_total is not None and (type(hdr_total) is not int or hdr_total != metadata_size):
                        raise ValueError("ut_metadata_total_size_mismatch")
                    expected_meta_len = min(BT_METADATA_PIECE_SIZE, metadata_size - p_idx * BT_METADATA_PIECE_SIZE)
                    m_chunk = body[split_pos:]
                    if len(m_chunk) != expected_meta_len or retained_meta_bytes + len(m_chunk) > max_metadata_bytes:
                        raise ValueError("ut_metadata_piece_length_mismatch")
                    meta_pieces[p_idx] = m_chunk
                    retained_meta_bytes += len(m_chunk)
                    _mark_progress()
                else:
                    _mark_non_progress()
            else:
                _mark_non_progress()

        info_bytes = b"".join(meta_pieces[i] for i in range(total_meta_pieces))
        if metadata_size is not None and len(info_bytes) != metadata_size:
            raise ValueError("ut_metadata_size_mismatch")
        if hashlib.sha1(info_bytes).digest() != bytes(infohash_20):
            raise ValueError("ut_metadata_infohash_mismatch")

        info_dict = bdecode(info_bytes)
        if not isinstance(info_dict, dict) or not isinstance(info_dict.get(b"files"), list) or not info_dict[b"files"]:
            raise ValueError("invalid_info_dict")
        raw_plen = info_dict.get(b"piece length")
        if type(raw_plen) is not int or raw_plen < 16384 or raw_plen > max_payload_bytes:
            raise ValueError("invalid_piece_length")
        piece_length = int(raw_plen)
        total_bytes = 0
        for f_entry in info_dict[b"files"]:
            flen = f_entry.get(b"length") if isinstance(f_entry, dict) else None
            if type(flen) is not int or flen < 0 or flen > max_payload_bytes:
                raise ValueError("invalid_file_length")
            total_bytes += int(flen)
        if total_bytes < 0 or total_bytes > max_payload_bytes:
            raise ValueError("payload_size_exceeds_limit")
        num_pieces = max(1, (total_bytes + piece_length - 1) // piece_length) if total_bytes > 0 else 0
        raw_pieces = info_dict.get(b"pieces")
        if not isinstance(raw_pieces, (bytes, bytearray)) or len(raw_pieces) != num_pieces * 20:
            raise ValueError("invalid_pieces_hash_vector")

        writer.write(struct.pack("!IB", 1, BT_MSG_INTERESTED))
        await writer.drain()

        block_size = 16384
        all_blocks: List[Tuple[int, int, int, int]] = []
        g_off = 0
        for p_idx in range(num_pieces):
            p_total = min(piece_length, total_bytes - p_idx * piece_length)
            p_off = 0
            while p_off < p_total:
                req_len = min(block_size, p_total - p_off)
                all_blocks.append((p_idx, p_off, req_len, g_off))
                p_off += req_len
                g_off += req_len

        payload_buf = bytearray(total_bytes)
        received_blocks: set[Tuple[int, int]] = set()
        inflight: Dict[Tuple[int, int], Tuple[int, int]] = {}
        next_block_idx = 0
        max_window = max(1, min(int(pipeline_depth), 64))

        async def _fill_pipeline() -> None:
            nonlocal next_block_idx
            if peer_choking:
                return
            wrote = False
            while len(inflight) < max_window and next_block_idx < len(all_blocks):
                b_pidx, b_poff, b_rlen, b_goff = all_blocks[next_block_idx]
                next_block_idx += 1
                if (b_pidx, b_poff) in received_blocks or (b_pidx, b_poff) in inflight:
                    continue
                inflight[(b_pidx, b_poff)] = (b_rlen, b_goff)
                writer.write(struct.pack("!IBIII", 13, BT_MSG_REQUEST, b_pidx, b_poff, b_rlen))
                wrote = True
            if wrote:
                await writer.drain()

        await _fill_pipeline()
        while len(received_blocks) < len(all_blocks):
            hdr = await _wait_io(reader.readexactly(4))
            (msg_len,) = struct.unpack("!I", hdr)
            if msg_len == 0:
                _mark_non_progress()
                continue
            if msg_len > 9 + block_size or msg_len > 131072:
                raise ValueError("oversized_peer_wire_piece_message")
            body = await _wait_io(reader.readexactly(msg_len))
            msg_id = body[0]
            if msg_id == BT_MSG_UNCHOKE:
                peer_choking = False
                _mark_non_progress()
                await _fill_pipeline()
                continue
            if msg_id == BT_MSG_CHOKE:
                peer_choking = True
                inflight.clear()
                next_block_idx = 0
                _mark_non_progress()
                continue
            if not peer_choking and msg_id == BT_MSG_PIECE and len(body) >= 9:
                r_idx, r_begin = struct.unpack("!II", body[1:9])
                exp_entry = inflight.pop((r_idx, r_begin), None)
                if exp_entry is not None:
                    exp_len, exp_goff = exp_entry
                    chunk = body[9:]
                    if len(chunk) != exp_len:
                        raise ValueError("short_piece_block")
                    payload_buf[exp_goff : exp_goff + exp_len] = chunk
                    received_blocks.add((r_idx, r_begin))
                    _mark_progress()
                    await _fill_pipeline()
                    continue
            _mark_non_progress()

        verified = verify_info_and_payload(info_bytes, bytes(payload_buf), expected_infohash_20=bytes(infohash_20), max_payload_bytes=max_payload_bytes)
        verified["info_bytes"] = info_bytes
        verified["payload_bytes"] = bytes(payload_buf)
        return verified
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


_MAX_INFLIGHT_DNS_WORKERS = 4
_DNS_ADMISSION_SEMAPHORE = threading.BoundedSemaphore(_MAX_INFLIGHT_DNS_WORKERS)
_DNS_INFLIGHT_LOCK = threading.Lock()
_DNS_INFLIGHT_COUNT = 0


def get_inflight_dns_worker_count() -> int:
    with _DNS_INFLIGHT_LOCK:
        return _DNS_INFLIGHT_COUNT


def _https_get_with_deadline(
    url: str,
    max_bytes: int,
    timeout: float,
    deadline: float,
    max_redirects: int = 5,
    connection_factory: Optional[Callable[..., Any]] = None,
    resolver_fn: Optional[Callable[[str, int], Sequence[Any]]] = None,
    socket_factory: Optional[Callable[[int, int, int], Any]] = None,
    tls_wrap: bool = True,
) -> bytes:
    """Perform an HTTPS GET bounded by an enforceable overall monotonic `deadline` covering
    DNS resolution, per-address TCP connection setup, TLS handshake, HTTP response headers,
    redirects, and body reads. When `deadline` expires, a watchdog timer shuts down and closes
    any registered socket immediately.
    """
    import http.client
    import queue
    import socket
    import time
    import urllib.parse

    if not url.lower().startswith("https://"):
        raise ValueError("bep19_webseed_requires_https")

    timed_out = [False]
    active_lock = threading.Lock()
    active_objs: List[Any] = []

    def _register_obj(obj: Any) -> None:
        if obj is None:
            return
        with active_lock:
            active_objs.append(obj)
            is_expired = timed_out[0]
        if is_expired:
            _force_close(obj)

    def _force_close(obj: Any) -> None:
        try:
            sock = getattr(obj, "sock", None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(obj, "shutdown"):
                obj.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            obj.close()
        except Exception:
            pass

    def _on_deadline() -> None:
        with active_lock:
            timed_out[0] = True
            objs = list(active_objs)
        for obj in objs:
            _force_close(obj)

    def _rem_timeout() -> float:
        rem = deadline - time.monotonic()
        if rem <= 0 or timed_out[0]:
            raise TimeoutError("webseed_total_deadline_exceeded")
        return min(float(timeout), rem)

    def _resolve_with_deadline(host: str, port: int) -> Sequence[Any]:
        global _DNS_INFLIGHT_COUNT
        rem_dns = _rem_timeout()
        if resolver_fn is None:
            import ipaddress
            try:
                ip_obj = ipaddress.ip_address(host)
                af = socket.AF_INET6 if ip_obj.version == 6 else socket.AF_INET
                sa = (host, int(port), 0, 0) if ip_obj.version == 6 else (host, int(port))
                return [(af, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sa)]
            except ValueError:
                pass
        if not _DNS_ADMISSION_SEMAPHORE.acquire(blocking=False):
            raise TimeoutError("webseed_dns_pool_saturated")
        with _DNS_INFLIGHT_LOCK:
            _DNS_INFLIGHT_COUNT += 1
        eff_resolver = resolver_fn or (
            lambda h, p: socket.getaddrinfo(h, p, socket.AF_UNSPEC, socket.SOCK_STREAM)
        )
        res_q: "queue.Queue[Tuple[bool, Any]]" = queue.Queue(maxsize=1)

        def _dns_worker() -> None:
            global _DNS_INFLIGHT_COUNT
            try:
                infos = eff_resolver(host, port)
                res_q.put((True, infos))
            except Exception as exc:
                res_q.put((False, exc))
            finally:
                with _DNS_INFLIGHT_LOCK:
                    _DNS_INFLIGHT_COUNT -= 1
                _DNS_ADMISSION_SEMAPHORE.release()

        try:
            t_dns = threading.Thread(target=_dns_worker, daemon=True)
            t_dns.start()
        except BaseException:
            with _DNS_INFLIGHT_LOCK:
                _DNS_INFLIGHT_COUNT -= 1
            _DNS_ADMISSION_SEMAPHORE.release()
            raise
        try:
            ok, payload = res_q.get(timeout=rem_dns)
        except queue.Empty as exc:
            if time.monotonic() >= deadline or timed_out[0]:
                raise TimeoutError("webseed_total_deadline_exceeded") from exc
            raise TimeoutError("webseed_dns_timeout") from exc
        _rem_timeout()
        if not ok:
            raise payload
        if not payload:
            raise OSError("dns_resolution_empty")
        return payload

    class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
        def connect(self) -> None:
            addrinfos = _resolve_with_deadline(self.host, self.port)
            eff_sock_factory = socket_factory or socket.socket
            last_err: Optional[Exception] = None
            connected_sock = None
            for res in addrinfos:
                af, socktype, proto, _canonname, sa = res
                rem_addr = _rem_timeout()
                raw_sock = eff_sock_factory(af, socktype, proto)
                _register_obj(raw_sock)
                try:
                    raw_sock.settimeout(rem_addr)
                    if self.source_address:
                        raw_sock.bind(self.source_address)
                    raw_sock.connect(sa)
                    connected_sock = raw_sock
                    break
                except Exception as exc:
                    last_err = exc
                    _force_close(raw_sock)
                    if timed_out[0] or time.monotonic() >= deadline:
                        raise TimeoutError("webseed_total_deadline_exceeded") from exc
            if connected_sock is None:
                if timed_out[0] or time.monotonic() >= deadline:
                    raise TimeoutError("webseed_total_deadline_exceeded")
                if last_err is not None:
                    raise last_err
                raise OSError("connect_failed")
            self.sock = connected_sock
            if self._tunnel_host:
                self._tunnel()
            rem_tls = _rem_timeout()
            connected_sock.settimeout(rem_tls)
            if tls_wrap:
                self.sock = self._context.wrap_socket(connected_sock, server_hostname=self.host)
                _register_obj(self.sock)

    rem_total = deadline - time.monotonic()
    if rem_total <= 0:
        raise TimeoutError("webseed_total_deadline_exceeded")
    watchdog = threading.Timer(rem_total, _on_deadline)
    watchdog.daemon = True
    watchdog.start()
    current_url = url
    try:
        for hop in range(max_redirects + 1):
            hop_timeout = _rem_timeout()
            parsed = urllib.parse.urlsplit(current_url)
            if parsed.scheme.lower() != "https" or not parsed.hostname:
                raise ValueError("bep19_webseed_requires_https")
            port = parsed.port or 443
            target_path = parsed.path or "/"
            if parsed.query:
                target_path = f"{target_path}?{parsed.query}"
            factory = connection_factory or _DeadlineHTTPSConnection
            conn = factory(parsed.hostname, port, timeout=hop_timeout)
            _register_obj(conn)
            try:
                conn.request(
                    "GET",
                    target_path,
                    headers={"User-Agent": "armiger-bep19-verifier/3.8.0", "Connection": "close"},
                )
                if getattr(conn, "sock", None) is not None:
                    _register_obj(conn.sock)
                    conn.sock.settimeout(_rem_timeout())
                resp = conn.getresponse()
                _rem_timeout()
                status = int(getattr(resp, "status", 0))
                if status in (301, 302, 303, 307, 308):
                    if hop >= max_redirects:
                        raise ValueError("bep19_webseed_too_many_redirects")
                    loc = resp.getheader("Location") if hasattr(resp, "getheader") else None
                    if not loc:
                        raise ValueError("bep19_webseed_redirect_missing_location")
                    resolved = urllib.parse.urljoin(current_url, str(loc))
                    if not resolved.lower().startswith("https://"):
                        raise ValueError("bep19_webseed_redirect_requires_https")
                    current_url = resolved
                    continue
                if status != 200:
                    raise ValueError(f"bep19_http_status_{status}")
                cl = resp.getheader("Content-Length") if hasattr(resp, "getheader") else None
                if cl is not None:
                    try:
                        if int(cl) > max_bytes:
                            raise ValueError("bep19_content_length_exceeds_expected")
                    except ValueError as exc:
                        if "bep19_content_length_exceeds_expected" in str(exc):
                            raise
                chunks = bytearray()
                while True:
                    rem = _rem_timeout()
                    if getattr(conn, "sock", None) is not None:
                        try:
                            conn.sock.settimeout(rem)
                        except Exception:
                            pass
                    read_limit = min(16384, max_bytes + 1 - len(chunks))
                    if hasattr(resp, "read1"):
                        chunk = resp.read1(read_limit)
                    elif hasattr(getattr(resp, "fp", None), "read1"):
                        chunk = resp.fp.read1(read_limit)
                    else:
                        chunk = resp.read(min(4096, read_limit))
                    if not chunk:
                        break
                    chunks.extend(chunk)
                    if len(chunks) > max_bytes:
                        raise ValueError("bep19_response_exceeds_expected_length")
                _rem_timeout()
                return bytes(chunks)
            finally:
                _force_close(conn)
        raise ValueError("bep19_webseed_too_many_redirects")
    except TimeoutError as exc:
        if timed_out[0] or time.monotonic() >= deadline or str(exc) == "webseed_total_deadline_exceeded":
            raise TimeoutError("webseed_total_deadline_exceeded") from exc
        if str(exc) in ("webseed_dns_timeout", "webseed_socket_timeout", "webseed_dns_pool_saturated"):
            raise
        raise TimeoutError("webseed_socket_timeout") from exc
    except ValueError:
        if timed_out[0] or time.monotonic() >= deadline:
            raise TimeoutError("webseed_total_deadline_exceeded")
        raise
    except Exception as exc:
        if timed_out[0] or time.monotonic() >= deadline:
            raise TimeoutError("webseed_total_deadline_exceeded") from exc
        if isinstance(exc, socket.timeout):
            raise TimeoutError("webseed_socket_timeout") from exc
        raise
    finally:
        watchdog.cancel()


def fetch_bundle_via_bep19_webseed(
    info_bytes: bytes,
    webseed_base_url: str,
    expected_infohash_20: Optional[bytes] = None,
    fetcher: Optional[Callable[..., bytes]] = None,
    max_payload_bytes: int = 67108864,
    timeout: float = 10.0,
    max_total_seconds: Optional[float] = None,
    connection_factory: Optional[Callable[..., Any]] = None,
    resolver_fn: Optional[Callable[[str, int], Sequence[Any]]] = None,
    socket_factory: Optional[Callable[[int, int, int], Any]] = None,
    tls_wrap: bool = True,
) -> Dict[str, Any]:
    """Download bundle files via BEP-19 HTTP/HTTPS webseed (`<base>/<bundle_name>/<file_path>`
    or flat root fallback) with strict per-file, aggregate byte, and overall transfer deadline limits,
    then verify every SHA-1 piece, SHA-256 Merkle root, and file SHA-256.
    """
    import inspect
    import time
    import urllib.parse
    from armiger.oci_vpn_guard import bdecode
    if max_payload_bytes <= 0:
        raise ValueError("invalid_max_payload_bytes")
    computed_ih = hashlib.sha1(info_bytes).digest()
    if expected_infohash_20 is not None and computed_ih != expected_infohash_20:
        raise ValueError("infohash_mismatch")
    info = bdecode(info_bytes)
    if not isinstance(info, dict) or not isinstance(info.get(b"files"), list) or not info[b"files"]:
        raise ValueError("invalid_info_dict")
    total_expected = 0
    for f in info[b"files"]:
        flen = f.get(b"length") if isinstance(f, dict) else None
        if isinstance(flen, bool) or not isinstance(flen, int) or flen < 0 or flen > max_payload_bytes:
            raise ValueError("invalid_or_oversized_file_length")
        total_expected += flen
    if total_expected > max_payload_bytes:
        raise ValueError("webseed_payload_size_exceeds_limit")
    _validate_torrent_info_geometry(info, max_payload_bytes=max_payload_bytes)
    bundle_name = info[b"name"].decode("utf-8")
    enc_bundle_name = urllib.parse.quote(bundle_name, safe="")
    base = webseed_base_url.rstrip("/")
    ws_budget = float(max_total_seconds) if max_total_seconds is not None else max(float(timeout) * 3.0, 15.0)
    ws_deadline = time.monotonic() + ws_budget

    def _rem_ws_timeout() -> float:
        rem = ws_deadline - time.monotonic()
        if rem <= 0:
            raise TimeoutError("webseed_total_deadline_exceeded")
        return min(float(timeout), rem)

    def _default_get(url: str, max_bytes: int) -> bytes:
        return _https_get_with_deadline(
            url=url,
            max_bytes=max_bytes,
            timeout=timeout,
            deadline=ws_deadline,
            connection_factory=connection_factory,
            resolver_fn=resolver_fn,
            socket_factory=socket_factory,
            tls_wrap=tls_wrap,
        )

    def _call_fetcher(url: str, exp_len: int) -> bytes:
        _rem_ws_timeout()
        if fetcher is None:
            return _default_get(url, exp_len)
        try:
            sig = inspect.signature(fetcher)
            if len(sig.parameters) >= 2:
                data = fetcher(url, exp_len)
            else:
                data = fetcher(url)
        except (ValueError, TypeError) as exc:
            if "bep19_" in str(exc) or "exceeds_expected" in str(exc):
                raise
            data = fetcher(url)
        _rem_ws_timeout()
        if len(data) > exp_len:
            raise ValueError("bep19_response_exceeds_expected_length")
        return data

    concat = bytearray()
    for f in info[b"files"]:
        exp_len = int(f[b"length"])
        rel_path = "/".join(p.decode("utf-8") for p in f[b"path"])
        enc_rel_path = "/".join(urllib.parse.quote(p.decode("utf-8"), safe="") for p in f[b"path"])
        bep19_url = f"{base}/{enc_bundle_name}/{enc_rel_path}"
        try:
            fbytes = _call_fetcher(bep19_url, exp_len)
        except TimeoutError:
            raise
        except ValueError as exc:
            if "exceeds_expected" in str(exc) or "redirect_requires_https" in str(exc):
                raise
            flat_url = f"{base}/{enc_rel_path}"
            fbytes = _call_fetcher(flat_url, exp_len)
        except Exception:
            flat_url = f"{base}/{enc_rel_path}"
            fbytes = _call_fetcher(flat_url, exp_len)
        if len(fbytes) != exp_len:
            raise ValueError(f"bep19_file_length_mismatch:{rel_path}")
        if len(concat) + len(fbytes) > max_payload_bytes:
            raise ValueError("webseed_payload_size_exceeds_limit")
        concat.extend(fbytes)

    verified = verify_info_and_payload(info_bytes, bytes(concat), expected_infohash_20=computed_ih, max_payload_bytes=max_payload_bytes)
    verified["info_bytes"] = info_bytes
    verified["payload_bytes"] = bytes(concat)
    return verified
