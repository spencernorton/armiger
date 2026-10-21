"""Zero-Cue Binary Catalog (`ZCB3`) Packer & Zero-Deserialization `mmap` Reader.

Implements Session 3's `ZCB3` sorted 9-byte binary table format for `validated_zero_cue_catalog.json`
(`929,408` BLAKE2b-64 entries across `anime_map`, `title_year_map`, and `title_only_map`):
  - Header (16 bytes): `4B magic b"ZCB3"` + `3 x uint32 (<3I)` counts `(n_anime, n_title_year, n_title_only)`
  - Body (`9 * N` bytes): 3 contiguous sections in order (`anime_map`, `title_year_map`, `title_only_map`),
    each sorted ascending by 8-byte big-endian BLAKE2b-64 key + 1-byte `uint8` category code.
  - Size reduction:
      * Raw JSON (`25,990,893 B` = `25.99 MB`)
      * Uncompressed `ZCB3` (`8,364,688 B` = `8.36 MB`, `-67.8%`, supports `O(log N)` zero-deserialization `mmap`)
      * `ZCB3` + `zlib-9` (`7,425,374 B` = `7.43 MB`, `-71.4%`)
      * `ZCB3` + `lzma-6` (`6,451,624 B` = `6.45 MB`, `-75.2%`)
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
import json
import lzma
import mmap
from pathlib import Path
import struct
from typing import Any
import zlib

ZCB3_MAGIC = b"ZCB3"
ZCB3_HEADER_STRUCT = struct.Struct("<4s3I")
ZCB3_HEADER_SIZE = ZCB3_HEADER_STRUCT.size  # 16 bytes
ZCB3_RECORD_SIZE = 9  # 8-byte BLAKE2b-64 key + 1-byte category code
ZCB3_SECTION_NAMES: tuple[str, str, str] = ("anime_map", "title_year_map", "title_only_map")
MAX_ZCB3_UNCOMPRESSED_BYTES = 64 * 1024 * 1024  # 64 MiB hard bound against decompression bombs
MAX_ZCB3_LZMA_MEMLIMIT_BYTES = 64 * 1024 * 1024  # 64 MiB LZMA decoder memory bound (supports preset 6)

CAT_TO_CODE: dict[str, int] = {
    "movie": 0,
    "tv_show": 1,
    "ambiguous": 2,
    "xxx": 3,
    "music": 4,
    "comic": 5,
    "game": 6,
    "audiobook": 7,
    "software": 8,
    "ebook": 9,
    "course": 10,
    "unknown": 11,
    "books": 12,
}
CODE_TO_CAT: dict[int, str] = {v: k for k, v in CAT_TO_CODE.items()}


def _hash_catalog_key_bytes(key: str) -> bytes:
    """Compute 8-byte keyed BLAKE2b digest for an unhashed lookup key."""
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"armiger-zc-v2").digest()


def _normalize_lookup_key(key: str | bytes) -> bytes | None:
    """Convert a 16-char hex digest, 8-byte raw digest, or plaintext lookup string into 8 raw bytes."""
    if isinstance(key, (bytes, bytearray, memoryview)):
        b = bytes(key)
        return b if len(b) == 8 else None
    if not isinstance(key, str):
        return None
    if len(key) == 16:
        try:
            return bytes.fromhex(key)
        except ValueError:
            pass
    return _hash_catalog_key_bytes(key)


def pack_zcb3_bytes(catalog: Mapping[str, Any]) -> bytes:
    """Pack `anime_map`, `title_year_map`, and `title_only_map` into deterministic `ZCB3` bytes."""
    packed_sections: list[bytes] = []
    counts: list[int] = []

    for section_name in ZCB3_SECTION_NAMES:
        raw_map = catalog.get(section_name) or {}
        if not isinstance(raw_map, Mapping):
            raise TypeError(f"Expected mapping for {section_name}, got {type(raw_map)!r}")

        parsed_items: list[tuple[bytes, int]] = []
        for k_raw, val_str in raw_map.items():
            if isinstance(k_raw, bytes) and len(k_raw) == 8:
                k_bytes = k_raw
            elif isinstance(k_raw, str) and len(k_raw) == 16:
                try:
                    k_bytes = bytes.fromhex(k_raw)
                except ValueError:
                    k_bytes = _hash_catalog_key_bytes(k_raw)
            elif isinstance(k_raw, str):
                k_bytes = _hash_catalog_key_bytes(k_raw)
            else:
                raise ValueError(f"Invalid key in {section_name}: {k_raw!r}")

            code = CAT_TO_CODE.get(str(val_str), 3)
            parsed_items.append((k_bytes, code))

        parsed_items.sort(key=lambda item: item[0])
        counts.append(len(parsed_items))

        buf = bytearray(len(parsed_items) * ZCB3_RECORD_SIZE)
        offset = 0
        for k_bytes, code in parsed_items:
            buf[offset : offset + 8] = k_bytes
            buf[offset + 8] = code
            offset += ZCB3_RECORD_SIZE
        packed_sections.append(bytes(buf))

    header = ZCB3_HEADER_STRUCT.pack(ZCB3_MAGIC, counts[0], counts[1], counts[2])
    return header + b"".join(packed_sections)


def compress_zcb3_zlib9(zcb3_bytes: bytes) -> bytes:
    """Compress packed `ZCB3` bytes with `zlib` level 9 (~`7.43 MB` for full catalog)."""
    return zlib.compress(zcb3_bytes, level=9)


def compress_zcb3_lzma6(zcb3_bytes: bytes) -> bytes:
    """Compress packed `ZCB3` bytes with `lzma` preset 6 (~`6.45 MB` for full catalog)."""
    return lzma.compress(zcb3_bytes, preset=6)


def decompress_zcb3(
    payload: bytes,
    *,
    max_bytes: int = MAX_ZCB3_UNCOMPRESSED_BYTES,
    lzma_memlimit: int = MAX_ZCB3_LZMA_MEMLIMIT_BYTES,
) -> bytes:
    """Safely decompress raw, `zlib`, or `lzma`-compressed `ZCB3` bytes with bounded output length
    and bounded LZMA decoder memory.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise ValueError("Empty ZCB3 payload")
    max_b = int(max_bytes)
    mem_b = int(lzma_memlimit)
    if max_b <= 0 or mem_b <= 0:
        raise ValueError("max_bytes and lzma_memlimit must be positive")
    raw = bytes(payload)
    if raw.startswith(ZCB3_MAGIC):
        if len(raw) > max_b:
            raise ValueError(f"ZCB3 payload exceeds max_bytes ({len(raw)} > {max_b})")
        return raw

    if raw.startswith(b"\xfd7zXZ\x00"):
        try:
            dec = lzma.LZMADecompressor(memlimit=mem_b)
            out = dec.decompress(raw, max_length=max_b + 1)
        except lzma.LZMAError as exc:
            raise ValueError(f"zcb3_lzma_decompression_error:{exc}") from exc
        if len(out) > max_b or not dec.eof:
            raise ValueError("zcb3_lzma_decompression_limit_exceeded")
        if not out.startswith(ZCB3_MAGIC):
            raise ValueError("Decompressed LZMA payload does not start with ZCB3 magic")
        return out

    try:
        zdec = zlib.decompressobj()
        out = zdec.decompress(raw, max_length=max_b + 1)
        if len(out) > max_b or zdec.unconsumed_tail:
            raise ValueError("zcb3_zlib_decompression_limit_exceeded")
        rem = max_b + 1 - len(out)
        out += zdec.flush(max(1, rem))
    except zlib.error as exc:
        raise ValueError(f"Invalid ZCB3 header or compressed stream: {exc}") from exc

    if len(out) > max_b or zdec.unconsumed_tail:
        raise ValueError("zcb3_zlib_decompression_limit_exceeded")
    if not zdec.eof:
        raise ValueError("zcb3_zlib_incomplete_stream")
    if not out.startswith(ZCB3_MAGIC):
        raise ValueError("Decompressed Zlib payload does not start with ZCB3 magic")
    return out


class ZCB3MapSectionView(Mapping[str, str]):
    """Read-only `O(log N)` binary-search mapping view over a single `ZCB3` section (`mmap` or `bytes`)."""

    def __init__(self, buf: mmap.mmap | bytes, base_offset: int, count: int) -> None:
        self._buf = buf
        self._base = int(base_offset)
        self._count = int(count)

    def __len__(self) -> int:
        return self._count

    def get(self, key: Any, default: Any = None) -> Any:
        target8 = _normalize_lookup_key(key)
        if target8 is None or self._count <= 0:
            return default
        lo = 0
        hi = self._count - 1
        base = self._base
        buf = self._buf
        while lo <= hi:
            mid = (lo + hi) >> 1
            off = base + mid * ZCB3_RECORD_SIZE
            k8 = buf[off : off + 8]
            if k8 == target8:
                code = buf[off + 8]
                return CODE_TO_CAT.get(code, "unknown")
            if k8 < target8:
                lo = mid + 1
            else:
                hi = mid - 1
        return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (str, bytes, bytearray, memoryview)):
            return False
        return self.get(key) is not None

    def __getitem__(self, key: str | bytes) -> str:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return str(val)

    def __iter__(self) -> Iterator[str]:
        base = self._base
        buf = self._buf
        for i in range(self._count):
            off = base + i * ZCB3_RECORD_SIZE
            yield bytes(buf[off : off + 8]).hex()

    def iter_records(self) -> Iterator[tuple[str, str]]:
        """Yield `(key_hex, category_str)` in sorted order without building a dict in memory."""
        base = self._base
        buf = self._buf
        for i in range(self._count):
            off = base + i * ZCB3_RECORD_SIZE
            k_hex = bytes(buf[off : off + 8]).hex()
            code = buf[off + 8]
            yield k_hex, CODE_TO_CAT.get(code, "unknown")


class ZCB3MmapCatalog(Mapping[str, Any]):
    """Zero-deserialization `ZCB3` catalog backed by an OS `mmap` or in-memory `bytes` buffer."""

    def __init__(
        self,
        path_or_bytes: str | Path | bytes | bytearray | memoryview,
    ) -> None:
        self._file_obj: Any = None
        self._mmap_obj: mmap.mmap | None = None
        self._raw_bytes: bytes | None = None
        self.source_path: Path | None = None
        self.storage_mode: str = "memory_zcb3"

        if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
            self._raw_bytes = decompress_zcb3(bytes(path_or_bytes))
            buf: mmap.mmap | bytes = self._raw_bytes
        else:
            p = Path(path_or_bytes)
            if not p.exists():
                raise FileNotFoundError(f"ZCB3 catalog file not found: {p}")
            self.source_path = p
            with p.open("rb") as f_peek:
                head4 = f_peek.read(4)
            if head4 == ZCB3_MAGIC:
                self._file_obj = p.open("rb")
                self._mmap_obj = mmap.mmap(self._file_obj.fileno(), 0, access=mmap.ACCESS_READ)
                buf = self._mmap_obj
                self.storage_mode = "mmap_zcb3"
            else:
                self._raw_bytes = decompress_zcb3(p.read_bytes())
                buf = self._raw_bytes
                self.storage_mode = "compressed_zcb3"

        if len(buf) < ZCB3_HEADER_SIZE:
            self.close()
            raise ValueError(f"ZCB3 buffer too short for 16-byte header: {len(buf)} bytes")

        magic, n_anime, n_title_year, n_title_only = ZCB3_HEADER_STRUCT.unpack(buf[:ZCB3_HEADER_SIZE])
        if magic != ZCB3_MAGIC:
            self.close()
            raise ValueError(f"Invalid ZCB3 magic bytes: {magic!r}")

        total_records = n_anime + n_title_year + n_title_only
        expected_size = ZCB3_HEADER_SIZE + total_records * ZCB3_RECORD_SIZE
        if len(buf) != expected_size:
            self.close()
            raise ValueError(
                f"ZCB3 buffer size mismatch: expected {expected_size} bytes for {total_records} records, got {len(buf)}"
            )

        self.counts: dict[str, int] = {
            "anime_map": int(n_anime),
            "title_year_map": int(n_title_year),
            "title_only_map": int(n_title_only),
        }
        self.total_entries: int = int(total_records)
        self.size_bytes: int = len(buf)

        off_anime = ZCB3_HEADER_SIZE
        off_title_year = off_anime + n_anime * ZCB3_RECORD_SIZE
        off_title_only = off_title_year + n_title_year * ZCB3_RECORD_SIZE

        self._sections: dict[str, ZCB3MapSectionView] = {
            "anime_map": ZCB3MapSectionView(buf, off_anime, n_anime),
            "title_year_map": ZCB3MapSectionView(buf, off_title_year, n_title_year),
            "title_only_map": ZCB3MapSectionView(buf, off_title_only, n_title_only),
        }

    @classmethod
    def from_json_catalog(
        cls,
        json_path_or_dict: str | Path | Mapping[str, Any],
        *,
        out_zcb3_path: str | Path | None = None,
    ) -> ZCB3MmapCatalog:
        """Pack a JSON catalog dict or file into `ZCB3` (optionally writing to disk for `mmap`) and return reader."""
        if isinstance(json_path_or_dict, Mapping):
            doc = json_path_or_dict
        else:
            doc = json.loads(Path(json_path_or_dict).read_text(encoding="utf-8"))
        packed = pack_zcb3_bytes(doc)
        if out_zcb3_path is not None:
            out_p = Path(out_zcb3_path)
            out_p.write_bytes(packed)
            return cls(out_p)
        return cls(packed)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "format":
            return "zcb3_blake2b64"
        return self._sections.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if key == "format":
            return "zcb3_blake2b64"
        if key in self._sections:
            return self._sections[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key == "format" or key in self._sections

    def __iter__(self) -> Iterator[str]:
        yield from self._sections.keys()

    def __len__(self) -> int:
        return len(self._sections)

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Materialize all 3 sections back into standard Python dicts for round-trip verification."""
        return {
            name: dict(view.iter_records())
            for name, view in self._sections.items()
        }

    def close(self) -> None:
        if self._mmap_obj is not None:
            try:
                self._mmap_obj.close()
            except Exception:
                pass
            self._mmap_obj = None
        if self._file_obj is not None:
            try:
                self._file_obj.close()
            except Exception:
                pass
            self._file_obj = None

    def __enter__(self) -> ZCB3MmapCatalog:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def write_zcb3_file(
    catalog_or_json_path: str | Path | Mapping[str, Any],
    out_path: str | Path,
    *,
    compression: str = "none",
) -> dict[str, Any]:
    """Pack a JSON catalog into `ZCB3` and write raw, `zlib-9`, or `lzma-6` output file."""
    if isinstance(catalog_or_json_path, Mapping):
        doc = catalog_or_json_path
        raw_json_bytes = len(json.dumps(doc).encode("utf-8"))
    else:
        src_p = Path(catalog_or_json_path)
        raw_bytes = src_p.read_bytes()
        raw_json_bytes = len(raw_bytes)
        doc = json.loads(raw_bytes.decode("utf-8"))

    zcb3_raw = pack_zcb3_bytes(doc)
    comp_mode = compression.strip().lower()
    if comp_mode in ("none", "raw", "zcb3"):
        out_bytes = zcb3_raw
    elif comp_mode in ("zlib", "zlib9", "zlib-9"):
        out_bytes = compress_zcb3_zlib9(zcb3_raw)
    elif comp_mode in ("lzma", "lzma6", "lzma-6", "xz"):
        out_bytes = compress_zcb3_lzma6(zcb3_raw)
    else:
        raise ValueError(f"Unsupported ZCB3 compression mode: {compression!r}")

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out_bytes)

    _, n_anime, n_title_year, n_title_only = ZCB3_HEADER_STRUCT.unpack(zcb3_raw[:ZCB3_HEADER_SIZE])
    return {
        "out_path": str(dest),
        "compression": comp_mode,
        "raw_json_bytes": raw_json_bytes,
        "zcb3_uncompressed_bytes": len(zcb3_raw),
        "written_bytes": len(out_bytes),
        "reduction_vs_json_pct": round((1.0 - len(out_bytes) / max(raw_json_bytes, 1)) * 100.0, 2),
        "total_entries": int(n_anime + n_title_year + n_title_only),
        "map_counts": {
            "anime_map": int(n_anime),
            "title_year_map": int(n_title_year),
            "title_only_map": int(n_title_only),
        },
        "zcb3_sha256": hashlib.sha256(zcb3_raw).hexdigest(),
        "written_sha256": hashlib.sha256(out_bytes).hexdigest(),
    }
