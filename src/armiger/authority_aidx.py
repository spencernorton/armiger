"""Zero-copy, byte-verified mmap binary authority index (.aidx) — zero pickle in production!"""
from __future__ import annotations

import hashlib
import json
import unicodedata
import math
import mmap
import os
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"AIDX0317"
_SEED_HI = 0x9E3779B9
_ALIGN = 8

_YEAR_RE = re.compile(r"(?<!\d)(19[2-9]\d|20[0-3]\d)(?!\d)")
_TOK_RE = re.compile(r"[a-z0-9]+")
_STOP_NOISE = frozenset({
    "1080p", "720p", "2160p", "4k", "480p", "x264", "x265", "h264", "h265", "hevc",
    "bluray", "bdrip", "brrip", "webrip", "webdl", "web", "hdrip", "dvdrip", "cam",
    "aac", "ac3", "dts", "ddp", "dd5", "flac", "mp3", "320kbps", "repack", "proper",
    "extended", "unrated", "remux", "hdr", "hdr10", "dv", "10bit", "8bit", "amzn",
    "nf", "dsnp", "hmax", "atvp", "multi", "dual", "dubbed", "subbed", "engsub",
    "complete", "season", "episode", "discography", "anthology", "collection",
    "fitgirl", "dodi", "codex", "rune", "elamigos", "gog", "skidrow", "plaza",
    "flt", "tenoke", "empress", "cpy", "razor1911", "yify", "yts", "rarbg", "ettv",
    "eztv", "tgx", "galaxyrg", "qxr", "psa", "pahe", "sparks", "geckos", "ntb", "flux",
})
_S_EP_RE = re.compile(r"\bs\d{1,2}(?:e\d{1,3})?\b|\b\d{1,2}x\d{2}\b", re.IGNORECASE)

AUTH_FEAT_DIM = 24


def _crc64(b: bytes) -> tuple[int, int, int]:
    lo = zlib.crc32(b) & 0xFFFFFFFF
    hi = zlib.crc32(b, _SEED_HI) & 0xFFFFFFFF
    return lo, hi, (hi << 32) | lo


def _pad8(buf: bytearray) -> int:
    rem = len(buf) % _ALIGN
    if rem:
        buf.extend(b"\x00" * (_ALIGN - rem))
    return len(buf)


def compile_v39_pkl_to_aidx(v39_pkl_path: str | Path, out_aidx_path: str | Path, v3161_bundle_dir: str | Path) -> dict[str, Any]:
    """Offline compiler: converts legacy 178MB v39.pkl into a single zero-copy mmap-backed .aidx binary."""
    import pickle, sys

    sys.path.insert(0, str(v3161_bundle_dir))
    with Path(v39_pkl_path).open("rb") as f:
        obj = pickle.load(f)

    idx = obj["index"]
    tidf = obj["token_idf"]
    N = len(idx._h64)

    # Convert sorted arrays to exact little-endian numpy buffers
    bm22 = np.frombuffer(idx._bm22, dtype=np.uint8)
    h64 = np.asarray(idx._h64, dtype=np.uint64)
    p16 = np.searchsorted(h64 >> np.uint64(48), np.arange(65537, dtype=np.uint64)).astype(np.uint32)
    key_bytes = np.frombuffer(idx._key_bytes, dtype=np.uint8)
    key_offsets = np.asarray(idx._key_offsets, dtype=np.uint32)
    f_orig9 = np.asarray(idx._f_orig9, dtype=np.float16)  # 1.59M x 9 x fp16 = 28.6 MB (vs 57.2 MB fp32!)
    f_ext5 = np.asarray(idx._f_ext5, dtype=np.float16)    # 1.59M x 5 x fp16 = 15.9 MB
    max_v = np.asarray(idx._max_v, dtype=np.int32)
    is_dict = np.frombuffer(idx._is_dict, dtype=np.uint8)
    yr_idx = np.asarray(idx._yr_idx, dtype=np.uint16)     # [N, 4]
    p_anime_u8 = np.asarray(idx._p_anime_u8, dtype=np.uint8) if idx._p_anime_u8 is not None else np.zeros(N, dtype=np.uint8)
    cn_dh_bits = np.frombuffer(idx._cn_dh_bits, dtype=np.uint8) if idx._cn_dh_bits is not None else np.zeros((N + 7) // 8, dtype=np.uint8)

    # Flatten _yr_pool (tuple of tuples of int years -> int16 values + uint32 offsets)
    yr_vals_list: list[int] = []
    yr_offs_list: list[int] = [0]
    for tup in idx._yr_pool:
        if isinstance(tup, (tuple, list)):
            for y in tup:
                yr_vals_list.append(int(y))
        yr_offs_list.append(len(yr_vals_list))
    yr_pool_vals = np.asarray(yr_vals_list, dtype=np.int16)
    yr_pool_offs = np.asarray(yr_offs_list, dtype=np.uint32)

    # Token IDF
    t_h64 = np.asarray(tidf._h64, dtype=np.uint64)
    t_p12 = np.searchsorted(t_h64 >> np.uint64(52), np.arange(4097, dtype=np.uint64)).astype(np.uint32)
    t_vals = np.asarray(tidf._vals, dtype=np.float32)

    sections: list[tuple[str, np.ndarray]] = [
        ("bm22", bm22),
        ("h64", h64),
        ("p16", p16),
        ("key_bytes", key_bytes),
        ("key_offsets", key_offsets),
        ("f_orig9", f_orig9),
        ("f_ext5", f_ext5),
        ("max_v", max_v),
        ("is_dict", is_dict),
        ("yr_idx", yr_idx),
        ("p_anime_u8", p_anime_u8),
        ("cn_dh_bits", cn_dh_bits),
        ("yr_pool_vals", yr_pool_vals),
        ("yr_pool_offs", yr_pool_offs),
        ("t_h64", t_h64),
        ("t_p12", t_p12),
        ("t_vals", t_vals),
    ]

    # Build aligned payload + JSON manifest header
    payload = bytearray()
    sec_meta: dict[str, dict[str, Any]] = {}
    for name, arr in sections:
        arr_c = np.ascontiguousarray(arr)
        off = _pad8(payload)
        raw = arr_c.tobytes()
        payload.extend(raw)
        sec_meta[name] = {
            "offset": off,
            "nbytes": len(raw),
            "dtype": str(arr_c.dtype),
            "shape": list(arr_c.shape),
        }

    header = {
        "format": "AIDX_V317",
        "n_keys": int(N),
        "n_idf": int(len(t_h64)),
        "sections": sec_meta,
    }
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header so payload starts on 8-byte boundary!
    prefix_len = 8 + 4 + len(hdr_bytes)
    pad_len = (_ALIGN - (prefix_len % _ALIGN)) % _ALIGN
    hdr_padded = hdr_bytes + (b" " * pad_len)

    out_p = Path(out_aidx_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(hdr_padded)))
        f.write(hdr_padded)
        f.write(payload)

    sz_mb = out_p.stat().st_size / (1024 * 1024)
    return {"aidx_path": str(out_p), "size_mb": round(sz_mb, 2), "n_keys": int(N), "n_idf": int(len(t_h64))}


class MmapAuthorityIndex:
    """Zero-copy, byte-verified mmap reader for authority_v317.aidx (<1 MiB private dirty RSS!)."""

    def __init__(self, aidx_path: str | Path):
        self._fp = open(aidx_path, "rb")
        self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        if self._mm[:8] != MAGIC:
            raise ValueError("Invalid .aidx magic")
        hdr_len = struct.unpack("<I", self._mm[8:12])[0]
        hdr = json.loads(self._mm[12 : 12 + hdr_len].decode("utf-8"))
        base = 12 + hdr_len
        self.n_keys = int(hdr["n_keys"])
        self.n_idf = int(hdr["n_idf"])
        self._views: list[Any] = []

        def view(name: str) -> np.ndarray:
            s = hdr["sections"][name]
            arr = np.frombuffer(self._mm, dtype=np.dtype(s["dtype"]), count=int(np.prod(s["shape"])), offset=base + s["offset"])
            arr = arr.reshape(tuple(s["shape"]))
            self._views.append(arr)
            return arr

        self.bm22 = view("bm22")
        self.h64 = view("h64")
        self.p16 = view("p16")
        self.key_bytes = view("key_bytes")
        self.key_offsets = view("key_offsets")
        self.f_orig9 = view("f_orig9")
        self.f_ext5 = view("f_ext5")
        self.max_v = view("max_v")
        self.is_dict = view("is_dict")
        self.yr_idx = view("yr_idx")
        self.p_anime_u8 = view("p_anime_u8")
        self.cn_dh_bits = view("cn_dh_bits")
        self.yr_pool_vals = view("yr_pool_vals")
        self.yr_pool_offs = view("yr_pool_offs")
        self.t_h64 = view("t_h64")
        self.t_p12 = view("t_p12")
        self.t_vals = view("t_vals")
        self._kb_mv = memoryview(self._mm)[base + hdr["sections"]["key_bytes"]["offset"] : base + hdr["sections"]["key_bytes"]["offset"] + hdr["sections"]["key_bytes"]["nbytes"]]

    def close(self) -> None:
        self._views.clear()
        for attr in ("bm22", "h64", "p16", "key_bytes", "key_offsets", "f_orig9", "f_ext5",
                     "max_v", "is_dict", "yr_idx", "p_anime_u8", "cn_dh_bits", "yr_pool_vals",
                     "yr_pool_offs", "t_h64", "t_p12", "t_vals"):
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, "_kb_mv") and self._kb_mv is not None:
            self._kb_mv.release()
            self._kb_mv = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def find_row(self, key: str) -> int:
        """O(1) Bloom + p16 narrowed binary search + 100% EXACT key-byte verification!"""
        if not key:
            return -1
        kb = key.encode("utf-8", "surrogatepass")
        lo, hi, hk = _crc64(kb)
        if not ((int(self.bm22[lo & 0x3FFFFF]) >> ((lo >> 22) & 7)) & 1):
            return -1
        b16 = hi >> 16
        lo_i = int(self.p16[b16])
        hi_i = int(self.p16[b16 + 1])
        if lo_i >= hi_i:
            return -1
        pos = lo_i + int(np.searchsorted(self.h64[lo_i:hi_i], np.uint64(hk), side="left"))
        n = self.n_keys
        klen = len(kb)
        while pos < n and int(self.h64[pos]) == hk:
            s0 = int(self.key_offsets[pos])
            s1 = int(self.key_offsets[pos + 1])
            if (s1 - s0) == klen and self._kb_mv[s0:s1] == kb:
                return pos
            pos += 1
        return -1

    def get_idf(self, tok: str, default: float = 0.0) -> float:
        if not tok:
            return default
        _, hi, hk = _crc64(tok.encode("utf-8", "surrogatepass"))
        b12 = hi >> 20
        lo_i = int(self.t_p12[b12])
        hi_i = int(self.t_p12[b12 + 1])
        if lo_i >= hi_i:
            return default
        pos = lo_i + int(np.searchsorted(self.t_h64[lo_i:hi_i], np.uint64(hk), side="left"))
        if pos < self.n_idf and int(self.t_h64[pos]) == hk:
            return float(self.t_vals[pos])
        return default

    def _has_year(self, pool_slot: int, target_years: set[int]) -> tuple[bool, bool]:
        """Return (yr_exact, has_any_years)."""
        a = int(self.yr_pool_offs[pool_slot])
        b = int(self.yr_pool_offs[pool_slot + 1])
        if a >= b:
            return False, False
        ys = self.yr_pool_vals[a:b]
        for ty in target_years:
            if ty in ys or (ty - 1) in ys or (ty + 1) in ys:
                return True, True
        return False, True

    def _score_candidates(
        self,
        cands: list[tuple[str, int, bool, float]],
        t_years: set[int],
    ) -> tuple[int, int, bool, bool, bool, float]:
        """Score candidate stems against byte-verified mmap .aidx table."""
        best_row = -1
        best_score = -1.0
        best_meta = (0, False, False, False)
        for st, nt, is_ex, src_w in cands:
            r = self.find_row(st)
            if r < 0:
                continue
            f9 = self.f_orig9[r]
            mw = float(f9[0]); tvw = float(f9[1]); gw = float(f9[2]); aw = float(f9[3]); sw = float(f9[4])
            max_v = float(self.max_v[r])
            is_d = bool(self.is_dict[r])
            yr_exact = False
            has_any_yr = False
            if t_years:
                for col in range(4):
                    ye, ha = self._has_year(int(self.yr_idx[r, col]), t_years)
                    yr_exact = yr_exact or ye
                    has_any_yr = has_any_yr or ha
            yr_conflict = bool(t_years and has_any_yr and not yr_exact)
            if nt == 1:
                if is_d:
                    if not (is_ex and yr_exact and (max_v >= 800 or (gw >= 4.0 and max_v >= 100))):
                        continue
                else:
                    if (not is_ex) and len(st) < 6:
                        continue
                    if len(st) < 5 and not (is_ex and len(st) == 4 and (gw >= 4.0 or aw >= 6.0 or sw >= 6.0)):
                        continue
            elif nt == 2:
                if is_d:
                    if not is_ex:
                        continue
                    if yr_conflict and not yr_exact:
                        continue
                    if (not yr_exact) and max_v < 350 and aw < 5.0 and gw < 3.5 and sw < 6.0:
                        continue
                else:
                    if (not is_ex) and (yr_conflict or ((not yr_exact) and max_v < 1200 and aw < 7.0 and sw < 8.0)):
                        continue
            tot_w = mw + tvw + gw + aw + sw + float(self.f_ext5[r].sum())
            sc = (
                src_w
                * (1.0 + 0.55 * min(nt, 5))
                * (1.35 if is_ex else 0.85)
                * (1.65 if yr_exact else (0.55 if yr_conflict else 1.0))
                * (0.62 if is_d else 1.0)
                * math.log1p(max(tot_w, 0.5) + 0.002 * max_v)
            )
            if sc > best_score:
                best_score = sc
                best_row = r
                best_meta = (nt, is_ex, yr_exact, yr_conflict)
        return best_row, best_meta[0], best_meta[1], best_meta[2], best_meta[3], max(0.0, best_score)

    def _row_to_24d(
        self,
        best_row: int,
        nt: int,
        is_ex: bool,
        yr_exact: bool,
        yr_conflict: bool,
        best_score: float,
        t_years: set[int],
    ) -> np.ndarray:
        out = np.zeros(AUTH_FEAT_DIM, dtype=np.float32)
        if best_row < 0:
            return out
        f9 = self.f_orig9[best_row]
        f5 = self.f_ext5[best_row]
        mw = float(f9[0]); tvw = float(f9[1]); gw = float(f9[2]); aw = float(f9[3]); sw = float(f9[4])
        if t_years:
            m_ym, _ = self._has_year(int(self.yr_idx[best_row, 0]), t_years)
            tv_ym, _ = self._has_year(int(self.yr_idx[best_row, 1]), t_years)
            g_ym, _ = self._has_year(int(self.yr_idx[best_row, 2]), t_years)
            a_ym, _ = self._has_year(int(self.yr_idx[best_row, 3]), t_years)
            if m_ym and not tv_ym:
                mw *= 2.5
            if tv_ym and not m_ym:
                tvw *= 2.5
            if g_ym:
                gw *= 2.0
            if a_ym:
                aw *= 2.0
        cw = float(f9[8])
        musw = float(f5[0]); ebw = float(f5[1]); crsw = float(f5[2]); abw = float(f5[3]); xxxw = float(f5[4])
        tot4 = mw + tvw + gw + aw + sw + 1e-6
        out[0] = 1.0
        out[1] = min(nt, 8) / 8.0
        out[2] = 1.0 if is_ex else 0.0
        out[3] = 1.0 if self.is_dict[best_row] else 0.0
        out[4] = 1.0 if yr_exact else 0.0
        out[5] = 1.0 if yr_conflict else 0.0
        out[6] = math.log1p(max(0.0, best_score))
        out[7] = math.log1p(max(0.0, float(self.max_v[best_row])))
        out[8] = math.log1p(max(0.0, mw))
        out[9] = math.log1p(max(0.0, tvw))
        out[10] = math.log1p(max(0.0, gw))
        out[11] = math.log1p(max(0.0, aw))
        out[12] = math.log1p(max(0.0, sw))
        out[13] = math.log1p(max(0.0, cw))
        out[14] = math.log1p(max(0.0, musw))
        out[15] = math.log1p(max(0.0, ebw))
        out[16] = math.log1p(max(0.0, crsw))
        out[17] = math.log1p(max(0.0, abw))
        out[18] = math.log1p(max(0.0, xxxw))
        out[19] = mw / tot4
        out[20] = tvw / tot4
        out[21] = gw / tot4
        out[22] = float(self.p_anime_u8[best_row]) / 255.0
        out[23] = 1.0 if ((int(self.cn_dh_bits[best_row >> 3]) >> (best_row & 7)) & 1) else 0.0
        return out

    def extract_24d_features(self, name: str, files: list[Any] | None = None) -> np.ndarray:
        """Full v3.9 space-separated + squashed + file-folder candidate authority lookup (24-dim)."""
        cands, t_years, _ = generate_v39_candidates(name, files)
        if not cands:
            return np.zeros(AUTH_FEAT_DIM, dtype=np.float32)
        best_row, nt, is_ex, yr_exact, yr_conflict, best_score = self._score_candidates(cands, t_years)
        return self._row_to_24d(best_row, nt, is_ex, yr_exact, yr_conflict, best_score, t_years)


# ---------------------------------------------------------------------------
# Full v3.9-grade Fast Candidate Generator (space-separated + squashed + files)
# ---------------------------------------------------------------------------
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z]{2})(?=\d)|(?<=\d)(?=[A-Za-z]{2})")
_V39_YEAR_RE = re.compile(r"(?<!\d)(19\d\d|20[0-2]\d)(?!\d)")
_V39_IA_META_RE = re.compile(
    r"(?i)(?:_meta\.(?:xml|sqlite)|_files\.xml|_archive\.torrent|_reviews\.xml|__ia_thumb\.jpg|"
    r"\.asr\.(?:srt|vtt|js)|_(?:thumb|itemimage|spectrogram)\.(?:jpg|png|gif)|^00icon)"
)
_V39_SIDECAR_RE = re.compile(
    r"(?i)\.(?:jpg|jpeg|png|gif|bmp|tif|tiff|webp|svg|ico|txt|nfo|sfv|md5|sha1|xml|sqlite|json|js|css|html|htm|srt|vtt|ass|ssa|sub|idx)$"
)
_V39_RDNS_RE = re.compile(r"(?i)^(?:com|org|net|io|de|fr|me|co|app|dev|edu|gov|uk|jp|ru)\.[a-z0-9_]+(?:\.[a-z0-9_]+)*\.")
_V39_TECH_CUT_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:s\d{1,2}[._\s-]*e\d{1,3}|\d{1,2}x\d{2,3}|ep(?:isode)?[._\s-]*\d{1,3}|"
    r"season[._\s-]*\d{1,2}|complete[._\s-]*series|4k|2160p|1080p|720p|480p|360p|bluray|blu[._\s-]*ray|"
    r"bdrip|brrip|web[._\s-]*dl|webrip|dvdrip|hdtv|x264|x265|h264|h265|hevc|xvid|divx|aac\d*|ac3|flac|"
    r"mp3|ddp?\d*|10bit|hi10p|remux|repack|proper|multi|dual[._\s-]*audio|v[._\s-]*\d+(?:[.\s]+\d+)+|"
    r"\d+\.\d+(?:\.\d+)*|setup|installer|portable|linux|amd64|x86_64|win32|win64|macos|android)(?![a-z0-9])"
)
_V39_GENERIC_RE = re.compile(
    r"(?i)^(?:season\s*\d+|s\s*\d+|vol(?:ume)?\s*\d+|part\s*\d+|pt\s*\d+|ep(?:isode)?\s*\d+|"
    r"chapter\s*\d+|ch\s*\d+|disc\s*\d+|disk\s*\d+|cd\s*\d+|dvd\s*\d+|track\s*\d+|"
    r"director\s*s?\s*cut|extended\s*cut|final\s*cut|theatrical\s*cut|special\s*edition|"
    r"complete\s*series|complete\s*season|untitled(?:\s*\d+)?|video(?:\s*\d+)?|audio(?:\s*\d+)?|"
    r"sample(?:\s*\d+)?|preview|trailer|teaser|cover|front|back)$"
)
_V39_NOISE_TOKS = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "and", "to", "by", "with", "at", "from", "or", "is",
    "mp4", "mkv", "avi", "mov", "m4v", "ogv", "webm", "wmv", "flv", "mpg", "mpeg", "vob", "ts",
    "zip", "rar", "7z", "tar", "gz", "xz", "iso", "bin", "cue", "exe", "msi", "apk", "ipa", "dmg",
    "pdf", "epub", "mobi", "cbz", "cbr", "mp3", "flac", "m4a", "m4b", "ogg", "wav",
    "h264", "x264", "x265", "hevc", "aac", "ac3", "dvd", "vhs",
    "4k", "2160p", "1080p", "720p", "480p", "360p", "hd", "sd", "uhd", "bluray", "webrip", "webdl", "dvdrip",
    "eng", "english", "sub", "subs", "subbed", "dub", "dubbed", "raw", "raws", "subsplease",
    "horriblesubs", "erairaws", "nyaa", "setup", "install", "installer", "version", "ver",
    "trailer", "teaser", "promo", "sample", "opening", "ending", "ost",
    "amd64", "linux", "win", "win32", "win64", "mod", "repack",
    "season", "vol", "volume", "episode", "ep", "part", "pt", "disc", "disk", "cd",
    "remastered", "definitive", "deluxe", "ultimate", "goty", "32bit", "64bit",
})
_V39_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\}")
_V39_EXT_RE = re.compile(r"\.[a-zA-Z0-9]{2,4}$")
_V39_SEP_RE = re.compile(r"[._-]+")
_V39_ALNUM_RE = re.compile(r"[a-z0-9]+")
_V39_SEG_SPLIT_RE = re.compile(r"\s+[-–—|~]+\s+|[:/|()]+")
_V39_LEAD_BRK_RE = re.compile(r"^(?:\[[^\]]*\]|\([^\)]*\))\s*")
_ZCB_NONWORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def _clean_v39_src(s: str) -> str:
    if not s:
        return ""
    if not s.isascii():
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("'", "").replace("\u2019", "")
    if "&" in s:
        s = s.replace("&", " and ")
    s = _V39_EXT_RE.sub("", s)
    s = _V39_RDNS_RE.sub("", s)
    s = _CAMEL_RE.sub(" ", s)
    s = _V39_BRACKET_RE.sub(" ", s)
    s = _V39_SEP_RE.sub(" ", s)
    return s.strip()


def _zcb_norm_title(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _ZCB_NONWORD_RE.sub(" ", s)
    return " ".join(s.split())


def generate_v39_candidates(
    name: str,
    files: list[Any] | None = None,
    max_files: int = 5,
) -> tuple[list[tuple[str, int, bool, float]], set[int], list[str]]:
    """
    Fast v3.9 candidate generator returning:
      - cands: [(stem, n_toks, is_exact, src_weight)] (space-separated + squashed, article-stripped)
      - t_years: explicit years found in title + top files
      - zcb_titles: NFKC-normalized titles (with leading articles preserved) for ZCB3 lookups
    """
    t_years: set[int] = set()
    sources: list[tuple[str, float]] = []
    zcb_titles: list[str] = []
    seen_zcb: set[str] = set()

    def _add_zcb(raw: str) -> None:
        nt = _zcb_norm_title(raw)
        if len(nt) >= 4 and nt not in seen_zcb:
            seen_zcb.add(nt)
            zcb_titles.append(nt)

    if name:
        nm = str(name)[:512]
        sources.append((nm, 1.0))
        for m in _V39_YEAR_RE.finditer(nm):
            y = int(m.group(1))
            if 1900 <= y <= 2028:
                t_years.add(y)
        s_clean = _V39_EXT_RE.sub("", nm).strip()
        s_nol = _V39_LEAD_BRK_RE.sub("", s_clean).strip()
        for base in (s_nol, s_clean):
            if not base:
                continue
            pre_t = _V39_TECH_CUT_RE.split(base)[0].strip()
            pre_y = _V39_YEAR_RE.split(pre_t)[0].strip()
            _add_zcb(pre_y)
            _add_zcb(pre_t)
            if " - " in pre_y:
                for part in pre_y.split(" - ")[:2]:
                    _add_zcb(part)

    if files:
        seen_src: set[str] = set()
        for f in files[:max_files]:
            p = str(f.get("path") or f.get("p") or f.get("name") or "") if isinstance(f, dict) else str(f or "")
            if not p or _V39_IA_META_RE.search(p):
                continue
            for m in _V39_YEAR_RE.finditer(p):
                y = int(m.group(1))
                if 1900 <= y <= 2028:
                    t_years.add(y)
            parts = [pt for pt in p.replace("\\", "/").split("/") if pt]
            if not parts:
                continue
            base = parts[-1]
            if not _V39_SIDECAR_RE.search(base):
                if base not in seen_src:
                    seen_src.add(base)
                    sources.append((base, 0.94))
                if len(parts) >= 2 and parts[0] not in seen_src:
                    seen_src.add(parts[0])
                    sources.append((parts[0], 0.90))
            elif len(parts) >= 2 and parts[0] not in seen_src:
                seen_src.add(parts[0])
                sources.append((parts[0], 0.78))

    cands_map: dict[str, tuple[int, bool, float]] = {}

    def _add(toks: list[str], is_exact: bool, w: float) -> None:
        toks = [t for t in toks if not (len(t) == 4 and t.isdigit() and 1900 <= int(t) <= 2028)]
        while toks and (toks[-1] in _V39_NOISE_TOKS or (toks[-1].isdigit() and len(toks) >= 2)):
            toks = toks[:-1]
        while toks and toks[0] in ("the", "a", "an"):
            toks = toks[1:]
        if not toks or len(toks) > 10:
            return
        if len(toks) == 1 and (len(toks[0]) < 4 or toks[0] in _V39_NOISE_TOKS or toks[0].isdigit()):
            return
        st = " ".join(toks)
        if _V39_GENERIC_RE.match(st):
            return
        prev = cands_map.get(st)
        if prev is None or (is_exact and not prev[1]) or (is_exact == prev[1] and w > prev[2]):
            cands_map[st] = (len(toks), is_exact, w)
        if 2 <= len(toks) <= 4:
            sq = "".join(toks)
            if len(sq) >= 6:
                prev_sq = cands_map.get(sq)
                if prev_sq is None or (is_exact and not prev_sq[1]):
                    cands_map[sq] = (len(toks), is_exact, w * 0.95)

    for raw_src, src_w in sources[:5]:
        cleaned = _clean_v39_src(raw_src)
        if not cleaned:
            continue
        pre_tech = _V39_TECH_CUT_RE.split(cleaned)[0].strip()
        pre_year = _V39_YEAR_RE.split(pre_tech)[0].strip()
        seen_v: set[str] = set()
        for variant in (pre_year, pre_tech, cleaned):
            if not variant or variant in seen_v:
                continue
            seen_v.add(variant)
            toks = _V39_ALNUM_RE.findall(variant.lower())
            if toks:
                _add(toks, True, src_w)
                if len(toks) >= 3:
                    for k in range(2, min(6, len(toks))):
                        _add(toks[:k], False, src_w * 0.70)
                        _add(toks[-k:], False, src_w * 0.65)
            segs = _V39_SEG_SPLIT_RE.split(variant)
            if len(segs) > 1:
                for seg in segs[:4]:
                    stoks = _V39_ALNUM_RE.findall(seg.lower())
                    if stoks:
                        is_seg_ex = len(stoks) >= max(2, int(len(toks) * 0.55))
                        _add(stoks, is_seg_ex, src_w * (0.94 if is_seg_ex else 0.78))

    cands = [(st, nt, ex, sw) for st, (nt, ex, sw) in cands_map.items()]
    return cands, t_years, zcb_titles


# ---------------------------------------------------------------------------
# Zero-Copy ZCB3 mmap Reader (validated_zero_cue_catalog.zcb3, 8.38 MB, 0 heap!)
# ---------------------------------------------------------------------------
_ZCB3_MAGIC = b"ZCB3"
_ZCB3_HDR = struct.Struct("<4s3I")
_ZCB3_REC = 9
_ZCB3_PERSON = b"armiger-zc-v2"


class ZCB3MmapReader:
    """Zero-copy mmap binary reader for validated_zero_cue_catalog.zcb3 (~0.005 ms/row)."""

    def __init__(self, zcb3_path: str | Path):
        self._fp = open(zcb3_path, "rb")
        self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        magic, n_ani, n_ty, n_to = _ZCB3_HDR.unpack_from(self._mm, 0)
        if magic != _ZCB3_MAGIC:
            raise ValueError("Invalid ZCB3 magic")
        off0 = 16
        off1 = off0 + n_ani * _ZCB3_REC
        off2 = off1 + n_ty * _ZCB3_REC
        off3 = off2 + n_to * _ZCB3_REC
        self._sec_ani = (off0, n_ani)
        self._sec_ty = (off1, n_ty)
        self._sec_to = (off2, n_to)
        self._mv = memoryview(self._mm)
        self._off_end = off3

    def close(self) -> None:
        if getattr(self, "_mv", None) is not None:
            self._mv.release()
            self._mv = None
        if getattr(self, "_mm", None) is not None:
            self._mm.close()
            self._mm = None
        if getattr(self, "_fp", None) is not None:
            self._fp.close()
            self._fp = None

    def _lookup_section(self, sec: tuple[int, int], key_str: str) -> int:
        base, n = sec
        if n <= 0 or not key_str:
            return -1
        target = hashlib.blake2b(key_str.encode("utf-8", errors="ignore"), digest_size=8, person=_ZCB3_PERSON).digest()
        mm = self._mm
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            pos = base + mid * 9
            k = mm[pos : pos + 8]
            if k == target:
                return mm[pos + 8]
            elif k < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def query_titles(self, zcb_titles: list[str], t_years: set[int]) -> tuple[int, int, int]:
        """Returns (title_year_code, anime_code, title_only_code); -1 if miss (0=movie,1=tv,2=ambig)."""
        ty_code = -1
        if t_years:
            yrs_sorted = sorted(t_years)
            for ct in zcb_titles[:6]:
                if len(ct) < 4:
                    continue
                for y in yrs_sorted[:2]:
                    c = self._lookup_section(self._sec_ty, f"{ct}|{y}")
                    if c >= 0:
                        ty_code = c
                        break
                if ty_code >= 0:
                    break
        ani_code = -1
        for ct in zcb_titles[:6]:
            if len(ct) >= 7:
                c = self._lookup_section(self._sec_ani, ct)
                if c >= 0:
                    ani_code = c
                    break
        to_code = -1
        for i, ct in enumerate(zcb_titles[:6]):
            min_len = 7 if i == 0 else 12
            if len(ct) >= min_len:
                c = self._lookup_section(self._sec_to, ct)
                if c >= 0:
                    to_code = c
                    break
        return ty_code, ani_code, to_code


# ---------------------------------------------------------------------------
# Unified External Authority Bundle (.aidx + ZCB3 + s3 Wikidata + ext_games)
# ---------------------------------------------------------------------------
FULL_AUTH_DIM = 38  # 24 (.aidx) + 6 (ZCB3) + 6 (s3 Wikidata) + 2 (ext_games)
AUDIO_P_AUTH_DIM = 12  # STRICTLY P(x)-only public authority features!


class ExternalAuthorityV317:
    """Zero-copy unified external authority reader (zero pickle, zero holdout leakage)."""

    def __init__(
        self,
        aidx_path: str | Path,
        zcb3_path: str | Path | None = None,
        ext_cat_json_path: str | Path | None = None,
    ):
        self.aidx = MmapAuthorityIndex(aidx_path)
        self.zcb3 = ZCB3MmapReader(zcb3_path) if (zcb3_path and Path(zcb3_path).exists()) else None
        self.s3: dict[str, list[int]] = {}
        self.ext_games: dict[str, list[int]] = {}
        if ext_cat_json_path and Path(ext_cat_json_path).exists():
            doc = json.loads(Path(ext_cat_json_path).read_text(encoding="utf-8"))
            self.s3 = doc.get("s3") or {}
            self.ext_games = doc.get("ext_games") or {}

    def close(self) -> None:
        self.aidx.close()
        if self.zcb3 is not None:
            self.zcb3.close()

    def extract_full_authority_38d(self, name: str, files: list[Any] | None = None) -> np.ndarray:
        """38-dim external authority features for Plane-2 cat11 / junk."""
        out = np.zeros(FULL_AUTH_DIM, dtype=np.float32)
        cands, t_years, zcb_titles = generate_v39_candidates(name, files)
        if not cands and not zcb_titles:
            return out
        if cands:
            best_row, nt, is_ex, yr_exact, yr_conflict, best_score = self.aidx._score_candidates(cands, t_years)
            out[:24] = self.aidx._row_to_24d(best_row, nt, is_ex, yr_exact, yr_conflict, best_score, t_years)
        if self.zcb3 is not None and zcb_titles:
            ty_c, ani_c, to_c = self.zcb3.query_titles(zcb_titles, t_years)
            if ty_c == 0:
                out[24] = 1.0
            elif ty_c == 1:
                out[25] = 1.0
            elif ty_c == 2:
                out[26] = 1.0
            if ani_c >= 0:
                out[27] = 1.0
            if to_c == 0:
                out[28] = 1.0
            elif to_c == 1:
                out[29] = 1.0
        if cands and (self.s3 or self.ext_games):
            s3_best = None
            eg_best = None
            for st, nt, is_ex, _ in cands:
                if s3_best is None and st in self.s3 and (nt >= 2 or (is_ex and len(st) >= 5)):
                    s3_best = self.s3[st]
                if eg_best is None and st in self.ext_games and (nt >= 2 or (is_ex and len(st) >= 5)):
                    eg_best = self.ext_games[st]
                if s3_best is not None and eg_best is not None:
                    break
            if s3_best is not None:
                cc, lc, rad, _, y0, y1 = s3_best
                out[30] = 1.0 if cc == 0 else (0.5 if cc == 2 else 0.0)
                out[31] = 1.0 if cc == 1 else (0.5 if cc == 2 else 0.0)
                if t_years and y0 > 0 and any((y0 - 1) <= ty <= (y1 + 1) for ty in t_years):
                    out[32] = 1.0
                out[33] = float(rad)
                out[34] = 1.0 if lc == 1 else 0.0
                out[35] = 1.0 if lc == 2 else 0.0
            if eg_best is not None:
                out[36] = 1.0
                y0, y1 = eg_best
                if t_years and y0 > 0 and any((y0 - 1) <= ty <= (y1 + 1) for ty in t_years):
                    out[37] = 1.0
        return out

    def extract_audio_p_authority_12d(self, px_title: str) -> np.ndarray:
        """
        STRICTLY P(x)-only 12-dim public authority vector for the isolated audio branch!
        Takes ONLY `px_title = project_audio_only(name, files)[0]` (zero subtitle text, zero files)
        so bitwise subtitle noninterference (`max |Δp_en| == 0.0000000000`) is guaranteed by construction.
        """
        out = np.zeros(AUDIO_P_AUTH_DIM, dtype=np.float32)
        if not px_title:
            return out
        cands, t_years, zcb_titles = generate_v39_candidates(px_title, None, max_files=0)
        if cands:
            best_row, nt, is_ex, yr_exact, yr_conflict, best_score = self.aidx._score_candidates(cands, t_years)
            if best_row >= 0:
                f9 = self.aidx.f_orig9[best_row]
                out[0] = 1.0
                out[1] = float(self.aidx.p_anime_u8[best_row]) / 255.0
                out[2] = 1.0 if ((int(self.aidx.cn_dh_bits[best_row >> 3]) >> (best_row & 7)) & 1) else 0.0
                out[3] = math.log1p(max(0.0, float(f9[3])))
                out[4] = math.log1p(max(0.0, float(f9[0])))
                out[5] = math.log1p(max(0.0, float(f9[1])))
        if self.zcb3 is not None and zcb_titles:
            ty_c, ani_c, to_c = self.zcb3.query_titles(zcb_titles, t_years)
            out[6] = 1.0 if ani_c >= 0 else 0.0
            out[7] = 1.0 if (ty_c == 0 or to_c == 0) else 0.0
            out[8] = 1.0 if (ty_c == 1 or to_c == 1) else 0.0
        if cands and self.s3:
            for st, nt, is_ex, _ in cands:
                rec = self.s3.get(st)
                if rec is not None and (nt >= 2 or (is_ex and len(st) >= 5)):
                    _, lc, _, loc, _, _ = rec
                    out[9] = 1.0 if lc == 1 else 0.0
                    out[10] = 1.0 if lc == 2 else 0.0
                    out[11] = float(loc)
                    break
        return out
