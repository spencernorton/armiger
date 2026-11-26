"""Zero-dependency pure-Python + NumPy deterministic signed word(1-2)+char_wb(3-5) feature hasher (100% train/prod bit-exact!)."""
from __future__ import annotations

from functools import lru_cache
import math
import os
import re
import unicodedata
import zlib
from typing import Any

import numpy as np

N_WORD_BUCKETS = 65536
N_CHAR_BUCKETS = 65536
TOTAL_BUCKETS = N_WORD_BUCKETS + N_CHAR_BUCKETS

_WORD_RE = re.compile(
    r"[a-z0-9_\u00c0-\u024f\u0400-\u052f\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0900-\u0d7f\u0600-\u06ff]+",
    re.I,
)
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_TECH_STRIP_LANG_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:"
    r"librivox|gutenberg|audiobook|unabridged|mp3|m4b|m4a|flac|ogg|opus|wav|aac|ac3|dts|ddp?\d*|"
    r"mkv|mp4|avi|mov|m4v|wmv|webm|ts|m2ts|iso|zip|rar|7z|pdf|epub|cbz|cbr|"
    r"64kb|128kb|192kb|256kb|320kb|kbps|vbr|cbr|44khz|48khz|"
    r"2160p|1080p|720p|480p|360p|4k|uhd|hdr\d*|dv|10bit|8bit|hi10p|"
    r"x264|x265|h264|h265|hevc|av1|xvid|divx|bluray|blu[\s._-]*ray|bdrip|brrip|"
    r"web[\s._-]*dl|webrip|web|dvdrip|hdtv|remux|repack|proper|nf|amzn|dsnp|atvp|hmax|"
    r"s\d{1,2}e\d{1,3}|s\d{1,2}|season[\s._-]*\d+|ep(?:isode)?[\s._-]*\d+|"
    r"complete|series|part[\s._-]*\d+|vol(?:ume)?[\s._-]*\d+|cd\d+|disc\d+|"
    r"yts|yify|rarbg|eztv|ettv|tgx|galaxy|psa|qxr|tigole|pahe|cmrg|ntb|flux|ion10|megusta|mkvcage|"
    r"19\d\d|20[0-2]\d|\d{3,6}"
    r")(?![a-z0-9])"
)
_EXT_VID = frozenset({".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".webm", ".flv"})
_EXT_AUD = frozenset({".mp3", ".flac", ".m4a", ".m4b", ".ogg", ".opus", ".wav", ".aac", ".wma"})
_EXT_HEX = frozenset("0123456789abcdef")
_DUMP_SUFS = (".warc", ".warc.gz", ".xml.bz2", ".sql.gz", ".jsonl.gz", ".tar.zst")


def build_enriched_lexical_text(name: str, files: list[Any] | None = None, size_bytes: int = 0) -> str:
    """
    Build rich structured+lexical text combining title, CamelCase-split tokens, top-14 unique
    file/folder stems, and coarse extension/count/size/opaque structural tokens so a single linear
    L-BFGS residual captures lexical x structural interactions (e.g. 'adobe' + __ext_exe__).
    """
    nm = str(name or "")[:256]
    nm_sp = _CAMEL_SPLIT_RE.sub(" ", nm)
    fl = files or []
    n_files = len(fl)
    ext_counts: dict[str, int] = {}
    n_vid = 0
    n_aud = 0
    tot_sz = int(size_bytes or 0)
    f_tokens: list[str] = []
    seen_f: set[str] = set()
    has_dump = nm.lower().endswith(_DUMP_SUFS)

    for f in fl[:128]:
        if isinstance(f, dict):
            p = str(f.get("path") or f.get("p") or f.get("name") or "")
            sz = max(0, int(f.get("size") or f.get("s") or f.get("length") or 0))
        else:
            p = str(f or "")
            sz = 0
        tot_sz += sz
        pl = p.lower().replace("\\", "/")
        if (not has_dump) and pl.endswith(_DUMP_SUFS):
            has_dump = True
        parts = [pt for pt in pl.split("/") if pt]
        if not parts:
            continue
        base = parts[-1]
        stem, ext = os.path.splitext(base)
        if ext and len(ext) <= 6:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            if ext in _EXT_VID:
                n_vid += 1
            elif ext in _EXT_AUD:
                n_aud += 1
        if len(f_tokens) < 12:
            if len(parts) >= 2 and parts[0] not in seen_f:
                seen_f.add(parts[0])
                f_tokens.append(parts[0][:40])
            if stem and stem not in seen_f:
                seen_f.add(stem)
                f_tokens.append(stem[:40])

    # Structural summary tokens (compact, deterministic, zero raw path bloat)
    stoks: list[str] = []
    if n_files == 0:
        stoks.append("__nf0__")
    elif n_files == 1:
        stoks.append("__nf1__")
    elif n_files <= 5:
        stoks.append("__nf2_5__")
    elif n_files <= 25:
        stoks.append("__nf6_25__")
    else:
        stoks.append("__nf26p__")

    if n_vid == 1:
        stoks.append("__nv1__")
    elif n_vid == 2:
        stoks.append("__nv2__")
    elif n_vid >= 3:
        stoks.append("__nv3p__")
    if n_aud >= 3:
        stoks.append("__na3p__")

    mb = tot_sz / (1024 * 1024)
    if tot_sz > 0:
        if mb < 5:
            stoks.append("__sz_lt5m__")
        elif mb < 100:
            stoks.append("__sz_5_100m__")
        elif mb < 1500:
            stoks.append("__sz_100m_1g__")
        elif mb < 15000:
            stoks.append("__sz_1_15g__")
        else:
            stoks.append("__sz_gt15g__")

    # Top 4 extensions by count
    for ex, _ in sorted(ext_counts.items(), key=lambda kv: -kv[1])[:4]:
        stoks.append("__ex_" + ex.lstrip(".") + "__")

    # Opaque hex / dump flags
    nl_s = nm.strip().lower()
    st_n = nl_s.rsplit(".", 1)[0] if "." in nl_s else nl_s
    if 10 <= len(st_n) <= 96 and all(ch in _EXT_HEX for ch in st_n):
        stoks.append("__opq_hex__")
    if has_dump:
        stoks.append("__opq_dump__")

    parts_out = [nm_sp]
    if f_tokens:
        parts_out.append(" ".join(f_tokens))
    if stoks:
        parts_out.append(" ".join(stoks))
    return " ".join(parts_out)


def clean_lang_title_p(clean_audio_title: str) -> str:
    """
    STRICTLY P(x)-only pure title language string!
    Strips container/LibriVox/bitrate/codec/resolution/scene-group metadata so pure orthographic
    char(2-5) & word(1) language ID classifies 'contes_temps_passe_librivox' as French rather than
    falling for the English LibriVox prior! Guaranteed 0.0000000000 subtitle-invariant.
    """
    if not clean_audio_title:
        return ""
    s = _CAMEL_SPLIT_RE.sub(" ", str(clean_audio_title))
    s = s.replace("_", " ").replace(".", " ").replace("-", " ")
    s = _TECH_STRIP_LANG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@lru_cache(maxsize=32768)
def _hash_word_and_chars(w: str) -> tuple[int, float, tuple[tuple[int, float], ...]]:
    b = w.encode("utf-8", "ignore")
    h = zlib.crc32(b) & 0xFFFFFFFF
    idx = h & 0xFFFF
    sgn = 1.0 if (h & 0x10000) else -1.0
    if w.startswith("__") and w.endswith("__"):
        return idx, sgn, ()
    wb = " " + w[:16] + " "
    L = len(wb)
    ch: list[tuple[int, float]] = []
    for n in (3, 4, 5):
        if L < n:
            continue
        for j in range(L - n + 1):
            cg = wb[j : j + n].encode("utf-8", "ignore")
            hc = zlib.crc32(cg, 0x27D4EB2D) & 0xFFFFFFFF
            ch.append((N_WORD_BUCKETS + (hc & 0xFFFF), 1.0 if (hc & 0x10000) else -1.0))
    return idx, sgn, tuple(ch)


def extract_hashed_counts(text: str, max_chars: int = 512, max_words: int = 60) -> dict[int, float]:
    """
    Deterministic crc32 signed feature hashing -> L2-normalized sparse dict {bucket_idx: signed_val}.
    # Vectorized multi-head linear lexical scoring
    """
    if not text:
        return {}
    s = str(text).lower()[:max_chars]
    words = _WORD_RE.findall(s)[:max_words]
    if not words:
        return {}

    w_counts: dict[int, float] = {}
    c_counts: dict[int, float] = {}
    hwc = _hash_word_and_chars
    crc = zlib.crc32

    nw = len(words)
    for i, w in enumerate(words):
        idx, sgn, ch = hwc(w)
        w_counts[idx] = w_counts.get(idx, 0.0) + sgn
        if i + 1 < nw:
            h2 = crc((w + " " + words[i + 1]).encode("utf-8", "ignore")) & 0xFFFFFFFF
            idx2 = h2 & 0xFFFF
            w_counts[idx2] = w_counts.get(idx2, 0.0) + (1.0 if (h2 & 0x10000) else -1.0)
        for ic, sc in ch:
            c_counts[ic] = c_counts.get(ic, 0.0) + sc

    out: dict[int, float] = {}
    if w_counts:
        inv_w = 1.0 / math.sqrt(sum(v * v for v in w_counts.values()))
        for k, v in w_counts.items():
            if v != 0.0:
                out[k] = v * inv_w
    if c_counts:
        inv_c = 1.0 / math.sqrt(sum(v * v for v in c_counts.values()))
        for k, v in c_counts.items():
            if v != 0.0:
                out[k] = v * inv_c
    return out


@lru_cache(maxsize=32768)
def _char_only_word(w: str) -> tuple[tuple[int, float], ...]:
    wb = " " + w[:24] + " "
    L = len(wb)
    ch: list[tuple[int, float]] = []
    crc = zlib.crc32
    for n in (2, 3, 4, 5):
        if L < n:
            continue
        for j in range(L - n + 1):
            hc = crc(wb[j : j + n].encode("utf-8", "ignore"), 0x517CC1B7) & 0xFFFFFFFF
            ch.append((hc & 0xFFFF, 1.0 if (hc & 0x10000) else -1.0))
    return tuple(ch)


def extract_char_only_counts(text: str, max_chars: int = 192) -> dict[int, float]:
    """
    Pure orthographic char_wb(2..5) hasher into 65,536 bins strictly for P(x) language-ID!
    Zero metadata tokens; captures Romance/Germanic/Slavic/Nordic/Romaji/ASCII slug phonetics.
    """
    if not text:
        return {}
    s = str(text)
    if not s.isascii():
        s = unicodedata.normalize("NFKC", s)
    words = _WORD_RE.findall(s.lower()[:max_chars])[:24]
    if not words:
        return {}
    c_counts: dict[int, float] = {}
    cow = _char_only_word
    for w in words:
        for ic, sc in cow(w):
            c_counts[ic] = c_counts.get(ic, 0.0) + sc
    if not c_counts:
        return {}
    inv = 1.0 / math.sqrt(sum(v * v for v in c_counts.values()))
    return {k: v * inv for k, v in c_counts.items() if v != 0.0}


def score_linear_sparse_vec_T(feats: dict[int, float], W_T: np.ndarray, b: np.ndarray | float) -> np.ndarray | float:
    """Cache-friendly contiguous float32 row-gather dot product (W_T shape [D, C] or [D])."""
    if not feats:
        return np.array(b, dtype=np.float32) if isinstance(b, np.ndarray) else float(b)
    idxs = np.fromiter(feats.keys(), dtype=np.int32, count=len(feats))
    vals = np.fromiter(feats.values(), dtype=np.float32, count=len(feats))
    if W_T.ndim == 2:
        return (vals @ W_T[idxs]) + b
    return float(np.dot(W_T[idxs], vals) + float(b))


def score_linear_sparse_vec(feats: dict[int, float], W: np.ndarray, b: np.ndarray | float) -> np.ndarray | float:
    """Zero-scipy pure-NumPy sparse dot product against FP16/FP32 weight table W."""
    if not feats:
        return np.array(b, dtype=np.float32) if isinstance(b, np.ndarray) else float(b)
    idxs = np.fromiter(feats.keys(), dtype=np.int32, count=len(feats))
    vals = np.fromiter(feats.values(), dtype=np.float32, count=len(feats))
    if W.ndim == 2:
        return (W[:, idxs].astype(np.float32) @ vals) + np.asarray(b, dtype=np.float32)
    return float(np.dot(W[idxs].astype(np.float32), vals) + float(b))


def _chunk_hash(ts: list[str]):
    rows_i, cols_i, vals_f = [], [], []
    for r_idx, t in enumerate(ts):
        d = extract_hashed_counts(t)
        for k, v in d.items():
            rows_i.append(r_idx)
            cols_i.append(k)
            vals_f.append(v)
    return (
        np.asarray(rows_i, dtype=np.int32),
        np.asarray(cols_i, dtype=np.int32),
        np.asarray(vals_f, dtype=np.float32),
        len(ts),
    )


def _chunk_char_hash(ts: list[str]):
    rows_i, cols_i, vals_f = [], [], []
    for r_idx, t in enumerate(ts):
        d = extract_char_only_counts(t)
        for k, v in d.items():
            rows_i.append(r_idx)
            cols_i.append(k)
            vals_f.append(v)
    return (
        np.asarray(rows_i, dtype=np.int32),
        np.asarray(cols_i, dtype=np.int32),
        np.asarray(vals_f, dtype=np.float32),
        len(ts),
    )


def build_csr_matrix(texts: list[str], n_workers: int = 12):
    """Training-only fast parallel CSR builder using the EXACT same extract_hashed_counts()."""
    import concurrent.futures
    from scipy.sparse import csr_matrix

    chunks = [texts[i : i + 2048] for i in range(0, len(texts), 2048)]
    all_r, all_c, all_v = [], [], []
    off = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for ri, ci, vf, m in ex.map(_chunk_hash, chunks):
            all_r.append(ri + off)
            all_c.append(ci)
            all_v.append(vf)
            off += m
    return csr_matrix(
        (np.concatenate(all_v), (np.concatenate(all_r), np.concatenate(all_c))),
        shape=(len(texts), TOTAL_BUCKETS),
        dtype=np.float32,
    )


def build_char_csr_matrix(texts: list[str], n_workers: int = 12):
    """Training-only fast parallel CSR builder using extract_char_only_counts() (65,536 bins)."""
    import concurrent.futures
    from scipy.sparse import csr_matrix

    chunks = [texts[i : i + 2048] for i in range(0, len(texts), 2048)]
    all_r, all_c, all_v = [], [], []
    off = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for ri, ci, vf, m in ex.map(_chunk_char_hash, chunks):
            all_r.append(ri + off)
            all_c.append(ci)
            all_v.append(vf)
            off += m
    return csr_matrix(
        (np.concatenate(all_v), (np.concatenate(all_r), np.concatenate(all_c))),
        shape=(len(texts), N_CHAR_BUCKETS),
        dtype=np.float32,
    )
