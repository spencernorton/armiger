"""Standalone Byte-Anchored Title, Year & Episode Span Extractor (`armiger/byte_span_extractor.py`).

Provides three explicitly separated extraction modes for scientific attribution:
  1. `decode_bio_tags_raw_neural(raw_name, bio_tags)`:
     Pure neural decoding strictly from the 9th neural head (`span_bio` 7-class per-byte BIO tagger:
     `0=O, 1=B-TITLE, 2=I-TITLE, 3=B-YEAR, 4=I-YEAR, 5=B-EP, 6=I-EP`) with zero regex fallback.
  2. `extract_byte_spans_v3(raw_name)`:
     Deterministic UTF-8 byte-offset structural boundary extractor with zero external parser fallback.
  3. `decode_bio_tags_to_spans(raw_name, bio_tags)`:
     Hybrid neural + structural span decoder that uses the neural BIO span when it overlaps (`IoU >= 0.65`)
     and satisfies strict non-technical token hygiene, or falls back to the structural byte boundary.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

BIO_O = 0
BIO_B_TITLE = 1
BIO_I_TITLE = 2
BIO_B_YEAR = 3
BIO_I_YEAR = 4
BIO_B_EP = 5
BIO_I_EP = 6

LEADING_PREFIX_RE = re.compile(
    r"^(?:\s*(?:\[[^\]]{1,48}\]|\([^\)]{1,36}\)|【[^】]{1,36}】|★[^★]{1,36}★)\s*[\.\-_ ]*)+"
)
LEADING_SITE_RE = re.compile(
    r"(?i)^(?:www\.[a-z0-9\-]+\.[a-z]{2,6}[\s\._\-]+|\[?[a-z0-9\-]+\.(?:com|org|net|to|me|cc|pe|mx|ro|lol|tv|ru|in)\]?[\s\._\-\|]+)+"
)

BOUNDARY_RE = re.compile(
    r"(?i)(?:"
    r"[\.\s_\-\(\[\{]+(?:19[2-9]\d|20[0-2]\d)(?:[\.\s_\-\)\]\}]|$)"
    r"|[\.\s_\-\(\[\{]+(?:S\d{1,2}(?:E\d{1,3})?|S\d{1,2}[\.\s_\-]*-[\.\s_\-]*S?\d{1,2}|\d{1,2}x\d{2}|Season[\.\s_\-]*\d+|Complete[\.\s_\-]+Series|Mini[\.\s_\-]*Series|E\d{2,3}\b|Ep\.?\s*\d{1,4}\b|Episode[\.\s_\-]*\d{1,4}\b|Cap\.\d{2,4}\b|Temporada[\.\s_\-]*\d+)"
    r"|[\.\s_\-]+-\s*(?:0\d|[1-9]\d{0,2})(?:v\d)?(?:[\.\s_\-\[\(]|$)"
    r"|[\.\s_\-\(\[\{]+v\d+\.\d+(?:\.\d+)?\b"
    r"|[\.\s_\-\(\[\{]+(?:2160p|1080p|1080i|720p|576p|480p|4k|uhd|bluray|blu-ray|bdrip|brrip|remux|web-dl|webdl|webrip|hdtv|dvdrip|hdrip|dvdr|dvd9|dvd5|x264|x265|h264|h265|h\.264|h\.265|hevc|av1|xvid|divx|proper|repack|rerip|extended|unrated|directors[\.\s_\-]*cut|remastered|multi|dual[\.\s_\-]*audio|dual[\.\s_\-]*lat|truefrench|vff|vfq|vfi|sub[\.\s_\-]*ita|ita[\.\s_\-]*eng|rus|castellano|latino|german|french|italian|spanish|hindi)\b"
    r"|\s*[\[\(\{](?:1080p|720p|2160p|480p|BD|WEB|BDRip|BluRay|HEVC|x264|x265|Dual|Multi|ENG|ITA|RUS|SPA|FRE|GER|Sub|AAC|AC3|DTS|FLAC)[^\)\]\}]*[\]\)\}]"
    r")"
)

YEAR_SPAN_RE = re.compile(r"(?:^|[\.\s_\-\(\[\{])((?:19[2-9]\d|20[0-2]\d))(?=[\.\s_\-\)\]\}]|$)")
EPISODE_SPAN_RE = re.compile(
    r"(?i)(?:^|[\.\s_\-\(\[\{])("
    r"S\d{1,2}(?:E\d{1,3}(?:[\-E]\d{1,3})?)?|S\d{1,2}[\.\s_\-]*-[\.\s_\-]*S?\d{1,2}|\d{1,2}x\d{2}|"
    r"Season[\.\s_\-]*\d+(?:[\.\s_\-]*-[\.\s_\-]*\d+)?|Episode[\.\s_\-]*\d{1,4}|Ep\.?\s*\d{1,4}|Cap\.\d{2,4}|Temporada[\.\s_\-]*\d+"
    r")(?=[\.\s_\-\)\]\}]|$)"
)
TRAILING_EXT_RE = re.compile(
    r"(?i)\.(?:mkv|mp4|avi|mov|wmv|m4v|webm|ts|m2ts|mpg|mpeg|iso|rar|zip|7z|tar|gz|epub|pdf|mobi|cbz|cbr|mp3|flac|m4b)$"
)
TRAILING_JUNK_RE = re.compile(
    r"(?i)[\.\s_\-]+(?:by[\.\s_\-]+[a-z0-9_]+|uploaded[\.\s_\-]+by.*|www\.[a-z0-9\.\-]+|\[[^\]]*\]|\([^\)]*\))$"
)
TECH_TOKEN_SET = frozenset({
    "1080p", "1080i", "720p", "2160p", "480p", "576p", "4k", "uhd", "sdr", "hdr", "hdr10", "dv",
    "web", "webdl", "webrip", "bluray", "bdrip", "brrip", "remux", "hdtv", "dvdrip", "hdrip", "dvdr",
    "x264", "x265", "h264", "h265", "hevc", "avc", "xvid", "divx", "av1", "10bit", "8bit",
    "aac", "ac3", "eac3", "ddp", "ddp5", "dts", "dtshd", "truehd", "atmos", "flac", "mp3", "opus",
    "mkv", "mp4", "avi", "proper", "repack", "rerip", "internal", "limited", "multi", "dual",
})


def key_norm(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = unicodedata.normalize("NFKC", stripped).casefold()
    return re.sub(r"[\W_]+", " ", folded, flags=re.UNICODE).strip()


def is_strict_clean_title(extracted: str, raw_name: str) -> bool:
    """Strict non-gameable title validity check:
    - Must have >= 2 normalized characters and >= 1 non-technical token.
    - Every normalized token must be anchored in `raw_name`.
    - Must NOT contain bare resolution/codec/source tokens (`1080p`, `WEB DL`, `x264`, `mkv`).
    - When `raw_name` contains technical/release metadata, `extracted` must strip it (`len(kn_ext) < len(kn_src)`).
    """
    if not extracted or not isinstance(extracted, str):
        return False
    kn_ext = key_norm(extracted)
    kn_src = key_norm(raw_name)
    if len(kn_ext) < 2 or not kn_src:
        return False
    ext_toks = kn_ext.split()
    src_toks = set(kn_src.split())
    if not ext_toks or any(t not in src_toks for t in ext_toks):
        return False
    if any(t in TECH_TOKEN_SET for t in ext_toks):
        return False
    src_has_tech_or_ext = bool(src_toks & TECH_TOKEN_SET) or bool(YEAR_SPAN_RE.search(raw_name)) or bool(EPISODE_SPAN_RE.search(raw_name))
    if src_has_tech_or_ext and len(kn_ext) >= len(kn_src):
        return False
    return True


def title_token_f1(pred_title: str, ref_title: str) -> float:
    """Compute exact normalized token-level F1 between `pred_title` and `ref_title`."""
    p_toks = key_norm(pred_title).split()
    r_toks = key_norm(ref_title).split()
    if not p_toks or not r_toks:
        return 0.0
    if p_toks == r_toks:
        return 1.0
    p_set, r_set = set(p_toks), set(r_toks)
    inter = len(p_set & r_set)
    if inter == 0:
        return 0.0
    prec = inter / len(p_set)
    rec = inter / len(r_set)
    return 2.0 * prec * rec / (prec + rec)


def _char_to_byte_span(s: str, c_start: int, c_end: int, max_bytes: int = 256) -> tuple[int, int]:
    b_start = len(s[:max(0, c_start)].encode("utf-8", errors="replace"))
    b_end = len(s[:max(0, c_end)].encode("utf-8", errors="replace"))
    return min(b_start, max_bytes), min(b_end, max_bytes)


def extract_byte_spans_v3(raw_name: str, max_bytes: int = 256) -> dict[str, Any]:
    """Extract byte-anchored `(title, year, episode_spec)` spans directly from `raw_name`."""
    nm = raw_name or ""
    work = TRAILING_EXT_RE.sub("", nm)

    c_start = 0
    m_site = LEADING_SITE_RE.match(work)
    if m_site and m_site.end() < len(work) - 3:
        c_start = m_site.end()
    m_pref = LEADING_PREFIX_RE.match(work[c_start:])
    if m_pref and (c_start + m_pref.end()) < len(work) - 2:
        c_start += m_pref.end()

    while c_start < len(work) and work[c_start] in " ._-|:/":
        c_start += 1

    tail = work[c_start:]
    c_end = len(work)

    for m in BOUNDARY_RE.finditer(tail):
        pos = m.start()
        prefix_candidate = tail[:pos].strip(" ._-([{|:")
        if len(key_norm(prefix_candidate)) >= 2:
            c_end = c_start + pos
            break

    raw_title_slice = work[c_start:c_end].strip(" ._-([{|:")
    raw_title_slice = TRAILING_JUNK_RE.sub("", raw_title_slice).strip(" ._-([{|:")
    words = re.sub(r"[\._]+", " ", raw_title_slice).split()
    while len(words) > 1 and key_norm(words[-1]) in TECH_TOKEN_SET:
        words.pop()
    clean_title = " ".join(words).strip()

    if len(key_norm(clean_title)) < 2:
        fallback = re.sub(r"[\._]+", " ", work).strip()
        fb_words = [w for w in fallback.split() if key_norm(w) not in TECH_TOKEN_SET]
        clean_title = " ".join(fb_words[:6]).strip()
        c_start, c_end = 0, min(len(nm), max(1, len(clean_title)))

    t_bstart, t_bend = _char_to_byte_span(nm, c_start, c_end, max_bytes=max_bytes)

    year_val = None
    year_bspan = None
    ym = YEAR_SPAN_RE.search(nm, pos=max(0, c_end - 1)) or YEAR_SPAN_RE.search(nm)
    if ym:
        year_val = int(ym.group(1))
        year_bspan = _char_to_byte_span(nm, ym.start(1), ym.end(1), max_bytes=max_bytes)

    ep_val = None
    ep_bspan = None
    em = EPISODE_SPAN_RE.search(nm, pos=max(0, c_end - 1)) or EPISODE_SPAN_RE.search(nm)
    if em:
        ep_val = em.group(1).strip()
        ep_bspan = _char_to_byte_span(nm, em.start(1), em.end(1), max_bytes=max_bytes)

    return {
        "title": clean_title,
        "title_byte_span": (t_bstart, t_bend),
        "year": year_val,
        "year_byte_span": year_bspan,
        "episode_spec": ep_val,
        "episode_byte_span": ep_bspan,
        "is_anchored": is_strict_clean_title(clean_title, nm),
        "extractor_source": "structural_byte_span",
    }


def decode_bio_tags_raw_neural(raw_name: str, bio_tags: Sequence[int] | Any) -> dict[str, Any]:
    """Pure neural BIO span decoder:
    Decodes `(title, year, episode_spec)` strictly from the 9th neural head `bio_tags`
    (`0=O, 1=B-TITLE, 2=I-TITLE, 3=B-YEAR, 4=I-YEAR, 5=B-EP, 6=I-EP`) with zero call to `extract_byte_spans_v3`.
    """
    nm = raw_name or ""
    raw_bytes = nm.encode("utf-8", errors="replace")[:256]
    L = min(len(raw_bytes), len(bio_tags))
    tags = [int(bio_tags[i]) for i in range(L)]

    title_idx = [i for i, t in enumerate(tags) if t in (1, 2)]
    if title_idx:
        runs: list[tuple[int, int]] = []
        run_s = title_idx[0]
        prev = run_s
        for ii in title_idx[1:]:
            if ii <= prev + 2:
                prev = ii
            else:
                runs.append((run_s, prev + 1))
                run_s = ii
                prev = ii
        runs.append((run_s, prev + 1))
        tb, te = max(runs, key=lambda r: r[1] - r[0])
        cand_raw = raw_bytes[tb:te].decode("utf-8", errors="replace")
        words = re.sub(r"[\._]+", " ", cand_raw).strip(" ._-[({/|:").split()
        clean_title = " ".join(words).strip()
    else:
        tb, te = 0, 0
        clean_title = ""

    year_idx = [i for i, t in enumerate(tags) if t in (3, 4)]
    year_val = None
    year_bspan = None
    if len(year_idx) >= 4:
        yb, ye = year_idx[0], year_idx[-1] + 1
        y_str = raw_bytes[yb:ye].decode("utf-8", errors="replace")
        m_y = re.search(r"(19[2-9]\d|20[0-2]\d)", y_str)
        if m_y:
            year_val = int(m_y.group(1))
            year_bspan = (yb, ye)

    ep_idx = [i for i, t in enumerate(tags) if t in (5, 6)]
    ep_val = None
    ep_bspan = None
    if len(ep_idx) >= 2:
        eb, ee = ep_idx[0], ep_idx[-1] + 1
        ep_val = raw_bytes[eb:ee].decode("utf-8", errors="replace").strip(" ._-[({/|:")
        ep_bspan = (eb, ee)

    return {
        "title": clean_title,
        "title_byte_span": (tb, te),
        "year": year_val,
        "year_byte_span": year_bspan,
        "episode_spec": ep_val,
        "episode_byte_span": ep_bspan,
        "is_anchored": is_strict_clean_title(clean_title, nm),
        "extractor_source": "raw_neural_span_bio",
    }


def decode_bio_tags_to_spans(raw_name: str, bio_tags: Sequence[int] | Any) -> dict[str, Any]:
    """Hybrid Neural + Structural Span Decoder."""
    neural_res = decode_bio_tags_raw_neural(raw_name, bio_tags)
    struct_res = extract_byte_spans_v3(raw_name)
    if neural_res["is_anchored"]:
        nb, ne = neural_res["title_byte_span"]
        sb, se = struct_res["title_byte_span"]
        inter = max(0, min(ne, se) - max(nb, sb))
        union = max(1, max(ne, se) - min(nb, sb))
        if inter / union >= 0.65:
            return neural_res
    return struct_res
