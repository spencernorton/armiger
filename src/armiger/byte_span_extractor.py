"""Standalone Byte-Anchored Title, Year & Episode Span Extractor (`armiger/byte_span_extractor.py`).

Provides four explicitly separated extraction and decoding modes for scientific attribution:
  1. `decode_bio_logits_viterbi(logits_2d, raw_bytes)`:
     Constrained 7x7 BIO Viterbi decoder with hard log-transition constraints (`0.00%` illegal
     transitions) and deterministic ASCII delimiter + 4-digit year boundary snapping
     (`snap_bio_span_boundaries`).
  2. `decode_bio_tags_raw_neural(raw_name, bio_tags)`:
     Pure neural decoding strictly from the 9th neural head (`span_bio` 7-class per-byte BIO tagger:
     `0=O, 1=B-TITLE, 2=I-TITLE, 3=B-YEAR, 4=I-YEAR, 5=B-EP, 6=I-EP`) with zero regex fallback.
  3. `extract_byte_spans_v3(raw_name)`:
     Deterministic UTF-8 byte-offset structural boundary extractor with zero external parser fallback.
  4. `decode_bio_tags_to_spans(raw_name, bio_tags)`:
     Hybrid neural + structural span decoder that uses the neural BIO span when it overlaps (`IoU >= 0.65`)
     and satisfies strict non-technical token hygiene, or falls back to the structural byte boundary.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

BIO_O = 0
BIO_B_TITLE = 1
BIO_I_TITLE = 2
BIO_B_YEAR = 3
BIO_I_YEAR = 4
BIO_B_EP = 5
BIO_I_EP = 6
SPAN_BIO_CLASSES: tuple[str, ...] = (
    "O",
    "B-TITLE",
    "I-TITLE",
    "B-YEAR",
    "I-YEAR",
    "B-EP",
    "I-EP",
)

SEP_BYTES: frozenset[int] = frozenset(b"_.- []()")
_YEAR_TOKEN_BYTES_RE = re.compile(rb"(?<![0-9])(19[2-9]\d|20[0-2]\d)(?![0-9])")
_EP_TOKEN_BYTES_RE = re.compile(
    rb"(?i)(?<![a-z0-9])("
    rb"s\d{1,2}(?:[\.\s_\-]*e\d{1,3})?"
    rb"|(?:season|episode|chapter|chapters|ep|vol|disc|part)[\.\s_\-]*\d{1,4}(?:[\.\s_\-]*-[\.\s_\-]*\d{1,4})?"
    rb")(?![a-z0-9])"
)
_NEG_INF: float = float("-inf")


def _build_bio_viterbi_tables_py() -> tuple[list[float], list[list[float]]]:
    """Construct the constrained 7-state initial log-prior vector and 7x7 log-transition matrix."""
    init_log_p = [_NEG_INF] * 7
    init_log_p[BIO_O] = 0.0
    init_log_p[BIO_B_TITLE] = 0.0
    init_log_p[BIO_B_YEAR] = 0.0
    init_log_p[BIO_B_EP] = 0.0

    trans = [[_NEG_INF] * 7 for _ in range(7)]
    trans[BIO_O][BIO_O] = 0.0
    trans[BIO_O][BIO_B_TITLE] = 0.0
    trans[BIO_O][BIO_B_YEAR] = 0.0
    trans[BIO_O][BIO_B_EP] = 0.0

    entity_pairs = (
        (BIO_B_TITLE, BIO_I_TITLE),
        (BIO_B_YEAR, BIO_I_YEAR),
        (BIO_B_EP, BIO_I_EP),
    )
    for b_x, i_x in entity_pairs:
        for src in (b_x, i_x):
            trans[src][BIO_O] = 0.0
            trans[src][i_x] = 0.0
            for b_y, _ in entity_pairs:
                trans[src][b_y] = 0.0
    return init_log_p, trans


_BIO_INIT_LOG_P_PY, _BIO_TRANS_LOG_P_PY = _build_bio_viterbi_tables_py()
_BIO_INIT_LOG_P = np.asarray(_BIO_INIT_LOG_P_PY, dtype=np.float64) if np is not None else _BIO_INIT_LOG_P_PY
_BIO_TRANS_LOG_P = np.asarray(_BIO_TRANS_LOG_P_PY, dtype=np.float64) if np is not None else _BIO_TRANS_LOG_P_PY


def is_illegal_bio_transition(prev_tag: int, cur_tag: int, *, is_start: bool = False) -> bool:
    """Return True if transitioning from `prev_tag` to `cur_tag` violates constrained 7-tag BIO grammar."""
    c = int(cur_tag)
    if c < 0 or c >= 7:
        return True
    if is_start:
        return not math.isfinite(float(_BIO_INIT_LOG_P_PY[c]))
    p = int(prev_tag)
    if p < 0 or p >= 7:
        return True
    return not math.isfinite(float(_BIO_TRANS_LOG_P_PY[p][c]))


def count_illegal_bio_transitions(tags: Sequence[int] | Any) -> int:
    """Count illegal BIO state transitions (including illegal `I-*` start state at t=0) in `tags`."""
    n = len(tags)
    if n == 0:
        return 0
    cnt = 1 if is_illegal_bio_transition(0, int(tags[0]), is_start=True) else 0
    for t in range(1, n):
        if is_illegal_bio_transition(int(tags[t - 1]), int(tags[t]), is_start=False):
            cnt += 1
    return cnt


def snap_bio_span_boundaries(
    tags: Sequence[int] | Any,
    raw_bytes: bytes | bytearray = b"",
) -> Any:
    """Deterministically snap BIO span boundaries on ASCII delimiters and 4-digit year tokens:
      - Strip leading/trailing ASCII separator bytes (`_`, `.`, `-`, ` `, `[`, `]`, `(`, `)`)
        from `B-*`/`I-*` spans back to `O`, reassigning `B-*` to the first non-separator byte.
      - Snap 4-digit year tokens `(19[2-9]\\d|20[0-2]\\d)` cleanly when `B-YEAR`/`I-YEAR` covers
        >= 2 bytes of the 4-digit token.
    """
    arr_list = [int(x) if 0 <= int(x) < 7 else BIO_O for x in tags]
    n = len(arr_list)
    if n == 0:
        return np.asarray(arr_list, dtype=np.int64) if np is not None else arr_list
    b_raw = bytes(raw_bytes or b"")
    L = min(n, len(b_raw))
    if L == 0:
        return np.asarray(arr_list, dtype=np.int64) if np is not None else arr_list

    entity_groups = (
        (BIO_B_TITLE, BIO_I_TITLE),
        (BIO_B_YEAR, BIO_I_YEAR),
        (BIO_B_EP, BIO_I_EP),
    )

    def _strip_and_normalize_spans() -> None:
        for b_tag, i_tag in entity_groups:
            min_len = 2 if b_tag in (BIO_B_YEAR, BIO_B_EP) else 1
            idx = 0
            while idx < L:
                if arr_list[idx] in (b_tag, i_tag):
                    s = idx
                    idx += 1
                    while idx < L and arr_list[idx] == i_tag:
                        idx += 1
                    e = idx
                    for k in range(s, e):
                        arr_list[k] = BIO_O
                    ns, ne = s, e
                    while ns < ne and b_raw[ns] in SEP_BYTES:
                        ns += 1
                    while ne > ns and b_raw[ne - 1] in SEP_BYTES:
                        ne -= 1
                    if ne - ns >= min_len:
                        arr_list[ns] = b_tag
                        for k in range(ns + 1, ne):
                            arr_list[k] = i_tag
                else:
                    idx += 1

    _strip_and_normalize_spans()

    year_positions = {i for i in range(L) if arr_list[i] in (BIO_B_YEAR, BIO_I_YEAR)}
    if year_positions:
        snapped_years: list[tuple[int, int]] = []
        for m in _YEAR_TOKEN_BYTES_RE.finditer(b_raw[:L]):
            ys, ye = m.start(1), m.end(1)
            if ye <= L:
                overlap = sum(1 for k in range(ys, ye) if k in year_positions)
                if overlap >= 2:
                    snapped_years.append((ys, ye))
        if snapped_years:
            for i in year_positions:
                arr_list[i] = BIO_O
            for ys, ye in snapped_years:
                arr_list[ys] = BIO_B_YEAR
                for k in range(ys + 1, ye):
                    arr_list[k] = BIO_I_YEAR
            _strip_and_normalize_spans()

    ep_spans: list[tuple[int, int]] = []
    idx = 0
    while idx < L:
        if arr_list[idx] in (BIO_B_EP, BIO_I_EP):
            s = idx
            while idx < L and arr_list[idx] in (BIO_B_EP, BIO_I_EP):
                idx += 1
            ep_spans.append((s, idx))
        else:
            idx += 1
    if ep_spans:
        for i in range(L):
            if arr_list[i] in (BIO_B_EP, BIO_I_EP):
                arr_list[i] = BIO_O
        s0, e0 = ep_spans[0]
        ep_set = set(range(s0, e0))
        max_ep_pos = e0 - 1
        for m in _EP_TOKEN_BYTES_RE.finditer(b_raw[:L]):
            es, ee = m.start(1), m.end(1)
            if ee <= L and max_ep_pos <= ee:
                overlap = sum(1 for k in range(es, ee) if k in ep_set)
                if overlap >= 2:
                    s0, e0 = es, ee
                    break
        arr_list[s0] = BIO_B_EP
        for k in range(s0 + 1, e0):
            arr_list[k] = BIO_I_EP
        _strip_and_normalize_spans()

    idx = 0
    while idx < L:
        if arr_list[idx] in (BIO_B_TITLE, BIO_I_TITLE):
            s = idx
            idx += 1
            while idx < L and arr_list[idx] == BIO_I_TITLE:
                idx += 1
            e = idx
            for k in range(s, e):
                arr_list[k] = BIO_O
            ns, ne = s, e
            while ns < ne and b_raw[ns] in SEP_BYTES:
                ns += 1
            while ne > ns and b_raw[ne - 1] in SEP_BYTES:
                ne -= 1
            if ne > ns:
                while ns > 0 and b_raw[ns - 1] not in SEP_BYTES and arr_list[ns - 1] == BIO_O:
                    ns -= 1
                while ne < L and b_raw[ne] not in SEP_BYTES and arr_list[ne] in (BIO_O, BIO_I_TITLE):
                    ne += 1
                while ns < ne and b_raw[ns] in SEP_BYTES:
                    ns += 1
                while ne > ns and b_raw[ne - 1] in SEP_BYTES:
                    ne -= 1
                if ne > ns:
                    arr_list[ns] = BIO_B_TITLE
                    for k in range(ns + 1, ne):
                        arr_list[k] = BIO_I_TITLE
                    idx = max(idx, ne)
        else:
            idx += 1

    for k in range(L, n):
        arr_list[k] = BIO_O
    return np.asarray(arr_list, dtype=np.int64) if np is not None else arr_list


def decode_bio_logits_viterbi(
    logits_2d: Any,
    raw_bytes: bytes | bytearray = b"",
) -> Any:
    """Decode per-byte 7-class `span_bio` logits via constrained 7x7 Viterbi + delimiter boundary snapping."""
    if np is not None:
        arr = np.asarray(logits_2d, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(f"Expected logits_2d of shape (T, 7), got {arr.shape}")

        n_total = int(arr.shape[0])
        if n_total == 0:
            return np.zeros((0,), dtype=np.int64)

        b_raw = bytes(raw_bytes or b"")
        L = min(n_total, len(b_raw)) if len(b_raw) > 0 else n_total
        if L == 0:
            return np.zeros((n_total,), dtype=np.int64)

        active = np.nan_to_num(arr[:L], nan=0.0, posinf=1e12, neginf=-1e12)
        shifted = active - np.max(active, axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.sum(np.exp(np.maximum(shifted, -700.0)), axis=-1, keepdims=True))

        dp = np.full((L, 7), _NEG_INF, dtype=np.float64)
        backptr = np.zeros((L, 7), dtype=np.int64)
        dp[0] = _BIO_INIT_LOG_P + log_probs[0]

        for t in range(1, L):
            scores = dp[t - 1][:, None] + _BIO_TRANS_LOG_P
            best_prev = np.argmax(scores, axis=0)
            backptr[t] = best_prev
            dp[t] = scores[best_prev, np.arange(7)] + log_probs[t]

        path = np.zeros((L,), dtype=np.int64)
        path[L - 1] = int(np.argmax(dp[L - 1]))
        for t in range(L - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        if len(b_raw) > 0:
            path = snap_bio_span_boundaries(path, b_raw[:L])

        full_out = np.zeros((n_total,), dtype=np.int64)
        full_out[:L] = path
        return full_out

    # Pure-Python fallback when numpy is absent
    rows = list(logits_2d)
    if len(rows) == 1 and hasattr(rows[0], "__len__") and len(rows[0]) > 0 and hasattr(rows[0][0], "__len__"):
        rows = list(rows[0])
    n_total = len(rows)
    if n_total == 0:
        return []
    for row in rows:
        if len(row) != 7:
            raise ValueError(f"Expected logits_2d rows of length 7, got {len(row)}")
    b_raw = bytes(raw_bytes or b"")
    L = min(n_total, len(b_raw)) if len(b_raw) > 0 else n_total
    if L == 0:
        return [0] * n_total
    dp_py = [[_NEG_INF] * 7 for _ in range(L)]
    bp_py = [[0] * 7 for _ in range(L)]
    for t in range(L):
        r = [
            0.0 if math.isnan(float(x)) else (1e12 if float(x) > 1e12 else (-1e12 if float(x) < -1e12 else float(x)))
            for x in rows[t]
        ]
        mx = max(r)
        exps = [math.exp(max(x - mx, -700.0)) for x in r]
        lse = math.log(sum(exps))
        lp = [x - mx - lse for x in r]
        if t == 0:
            for j in range(7):
                dp_py[0][j] = _BIO_INIT_LOG_P_PY[j] + lp[j]
        else:
            for j in range(7):
                best_i = 0
                best_sc = dp_py[t - 1][0] + _BIO_TRANS_LOG_P_PY[0][j]
                for i in range(1, 7):
                    sc = dp_py[t - 1][i] + _BIO_TRANS_LOG_P_PY[i][j]
                    if sc > best_sc:
                        best_sc = sc
                        best_i = i
                bp_py[t][j] = best_i
                dp_py[t][j] = best_sc + lp[j]
    path_py = [0] * L
    path_py[L - 1] = max(range(7), key=lambda j: dp_py[L - 1][j])
    for t in range(L - 2, -1, -1):
        path_py[t] = bp_py[t + 1][path_py[t + 1]]
    if len(b_raw) > 0:
        path_py = list(snap_bio_span_boundaries(path_py, b_raw[:L]))
    return path_py + [0] * (n_total - L)


LEADING_PREFIX_RE = re.compile(
    r"^(?:\s*(?:\[[^\]]{1,48}\]|\([^\)]{1,36}\)|【[^】]{1,36}】|★[^★]{1,36}★)\s*[\.\-_ ]*)+"
)
LEADING_SITE_RE = re.compile(
    r"(?i)^(?:www\.[a-z0-9\-]+\.[a-z]{2,6}[\s\._\-]+|\[?[a-z0-9\-]+\.(?:com|org|net|to|me|cc|pe|mx|ro|lol|tv|ru|in)\]?[\s\._\-\|]+)+"
)

BOUNDARY_RE = re.compile(
    r"(?i)(?:"
    r"[\.\s_\-\(\[\{]+(?:19[2-9]\d|20[0-2]\d)(?:[\.\s_\-\)\]\}]|$)"
    r"|[\.\s_\-\(\[\{]+(?:S\d{1,2}[\.\s_\-]*-[\.\s_\-]*S?\d{1,2}|S\d{1,2}(?:E\d{1,3})?|\d{1,2}x\d{2}|Season[\.\s_\-]*\d+|Complete[\.\s_\-]+Series|Mini[\.\s_\-]*Series|E\d{2,3}\b|Ep\.?\s*\d{1,4}\b|Episode[\.\s_\-]*\d{1,4}\b|Cap\.\d{2,4}\b|Temporada[\.\s_\-]*\d+)"
    r"|[\.\s_\-]+-\s*(?:0\d|[1-9]\d{0,2})(?:v\d)?(?:[\.\s_\-\[\(]|$)"
    r"|[\.\s_\-\(\[\{]+v\d+\.\d+(?:\.\d+)?\b"
    r"|[\.\s_\-\(\[\{]+(?:2160p|1080p|1080i|720p|576p|480p|4k|uhd|bluray|blu-ray|bdrip|brrip|remux|web-dl|webdl|webrip|hdtv|dvdrip|hdrip|dvdr|dvd9|dvd5|x264|x265|h264|h265|h\.264|h\.265|hevc|av1|xvid|divx|proper|repack|rerip|extended|unrated|directors[\.\s_\-]*cut|remastered|multi|dual[\.\s_\-]*audio|dual[\.\s_\-]*lat|truefrench|vff|vfq|vfi|vostfr|sub[\.\s_\-]*ita|ita[\.\s_\-]*eng)\b"
    r"|[\.\s_\-\(\[\{]+(?:rus|russian|castellano|latino|german|french|italian|spanish|hindi|korean|japanese|chinese)\b(?=[\.\s_\-\)\]\}]*(?:2160p|1080p|1080i|720p|576p|480p|4k|uhd|bluray|blu-ray|bdrip|brrip|remux|web-dl|webdl|webrip|web|hdtv|dvdrip|hdrip|dvdr|dvd9|dvd5|x264|x265|h264|h265|h\.264|h\.265|hevc|av1|xvid|divx|aac|ac3|eac3|ddp\d*(?:\.\d)?|dts(?:-?hd)?|truehd|atmos|flac|mp3|m4b|dl|dub|dubbed|sub|subbed|subs|vostfr|multi|dual|audio|proper|repack|rerip|internal|limited|complete)\b)"
    r"|\s*[\[\(\{](?:1080p|720p|2160p|480p|BD|WEB|BDRip|BluRay|HEVC|x264|x265|Dual|Multi|ENG|ITA|RUS|SPA|FRE|GER|Sub|AAC|AC3|DTS|FLAC)[^\)\]\}]*[\]\)\}]"
    r")"
)

YEAR_SPAN_RE = re.compile(r"(?:^|[\.\s_\-\(\[\{])((?:19[2-9]\d|20[0-2]\d))(?=[\.\s_\-\)\]\}]|$)")
EPISODE_SPAN_RE = re.compile(
    r"(?i)(?:^|[\.\s_\-\(\[\{])("
    r"S\d{1,2}[\.\s_\-]*-[\.\s_\-]*S?\d{1,2}|S\d{1,2}(?:E\d{1,3}(?:[\-E]\d{1,3})?)?|\d{1,2}x\d{2}|"
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


def _utf8_safe_window(raw_name: str, max_bytes: int = 256) -> tuple[str, bytes, bool]:
    """Return `(window_str, window_bytes, was_truncated)` decoded strictly from a UTF-8-safe `max_bytes` prefix."""
    if not raw_name or max_bytes <= 0:
        return "", b"", False
    full_bytes = raw_name.encode("utf-8", errors="replace")
    if len(full_bytes) <= max_bytes:
        win_str = full_bytes.decode("utf-8", errors="ignore")
        return win_str, win_str.encode("utf-8"), False
    win_str = full_bytes[:max_bytes].decode("utf-8", errors="ignore")
    return win_str, win_str.encode("utf-8"), True


def _char_to_byte_span(s: str, c_start: int, c_end: int, max_bytes: int = 256) -> tuple[int, int]:
    b_start = len(s[:max(0, c_start)].encode("utf-8", errors="replace"))
    b_end = len(s[:max(0, c_end)].encode("utf-8", errors="replace"))
    return min(b_start, max_bytes), min(b_end, max_bytes)


def extract_byte_spans_v3(raw_name: str, max_bytes: int = 256) -> dict[str, Any]:
    """Extract byte-anchored `(title, year, episode_spec)` spans from a UTF-8-safe `max_bytes` window of `raw_name`."""
    nm, _nm_bytes, was_truncated = _utf8_safe_window(raw_name or "", max_bytes=max_bytes)
    work = TRAILING_EXT_RE.sub("", nm) if not was_truncated else nm

    c_start = 0
    m_site = LEADING_SITE_RE.match(work)
    if m_site and (was_truncated or m_site.end() < len(work) - 3):
        c_start = m_site.end()
    m_pref = LEADING_PREFIX_RE.match(work[c_start:])
    if m_pref and (was_truncated or (c_start + m_pref.end()) < len(work) - 2):
        c_start += m_pref.end()

    if was_truncated:
        m_unclosed = re.match(r"^\s*(?:\[[^\]]*|\([^\)]*|【[^】]*|★[^★]*)$", work[c_start:])
        if m_unclosed:
            c_start = len(work)

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

    while c_end > c_start and work[c_end - 1] in " ._-([{|:/":
        c_end -= 1
    m_junk = TRAILING_JUNK_RE.search(work[c_start:c_end])
    if m_junk:
        c_end = c_start + m_junk.start()
        while c_end > c_start and work[c_end - 1] in " ._-([{|:/":
            c_end -= 1
    base_c_start = c_start
    token_spans = list(re.finditer(r"[^\._\s]+", work[base_c_start:c_end]))
    while len(token_spans) > 1 and key_norm(token_spans[-1].group(0)) in TECH_TOKEN_SET:
        token_spans.pop()
        c_end = base_c_start + token_spans[-1].end()
        while c_end > base_c_start and work[c_end - 1] in " ._-([{|:/":
            c_end -= 1

    valid_tok_matches = [m for m in token_spans if m.group(0).strip(" ._-([{|:") != ""]
    if valid_tok_matches:
        c_start = base_c_start + valid_tok_matches[0].start()
        c_end = base_c_start + valid_tok_matches[-1].end()
        while c_start < c_end and work[c_start] in " ._-([{|:/":
            c_start += 1
        while c_end > c_start and work[c_end - 1] in " ._-([{|:/":
            c_end -= 1
        words = [
            m.group(0).strip(" ._-([{|:")
            for m in re.finditer(r"[^\._\s]+", work[c_start:c_end])
            if m.group(0).strip(" ._-([{|:") != ""
        ]
        clean_title = " ".join(words).strip()
    else:
        clean_title = ""
        c_start, c_end = 0, 0

    if len(key_norm(clean_title)) < 2:
        if base_c_start < len(work):
            fb_matches = [
                m
                for m in re.finditer(r"[^\._\s]+", work[base_c_start:])
                if m.group(0).strip(" ._-([{|:") != "" and key_norm(m.group(0)) not in TECH_TOKEN_SET
            ][:6]
        else:
            fb_matches = []
        if fb_matches:
            c_start = base_c_start + fb_matches[0].start()
            c_end = base_c_start + fb_matches[-1].end()
            while c_start < c_end and work[c_start] in " ._-([{|:/":
                c_start += 1
            while c_end > c_start and work[c_end - 1] in " ._-([{|:/":
                c_end -= 1
            clean_title = " ".join(
                m.group(0).strip(" ._-([{|:")
                for m in re.finditer(r"[^\._\s]+", work[c_start:c_end])
                if m.group(0).strip(" ._-([{|:") != "" and key_norm(m.group(0)) not in TECH_TOKEN_SET
            ).strip()
        if len(key_norm(clean_title)) < 2:
            clean_title = ""
            c_start, c_end = 0, 0

    t_bstart, t_bend = _char_to_byte_span(nm, c_start, c_end, max_bytes=max_bytes)
    if not clean_title or t_bstart >= t_bend:
        clean_title = ""
        t_bstart, t_bend = 0, 0

    year_val = None
    year_bspan = None
    ym = YEAR_SPAN_RE.search(nm, pos=max(0, c_end - 1)) or YEAR_SPAN_RE.search(nm)
    if ym:
        yb_s, yb_e = _char_to_byte_span(nm, ym.start(1), ym.end(1), max_bytes=max_bytes)
        if yb_e - yb_s == 4:
            year_val = int(ym.group(1))
            year_bspan = (yb_s, yb_e)

    ep_val = None
    ep_bspan = None
    em = EPISODE_SPAN_RE.search(nm, pos=max(0, c_end - 1)) or EPISODE_SPAN_RE.search(nm)
    if em:
        eb_s, eb_e = _char_to_byte_span(nm, em.start(1), em.end(1), max_bytes=max_bytes)
        if eb_s < eb_e:
            ep_val = em.group(1).strip()
            ep_bspan = (eb_s, eb_e)

    anchored = bool(clean_title and t_bstart < t_bend and is_strict_clean_title(clean_title, nm))
    return {
        "title": clean_title,
        "title_byte_span": (t_bstart, t_bend),
        "year": year_val,
        "year_byte_span": year_bspan,
        "episode_spec": ep_val,
        "episode_byte_span": ep_bspan,
        "is_anchored": anchored,
        "extractor_source": "structural_byte_span",
    }


def _extract_contiguous_bio_runs(tags: Sequence[int], b_tag: int, i_tag: int) -> list[tuple[int, int]]:
    """Decode contiguous BIO entity spans `[start, end)`, honoring `b_tag` as starting a new entity."""
    runs: list[tuple[int, int]] = []
    cur_s: int | None = None
    cur_e: int = 0
    for idx, t in enumerate(tags):
        if t == b_tag:
            if cur_s is not None:
                runs.append((cur_s, cur_e))
            cur_s = idx
            cur_e = idx + 1
        elif t == i_tag:
            if cur_s is not None and idx == cur_e:
                cur_e = idx + 1
            else:
                if cur_s is not None:
                    runs.append((cur_s, cur_e))
                cur_s = idx
                cur_e = idx + 1
        else:
            if cur_s is not None:
                runs.append((cur_s, cur_e))
                cur_s = None
    if cur_s is not None:
        runs.append((cur_s, cur_e))
    return runs


def decode_bio_tags_raw_neural(raw_name: str, bio_tags: Sequence[int] | Any) -> dict[str, Any]:
    """Pure neural BIO span decoder:
    Decodes `(title, year, episode_spec)` strictly from the 9th neural head `bio_tags`
    (`0=O, 1=B-TITLE, 2=I-TITLE, 3=B-YEAR, 4=I-YEAR, 5=B-EP, 6=I-EP`) with zero call to `extract_byte_spans_v3`.
    """
    nm, raw_bytes, _ = _utf8_safe_window(raw_name or "", max_bytes=256)
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

    year_runs = _extract_contiguous_bio_runs(tags, BIO_B_YEAR, BIO_I_YEAR)
    year_candidates: list[tuple[int, tuple[int, int]]] = []
    for ys, ye in year_runs:
        if ye - ys < 4:
            continue
        for m_y in re.finditer(rb"(?<!\d)(19[2-9]\d|20[0-2]\d)(?!\d)", raw_bytes[ys:ye]):
            exact_yb, exact_ye = ys + m_y.start(1), ys + m_y.end(1)
            if exact_ye - exact_yb == 4:
                year_candidates.append((int(m_y.group(1).decode("ascii")), (exact_yb, exact_ye)))

    year_val = None
    year_bspan = None
    if year_candidates:
        post_title = [c for c in year_candidates if c[1][0] >= te]
        selected_year = post_title[0] if post_title else year_candidates[0]
        year_val, year_bspan = selected_year

    ep_runs = _extract_contiguous_bio_runs(tags, BIO_B_EP, BIO_I_EP)
    ep_val = None
    ep_bspan = None
    for es, ee in ep_runs:
        if ee - es < 2:
            continue
        while es < ee and raw_bytes[es : es + 1] in b" ._-[({/|:)]}":
            es += 1
        while ee > es and raw_bytes[ee - 1 : ee] in b" ._-[({/|:)]}":
            ee -= 1
        if ee - es >= 2:
            cand_ep = raw_bytes[es:ee].decode("utf-8", errors="replace").strip()
            if cand_ep:
                ep_val = cand_ep
                ep_bspan = (es, ee)
                if es >= te:
                    break

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
