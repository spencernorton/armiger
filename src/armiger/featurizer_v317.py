"""Fast deterministic Plane-1/2 featurizer + UTF-8-safe BIO+Pointer title span decoder."""
from __future__ import annotations

import math
import os
import re
from typing import Any

import numpy as np

from canonical_views import (
    policy2_v2_signals,
    project_audio_text,
    view_a_raw_nfkc,
    view_c_byte_obf_stats,
)
from tokenizer_v317 import PurePythonUnigramTokenizer

MAX_T = 48        # Main subword length (View A title + top file summary)
MAX_TA = 32       # Isolated audio subword length (strictly P(x)!)
MAX_L = 192       # Raw UTF-8 byte length (head+tail for >192; 0..192 exact for spans!)
MAX_F = 64        # Max files per torrent
FILE_TOK_K = 8    # Subword tokens per file (2x richer than v3.16.1's 4 words!)
FILE_NUM_D = 12   # Per-file numeric features
TREE_SUM_D = 48   # Global tree + View-C + Policy-2 summary vector

_VID_EXT = frozenset({".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".webm", ".flv", ".vob", ".rmvb"})
_AUD_MUSIC_EXT = frozenset({".mp3", ".flac", ".wav", ".ogg", ".opus", ".aac", ".wma", ".alac", ".ape", ".wv", ".aiff", ".m4a"})
_AUD_BOOK_EXT = frozenset({".m4b", ".aa", ".aax"})
_IMG_EXT = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".avif", ".heic"})
_COMIC_EXT = frozenset({".cbz", ".cbr", ".cb7", ".cbt", ".cba"})
_EBOOK_EXT = frozenset({".epub", ".mobi", ".azw", ".azw3", ".fb2", ".djvu", ".lit", ".pdb", ".pdf", ".chm", ".rtf", ".docx", ".txt"})
_ARC_EXT = frozenset({".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".r00", ".r01", ".part1"})
_EXEC_EXT = frozenset({".exe", ".msi", ".com", ".apk", ".dmg", ".pkg", ".deb", ".rpm", ".appimage", ".bin"})
_DROPPER_EXT = frozenset({".lnk", ".scr", ".hta", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".ps1", ".cmd", ".bat", ".pif", ".cpl", ".reg", ".url", ".webloc"})
_GAME_EXT = frozenset({
    ".pak", ".vpk", ".bsa", ".ba2", ".wad", ".pk3", ".pk4", ".rpf", ".forge", ".bik", ".bk2",
    ".ucas", ".utoc", ".pck", ".unity3d", ".assets", ".ress", ".bnk", ".fsb", ".xnb", ".esm",
    ".esp", ".scs", ".sii", ".cpk", ".obb", ".nsp", ".xci", ".cia", ".3ds", ".nds", ".iso",
    ".cso", ".chd", ".rvz", ".wbfs", ".gba", ".smc", ".sfc", ".nes", ".z64", ".dll", ".so",
})
_SUB_EXT = frozenset({".srt", ".ass", ".ssa", ".sub", ".idx", ".vtt", ".sup", ".mks"})
_INFO_EXT = frozenset({".nfo", ".sfv", ".md5", ".sha1", ".sha256", ".cue", ".log", ".m3u", ".m3u8", ".pls"})

_CHAP_RE = re.compile(r"(?:chapter|ch[\s._-]?\d+|part[\s._-]?\d+|disc[\s._-]?\d+|cd[\s._-]?\d+|unabridged|audiobook|audible)", re.IGNORECASE)
_EP_RE = re.compile(r"(?:s\d{1,2}e\d{1,3}|\b\d{1,2}x\d{2}\b|episode[\s._-]?\d+|ep[\s._-]?\d+)", re.IGNORECASE)
_MUSIC_RE = re.compile(r"(?:flac|320\s?kbps|24bit|vinyl|discography|remix|ost|soundtrack|\bep\b|track[\s._-]?\d+|v0\b|mp3\s?320)", re.IGNORECASE)
_COURSE_RE = re.compile(r"(?:udemy|coursera|pluralsight|masterclass|lynda|linkedin[\s._-]?learning|tutorial|lesson[\s._-]?\d+|module[\s._-]?\d+|lecture[\s._-]?\d+|bootcamp)", re.IGNORECASE)
_COMIC_RE = re.compile(r"(?:manga|comic|issue[\s._-]?#?\d+|vol(?:ume)?[\s._-]?\d+|tome[\s._-]?\d+|cbz|cbr|c2c|digital[\s._-]?empire)", re.IGNORECASE)
_SAMPLE_RE = re.compile(r"(?:sample|trailer|preview|readme)", re.IGNORECASE)


def encode_raw_bytes_head_tail(name: str, max_l: int = MAX_L) -> np.ndarray:
    """Encode raw UTF-8 bytes preserving exact byte indices [0..128) for spans + tail bytes for long titles."""
    b = (name or "").encode("utf-8", errors="ignore")
    out = np.zeros(max_l, dtype=np.int64)
    n = len(b)
    if n == 0:
        return out
    if n <= max_l:
        out[:n] = np.frombuffer(b, dtype=np.uint8).astype(np.int64) + 1  # 0=PAD, 1..256=byte+1
    else:
        head_k = 128
        tail_k = max_l - head_k
        out[:head_k] = np.frombuffer(b[:head_k], dtype=np.uint8).astype(np.int64) + 1
        out[head_k:] = np.frombuffer(b[-tail_k:], dtype=np.uint8).astype(np.int64) + 1
    return out


def featurize_single_row(
    row: dict[str, Any],
    tok: PurePythonUnigramTokenizer,
    max_t: int = MAX_T,
    max_ta: int = MAX_TA,
    max_l: int = MAX_L,
    max_f: int = MAX_F,
    aud_txt: str | None = None,
) -> dict[str, np.ndarray]:
    """Produce all 7 Plane-1 inputs + Plane-2 tree summary vector for one torrent row."""
    name = str(row.get("name") or "")
    files = row.get("files") or []
    n_files_raw = len(files)
    row_size = max(0, int(row.get("size") or 0))

    # View A (multilingual NFKC) -> main subwords
    va = view_a_raw_nfkc(name)
    sub_ids = tok.encode(va, max_len=max_t, add_cls=True)
    subword_ids = np.zeros(max_t, dtype=np.int64)
    if sub_ids:
        subword_ids[: len(sub_ids)] = sub_ids

    # Deterministic Audio-Only Projection P(x) -> isolated audio branch subwords
    if aud_txt is None:
        aud_txt = project_audio_text(name, files)
    aud_ids = tok.encode(aud_txt, max_len=max_ta, add_cls=True)
    audio_subword_ids = np.zeros(max_ta, dtype=np.int64)
    if aud_ids:
        audio_subword_ids[: len(aud_ids)] = aud_ids

    # Raw UTF-8 bytes -> Byte ConvNeXt + span head
    byte_ids = encode_raw_bytes_head_tail(name, max_l=max_l)

    # Parse files (deterministic descending-size + path tiebreak, top max_f for set encoder, up to 256 for tree stats)
    parsed: list[tuple[int, str, str, str, int]] = []
    tot_file_bytes = 0
    for f in files[:256]:
        if isinstance(f, dict):
            p = str(f.get("path") or f.get("name") or f.get("p") or "")
            sz = max(0, int(f.get("size") or f.get("length") or f.get("s") or 0))
        else:
            p = str(f)
            sz = 0
        if not p:
            continue
        pl = p.replace("\\", "/")
        parts = [seg for seg in pl.split("/") if seg]
        depth = max(0, len(parts) - 1)
        base = parts[-1] if parts else pl
        _, ext = os.path.splitext(base.lower())
        tot_file_bytes += sz
        parsed.append((sz, pl, base, ext, depth))

    tot_bytes = max(row_size, tot_file_bytes, 1)

    # Compute rich tree-summary counts & byte shares across up to 256 files
    cnt_vid = cnt_aud = cnt_m4b = cnt_img = cnt_comic = cnt_ebook = cnt_pdf = 0
    cnt_arc = cnt_exec = cnt_drop = cnt_game = cnt_sub = cnt_info = cnt_other = 0
    b_vid = b_aud = b_m4b = b_img = b_comic = b_ebook = b_arc = b_exec = b_game = b_sub = 0
    aud_sizes: list[int] = []
    depths: list[int] = []
    top_dirs: set[str] = set()
    n_chap_hit = n_ep_hit = n_mus_hit = n_course_hit = n_comic_hit = 0
    ext_counts: dict[str, int] = {}

    for sz, pl, base, ext, depth in parsed:
        depths.append(depth)
        if "/" in pl:
            top_dirs.add(pl.split("/", 1)[0].lower())
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        if ext in _VID_EXT:
            cnt_vid += 1; b_vid += sz
        elif ext in _AUD_BOOK_EXT:
            cnt_m4b += 1; cnt_aud += 1; b_m4b += sz; b_aud += sz; aud_sizes.append(sz)
        elif ext in _AUD_MUSIC_EXT:
            cnt_aud += 1; b_aud += sz; aud_sizes.append(sz)
        elif ext in _COMIC_EXT:
            cnt_comic += 1; b_comic += sz
        elif ext in _IMG_EXT:
            cnt_img += 1; b_img += sz
        elif ext in _EBOOK_EXT:
            cnt_ebook += 1; b_ebook += sz
            if ext == ".pdf":
                cnt_pdf += 1
        elif ext in _ARC_EXT:
            cnt_arc += 1; b_arc += sz
        elif ext in _DROPPER_EXT:
            cnt_drop += 1; b_exec += sz
        elif ext in _EXEC_EXT:
            cnt_exec += 1; b_exec += sz
        elif ext in _GAME_EXT:
            cnt_game += 1; b_game += sz
        elif ext in _SUB_EXT:
            cnt_sub += 1; b_sub += sz
        elif ext in _INFO_EXT:
            cnt_info += 1
        else:
            cnt_other += 1

        pass

    if parsed:
        joined_pl = "\n".join(x[1] for x in parsed)
        joined_bs = "\n".join(x[2] for x in parsed)
        if _CHAP_RE.search(joined_pl):
            n_chap_hit = sum(1 for x in parsed if _CHAP_RE.search(x[1]))
        if _EP_RE.search(joined_bs):
            n_ep_hit = sum(1 for x in parsed if _EP_RE.search(x[2]))
        if _MUSIC_RE.search(joined_pl):
            n_mus_hit = sum(1 for x in parsed if _MUSIC_RE.search(x[1]))
        if _COURSE_RE.search(joined_pl):
            n_course_hit = sum(1 for x in parsed if _COURSE_RE.search(x[1]))
        if _COMIC_RE.search(joined_pl):
            n_comic_hit = sum(1 for x in parsed if _COMIC_RE.search(x[1]))

    # Sort files deterministically for set encoder (descending size, then path)
    parsed.sort(key=lambda x: (-x[0], x[1]))
    top_parsed = parsed[:max_f]
    F_eff = len(top_parsed)

    file_tok_ids = np.zeros((max_f, FILE_TOK_K), dtype=np.int64)
    file_num_feats = np.zeros((max_f, FILE_NUM_D), dtype=np.float32)
    file_mask = np.zeros(max_f, dtype=np.float32)

    for j, (sz, pl, base, ext, depth) in enumerate(top_parsed):
        file_mask[j] = 1.0
        # Keep parent/basename for rich context (e.g. "Crack/steam_api64.dll" or "Chapter 01/part1.mp3")
        parts = [seg for seg in pl.split("/") if seg]
        ftxt = "/".join(parts[-2:]) if len(parts) >= 2 else base
        fids = tok.encode(ftxt, max_len=FILE_TOK_K, add_cls=False)
        if fids:
            file_tok_ids[j, : len(fids)] = fids
        file_num_feats[j, 0] = math.log1p(sz) / 25.0
        file_num_feats[j, 1] = float(sz) / tot_bytes
        file_num_feats[j, 2] = min(depth, 8) / 8.0
        file_num_feats[j, 3] = 1.0 if ext in _VID_EXT else 0.0
        file_num_feats[j, 4] = 1.0 if (ext in _AUD_MUSIC_EXT or ext in _AUD_BOOK_EXT) else 0.0
        file_num_feats[j, 5] = 1.0 if ext in _IMG_EXT else 0.0
        file_num_feats[j, 6] = 1.0 if ext in _ARC_EXT else 0.0
        file_num_feats[j, 7] = 1.0 if (ext in _EXEC_EXT or ext in _DROPPER_EXT) else 0.0
        file_num_feats[j, 8] = 1.0 if ext in _GAME_EXT else 0.0
        file_num_feats[j, 9] = 1.0 if (ext in _COMIC_EXT or ext in _EBOOK_EXT) else 0.0
        file_num_feats[j, 10] = 1.0 if ext in _SUB_EXT else 0.0
        file_num_feats[j, 11] = 1.0 if (ext in _INFO_EXT or _SAMPLE_RE.search(base)) else 0.0

    # Assemble 48-dim tree_summary + View-C + Policy-2 v2 vector
    nf = max(1, len(parsed))
    vc = view_c_byte_obf_stats(name)
    p2 = policy2_v2_signals(name, files)

    # Extension entropy
    ext_ent = 0.0
    if parsed:
        for c in ext_counts.values():
            pr = c / nf
            ext_ent -= pr * math.log(pr + 1e-12)
    avg_aud_mb = (sum(aud_sizes) / len(aud_sizes) / 1e6) if aud_sizes else 0.0
    max_aud_mb = (max(aud_sizes) / 1e6) if aud_sizes else 0.0
    mean_depth = (sum(depths) / len(depths)) if depths else 0.0
    max_depth = max(depths) if depths else 0

    ts = np.zeros(TREE_SUM_D, dtype=np.float32)
    # 0..3: regime & scale
    ts[0] = 1.0 if n_files_raw == 0 else 0.0
    ts[1] = 1.0 if n_files_raw == 1 else 0.0
    ts[2] = math.log1p(n_files_raw) / 7.0
    ts[3] = math.log1p(tot_bytes) / 26.0
    # 4..13: byte shares by media/structural type
    ts[4] = b_vid / tot_bytes
    ts[5] = b_aud / tot_bytes
    ts[6] = b_m4b / tot_bytes
    ts[7] = b_img / tot_bytes
    ts[8] = b_comic / tot_bytes
    ts[9] = b_ebook / tot_bytes
    ts[10] = b_arc / tot_bytes
    ts[11] = b_exec / tot_bytes
    ts[12] = b_game / tot_bytes
    ts[13] = b_sub / tot_bytes
    # 14..25: file count shares by type
    ts[14] = cnt_vid / nf
    ts[15] = cnt_aud / nf
    ts[16] = cnt_m4b / nf
    ts[17] = cnt_img / nf
    ts[18] = cnt_comic / nf
    ts[19] = cnt_ebook / nf
    ts[20] = cnt_pdf / nf
    ts[21] = cnt_arc / nf
    ts[22] = cnt_exec / nf
    ts[23] = cnt_drop / nf
    ts[24] = cnt_game / nf
    ts[25] = cnt_sub / nf
    # 26..35: fine-grained structural disambiguators (music vs audiobook, comic, course, game, hierarchy)
    ts[26] = math.log1p(avg_aud_mb) / 6.0
    ts[27] = math.log1p(max_aud_mb) / 7.0
    ts[28] = 1.0 if (cnt_aud > 0 and avg_aud_mb >= 20.0) else 0.0
    ts[29] = mean_depth / 5.0
    ts[30] = min(max_depth, 10) / 10.0
    ts[31] = ext_ent / 3.0
    ts[32] = math.log1p(len(top_dirs)) / 4.0
    ts[33] = (n_chap_hit / nf) + (0.5 if _CHAP_RE.search(name) else 0.0)
    ts[34] = (n_ep_hit / nf) + (0.5 if _EP_RE.search(name) else 0.0)
    ts[35] = (n_mus_hit / nf) + (0.5 if _MUSIC_RE.search(name) else 0.0)
    # 36..39: domain lexical signals + Policy-2 v2 flags
    ts[36] = (n_course_hit / nf) + (0.5 if _COURSE_RE.search(name) else 0.0)
    ts[37] = (n_comic_hit / nf) + (0.5 if _COMIC_RE.search(name) else 0.0)
    ts[38] = 1.0 if p2["hard_dropper_trap"] else 0.0
    ts[39] = 1.0 if p2["game_demotion_eligible"] else 0.0
    # 40..47: Policy-2 extra counts + View-C byte obfuscation signals
    ts[40] = min(p2["n_pass_trap"], 5) / 5.0
    ts[41] = min(p2["n_redirect"], 5) / 5.0
    ts[42] = min(vc["zw_count"], 10.0) / 10.0
    ts[43] = vc["non_ascii_ratio"]
    ts[44] = vc["digit_ratio"]
    ts[45] = vc["punct_ratio"]
    ts[46] = vc["mixed_script_flag"]
    ts[47] = math.log1p(vc["byte_len"]) / 6.0

    return {
        "subword_ids": subword_ids,
        "audio_subword_ids": audio_subword_ids,
        "byte_ids": byte_ids,
        "file_tok_ids": file_tok_ids,
        "file_num_feats": file_num_feats,
        "file_mask": file_mask,
        "tree_summary": ts,
        "p2": p2,
        "vc": vc,
    }


def _is_word_char_nfkc(ch: str) -> bool:
    import unicodedata as _ud
    return any(c.isalnum() for c in _ud.normalize("NFKC", ch))


def decode_span_utf8_safe(
    raw_name: str,
    bio_logits: np.ndarray,
    ptr_logits: np.ndarray | None = None,
) -> tuple[str, int, int, bool]:
    """
    Decode byte-level BIO + Start/End Pointer logits into a clean title string with
    UNCONDITIONAL UTF-8 codepoint AND whole-word boundary enforcement (100% valid UTF-8 + 100% whole words!).
    """
    s_in = str(raw_name or "")
    try:
        raw_b = s_in.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        s_in = s_in.encode("utf-8", errors="replace").decode("utf-8")
        raw_b = s_in.encode("utf-8", errors="strict")
    n = min(len(raw_b), len(bio_logits))
    if n <= 0:
        return "", 0, 0, True

    # Ensure terminal byte index end_n does not bisect a multi-byte UTF-8 codepoint
    end_n = n
    while end_n > 0 and end_n < len(raw_b) and (raw_b[end_n] & 0xC0) == 0x80:
        end_n -= 1
    if end_n <= 0:
        return "", 0, 0, True

    prefix_s = raw_b[:end_n].decode("utf-8", errors="ignore")
    if not prefix_s:
        return "", 0, 0, True

    # Map character positions in prefix_s to exact UTF-8 byte offsets and word flags
    m_chars = len(prefix_s)
    char_b_off = [0] * (m_chars + 1)
    is_w = [False] * m_chars
    acc_b = 0
    for ci, ch in enumerate(prefix_s):
        char_b_off[ci] = acc_b
        acc_b += len(ch.encode("utf-8"))
        is_w[ci] = _is_word_char_nfkc(ch)
    char_b_off[m_chars] = acc_b

    # Word-start and word-end byte offsets (prevents mid-word slices like "Matri" from "Matrix")
    word_starts = [char_b_off[ci] for ci in range(m_chars) if is_w[ci] and (ci == 0 or not is_w[ci - 1])]
    word_ends = [char_b_off[ci + 1] for ci in range(m_chars) if is_w[ci] and (ci + 1 == m_chars or not is_w[ci + 1])]
    # If the last word in prefix_s was truncated at end_n < len(raw_b), exclude that mid-word cut if an earlier word end exists
    if end_n < len(raw_b) and len(word_ends) >= 2:
        full_s = raw_b.decode("utf-8", errors="ignore")
        if len(full_s) > m_chars and is_w[m_chars - 1] and _is_word_char_nfkc(full_s[m_chars]):
            word_ends = word_ends[:-1]

    valid_starts = word_starts if word_starts else [i for i in range(end_n) if (raw_b[i] & 0xC0) != 0x80]
    valid_ends = word_ends if word_ends else [i for i in range(1, end_n + 1) if i == end_n or (raw_b[i] & 0xC0) != 0x80]
    if not valid_starts or not valid_ends:
        return "", 0, 0, True

    bl = bio_logits[:end_n].astype(np.float64)
    bl = bl - bl.max(axis=1, keepdims=True)
    bp = np.exp(bl)
    bp /= bp.sum(axis=1, keepdims=True)
    p_b = bp[:, 1]
    p_in = bp[:, 1] + bp[:, 2]

    if ptr_logits is not None and len(ptr_logits) >= end_n:
        pl = ptr_logits[:end_n].astype(np.float64)
        s_log = pl[:, 0] - np.max(pl[:, 0])
        e_log = pl[:, 1] - np.max(pl[:, 1])
        ps = np.exp(s_log); ps /= ps.sum() + 1e-12
        pe = np.exp(e_log); pe /= pe.sum() + 1e-12
        pref = np.zeros(end_n + 1, dtype=np.float64)
        pref[1:] = np.cumsum(p_in - 0.35)
        s_cands = sorted(valid_starts, key=lambda i: float(ps[i] + 0.35 * p_b[i]), reverse=True)[:10]
        e_cands = sorted(valid_ends, key=lambda j: float(pe[j - 1]), reverse=True)[:10]
        best_sc = -1e18
        best_s, best_e = valid_starts[0], valid_ends[-1]
        for s in s_cands:
            for e in e_cands:
                if e <= s or (e - s) > 160:
                    continue
                sc = math.log(ps[s] + 1e-9) + math.log(pe[e - 1] + 1e-9) + 0.45 * (pref[e] - pref[s])
                if sc > best_sc:
                    best_sc = sc
                    best_s, best_e = s, e
    else:
        best_s, best_e = valid_starts[0], valid_ends[-1]

    # Snap best_s and best_e to whole-word boundaries if any word char exists
    if word_starts and word_ends:
        if best_s not in word_starts:
            best_s = min(word_starts, key=lambda ws: abs(ws - best_s))
        valid_ge = [we for we in word_ends if we > best_s]
        if valid_ge:
            if best_e not in valid_ge:
                best_e = min(valid_ge, key=lambda we: abs(we - best_e))
        else:
            best_s, best_e = word_starts[0], word_ends[0]

    seg = raw_b[best_s:best_e]
    try:
        decoded = seg.decode("utf-8", errors="strict").strip(" ._-[]()")
        utf8_ok = True
    except UnicodeDecodeError:
        decoded = seg.decode("utf-8", errors="ignore").strip(" ._-[]()")
        utf8_ok = False
    if not decoded and word_starts and word_ends:
        best_s, best_e = word_starts[0], word_ends[min(len(word_ends) - 1, 3)]
        decoded = raw_b[best_s:best_e].decode("utf-8", errors="ignore").strip(" ._-[]()")
    return decoded, int(best_s), int(best_e), utf8_ok


# ---------------------------------------------------------------------------
# Fine-Grained File-Tree & Scene Disambiguator (18-dim) for Plane-2 Stage-2
# Directly targets movie<->tv_show, software<->game, comic<->ebook, xxx<->comic, music<->audiobook
# ---------------------------------------------------------------------------
TREE_FINE_DIM = 18

_TF_SXXEYY_RE = re.compile(r"(?i)(?<![a-z0-9])(?:s\d{1,2}[._\s-]*e\d{1,3}|\d{1,2}x\d{2,3}|ep(?:isode)?[._\s-]*\d{1,3}|season[._\s-]*\d{1,2})(?![a-z0-9])")
_TF_SEASON_NUM_RE = re.compile(r"(?i)(?<![a-z0-9])(?:s(\d{1,2})e\d{1,3}|season[._\s-]*(\d{1,2}))(?![a-z0-9])")
_TF_COMPLETE_SER_RE = re.compile(r"(?i)(?:complete[._\s-]*series|complete[._\s-]*season|s\d{1,2}[._\s-]*complete|seasons?[._\s-]*\d+[._\s-]*[-–to]+[._\s-]*\d+)")
_TF_SPORTS_RE = re.compile(r"(?i)(?<![a-z0-9])(?:ufc|nfl|nba|nhl|mlb|f1|formula[._\s-]*1|motogp|wwe|aew|premier[._\s-]*league|champions[._\s-]*league|la[._\s-]*liga|bundesliga|serie[._\s-]*a|world[._\s-]*cup|olympics|wimbledon|pga|nascar|boxing|match[._\s-]*of[._\s-]*the[._\s-]*day)(?![a-z0-9])")
_TF_GAME_SCENE_RE = re.compile(r"(?i)(?<![a-z0-9])(?:fitgirl|dodi|codex|skidrow|rune|empress|elamigos|gog|flt|plaza|cpy|tenoke|reloaded|hoodlum|razor1911|darksiders|tinyiso|kaos|xatab|goldberg|steamrip|insaneramzes|chronos|p2p[._\s-]*game|repack[._\s-]*by[._\s-]*fitgirl|switch[._\s-]*nsp|ps4[._\s-]*pkg|xbox360|pc[._\s-]*game)(?![a-z0-9])")
_TF_SW_CUE_RE = re.compile(r"(?i)(?<![a-z0-9])(?:keygen|patch|portable|pre[\s._-]*activated|multilingual|x64|x86|win64|win32|macos|vst|vst3|aax|plugin|plugins|sdk|firmware|driver|drivers|adobe|autodesk|jetbrains|microsoft[._\s-]*office|windows[._\s-]*(?:10|11|server)|photoshop|illustrator|premiere|acrobat|autocad|solidworks|matlab|vmware|parallels|coreldraw|ableton|fl[._\s-]*studio|kontakt|cubase|daw|cracked[._\s-]*mac|linux[._\s-]*x86_64|iso[._\s-]*bootable)(?![a-z0-9])")
_TF_SW_VER_RE = re.compile(r"(?i)(?<![a-z0-9])(?:v?\d{1,2}\.\d{1,3}(?:\.\d{1,4})+|build[._\s-]*\d{3,6}|r\d{4}[ab]?)(?![a-z0-9])")
_TF_NSFW_CG_RE = re.compile(r"(?i)(?:成年コミック|r18|r-18|hentai|doujin|doujinshi|eroge|ecchi|sukebei|fanbox|patreon[._\s-]*(?:nsfw|lewd|uncensored)|uncensored[._\s-]*(?:cg|pack)|jav\b|onlyfans|manyvids|brazzers|naughtyamerica|blacked|vixen|bangbros|pornhub|xvideos|chaturbate|camrip|xxx)")
_TF_AB_TREE_RE = re.compile(r"(?i)(?:_librivox\b|\blibrivox\b|\baudiobook\b|\bunabridged\b|\bh[oö]rbuch\b|\bchapter[\s._-]*\d{1,3}\b|\bpart[\s._-]*\d{1,2}\b.*\.(?:mp3|m4b|m4a))")
_TF_ROM_EXTS = frozenset({".nsp", ".xci", ".cia", ".3ds", ".nds", ".gba", ".gbc", ".smc", ".sfc", ".nes", ".n64", ".z64", ".rvz", ".wbfs", ".chd", ".cso", ".vpk", ".pkg", ".pbp", ".wad"})
_TF_GAME_AST_EXTS = frozenset({".pak", ".bsa", ".ba2", ".pk3", ".pk4", ".rpf", ".forge", ".ucas", ".utoc", ".unity3d", ".assets", ".bnk", ".fsb", ".xnb", ".esm", ".esp", ".bgl", ".sii", ".scs", ".cpk"})
_TF_COMIC_EXTS = frozenset({".cbz", ".cbr", ".cb7", ".cba", ".cbt"})
_TF_EBOOK_EXTS = frozenset({".epub", ".mobi", ".azw", ".azw3", ".fb2", ".djvu", ".lit", ".opf"})
_TF_IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"})
_TF_VID_EXTS = frozenset({".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".webm", ".flv", ".vob"})


def extract_tree_fine_18d(name: str, files: list[Any] | None = None) -> np.ndarray:
    """18-dim fine-grained file-tree & scene disambiguator features."""
    out = np.zeros(TREE_FINE_DIM, dtype=np.float32)
    nm = str(name or "")
    fl = files or []
    n_ep_files = 0
    seasons: set[int] = set()
    n_vid = 0
    n_rom = 0
    n_gast = 0
    n_cbz = 0
    n_ebk = 0
    n_pdf = 0
    n_img = 0
    n_other_non_meta = 0
    ab_tree_hit = bool(_TF_AB_TREE_RE.search(nm))
    nsfw_hit = bool(_TF_NSFW_CG_RE.search(nm))
    game_scene_hit = bool(_TF_GAME_SCENE_RE.search(nm))
    sw_cue_hit = bool(_TF_SW_CUE_RE.search(nm))
    for m in _TF_SEASON_NUM_RE.finditer(nm):
        s_num = m.group(1) or m.group(2)
        if s_num:
            seasons.add(int(s_num))
    for f in fl[:256]:
        p = str(f.get("path") or f.get("p") or f.get("name") or "") if isinstance(f, dict) else str(f or "")
        pl = p.lower().replace("\\", "/")
        base = pl.rsplit("/", 1)[-1]
        _, ext = os.path.splitext(base)
        if ext in _TF_VID_EXTS:
            n_vid += 1
            if _TF_SXXEYY_RE.search(pl):
                n_ep_files += 1
                for m in _TF_SEASON_NUM_RE.finditer(pl):
                    s_num = m.group(1) or m.group(2)
                    if s_num:
                        seasons.add(int(s_num))
        elif ext in _TF_ROM_EXTS:
            n_rom += 1
        elif ext in _TF_GAME_AST_EXTS:
            n_gast += 1
        elif ext in _TF_COMIC_EXTS:
            n_cbz += 1
        elif ext in _TF_EBOOK_EXTS:
            n_ebk += 1
        elif ext == ".pdf":
            n_pdf += 1
        elif ext in _TF_IMG_EXTS:
            n_img += 1
        elif ext not in (".txt", ".nfo", ".sfv", ".md5", ".srt", ".ass", ".sub", ".idx", ".vtt"):
            n_other_non_meta += 1
        pass
    if fl and not (ab_tree_hit and game_scene_hit and sw_cue_hit):
        top_joined = " ".join(
            (str(f.get("path") or f.get("p") or f.get("name") or "") if isinstance(f, dict) else str(f or ""))
            for f in fl[:16]
        )
        if (not ab_tree_hit) and _TF_AB_TREE_RE.search(top_joined):
            ab_tree_hit = True
        if (not game_scene_hit) and _TF_GAME_SCENE_RE.search(top_joined):
            game_scene_hit = True
        if (not sw_cue_hit) and _TF_SW_CUE_RE.search(top_joined):
            sw_cue_hit = True
    tot_c = max(1, n_vid + n_rom + n_gast + n_cbz + n_ebk + n_pdf + n_img + n_other_non_meta)
    out[0] = 1.0 if (_TF_SXXEYY_RE.search(nm) or n_ep_files > 0) else 0.0
    out[1] = math.log1p(n_ep_files) / 4.0
    out[2] = min(len(seasons), 10) / 10.0
    out[3] = 1.0 if n_vid == 1 else 0.0
    out[4] = 1.0 if n_vid >= 3 else 0.0
    out[5] = 1.0 if _TF_COMPLETE_SER_RE.search(nm) else 0.0
    out[6] = 1.0 if _TF_SPORTS_RE.search(nm) else 0.0
    out[7] = 1.0 if game_scene_hit else 0.0
    out[8] = 1.0 if n_rom > 0 else 0.0
    out[9] = 1.0 if n_gast > 0 else 0.0
    out[10] = 1.0 if sw_cue_hit else 0.0
    out[11] = 1.0 if _TF_SW_VER_RE.search(nm) else 0.0
    out[12] = 1.0 if n_cbz > 0 else 0.0
    out[13] = 1.0 if n_ebk > 0 else 0.0
    out[14] = 1.0 if (n_pdf > 0 and n_vid == 0 and n_ebk == 0 and n_cbz == 0 and n_other_non_meta == 0) else 0.0
    out[15] = float((n_img + n_cbz) / tot_c)
    out[16] = 1.0 if nsfw_hit else 0.0
    out[17] = 1.0 if ab_tree_hit else 0.0
    return out


OPAQUE_DIM = 6
_OPQ_HEX = frozenset("0123456789abcdef")
_OPQ_HEX_EXTS = frozenset({"zip", "rar", "7z", "tar", "gz", "bz2", "zst", "txt", "bin", "dat", "gen", "sql", "", "mp4", "mkv", "avi", "mov", "wmv", "webm", "flv", "ts", "m4v"})
_OPQ_DUMP_SUFS = (".warc", ".warc.gz", ".xml.bz2", ".sql.gz", ".jsonl.gz", ".tar.zst", ".csv.gz", ".parquet")
_OPQ_CHUNK_RE = re.compile(r"(?i)(?:live[\s._-]*archive.*chunk|split[\s._-]*archive|dataset[\s._-]*dump|wiki[\s._-]*dump|crawl[\s._-]*dump|^chunk[\s._-]*\d+)")
_OPQ_IA_SUFS = (".ia.mp4", "_meta.xml", "_files.xml", "_archive.torrent")
_OPQ_ALPHA_RE = re.compile(r"[a-z]{3,}", re.I)


def extract_opaque_6d(name: str, files: list[Any] | None = None) -> np.ndarray:
    """6-dim opaque/unknown structural signals: [is_strict_hex, is_dump, is_chunk, alpha_wc_norm, max_alpha_len_norm, hex_ratio]."""
    out = np.zeros(OPAQUE_DIM, dtype=np.float32)
    nm = str(name or "").strip()
    nl = nm.lower()
    fl = files or []
    n_files = len(fl)
    has_ia = any(nl.endswith(suf) for suf in _OPQ_IA_SUFS)
    if not has_ia:
        for f in fl[:12]:
            p = str(f.get("path") or f.get("p") or f.get("name") or "") if isinstance(f, dict) else str(f or "")
            if p.lower().endswith(_OPQ_IA_SUFS):
                has_ia = True
                break
    stem = nl.rsplit(".", 1)[0] if "." in nl else nl
    ext = nl.rsplit(".", 1)[-1] if "." in nl else ""
    is_hex = False
    if (not has_ia) and ext in _OPQ_HEX_EXTS and 10 <= len(stem) <= 96 and all(ch in _OPQ_HEX for ch in stem):
        if n_files <= 1:
            is_hex = True
        elif n_files <= 4:
            all_h = True
            for f in fl:
                p = str(f.get("path") or f.get("p") or f.get("name") or "") if isinstance(f, dict) else str(f or "")
                b = p.lower().replace("\\", "/").rsplit("/", 1)[-1]
                fs = b.rsplit(".", 1)[0] if "." in b else b
                if not (10 <= len(fs) <= 96 and all(ch in _OPQ_HEX for ch in fs)):
                    all_h = False
                    break
            is_hex = all_h
    is_dump = False
    if not has_ia:
        if nl.endswith(_OPQ_DUMP_SUFS):
            is_dump = True
        else:
            for f in fl[:16]:
                p = str(f.get("path") or f.get("p") or f.get("name") or "") if isinstance(f, dict) else str(f or "")
                if p.lower().endswith(_OPQ_DUMP_SUFS):
                    is_dump = True
                    break
    is_chunk = bool(_OPQ_CHUNK_RE.search(nm))
    aw = _OPQ_ALPHA_RE.findall(nl)
    out[0] = 1.0 if is_hex else 0.0
    out[1] = 1.0 if is_dump else 0.0
    out[2] = 1.0 if is_chunk else 0.0
    out[3] = min(len(aw), 10) / 10.0
    out[4] = min(max((len(w) for w in aw), default=0), 16) / 16.0
    alnum = [ch for ch in nl if ch.isalnum()]
    out[5] = (sum(1 for ch in alnum if ch in _OPQ_HEX) / len(alnum)) if alnum else 0.0
    return out
