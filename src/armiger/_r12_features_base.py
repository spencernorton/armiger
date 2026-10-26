"""Standalone Armiger v2.1 / Armiger Ultra ONNX Featurizer & Inference Runtime (`armiger_inference.py`).

Implements exact feature parity with the Armiger v2.1 training pipeline (`sift/prepare.py` +
`enrich_side21_v2` 36-dim `stats` vector + `vocab.json` 4,096-extension lookup + CRC32
digit-folded path-word hashing) with zero external codebase dependencies beyond `numpy`
and `onnxruntime`.

All release-group and publisher token tables are stored as 8-byte BLAKE2b-64 hex digests
so zero copyrighted or scene group strings appear in plaintext.

R12 Session 2: derivative-group file-tree features
(`U_deriv_groups`, `E_episode_keys`, `P_multipart_idx`, `s_max`) are computed by
`compute_derivative_group_features` and appended to the side vector at
`side[24..27]`; negated subtitle clauses are stripped before the `side[13]`
subtitle cue; opaque top-level names gain a `|FILES: ...` primary-media suffix
in `name_bytes`. Legacy `side[0..23]` semantics are unchanged (only previously
dead `side[13]` becomes live), so `stats[:36]` remains the legacy R10/R11
prefix. Training (R12 Sessions 5 & 7) and inference must import the shared
`SIDE_DIM` / `STATS_DIM` constants below instead of hard-coding `24` / `36`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import zlib

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

SPAN_BIO_CLASSES: Tuple[str, ...] = (
    "O",
    "B-TITLE",
    "I-TITLE",
    "B-YEAR",
    "I-YEAR",
    "B-EP",
    "I-EP",
)

# Side/stats dimensionality contract (R12 Session 2). `stats = base_stats + side`.
BASE_STATS_DIM: int = 12
LEGACY_SIDE_DIM: int = 24
SIDE_DIM: int = 28  # 24 legacy slots + 4 derivative-group features at side[24..27]
N_SIDE: int = SIDE_DIM
STATS_DIM: int = BASE_STATS_DIM + SIDE_DIM  # 40 (legacy R10/R11 width was 36)
LEGACY_STATS_DIM: int = BASE_STATS_DIM + LEGACY_SIDE_DIM  # 36
SIDE_FEATURE_NAMES: Tuple[str, ...] = (
    "ebook_share", "pdf_share", "m4b_share", "comic_share", "console_share",
    "app_share", "exec_share", "iso_share", "archive_share", "html_share",
    "script_ext_present", "trap_flag", "small_video_flag", "subtitle_cue",
    "sub_only_group", "anime_keyword", "dub_keyword", "game_group",
    "software_group", "course_keyword", "book_comic_keyword",
    "lang_english", "lang_non_english", "lang_unknown",
    "deriv_u_log", "deriv_e_log", "deriv_p_multipart", "deriv_s_max",
)

NEURAL_HEAD_CLASSES: Dict[str, List[str]] = {
    "category": [
        "movie", "tv_show", "xxx", "music", "comic",
        "game", "audiobook", "software", "ebook", "course", "unknown",
    ],
    "adult": ["non_adult", "adult"],
    "junk": ["clean", "junk"],
    "anime": ["non_anime", "anime"],
    "english_subs": ["no_eng_subs", "has_eng_subs", "unknown"],
    "lang": ["english", "non_english", "multi_or_unknown"],
    "owner_scope": ["allowed_type", "excluded"],
    "title_language": ["english_title", "non_english_title"],
    "span_bio": list(SPAN_BIO_CLASSES),
}

_VOCAB_EXT_MAP: Optional[Dict[str, int]] = None
_PREP_KIND_EXTS = {
    0: frozenset("mkv mp4 avi ts m2ts mts vob wmv mpg mpeg mov flv rmvb rm m4v divx webm 3gp asf f4v ogv mxf m2v bik".split()),
    1: frozenset("mp3 flac m4a m4b aac ogg opus wav wma ape alac aif aiff aifc ac3 dts mka mpc wv shn mid midi dff dsf".split()),
    2: frozenset("zip rar 7z gz tgz bz2 xz zst lz lzma tar cab iso img dmg mdf mds nrg wim par par2 z".split()),
    3: frozenset("exe msi apk ipa dll bat cmd sh vbs ps1 com scr jar deb rpm appimage run lnk".split()),
    4: frozenset("jpg jpeg png gif webp bmp tif tiff avif jxl heic heif psd ai svg ico tga cr2 nef arw dng raw eps".split()),
    5: frozenset("pdf epub mobi azw azw3 djvu djv fb2 txt rtf doc docx odt xls xlsx ppt pptx csv chm cbz cbr cb7 html htm mht mhtml md tex lit".split()),
    6: frozenset("srt ass ssa sub idx vtt sup smi sbv ttml dfxp usf scc stl mks".split()),
}
_PREP_EXT_TO_KIND = {e: k for k, es in _PREP_KIND_EXTS.items() for e in es}
_PREP_SPLIT_ARC_RE = re.compile(r"[rsz]\d{2,3}|\d{3}")
_PREP_WORD_RE = re.compile(r"[^\W_]+")
_PREP_DIGIT_RE = re.compile(r"\d")
_PREP_SAMPLE_RE = re.compile(r"(?<![a-z])samples?(?![a-z])", re.IGNORECASE)

# BLAKE2b-64 digest sets for release-tag disambiguation (zero plaintext group names)
_SUB_ONLY_GRP_H = frozenset(['00d47045a1f1cd70', '0380c309e464115a', '0665f92d474878ff', '0f8f3809c6074d35', '1bc080c5d14b4eae', '1bf62e9b91086379', '22325081668ed28b', '26503f6826942362', '33b21e80d5dbfa62', '3e370e64b7539495', '4297edae7e25ba02', '4491be5b41a66832', '44e08227cfda9421', '4b956de9c462ee70', '519230ab4d0911f7', '581a8006ea8bd801', '65dfe6849ae4bb0d', '66ced9df20d8f558', '6a437c446b5778d6', '6cff013c3e752ac8', '6d3b6a62beedc878', '7bc52c5f863149cc', '867c636203d0524d', 'a7637cb5c531d8e9', 'b61d75b73d8e0c59', 'bed3322685febfa5', 'c603a601b7cb9d4c', 'd4eaba2bc7ef10d6', 'e20a8d50ddf1aa2e', 'e3fd326207da261c', 'fc94bcdf9fcb0164'])
_DUAL_DUB_GRP_H = frozenset(['16bdcdbbbd493d9e', '17addcc669201a7b', '1cb486cd76cea8c6', '1cc2f6b1aa24d6f7', '236a71bfa2d47ca3', '24eee113ee4bddb0', '341556a463a4784b', '3658912d4f05f762', '6149de24f7784f27', '627dcb38527b8e4c', '6f09d409143a679c', '727c5e1f0f05ad51', '8b7c320e1df4bf3e', '9e64e0c8c213bf00', 'ac3675ff37aaf11e', 'b2c26515563660f5', 'bd672057ff15b985', 'd9621931df7eb664', 'e20db1736162ba0c', 'e2ee075c6b5545ad', 'eb446df9e0b5f645', 'f7a175e4cf1907fb', 'fe0fe2b14bb48506'])
_GAME_GRP_H = frozenset(['03f22fe2a782da12', '0ae36a7939ac0be1', '1217fb0deda20335', '1c8f3166691fa2c6', '306372c75c8ceaa2', '32ba70bb6b5ae022', '4509f87fe97d6ed9', '4afe068be7dcf016', '6f7a14ce24f000a0', '7403ff2453c8a173', '7de6d237db23b5fa', '9641051915abdbb6', 'a491e84e39e2064e', 'a65a1816debe0915', 'aa89d2a9d5d18f01', 'aaac60aa2d75341b', 'bb98906accdd6e05', 'c200372b38fb6d26', 'c430ddffd526c993', 'cc81415c4d797870', 'e37a0da20d8d24ac', 'ed5d3b0f00026714', 'ee7b45399bcf968a', 'f0982ea7c2028c50', 'f1737c198253be9d', 'f766dbb74e38db04', 'fbe41c3bbd33ac19'])
_SW_GRP_H = frozenset(['09455d377a262608', '0e30392b37df5495', '103064b4fea08f1c', '11a120607b793e4c', '158ba6de7392b368', '193a01c7e1bd0f62', '2199b8b3ff553486', '240843438078b817', '2508ee34b17c2225', '2a4eb6e114be0fab', '352836dac99f04b7', '391bd8a58914224a', '4185301675b9d80c', '4472c2744066912a', '4c4684b615a127b2', '522a470219d3a4f1', '5a6bb90382b6c1f6', '5ca7613c608a9ea5', '63f15e4687bfc135', '66a7555e70f0872e', '73d877e885d34562', '77c343b865d05e31', '780409423e866d09', '7e0a43f55a6714fd', '97409999bccb98f4', '996bd865c828c8f9', '9b60cdfff0e2ab1c', 'a330b94802e4b6c6', 'a9c5a9029bc6a596', 'b48e9326bf6316ef', 'c595d125bfd5ebe9', 'c8575232dac0347d', 'd64fa92611ac0eb2', 'e182e241e26cb119', 'e88087f130a6e7d6'])
_BOOK_COMIC_GRP_H = frozenset(['0cad54552908fa65', '338c3fad71b18311', '5913147687b3c1e5', '712f82b4b85131a0', '7f5fc23af41380b2', '8d8b63a8bb00b7e4', '913bd80742a62e95', 'c00722c39b756d92', 'ecfebbed5476af20', 'ee2807f40b38f779', 'f89c7ccfb7b1338c', 'fdffac028227e18e', 'ffb2c2ead4c7ba7e'])
_BENIGN_README_EXE_H = frozenset(['062b96e72d602b6e', '9a9f15179701c6ae', 'ddf05591355cf7e0'])

_TOKEN_SCAN_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_SIDE_ANIME_KW_RE = re.compile(
    r"(?i)\b(?:anime|ova|ona|donghua|aeni|bdrip[ ._-]*1080p[ ._-]*flac|dual[ ._-]*audio[ ._-]*\[?(?:jpn|jap))\b"
)
_SIDE_VID_CLAIM_RE = re.compile(
    r"(?i)\b(?:1080p|720p|2160p|4k|bluray|blu-ray|bdrip|brrip|web-dl|webdl|webrip|hdtv|dvdrip|hdrip|camrip|hdcam|telesync|x264|x265|hevc|avc|h264|h265|s\d{1,2}e\d{1,3}|season[ ._-]*\d+|fc2-?ppv)\b|高清首发|更新至第"
)
_SIDE_DECEPTIVE_SUB_STRIP_RE = re.compile(
    r"(?i)\b(?:eng(?:lish)?|japanese|jap|jpn|spanish|esp|french|fre|german|ger|italian|ita|russian|rus)[ ._-]+(?:hard|soft)?sub(?:s|bed|titled|titles)?\b|"
    r"\b(?:hard|soft)?sub(?:s|bed|titled|titles)?[ ._-]+(?:eng|english|esp|ja|jpn|fr|de|it|ru)\b|"
    r"\b(?:m[ ._-]?subs?|multi[ ._-]?subs?|esubs?|vostfr|hardsubs?)\b"
)
_SIDE_ANIME_DUB_KW_RE = re.compile(
    r"(?i)\b(?:dual[ ._-]?audio|dubbed|eng(?:lish)?[ ._-]?dub(?:bed)?|multi[ ._-]?audio)\b"
)
_SIDE_SW_APP_TITLE_RE = re.compile(
    r"(?i)\b(?:v?\d+\.\d+\.\d+|repack|portable|cracked|keygen|patch|patcher|activat(?:ed|or)|"
    r"multilingual|win(?:32|64)|x64|x86|macos|macosx|android|apk|pc[ ._-]?game|nintendo[ ._-]?switch|ps4|ps5|"
    r"emulator|driver|firmware|setup|installer|downloader|studio|vst3?|"
    r"pro[ ._-]+v?\d|ce[ ._-]+v?\d)\b"
)
_SIDE_COURSE_RE = re.compile(
    r"(?i)\b(?:complete[ ._-]+course|video[ ._-]+course|masterclass|bootcamp|tutorial[ ._-]+series|certification[ ._-]+course|hands-on[ ._-]+training)\b"
)
_SIDE_BOOK_COMIC_KW_RE = re.compile(
    r"(?i)\b(?:audiobooks?|audible|m4b|unabridged|narrated[ ._-]?by|ebook|e-book|epub|mobi|azw3|calibre)\b"
)
_SIDE_VPN_TRAP_RE = re.compile(
    r"(?i)免费安装下载vpn|打不开网页|codec[ ._-]?(?:pack|setup|install)|player[ ._-]?setup|instructions?[ ._-]?to[ ._-]?watch"
)
_SIDE_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SIDE_EP_ZIP_RE = re.compile(
    r"(?i)(?:^\[[^\[\]]{2,30}\].*?(?:\s-\s*\d{1,3}\b|\bS\d{1,2}E\d{1,3}\b).*?\b(?:1080p|720p|2160p|480p)\b|"
    r"\bS\d{1,2}E\d{1,3}\b.*?\b(?:1080p|720p|2160p|480p|WEB-DL|WEBRip|BluRay|BDRip|HDTV|x264|x265|HEVC)\b|"
    r"\b(?:1080p|720p|2160p|480p|576p|4k)\b.*?\b(?:WEB-DL|WEBRip|BluRay|BDRip|BRRip|HDRip|HDTV|HDCAM|CAMRip|DVDRip|x264|x265|H\.?264|H\.?265|HEVC|AV1|XviD|DivX)\b|"
    r"\b(?:WEB-DL|WEBRip|BDRip|BRRip|HDRip|HDTV|HDCAM|CAMRip|DVDRip)\b(?:.*?\b(?:x264|x265|H\.?264|H\.?265|HEVC|AV1|XviD|DivX))?)"
)
_SIDE_BENIGN_ZIP_RE = re.compile(
    r"(?i)\b(?:batch|complete|season[ ._-]*\d+|s\d{1,2}[ ._-]*complete|e?\d{1,3}[ ._-]*[~-][ ._-]*e?\d{1,3}|시즌\s*\d+|완결|全集|合集|"
    r"ost|soundtrack|artbook|scans?|wallpapers?|fonts?|subs?|subtitles?)\b"
)


# --- R12 Session 2: derivative-group file-tree features ---------------------
# Brief-mandated video set (.mkv .mp4 .avi .ogv .mpg .mpeg .m4v .ts .wmv .mov
# .webm .vob .iso) plus same-family containers seen in the gold corpus
# (.m2ts/.mts/.divx/.flv/.rmvb/.3gp/.asf) so primary videos are never missed.
_DERIV_VIDEO_EXTS = frozenset(
    "mkv mp4 avi ogv mpg mpeg m4v ts wmv mov webm vob iso m2ts mts divx flv rmvb 3gp asf".split()
)
_DERIV_SAMPLE_RE = re.compile(r"(?i)\bsample\b")
# IA/transcode rendition suffixes stripped repeatedly from a video stem.
_DERIV_SUFFIX_RE = re.compile(
    r"(?i)(?:\.(?:ia|512kb|h264|h\.264|x264|hd|sd)|_(?:archive|thumbs|files|512kb|h264|ia))+$"
)
_EP_SXE_RE = re.compile(r"(?i)(?<![a-z0-9])s(\d{1,2})[\s._\-]*e(\d{1,3})(?![0-9])")
_EP_NXN_RE = re.compile(r"(?<![0-9a-zA-Z])(\d{1,2})x(\d{2})(?![0-9a-zA-Z])", re.IGNORECASE)
_EP_WORD_RE = re.compile(r"(?i)\b(?:episode)[\s._\-]*(\d{1,3})(?![0-9])")
_EP_SHORT_RE = re.compile(r"(?i)(?<![a-z0-9])ep\.?\s*(\d{1,3})(?![0-9])")
_EP_BARE_E_RE = re.compile(r"(?i)(?<![a-z0-9])e(\d{1,3})(?![a-z0-9])")
# Anime fansub dash numbering (`[Group] Show - 01v2 [CRC].mkv`), train-validated in WS-1.
_EP_DASH_NUM_RE = re.compile(r"\s-\s(\d{1,3})(?:v\d)?(?![0-9a-zA-Z])")
_SEASON_HINT_RE = re.compile(
    r"(?i)(?:(?<![a-z0-9])s(\d{1,2})(?![0-9])|\bseason[\s._\-]*(\d{1,2})(?![0-9]))"
)
_MULTIPART_TOKEN_RE = re.compile(
    r"(?i)\b(?:cd|disc|disk|pt|part|side)[\s._\-]*([1-4])\b"
)
_VTS_STEM_RE = re.compile(r"(?i)^vts_\d+_\d+$")
_PILOT_RE = re.compile(r"(?i)\b(?:unaired[\s._\-]*)?pilot\b")
_YEAR_TOKEN_RE = re.compile(r"(?<![0-9])(?:19[2-9]\d|20[0-2]\d)(?![0-9])")
# Negated subtitle clauses are removed before the side[13] subtitle cue so
# "no english subs" / "without eng subs" / "non-english subtitles" cannot fire it.
_NEGATED_SUB_RE = re.compile(
    r"(?i)\b(?:no|without|non|none|pas\s+de|kein[e]?|sin)[\s._\-]+"
    r"(?:[-\w]+[\s._\-]+){0,2}?(?:eng(?:lish)?[\s._\-]+)?"
    r"(?:sub(?:title)?s?|srt|ass|ssa|cc)\b"
    r"|\bsub(?:title)?s?[\s:_-]*(?:no|none|n/a|nil|false)\b"
)
_SUB_CUE_RE = re.compile(
    r"(?i)(?:\b(?:eng(?:lish)?|en|hard|soft|multi)[\s._\-]*"
    r"(?:sub(?:title)?s?|srt|ass|ssa|cc)\b"
    r"|\b(?:subtitles?|subs|subbed|esubs?|closed[\s._\-]*captions?)\b"
    r"|\.(?:srt|ass|ssa)\b)"
)
_MEDIA_SUMMARY_EXTS = frozenset(
    set(_DERIV_VIDEO_EXTS)
    | set("mp3 flac m4b m4a ogg opus epub mobi azw azw3 pdf cbz cbr cb7 "
          "exe msi apk dmg nsp xci xcz nsz".split())
)
_DERIV_LOG_NORM: float = math.log1p(64.0)


def _normalize_file_entries(files: Any) -> List[Tuple[str, int]]:
    """Normalize torrent `files` (dicts / (path, size) tuples / strings) to `(path, size)` pairs."""
    out: List[Tuple[str, int]] = []
    for f in files or []:
        if isinstance(f, Mapping):
            p = str(f.get("path") or f.get("p") or f.get("name") or "")
            try:
                sz = int(f.get("size") or f.get("s") or 0)
            except (TypeError, ValueError, OverflowError):
                sz = 0
        elif isinstance(f, (tuple, list)) and len(f) >= 1:
            p = str(f[0] or "")
            try:
                sz = int(f[1]) if len(f) > 1 else 0
            except (TypeError, ValueError, OverflowError):
                sz = 0
        else:
            p, sz = str(f or ""), 0
        if p and "/.____padding_file/" not in p and not p.startswith(".____padding_file"):
            out.append((p, max(sz, 0)))
    return out


def _deriv_stem_key(parent: str, stem: str) -> Tuple[str, str]:
    """Collapse transcode renditions of one work to a single derivative-group key."""
    s = stem
    for _ in range(4):
        stripped = _DERIV_SUFFIX_RE.sub("", s)
        if stripped == s:
            break
        s = stripped
    norm_s = re.sub(r"[\W_]+", " ", s.lower(), flags=re.UNICODE).strip() or s.lower().strip()
    norm_p = re.sub(r"[\W_]+", " ", parent.lower(), flags=re.UNICODE).strip()
    return norm_p, norm_s


def _episode_keys_from_text(text: str, default_season: Optional[int] = None) -> set:
    """Distinct episode keys in one path/name string (multipart CD/Part tokens never count)."""
    keys: set = set()
    if not text:
        return keys
    seasons = {
        int(g) for m in _SEASON_HINT_RE.finditer(text)
        for g in m.groups() if g is not None
    }
    for m_sxe in _EP_SXE_RE.finditer(text):
        seasons.add(int(m_sxe.group(1)))
    for m_nxn in _EP_NXN_RE.finditer(text):
        seasons.add(int(m_nxn.group(1)))
    season = seasons.pop() if len(seasons) == 1 else (default_season if not seasons else None)

    def add_bare(num: int) -> None:
        keys.add(f"s{season:02d}e{num:03d}" if season is not None else f"ep{num:04d}")

    for m in _EP_SXE_RE.finditer(text):
        keys.add(f"s{int(m.group(1)):02d}e{int(m.group(2)):03d}")
    for m in _EP_NXN_RE.finditer(text):
        keys.add(f"s{int(m.group(1)):02d}e{int(m.group(2)):03d}")
    for m in _EP_WORD_RE.finditer(text):
        add_bare(int(m.group(1)))
    for m in _EP_SHORT_RE.finditer(text):
        add_bare(int(m.group(1)))
    # Bare `E01` stems (brief's `Ep?` with optional `p`), e.g. `Show.S01/E01.mkv`.
    # The lookbehind already excludes the `E` inside `S01E01` (digit before it).
    for m in _EP_BARE_E_RE.finditer(text):
        add_bare(int(m.group(1)))
    for m in _EP_DASH_NUM_RE.finditer(text):
        add_bare(int(m.group(1)))
    if not keys and _PILOT_RE.search(text):
        keys.add("pilot")
    return keys


def compute_derivative_group_features(
    files: Any,
    raw_name: str = "",
) -> Tuple[float, float, float, float]:
    """Compute `(U_deriv_groups, E_episode_keys, P_multipart_idx, s_max)` for a torrent file tree.

    - `U`: unique normalized video derivative stems; IA transcodes of one stem
      (`.mp4`/`.ogv`/`.ia.mp4`/`.512kb` ...) collapse to one group and all
      `VIDEO_TS/VTS_*` files collapse to one group.
    - `E`: distinct episode keys (`SxxExx`, `NxNN`, `Ep N`, `Episode N`, bare
      `ENN` under a season folder). Pure multipart splits (`CD1/CD2`,
      `Part 1/2` of one film) contribute `0`; a two-part TV pilot carrying an
      explicit `SxxExx`/`Pilot`/season marker counts its parts (`E >= 2`).
    - `P`: `1.0` for a multi-disc/part movie split or `VIDEO_TS` with `E <= 1`.
    - `s_max`: largest deduplicated primary stem share, using the largest file
      per derivative stem so IA renditions cannot depress it. Sample files
      (`sample` token, or `< 15 MB` when total video exceeds `100 MB`) are ignored.
    """
    entries = _normalize_file_entries(files)
    videos: List[Tuple[str, str, str, int]] = []  # (path, parent, stem, size)
    for path, sz in entries:
        p_norm = path.replace("\\", "/")
        parent, _, base = p_norm.rpartition("/")
        if "." not in base:
            continue
        stem, ext = base.rsplit(".", 1)
        if ext.lower() in _DERIV_VIDEO_EXTS:
            videos.append((p_norm, parent, stem, sz))

    total_video = sum(sz for _, _, _, sz in videos)
    kept = [
        v for v in videos
        if not _DERIV_SAMPLE_RE.search(v[0].rsplit("/", 1)[-1])
        and not (0 < v[3] < 15_000_000 and total_video > 100_000_000)
    ]
    if kept:
        videos = kept

    rep_bytes: Dict[Tuple[str, str], int] = {}
    ep_keys: set = set()
    part_indices: set = set()
    has_video_ts = False
    tv_marker = bool(_EP_SXE_RE.search(raw_name or "") or _PILOT_RE.search(raw_name or "")
                     or _SEASON_HINT_RE.search(raw_name or ""))
    title_seasons = {
        int(g) for m in _SEASON_HINT_RE.finditer(raw_name or "")
        for g in m.groups() if g is not None
    } | {int(m.group(1)) for m in _EP_SXE_RE.finditer(raw_name or "")} | {int(m.group(1)) for m in _EP_NXN_RE.finditer(raw_name or "")}
    title_default_season = next(iter(title_seasons)) if len(title_seasons) == 1 else None
    ep_keys.update(_episode_keys_from_text(raw_name or "", default_season=title_default_season))

    for p_norm, parent, stem, sz in videos:
        ep_keys.update(_episode_keys_from_text(p_norm, default_season=title_default_season))
        part_indices.update(int(m.group(1)) for m in _MULTIPART_TOKEN_RE.finditer(p_norm))
        if _EP_SXE_RE.search(p_norm) or _PILOT_RE.search(p_norm) or _SEASON_HINT_RE.search(p_norm):
            tv_marker = True
        if "/video_ts/" in f"/{p_norm.lower()}" or _VTS_STEM_RE.match(stem):
            has_video_ts = True
            key = ("__video_ts__", "")
        else:
            key = _deriv_stem_key(parent, stem)
        rep_bytes[key] = max(rep_bytes.get(key, 0), sz)

    e_count = len(ep_keys)
    # Two-part TV pilot: explicit TV marker + distinct parts of one episode.
    if tv_marker and len(part_indices) >= 2 and e_count <= 1:
        e_count = len(part_indices)

    u_count = len(rep_bytes)
    p_idx = 1.0 if (has_video_ts or len(part_indices) >= 2) and e_count <= 1 else 0.0
    rep_total = sum(rep_bytes.values())
    if u_count == 0:
        s_max = 0.0
    elif rep_total > 0:
        s_max = max(rep_bytes.values()) / rep_total
    else:  # sizes unknown: equal-share assumption over derivative groups
        s_max = 1.0 / u_count
    return float(u_count), float(e_count), float(p_idx), float(s_max)


def primary_media_file_summary(files: Any, top_n: int = 3) -> str:
    """Top-`top_n` largest primary media relative paths, ` ; `-joined (largest first)."""
    media = [
        (p, sz) for p, sz in _normalize_file_entries(files)
        if "." in p.rsplit("/", 1)[-1]
        and p.rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower() in _MEDIA_SUMMARY_EXTS
    ]
    media.sort(key=lambda x: -x[1])
    return " ; ".join(p for p, _ in media[:top_n])


def build_enriched_name_text(name: str, files: Any, max_bytes: int = 256) -> str:
    """Append |FILES: <top-3 media basenames> when name is opaque or hides multi-episode files."""
    summary = primary_media_file_summary(files, top_n=3)
    if not summary:
        return name
    _, e_count, _, _ = compute_derivative_group_features(files, name)
    name_has_ep = bool(_episode_keys_from_text(name))
    is_opaque_slug = not bool(re.search(r"[A-Za-z]{3,}", name))
    hidden_ep_pack = (e_count >= 2 and not name_has_ep)
    if not (is_opaque_slug or hidden_ep_pack):
        return name
    basenames = " ; ".join(
        p.rsplit("/", 1)[-1] for p in summary.split(" ; ")
    )
    return f"{name} |FILES: {basenames}"


def _h64(s: str) -> str:
    return hashlib.blake2b(s.strip().lower().encode("utf-8"), digest_size=8).hexdigest()


def _extract_token_hashes(text: str) -> frozenset[str]:
    toks = [m.group(0).lower() for m in _TOKEN_SCAN_RE.finditer(text)]
    out = {_h64(t) for t in toks}
    for i in range(len(toks) - 1):
        out.add(_h64(f"{toks[i]} {toks[i + 1]}"))
    return frozenset(out)


_VOCAB_CACHE_BY_PATH: Dict[str, Dict[str, int]] = {}


def _get_vocab_ext_map(vocab_path: Optional[Path] = None) -> Dict[str, int]:
    global _VOCAB_EXT_MAP
    if vocab_path is not None:
        vp = Path(vocab_path)
        if vp.exists():
            key = str(vp.resolve())
            if key not in _VOCAB_CACHE_BY_PATH:
                data = json.loads(vp.read_text(encoding="utf-8"))
                _VOCAB_CACHE_BY_PATH[key] = {e: i + 1 for i, e in enumerate(data.get("exts", []))}
            _VOCAB_EXT_MAP = _VOCAB_CACHE_BY_PATH[key]
            return _VOCAB_EXT_MAP
    if vocab_path is None and _VOCAB_EXT_MAP is not None and len(_VOCAB_EXT_MAP) > 64:
        return _VOCAB_EXT_MAP
    vpath = Path(__file__).resolve().with_name("vocab.json")
    if not vpath.exists() and os.environ.get("ARMIGER_MODEL_DIR"):
        vpath = Path(os.environ["ARMIGER_MODEL_DIR"]) / "vocab.json"
    if vpath.exists():
        key = str(vpath.resolve())
        if key not in _VOCAB_CACHE_BY_PATH:
            data = json.loads(vpath.read_text(encoding="utf-8"))
            _VOCAB_CACHE_BY_PATH[key] = {e: i + 1 for i, e in enumerate(data.get("exts", []))}
        _VOCAB_EXT_MAP = _VOCAB_CACHE_BY_PATH[key]
    else:
        _VOCAB_EXT_MAP = {
            "mp3": 2, "mp4": 3, "mkv": 4, "avi": 7, "srt": 10, "zip": 12,
            "rar": 17, "exe": 23, "flac": 24, "png": 25, "dll": 31, "m4b": 53,
            "epub": 81, "cbz": 96, "lnk": 150, "scr": 310,
        }
    return _VOCAB_EXT_MAP


def _hash_ext(ext: str) -> int:
    if not ext:
        return 4096
    return _get_vocab_ext_map().get(ext.lower(), 4096)


def _hash_word(w: str) -> int:
    if not w:
        return 0
    folded = _PREP_DIGIT_RE.sub("0", w.lower())
    return (zlib.crc32(folded.encode("utf-8")) % 32767) + 1


def _log2_size_bucket(sz: int) -> int:
    return min(47, max(int(sz), 1).bit_length() - 1)


def _prep_split_path(path: str) -> Tuple[str, str, str, int]:
    p_norm = path.replace("\\", "/")
    parent, _, base = p_norm.rpartition("/")
    i = base.rfind(".")
    ext = base[i + 1:].lower() if i > 0 else ""
    if len(ext) <= 10 and ext.isascii() and ext.isalnum():
        stem = base[:i]
    else:
        ext, stem = "", base
    return parent.rpartition("/")[2], stem, ext, p_norm.count("/")


def _prep_path_words(parent: str, stem: str) -> List[int]:
    p_words = [_hash_word(w) for w in _PREP_WORD_RE.findall(parent.lower())][-2:] if parent else []
    s_words = [_hash_word(w) for w in _PREP_WORD_RE.findall(stem.lower())][-(4 - len(p_words)):]
    res = p_words + s_words
    while len(res) < 4:
        res.append(0)
    return res[:4]


def encode_torrent_features(
    row: Mapping[str, Any],
    max_name_len: int = 256,
    max_files: int = 64,
) -> Dict[str, Any]:
    """Encode a torrent metadata dictionary into the exact ONNX tensor inputs for Armiger v2.1."""
    name = str(row.get("name") or row.get("title") or "")

    raw_files = row.get("files") or []
    try:
        total_size = max(0, int(row.get("size") or 0))
    except (TypeError, ValueError, OverflowError):
        total_size = 0
    all_items: List[Tuple[str, int]] = _normalize_file_entries(raw_files)
    all_items.sort(key=lambda x: -x[1])
    items: List[Tuple[str, int]] = all_items[:max_files]
    if not items and (row.get("ext") or row.get("extension") or "." in name):
        items.append((name, total_size))

    # R12: opaque/folder names expose their primary media filenames to the
    # byte/char/word encoders; derivative-group features use the full tree.
    name_text = build_enriched_name_text(name, all_items or items, max_bytes=max_name_len)
    b_name = name_text.encode("utf-8", "replace")
    if len(b_name) > max_name_len:
        b_name = b_name[:max_name_len].decode("utf-8", "ignore").encode("utf-8")
    raw_bytes = list(b_name) if b_name else [32]
    deriv_features = compute_derivative_group_features(all_items or items, name)

    ext_map = _get_vocab_ext_map()
    exts: List[int] = []
    sizes: List[int] = []
    depths: List[int] = []
    words_list: List[List[int]] = []
    f_exts_str: List[str] = []

    kbytes = [0.0] * 8
    kcount = [0] * 8
    has_sample = False

    for path, sz in items:
        parent, stem, ext, depth = _prep_split_path(path)
        f_exts_str.append(ext)
        exts.append(ext_map.get(ext, 4096))
        sizes.append(_log2_size_bucket(sz))
        depths.append(min(depth, 7))
        words_list.append(_prep_path_words(parent, stem))
        k = _PREP_EXT_TO_KIND.get(ext)
        if k is None:
            k = 2 if _PREP_SPLIT_ARC_RE.fullmatch(ext) else 7
        kbytes[k] += max(sz, 0)
        kcount[k] += 1
        has_sample = has_sample or (_PREP_SAMPLE_RE.search(path) is not None)

    tot_k = sum(kbytes)
    weights = kbytes if tot_k > 0 else [float(c) for c in kcount]
    wsum = sum(weights)
    try:
        raw_fc = max(0, int(row.get("file_count") or 0))
    except (TypeError, ValueError, OverflowError):
        raw_fc = 0
    count = max(raw_fc, len(all_items), len(items))
    biggest = max((max(s, 0) for _, s in items), default=0)
    max_share = (biggest / tot_k if tot_k > 0 else (1.0 / len(items) if items else 0.0))

    base_stats = [
        math.log1p(max(count, 0)),
        math.log1p(max(total_size if total_size > 0 else int(tot_k), 0)),
    ]
    base_stats += [w / wsum if wsum > 0 else 0.0 for w in weights]
    base_stats += [max_share, float(has_sample)]

    side = [0.0] * SIDE_DIM
    n_f = float(max(1, len(items)))

    m_grp = re.match(r"^\[([^\[\]]{2,30})\]", name)
    grp_lc = m_grp.group(1).strip().lower() if m_grp else ""
    grp_h = _h64(grp_lc) if grp_lc else ""
    title_tok_h = _extract_token_hashes(name)

    has_sub_grp = (grp_h in _SUB_ONLY_GRP_H) or bool(title_tok_h & _SUB_ONLY_GRP_H)
    has_dub_grp = (grp_h in _DUAL_DUB_GRP_H) or bool(title_tok_h & _DUAL_DUB_GRP_H)
    has_game_grp = bool(title_tok_h & _GAME_GRP_H)
    has_sw_grp = bool(title_tok_h & _SW_GRP_H)
    has_book_grp = bool(title_tok_h & _BOOK_COMIC_GRP_H)

    if items:
        ebk_set = {"epub", "mobi", "azw3", "azw", "djvu", "fb2", "lit", "iba"}
        pdf_set = {"pdf"}
        m4b_set = {"m4b", "aa", "aax"}
        com_set = {"cbz", "cbr", "cb7", "cbt"}
        con_set = {"nsp", "xci", "xcz", "nsz", "wbfs", "rvz", "gcz", "cia", "3ds", "vpk", "ps4", "fpkg"}
        app_set = {"dmg", "pkg", "apk", "ipa", "deb", "rpm", "appimage"}
        exe_dll_set = {"exe", "msi", "bat", "cmd", "sh", "com", "dll"}
        iso_set = {"iso", "bin", "cue", "mdf", "mds", "nrg", "img"}
        arc_set = {"rar", "zip", "7z", "tar", "gz", "xz", "bz2", "001"}
        htm_set = {"html", "htm", "url", "mht", "torrent"}
        scr_set = {"scr", "lnk", "vbs", "hta", "pif", "ps1", "wsf", "reg"}
        vid_set = {"mkv", "mp4", "avi", "ts", "m2ts", "wmv", "mov", "webm", "flv", "mpg", "mpeg", "m4v", "vob", "ogv", "rmvb", "3gp", "asf", "divx"}
        bd_game_benign = {"vcdiff", "apk", "bdmv", "clpi", "mpls", "ifo", "bup", "vob", "npa", "vpk", "pck", "umd", "pak", "xp3", "rgss3a", "wolf", "nsp", "xci", "wbfs"}

        side[0] = sum(1 for e in f_exts_str if e in ebk_set) / n_f
        side[1] = sum(1 for e in f_exts_str if e in pdf_set) / n_f
        side[2] = sum(1 for e in f_exts_str if e in m4b_set) / n_f
        side[3] = sum(1 for e in f_exts_str if e in com_set) / n_f
        side[4] = sum(1 for e in f_exts_str if e in con_set) / n_f
        side[5] = sum(1 for e in f_exts_str if e in app_set) / n_f

        vc = bool(_SIDE_VID_CLAIM_RE.search(name) or _SIDE_ANIME_KW_RE.search(name) or has_sub_grp or has_dub_grp)
        has_any_vid_inf = any(e in vid_set for e in f_exts_str)
        has_renpy_structure = (
            not vc
            and not has_any_vid_inf
            and any(
                p_i.replace("\\", "/").lower().endswith((".rpa", ".rpyb")) and int(sz_i or 0) >= 65536
                for p_i, sz_i in items
            )
        )
        renpy_say_re = re.compile(r"(?i)(?:^|/)lib/(?:py\d+-)?windows-[^/]+/say\.vbs$")

        n_real_exec_dll = 0
        has_real_exe = False
        has_dll = False
        has_scr_trap = False
        has_oversized_lnk = False
        has_root_launcher = False
        has_unconditional_trap_ext = False
        has_any_scr_ext = False
        vid_depths: List[int] = []

        for (p_i, _), e_str, s_b, d_i, w_i in zip(items, f_exts_str, sizes, depths, words_list):
            p_posix = p_i.replace("\\", "/")
            if has_renpy_structure and renpy_say_re.search(p_posix):
                continue
            if e_str in scr_set:
                has_any_scr_ext = True
            if e_str in {"scr", "vbs", "vbe", "hta", "pif", "wsf", "wsh", "cpl"}:
                has_unconditional_trap_ext = True
            p_tok_h = _extract_token_hashes(p_i)
            is_benign_stub = (e_str == "exe") and (s_b < 15 or 16570 in w_i or bool(p_tok_h & _BENIGN_README_EXE_H))
            if e_str in exe_dll_set and not is_benign_stub:
                n_real_exec_dll += 1
            if e_str == "exe" and not is_benign_stub and s_b >= 15:
                has_real_exe = True
                if d_i == 0:
                    has_root_launcher = True
            elif e_str == "dll":
                has_dll = True
            if e_str in {"scr", "lnk", "hta", "vbs", "pif", "ps1", "wsf", "cmd", "com", "one", "xll", "xlam"} and s_b >= 12:
                has_scr_trap = True
                if d_i == 0:
                    has_root_launcher = True
                if e_str == "lnk" and s_b >= 15:
                    has_oversized_lnk = True
            if e_str in vid_set:
                vid_depths.append(d_i)

        side[6] = n_real_exec_dll / n_f
        side[7] = sum(1 for e in f_exts_str if e in iso_set) / n_f
        side[8] = sum(1 for e in f_exts_str if e in arc_set) / n_f
        side[9] = sum(1 for e in f_exts_str if e in htm_set) / n_f
        side[10] = 1.0 if has_any_scr_ext else 0.0

        is_sw_or_game = bool(
            has_game_grp
            or has_sw_grp
            or (not vc and _SIDE_SW_APP_TITLE_RE.search(name))
            or any(e in bd_game_benign for e in f_exts_str)
        )
        vid_share = base_stats[2]
        flagged = has_unconditional_trap_ext
        if not is_sw_or_game:
            if vc and has_real_exe and has_dll:
                flagged = True
            elif vc and len(vid_depths) >= 1 and (has_real_exe or has_root_launcher or has_scr_trap):
                flagged = True
            elif (vc or len(vid_depths) > 0) and has_oversized_lnk:
                flagged = True
            elif vc and vid_share == 0.0 and (has_real_exe or has_scr_trap):
                flagged = True
            elif len(items) <= 2 and any(e in ("zip", "zipx", "rar", "7z") for e in f_exts_str) and vid_share == 0.0:
                if _SIDE_EP_ZIP_RE.search(name) and not _SIDE_BENIGN_ZIP_RE.search(name):
                    flagged = True
            elif len(items) <= 2 and any(e in ("iso", "img", "vhd", "vhdx", "udf") for e in f_exts_str) and vid_share == 0.0:
                name_stem = re.sub(r"\.(?:iso|img|vhd|vhdx|udf|dmg)$", "", name.strip(), flags=re.IGNORECASE)
                if _SIDE_EP_ZIP_RE.search(name_stem) and not re.search(r"(?i)\b(?:dvd|bd25|bd50|remux|untouched|pal|ntsc|iso)\b", name_stem):
                    flagged = True

        if flagged or _SIDE_VPN_TRAP_RE.search(name) or (vc and vid_share == 0.0 and (side[6] > 0 or side[9] > 0 or side[10] > 0)):
            side[11] = 1.0
        if vc and vid_share > 0 and 0 < base_stats[1] < 14.22:
            side[12] = 1.0

    # R12: strip negated subtitle clauses ("no english subs", ...) before the
    # side[13] subtitle cue; previously side[13] was never populated at all.
    if _SUB_CUE_RE.search(_NEGATED_SUB_RE.sub(" ", name)):
        side[13] = 1.0

    ag = bool(_SIDE_ANIME_KW_RE.search(name)) or has_sub_grp or has_dub_grp
    if ag:
        side[15] = 1.0
    clean_audio_name = _SIDE_DECEPTIVE_SUB_STRIP_RE.sub(" ", name)
    clean_tok_h = _extract_token_hashes(clean_audio_name)
    ad = bool(_SIDE_ANIME_DUB_KW_RE.search(clean_audio_name)) or (grp_h in _DUAL_DUB_GRP_H) or bool(clean_tok_h & _DUAL_DUB_GRP_H)
    if ad:
        side[16] = 1.0
    if (grp_h in _SUB_ONLY_GRP_H) and not ad:
        side[14] = 1.0
    if has_game_grp:
        side[17] = 1.0
    if has_sw_grp and not _SIDE_CJK_RE.search(name):
        side[18] = 1.0
    if _SIDE_COURSE_RE.search(name) and not ag:
        side[19] = 1.0
    if _SIDE_BOOK_COMIC_KW_RE.search(name) or has_book_grp:
        side[20] = 1.0

    lang_str = (row.get("language") or "").strip().lower()
    if lang_str in ("ja", "ru", "fr", "es", "de", "it", "pt", "zh", "ko", "hi", "pl", "nl", "tr", "ar") or _SIDE_CJK_RE.search(name) or ((grp_h in _SUB_ONLY_GRP_H) and not ad):
        side[22] = 1.0
    elif lang_str in ("en", "eng", "english") or ad:
        side[21] = 1.0
    else:
        side[23] = 1.0

    # R12 derivative-group features (raw counts stay in `deriv_features`;
    # the network sees bounded log-normalized U/E, binary P, raw s_max).
    _u, _e, _p, _smax = deriv_features
    side[24] = min(1.0, math.log1p(_u) / _DERIV_LOG_NORM)
    side[25] = min(1.0, math.log1p(_e) / _DERIV_LOG_NORM)
    side[26] = float(_p)
    side[27] = float(_smax)

    return {
        "name_bytes": raw_bytes,
        "ext": exts if exts else [0],
        "size": sizes if sizes else [0],
        "depth": depths if depths else [0],
        "words": words_list if words_list else [[0, 0, 0, 0]],
        "stats": base_stats + side,
        "deriv_features": deriv_features,
        "name_text": name_text,
    }


def build_onnx_batch_inputs(
    rows: Sequence[Mapping[str, Any]],
    max_name_len: int = 256,
    max_files: int = 64,
    stats_dim: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Collate a sequence of torrent dicts into padded NumPy arrays for ONNX Runtime.

    `stats_dim` defaults to `STATS_DIM` (40). Pass a legacy checkpoint's width
    (e.g. `36`) to emit the legacy `stats` prefix during the R12 transition;
    `ArmigerStandaloneONNX.predict_batch` does this automatically from the
    loaded session's `stats` input shape.
    """
    encoded = [encode_torrent_features(r, max_name_len=max_name_len, max_files=max_files) for r in rows]
    bsz = len(encoded)
    max_l = max(len(e["name_bytes"]) for e in encoded)
    max_f = max(len(e["ext"]) for e in encoded)
    width = STATS_DIM if stats_dim is None else int(stats_dim)

    name_b = np.zeros((bsz, max_l), dtype=np.int64)
    name_l = np.zeros((bsz,), dtype=np.int64)
    ext_t = np.zeros((bsz, max_f), dtype=np.int32)
    size_t = np.zeros((bsz, max_f), dtype=np.int64)
    depth_t = np.zeros((bsz, max_f), dtype=np.int64)
    words_t = np.zeros((bsz, max_f, 4), dtype=np.int32)
    stats_t = np.zeros((bsz, width), dtype=np.float32)

    for i, e in enumerate(encoded):
        lb = len(e["name_bytes"])
        lf = len(e["ext"])
        name_b[i, :lb] = e["name_bytes"]
        name_l[i] = lb
        ext_t[i, :lf] = e["ext"]
        size_t[i, :lf] = e["size"]
        depth_t[i, :lf] = e["depth"]
        words_t[i, :lf, :] = e["words"]
        stats_t[i, : min(width, len(e["stats"]))] = e["stats"][:width]

    return {
        "name_bytes": name_b,
        "name_len": name_l,
        "ext": ext_t,
        "size": size_t,
        "depth": depth_t,
        "words": words_t,
        "stats": stats_t,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


_DEFAULT_MONDRIAN_THRESHOLDS: Dict[str, float] = {"adult": 0.5451, "junk": 0.1010, "owner_scope": 0.4770}


def _mondrian_value(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 < float(v) < 1.0 else None


def load_mondrian_thresholds(calibration_path: str | Path | None) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """(global gate thresholds, per-gate strata) from `calibration.json`; defaults only where nothing is fitted.

    Same precedence as `calibration_and_dht.ArmigerCalibrator`: `mondrian_thresholds[gate]` (a dict replaces the
    legacy block entirely), else legacy `mondrian_conformal_thresholds[gate]["threshold" | "threshold_*"]`;
    strata from `mondrian_stratified_thresholds[gate]["strata"][<category>]` (fitted per predicted category).
    """
    thresholds = dict(_DEFAULT_MONDRIAN_THRESHOLDS)
    strata: Dict[str, Dict[str, float]] = {}
    try:
        cfg = json.loads(Path(calibration_path).read_text(encoding="utf-8")) if calibration_path else {}
    except (OSError, ValueError):
        cfg = {}
    if not isinstance(cfg, dict):
        return thresholds, strata
    flat = cfg.get("mondrian_thresholds")
    if not isinstance(flat, dict):
        legacy = cfg.get("mondrian_conformal_thresholds")
        flat = {}
        for gate, block in (legacy.items() if isinstance(legacy, dict) else ()):
            if isinstance(block, dict):
                keys = ["threshold"] + sorted(k for k in block if str(k).startswith("threshold_"))
                flat[gate] = next((block[k] for k in keys if _mondrian_value(block.get(k)) is not None), None)
            else:
                flat[gate] = block
    for gate in thresholds:
        v = _mondrian_value(flat.get(gate, flat.get("scope") if gate == "owner_scope" else None))
        if v is not None:
            thresholds[gate] = v
    raw_strata = cfg.get("mondrian_stratified_thresholds")
    for gate, block in (raw_strata.items() if isinstance(raw_strata, dict) else ()):
        per = block.get("strata") if isinstance(block, dict) else None
        if isinstance(per, dict):
            g_key = "owner_scope" if gate == "scope" else str(gate)
            strata[g_key] = {str(k): t for k, v in per.items() if (t := _mondrian_value(v)) is not None}
    return thresholds, strata


class ArmigerStandaloneONNX:
    """Self-contained ONNX Runtime classifier for `NorviTech/armiger` and `NorviTech/armiger-ultra`."""

    def __init__(
        self,
        onnx_path: Optional[str | Path] = None,
        vocab_path: Optional[str | Path] = None,
        calibration_path: Optional[str | Path] = None,
    ) -> None:
        import onnxruntime as ort

        base = Path(__file__).resolve().parent
        if onnx_path is None:
            for cand in ("armiger.onnx", "armiger.int8.onnx", "armiger_ultra.fp32.onnx"):
                if (base / cand).exists():
                    onnx_path = base / cand
                    break
        if onnx_path is None:
            raise FileNotFoundError("Could not find armiger.onnx or armiger_ultra.fp32.onnx")
        self.onnx_path = Path(onnx_path)
        default_vocab = base / "vocab.json"
        if not default_vocab.exists() and self.onnx_path.with_name("vocab.json").exists():
            default_vocab = self.onnx_path.with_name("vocab.json")
        self.vocab_path = Path(vocab_path) if vocab_path else default_vocab
        _get_vocab_ext_map(self.vocab_path)
        default_cal = base / "calibration.json"
        if not default_cal.exists() and self.onnx_path.with_name("calibration.json").exists():
            default_cal = self.onnx_path.with_name("calibration.json")
        self.calibration_path = Path(calibration_path) if calibration_path else default_cal
        self.mondrian_thresholds, self.mondrian_strata = load_mondrian_thresholds(self.calibration_path)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        self.head_names = [o.name for o in self.session.get_outputs()]

    @staticmethod
    def _apply_v3_post_neural_guard(row: Mapping[str, Any], item_res: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_v3_post_neural_guard(row, item_res)

    def mondrian_threshold(self, gate: str, stratum: Optional[str] = None) -> float:
        """Stratified Mondrian threshold for `gate` (per predicted category when fitted, else global)."""
        per = self.mondrian_strata.get(gate, {})
        if stratum is not None and str(stratum) in per:
            return per[str(stratum)]
        return self.mondrian_thresholds[gate]

    def _session_stats_dim(self) -> int:
        """`stats` width expected by the loaded checkpoint (legacy 36 or R12 40)."""
        try:
            for inp in self.session.get_inputs():
                if inp.name == "stats" and len(inp.shape) == 2 and isinstance(inp.shape[1], int):
                    return int(inp.shape[1])
        except Exception:
            pass
        return STATS_DIM

    def predict_batch(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        feed = build_onnx_batch_inputs(rows, stats_dim=self._session_stats_dim())
        raw_outs = self.session.run(self.head_names, feed)
        logits_by_head = dict(zip(self.head_names, raw_outs))
        results: List[Dict[str, Any]] = []
        for i in range(len(rows)):
            item_res: Dict[str, Any] = {"heads": {}, "probabilities": {}}
            for h, arr in logits_by_head.items():
                vec = arr[i].astype(np.float64)
                if vec.ndim == 2:
                    # 3D batch tensor (e.g. span_bio [B, 256, 7]) -> 2D per-item [256, 7]
                    shifted = vec - np.max(vec, axis=-1, keepdims=True)
                    exps = np.exp(shifted)
                    probs_2d = exps / np.maximum(1e-12, np.sum(exps, axis=-1, keepdims=True))
                    raw_nm = str(rows[i].get("name") or rows[i].get("title") or "")
                    try:
                        try:
                            from armiger.byte_span_extractor import decode_bio_logits_viterbi, decode_bio_tags_to_spans
                        except ImportError:
                            from byte_span_extractor import decode_bio_logits_viterbi, decode_bio_tags_to_spans
                        raw_b = raw_nm.encode("utf-8", errors="replace")[: vec.shape[0]]
                        tag_ids = [int(x) for x in decode_bio_logits_viterbi(vec, raw_b).tolist()]
                    except Exception:
                        tag_ids = [int(x) for x in np.argmax(probs_2d, axis=-1).tolist()]
                    classes = NEURAL_HEAD_CLASSES.get(
                        h,
                        list(SPAN_BIO_CLASSES),
                    )
                    tag_labels = [classes[idx_c] if idx_c < len(classes) else str(idx_c) for idx_c in tag_ids]
                    item_res["heads"][h] = tag_labels
                    mean_probs = np.mean(probs_2d, axis=0)
                    item_res["probabilities"][h] = {
                        classes[k]: round(float(mean_probs[k]), 6) for k in range(min(len(classes), len(mean_probs)))
                    }
                    try:
                        item_res["span_bio"] = decode_bio_tags_to_spans(raw_nm, tag_ids)
                    except Exception:
                        item_res["span_bio"] = {"tag_ids": tag_ids, "tags": tag_labels}
                    continue
                probs = _softmax(vec)
                classes = NEURAL_HEAD_CLASSES.get(h, [str(k) for k in range(len(probs))])
                idx_c = int(np.argmax(probs))
                item_res["heads"][h] = classes[idx_c] if idx_c < len(classes) else str(idx_c)
                item_res["probabilities"][h] = {
                    classes[k]: round(float(probs[k]), 6) for k in range(min(len(classes), len(probs)))
                }
            self._apply_v3_post_neural_guard(rows[i], item_res)
            p_junk = item_res["probabilities"].get("junk", {}).get("junk", 0.0)
            p_adult = item_res["probabilities"].get("adult", {}).get("adult", 0.0)
            # Calibrated excluded probability is kept as-is: the conformal gate (not the argmax label) decides.
            p_excl = item_res["probabilities"].get("owner_scope", {}).get("excluded", 0.0)
            stratum = item_res["heads"].get("category")
            thr = {g: self.mondrian_threshold(g, stratum) for g in ("adult", "junk", "owner_scope")}
            item_res["p_junk"] = p_junk
            item_res["mondrian_gates"] = {
                "adult_flagged": bool(p_adult >= thr["adult"]),
                "junk_flagged": bool(p_junk >= thr["junk"]),
                "scope_excluded": bool(p_excl >= thr["owner_scope"]),
                "passed_all": bool(p_adult < thr["adult"] and p_junk < thr["junk"] and p_excl < thr["owner_scope"]),
                "thresholds": thr,
                "owner_scope_stratum": stratum,
            }
            results.append(item_res)
        return results

    def predict(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([row])[0]


ArmigerONNXClassifier = ArmigerStandaloneONNX


def _apply_v3_post_neural_guard(row, result):
    return result
