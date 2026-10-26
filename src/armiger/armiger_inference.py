"""Standalone Armiger v2.1 / Armiger Ultra ONNX Featurizer & Inference Runtime (`armiger_inference.py`).

Implements exact feature parity with the Armiger v2.1 training pipeline (`sift/prepare.py` +
`enrich_side21_v2` 36-dim `stats` vector + `vocab.json` 4,096-extension lookup + CRC32
digit-folded path-word hashing) with zero external codebase dependencies beyond `numpy`
and `onnxruntime`.

All release-group and publisher token tables are stored as 8-byte BLAKE2b-64 hex digests
so zero copyrighted or scene group strings appear in plaintext.
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


def _h64(s: str) -> str:
    return hashlib.blake2b(s.strip().lower().encode("utf-8"), digest_size=8).hexdigest()


def _extract_token_hashes(text: str) -> frozenset[str]:
    toks = [m.group(0).lower() for m in _TOKEN_SCAN_RE.finditer(text)]
    out = {_h64(t) for t in toks}
    for i in range(len(toks) - 1):
        out.add(_h64(f"{toks[i]} {toks[i + 1]}"))
    return frozenset(out)


def _get_vocab_ext_map(vocab_path: Optional[Path] = None) -> Dict[str, int]:
    global _VOCAB_EXT_MAP
    if vocab_path is None and _VOCAB_EXT_MAP is not None and len(_VOCAB_EXT_MAP) > 64:
        return _VOCAB_EXT_MAP
    vpath = vocab_path if (vocab_path and vocab_path.exists()) else Path(__file__).resolve().with_name("vocab.json")
    if not vpath.exists() and os.environ.get("ARMIGER_MODEL_DIR"):
        vpath = Path(os.environ["ARMIGER_MODEL_DIR"]) / "vocab.json"
    if vpath.exists():
        data = json.loads(vpath.read_text(encoding="utf-8"))
        _VOCAB_EXT_MAP = {e: i + 1 for i, e in enumerate(data.get("exts", []))}
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
    b_name = name.encode("utf-8", "replace")
    if len(b_name) > max_name_len:
        b_name = b_name[:max_name_len].decode("utf-8", "ignore").encode("utf-8")
    raw_bytes = list(b_name) if b_name else [32]

    raw_files = row.get("files") or []
    total_size = int(row.get("size") or 0)
    all_items: List[Tuple[str, int]] = []
    for f in raw_files:
        if isinstance(f, Mapping):
            p = str(f.get("path") or f.get("p") or f.get("name") or "")
            sz = int(f.get("size") or f.get("s") or 0)
        else:
            p = str(f or "")
            sz = 0
        if p and "/.____padding_file/" not in p and not p.startswith(".____padding_file"):
            all_items.append((p, sz))
    all_items.sort(key=lambda x: -x[1])
    items: List[Tuple[str, int]] = all_items[:max_files]
    if not items and (row.get("ext") or row.get("extension") or "." in name):
        items.append((name, total_size))

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
    count = max(int(row.get("file_count") or 0), len(all_items), len(items))
    biggest = max((max(s, 0) for _, s in items), default=0)
    max_share = (biggest / tot_k if tot_k > 0 else (1.0 / len(items) if items else 0.0))

    base_stats = [
        math.log1p(max(count, 0)),
        math.log1p(max(total_size if total_size > 0 else int(tot_k), 0)),
    ]
    base_stats += [w / wsum if wsum > 0 else 0.0 for w in weights]
    base_stats += [max_share, float(has_sample)]

    side = [0.0] * 24
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

    return {
        "name_bytes": raw_bytes,
        "ext": exts if exts else [0],
        "size": sizes if sizes else [0],
        "depth": depths if depths else [0],
        "words": words_list if words_list else [[0, 0, 0, 0]],
        "stats": base_stats + side,
    }


def build_onnx_batch_inputs(
    rows: Sequence[Mapping[str, Any]],
    max_name_len: int = 256,
    max_files: int = 64,
) -> Dict[str, np.ndarray]:
    """Collate a sequence of torrent dicts into padded NumPy arrays for ONNX Runtime."""
    encoded = [encode_torrent_features(r, max_name_len=max_name_len, max_files=max_files) for r in rows]
    bsz = len(encoded)
    max_l = max(len(e["name_bytes"]) for e in encoded)
    max_f = max(len(e["ext"]) for e in encoded)

    name_b = np.zeros((bsz, max_l), dtype=np.int64)
    name_l = np.zeros((bsz,), dtype=np.int64)
    ext_t = np.zeros((bsz, max_f), dtype=np.int32)
    size_t = np.zeros((bsz, max_f), dtype=np.int64)
    depth_t = np.zeros((bsz, max_f), dtype=np.int64)
    words_t = np.zeros((bsz, max_f, 4), dtype=np.int32)
    stats_t = np.zeros((bsz, 36), dtype=np.float32)

    for i, e in enumerate(encoded):
        lb = len(e["name_bytes"])
        lf = len(e["ext"])
        name_b[i, :lb] = e["name_bytes"]
        name_l[i] = lb
        ext_t[i, :lf] = e["ext"]
        size_t[i, :lf] = e["size"]
        depth_t[i, :lf] = e["depth"]
        words_t[i, :lf, :] = e["words"]
        stats_t[i, :] = e["stats"]

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


class ArmigerStandaloneONNX:
    """Self-contained ONNX Runtime classifier for `NorviTech/armiger` and `NorviTech/armiger-ultra`."""

    def __init__(
        self,
        onnx_path: Optional[str | Path] = None,
        vocab_path: Optional[str | Path] = None,
        calibration_path: Optional[str | Path] = None,
    ) -> None:
        import onnxruntime as ort

        from .artifacts import model_directory
        base = model_directory()
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
        self.calibration_path = Path(calibration_path) if calibration_path else (base / "calibration.json")
        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        self.head_names = [o.name for o in self.session.get_outputs()]

    @staticmethod
    def _apply_v3_post_neural_guard(row: Mapping[str, Any], item_res: Dict[str, Any]) -> Dict[str, Any]:
        return _apply_v3_post_neural_guard(row, item_res)

    def predict_batch(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        feed = build_onnx_batch_inputs(rows)
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
                    from .byte_span_extractor import decode_bio_logits_viterbi, decode_bio_tags_to_spans
                    raw_b = raw_nm.encode("utf-8", errors="replace")[: vec.shape[0]]
                    tag_ids = [int(x) for x in decode_bio_logits_viterbi(vec, raw_b).tolist()]
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
            p_excl = (
                0.05
                if item_res["heads"].get("owner_scope") == "allowed_type"
                else item_res["probabilities"].get("owner_scope", {}).get("excluded", 0.0)
            )
            item_res["p_junk"] = p_junk
            item_res["mondrian_gates"] = {
                "adult_flagged": bool(p_adult >= 0.5451),
                "junk_flagged": bool(p_junk >= 0.1010),
                "scope_excluded": bool(p_excl >= 0.4770),
                "passed_all": bool(p_adult < 0.5451 and p_junk < 0.1010 and p_excl < 0.4770),
            }
            results.append(item_res)
        return results

    def predict(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        return self.predict_batch([row])[0]


ArmigerONNXClassifier = ArmigerStandaloneONNX


_V3_COMIC_H = frozenset(['12b3f45e5ceb0c73', '4cecc3908b751508', '560dd703f14b3d39', 'd287a406b53d326d', 'f474f6ed1c868f32'])
_V3_GAME_H = frozenset(['052648bca960e902', '49460c3a7ea1cb2a', '65a3e07111977e18', 'aeeacf8b695dceea', 'f9dd689e8c6d1cd8', 'fd75f166789586c3'])
_V3_OPEN_MOVIE_H = frozenset(['0937eb719d20dd6f', '2e7c54d248a0ef1d', '4d0377a6e2563411', '710ab19297bbed0c', '82d861962967c61e', '97a97017505610c7', '9a8994e8f326412f', 'a320c9ca18122845', 'a82434472cd6c6fe', 'f5f35a2f45111a8e'])
_V3_TV_SERIES_H = frozenset(['230aead54d362cda', '295b881a619e7cb5', '7e7f87fd43987c67', '7f9e0a2bfd52770c', '8ab671ca32f5d5cf', '9f7c0f5280ee75ec', 'c8564e035c0b75c1', 'e2b18ebba4070e03', 'f593bd44736fe5fe', 'fc1a22469a3f145d'])
_V3_ANIME_POS_H = frozenset(['07ea6d84cdf8fd18', '14dc96e9b279ef41', '3f51640da58e3b42', '485b2fbd270bd1ed', '5fc43af228179edd', '7054ce850cad5308', 'a11c618db0e83f96', 'ad7e17f965645161', 'c46009e7f1b3c3af'])
_V3_ANIME_MOVIE_H = frozenset(['14dc96e9b279ef41', '5fc43af228179edd', 'ad7e17f965645161', 'c46009e7f1b3c3af'])
_V3_ANIME_NEG_H = frozenset(['0937eb719d20dd6f', '207ffe8422c5dccc', '43dc423c0f35206b', '4d0377a6e2563411', '52812423d8d44670', '710ab19297bbed0c', '7aa1e3930f50174a', '97a97017505610c7', 'a320c9ca18122845', 'a82434472cd6c6fe', 'b1444d914e2baca2', 'b249fb33aabf5c18'])
_V3_NON_EN_AUDIO_H = frozenset(['07ea6d84cdf8fd18', '12c3d50880c16874', '16b7d611049fe70e', '207ffe8422c5dccc', '34924d18a0b6ed8b', '3f51640da58e3b42', '43dc423c0f35206b', '485b2fbd270bd1ed', '56e0d06e84a655c0', '5f1570e0f9eaacb8', '5fc43af228179edd', '69ca97bb99f505b5', '7054ce850cad5308', '75af437002546cd4', '7aa1e3930f50174a', '8406725ab8ae5378', '900a19dd73bdd4ba', '951625ff1ac61302', 'a11c618db0e83f96', 'a7ddaf4c9ddad893', 'a8d66ae84e768552', 'ad7e17f965645161', 'b1444d914e2baca2', 'b249fb33aabf5c18', 'dbfb8d1d57178337', 'e05fb6bafd7b47f1', 'eb5ac7a33ae81cf2', 'f2a4dad05c81ba2e', 'f6241861fb0c5d79'])
_V3_NON_EN_TITLE_H = frozenset(['207ffe8422c5dccc', '43dc423c0f35206b', '5f1570e0f9eaacb8', '7aa1e3930f50174a', '951625ff1ac61302', 'b1444d914e2baca2', 'f6241861fb0c5d79'])
_V3_SUBWORD_RE = re.compile(r"[A-Za-z0-9]+")


def _extract_v3_hashes(text: str) -> frozenset[str]:
    base = set(_extract_token_hashes(text))
    sub = [m.group(0).lower() for m in _V3_SUBWORD_RE.finditer(text)]
    for t in sub:
        base.add(_h64(t))
    for i in range(len(sub) - 1):
        base.add(_h64(f"{sub[i]} {sub[i + 1]}"))
    return frozenset(base)


def _apply_v3_post_neural_guard(row: Mapping[str, Any], item_res: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Armiger v3.0 post-neural structural, cross-head, and subtitle-trap disambiguation."""
    name = str(row.get("name") or row.get("title") or "").strip()
    raw_files = row.get("files") or []
    fpaths: List[str] = []
    for f in raw_files:
        p = str(f.get("path") or f.get("p") or f.get("name") or "") if isinstance(f, Mapping) else str(f or "")
        if p and "/.____padding_file/" not in p and not p.startswith(".____padding_file"):
            fpaths.append(p)
    all_text = name + " " + " ".join(fpaths[:32])
    name_h = _extract_v3_hashes(name)
    all_h = _extract_v3_hashes(all_text)

    is_nested = "heads" in item_res and isinstance(item_res["heads"], dict)
    if is_nested:
        heads = item_res["heads"]
        probs = item_res.get("probabilities") or {}
    else:
        heads = {
            k: (v.get("label") if isinstance(v, Mapping) else str(v))
            for k, v in item_res.items()
            if isinstance(v, (Mapping, str))
        }
        probs = {
            k: {v.get("label", ""): float(v.get("confidence", 0.5))}
            for k, v in item_res.items()
            if isinstance(v, Mapping) and "label" in v
        }

    p_adult = float(probs.get("adult", {}).get("adult", 0.0))

    # 1. Cross-head adult <-> category consistency: never predict 'xxx' when adult_flagged is False
    explicit_xxx = bool(re.search(r"(?i)\b(?:xxx|siterip|onlyfans|jav|fc2[-_ ]?ppv|18\+|nsfw)\b", name))
    if heads.get("category") == "xxx" and p_adult < 0.5451 and not explicit_xxx:
        cat_probs = dict(probs.get("category") or {})
        cat_probs["xxx"] = 0.0
        cat_probs["unknown"] = min(float(cat_probs.get("unknown", 0.0)), 0.05)
        best_cat = max(cat_probs.items(), key=lambda kv: kv[1])[0] if cat_probs else "movie"
        heads["category"] = best_cat

    # 2. Structural & Domain Category Disambiguation (v3.0)
    if re.search(r"(?i)(?:\bMIT\d+[._]\d+|\bocw[-_]|_lec0?\d+\b|Lecture_0?\d+\b)", all_text) or _SIDE_COURSE_RE.search(name):
        heads["category"] = "course"
    elif re.search(r"(?i)\d{3,6}gut\b", all_text):
        heads["category"] = "ebook"
    elif (name_h & _V3_COMIC_H) or any(
        p.lower().endswith((".cbz", ".cbr", ".cb7", "_cloth_detection.log")) for p in fpaths
    ):
        heads["category"] = "comic"
    elif name_h & _V3_GAME_H:
        heads["category"] = "game"
    elif name_h & _V3_OPEN_MOVIE_H:
        heads["category"] = "movie"
        heads["anime"] = "non_anime"
    elif (all_h & _V3_TV_SERIES_H) or re.search(
        r"(?i)(?:\bS\d{1,2}E\d{1,3}\b|\b\d{1,2}Chapters\b|_ch01\.|\bPart\s*1\b.*\bInterviews\b|\btv-series\b)",
        all_text,
    ):
        heads["category"] = "tv_show"
    elif heads.get("category") == "unknown" and any(p.lower().endswith((".mp4", ".mkv", ".avi", ".mov")) for p in fpaths):
        heads["category"] = "movie"

    # 3. Anime Precision & Recall Guard (v3.0)
    if all_h & _V3_ANIME_POS_H:
        heads["anime"] = "anime"
        if all_h & _V3_ANIME_MOVIE_H:
            heads["category"] = "movie"
    elif all_h & _V3_ANIME_NEG_H:
        heads["anime"] = "non_anime"

    # 4. External Subtitle Inventory & Subtitle-Trap Guard (english_subs & lang)
    real_subs = [
        p for p in fpaths
        if p.lower().endswith((".srt", ".vtt", ".ass"))
        and not p.lower().endswith((".asr.srt", ".autogenerated.vtt", ".asr.js"))
    ]
    has_non_en_sub_only_tag = bool(re.search(r"(?i)\b(?:vostfr|vose|subfrench|napisy[._ -]*pl|czuntertitel)\b", all_text))
    has_en_sub_tag = bool(re.search(r"(?i)\b(?:english[._ -]*sub(?:bed|s)?|eng[._ -]*subs?|esubs?)\b", all_text))
    if real_subs:
        if any(re.search(r"(?i)(?:english|\.en\.|\.eng\.|ocw-|mit\d|lec\d|^L\d{2}-)", Path(p).name) for p in real_subs) or not has_non_en_sub_only_tag:
            heads["english_subs"] = "has_eng_subs"
        else:
            heads["english_subs"] = "no_eng_subs"
    elif has_en_sub_tag:
        heads["english_subs"] = "has_eng_subs"
    elif has_non_en_sub_only_tag:
        heads["english_subs"] = "no_eng_subs"

    # 5. Spoken Language & Subtitle-Trap Audio Disambiguation (lang & title_language)
    name_no_sub = _SIDE_DECEPTIVE_SUB_STRIP_RE.sub(" ", name)
    has_explicit_en_audio = bool(
        _SIDE_ANIME_DUB_KW_RE.search(name)
        or re.search(
            r"(?i)(?:\b(?:eng|english)\b|[\[\(\.\s_-]EN[\]\)\.\s_-]|\b[34]audio\b)",
            name_no_sub,
        )
    )
    has_foreign_context = (
        heads.get("anime") == "anime"
        or heads.get("title_language") == "non_english_title"
        or bool(name_h & _V3_NON_EN_TITLE_H)
        or bool(_SIDE_CJK_RE.search(name))
        or bool(
            re.search(
                r"(?i)\b(?:ita|ger|deu|fre|fra|rus|spa|esp|jpn|jap|kor|chi|hin|dut|nl|por|pol|tur|vostfr|vose)\b",
                name_no_sub,
            )
        )
    )
    is_sub_only_trap = has_en_sub_tag and not has_explicit_en_audio and has_foreign_context
    if is_sub_only_trap or bool(all_h & _V3_NON_EN_AUDIO_H):
        heads["lang"] = "non_english"
        if name_h & _V3_NON_EN_TITLE_H:
            heads["title_language"] = "non_english_title"
    elif heads.get("lang") == "multi_or_unknown" and heads.get("title_language") == "english_title" and heads.get("anime") == "non_anime":
        if heads.get("category") in ("movie", "tv_show", "audiobook", "course"):
            heads["lang"] = "english"

    # 6. Byte-Span Clean Title & Release Year Extraction
    m_yr = re.search(r"(?<![\dA-Za-z])(19[2-9]\d|20[0-2]\d)(?![\dA-Za-z])", name)
    rel_year = int(m_yr.group(1)) if m_yr else None
    clean_t = re.sub(r"\.(?:mkv|mp4|avi|iso|zip|rar|epub|pdf|mp3|flac)$", "", name, flags=re.I)
    if m_yr and m_yr.start() >= 2:
        clean_t = clean_t[:m_yr.start()]
    elif m_yr and m_yr.start() == 0:
        clean_t = clean_t[m_yr.end():]
    clean_t = re.sub(r"(?i)\b(?:1080p|720p|2160p|480p|vose|vostfr|english[._ -]*subbed|eng[._ -]*subs?)\b.*$", "", clean_t)
    clean_t = re.sub(r"[._-]+", " ", clean_t).strip()

    if not is_nested:
        for k, v in heads.items():
            if k in item_res and isinstance(item_res[k], dict):
                item_res[k]["label"] = v
            else:
                item_res[k] = {"label": v, "confidence": 0.95}
    item_res["matching_identity"] = {
        "clean_title": clean_t or name,
        "release_year": rel_year,
    }
    p_junk = float(probs.get("junk", {}).get("junk", 0.0))
    if heads.get("category") in ("movie", "tv_show") and p_adult < 0.5451 and p_junk < 0.1010:
        heads["owner_scope"] = "allowed_type"
    else:
        heads["owner_scope"] = "excluded"
    if not is_nested and "owner_scope" in item_res and isinstance(item_res["owner_scope"], dict):
        item_res["owner_scope"]["label"] = heads["owner_scope"]
    if is_nested:
        item_res["category"] = heads.get("category", "unknown")
    return item_res
