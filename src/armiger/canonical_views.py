"""Plane 0 canonicalization, 3 parallel text views, deterministic P(x) audio projection, and Policy-2 v2."""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Any

# Zero-width / invisible / bidi / format characters stripped in View A/B (counted in View C!)
_ZERO_WIDTH_RE = re.compile(
    r"[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e"
    r"\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\uffa0]"
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")
_ALNUM_SEP_RE = re.compile(r"[^a-z0-9]+")

# Compact UTS #39 confusable / homoglyph fold table for safety skeleton (View B ONLY!)
# Covers Cyrillic, Greek, common Latin lookalikes, fullwidth, and leetspeak digits inside words.
_CONFUSABLE_MAP = str.maketrans({
    # Cyrillic -> Latin lookalikes
    "\u0430": "a", "\u0410": "a", "\u0432": "b", "\u0412": "b",
    "\u0441": "c", "\u0421": "c", "\u0435": "e", "\u0415": "e",
    "\u0451": "e", "\u0401": "e", "\u043d": "h", "\u041d": "h",
    "\u0456": "i", "\u0406": "i", "\u0458": "j", "\u0408": "j",
    "\u043a": "k", "\u041a": "k", "\u043c": "m", "\u041c": "m",
    "\u043e": "o", "\u041e": "o", "\u0440": "p", "\u0420": "p",
    "\u0455": "s", "\u0405": "s", "\u0442": "t", "\u0422": "t",
    "\u0443": "y", "\u0423": "y", "\u0445": "x", "\u0425": "x",
    "\u0437": "3", "\u0417": "3", "\u043f": "n", "\u0438": "u",
    # Greek -> Latin lookalikes
    "\u03b1": "a", "\u0391": "a", "\u03b2": "b", "\u0392": "b",
    "\u03b5": "e", "\u0395": "e", "\u03b6": "z", "\u0396": "z",
    "\u03b7": "n", "\u0397": "h", "\u03b9": "i", "\u0399": "i",
    "\u03ba": "k", "\u039a": "k", "\u03bc": "m", "\u039c": "m",
    "\u03bd": "v", "\u039d": "n", "\u03bf": "o", "\u039f": "o",
    "\u03c1": "p", "\u03a1": "p", "\u03c4": "t", "\u03a4": "t",
    "\u03c5": "u", "\u03a5": "y", "\u03c7": "x", "\u03a7": "x",
    # Common Latin/Turkish/phonetic lookalikes
    "\u0131": "i", "\u0130": "i", "\u00df": "ss", "\u00f8": "o",
    "\u00d8": "o", "\u0142": "l", "\u0141": "l", "\u0111": "d",
    "\u0110": "d", "\u00fe": "th", "\u00de": "th", "\u00e6": "ae",
    "\u00c6": "ae", "\u0153": "oe", "\u0152": "oe",
})
_LEET_MAP = str.maketrans({
    "@": "a", "$": "s", "!": "i", "|": "i", "+": "t",
})


def view_a_raw_nfkc(text: str) -> str:
    """View A: Multilingual-preserving NFKC view (NO confusable/diacritic stripping)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _CTRL_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def view_b_safety_skeleton(text: str) -> str:
    """View B: Confusable/diacritic-folded ASCII skeleton for Plane-0 safety & dedup ONLY."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _CTRL_RE.sub(" ", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.translate(_CONFUSABLE_MAP).translate(_LEET_MAP).lower()
    return _WS_RE.sub(" ", s).strip()


def view_c_byte_obf_stats(text: str) -> dict[str, float]:
    """View C: Obfuscation & byte-surface statistics before normalization."""
    if not text:
        return {
            "zw_count": 0.0, "non_ascii_ratio": 0.0, "digit_ratio": 0.0,
            "punct_ratio": 0.0, "mixed_script_flag": 0.0, "byte_len": 0.0,
        }
    s = str(text)
    n = max(1, len(s))
    zw = len(_ZERO_WIDTH_RE.findall(s))
    non_ascii = sum(1 for c in s if ord(c) > 127)
    digits = sum(1 for c in s if c.isdigit())
    punct = sum(1 for c in s if unicodedata.category(c).startswith("P"))
    scripts = set()
    for c in s:
        if c.isalpha():
            o = ord(c)
            if o < 128:
                scripts.add("LATIN_ASCII")
            elif 0x0400 <= o <= 0x04FF:
                scripts.add("CYRILLIC")
            elif 0x0370 <= o <= 0x03FF:
                scripts.add("GREEK")
            elif 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF:
                scripts.add("CJK")
            elif 0x0600 <= o <= 0x06FF:
                scripts.add("ARABIC")
            elif 0x0900 <= o <= 0x097F:
                scripts.add("DEVANAGARI")
            else:
                scripts.add("OTHER")
    return {
        "zw_count": float(zw),
        "non_ascii_ratio": float(non_ascii) / n,
        "digit_ratio": float(digits) / n,
        "punct_ratio": float(punct) / n,
        "mixed_script_flag": 1.0 if len(scripts) >= 2 else 0.0,
        "byte_len": float(len(s.encode("utf-8", errors="ignore"))),
    }


def norm_name(text: str) -> str:
    """Legacy-compatible alnum space-folded normalized name (matches v3.16.1 firewall)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    return _WS_RE.sub(" ", re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)).strip()


def canon_skeleton(text: str) -> str:
    """Strict confusable-folded alphanumeric skeleton for leak-proof firewall & group dedup."""
    sk = view_b_safety_skeleton(text)
    return _ALNUM_SEP_RE.sub("", sk)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------------------------------------------------------
# Deterministic Audio-Only Projection P(x) — STRICTLY subtitle-invariant!
# ---------------------------------------------------------------------------
_SUBTITLE_EXTS = frozenset({
    ".srt", ".ass", ".ssa", ".sub", ".idx", ".vtt", ".sup", ".smi", ".sami",
    ".pgs", ".ttml", ".dfxp", ".sbv", ".lrc", ".usf", ".mks",
})
_SUBTITLE_DIR_RE = re.compile(
    r"(?:^|/)((?:eng(?:lish)?[\s._-]*)?subs?|subtitles?|subtitulos?|sous[\s._-]*titres?|"
    r"untertitel|sottotitoli|legendas?|captions?|cc|sdh)(?:/|$)",
    re.IGNORECASE,
)
# Subtitle-only tags stripped completely from titles/paths BEFORE entering the isolated audio branch.
# Carefully avoids stripping spoken-audio tags (e.g. dual audio, dubbed, multi audio, french audio).
_LANG_TOK = (
    r"(?:multi|dual|eng(?:lish)?|en|es|esp|spa(?:nish)?|fr|fre(?:nch)?|ger(?:man)?|de|deu|"
    r"ita(?:lian)?|pt|por|br|ptbr|ru|rus(?:sian)?|jp|jpn|jap|kor(?:ean)?|zh|chi|chs|cht|cmn|"
    r"ar|ara(?:bic)?|nl|dut|pl|pol|tr|tur|he|heb|sv|swe|no|nor|da|dan|fi|fin|hu|hun|cs|cze|"
    r"ro|rum|el|gre|vi|vie|th|tha|id|ind|ms|may|hi|hin|uk|ukr|sr|hr|bg|sk|sl|et|lv|lt|fa|per|"
    r"auto|forced|full|retail|with|incl(?:uding)?|plus|no|only|raw|cc|sdh|hoh)"
)
_SUB_COMPOUND = (
    r"(?:hardsub(?:s|bed)?|softsub(?:s|bed)?|esub(?:s)?|msub(?:s)?|engsub(?:s)?|ensub(?:s)?|"
    r"subtitled|subbed|subita|subesp|subeng|subfr|subger|subpt|subbr|"
    r"subru|subpl|subnl|subkor|subjp|subzh|multisub(?:s)?|vostfr|vost|legendado|omu|fsub|"
    r"sous[\s._-]*titres?|untertitel|sottotitoli|legendas?|napisy)"
)
_SUB_BRACKET_RE = re.compile(
    rf"[\[\(\{{]\s*(?:{_LANG_TOK}[\s._+/&,-]+)*(?:{_SUB_COMPOUND}|subtitles?|subtitulos?|sub(?:s)?)"
    rf"(?:[\s._+/&,-]+(?:{_LANG_TOK}|{_SUB_COMPOUND}|subtitles?|sub(?:s)?))*\s*[\]\)\}}]"
    rf"|[\[\(\{{]\s*(?:cc|sdh|hoh)(?:[\s._+/&,-]+(?:{_LANG_TOK}|cc|sdh|hoh))*\s*[\]\)\}}]",
    re.IGNORECASE,
)
_SUB_MOD = r"(?:with|incl(?:uding)?|plus|eng(?:lish)?|en|multi)"
_SUB_TOKEN_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?:"
    rf"(?:(?:with|incl(?:uding)?|plus)[\s._+-]+)?{_SUB_COMPOUND}"
    rf"|(?:(?:with|incl(?:uding)?|plus)[\s._+-]+)?(?:{_SUB_MOD}[\s._+-]+)?(?:subtitles?|subtitulos?)"
    rf"|(?:(?:with|incl(?:uding)?|plus)[\s._+-]+)(?:{_LANG_TOK}[\s._+-]+)?subs?"
    rf"|(?:(?:eng(?:lish)?|en|multi)[\s._+-]+subs?)"
    rf"|(?:subs?[\s._+-]+{_LANG_TOK}(?:[\s._+/&,-]+{_LANG_TOK})*)"
    rf"|(?:cc|sdh|hoh)(?=[\s._+\-\]\)\}}]|$)"
    rf")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_MEDIA_AUDIO_EXTS = frozenset({
    ".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg", ".webm", ".flv", ".vob",
    ".mp3", ".flac", ".m4a", ".m4b", ".aac", ".ogg", ".opus", ".wav", ".wma", ".alac", ".ac3", ".dts", ".mka",
})


def _strip_sub_tokens(s: str) -> str:
    s = view_a_raw_nfkc(s)
    prev = None
    for _ in range(4):
        if s == prev:
            break
        prev = s
        s = _SUB_BRACKET_RE.sub(" ", s)
        s = _SUB_TOKEN_RE.sub(" ", s)
        # Clean empty leftover brackets/parens
        s = re.sub(r"[\[\(\{]\s*[\]\)\}]", " ", s)
        s = re.sub(r"[\s._+-]{2,}", " ", s)
    return _WS_RE.sub(" ", s).strip(" ._-+")


def project_audio_only(name: str, files: list[Any] | None = None) -> tuple[str, list[str]]:
    """
    Deterministic Audio-Only Projection P(x):
    - Strips ALL subtitle filenames (.srt/.ass/.sub/.idx/.vtt/.mks/...) and subtitle directories.
    - Strips ALL subtitle tags/tokens from title & media filenames while preserving spoken-audio
      tags, title language/script, release group, and video/audio media filenames.
    - Returns (clean_audio_title, clean_media_filenames[:8]).
    """
    clean_title = _strip_sub_tokens(name or "")
    kept_media: list[tuple[int, str]] = []
    for f in (files or [])[:256]:
        if isinstance(f, dict):
            p = str(f.get("path") or f.get("name") or f.get("p") or "")
            sz = int(f.get("size") or f.get("length") or f.get("s") or 0)
        else:
            p = str(f)
            sz = 0
        pl = p.lower().replace("\\", "/")
        _, ext = os.path.splitext(pl)
        if ext in _SUBTITLE_EXTS:
            continue
        if _SUBTITLE_DIR_RE.search(pl):
            continue
        if ext in _MEDIA_AUDIO_EXTS:
            # Strip directory path components that are pure subtitle dirs; keep basename + parent
            parts = [seg for seg in pl.split("/") if seg]
            tail = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "")
            cleaned = _strip_sub_tokens(tail)
            if cleaned:
                kept_media.append((sz, cleaned))
    # Sort deterministically by descending size then lexicographically
    kept_media.sort(key=lambda x: (-x[0], x[1]))
    media_names = [m for _, m in kept_media[:8]]
    return clean_title, media_names


def project_audio_text(name: str, files: list[Any] | None = None) -> str:
    """Single deterministic string passed ONLY to the isolated Plane-1 audio branch."""
    ct, media = project_audio_only(name, files)
    if not media:
        return ct
    # Append up to 3 distinct representative media stems not identical to title
    ct_l = ct.lower()
    extras = []
    for m in media:
        if m.lower() != ct_l and m not in extras:
            extras.append(m)
        if len(extras) >= 3:
            break
    return ct if not extras else f"{ct} | " + " ; ".join(extras)


# ---------------------------------------------------------------------------
# Policy-2 v2 (Exact Contract v2 Overlay + Demotion-Only Game Exemption)
# ---------------------------------------------------------------------------
# Unambiguous malware dropper extensions NEVER exempted even inside game/software payloads.
_HARD_DROPPER_EXTS = frozenset({
    ".lnk", ".scr", ".hta", ".vbs", ".vbe", ".jse", ".wsf", ".wsh", ".pif", ".cpl", ".msc",
})
# Shell script extensions that are Policy-2 on non-SG media and hard-droppers on small-ISO/decoy traps.
_SCRIPT_EXTS = frozenset({".ps1", ".cmd", ".bat"})
# Exact Label Contract v1/v2 Policy-2 multi-file always-junk extensions (for non-software/non-game).
_POLICY2_ALWAYS_EXTS = frozenset({
    ".apk", ".chm", ".url", ".lnk", ".scr", ".vbs", ".vbe", ".hta", ".pif", ".bat", ".cmd", ".ps1",
})
_POLICY2_SMALL_EXE_EXTS = frozenset({".exe", ".msi", ".com"})
_POLICY2_SINGLE_FILE_EXTS = frozenset({
    ".torrent", ".ps1", ".vbs", ".vbe", ".hta", ".scr", ".pif", ".lnk", ".url",
})
_POLICY2_SMALL_EXE_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB

_VIDEO_EXTS = frozenset({
    ".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".mpg", ".mpeg",
    ".webm", ".flv", ".vob", ".ogm", ".rmvb", ".3gp", ".asf", ".divx", ".f4v", ".mts",
})
_GAME_ASSET_EXTS = frozenset({
    ".pak", ".vpk", ".bsa", ".ba2", ".wad", ".pk3", ".pk4", ".rpf", ".forge", ".bik", ".bk2",
    ".ucas", ".utoc", ".pck", ".unity3d", ".assets", ".ress", ".resource", ".bnk", ".fsb",
    ".xnb", ".esm", ".esp", ".esl", ".bgl", ".sii", ".scs", ".bin", ".cpk", ".arc", ".dat",
    ".obb", ".nsp", ".xci", ".cia", ".3ds", ".nds", ".iso", ".cso", ".chd", ".rvz", ".wbfs",
    ".gba", ".smc", ".sfc", ".nes", ".n64", ".z64", ".pkg", ".vdf", ".so", ".dll",
})
_ARCHIVE_EXTS = frozenset({".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".iso"})
_REDIRECT_TRAP_EXTS = frozenset({".url", ".webloc", ".website"})
_IA_SUFS = (".ia.mp4", "_meta.xml", "_files.xml", "_archive.torrent")
# Strict password-trap filename regex (excludes innocent readme.txt / codec / passenger / pw substrings)
_PASS_TRAP_RE = re.compile(
    r"(?:^|[\s._/-])(?:password|pass[\s._-]*word|extract[\s._-]*pass|unzip[\s._-]*pass|"
    r"how[\s._-]*to[\s._-]*open|readme[\s._-]*first|get[\s._-]*password)"
    r"(?:[\s._/-].*)?\.(?:txt|url|html?|rtf|pdf|doc|docx|nfo)$",
    re.IGNORECASE,
)
_COURSE_HINT_RE = re.compile(
    r"(?:^|[/\\\s._\-\[\(])(?:module|lesson|lecture|section|week|unit|chapter|udemy|coursera|masterclass)[\s._-]*\d*",
    re.IGNORECASE,
)


def policy2_v2_signals(name: str, files: list[Any] | None) -> dict[str, Any]:
    """
    Compute deterministic structural trap signals + demotion-only game-asset evidence.
    Two clean orthogonal flags:
      1) `hard_dropper_trap`: Unambiguous malware droppers (.lnk/.scr/.hta/.vbs/.vbe/.pif,
         explicit password-archive+url traps, small-ISO script launchers, video-decoy+exe traps).
         NEVER exempted by game assets or software/game category prediction.
      2) `policy2_non_sg_trap`: Exact Label Contract v1/v2 Policy-2 multi-file overlay
         (.apk/.chm/.url/.lnk/.scr/.vbs/.hta/.pif/.bat/.cmd/.ps1 or <=10MiB .exe/.msi/.com
         excluding RARBG_DO_NOT_MIRROR<=4KiB, or single-file .torrent/dropper) for non-IA payloads.
         Applied at arbitration ONLY when predicted category is NOT software/game AND NOT
         `game_demotion_eligible` AND NOT pure `.chm` ebook.
    """
    fl = files or []
    nl = (name or "").lower()
    total_bytes = 0
    vid_bytes = 0
    game_asset_bytes = 0
    exec_bytes = 0
    archive_bytes = 0
    n_hard_dropper = 0
    n_script = 0
    n_ps1 = 0
    n_exe_msi = 0
    n_small_exe = 0
    n_vid = 0
    n_game_asset = 0
    n_redirect = 0
    n_pass_trap = 0
    n_policy2_non_chm = 0
    n_chm = 0
    n_creative_media = 0
    n_small_apk = 0
    n_files = len(fl)
    has_ia = any(nl.endswith(suf) for suf in _IA_SUFS)
    course_like = bool(_COURSE_HINT_RE.search(nl))

    for f in fl[:512]:
        if isinstance(f, dict):
            p = str(f.get("path") or f.get("name") or f.get("p") or "")
            sz = max(0, int(f.get("size") or f.get("length") or f.get("s") or 0))
        else:
            p = str(f)
            sz = 0
        pl = p.lower().replace("\\", "/")
        if not has_ia and any(pl.endswith(suf) for suf in _IA_SUFS):
            has_ia = True
        base = pl.rsplit("/", 1)[-1]
        _, ext = os.path.splitext(base)
        total_bytes += sz
        if ext in _HARD_DROPPER_EXTS:
            n_hard_dropper += 1
            exec_bytes += sz
        elif ext in _SCRIPT_EXTS:
            n_script += 1
            if ext == ".ps1":
                n_ps1 += 1
            exec_bytes += sz
        elif ext in _POLICY2_SMALL_EXE_EXTS or ext in (".apk", ".dmg", ".appimage"):
            is_rarbg = ("rarbg_do_not_mirror" in base and sz <= 4096)
            if not is_rarbg:
                n_exe_msi += 1
                exec_bytes += sz
                if ext in _POLICY2_SMALL_EXE_EXTS and sz <= _POLICY2_SMALL_EXE_MAX_BYTES:
                    n_small_exe += 1
                if ext == ".apk" and sz <= 2_000_000:
                    n_small_apk += 1
        if ext in _POLICY2_ALWAYS_EXTS:
            if ext == ".chm":
                n_chm += 1
            else:
                n_policy2_non_chm += 1
        if ext in _VIDEO_EXTS:
            n_vid += 1
            vid_bytes += sz
        if ext in (".epub", ".mobi", ".azw3", ".cbz", ".cbr", ".cb7", ".mp3", ".flac", ".m4b", ".ogg", ".wav"):
            n_creative_media += 1
        if ext in _GAME_ASSET_EXTS and sz >= 1_000_000:
            n_game_asset += 1
            game_asset_bytes += sz
        if ext in _ARCHIVE_EXTS:
            archive_bytes += sz
        if ext in _REDIRECT_TRAP_EXTS:
            n_redirect += 1
        if _PASS_TRAP_RE.search(base):
            n_pass_trap += 1

    if n_vid >= 5:
        course_like = True

    tb = max(1, total_bytes)
    game_asset_byte_ratio = game_asset_bytes / tb
    archive_byte_ratio = archive_bytes / tb

    # Single-file dropper/torrent check
    _, name_ext = os.path.splitext(nl.rsplit("/", 1)[-1])
    single_file_trap = False
    if not has_ia:
        if n_files == 0 and name_ext in _POLICY2_SINGLE_FILE_EXTS:
            single_file_trap = True
        elif n_files == 1:
            f0 = fl[0]
            p0 = str(f0.get("path") or f0.get("name") or f0.get("p") or "") if isinstance(f0, dict) else str(f0)
            _, ext0 = os.path.splitext(p0.lower().replace("\\", "/").rsplit("/", 1)[-1])
            if ext0 in _POLICY2_SINGLE_FILE_EXTS:
                single_file_trap = True

    # 1. Strict hard-malware dropper trigger (NEVER exempted by game assets or game/sw category!)
    hard_dropper_trap = (
        (n_hard_dropper > 0)
        or (n_pass_trap > 0 and (n_redirect > 0 or archive_byte_ratio > 0.35 or n_script > 0 or n_exe_msi > 0))
        or (n_script > 0 and total_bytes < 50_000_000 and archive_byte_ratio > 0.25)
        or (n_ps1 > 0 and n_game_asset > 0)
        or (n_small_exe > 0 and 1 <= n_vid <= 4 and game_asset_byte_ratio < 0.25 and not course_like)
        or (n_small_apk > 0 and (n_vid > 0 or n_creative_media > 0) and game_asset_byte_ratio < 0.25 and exec_bytes < total_bytes * 0.25 and not course_like)
        or (n_small_exe > 0 and n_creative_media > 0 and game_asset_byte_ratio < 0.25 and exec_bytes <= 2_000_000 and exec_bytes < total_bytes * 0.25 and not course_like)
    )

    # 2. Exact Owner Policy-2 structural trap for non-software / non-game payloads
    has_any_p2_ext = (n_policy2_non_chm > 0) or (n_chm > 0) or (n_small_exe > 0) or (n_exe_msi > 0)
    pure_chm_only = (n_chm > 0) and (n_policy2_non_chm == 0) and (n_small_exe == 0) and (n_vid == 0) and (n_creative_media == 0)
    policy2_non_sg_trap = (not has_ia) and (
        single_file_trap
        or (n_files >= 2 and has_any_p2_ext)
    )

    # 3. Demotion-only legitimate multi-file game evidence (forbids Policy-2 hard floor, hands to junk_gbdt!)
    game_demotion_eligible = (
        (not hard_dropper_trap)
        and n_hard_dropper == 0
        and n_pass_trap == 0
        and (
            (game_asset_byte_ratio >= 0.45 and total_bytes >= 50_000_000 and n_files >= 2)
            or (n_game_asset >= 3 and total_bytes >= 200_000_000)
        )
    )
    return {
        "n_files": n_files,
        "total_bytes": total_bytes,
        "n_dropper": n_hard_dropper + n_script,
        "n_hard_dropper": n_hard_dropper,
        "n_script": n_script,
        "n_exe_msi": n_exe_msi,
        "n_small_exe": n_small_exe,
        "n_vid": n_vid,
        "n_game_asset": n_game_asset,
        "n_redirect": n_redirect,
        "n_pass_trap": n_pass_trap,
        "game_asset_byte_ratio": float(game_asset_byte_ratio),
        "exec_byte_ratio": float(exec_bytes / tb),
        "archive_byte_ratio": float(archive_byte_ratio),
        "has_ia": bool(has_ia),
        "pure_chm_only": bool(pure_chm_only),
        "hard_dropper_trap": bool(hard_dropper_trap),
        "policy2_non_sg_trap": bool(policy2_non_sg_trap),
        "game_demotion_eligible": bool(game_demotion_eligible),
    }
