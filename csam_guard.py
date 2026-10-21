"""Multi-Layer CSAM & Harmful Content Protection Engine for Armiger DHT Sync.

Enforces:
  1. Multi-stage ContentFilter normalization parity with multi-stage text normalization, leetspeak deobfuscation,
     zero-width character stripping, and pattern matching.
  2. Deterministic & structural pre-ingest CSAM/abuse blocker checking title, file trees,
     file extensions, and disguised malware droppers before any shard serialization.
  3. Ed25519-signed blinded SHA-256 tombstone purge list with monotonic sequence enforcement
     and cryptographic revocation verification.
  4. Mandatory Armiger Mondrian conformal safety gate evaluation (junk/malware, out-of-scope,
     and strict adult namespace isolation).
"""
from __future__ import annotations

import dataclasses
import enum
import fcntl
import functools
import hashlib
import hmac
import json
import math
import os
import pathlib
import re
import tempfile
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - stdlib fallback for bare CI
    InvalidSignature = Exception  # type: ignore[misc,assignment]
    Ed25519PrivateKey = None  # type: ignore[misc,assignment]
    Ed25519PublicKey = None  # type: ignore[misc,assignment]


class CSAMViolationType(str, enum.Enum):
    """Specific categories of content filter violations."""

    CSAM_EXPLOITATION = "csam_exploitation"
    UNDERAGE_SEXUAL_CONTENT = "underage_sexual_content"
    NONCONSENSUAL_ABUSE = "nonconsensual_abuse"
    EXTREME_VIOLENCE_GORE = "extreme_violence_gore"
    MALWARE_PAYLOAD = "malware_payload"
    TOMBSTONED_HASH = "tombstoned_hash"
    SAFETY_GATE_JUNK = "safety_gate_junk"
    SAFETY_GATE_OUT_OF_SCOPE = "safety_gate_out_of_scope"
    UNISOLATED_ADULT = "unisolated_adult"


class CSAMViolationError(Exception):
    """Raised when torrent metadata or payload violates CSAM/abuse safety policies."""

    def __init__(self, violation_type: CSAMViolationType, reason: str, pattern: str | None = None) -> None:
        super().__init__(f"Safety violation [{violation_type.value}]: {reason}")
        self.violation_type = violation_type
        self.reason = reason
        self.pattern = pattern


@dataclasses.dataclass(frozen=True)
class ContentFilterResult:
    """Detailed result of content filter inspection."""

    is_safe: bool
    violation_type: CSAMViolationType | None
    reason: str | None
    matched_pattern: str | None
    risk_score: float
    sanitized_title: str


# ---------------------------------------------------------------------------
# Leetspeak & Normalization Dictionaries
# ---------------------------------------------------------------------------

_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "@": "a",
    "$": "s",
    "5": "s",
    "7": "t",
    "+": "t",
    "!": "i",
    "8": "b",
    "9": "g",
    "|": "l",
}

# Confusable Cyrillic / Greek homoglyphs mapped to ASCII Latin equivalents
_HOMOGLYPH_MAP: dict[str, str] = {
    "\u0430": "a",  # Cyrillic a
    "\u0432": "b",  # Cyrillic ve
    "\u0435": "e",  # Cyrillic ie
    "\u0438": "i",  # Cyrillic i
    "\u0456": "i",  # Cyrillic byelorussian-ukrainian i
    "\u0457": "i",  # Cyrillic yi
    "\u0458": "j",  # Cyrillic je
    "\u043a": "k",  # Cyrillic ka
    "\u043c": "m",  # Cyrillic em
    "\u043d": "h",  # Cyrillic en
    "\u043e": "o",  # Cyrillic o
    "\u0440": "p",  # Cyrillic er
    "\u0441": "c",  # Cyrillic es
    "\u0455": "s",  # Cyrillic dze
    "\u0442": "t",  # Cyrillic te
    "\u0443": "y",  # Cyrillic u
    "\u0445": "x",  # Cyrillic ha
    "\u03b1": "a",  # Greek alpha
    "\u03b2": "b",  # Greek beta
    "\u03b5": "e",  # Greek epsilon
    "\u03b9": "i",  # Greek iota
    "\u03ba": "k",  # Greek kappa
    "\u03bd": "v",  # Greek nu
    "\u03bf": "o",  # Greek omicron
    "\u03c1": "p",  # Greek rho
    "\u03c4": "t",  # Greek tau
    "\u03c5": "u",  # Greek upsilon
    "\u03c7": "x",  # Greek chi
    "\U0001d455": "h",  # Reserved slot between math italic g (1D454) and i (1D456)
}

# Regex to strip zero-width characters, soft hyphens, word joiners, and bidi overrides/isolates
_ZERO_WIDTH_RE = re.compile(
    r"[\u00AD\u034F\u061C\u180E\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]"
)

# Collapse obfuscated single-character runs (e.g., "c h i l d" -> "child")
_SINGLE_CHAR_RUN_RE = re.compile(r"\b([a-z0-9])\s+(?=[a-z0-9]\b)")

# Delimiters and non-alphanumeric punctuation
_PUNCT_RE = re.compile(r"[\._\-\+\[\]\(\)\{\}\\\/\|\:\,\;\*\^\~\`\'\"]+")

# Repeated characters (e.g. "teeeen" -> "teen")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")

# Private tracker passkey / auth tokens in torrent titles
_TRACKER_TAGS_RE = re.compile(
    r"(\[passkey=[^\]]+\]|\?passkey=[a-zA-Z0-9]+|\[auth=[^\]]+\]|torrentpass=[a-zA-Z0-9]+)",
    re.IGNORECASE,
)

_HOMOGLYPH_TRANS = str.maketrans(_HOMOGLYPH_MAP)
_LEET_TRANS = str.maketrans(_LEET_MAP)
_PRESERVE_ALNUM_TOKENS = frozenset({
    "1080p", "720p", "2160p", "480p", "1080i", "720i",
    "x264", "x265", "h264", "h265", "av1", "vp8", "vp9", "vc1",
    "mp3", "mp4", "m4a", "m4b", "m4v", "m2ts", "3gp", "ac3", "eac3",
    "bd25", "bd50", "dvd5", "dvd9", "win32", "win64", "x64", "x86",
    "8yo", "9yo", "10yo", "11yo", "12yo", "13yo", "14yo", "15yo", "16yo", "17yo",
})


def normalize_text(text: str) -> str:
    """Normalize text through unicode decomposition, leetspeak mapping, and token cleaning.

    Steps:
      1. Unicode NFKD decomposition + strip non-spacing diacritic marks.
      2. Strip zero-width & bidirectional formatting characters.
      3. Lowercase & map confusable Cyrillic/Greek homoglyphs.
      4. Normalize punctuation and delimiters to single spaces.
      5. Map leetspeak characters on mixed/obfuscated tokens while preserving pure numbers
         (so file numbers like 'CSS/43' never become 'cs ae') and age tokens ('14yo').
      6. Collapse repetitive runs of 3+ identical characters.
    """
    if not text:
        return ""

    # 1. Strip Unicode tag characters, variation selectors, and format/control characters pre- and post-NFKD
    pre_cleaned = "".join(
        c for c in text
        if unicodedata.category(c) not in ("Cf", "Mn", "Me", "Cc")
        and not (0xFE00 <= ord(c) <= 0xFE0F)
        and not (0xE0000 <= ord(c) <= 0xE01EF)
    )
    decomp = unicodedata.normalize("NFKD", pre_cleaned)
    stripped = "".join(
        c for c in decomp
        if not unicodedata.combining(c)
        and unicodedata.category(c) not in ("Cf", "Mn", "Me", "Cc")
        and not (0xFE00 <= ord(c) <= 0xFE0F)
        and not (0xE0000 <= ord(c) <= 0xE01EF)
    )

    # 2. Strip zero-width characters & map homoglyphs
    cleaned = _ZERO_WIDTH_RE.sub("", stripped).lower().translate(_HOMOGLYPH_TRANS)

    # 3. Punctuation (except leet symbols @$!+|) to space first so tokens are cleanly bounded
    spaced = re.sub(r"[\._\-\[\]\(\)\{\}\\\/\:\,\;\*\^\~\`\'\"\?\&\=]+", " ", cleaned)

    # 4. Token-aware leetspeak translation: pure digits ("43", "2026") and preserved media/age tokens
    # are kept intact; any token containing letters or leet symbols is leet-decoded.
    out_tokens: list[str] = []
    for tok in spaced.split():
        if (len(tok) >= 2 and tok.isdigit()) or tok in _PRESERVE_ALNUM_TOKENS:
            out_tokens.append(tok)
        else:
            decoded = tok.translate(_LEET_TRANS)
            collapsed = _REPEATED_CHARS_RE.sub(r"\1\1", decoded)
            out_tokens.append(collapsed)

    joined = " ".join(out_tokens)
    punct_normalized = _PUNCT_RE.sub(" ", joined)
    return " ".join(punct_normalized.split())


def sanitize_title(title: str) -> str:
    """Strip private tracker auth tokens, passkeys, and user credentials for complete anonymity."""
    clean = _TRACKER_TAGS_RE.sub("", title)
    return " ".join(clean.split()).strip()


# ---------------------------------------------------------------------------
# Strict Pre-Ingest CSAM & Abuse Patterns
# ---------------------------------------------------------------------------

# Forbidden primary CSAM terms (zero-tolerance keywords and stems; [li] handles 1->i/l ambiguity)
_CSAM_DIRECT_PATTERNS = [
    re.compile(r"\b(c\s*s\s*a\s*m|c\s*s\s*a\s*e)\b", re.IGNORECASE),
    re.compile(r"\b(pedophi[li]|paedophi[li]|pedofi[li]|paedofi[li])\w*", re.IGNORECASE),
    re.compile(r"\b(chi[li]d\s*porn|cp\s*vids?|cp\s*[li]inks?|cp\s*megas?|cp\s*pack\s*co[li]+ection|chi[li]d\s*erotic|underage\s*porn)\b", re.IGNORECASE),
    re.compile(r"\b(pre\s*teen\s*(sex|nude|porn|erotic|hardcore)|preteen[_\-\s]*(sex|nude|porn))\b", re.IGNORECASE),
    re.compile(r"\b([li]o[li]ita\s*(hardcore|hentai|sex|porn|nude|uncensored))\b", re.IGNORECASE),
    re.compile(r"\b(shotacon|shota\s*(sex|porn|hentai|hardcore))\b", re.IGNORECASE),
    re.compile(r"\b(todd[li]er|kindergarten|infant|e[li]ementary)\s*(sex|nude|erotic|porn)\b", re.IGNORECASE),
    re.compile(r"\b(jai[li]\s*bait|bare[li]y\s*[li]ega[li]\s*(underage|minor|chi[li]d))\b", re.IGNORECASE),
    re.compile(r"\b(incest\s*(chi[li]d|daughter|son|minor|underage))\b", re.IGNORECASE),
    re.compile(r"\b(?:[89]|1[0-7])\s*y\s*o\s*(?:fuck\w*|sex|porn|nude|erotic|hardcore|anal|creampie)\b", re.IGNORECASE),
]

# Non-consensual exploitation & sexual violence
_NONCONSENSUAL_PATTERNS = [
    re.compile(r"\b(non\s*consensua[li]|rape\s*(porn|video|scene)|gang\s*rape|forced\s*sex)\b", re.IGNORECASE),
    re.compile(r"\b(revenge\s*porn|spycam\s*(bathroom|shower|toi[li]et|upskirt))\b", re.IGNORECASE),
    re.compile(r"\b(b[li]ackmai[li]\s*sex|sextortion)\b", re.IGNORECASE),
]

# Extreme violence, snuff, execution, gore
_EXTREME_VIOLENCE_PATTERNS = [
    re.compile(r"\b(snuff\s*(fi[li]m|video|movie)|rea[li]\s*beheading|rea[li]\s*execution)\b", re.IGNORECASE),
    re.compile(r"\b([li]ive\s*suicide\s*(video|stream)|massacre\s*[li]ivestream)\b", re.IGNORECASE),
    re.compile(r"\b(rea[li]\s*torture\s*(video|fi[li]m)|crush\s*fetish\s*(anima[li]|ki[li]+))\b", re.IGNORECASE),
]

# High-risk dropper extensions that are NEVER legitimate standalone software/game/media torrents
_ALWAYS_MALWARE_EXTENSIONS = {
    ".scr", ".pif", ".hta", ".vbs", ".vbe", ".wsf", ".wsh",
    ".cpl", ".gadget", ".application",
    ".appref-ms", ".library-ms", ".search-ms", ".settingcontent-ms", ".diagcab",
}

# Suspicious executable & dropper extensions that must never appear disguised as media
_MALWARE_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".vbs", ".vbe", ".js", ".jse",
    ".wsf", ".wsh", ".ps1", ".psm1", ".pif", ".application", ".gadget",
    ".msi", ".msp", ".com", ".hta", ".cpl", ".msc", ".jar", ".lnk", ".dll",
    ".sys", ".drv", ".ocx", ".reg", ".url",
    ".chm", ".one", ".xll", ".xlam", ".docm", ".xlsm",
    ".appref-ms", ".library-ms", ".search-ms", ".settingcontent-ms", ".diagcab",
    ".pkg", ".command",
}

_SAMPLE_SUFFIX_RE = re.compile(r"\.sample\d*$", re.IGNORECASE)
_RTLO_SPOOF_RE = re.compile(r"[\u202d\u202e]")
_BDMV_JAR_PATH_RE = re.compile(r"(?:^|/)bdmv/(?:backup/)?jar/[^/]+\.jar$", re.IGNORECASE)

_RENPY_SAY_VBS_CG_RE = re.compile(r"(?i)(?:^|/)lib/(?:py\d-)?windows-[^/]+/say\.vbs$")

_MEDIA_CODEC_LURE_RE = re.compile(
    r"(?i)(?:^|(?<=[^a-z0-9]))(?:"
    r"codec[\s._-]*pack|"
    r"xvid[\s._-]*codec|"
    r"x264[\s._-]*video[\s._-]*codec"
    r")(?:$|(?=[^a-z0-9]))"
)

_MALWARE_LURE_RE = re.compile(
    r"(?i)(?:^|(?<=[^a-z0-9]))(?:"
    r"required[\s._-]*to[\s._-]*play|"
    r"install[\s._-]*codec|"
    r"nitro[\s._-]*generator|"
    r"free[\s._-]*nitro|"
    r"fake[\s._-]*seeder[\s._-]*swarm[\s._-]*trap|"
    r"adware[\s._-]*bundle[\s._-]*installer|"
    r"password[\s._-]*protected[\s._-]*trap|"
    r"password[\s._-]*in[\s._-]*description|"
    r"corrupted[\s._-]*dead[\s._-]*sample|"
    r"corrupt\.part|"
    r"spam[\s._-]*link[\s._-]*redirect|"
    r"watch[\s._-]*full[\s._-]*movie\.url|"
    r"activar[\s._-]*sonido|"
    r"\d+(?:\.\d+)?\s*tb\s+collection\.torrent"
    r")(?:$|(?=[^a-z0-9]))"
)

_BENIGN_BOOKMARK_URL_RE = re.compile(
    r"(?i)^(?:torrent[\s._-]*downloaded[\s._-]*from|downloaded[\s._-]*from|visit[\s._-]*us|rarbg|1337x|yts|tgx|torrentgalaxy|rutracker|nyaa)"
)

_PASSWORD_COMPANION_RE = re.compile(
    r"(?i)(?:password|contrase[nñ]a|parola|mot[\s._-]*de[\s._-]*passe|passwort|how[\s._-]*to[\s._-]*extract|readme[\s._-]*pass)"
)

_ANIME_OR_EP_ZIP_CG_RE = re.compile(
    r"(?i)(?:^\[[^\[\]]{2,30}\].*?(?:\s-\s*\d{1,3}\b|\bS\d{1,2}E\d{1,3}\b).*?\b(?:1080p|720p|2160p|480p)\b|"
    r"\b(?:S\d{1,2}[._ -]?E\d{1,3}|\d{1,2}x\d{2})\b.*?\b(?:1080p|720p|2160p|480p|WEB-DL|WEBRip|BluRay|BDRip|HDTV|x264|x265|HEVC)\b|"
    r"\s-\s\d{1,3}\s*(?:\(|\[).*?\b(?:1080p|720p|2160p|480p)\b)"
)
_VIDEO_RIP_CLAIM_CG_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:1080p|720p|2160p|480p|576p|4k)\b.*?\b(?:web-?dl|webrip|bluray|bdrip|brrip|hdrip|hdtv|hdcam|camrip|dvdrip|x264|x265|h\.?264|h\.?265|hevc|av1|xvid|divx)\b|"
    r"\b(?:web-?dl|webrip|bdrip|brrip|hdrip|hdtv|hdcam|camrip|dvdrip)\b(?:.*?\b(?:x264|x265|h\.?264|h\.?265|hevc|av1|xvid|divx|aac|ac3|ddp?5\.?1))?|"
    r"\b(?:19\d\d|20\d\d|s\d{1,2}[._ -]?e\d{1,3}|\d{1,2}x\d{2})\b.*?\b(?:x264|x265|h\.?264|h\.?265|hevc|av1|xvid|divx)\b"
    r")"
)
_OPTICAL_DISC_TITLE_CG_RE = re.compile(
    r"(?i)\b(?:dvd(?:[59]|r|-r|-rom)?|bd(?:25|50|66|100|mv|rom|-r)|blu-?ray[\s._-]*disc|full[\s._-]*bd|untouched|remux|pal|ntsc|iso|udf|sacd|dvd-?audio|3d[\s._-]*iso)\b"
)
_BENIGN_ZIP_CG_RE = re.compile(
    r"(?i)\b(?:batch|complete|season[ ._-]*\d+|s\d{1,2}[ ._-]*complete|e?\d{1,3}[ ._-]*[~-][ ._-]*e?\d{1,3}|시즌\s*\d+|완결|全集|合集|"
    r"ost|soundtrack|flac|mp3|wav|lossless|album|discography|artbook|scans?|wallpapers?|fonts?|subs?|subtitles?|"
    r"cbz|cbr|comic|manga|magazine|ebook|epub|pdf|audiobook|m4b|templates?|presets?|samples?|stems)\b"
)

_SINGLE_ARCHIVE_TRAP_EXTS = {".zip", ".zipx", ".rar", ".7z", ".ace", ".cab"}
_DISK_IMAGE_TRAP_EXTS = {".iso", ".img", ".vhd", ".vhdx", ".udf", ".dmg"}
_VIDEO_MEDIA_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm", ".flv", ".m4v", ".mpg", ".mpeg", ".ts", ".m2ts", ".vob", ".ogv", ".3gp"}
_BENIGN_COURSE_EXTS = _VIDEO_MEDIA_EXTS | {
    ".mp3", ".flac", ".m4a", ".m4b", ".aac", ".ogg", ".wav",
    ".pdf", ".epub", ".mobi", ".djvu", ".cbz", ".cbr",
    ".srt", ".ass", ".ssa", ".vtt", ".sub",
    ".gp3", ".gp4", ".gp5", ".gpx", ".ptb", ".ipynb", ".md", ".html",
}

_ARCHIVE_EXTENSIONS = {".zip", ".zipx", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".iso", ".img"}

_DISGUISED_MEDIA_STEMS = (
    ".mp4.", ".mkv.", ".avi.", ".mov.", ".wmv.", ".webm.", ".flv.", ".m4v.",
    ".mp4 ", ".mkv ", ".avi ", ".mov ",
    ".mp3.", ".flac.", ".wav.", ".m4a.", ".ogg.", ".aac.", ".opus.",
    ".pdf.", ".epub.", ".mobi.", ".cbz.", ".cbr.", ".djvu.",
    ".jpg.", ".jpeg.", ".png.", ".gif.", ".webp.", ".srt.", ".ass.", ".sub.",
)


def _clean_filename_for_ext(raw_name: str) -> str:
    s = raw_name.lower().strip().rstrip(". \t\r\n\x00")
    s = _SAMPLE_SUFFIX_RE.sub("", s).rstrip(". \t\r\n\x00")
    return s


def _match_malware_ext(clean_lower: str) -> str | None:
    dot = clean_lower.rfind(".")
    if dot == -1:
        return None
    ext = clean_lower[dot:]
    return ext if ext in _MALWARE_EXTENSIONS else None


def _is_disk_image_mount_trap_cg(title_raw: str) -> bool:
    stem = re.sub(r"\.(?:iso|img|vhd|vhdx|udf|dmg)$", "", title_raw.strip(), flags=re.IGNORECASE)
    if _OPTICAL_DISC_TITLE_CG_RE.search(stem) or _BENIGN_ZIP_CG_RE.search(stem):
        return False
    return bool(_VIDEO_RIP_CLAIM_CG_RE.search(stem) or _ANIME_OR_EP_ZIP_CG_RE.search(stem))


# ---------------------------------------------------------------------------
# CSAM & Abuse Guard Implementation
# ---------------------------------------------------------------------------

class CSAMGuard:
    """Comprehensive content safety guard enforcing CSAM, abuse, and malware blocking."""

    def __init__(self, tombstone_store: TombstoneStore | None = None) -> None:
        self.tombstone_store = tombstone_store or TombstoneStore()

    def inspect_text(self, text: str) -> ContentFilterResult:
        """Inspect a single string (title, file path, tag) for CSAM and harmful content."""
        if not text:
            return ContentFilterResult(
                is_safe=True,
                violation_type=None,
                reason=None,
                matched_pattern=None,
                risk_score=0.0,
                sanitized_title="",
            )

        sanitized = sanitize_title(text)
        normalized = normalize_text(sanitized)
        # Preserve standalone "css" (Cascading Style Sheets) before collapsing single-char runs or stretching
        norm_css_safe = re.sub(r"\bcss\b", "css_style_token", normalized)
        de_spaced = _SINGLE_CHAR_RUN_RE.sub(r"\1", norm_css_safe)
        # Collapse arbitrary character stretching while preserving valid English double letters in target stems
        de_stretched = re.sub(r"(.)\1+", r"\1", de_spaced).replace("cs_style_token", "css")
        de_spaced = de_spaced.replace("css_style_token", "css")
        for raw_stem, canonical_stem in (
            ("ten", "teen"),
            ("todler", "toddler"),
            ("snuf", "snuff"),
            ("masacre", "massacre"),
            ("kil", "kill"),
            ("bathrom", "bathroom"),
        ):
            de_stretched = de_stretched.replace(raw_stem, canonical_stem)
        candidates = tuple(dict.fromkeys((normalized, de_spaced, de_stretched)))

        for cand in candidates:
            # 1. Direct CSAM patterns
            for pat in _CSAM_DIRECT_PATTERNS:
                m = pat.search(cand)
                if m:
                    return ContentFilterResult(
                        is_safe=False,
                        violation_type=CSAMViolationType.CSAM_EXPLOITATION,
                        reason=f"Matched direct CSAM pattern: '{m.group()}'",
                        matched_pattern=pat.pattern,
                        risk_score=1.0,
                        sanitized_title=sanitized,
                    )

            # 2. Non-consensual exploitation patterns
            for pat in _NONCONSENSUAL_PATTERNS:
                m = pat.search(cand)
                if m:
                    return ContentFilterResult(
                        is_safe=False,
                        violation_type=CSAMViolationType.NONCONSENSUAL_ABUSE,
                        reason=f"Matched non-consensual exploitation pattern: '{m.group()}'",
                        matched_pattern=pat.pattern,
                        risk_score=0.98,
                        sanitized_title=sanitized,
                    )

            # 3. Extreme violence & gore patterns
            for pat in _EXTREME_VIOLENCE_PATTERNS:
                m = pat.search(cand)
                if m:
                    return ContentFilterResult(
                        is_safe=False,
                        violation_type=CSAMViolationType.EXTREME_VIOLENCE_GORE,
                        reason=f"Matched extreme violence pattern: '{m.group()}'",
                        matched_pattern=pat.pattern,
                        risk_score=0.95,
                        sanitized_title=sanitized,
                    )

        return ContentFilterResult(
            is_safe=True,
            violation_type=None,
            reason=None,
            matched_pattern=None,
            risk_score=0.0,
            sanitized_title=sanitized,
        )

    def inspect_torrent(
        self,
        infohash: str,
        title: str,
        file_list: Sequence[str | dict[str, Any]] | None = None,
        tags: Sequence[str] | None = None,
        category: str | None = None,
    ) -> ContentFilterResult:
        """Structural pre-ingest inspection over entire torrent metadata.

        Verifies:
          1. Infohash is not on the Ed25519-signed blinded tombstone purge list.
          2. Torrent title contains no CSAM/abuse keywords or passkey leakage.
          3. None of the internal files contain CSAM tokens.
          4. No disguised executable payloads, RTLO spoofing, mount traps, or archive traps exist.
          5. Tags contain no prohibited content.
        """
        clean_infohash = str(infohash).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", clean_infohash):
            return ContentFilterResult(
                is_safe=False,
                violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                reason="Malformed non-hexadecimal 40-char infohash rejected",
                matched_pattern=clean_infohash[:40],
                risk_score=1.0,
                sanitized_title=sanitize_title(title),
            )

        # 1. Tombstone purge check
        if self.tombstone_store.is_tombstoned(clean_infohash):
            return ContentFilterResult(
                is_safe=False,
                violation_type=CSAMViolationType.TOMBSTONED_HASH,
                reason=f"Infohash {clean_infohash} is present on the signed revocation tombstone list",
                matched_pattern=clean_infohash,
                risk_score=1.0,
                sanitized_title=sanitize_title(title),
            )

        # 2. Inspect title for CSAM / non-consensual / extreme violence
        res = self.inspect_text(title)
        if not res.is_safe:
            return res

        # 3. Inspect tags
        if tags:
            for tag in tags:
                tag_res = self.inspect_text(tag)
                if not tag_res.is_safe:
                    return tag_res

        # 4. Inspect file list text for CSAM / abuse first (preserves violation_type priority)
        parsed_files: list[tuple[str, str, int]] = []
        if file_list:
            for f in file_list:
                fname = f if isinstance(f, str) else str(f.get("path") or f.get("p") or f.get("name") or "")
                fsize = int((f.get("size") or f.get("s") or 0) if isinstance(f, dict) else 0)
                f_res = self.inspect_text(fname)
                if not f_res.is_safe:
                    return f_res
                parsed_files.append((fname, _clean_filename_for_ext(fname), fsize))

        # 5. Check Unicode Right-to-Left Override (RTLO) extension spoofing in title or files
        if _RTLO_SPOOF_RE.search(title) or any(_RTLO_SPOOF_RE.search(fn) for fn, _, _ in parsed_files):
            return ContentFilterResult(
                is_safe=False,
                violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                reason=f"Blocked Unicode RTLO extension spoofing character in torrent: '{title}'",
                matched_pattern="U+202E_RTLO",
                risk_score=1.0,
                sanitized_title=res.sanitized_title,
            )

        title_clean = _clean_filename_for_ext(title)

        # 6. Universal malware/honeypot lure patterns (blocked across ALL categories)
        m_lure = _MALWARE_LURE_RE.search(title_clean)
        if m_lure:
            return ContentFilterResult(
                is_safe=False,
                violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                reason=f"Blocked explicit malware/honeypot lure in title: '{title}'",
                matched_pattern=m_lure.group(0),
                risk_score=0.99,
                sanitized_title=res.sanitized_title,
            )
        for fname, f_lower, _ in parsed_files:
            m_flure = _MALWARE_LURE_RE.search(f_lower)
            if m_flure:
                return ContentFilterResult(
                    is_safe=False,
                    violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                    reason=f"Blocked explicit malware/honeypot lure in file: '{fname}'",
                    matched_pattern=m_flure.group(0),
                    risk_score=0.99,
                    sanitized_title=res.sanitized_title,
                )

        title_claims_video = bool(_ANIME_OR_EP_ZIP_CG_RE.search(title_clean) or _VIDEO_RIP_CLAIM_CG_RE.search(title_clean))

        # 7. Title-level disguised media double-extension & unconditional dropper extension checks
        t_ext = _match_malware_ext(title_clean)
        if t_ext is not None:
            if (
                any(stem in title_clean for stem in _DISGUISED_MEDIA_STEMS)
                or t_ext in _ALWAYS_MALWARE_EXTENSIONS
                or (title_claims_video and t_ext in (".one", ".xll", ".xlam"))
            ):
                return ContentFilterResult(
                    is_safe=False,
                    violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                    reason=f"Disguised executable or high-risk dropper in torrent title: '{title}'",
                    matched_pattern=t_ext,
                    risk_score=0.99,
                    sanitized_title=res.sanitized_title,
                )

        cat_lower = (category or "").lower().strip()
        is_strict_media_category = cat_lower in {
            "movie", "tv_show", "music", "anime", "book", "ebook", "audiobook", "comic", "xxx"
        }
        is_course_category = cat_lower == "course"

        has_large_real_video = False
        has_any_video = False
        has_bdmv_structure = "bdmv" in title_clean
        has_renpy_structure = False
        benign_course_files = 0
        benign_course_bytes = 0
        has_password_companion = False
        image_file_count = 0
        image_total_bytes = 0

        for _, f_lower, fsize in parsed_files:
            f_posix = f_lower.replace("\\", "/")
            if "/bdmv/" in f_posix or f_posix.startswith("bdmv/") or f_posix.endswith((".m2ts", ".bdmv", ".mpls")):
                has_bdmv_structure = True
            if f_posix.endswith((".rpa", ".rpyb")) and fsize >= 65536:
                has_renpy_structure = True
            if _PASSWORD_COMPANION_RE.search(f_posix):
                has_password_companion = True
            dot = f_posix.rfind(".")
            f_ext = f_posix[dot:] if dot != -1 else ""
            if f_ext in _BENIGN_COURSE_EXTS and fsize >= 65536:
                benign_course_files += 1
                benign_course_bytes += fsize
            if f_ext in (".jpg", ".jpeg", ".png", ".webp") and fsize > 0:
                image_file_count += 1
                image_total_bytes += fsize
            if f_ext in _VIDEO_MEDIA_EXTS:
                has_any_video = True
                if fsize >= 100_000_000:
                    has_large_real_video = True
        has_real_image_set = image_file_count >= 5 and image_total_bytes >= 5_000_000
        has_real_course_media = benign_course_files >= 3 and benign_course_bytes >= 50_000_000

        # Multi-lesson courses (>= 5 files, >= 3 sized lesson files, >= 50 MB total lesson media) legitimately
        # include .js, .ps1, .bat, or interactive player .exe files; zero-byte or tiny decoys are still checked.
        check_media_executables = is_strict_media_category or (
            is_course_category and (len(parsed_files) < 5 or not has_real_course_media)
        )

        # Contextual codec-pack lure check: blocks codec-pack installers disguised as or bundled with media
        # while sparing legitimate standalone codec-pack software utilities.
        if check_media_executables or has_any_video or title_claims_video:
            m_clure = _MEDIA_CODEC_LURE_RE.search(title_clean)
            if m_clure and (has_any_video or title_claims_video):
                return ContentFilterResult(
                    is_safe=False,
                    violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                    reason=f"Blocked media codec-pack lure in title: '{title}'",
                    matched_pattern=m_clure.group(0),
                    risk_score=0.99,
                    sanitized_title=res.sanitized_title,
                )
            for fname, f_lower, _ in parsed_files:
                m_fclure = _MEDIA_CODEC_LURE_RE.search(f_lower)
                if m_fclure:
                    return ContentFilterResult(
                        is_safe=False,
                        violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                        reason=f"Blocked explicit malware/honeypot lure in file: '{fname}'",
                        matched_pattern=m_fclure.group(0),
                        risk_score=0.99,
                        sanitized_title=res.sanitized_title,
                    )

        # 8. Cold / bare-title checks when file_list is empty
        if not parsed_files and (check_media_executables or cat_lower in ("unknown", "")):
            if t_ext is not None and not _BENIGN_ZIP_CG_RE.search(title_clean):
                return ContentFilterResult(
                    is_safe=False,
                    violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                    reason=f"Disguised executable or script extension in media title: '{title}'",
                    matched_pattern=t_ext,
                    risk_score=0.99,
                    sanitized_title=res.sanitized_title,
                )
            dot_t = title_clean.rfind(".")
            bare_ext = title_clean[dot_t:] if dot_t != -1 else ""
            if bare_ext in _SINGLE_ARCHIVE_TRAP_EXTS and not _BENIGN_ZIP_CG_RE.search(title_clean):
                if title_claims_video:
                    return ContentFilterResult(
                        is_safe=False,
                        violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                        reason=f"Single-archive media malware trap in title: '{title}'",
                        matched_pattern=bare_ext,
                        risk_score=0.99,
                        sanitized_title=res.sanitized_title,
                    )
            if bare_ext in _DISK_IMAGE_TRAP_EXTS and (
                any(stem in title_clean for stem in _DISGUISED_MEDIA_STEMS) or _is_disk_image_mount_trap_cg(title_clean)
            ):
                return ContentFilterResult(
                    is_safe=False,
                    violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                    reason=f"Disk image mount malware trap ({bare_ext}) in media title: '{title}'",
                    matched_pattern=bare_ext,
                    risk_score=0.99,
                    sanitized_title=res.sanitized_title,
                )

        # 9. File-list malware, mount trap, and archive trap inspection
        if parsed_files:
            for fname, f_lower, fsize in parsed_files:
                f_posix = f_lower.replace("\\", "/")
                base_lower = f_posix.rsplit("/", 1)[-1]
                if (
                    base_lower in ("do_not_mirror.exe", "rarbg_do_not_mirror.exe", "rarbg.com_do_not_mirror.exe")
                    and (0 < fsize <= 512)
                    and (has_large_real_video or has_real_image_set)
                ):
                    continue
                if (
                    _RENPY_SAY_VBS_CG_RE.search(f_posix)
                    and not is_strict_media_category
                    and not title_claims_video
                    and not has_any_video
                    and has_renpy_structure
                    and (0 < fsize <= 4096)
                ):
                    continue
                is_disguised = any(stem in f_lower for stem in _DISGUISED_MEDIA_STEMS)
                dot_f = f_lower.rfind(".")
                raw_ext = f_lower[dot_f:] if dot_f != -1 else ""
                if is_disguised and raw_ext in _DISK_IMAGE_TRAP_EXTS:
                    return ContentFilterResult(
                        is_safe=False,
                        violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                        reason=f"Disguised disk-image payload found in torrent: '{fname}'",
                        matched_pattern=raw_ext,
                        risk_score=0.99,
                        sanitized_title=res.sanitized_title,
                    )
                ext = _match_malware_ext(f_lower)
                if ext is not None:
                    # Exempt legitimate Internet Archive / Gutenberg ASR (.asr.js) and J2ME reader (.qioo.jar) sidecars when not disguised
                    if not is_disguised and f_lower.endswith((".asr.js", ".qioo.jar")):
                        continue
                    # Exempt legitimate Blu-ray BD-J Java menu JARs (BDMV/JAR/00000.jar)
                    if ext == ".jar" and has_bdmv_structure and _BDMV_JAR_PATH_RE.search(f_posix):
                        continue
                    # Exempt known small tracker bookmark (.url) only when not disguised and real media is present
                    if (
                        ext == ".url"
                        and not is_disguised
                        and (0 < fsize <= 2048)
                        and _BENIGN_BOOKMARK_URL_RE.search(base_lower)
                        and (has_large_real_video or has_real_image_set or has_real_course_media)
                    ):
                        continue
                    if (
                        check_media_executables
                        or is_disguised
                        or ext in _ALWAYS_MALWARE_EXTENSIONS
                        or (title_claims_video and ext in (".one", ".xll", ".xlam"))
                    ):
                        return ContentFilterResult(
                            is_safe=False,
                            violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                            reason=f"Disguised executable or dropper found in media torrent: '{fname}'",
                            matched_pattern=ext,
                            risk_score=0.99,
                            sanitized_title=res.sanitized_title,
                        )

            if check_media_executables and not has_any_video and len(parsed_files) <= 3:
                # Optical / Disk Image Mount Trap (.iso, .img, .vhd, .vhdx, .udf, .dmg)
                for fname, f_lower, _ in parsed_files:
                    dot = f_lower.rfind(".")
                    f_ext = f_lower[dot:] if dot != -1 else ""
                    if f_ext in _DISK_IMAGE_TRAP_EXTS and _is_disk_image_mount_trap_cg(title_clean):
                        return ContentFilterResult(
                            is_safe=False,
                            violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                            reason=f"Disk image mount malware trap ({f_ext}) in stream/rip-claiming media torrent: '{fname}'",
                            matched_pattern=f_ext,
                            risk_score=0.99,
                            sanitized_title=res.sanitized_title,
                        )

                # Single-archive media trap (.zip, .zipx, .rar, .7z, .ace, .cab)
                if not _BENIGN_ZIP_CG_RE.search(title_clean):
                    arch_matches: list[tuple[str, str]] = []
                    for fname, f_lower, _ in parsed_files:
                        dot = f_lower.rfind(".")
                        f_ext = f_lower[dot:] if dot != -1 else ""
                        if f_ext in _SINGLE_ARCHIVE_TRAP_EXTS:
                            arch_matches.append((fname, f_ext))
                    if len(arch_matches) == 1:
                        arch_fname, arch_ext = arch_matches[0]
                        if has_password_companion or (
                            len(parsed_files) <= 2
                            and cat_lower in ("movie", "tv_show", "anime", "unknown", "")
                            and (_ANIME_OR_EP_ZIP_CG_RE.search(title_clean) or _VIDEO_RIP_CLAIM_CG_RE.search(title_clean))
                        ):
                            return ContentFilterResult(
                                is_safe=False,
                                violation_type=CSAMViolationType.MALWARE_PAYLOAD,
                                reason=f"Single-archive episode/movie malware trap in media torrent: '{arch_fname}'",
                                matched_pattern=arch_ext,
                                risk_score=0.99,
                                sanitized_title=res.sanitized_title,
                            )

        return ContentFilterResult(
            is_safe=True,
            violation_type=None,
            reason=None,
            matched_pattern=None,
            risk_score=0.0,
            sanitized_title=res.sanitized_title,
        )


# ---------------------------------------------------------------------------
# Armiger Conformal Safety Gate Interface
# ---------------------------------------------------------------------------

class ArmigerSafetyGate:
    """Evaluates Mondrian conformal thresholds to prevent malware injection and adult leakage."""

    def __init__(
        self,
        adult_threshold: float = 0.5451,
        junk_threshold: float = 0.1010,
        scope_threshold: float = 0.4770,
    ) -> None:
        self.adult_threshold = adult_threshold
        self.junk_threshold = junk_threshold
        self.scope_threshold = scope_threshold

    def evaluate(
        self,
        p_adult: float,
        p_junk: float,
        p_excluded_scope: float,
        target_category: str,
    ) -> tuple[bool, str, CSAMViolationType | None]:
        """Validate safety gates for a torrent candidate.

        Rules:
          - If p_junk >= junk_threshold: Drop as malware/phishing trap.
          - If p_excluded_scope >= scope_threshold: Drop as out-of-scope.
          - If p_adult >= adult_threshold: Must ONLY be routed to 'xxx' category.
            If target_category is not 'xxx', reject to maintain strict mainstream isolation.
        """
        for pname, pval in (("p_adult", p_adult), ("p_junk", p_junk), ("p_excluded_scope", p_excluded_scope)):
            if isinstance(pval, bool) or not isinstance(pval, (int, float)) or not math.isfinite(pval) or pval < 0.0 or pval > 1.0:
                return False, f"invalid_probability_{pname}:{pval}", CSAMViolationType.SAFETY_GATE_JUNK

        if p_junk >= self.junk_threshold:
            return False, f"junk_probability_exceeded:{p_junk:.4f}>={self.junk_threshold:.4f}", CSAMViolationType.SAFETY_GATE_JUNK

        if p_excluded_scope >= self.scope_threshold:
            return False, f"out_of_scope_probability_exceeded:{p_excluded_scope:.4f}>={self.scope_threshold:.4f}", CSAMViolationType.SAFETY_GATE_OUT_OF_SCOPE

        is_adult = bool(p_adult >= self.adult_threshold)
        if is_adult and target_category != "xxx":
            return False, f"unisolated_adult_leakage:{p_adult:.4f}>={self.adult_threshold:.4f}_into_{target_category}", CSAMViolationType.UNISOLATED_ADULT

        if not is_adult and target_category == "xxx":
            # Target is xxx but confidence of adult is below threshold; allow or flag
            pass

        return True, "passed", None




# ---------------------------------------------------------------------------
# Post-Quantum Primitives (ML-KEM-768 FIPS 203 & WOTS+-SHAKE256 FIPS 205/RFC 8391)
# ---------------------------------------------------------------------------

class MLKEM768:
    """Self-contained NIST FIPS 203 ML-KEM-768 (Module-LWE Post-Quantum KEM) implementation."""

    Q = 3329
    N = 256
    K = 3
    EK_BYTES = 1184
    DK_BYTES = 2400
    CT_BYTES = 1088
    SS_BYTES = 32

    ZETAS = (
        1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746,
        296, 2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821,
        289, 331, 3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915,
        2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910,
        17, 2761, 583, 2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050,
        1703, 1651, 2789, 1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641,
        1584, 2298, 2037, 3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594,
        2804, 1092, 403, 1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154,
    )
    GAMMAS = tuple(pow(17, 2 * int(f"{i:07b}"[::-1], 2) + 1, 3329) for i in range(128))
    _CBD2_LO = tuple(((0, 1, 1, 2)[b & 3] - (0, 1, 1, 2)[(b >> 2) & 3]) % 3329 for b in range(256))
    _CBD2_HI = tuple(((0, 1, 1, 2)[(b >> 4) & 3] - (0, 1, 1, 2)[(b >> 6) & 3]) % 3329 for b in range(256))

    _NTT_STEPS: List[List[Tuple[int, int, int, int]]] = []
    _INTT_STEPS: List[List[Tuple[int, int, int, int]]] = []

    @classmethod
    def _init_tables(cls) -> None:
        if cls._NTT_STEPS:
            return
        k = 1
        length = 128
        while length >= 2:
            layer = []
            for start in range(0, 256, 2 * length):
                zeta = cls.ZETAS[k]
                k += 1
                layer.append((zeta, start, start + length, length))
            cls._NTT_STEPS.append(layer)
            length >>= 1

        k = 127
        length = 2
        while length <= 128:
            layer = []
            for start in range(0, 256, 2 * length):
                zeta = cls.ZETAS[k]
                k -= 1
                layer.append((zeta, start, start + length, length))
            cls._INTT_STEPS.append(layer)
            length <<= 1

    @classmethod
    def ntt(cls, f: List[int]) -> List[int]:
        p = list(f)
        q = 3329
        for layer in cls._NTT_STEPS:
            for zeta, s, e, l in layer:
                for j in range(s, e):
                    t = (zeta * p[j + l]) % q
                    p[j + l] = (p[j] - t) % q
                    p[j] = (p[j] + t) % q
        return p

    @classmethod
    def intt(cls, f: List[int]) -> List[int]:
        p = list(f)
        q = 3329
        for layer in cls._INTT_STEPS:
            for zeta, s, e, l in layer:
                for j in range(s, e):
                    t = p[j]
                    p[j] = (t + p[j + l]) % q
                    p[j + l] = (zeta * (p[j + l] - t)) % q
        return [(x * 3303) % q for x in p]

    @classmethod
    def basemul_acc(cls, a_vec: List[List[int]], b_vec: List[List[int]]) -> List[int]:
        res = [0] * 256
        q = 3329
        gammas = cls.GAMMAS
        for a, b in zip(a_vec, b_vec):
            for i in range(128):
                idx = i << 1
                a0, a1 = a[idx], a[idx + 1]
                b0, b1 = b[idx], b[idx + 1]
                res[idx] = (res[idx] + a0 * b0 + a1 * b1 * gammas[i]) % q
                res[idx + 1] = (res[idx + 1] + a0 * b1 + a1 * b0) % q
        return res

    @staticmethod
    def sample_ntt(rho: bytes, j: int, i: int) -> List[int]:
        stream = hashlib.shake_128(rho + bytes([j, i])).digest(672)
        a: List[int] = []
        idx = 0
        while len(a) < 256:
            b0, b1, b2 = stream[idx], stream[idx + 1], stream[idx + 2]
            idx += 3
            d1 = b0 + ((b1 & 0x0F) << 8)
            d2 = (b1 >> 4) + (b2 << 4)
            if d1 < 3329:
                a.append(d1)
            if d2 < 3329 and len(a) < 256:
                a.append(d2)
        return a

    @classmethod
    def sample_cbd2(cls, prf_bytes: bytes) -> List[int]:
        f = [0] * 256
        lo = cls._CBD2_LO
        hi = cls._CBD2_HI
        for i, b in enumerate(prf_bytes):
            idx = i << 1
            f[idx] = lo[b]
            f[idx + 1] = hi[b]
        return f

    @staticmethod
    def encode_12(f: List[int]) -> bytes:
        out = bytearray(384)
        for i in range(128):
            a = f[2 * i] % 3329
            b = f[2 * i + 1] % 3329
            idx = 3 * i
            out[idx] = a & 0xFF
            out[idx + 1] = ((a >> 8) & 0x0F) | ((b & 0x0F) << 4)
            out[idx + 2] = (b >> 4) & 0xFF
        return bytes(out)

    @staticmethod
    def decode_12(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(128):
            idx = 3 * i
            b0, b1, b2 = buf[idx], buf[idx + 1], buf[idx + 2]
            f[2 * i] = (b0 + ((b1 & 0x0F) << 8)) % 3329
            f[2 * i + 1] = ((b1 >> 4) + (b2 << 4)) % 3329
        return f

    @staticmethod
    def compress_encode_10(f: List[int]) -> bytes:
        out = bytearray(320)
        for i in range(64):
            c0 = ((f[4 * i] << 10) + 1664) // 3329 & 0x3FF
            c1 = ((f[4 * i + 1] << 10) + 1664) // 3329 & 0x3FF
            c2 = ((f[4 * i + 2] << 10) + 1664) // 3329 & 0x3FF
            c3 = ((f[4 * i + 3] << 10) + 1664) // 3329 & 0x3FF
            idx = 5 * i
            out[idx] = c0 & 0xFF
            out[idx + 1] = ((c0 >> 8) & 0x03) | ((c1 & 0x3F) << 2)
            out[idx + 2] = ((c1 >> 6) & 0x0F) | ((c2 & 0x0F) << 4)
            out[idx + 3] = ((c2 >> 4) & 0x3F) | ((c3 & 0x03) << 6)
            out[idx + 4] = (c3 >> 2) & 0xFF
        return bytes(out)

    @staticmethod
    def decode_decompress_10(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(64):
            idx = 5 * i
            b0, b1, b2, b3, b4 = buf[idx], buf[idx + 1], buf[idx + 2], buf[idx + 3], buf[idx + 4]
            c0 = b0 | ((b1 & 0x03) << 8)
            c1 = (b1 >> 2) | ((b2 & 0x0F) << 6)
            c2 = (b2 >> 4) | ((b3 & 0x3F) << 4)
            c3 = (b3 >> 6) | (b4 << 2)
            f[4 * i] = (c0 * 3329 + 512) >> 10
            f[4 * i + 1] = (c1 * 3329 + 512) >> 10
            f[4 * i + 2] = (c2 * 3329 + 512) >> 10
            f[4 * i + 3] = (c3 * 3329 + 512) >> 10
        return f

    @staticmethod
    def compress_encode_4(f: List[int]) -> bytes:
        out = bytearray(128)
        for i in range(128):
            c0 = ((f[2 * i] << 4) + 1664) // 3329 & 0x0F
            c1 = ((f[2 * i + 1] << 4) + 1664) // 3329 & 0x0F
            out[i] = c0 | (c1 << 4)
        return bytes(out)

    @staticmethod
    def decode_decompress_4(buf: bytes) -> List[int]:
        f = [0] * 256
        for i in range(128):
            b = buf[i]
            f[2 * i] = ((b & 0x0F) * 3329 + 8) >> 4
            f[2 * i + 1] = (((b >> 4) & 0x0F) * 3329 + 8) >> 4
        return f

    @staticmethod
    def encode_1(m: bytes) -> List[int]:
        f = [0] * 256
        for i in range(32):
            b = m[i]
            base = i << 3
            for j in range(8):
                f[base + j] = 1665 if ((b >> j) & 1) else 0
        return f

    @staticmethod
    def decode_1(f: List[int]) -> bytes:
        out = bytearray(32)
        for i in range(32):
            byte_val = 0
            base = i << 3
            for j in range(8):
                c = (((f[base + j] << 1) + 1664) // 3329) & 1
                byte_val |= (c << j)
            out[i] = byte_val
        return bytes(out)

    @classmethod
    @functools.lru_cache(maxsize=64)
    def _expand_ek(cls, ek: bytes) -> Tuple[List[List[int]], List[List[List[int]]], bytes]:
        cls._init_tables()
        t_hat = [cls.decode_12(ek[384 * i : 384 * (i + 1)]) for i in range(3)]
        rho = ek[1152:1184]
        a_hat_t = [[cls.sample_ntt(rho, i, j) for j in range(3)] for i in range(3)]
        ek_hash = hashlib.sha3_256(ek).digest()
        return t_hat, a_hat_t, ek_hash

    @classmethod
    @functools.lru_cache(maxsize=64)
    def _expand_dk(cls, dk: bytes) -> Tuple[List[List[int]], bytes, bytes, bytes, List[List[int]], List[List[List[int]]]]:
        cls._init_tables()
        s_hat = [cls.decode_12(dk[384 * i : 384 * (i + 1)]) for i in range(3)]
        ek = dk[1152:2336]
        h = dk[2336:2368]
        z = dk[2368:2400]
        t_hat, a_hat_t, _ = cls._expand_ek(ek)
        return s_hat, ek, h, z, t_hat, a_hat_t

    @classmethod
    def keygen(cls, seed_d: Optional[bytes] = None, seed_z: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        cls._init_tables()
        d = seed_d if seed_d is not None else os.urandom(32)
        z = seed_z if seed_z is not None else os.urandom(32)
        g = hashlib.sha3_512(d + bytes([3])).digest()
        rho, sigma = g[:32], g[32:]
        a_hat = [[cls.sample_ntt(rho, j, i) for j in range(3)] for i in range(3)]
        s_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(sigma + bytes([i])).digest(128))) for i in range(3)]
        e_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(sigma + bytes([i + 3])).digest(128))) for i in range(3)]
        t_hat = []
        for i in range(3):
            ti = cls.basemul_acc(a_hat[i], s_hat)
            t_hat.append([(a + b) % 3329 for a, b in zip(ti, e_hat[i])])
        ek_pke = b"".join(cls.encode_12(p) for p in t_hat) + rho
        dk_pke = b"".join(cls.encode_12(p) for p in s_hat)
        h = hashlib.sha3_256(ek_pke).digest()
        return ek_pke, dk_pke + ek_pke + h + z

    @classmethod
    def _pke_encrypt_expanded(
        cls,
        t_hat: List[List[int]],
        a_hat_t: List[List[List[int]]],
        m: bytes,
        r: bytes,
    ) -> bytes:
        r_hat = [cls.ntt(cls.sample_cbd2(hashlib.shake_256(r + bytes([i])).digest(128))) for i in range(3)]
        e1 = [cls.sample_cbd2(hashlib.shake_256(r + bytes([i + 3])).digest(128)) for i in range(3)]
        e2 = cls.sample_cbd2(hashlib.shake_256(r + bytes([6])).digest(128))
        u = []
        for i in range(3):
            ui = cls.intt(cls.basemul_acc(a_hat_t[i], r_hat))
            u.append([(a + b) % 3329 for a, b in zip(ui, e1[i])])
        mu = cls.encode_1(m)
        v_ntt = cls.basemul_acc(t_hat, r_hat)
        v_poly = cls.intt(v_ntt)
        v = [(a + b + c) % 3329 for a, b, c in zip(v_poly, e2, mu)]
        return b"".join(cls.compress_encode_10(p) for p in u) + cls.compress_encode_4(v)

    @classmethod
    def encaps(cls, ek: bytes, seed_m: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        if len(ek) != cls.EK_BYTES:
            raise ValueError(f"ML-KEM-768 ek must be {cls.EK_BYTES} bytes")
        if ek[:1152] == b"\x00" * 1152 or ek[1152:] == b"\x00" * 32:
            raise ValueError("ML-KEM-768 degenerate all-zero encapsulation key rejected")
        t_hat, a_hat_t, ek_hash = cls._expand_ek(ek)
        if b"".join(cls.encode_12(p) for p in t_hat) != ek[:1152]:
            raise ValueError("ML-KEM-768 non-canonical encapsulation key (coefficients >= q)")
        m = seed_m if seed_m is not None else os.urandom(32)
        kr = hashlib.sha3_512(m + ek_hash).digest()
        k_shared, r = kr[:32], kr[32:]
        c = cls._pke_encrypt_expanded(t_hat, a_hat_t, m, r)
        return c, k_shared

    @classmethod
    def decaps(cls, dk: bytes, c: bytes) -> bytes:
        if len(dk) != cls.DK_BYTES or len(c) != cls.CT_BYTES:
            raise ValueError("ML-KEM-768 invalid dk or ciphertext length")
        s_hat, _ek, h, z, t_hat, a_hat_t = cls._expand_dk(dk)
        u = [cls.decode_decompress_10(c[320 * i : 320 * (i + 1)]) for i in range(3)]
        v = cls.decode_decompress_4(c[960:1088])
        u_hat = [cls.ntt(p) for p in u]
        s_u = cls.intt(cls.basemul_acc(s_hat, u_hat))
        w = [(vi - sui) % 3329 for vi, sui in zip(v, s_u)]
        m_prime = cls.decode_1(w)
        kr_prime = hashlib.sha3_512(m_prime + h).digest()
        k_prime, r_prime = kr_prime[:32], kr_prime[32:]
        k_bar = hashlib.shake_256(z + c).digest(32)
        c_prime = cls._pke_encrypt_expanded(t_hat, a_hat_t, m_prime, r_prime)
        if hmac.compare_digest(c, c_prime):
            return k_prime
        return k_bar


MLKEM768._init_tables()


class WOTSPlusSHAKE256:
    """Post-Quantum Winternitz Hash-Based Signature over SHAKE-256 with Merkle One-Time Leaf Trees (AWM2)."""

    N = 32
    W = 16
    LEN1 = 64
    LEN2 = 3
    CHAINS = 67
    PRIV_BYTES = 64
    PUB_BYTES = 32
    SIG_BYTES = 2176
    TREE_HEIGHT = 4
    TREE_LEAVES = 16
    USABLE_LEAVES_PER_SUBTREE = 15
    MAX_SUBTREE_HOPS = 64
    MAX_SUPPORTED_LEAVES = MAX_SUBTREE_HOPS * USABLE_LEAVES_PER_SUBTREE
    AWM2_MAGIC = b"AWM2"
    HOP_BYTES = 4 + 32 + 32 + 128 + 2176  # subtree_id(4) + merkle_root(32) + hop_leaf_pub(32) + auth_path(128) + cert_sig(2176)

    _LOCK = threading.RLock()
    _STATE_DIR: Optional[pathlib.Path] = None
    _SEAL_DIR: Optional[pathlib.Path] = None
    _REQUIRE_EXPLICIT_STATE_DIR: bool = False
    _REQUIRE_EXISTING_STATE: bool = False
    _IN_MEM_NEXT_LEAF: Dict[str, int] = {}
    _IN_MEM_USED_LEAVES: Dict[str, Dict[int, str]] = {}
    _IN_MEM_MSG_TO_LEAF: Dict[Tuple[str, str], int] = {}
    _IN_MEM_GENERATION: Dict[str, int] = {}
    _IN_MEM_NONCE: Dict[str, str] = {}
    _IN_MEM_BOUND_DIR: Dict[str, str] = {}
    _VERIFIED_ONE_TIME_KEYS: Dict[str, bytes] = {}
    _ED25519_TO_WOTS_PUBS: dict[str, set[str]] = {}
    _ED25519_ROOT_WOTS_PUB: dict[str, str] = {}
    _ED25519_SEQ_TO_WOTS_PUB: dict[tuple[str, int], str] = {}
    _WOTS_TO_ED25519_PUB: dict[str, str] = {}

    @classmethod
    def set_state_dir(
        cls,
        state_dir: Optional[Union[str, pathlib.Path]],
        *,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        with cls._LOCK:
            cls._STATE_DIR = pathlib.Path(state_dir) if state_dir is not None else None
            if seal_dir is not None:
                cls._SEAL_DIR = pathlib.Path(seal_dir)

    @classmethod
    def set_seal_dir(cls, seal_dir: Optional[Union[str, pathlib.Path]]) -> None:
        with cls._LOCK:
            cls._SEAL_DIR = pathlib.Path(seal_dir) if seal_dir is not None else None

    @classmethod
    def set_require_explicit_state_dir(cls, required: bool = True) -> None:
        with cls._LOCK:
            cls._REQUIRE_EXPLICIT_STATE_DIR = bool(required)

    @classmethod
    def set_require_existing_state(cls, required: bool = True) -> None:
        with cls._LOCK:
            cls._REQUIRE_EXISTING_STATE = bool(required)

    @classmethod
    def get_state_dir(cls) -> pathlib.Path:
        """Resolve the durable WOTS+ signing-state directory (never defaulting to temporary storage)."""
        env_dir = os.environ.get("ARMIGER_WOTS_STATE_DIR", "").strip()
        req_explicit = cls._REQUIRE_EXPLICIT_STATE_DIR or (
            os.environ.get("ARMIGER_WOTS_REQUIRE_EXPLICIT_STATE_DIR", "").strip() in ("1", "true", "yes")
        )
        if cls._STATE_DIR is not None:
            d = cls._STATE_DIR
        elif env_dir:
            d = pathlib.Path(env_dir)
        elif req_explicit:
            raise RuntimeError(
                "wots_explicit_durable_state_dir_required: configure ARMIGER_WOTS_STATE_DIR or call "
                "WOTSPlusSHAKE256.set_state_dir(...) on a persistent volume before signing"
            )
        else:
            xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
            base_state = pathlib.Path(xdg_state) if xdg_state else (pathlib.Path.home() / ".local" / "state")
            d = base_state / "armiger" / f"wots_state_{os.getuid()}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def get_seal_dir(cls, state_dir: pathlib.Path) -> pathlib.Path:
        """Resolve an independent durable freshness-checkpoint directory distinct from `state_dir`."""
        env_seal = os.environ.get("ARMIGER_WOTS_SEAL_DIR", "").strip()
        if cls._SEAL_DIR is not None:
            sdir = cls._SEAL_DIR
        elif env_seal:
            sdir = pathlib.Path(env_seal)
        else:
            xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
            base_state = pathlib.Path(xdg_state) if xdg_state else (pathlib.Path.home() / ".local" / "state")
            sdir = base_state / "armiger" / f"wots_freshness_seals_{os.getuid()}"
            if sdir.resolve() == state_dir.resolve():
                sdir = state_dir.resolve().parent / f".{state_dir.name}_freshness_seals"
        sdir.mkdir(parents=True, exist_ok=True)
        if sdir.resolve() == state_dir.resolve():
            raise RuntimeError(
                "wots_independent_seal_dir_required: seal_dir must not be the same directory as state_dir"
            )
        return sdir

    @staticmethod
    def _sync_dir(dir_path: pathlib.Path) -> None:
        dir_fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    @staticmethod
    def _compute_wots_state_mac(
        norm_root: str,
        next_leaf_index: int,
        used_map: Mapping[str, str],
        state_generation: int,
        state_nonce: str,
        bound_state_dir: str = "",
        domain: bytes = b"armiger-wots-durable-state-mac-v1:",
    ) -> str:
        mac_key = hashlib.shake_256(domain + norm_root.encode("ascii")).digest(32)
        doc_obj: Dict[str, Any] = {
            "root_pq_pub_hex": norm_root,
            "next_leaf_index": int(next_leaf_index),
            "used_leaves": {str(k): str(v) for k, v in sorted(used_map.items())},
            "state_generation": int(state_generation),
            "state_nonce": str(state_nonce),
        }
        if bound_state_dir:
            doc_obj["bound_state_dir"] = str(bound_state_dir)
        canonical = json.dumps(
            doc_obj,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hmac.new(mac_key, canonical, hashlib.sha256).hexdigest()

    @classmethod
    def _read_and_verify_wots_seal(
        cls,
        seal_path: pathlib.Path,
        norm_root: str,
        expected_bound_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            raw_s = seal_path.read_text(encoding="utf-8")
            sdoc = json.loads(raw_s)
        except Exception as exc:
            raise RuntimeError(f"wots_durable_seal_corrupted:{seal_path}:{exc}") from exc
        if (
            not isinstance(sdoc, dict)
            or sdoc.get("root_pq_pub_hex") != norm_root
            or isinstance(sdoc.get("next_leaf_index"), bool)
            or not isinstance(sdoc.get("next_leaf_index"), int)
            or isinstance(sdoc.get("state_generation"), bool)
            or not isinstance(sdoc.get("state_generation"), int)
            or int(sdoc["state_generation"]) < 1
            or not isinstance(sdoc.get("state_nonce"), str)
            or not sdoc["state_nonce"]
            or not isinstance(sdoc.get("used_leaves"), dict)
        ):
            raise RuntimeError(f"wots_durable_seal_invalid_schema:{seal_path}")
        s_next = int(sdoc["next_leaf_index"])
        s_gen = int(sdoc["state_generation"])
        s_nonce = str(sdoc["state_nonce"])
        s_bound_dir = str(sdoc.get("bound_state_dir", ""))
        s_used = {str(k): str(v) for k, v in sdoc["used_leaves"].items()}
        exp_mac = cls._compute_wots_state_mac(
            norm_root,
            s_next,
            s_used,
            s_gen,
            s_nonce,
            bound_state_dir=s_bound_dir,
            domain=b"armiger-wots-freshness-seal-mac-v1:",
        )
        if not hmac.compare_digest(str(sdoc.get("seal_mac", "")), exp_mac):
            raise RuntimeError(f"wots_durable_seal_mac_mismatch:{seal_path}")
        if expected_bound_dir is not None and s_bound_dir and s_bound_dir != expected_bound_dir:
            raise RuntimeError(
                f"wots_durable_state_directory_mismatch:{expected_bound_dir}!={s_bound_dir}"
            )
        return {
            "next_leaf_index": s_next,
            "state_generation": s_gen,
            "state_nonce": s_nonce,
            "bound_state_dir": s_bound_dir,
            "used_leaves": s_used,
        }

    @classmethod
    def provision_state(
        cls,
        priv_seed: bytes,
        *,
        state_dir: Optional[Union[str, pathlib.Path]] = None,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> pathlib.Path:
        """Explicitly provision durable initial WOTS+ state and independent freshness checkpoint for a new root key."""
        _, root_pq_pub = cls._normalize_root_priv(priv_seed)
        norm_root = root_pq_pub.hex().strip().lower()
        sdir = pathlib.Path(state_dir) if state_dir is not None else cls.get_state_dir()
        sdir.mkdir(parents=True, exist_ok=True)
        resolved_sdir = sdir.resolve()
        ext_sdir = pathlib.Path(seal_dir) if seal_dir is not None else cls.get_seal_dir(sdir)
        ext_sdir.mkdir(parents=True, exist_ok=True)
        if ext_sdir.resolve() == resolved_sdir:
            raise RuntimeError("wots_independent_seal_dir_required")
        state_path = sdir / f"wots_state_{norm_root[:32]}.json"
        sibling_seal_path = sdir / f".wots_state_{norm_root[:32]}.json.seal"
        ext_seal_path = ext_sdir / f"wots_seal_{norm_root[:32]}.seal"
        if (
            state_path.exists()
            or sibling_seal_path.exists()
            or ext_seal_path.exists()
            or norm_root in cls._IN_MEM_GENERATION
        ):
            raise RuntimeError(f"wots_state_already_provisioned:{state_path}")
        cls._load_and_update_durable_state(
            norm_root,
            explicit_leaf_index=None,
            reserve_fresh=False,
            state_dir=sdir,
            seal_dir=ext_sdir,
            _provision_only=True,
        )
        return state_path

    @classmethod
    def migrate_legacy_state(
        cls,
        priv_seed: bytes,
        *,
        state_dir: Optional[Union[str, pathlib.Path]] = None,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> pathlib.Path:
        """Explicitly upgrade a legacy generation-less WOTS+ state file to MAC-authenticated state + freshness seals."""
        _, root_pq_pub = cls._normalize_root_priv(priv_seed)
        norm_root = root_pq_pub.hex().strip().lower()
        sdir = pathlib.Path(state_dir) if state_dir is not None else cls.get_state_dir()
        sdir.mkdir(parents=True, exist_ok=True)
        resolved_sdir = sdir.resolve()
        ext_sdir = pathlib.Path(seal_dir) if seal_dir is not None else cls.get_seal_dir(sdir)
        ext_sdir.mkdir(parents=True, exist_ok=True)
        if ext_sdir.resolve() == resolved_sdir:
            raise RuntimeError("wots_independent_seal_dir_required")
        state_path = sdir / f"wots_state_{norm_root[:32]}.json"
        sibling_seal_path = sdir / f".wots_state_{norm_root[:32]}.json.seal"
        ext_seal_path = ext_sdir / f"wots_seal_{norm_root[:32]}.seal"
        if not state_path.exists():
            raise RuntimeError(f"wots_legacy_state_not_found:{state_path}")
        if sibling_seal_path.exists() or ext_seal_path.exists() or norm_root in cls._IN_MEM_GENERATION:
            raise RuntimeError(f"wots_state_already_established_cannot_downgrade_via_legacy_migration:{state_path}")
        raw_txt = state_path.read_text(encoding="utf-8")
        parsed = json.loads(raw_txt)
        if (
            not isinstance(parsed, dict)
            or isinstance(parsed.get("next_leaf_index"), bool)
            or not isinstance(parsed.get("next_leaf_index"), int)
            or not isinstance(parsed.get("used_leaves", {}), dict)
        ):
            raise ValueError("invalid_legacy_wots_state_schema")
        if "state_generation" in parsed or "state_mac" in parsed:
            raise RuntimeError("wots_state_is_not_legacy")
        next_idx = int(parsed["next_leaf_index"])
        used_map = {str(k): str(v) for k, v in parsed.get("used_leaves", {}).items()}
        next_gen = 1
        next_nonce = os.urandom(16).hex()
        bound_dir_str = str(resolved_sdir)
        state_mac = cls._compute_wots_state_mac(
            norm_root, next_idx, used_map, next_gen, next_nonce, bound_state_dir=bound_dir_str
        )
        state_bytes = json.dumps(
            {
                "root_pq_pub_hex": norm_root,
                "next_leaf_index": next_idx,
                "used_leaves": used_map,
                "state_generation": next_gen,
                "state_nonce": next_nonce,
                "bound_state_dir": bound_dir_str,
                "state_mac": state_mac,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        state_path.write_bytes(state_bytes)
        seal_mac = cls._compute_wots_state_mac(
            norm_root,
            next_idx,
            used_map,
            next_gen,
            next_nonce,
            bound_state_dir=bound_dir_str,
            domain=b"armiger-wots-freshness-seal-mac-v1:",
        )
        seal_bytes = json.dumps(
            {
                "root_pq_pub_hex": norm_root,
                "next_leaf_index": next_idx,
                "used_leaves": used_map,
                "state_generation": next_gen,
                "state_nonce": next_nonce,
                "bound_state_dir": bound_dir_str,
                "seal_mac": seal_mac,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        sibling_seal_path.write_bytes(seal_bytes)
        ext_seal_path.write_bytes(seal_bytes)
        cls._IN_MEM_GENERATION[norm_root] = next_gen
        cls._IN_MEM_NONCE[norm_root] = next_nonce
        cls._IN_MEM_BOUND_DIR[norm_root] = bound_dir_str
        return state_path

    @classmethod
    def migrate_state_directory(
        cls,
        priv_seed: bytes,
        old_state_dir: Union[str, pathlib.Path],
        new_state_dir: Union[str, pathlib.Path],
        *,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
    ) -> pathlib.Path:
        """Explicitly migrate an established WOTS+ state directory to `new_state_dir`, re-binding seals and high-water marks."""
        _, root_pq_pub = cls._normalize_root_priv(priv_seed)
        norm_root = root_pq_pub.hex().strip().lower()
        old_sdir = pathlib.Path(old_state_dir)
        new_sdir = pathlib.Path(new_state_dir)
        new_sdir.mkdir(parents=True, exist_ok=True)
        old_resolved = old_sdir.resolve()
        new_resolved = new_sdir.resolve()
        ext_sdir = pathlib.Path(seal_dir) if seal_dir is not None else cls.get_seal_dir(new_sdir)
        ext_sdir.mkdir(parents=True, exist_ok=True)
        if ext_sdir.resolve() == new_resolved:
            raise RuntimeError("wots_independent_seal_dir_required")
        old_state_path = old_sdir / f"wots_state_{norm_root[:32]}.json"
        old_sibling_seal = old_sdir / f".wots_state_{norm_root[:32]}.json.seal"
        ext_seal_path = ext_sdir / f"wots_seal_{norm_root[:32]}.seal"
        if not old_state_path.exists() or not old_sibling_seal.exists() or not ext_seal_path.exists():
            raise RuntimeError(f"wots_migration_source_state_or_seal_missing:{old_state_path}")
        s_info = cls._read_and_verify_wots_seal(ext_seal_path, norm_root, expected_bound_dir=str(old_resolved))
        cls._read_and_verify_wots_seal(old_sibling_seal, norm_root, expected_bound_dir=str(old_resolved))
        next_idx = int(s_info["next_leaf_index"])
        used_map = dict(s_info["used_leaves"])
        next_gen = max(int(s_info["state_generation"]), cls._IN_MEM_GENERATION.get(norm_root, 0)) + 1
        next_nonce = os.urandom(16).hex()
        new_bound_str = str(new_resolved)
        state_mac = cls._compute_wots_state_mac(
            norm_root, next_idx, used_map, next_gen, next_nonce, bound_state_dir=new_bound_str
        )
        new_state_path = new_sdir / f"wots_state_{norm_root[:32]}.json"
        new_sibling_seal = new_sdir / f".wots_state_{norm_root[:32]}.json.seal"
        new_state_path.write_bytes(
            json.dumps(
                {
                    "root_pq_pub_hex": norm_root,
                    "next_leaf_index": next_idx,
                    "used_leaves": used_map,
                    "state_generation": next_gen,
                    "state_nonce": next_nonce,
                    "bound_state_dir": new_bound_str,
                    "state_mac": state_mac,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        seal_mac = cls._compute_wots_state_mac(
            norm_root,
            next_idx,
            used_map,
            next_gen,
            next_nonce,
            bound_state_dir=new_bound_str,
            domain=b"armiger-wots-freshness-seal-mac-v1:",
        )
        seal_bytes = json.dumps(
            {
                "root_pq_pub_hex": norm_root,
                "next_leaf_index": next_idx,
                "used_leaves": used_map,
                "state_generation": next_gen,
                "state_nonce": next_nonce,
                "bound_state_dir": new_bound_str,
                "seal_mac": seal_mac,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        new_sibling_seal.write_bytes(seal_bytes)
        ext_seal_path.write_bytes(seal_bytes)
        old_state_path.unlink(missing_ok=True)
        old_sibling_seal.unlink(missing_ok=True)
        cls._IN_MEM_GENERATION[norm_root] = next_gen
        cls._IN_MEM_NONCE[norm_root] = next_nonce
        cls._IN_MEM_BOUND_DIR[norm_root] = new_bound_str
        return new_state_path

    @classmethod
    def _load_and_update_durable_state(
        cls,
        root_pq_pub_hex: str,
        *,
        msg_sha256_hex: Optional[str] = None,
        explicit_leaf_index: Optional[int] = None,
        reserve_fresh: bool = False,
        state_dir: Optional[Union[str, pathlib.Path]] = None,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
        require_existing_state: Optional[bool] = None,
        _provision_only: bool = False,
    ) -> int:
        norm_root = root_pq_pub_hex.strip().lower()
        sdir = pathlib.Path(state_dir) if state_dir is not None else cls.get_state_dir()
        sdir.mkdir(parents=True, exist_ok=True)
        resolved_sdir = sdir.resolve()
        bound_dir_str = str(resolved_sdir)
        ext_sdir = pathlib.Path(seal_dir) if seal_dir is not None else cls.get_seal_dir(sdir)
        ext_sdir.mkdir(parents=True, exist_ok=True)
        if ext_sdir.resolve() == resolved_sdir:
            raise RuntimeError("wots_independent_seal_dir_required")

        state_path = sdir / f"wots_state_{norm_root[:32]}.json"
        sibling_seal_path = sdir / f".wots_state_{norm_root[:32]}.json.seal"
        ext_seal_path = ext_sdir / f"wots_seal_{norm_root[:32]}.seal"
        lock_path = sdir / f"wots_state_{norm_root[:32]}.lock"
        req_existing = (
            bool(require_existing_state)
            if require_existing_state is not None
            else (
                cls._REQUIRE_EXISTING_STATE
                or os.environ.get("ARMIGER_WOTS_REQUIRE_EXISTING_STATE", "").strip() in ("1", "true", "yes")
            )
        )

        with cls._LOCK:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                persisted_next = 0
                persisted_gen = 0
                persisted_nonce = ""
                used_map: Dict[str, str] = {}
                mem_bound_dir = cls._IN_MEM_BOUND_DIR.get(norm_root)
                if mem_bound_dir is not None and mem_bound_dir != bound_dir_str:
                    raise RuntimeError(
                        f"wots_durable_state_directory_mismatch:{bound_dir_str}!={mem_bound_dir}"
                    )
                if not state_path.exists():
                    if (
                        req_existing
                        or sibling_seal_path.exists()
                        or ext_seal_path.exists()
                        or norm_root in cls._IN_MEM_GENERATION
                    ):
                        raise RuntimeError(
                            f"wots_durable_state_missing_after_establishment:{state_path}"
                        )
                else:
                    try:
                        raw_txt = state_path.read_text(encoding="utf-8")
                        parsed = json.loads(raw_txt)
                        if (
                            not isinstance(parsed, dict)
                            or isinstance(parsed.get("next_leaf_index"), bool)
                            or not isinstance(parsed.get("next_leaf_index"), int)
                        ):
                            raise ValueError("invalid_wots_state_schema")
                        persisted_next = int(parsed["next_leaf_index"])
                        raw_used = parsed.get("used_leaves", {})
                        if not isinstance(raw_used, dict):
                            raise ValueError("invalid_wots_used_leaves")
                        used_map = {str(k): str(v) for k, v in raw_used.items()}
                        if (
                            "state_generation" not in parsed
                            or "state_nonce" not in parsed
                            or "state_mac" not in parsed
                        ):
                            raise ValueError("unauthenticated_legacy_wots_state_requires_migration")
                        raw_gen = parsed.get("state_generation")
                        raw_nonce = parsed.get("state_nonce")
                        if (
                            isinstance(raw_gen, bool)
                            or not isinstance(raw_gen, int)
                            or int(raw_gen) < 1
                            or not isinstance(raw_nonce, str)
                            or not raw_nonce
                        ):
                            raise ValueError("invalid_wots_state_generation_or_nonce")
                        persisted_gen = int(raw_gen)
                        persisted_nonce = str(raw_nonce)
                        persisted_bound_dir = str(parsed.get("bound_state_dir", ""))
                        if persisted_bound_dir and persisted_bound_dir != bound_dir_str:
                            raise ValueError(
                                f"wots_state_bound_dir_mismatch:{bound_dir_str}!={persisted_bound_dir}"
                            )
                        exp_state_mac = cls._compute_wots_state_mac(
                            norm_root,
                            persisted_next,
                            used_map,
                            persisted_gen,
                            persisted_nonce,
                            bound_state_dir=persisted_bound_dir,
                        )
                        if not hmac.compare_digest(str(parsed.get("state_mac", "")), exp_state_mac):
                            raise ValueError("wots_state_mac_verification_failed")
                    except Exception as exc:
                        raise RuntimeError(f"wots_durable_state_corrupted:{state_path}:{exc}") from exc

                    # Verify against in-process high-water mark and independent durable freshness checkpoint
                    mem_gen = cls._IN_MEM_GENERATION.get(norm_root, 0)
                    mem_nonce = cls._IN_MEM_NONCE.get(norm_root, "")
                    if persisted_gen < mem_gen or (
                        persisted_gen == mem_gen and mem_nonce and persisted_nonce != mem_nonce
                    ):
                        raise RuntimeError(f"wots_durable_state_rollback_detected:in_memory_hwm:{state_path}")

                    if not ext_seal_path.exists():
                        raise RuntimeError(
                            f"wots_durable_state_missing_independent_seal:{ext_seal_path}"
                        )
                    if not sibling_seal_path.exists():
                        raise RuntimeError(
                            f"wots_durable_state_missing_sibling_seal:{sibling_seal_path}"
                        )
                    for sp in (ext_seal_path, sibling_seal_path):
                        s_info = cls._read_and_verify_wots_seal(
                            sp, norm_root, expected_bound_dir=bound_dir_str
                        )
                        if (
                            persisted_next < s_info["next_leaf_index"]
                            or persisted_gen < s_info["state_generation"]
                            or (
                                persisted_gen == s_info["state_generation"]
                                and persisted_nonce != s_info["state_nonce"]
                            )
                            or any(used_map.get(k) != v for k, v in s_info["used_leaves"].items())
                        ):
                            raise RuntimeError(
                                f"wots_durable_state_rollback_detected:freshness_seal_mismatch:{sp}"
                            )

                persisted_used_map = dict(used_map)
                staged_used_int: Dict[int, str] = {}
                for k_str, v_hex in used_map.items():
                    staged_used_int[int(k_str)] = v_hex

                next_idx = max(
                    persisted_next,
                    (max(staged_used_int.keys()) + 1) if staged_used_int else 0,
                )

                staged_msg_leaf: Optional[Tuple[Tuple[str, str], int]] = None
                if _provision_only:
                    idx = 0
                elif explicit_leaf_index is not None:
                    if isinstance(explicit_leaf_index, bool) or not isinstance(explicit_leaf_index, int):
                        raise ValueError("leaf_index must be an integer")
                    idx = int(explicit_leaf_index)
                    if idx < 0:
                        raise ValueError("leaf_index must be non-negative")
                    if idx >= cls.MAX_SUPPORTED_LEAVES:
                        raise ValueError(f"wots_leaf_index_exhausted:{idx}>={cls.MAX_SUPPORTED_LEAVES}")
                    prev_msg = used_map.get(str(idx))
                    if prev_msg is not None and msg_sha256_hex is not None and prev_msg != msg_sha256_hex:
                        raise ValueError(f"wots_leaf_index_already_used:{idx}")
                    if msg_sha256_hex is not None:
                        used_map[str(idx)] = msg_sha256_hex
                        staged_used_int[idx] = msg_sha256_hex
                        staged_msg_leaf = ((norm_root, msg_sha256_hex), idx)
                    next_idx = max(next_idx, idx + 1)
                elif not reserve_fresh and msg_sha256_hex is not None:
                    existing_idx = next(
                        (int(k) for k, v in persisted_used_map.items() if v == msg_sha256_hex),
                        None,
                    )
                    if existing_idx is not None:
                        if existing_idx < 0 or existing_idx >= cls.MAX_SUPPORTED_LEAVES:
                            raise ValueError(f"wots_leaf_index_exhausted:{existing_idx}>={cls.MAX_SUPPORTED_LEAVES}")
                        cls._IN_MEM_MSG_TO_LEAF[(norm_root, msg_sha256_hex)] = existing_idx
                        return existing_idx
                    idx = next_idx
                    if idx < 0 or idx >= cls.MAX_SUPPORTED_LEAVES:
                        raise ValueError(f"wots_leaf_index_exhausted:{idx}>={cls.MAX_SUPPORTED_LEAVES}")
                    next_idx = idx + 1
                    used_map[str(idx)] = msg_sha256_hex
                    staged_used_int[idx] = msg_sha256_hex
                    staged_msg_leaf = ((norm_root, msg_sha256_hex), idx)
                else:
                    idx = next_idx
                    if idx < 0 or idx >= cls.MAX_SUPPORTED_LEAVES:
                        raise ValueError(f"wots_leaf_index_exhausted:{idx}>={cls.MAX_SUPPORTED_LEAVES}")
                    next_idx = idx + 1
                    if msg_sha256_hex is not None:
                        used_map[str(idx)] = msg_sha256_hex
                        staged_used_int[idx] = msg_sha256_hex
                        staged_msg_leaf = ((norm_root, msg_sha256_hex), idx)

                next_gen = max(persisted_gen, cls._IN_MEM_GENERATION.get(norm_root, 0)) + 1
                next_nonce = os.urandom(16).hex()
                state_mac = cls._compute_wots_state_mac(
                    norm_root,
                    next_idx,
                    used_map,
                    next_gen,
                    next_nonce,
                    bound_state_dir=bound_dir_str,
                )
                payload_bytes = json.dumps(
                    {
                        "root_pq_pub_hex": norm_root,
                        "next_leaf_index": next_idx,
                        "used_leaves": used_map,
                        "state_generation": next_gen,
                        "state_nonce": next_nonce,
                        "bound_state_dir": bound_dir_str,
                        "state_mac": state_mac,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")

                tmp_path = sdir / f".wots_state_{norm_root[:32]}_{os.getpid()}_{threading.get_ident()}.tmp"
                fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, payload_bytes)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(str(tmp_path), str(state_path))
                cls._sync_dir(sdir)

                seal_mac = cls._compute_wots_state_mac(
                    norm_root,
                    next_idx,
                    used_map,
                    next_gen,
                    next_nonce,
                    bound_state_dir=bound_dir_str,
                    domain=b"armiger-wots-freshness-seal-mac-v1:",
                )
                seal_bytes = json.dumps(
                    {
                        "root_pq_pub_hex": norm_root,
                        "next_leaf_index": next_idx,
                        "used_leaves": used_map,
                        "state_generation": next_gen,
                        "state_nonce": next_nonce,
                        "bound_state_dir": bound_dir_str,
                        "seal_mac": seal_mac,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                for target_seal in (sibling_seal_path, ext_seal_path):
                    target_seal.parent.mkdir(parents=True, exist_ok=True)
                    stmp = target_seal.with_name(
                        f".{target_seal.name}_{os.getpid()}_{threading.get_ident()}.tmp"
                    )
                    sfd = os.open(str(stmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    try:
                        os.write(sfd, seal_bytes)
                        os.fsync(sfd)
                    finally:
                        os.close(sfd)
                    os.replace(str(stmp), str(target_seal))
                    cls._sync_dir(target_seal.parent)

                # Commit in-memory caches ONLY after durable persistence succeeds
                cls._IN_MEM_USED_LEAVES[norm_root] = staged_used_int
                cls._IN_MEM_NEXT_LEAF[norm_root] = next_idx
                cls._IN_MEM_GENERATION[norm_root] = next_gen
                cls._IN_MEM_NONCE[norm_root] = next_nonce
                cls._IN_MEM_BOUND_DIR[norm_root] = bound_dir_str
                if staged_msg_leaf is not None:
                    cls._IN_MEM_MSG_TO_LEAF[staged_msg_leaf[0]] = staged_msg_leaf[1]
                return idx
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    @staticmethod
    def _chain(x: bytes, start: int, steps: int, pub_seed: bytes, chain_idx: int) -> bytes:
        cur = x
        prefix = pub_seed + bytes([chain_idx])
        for step in range(start, start + steps):
            cur = hashlib.shake_256(prefix + bytes([step]) + cur).digest(32)
        return cur

    @classmethod
    def _msg_to_base_w(cls, msg_digest_32: bytes) -> List[int]:
        digits: List[int] = []
        for b in msg_digest_32:
            digits.append((b >> 4) & 0x0F)
            digits.append(b & 0x0F)
        csum = sum(15 - d for d in digits)
        digits.append((csum >> 8) & 0x0F)
        digits.append((csum >> 4) & 0x0F)
        digits.append(csum & 0x0F)
        return digits

    @classmethod
    def generate_keypair(cls, seed: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        raw = seed if seed is not None else os.urandom(64)
        return cls._generate_keypair_from_seed(raw)

    @classmethod
    @functools.lru_cache(maxsize=512)
    def _generate_keypair_from_seed(cls, raw: bytes) -> Tuple[bytes, bytes]:
        if len(raw) == 32:
            raw = hashlib.shake_256(b"armiger-wots-keygen-v1:" + raw).digest(64)
        elif len(raw) != 64:
            raise ValueError("WOTS+-SHAKE256 seed must be 32 or 64 bytes")
        sk_seed, pub_seed = raw[:32], raw[32:]
        ends: List[bytes] = []
        for i in range(cls.CHAINS):
            sk_i = hashlib.shake_256(sk_seed + bytes([i])).digest(32)
            ends.append(cls._chain(sk_i, 0, 15, pub_seed, i))
        pk = hashlib.shake_256(pub_seed + b"".join(ends)).digest(32)
        return raw, pk

    @classmethod
    def _raw_sign(cls, priv_seed_64: bytes, message: bytes) -> bytes:
        raw, pk = cls._generate_keypair_from_seed(priv_seed_64)
        sk_seed, pub_seed = raw[:32], raw[32:]
        digest = hashlib.shake_256(b"armiger-wots-msg-v2:" + pub_seed + pk + message).digest(32)
        digits = cls._msg_to_base_w(digest)
        sig_parts = [pub_seed]
        for i, d in enumerate(digits):
            sk_i = hashlib.shake_256(sk_seed + bytes([i])).digest(32)
            sig_parts.append(cls._chain(sk_i, 0, d, pub_seed, i))
        return b"".join(sig_parts)

    @classmethod
    def _raw_verify(cls, pub_key_32: bytes, message: bytes, sig_2176: bytes) -> bool:
        if len(pub_key_32) != cls.PUB_BYTES or len(sig_2176) != cls.SIG_BYTES:
            return False
        pub_seed = sig_2176[:32]
        digest = hashlib.shake_256(b"armiger-wots-msg-v2:" + pub_seed + pub_key_32 + message).digest(32)
        digits = cls._msg_to_base_w(digest)
        ends: List[bytes] = []
        for i, d in enumerate(digits):
            sig_i = sig_2176[32 * (i + 1) : 32 * (i + 2)]
            ends.append(cls._chain(sig_i, d, 15 - d, pub_seed, i))
        computed_pk = hashlib.shake_256(pub_seed + b"".join(ends)).digest(32)
        return hmac.compare_digest(computed_pk, pub_key_32)

    @classmethod
    @functools.lru_cache(maxsize=256)
    def _build_subtree(cls, root_priv_64: bytes, subtree_id: int) -> Dict[str, Any]:
        leaf_privs: List[bytes] = []
        leaf_pubs: List[bytes] = []
        level0: List[bytes] = []
        sub_bytes = int(subtree_id).to_bytes(4, "big")
        for j in range(cls.TREE_LEAVES):
            j_bytes = j.to_bytes(4, "big")
            l_seed = hashlib.shake_256(
                b"armiger-wots-merkle-leaf-v1:" + root_priv_64 + sub_bytes + j_bytes
            ).digest(64)
            l_priv, l_pub = cls._generate_keypair_from_seed(l_seed)
            leaf_privs.append(l_priv)
            leaf_pubs.append(l_pub)
            level0.append(
                hashlib.sha256(b"armiger-wots-leaf-node-v1:" + sub_bytes + j_bytes + l_pub).digest()
            )
        levels: List[List[bytes]] = [level0]
        cur = level0
        for lvl in range(cls.TREE_HEIGHT):
            nxt: List[bytes] = []
            for k in range(0, len(cur), 2):
                nxt.append(
                    hashlib.sha256(b"armiger-wots-int-node-v1:" + bytes([lvl]) + cur[k] + cur[k + 1]).digest()
                )
            levels.append(nxt)
            cur = nxt
        merkle_root = levels[-1][0]
        cert_msg = b"armiger-wots-tree-root-v1:" + sub_bytes + merkle_root
        if subtree_id == 0:
            cert_sig = cls._raw_sign(root_priv_64, cert_msg)
        else:
            prev_tree = cls._build_subtree(root_priv_64, subtree_id - 1)
            cert_sig = cls._raw_sign(prev_tree["leaf_privs"][cls.USABLE_LEAVES_PER_SUBTREE], cert_msg)
        return {
            "merkle_root": merkle_root,
            "levels": levels,
            "leaf_privs": leaf_privs,
            "leaf_pubs": leaf_pubs,
            "cert_sig": cert_sig,
        }

    @classmethod
    def _auth_path_bytes(cls, levels: List[List[bytes]], leaf_offset: int) -> bytes:
        parts: List[bytes] = []
        idx = int(leaf_offset)
        for lvl in range(cls.TREE_HEIGHT):
            sib = idx ^ 1
            parts.append(levels[lvl][sib])
            idx >>= 1
        return b"".join(parts)

    @classmethod
    def _normalize_root_priv(cls, priv_seed: bytes) -> Tuple[bytes, bytes]:
        if len(priv_seed) == 32:
            return cls.derive_keypair_from_ed25519_priv(priv_seed, seq=0)
        return cls.generate_keypair(priv_seed)

    @classmethod
    def _register_binding(cls, ed_pub_hex: str, wots_pub_hex: str, seq: int | None = None) -> None:
        ed_norm = ed_pub_hex.strip().lower()
        pq_norm = wots_pub_hex.strip().lower()
        cls._ED25519_TO_WOTS_PUBS.setdefault(ed_norm, set()).add(pq_norm)
        cls._WOTS_TO_ED25519_PUB[pq_norm] = ed_norm
        if seq is None or int(seq) == 0:
            cls._ED25519_ROOT_WOTS_PUB[ed_norm] = pq_norm
        elif int(seq) > 0:
            cls._ED25519_SEQ_TO_WOTS_PUB[(ed_norm, int(seq))] = pq_norm

    @classmethod
    def get_root_wots_pub(cls, ed_pub_hex: str) -> Optional[str]:
        return cls._ED25519_ROOT_WOTS_PUB.get(ed_pub_hex.strip().lower())

    @classmethod
    def derive_pq_root_pubkey(cls, ed25519_priv_bytes: bytes) -> bytes:
        _, pk0 = cls.derive_keypair_from_ed25519_priv(ed25519_priv_bytes, seq=0)
        return pk0

    @classmethod
    def derive_pq_root_pubkey_hex(cls, ed25519_priv_bytes: bytes) -> str:
        return cls.derive_pq_root_pubkey(ed25519_priv_bytes).hex()

    @classmethod
    def is_bound_to_ed25519_pub(cls, ed_pub_hex: str, wots_pub_hex: str, seq: int | None = None) -> bool:
        ed_norm = ed_pub_hex.strip().lower()
        pq_norm = wots_pub_hex.strip().lower()
        if seq is not None and (ed_norm, int(seq)) in cls._ED25519_SEQ_TO_WOTS_PUB:
            return cls._ED25519_SEQ_TO_WOTS_PUB[(ed_norm, int(seq))] == pq_norm
        return pq_norm in cls._ED25519_TO_WOTS_PUBS.get(ed_norm, set())

    @classmethod
    def get_bound_wots_pubs(cls, ed_pub_hex: str) -> set[str]:
        return set(cls._ED25519_TO_WOTS_PUBS.get(ed_pub_hex.strip().lower(), set()))

    @classmethod
    def derive_keypair_from_ed25519_priv(cls, ed25519_priv_bytes: bytes, seq: int = 0) -> Tuple[bytes, bytes]:
        ctx = b"armiger-hybrid-ed25519-wots-v1:" + ed25519_priv_bytes + (f":seq:{int(seq)}".encode("utf-8") if seq else b"")
        seed64 = hashlib.shake_256(ctx).digest(64)
        raw, pk = cls.generate_keypair(seed64)
        try:
            from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
            ed_pub = Ed25519PrivateKey.from_private_bytes(ed25519_priv_bytes).public_key().public_bytes(
                Encoding.Raw, PublicFormat.Raw
            ).hex()
            cls._register_binding(ed_pub, pk.hex(), seq=int(seq) if seq else 0)
            if seq != 0:
                ctx0 = b"armiger-hybrid-ed25519-wots-v1:" + ed25519_priv_bytes
                _, pk0 = cls.generate_keypair(hashlib.shake_256(ctx0).digest(64))
                cls._register_binding(ed_pub, pk0.hex(), seq=0)
        except Exception:
            pass
        return raw, pk

    @classmethod
    def reserve_next_leaf(
        cls,
        priv_seed: bytes,
        *,
        state_dir: Optional[Union[str, pathlib.Path]] = None,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
        require_existing_state: Optional[bool] = None,
    ) -> Tuple[int, bytes, bytes]:
        """Atomically reserve and persist the next unused one-time leaf index, returning (leaf_index, leaf_pub_32B, root_pq_pub_32B)."""
        root_priv_64, root_pq_pub = cls._normalize_root_priv(priv_seed)
        leaf_idx = cls._load_and_update_durable_state(
            root_pq_pub.hex(),
            reserve_fresh=True,
            state_dir=state_dir,
            seal_dir=seal_dir,
            require_existing_state=require_existing_state,
        )
        subtree_id = leaf_idx // cls.USABLE_LEAVES_PER_SUBTREE
        leaf_offset = leaf_idx % cls.USABLE_LEAVES_PER_SUBTREE
        tree = cls._build_subtree(root_priv_64, subtree_id)
        leaf_pub = tree["leaf_pubs"][leaf_offset]
        return leaf_idx, leaf_pub, root_pq_pub

    @classmethod
    def sign_reserved_leaf(
        cls,
        priv_seed: bytes,
        leaf_index: int,
        message: bytes,
        *,
        state_dir: Optional[Union[str, pathlib.Path]] = None,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
        require_existing_state: Optional[bool] = None,
    ) -> bytes:
        """Sign `message` using a reserved `leaf_index` and bind `(root_pq_pub, leaf_index)` to `SHA-256(message)`."""
        return cls.sign(
            priv_seed,
            message,
            leaf_index=leaf_index,
            state_dir=state_dir,
            seal_dir=seal_dir,
            require_existing_state=require_existing_state,
        )

    @classmethod
    def allocate_and_sign(
        cls,
        priv_seed: bytes,
        message: bytes,
        *,
        leaf_index: Optional[int] = None,
        reserve_fresh: bool = True,
        state_dir: Optional[Union[str, pathlib.Path]] = None,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
        require_existing_state: Optional[bool] = None,
    ) -> Tuple[bytes, bytes, int, bytes]:
        """Allocate a fresh one-time leaf and sign `message`, returning (leaf_pub_32B, sig_bytes, leaf_index, root_pq_pub_32B)."""
        root_priv_64, root_pq_pub = cls._normalize_root_priv(priv_seed)
        msg_sha = hashlib.sha256(message).hexdigest()
        idx = cls._load_and_update_durable_state(
            root_pq_pub.hex(),
            msg_sha256_hex=msg_sha,
            explicit_leaf_index=leaf_index,
            reserve_fresh=reserve_fresh if leaf_index is None else False,
            state_dir=state_dir,
            seal_dir=seal_dir,
            require_existing_state=require_existing_state,
        )
        sig_bytes = cls._build_awm2_signature(root_priv_64, root_pq_pub, idx, message)
        subtree_id = idx // cls.USABLE_LEAVES_PER_SUBTREE
        leaf_offset = idx % cls.USABLE_LEAVES_PER_SUBTREE
        leaf_pub = cls._build_subtree(root_priv_64, subtree_id)["leaf_pubs"][leaf_offset]
        return leaf_pub, sig_bytes, idx, root_pq_pub

    @classmethod
    def _build_awm2_signature(
        cls,
        root_priv_64: bytes,
        root_pq_pub: bytes,
        leaf_index: int,
        message: bytes,
    ) -> bytes:
        if isinstance(leaf_index, bool) or not isinstance(leaf_index, int) or leaf_index < 0:
            raise ValueError("leaf_index must be a non-negative integer")
        if leaf_index >= cls.MAX_SUPPORTED_LEAVES:
            raise ValueError(f"wots_leaf_index_exhausted:{leaf_index}>={cls.MAX_SUPPORTED_LEAVES}")
        subtree_id = int(leaf_index) // cls.USABLE_LEAVES_PER_SUBTREE
        leaf_offset = int(leaf_index) % cls.USABLE_LEAVES_PER_SUBTREE
        num_hops = subtree_id + 1
        if num_hops < 1 or num_hops > cls.MAX_SUBTREE_HOPS:
            raise ValueError(f"wots_num_hops_out_of_range:{num_hops}")
        hop_blobs: List[bytes] = []
        for s in range(num_hops):
            t = cls._build_subtree(root_priv_64, s)
            target_offset = cls.USABLE_LEAVES_PER_SUBTREE if s < subtree_id else leaf_offset
            hop_leaf_pub = t["leaf_pubs"][target_offset]
            auth_path = cls._auth_path_bytes(t["levels"], target_offset)
            hop_blobs.append(
                s.to_bytes(4, "big") + t["merkle_root"] + hop_leaf_pub + auth_path + t["cert_sig"]
            )
        final_tree = cls._build_subtree(root_priv_64, subtree_id)
        leaf_priv = final_tree["leaf_privs"][leaf_offset]
        leaf_pub = final_tree["leaf_pubs"][leaf_offset]
        leaf_sig = cls._raw_sign(leaf_priv, message)
        return (
            cls.AWM2_MAGIC
            + bytes([num_hops])
            + root_pq_pub
            + int(leaf_index).to_bytes(4, "big")
            + leaf_pub
            + b"".join(hop_blobs)
            + leaf_sig
        )

    @classmethod
    def sign(
        cls,
        priv_seed: bytes,
        message: bytes,
        *,
        leaf_index: Optional[int] = None,
        raw_one_time: bool = False,
        state_dir: Optional[Union[str, pathlib.Path]] = None,
        seal_dir: Optional[Union[str, pathlib.Path]] = None,
        require_existing_state: Optional[bool] = None,
    ) -> bytes:
        """Sign `message` using a Merkle-authenticated one-time WOTS+ leaf key (or raw 2,176B mode if `raw_one_time=True`)."""
        root_priv_64, root_pq_pub = cls._normalize_root_priv(priv_seed)
        if raw_one_time:
            return cls._raw_sign(root_priv_64, message)
        msg_sha = hashlib.sha256(message).hexdigest()
        idx = cls._load_and_update_durable_state(
            root_pq_pub.hex(),
            msg_sha256_hex=msg_sha,
            explicit_leaf_index=leaf_index,
            reserve_fresh=False,
            state_dir=state_dir,
            seal_dir=seal_dir,
            require_existing_state=require_existing_state,
        )
        return cls._build_awm2_signature(root_priv_64, root_pq_pub, idx, message)

    @classmethod
    def extract_signature_metadata(
        cls,
        pub_key_or_hex: Union[bytes, str],
        sig_or_hex: Union[bytes, str],
    ) -> Optional[Dict[str, Any]]:
        try:
            pub_bytes = bytes.fromhex(pub_key_or_hex.strip()) if isinstance(pub_key_or_hex, str) else bytes(pub_key_or_hex)
            sig_bytes = bytes.fromhex(sig_or_hex.strip()) if isinstance(sig_or_hex, str) else bytes(sig_or_hex)
        except ValueError:
            return None
        if len(pub_bytes) != cls.PUB_BYTES:
            return None
        if len(sig_bytes) == cls.SIG_BYTES:
            return {
                "mode": "raw",
                "root_pq_pub": pub_bytes,
                "root_pq_pub_hex": pub_bytes.hex().lower(),
                "leaf_index": None,
                "leaf_pub": pub_bytes,
                "leaf_pub_hex": pub_bytes.hex().lower(),
                "one_time_key_id": f"raw:{pub_bytes.hex().lower()}",
            }
        if sig_bytes.startswith(cls.AWM2_MAGIC) and len(sig_bytes) >= 73:
            num_hops = sig_bytes[4]
            expected_len = 73 + num_hops * cls.HOP_BYTES + cls.SIG_BYTES
            if num_hops < 1 or num_hops > 64 or len(sig_bytes) != expected_len:
                return None
            root_pq_pub = sig_bytes[5:37]
            leaf_index = int.from_bytes(sig_bytes[37:41], "big")
            leaf_pub = sig_bytes[41:73]
            return {
                "mode": "awm2",
                "root_pq_pub": root_pq_pub,
                "root_pq_pub_hex": root_pq_pub.hex().lower(),
                "leaf_index": leaf_index,
                "leaf_pub": leaf_pub,
                "leaf_pub_hex": leaf_pub.hex().lower(),
                "one_time_key_id": f"awm2:{root_pq_pub.hex().lower()}:leaf:{leaf_index}",
            }
        return None

    @classmethod
    def verify_against_trusted_roots(
        cls,
        trusted_pq_roots: Sequence[bytes],
        claimed_pq_pub: bytes,
        message: bytes,
        signature: bytes,
        *,
        used_keys: Optional[Dict[str, bytes]] = None,
    ) -> Tuple[bool, str]:
        if not trusted_pq_roots:
            return False, "untrusted_pq_root_pubkey"
        meta = cls.extract_signature_metadata(claimed_pq_pub, signature)
        if meta is None:
            return False, "pq_wots_signature_verification_failed"
        if meta["root_pq_pub"] not in trusted_pq_roots:
            return False, "untrusted_pq_root_pubkey"
        if claimed_pq_pub not in (meta["leaf_pub"], meta["root_pq_pub"]):
            return False, "untrusted_pq_root_pubkey"

        msg_digest = hashlib.sha256(message).digest()
        key_ids = (meta["leaf_pub_hex"], meta["one_time_key_id"])
        if used_keys is not None:
            for kid in key_ids:
                prev = used_keys.get(kid)
                if prev is not None and prev != msg_digest:
                    return False, "wots_one_time_key_reused"
        with cls._LOCK:
            for kid in key_ids:
                prev_g = cls._VERIFIED_ONE_TIME_KEYS.get(kid)
                if prev_g is not None and prev_g != msg_digest:
                    return False, "wots_one_time_key_reused"

        if not cls.verify(claimed_pq_pub, message, signature):
            return False, "pq_wots_signature_verification_failed"

        if used_keys is not None:
            for kid in key_ids:
                used_keys[kid] = msg_digest
        return True, "verified"

    @classmethod
    def verify(cls, pub_key: bytes, message: bytes, signature: bytes) -> bool:
        if len(pub_key) != cls.PUB_BYTES:
            return False
        msg_digest = hashlib.sha256(message).digest()

        if len(signature) == cls.SIG_BYTES:
            kid = f"raw:{pub_key.hex().lower()}"
            with cls._LOCK:
                prev = cls._VERIFIED_ONE_TIME_KEYS.get(kid)
                if prev is not None and prev != msg_digest:
                    return False
            if not cls._raw_verify(pub_key, message, signature):
                return False
            with cls._LOCK:
                cls._VERIFIED_ONE_TIME_KEYS[kid] = msg_digest
                cls._VERIFIED_ONE_TIME_KEYS[pub_key.hex().lower()] = msg_digest
            return True

        if not signature.startswith(cls.AWM2_MAGIC) or len(signature) < 73:
            return False
        num_hops = signature[4]
        if num_hops < 1 or num_hops > 64:
            return False
        expected_len = 73 + num_hops * cls.HOP_BYTES + cls.SIG_BYTES
        if len(signature) != expected_len:
            return False

        root_pq_pub = signature[5:37]
        leaf_index = int.from_bytes(signature[37:41], "big")
        leaf_pub = signature[41:73]

        if not (hmac.compare_digest(pub_key, root_pq_pub) or hmac.compare_digest(pub_key, leaf_pub)):
            return False
        if (leaf_index // cls.USABLE_LEAVES_PER_SUBTREE) + 1 != num_hops:
            return False

        kid_leaf = f"awm2:{root_pq_pub.hex().lower()}:leaf:{leaf_index}"
        kid_pub = leaf_pub.hex().lower()
        with cls._LOCK:
            for kid in (kid_leaf, kid_pub):
                prev = cls._VERIFIED_ONE_TIME_KEYS.get(kid)
                if prev is not None and prev != msg_digest:
                    return False

        cur_signer_pub = root_pq_pub
        offset = 73
        for s in range(num_hops):
            hop = signature[offset : offset + cls.HOP_BYTES]
            offset += cls.HOP_BYTES
            sub_id = int.from_bytes(hop[:4], "big")
            if sub_id != s:
                return False
            merkle_root = hop[4:36]
            hop_leaf_pub = hop[36:68]
            auth_path = hop[68:196]
            cert_sig = hop[196 : 196 + cls.SIG_BYTES]

            cert_msg = b"armiger-wots-tree-root-v1:" + sub_id.to_bytes(4, "big") + merkle_root
            if not cls._raw_verify(cur_signer_pub, cert_msg, cert_sig):
                return False

            target_offset = cls.USABLE_LEAVES_PER_SUBTREE if s < num_hops - 1 else (leaf_index % cls.USABLE_LEAVES_PER_SUBTREE)
            cur_hash = hashlib.sha256(
                b"armiger-wots-leaf-node-v1:" + sub_id.to_bytes(4, "big") + target_offset.to_bytes(4, "big") + hop_leaf_pub
            ).digest()
            idx = target_offset
            for lvl in range(cls.TREE_HEIGHT):
                sib = auth_path[lvl * 32 : (lvl + 1) * 32]
                if (idx & 1) == 0:
                    cur_hash = hashlib.sha256(b"armiger-wots-int-node-v1:" + bytes([lvl]) + cur_hash + sib).digest()
                else:
                    cur_hash = hashlib.sha256(b"armiger-wots-int-node-v1:" + bytes([lvl]) + sib + cur_hash).digest()
                idx >>= 1
            if not hmac.compare_digest(cur_hash, merkle_root):
                return False
            cur_signer_pub = hop_leaf_pub

        if not hmac.compare_digest(cur_signer_pub, leaf_pub):
            return False

        leaf_sig = signature[offset:]
        if not cls._raw_verify(leaf_pub, message, leaf_sig):
            return False

        with cls._LOCK:
            cls._VERIFIED_ONE_TIME_KEYS[kid_leaf] = msg_digest
            cls._VERIFIED_ONE_TIME_KEYS[kid_pub] = msg_digest
        return True


# ---------------------------------------------------------------------------
# Hybrid Ed25519 + WOTS+-SHAKE256 Blinded SHA-256 Tombstone Purge Engine
# ---------------------------------------------------------------------------

def blind_infohash(infohash: str) -> str:
    """Compute deterministic Post-Quantum blinded SHA-256 hash for an infohash.

    Formula: SHA-256("armiger-tombstone-v1:" || lowercase_hex_infohash)
    Preserves anonymity while allowing deterministic matching on known bad hashes (128-bit Grover PQ preimage resistance).
    """
    clean = str(infohash).strip().lower().encode("utf-8", errors="replace")
    return hashlib.sha256(b"armiger-tombstone-v1:" + clean).hexdigest()


@dataclasses.dataclass(frozen=True)
class SignedTombstoneList:
    """Hybrid Post-Quantum (Ed25519 + WOTS+-SHAKE256) signed revocation manifest."""

    seq: int
    timestamp: int
    tombstones: list[str]
    reasons: dict[str, str]
    signature_hex: str
    pq_pubkey_hex: str = ""
    pq_signature_hex: str = ""

    def canonical_payload(self) -> bytes:
        """Deterministic, sorted JSON bytes for Hybrid Ed25519 + WOTS+-SHAKE256 signature verification."""
        norm_reasons = {str(k).lower(): v for k, v in (self.reasons or {}).items()}
        norm_tombstones = sorted({str(t).lower() for t in self.tombstones})
        body: dict[str, Any] = {
            "reasons": {k: norm_reasons.get(k, "CSAM_ABUSE") for k in norm_tombstones},
            "seq": int(self.seq),
            "timestamp": int(self.timestamp),
            "tombstones": norm_tombstones,
        }
        if self.pq_pubkey_hex:
            body["pq_pubkey_hex"] = self.pq_pubkey_hex.strip().lower()
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


class TombstoneStore:
    """Maintains active blinded tombstone revocation set and verifies Hybrid Ed25519 + WOTS+-SHAKE256 updates."""

    MAX_SEQ_JUMP = 1024
    MAX_PAST_AGE_SECONDS = 7 * 86400
    MAX_FUTURE_DRIFT_SECONDS = 300

    def __init__(
        self,
        trusted_root_pubkeys_hex: Sequence[str] | None = None,
        initial_seq: int = 0,
        trusted_root_pq_pubkeys_hex: Sequence[str] | None = None,
        *,
        require_pq: bool = True,
    ) -> None:
        if isinstance(initial_seq, bool) or not isinstance(initial_seq, int) or initial_seq < 0:
            raise TypeError("initial_seq must be a non-negative integer")
        self.trusted_pubkeys: list[bytes] = []
        self.trusted_pq_pubkeys: list[bytes] = [
            bytes.fromhex(k.strip().lower())
            for k in (trusted_root_pq_pubkeys_hex or [])
            if len(k.strip()) == 64
        ]
        self.ed_to_pq_pubkeys: dict[bytes, list[bytes]] = {}
        raw_ed_entries = list(trusted_root_pubkeys_hex or [])
        for idx, raw_entry in enumerate(raw_ed_entries):
            cleaned = str(raw_entry).strip().lower()
            if ":" in cleaned:
                ed_part, pq_part = cleaned.split(":", 1)
                ed_b: Optional[bytes] = None
                if len(ed_part.strip()) == 64:
                    ed_b = bytes.fromhex(ed_part.strip())
                    if ed_b not in self.trusted_pubkeys:
                        self.trusted_pubkeys.append(ed_b)
                if len(pq_part.strip()) == 64:
                    pq_b = bytes.fromhex(pq_part.strip())
                    if pq_b not in self.trusted_pq_pubkeys:
                        self.trusted_pq_pubkeys.append(pq_b)
                    if ed_b is not None:
                        bound_list = self.ed_to_pq_pubkeys.setdefault(ed_b, [])
                        if pq_b not in bound_list:
                            bound_list.append(pq_b)
            elif len(cleaned) == 64:
                ed_b = bytes.fromhex(cleaned)
                if ed_b not in self.trusted_pubkeys:
                    self.trusted_pubkeys.append(ed_b)
                bound_list = self.ed_to_pq_pubkeys.setdefault(ed_b, [])
                if self.trusted_pq_pubkeys:
                    if len(self.trusted_pq_pubkeys) == len(raw_ed_entries) and idx < len(self.trusted_pq_pubkeys):
                        pq_b = self.trusted_pq_pubkeys[idx]
                        if pq_b not in bound_list:
                            bound_list.append(pq_b)
                    else:
                        bound_root_pq = WOTSPlusSHAKE256.get_root_wots_pub(cleaned)
                        if bound_root_pq and len(bound_root_pq) == 64:
                            pq_b = bytes.fromhex(bound_root_pq)
                            if pq_b in self.trusted_pq_pubkeys and pq_b not in bound_list:
                                bound_list.append(pq_b)
                        elif len(raw_ed_entries) == 1:
                            for pq_b in self.trusted_pq_pubkeys:
                                if pq_b not in bound_list:
                                    bound_list.append(pq_b)
                else:
                    bound_root_pq = WOTSPlusSHAKE256.get_root_wots_pub(cleaned)
                    if bound_root_pq and len(bound_root_pq) == 64:
                        pq_b = bytes.fromhex(bound_root_pq)
                        if pq_b not in self.trusted_pq_pubkeys:
                            self.trusted_pq_pubkeys.append(pq_b)
                        if pq_b not in bound_list:
                            bound_list.append(pq_b)
        self.require_pq = bool(require_pq)
        self.current_seq = int(initial_seq)
        self.blinded_hashes: set[str] = set()
        self.reasons: dict[str, str] = {}
        self._used_pq_pubkeys: dict[str, bytes] = {}

    def add_trusted_pubkey(self, pubkey_hex: str, pq_pubkey_hex: str = "") -> None:
        cleaned = pubkey_hex.strip().lower()
        if ":" in cleaned and not pq_pubkey_hex:
            ed_part, pq_part = cleaned.split(":", 1)
            cleaned = ed_part.strip()
            pq_pubkey_hex = pq_part.strip()
        raw = bytes.fromhex(cleaned)
        if len(raw) == 32 and raw not in self.trusted_pubkeys:
            self.trusted_pubkeys.append(raw)
        bound_list = self.ed_to_pq_pubkeys.setdefault(raw, []) if len(raw) == 32 else []
        if pq_pubkey_hex:
            raw_pq = bytes.fromhex(pq_pubkey_hex.strip().lower())
            if len(raw_pq) == 32:
                if raw_pq not in self.trusted_pq_pubkeys:
                    self.trusted_pq_pubkeys.append(raw_pq)
                if raw_pq not in bound_list:
                    bound_list.append(raw_pq)
        else:
            bound_root_pq = WOTSPlusSHAKE256.get_root_wots_pub(cleaned)
            if bound_root_pq and len(bound_root_pq) == 64:
                raw_pq = bytes.fromhex(bound_root_pq)
                if raw_pq not in self.trusted_pq_pubkeys:
                    self.trusted_pq_pubkeys.append(raw_pq)
                if raw_pq not in bound_list:
                    bound_list.append(raw_pq)

    def is_tombstoned(self, infohash: str) -> bool:
        """Check whether an infohash matches any active blinded tombstone."""
        blinded = blind_infohash(infohash)
        return blinded in self.blinded_hashes

    def apply_signed_tombstones(self, manifest: SignedTombstoneList) -> tuple[bool, str, int]:
        """Verify Hybrid Ed25519 + WOTS+-SHAKE256 signature, monotonic sequence, and ingest new tombstones.

        Returns (success: bool, message: str, new_tombstones_count: int).
        """
        if isinstance(manifest.seq, bool) or not isinstance(manifest.seq, int) or manifest.seq <= 0:
            return False, "invalid_tombstone_seq_type", 0
        if isinstance(manifest.timestamp, bool) or not isinstance(manifest.timestamp, int) or manifest.timestamp <= 0:
            return False, "invalid_tombstone_timestamp_type", 0
        if manifest.seq <= self.current_seq:
            return False, f"stale_or_replayed_tombstone_seq:{manifest.seq}<={self.current_seq}", 0
        if manifest.seq > (2**63 - 1) or manifest.seq > self.current_seq + self.MAX_SEQ_JUMP:
            return False, f"excessive_or_overflow_tombstone_seq_jump:{manifest.seq}", 0

        now_ts = int(time.time())
        if (manifest.timestamp - now_ts) > self.MAX_FUTURE_DRIFT_SECONDS or (now_ts - manifest.timestamp) > self.MAX_PAST_AGE_SECONDS:
            return False, f"stale_or_future_tombstone_timestamp:{manifest.timestamp}", 0

        for hex_field in (manifest.signature_hex, manifest.pq_pubkey_hex, manifest.pq_signature_hex):
            if hex_field and hex_field != hex_field.lower():
                return False, "non_canonical_uppercase_hex_in_tombstone_manifest", 0

        payload = manifest.canonical_payload()
        try:
            sig_bytes = bytes.fromhex(manifest.signature_hex)
        except ValueError:
            return False, "ed25519_signature_verification_failed", 0

        # 1. Classical Ed25519 signature verification against trusted roots
        verified = False
        matched_ed_pub: Optional[bytes] = None
        if not self.trusted_pubkeys:
            return False, "no_trusted_root_keys_configured", 0

        for pub_bytes in self.trusted_pubkeys:
            try:
                vk = Ed25519PublicKey.from_public_bytes(pub_bytes)
                vk.verify(sig_bytes, payload)
                verified = True
                matched_ed_pub = pub_bytes
                break
            except (InvalidSignature, ValueError):
                continue

        if not verified or matched_ed_pub is None:
            return False, "ed25519_signature_verification_failed", 0

        # 2. Post-Quantum WOTS+-SHAKE256 signature verification (mandatory when require_pq=True)
        has_pq_fields = bool(manifest.pq_signature_hex or manifest.pq_pubkey_hex)
        if self.require_pq or has_pq_fields or self.trusted_pq_pubkeys:
            if not manifest.pq_pubkey_hex.strip() or not manifest.pq_signature_hex.strip():
                return False, "missing_pq_wots_signature", 0
            try:
                pq_sig_bytes = bytes.fromhex(manifest.pq_signature_hex)
                norm_pq_hex = manifest.pq_pubkey_hex.strip().lower()
                man_pq_bytes = bytes.fromhex(norm_pq_hex)
            except ValueError:
                return False, "pq_wots_signature_verification_failed", 0

            bound_pq_roots = list(self.ed_to_pq_pubkeys.get(matched_ed_pub, []))
            if not bound_pq_roots:
                dyn_pq_hex = WOTSPlusSHAKE256.get_root_wots_pub(matched_ed_pub.hex())
                if dyn_pq_hex and len(dyn_pq_hex) == 64:
                    dyn_b = bytes.fromhex(dyn_pq_hex)
                    if not self.trusted_pq_pubkeys or dyn_b in self.trusted_pq_pubkeys:
                        bound_pq_roots.append(dyn_b)

            ok_pq, pq_reason = WOTSPlusSHAKE256.verify_against_trusted_roots(
                bound_pq_roots,
                man_pq_bytes,
                payload,
                pq_sig_bytes,
                used_keys=self._used_pq_pubkeys,
            )
            if not ok_pq:
                return False, pq_reason, 0

        # Monotonic update
        self.current_seq = int(manifest.seq)
        norm_reasons = {str(k).lower(): v for k, v in (manifest.reasons or {}).items()}
        new_count = 0
        for thash in manifest.tombstones:
            clean_hash = thash.lower()
            if clean_hash not in self.blinded_hashes:
                self.blinded_hashes.add(clean_hash)
                self.reasons[clean_hash] = norm_reasons.get(clean_hash, "CSAM_ABUSE")
                new_count += 1

        return True, "verified_and_applied", new_count


def derive_pq_root_pubkey_hex(private_key_bytes: bytes) -> str:
    """Return the distributable 64-char hex PQ root public key commitment for a 32-byte Ed25519 private key."""
    return WOTSPlusSHAKE256.derive_pq_root_pubkey(private_key_bytes).hex()


def generate_ed25519_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh Ed25519 private key (32 bytes) and public key (32 bytes)."""
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
    priv_bytes = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub_bytes = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(priv_bytes, seq=0)
    return priv_bytes, pub_bytes


def sign_tombstone_list(
    seq: int,
    infohashes_or_blinded: Sequence[str],
    private_key_bytes: bytes,
    reasons: dict[str, str] | None = None,
    are_blinded: bool = False,
    timestamp: int | None = None,
) -> SignedTombstoneList:
    """Create and dual-sign a SignedTombstoneList manifest with Ed25519 + Merkle-Authenticated WOTS+-SHAKE256."""
    ts = timestamp if timestamp is not None else int(time.time())
    blinded_list = [
        h.lower() if are_blinded else blind_infohash(h)
        for h in infohashes_or_blinded
    ]
    reasons_map = {}
    for h in blinded_list:
        reasons_map[h] = (reasons or {}).get(h, "CSAM_ABUSE")

    leaf_idx, leaf_pub, _root_pq_pub = WOTSPlusSHAKE256.reserve_next_leaf(private_key_bytes)
    dummy = SignedTombstoneList(
        seq=seq,
        timestamp=ts,
        tombstones=blinded_list,
        reasons=reasons_map,
        signature_hex="",
        pq_pubkey_hex=leaf_pub.hex(),
    )
    payload = dummy.canonical_payload()
    sk = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    sig = sk.sign(payload)
    pq_sig = WOTSPlusSHAKE256.sign_reserved_leaf(private_key_bytes, leaf_idx, payload)

    return SignedTombstoneList(
        seq=seq,
        timestamp=ts,
        tombstones=blinded_list,
        reasons=reasons_map,
        signature_hex=sig.hex(),
        pq_pubkey_hex=leaf_pub.hex(),
        pq_signature_hex=pq_sig.hex(),
    )
