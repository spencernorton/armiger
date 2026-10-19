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
import functools
import hashlib
import hmac
import json
import math
import os
import re
import time
import unicodedata
from typing import Any, List, Optional, Sequence, Tuple

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
    r"importante[\s._-]*leeme|"
    r"activar[\s._-]*sonido|"
    r"\d+(?:\.\d+)?\s*tb\s+collection\.torrent"
    r")(?:$|(?=[^a-z0-9]))"
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
            if f_posix.endswith((".rpa", ".rpy", ".rpyb")) or "/renpy/" in f"/{f_posix}":
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
                if _RENPY_SAY_VBS_CG_RE.search(f_posix) and not title_claims_video and (has_renpy_structure or not check_media_executables):
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
                    # Exempt legitimate Blu-ray BD-J Java menu JARs (BDMV/JAR/00000.jar)
                    if ext == ".jar" and has_bdmv_structure and _BDMV_JAR_PATH_RE.search(f_posix):
                        continue
                    # Exempt companion tracker bookmark (.url) when real video/audio/course media is present
                    if ext == ".url" and (has_large_real_video or has_real_image_set or has_real_course_media):
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
    """Post-Quantum Winternitz Hash-Based Signature over SHAKE-256 (RFC 8391 / NIST FIPS 205 building block)."""

    N = 32
    W = 16
    CHAINS = 67
    PRIV_BYTES = 64
    PUB_BYTES = 32
    SIG_BYTES = 2176

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
    @functools.lru_cache(maxsize=256)
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
    def derive_keypair_from_ed25519_priv(cls, ed25519_priv_bytes: bytes, seq: int = 0) -> Tuple[bytes, bytes]:
        ctx = b"armiger-hybrid-ed25519-wots-v1:" + ed25519_priv_bytes + (f":seq:{int(seq)}".encode("utf-8") if seq else b"")
        seed64 = hashlib.shake_256(ctx).digest(64)
        return cls.generate_keypair(seed64)

    @classmethod
    def sign(cls, priv_seed: bytes, message: bytes) -> bytes:
        if len(priv_seed) == 32:
            priv_seed, pk = cls.derive_keypair_from_ed25519_priv(priv_seed)
        else:
            _, pk = cls.generate_keypair(priv_seed)
        sk_seed, pub_seed = priv_seed[:32], priv_seed[32:]
        digest = hashlib.shake_256(b"armiger-wots-msg-v2:" + pub_seed + pk + message).digest(32)
        digits = cls._msg_to_base_w(digest)
        sig_parts = [pub_seed]
        for i, d in enumerate(digits):
            sk_i = hashlib.shake_256(sk_seed + bytes([i])).digest(32)
            sig_parts.append(cls._chain(sk_i, 0, d, pub_seed, i))
        return b"".join(sig_parts)

    @classmethod
    def verify(cls, pub_key: bytes, message: bytes, signature: bytes) -> bool:
        if len(pub_key) != cls.PUB_BYTES or len(signature) != cls.SIG_BYTES:
            return False
        pub_seed = signature[:32]
        digest = hashlib.shake_256(b"armiger-wots-msg-v2:" + pub_seed + pub_key + message).digest(32)
        digits = cls._msg_to_base_w(digest)
        ends: List[bytes] = []
        for i, d in enumerate(digits):
            sig_i = signature[32 * (i + 1) : 32 * (i + 2)]
            ends.append(cls._chain(sig_i, d, 15 - d, pub_seed, i))
        computed_pk = hashlib.shake_256(pub_seed + b"".join(ends)).digest(32)
        return hmac.compare_digest(computed_pk, pub_key)


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
        body = {
            "reasons": {k: self.reasons.get(k, "CSAM_ABUSE") for k in sorted(self.tombstones)},
            "seq": int(self.seq),
            "timestamp": int(self.timestamp),
            "tombstones": sorted([t.lower() for t in self.tombstones]),
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


class TombstoneStore:
    """Maintains active blinded tombstone revocation set and verifies Hybrid Ed25519 + WOTS+-SHAKE256 updates."""

    def __init__(
        self,
        trusted_root_pubkeys_hex: Sequence[str] | None = None,
        initial_seq: int = 0,
        trusted_root_pq_pubkeys_hex: Sequence[str] | None = None,
    ) -> None:
        self.trusted_pubkeys = [
            bytes.fromhex(k.strip().lower())
            for k in (trusted_root_pubkeys_hex or [])
            if len(k.strip()) == 64
        ]
        self.trusted_pq_pubkeys = [
            bytes.fromhex(k.strip().lower())
            for k in (trusted_root_pq_pubkeys_hex or [])
            if len(k.strip()) == 64
        ]
        self.current_seq = int(initial_seq)
        self.blinded_hashes: set[str] = set()
        self.reasons: dict[str, str] = {}

    def add_trusted_pubkey(self, pubkey_hex: str, pq_pubkey_hex: str = "") -> None:
        raw = bytes.fromhex(pubkey_hex.strip().lower())
        if len(raw) == 32 and raw not in self.trusted_pubkeys:
            self.trusted_pubkeys.append(raw)
        if pq_pubkey_hex:
            raw_pq = bytes.fromhex(pq_pubkey_hex.strip().lower())
            if len(raw_pq) == 32 and raw_pq not in self.trusted_pq_pubkeys:
                self.trusted_pq_pubkeys.append(raw_pq)

    def is_tombstoned(self, infohash: str) -> bool:
        """Check whether an infohash matches any active blinded tombstone."""
        blinded = blind_infohash(infohash)
        return blinded in self.blinded_hashes

    def apply_signed_tombstones(self, manifest: SignedTombstoneList) -> tuple[bool, str, int]:
        """Verify Hybrid Ed25519 + WOTS+-SHAKE256 signature, monotonic sequence, and ingest new tombstones.

        Returns (success: bool, message: str, new_tombstones_count: int).
        """
        if manifest.seq <= self.current_seq:
            return False, f"stale_or_replayed_tombstone_seq:{manifest.seq}<={self.current_seq}", 0

        payload = manifest.canonical_payload()
        try:
            sig_bytes = bytes.fromhex(manifest.signature_hex)
        except ValueError:
            return False, "ed25519_signature_verification_failed", 0

        # 1. Classical Ed25519 signature verification against trusted roots
        verified = False
        if not self.trusted_pubkeys:
            return False, "no_trusted_root_keys_configured", 0

        for pub_bytes in self.trusted_pubkeys:
            try:
                vk = Ed25519PublicKey.from_public_bytes(pub_bytes)
                vk.verify(sig_bytes, payload)
                verified = True
                break
            except (InvalidSignature, ValueError):
                continue

        if not verified:
            return False, "ed25519_signature_verification_failed", 0

        # 2. Post-Quantum WOTS+-SHAKE256 signature verification
        if manifest.pq_signature_hex or manifest.pq_pubkey_hex or self.trusted_pq_pubkeys:
            try:
                pq_sig_bytes = bytes.fromhex(manifest.pq_signature_hex)
            except ValueError:
                return False, "pq_wots_signature_verification_failed", 0

            candidate_pq_pubs: List[bytes] = []
            if self.trusted_pq_pubkeys:
                candidate_pq_pubs.extend(self.trusted_pq_pubkeys)
            elif manifest.pq_pubkey_hex:
                try:
                    candidate_pq_pubs.append(bytes.fromhex(manifest.pq_pubkey_hex.strip().lower()))
                except ValueError:
                    return False, "pq_wots_signature_verification_failed", 0

            pq_verified = any(
                WOTSPlusSHAKE256.verify(pq_pub, payload, pq_sig_bytes)
                for pq_pub in candidate_pq_pubs
            )
            if not pq_verified:
                return False, "pq_wots_signature_verification_failed", 0

        # Monotonic update
        self.current_seq = manifest.seq
        new_count = 0
        for thash in manifest.tombstones:
            clean_hash = thash.lower()
            if clean_hash not in self.blinded_hashes:
                self.blinded_hashes.add(clean_hash)
                self.reasons[clean_hash] = manifest.reasons.get(clean_hash, "CSAM_ABUSE")
                new_count += 1

        return True, "verified_and_applied", new_count


def generate_ed25519_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh Ed25519 private key (32 bytes) and public key (32 bytes)."""
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, PrivateFormat, NoEncryption
    priv_bytes = priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub_bytes = pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv_bytes, pub_bytes


def sign_tombstone_list(
    seq: int,
    infohashes_or_blinded: Sequence[str],
    private_key_bytes: bytes,
    reasons: dict[str, str] | None = None,
    are_blinded: bool = False,
    timestamp: int | None = None,
) -> SignedTombstoneList:
    """Create and dual-sign a SignedTombstoneList manifest with Ed25519 + Post-Quantum WOTS+-SHAKE256."""
    ts = timestamp if timestamp is not None else int(time.time())
    blinded_list = [
        h.lower() if are_blinded else blind_infohash(h)
        for h in infohashes_or_blinded
    ]
    reasons_map = {}
    for h in blinded_list:
        reasons_map[h] = (reasons or {}).get(h, "CSAM_ABUSE")

    dummy = SignedTombstoneList(
        seq=seq,
        timestamp=ts,
        tombstones=blinded_list,
        reasons=reasons_map,
        signature_hex="",
    )
    payload = dummy.canonical_payload()
    sk = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    sig = sk.sign(payload)
    pq_priv, pq_pub = WOTSPlusSHAKE256.derive_keypair_from_ed25519_priv(private_key_bytes, seq=seq)
    pq_sig = WOTSPlusSHAKE256.sign(pq_priv, payload)

    return SignedTombstoneList(
        seq=seq,
        timestamp=ts,
        tombstones=blinded_list,
        reasons=reasons_map,
        signature_hex=sig.hex(),
        pq_pubkey_hex=pq_pub.hex(),
        pq_signature_hex=pq_sig.hex(),
    )
