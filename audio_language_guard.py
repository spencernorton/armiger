"""3-Layer Cross-Head Audio Language & Anime-Dub Disambiguator Plugin (`armiger/audio_language_guard.py`).

Couples Armiger's neural `title_language` head (`P(non_english_title)`) with its `lang` head
(`P(english_audio)`) and resolves linguistic and release-tag edge cases:
  1. Strips subtitle-only clauses (`[Napisy ENG-PL]`, `Sub Ita Eng`, `[JPN - ENG Sub]`, `Eng Subs`)
     before testing for explicit English audio tokens.
  2. Distinguishes single-track voiceover tags (`[Lektor PL]`) from dual-track (`[ENG-Lektor PL]`),
     and multi-audio tags (`3Audio`, `4Audio`, `国粤英多音轨`, `MULTi (FR EN)`, `[RUS + JAP + ENG]`)
     from single-title subtitle tags.
  3. Restricts post-title foreign audio/tracker markers so English work titles containing language
     adjectives are not penalized when English audio is present.
  4. Separates general English-compatible audio (`is_english_compatible`) from high-precision Anime
     English-Dub / Dual-Audio classification (`is_anime_dub_or_dual`).
"""
from __future__ import annotations

import re
from typing import Any

SUB_SEGMENT_STRIP_RE = re.compile(
    r"(?i)(?:\[\s*napisy[^\]]*\]|"
    r"\[\s*(?:jpn|jap|chi|chs|cht|kor|ita|esp|spa|fre|ger)[\s._-]*[\-\+/][\s._-]*(?:en|eng|english)[\s._-]*subs?\s*\]|"
    r"\b(?:en|eng|english)[\s._-]*(?:fan[\s._-]*)?subs?\b|"
    r"\bsub(?:s|titles?|t)?[\s._:-]+(?:ita|spa|esp|fre|ger|rus|por|pol|dut|nl|eng|en|es|fr|de|it|pt|ingles|ingl[eé]s)"
    r"(?:[\s._,/-]+(?:ita|spa|esp|fre|ger|rus|por|pol|dut|nl|eng|en|es|fr|de|it|pt|ingles|ingl[eé]s))*)"
)

MULTI_ENG_AUDIO_RE = re.compile(
    r"(?i)(?:\bmulti[\s._-]*\([^\)]*\b(?:en|eng|english)\b|"
    r"\[\s*(?:rus|ukr|jpn|jap|ita|fre|fra|ger|deu|spa|esp|lat)[^\]]*[\+\-\/]\s*(?:en|eng|english)\b(?!\s*sub)|"
    r"\[esp(?:-[a-z]+)*-eng\]|"
    r"\bcastellano[^\n\r]*\bingl[eé]s(?![\s._-]*sub)\b|"
    r"\b(?:en|eng|english)[\.\s_-]+(?:dts(?:-hd|-x)?|ac3|aac|ddp|truehd|flac)\b|"
    r"国粤英多音轨|"
    r"\b[34]audio\b|"
    r"\bENGLISH\s+XXX\b|"
    r"\b(?:ita|ger|deu|fre|fra|rus|spa|esp|jpn)[\.\s_-]+(?:ac3|aac|dts|ddp|mp3|flac|5\.1|2\.0)[\.\s_-]+(?:en|eng|english)\b)"
)

FOREIGN_AUDIO_POST_RE = re.compile(
    r"(?i)(?:[\.\s_-]bg[\.\s_-]+audio\b|"
    r"\bhindi[\.\s_-]+(?:dub|dubbed|web[\.\s_-]+series)\b|"
    r"\bgerman[\.\s_-]+(?:ac3ld|ld|dubbed|dts|ac3)\b|"
    r"\(bnn\).*\bnl\b|"
    r"\[esp\]|\bcastellano\b|"
    r"\b(?:divxtotal|maxitorrent|nucleohd|pctnew|descargas2020|atomohd|atomixhq|pctreload|elitetorrent)\b|"
    r"\b(?:cze|czech|lithuanian|latvian|estonian|finnish|hungarian|romanian|bulgarian|serbian|croatian|"
    r"slovak|slovenian|ukrainian|hebrew|persian|farsi|vietnamese|thai|indonesian|malay|tagalog)\b)"
)

FOREIGN_TITLE_MORPH_RE = re.compile(
    r"(?i)(?:\b(?:frauen|ohne|unschuld|hasslich|hässlich|alarmstufe)\b|"
    r"\b(?:a\s+vendre|à\s+vendre|en\s+attendant)\b|"
    r"\b\w*(?:aa|ee|ii|oo|uu|yy|ä|ö)\w*(?:tut|tyt|nut|nyt)\b|"
    r"\b\w*(?:sz|cz|rz|szcz)\w*\s+\w*(?:sz|cz|rz|szcz|mistrz)\w*\b)"
)


def apply_audio_language_guard_v3(
    raw_name: str,
    p_en: float,
    p_title_foreign: float = 0.0,
) -> float:
    """Return calibrated English-compatible audio probability `p_en_adjusted` in `[0.0, 1.0]`."""
    nm = raw_name or ""

    if MULTI_ENG_AUDIO_RE.search(nm):
        if re.search(r"(?i)\b(?:d'amore|l'amore)\b", nm) and " - " not in nm and "mircrew" not in nm.lower():
            return min(p_en, 0.20)
        return max(p_en, 0.88)

    if re.search(r"(?i)\blektor[\s._-]*pl\b", nm):
        nm_without_lektor = re.sub(r"(?i)\blektor[\s._-]*pl\b", " ", SUB_SEGMENT_STRIP_RE.sub(" ", nm))
        has_eng_alongside_lektor = bool(
            re.search(r"(?i)\b(?:eng|english)\b", nm_without_lektor)
            or re.search(r"[\[\(\.\s_-]EN[\]\)\.\s_-]", nm_without_lektor)
            or re.search(r"(?i)\bdual[\.\s_-]*audio\b", nm_without_lektor)
        )
        if not has_eng_alongside_lektor:
            return min(p_en, 0.10)
        return max(p_en, 0.85)

    if re.search(
        r"(?i)\b(?:ita|ger|deu|fre|fra|rus|spa|jpn|kor|chi|hin|dut|nl|por|pol|tur)[\.\s_-]+(?:en|eng|english)[\.\s_-]+sub",
        nm,
    ):
        if not re.search(
            r"(?i)(?:\b(?:en|eng|english)[\.\s_-]+(?:ac3|aac|dts|ddp|truehd|flac|mp3|audio|dub)\b|"
            r"sub[\.\s_-]+(?:ita|ger|fre|spa)[\.\s_-]+(?:en|eng)\b|mircrew)",
            nm,
        ):
            return min(p_en, 0.10)

    if re.match(r"^\s*[А-Яа-яЁё]{3,}", nm) and re.search(r"(?i)\.ПМ\.Eng\b", nm):
        return min(p_en, 0.12)
    if re.match(r"^\s*[\u4e00-\u9fff]+\.\d{4}\.", nm) and "英语中字" in nm:
        return min(p_en, 0.12)

    nm_nosub = SUB_SEGMENT_STRIP_RE.sub(" ", nm)

    has_explicit_eng_audio = bool(
        re.search(r"(?i)\b(?:eng|english)\b", nm_nosub)
        or re.search(r"[\[\(\.\s_-]EN[\]\)\.\s_-]", nm_nosub)
        or re.search(r"英语", nm_nosub)
        or re.search(r"(?i)\bdual[\.\s_-]*audio\b", nm_nosub)
    )

    if not has_explicit_eng_audio:
        if FOREIGN_AUDIO_POST_RE.search(nm) or FOREIGN_TITLE_MORPH_RE.search(nm):
            return min(p_en, 0.10)
        if re.search(r"(?i)[\.\s_-]dual[\.\s_-]*lat\b", nm) and (p_en < 0.50 or (p_en < 0.84 and p_title_foreign >= 0.30)):
            return min(p_en, 0.10)
        if re.search(r"(?i)\bexkinoray\b", nm) and p_en < 0.85:
            return min(p_en, 0.10)
        if re.search(r"[А-Яа-яЁё]{4,}", nm) and not re.search(
            r"(?:[А-Яа-яЁё]+\s*/\s*[A-Za-z]{3,}|[A-Za-z]{3,}[^А-Яа-яЁё]*\s-\s*[А-Яа-яЁё]{4,})",
            nm,
        ):
            return min(p_en, 0.12)
        if p_title_foreign >= 0.85 and p_en < 0.75:
            return min(p_en, 0.25)

    return float(p_en)


def classify_audio_and_anime_dub_v3(
    raw_name: str,
    p_en: float,
    p_anime: float,
    p_title_foreign: float = 0.0,
) -> dict[str, Any]:
    """Classify general English-compatible audio and high-precision anime English-dub/dual-audio status."""
    nm = raw_name or ""
    p_en_adj = apply_audio_language_guard_v3(nm, p_en=p_en, p_title_foreign=p_title_foreign)
    is_english_compatible = p_en_adj >= 0.50
    is_anime = p_anime >= 0.50

    has_foreign_only_anime_tag = bool(
        re.search(r"[А-Яа-яЁё]{3,}|\b(?:vfi|vfq|vff|truefrench|dual[\s._-]*lat)\b", nm, re.I)
        and not re.search(r"(?i)\[\s*(?:rus|ukr)[^\]]*\+\s*(?:en|eng|english)\b", nm)
    ) or bool(
        re.search(r"(?i)\bmulti[\s._-]*subs\b", nm)
        and not re.search(r"(?i)\b(?:dual[\s._-]*audio|english[\s._-]*dub(?:bed)?|eng[\s._-]*dub(?:bed)?)\b", nm)
    )
    clean_anime = re.sub(r"(?i)\[(?:eng(?:lish)?[\s._-]*subs?|sub[\s._-]*eng)\]", " ", nm)
    has_verified_anime_dub_cue = bool(
        not has_foreign_only_anime_tag
        and re.search(
            r"(?i)(?:\b(?:english[\s._-]*dub(?:bed)?|eng[\s._-]*dub(?:bed)?|funidub|golumpa|arukoru)\b|"
            r"\bdual[\s._-]*audio\b|"
            r"\[esp(?:-[a-z]+)*-eng\]|"
            r"\[\s*(?:rus|jpn|jap|ita|fre|ger|spa)[^\]]*\+\s*(?:en|eng|english)\b|"
            r"\bita[\.\s_-]+eng\b|"
            r"\b[34]audio\b|"
            r"\b(?:cr|nf|hidive|amzn|funimation)[\s._-]+web-dl[\s._-]+dual\b|"
            r"\.dual\.(?:aac|ddp|ac3|flac|opus)|"
            r"\bmulti-varyg\b)",
            clean_anime,
        )
    )
    if is_anime and has_verified_anime_dub_cue:
        p_en_adj = max(p_en_adj, 0.85)
        is_english_compatible = True

    is_anime_dub = bool(is_anime and has_verified_anime_dub_cue)
    return {
        "p_english_compatible_audio": round(p_en_adj, 4),
        "is_english_compatible": is_english_compatible,
        "is_anime": is_anime,
        "is_anime_dub_or_dual": is_anime_dub,
        "audio_class": "english_compatible" if is_english_compatible else "non_english",
        "anime_audio_subtype": ("dub_or_dual" if is_anime_dub else "sub_or_raw") if is_anime else "not_anime",
    }


# Backward-compatible entrypoints used by armiger/ultra_bridge.py and existing tests
apply_audio_language_guard = apply_audio_language_guard_v3
classify_audio_and_anime_dub = classify_audio_and_anime_dub_v3
