"""Cross-Head Audio Language & Anime Dub Disambiguator Plugin for Armiger & Armiger Ultra.

Couples Armiger's neural `title_language` head (`P(non_english_title)`) with its `lang` head
(`P(english_audio)`) and resolves five high-frequency linguistic & release-tag edge cases:
  1. Romance-language lowercase `.en.` prepositions (`viaje.en.el.tiempo`, `en.plein.soleil`)
     that naive tokenizers confuse with uppercase `EN` audio tags.
  2. `<Foreign>.ENG.Sub` / `<Foreign>.Sub.Eng` releases where `ENG` modifies `Sub` rather than audio.
  3. Explicit regional dub tags (`Hindi Dubbed`, `German.AC3LD`, `Dual-Lat`, `BG Audio`, `(BNN) ... NL`)
     when no explicit English audio track token is present.
  4. Cyrillic-titled releases without an explicit English audio tag.
  5. Multi-audio brackets/parentheses (`MULTi (FR EN)`, `[RUS + JAP + ENG]`) that include an English audio track.
"""
from __future__ import annotations

import re
from typing import Any


def apply_audio_language_guard(
    raw_name: str,
    p_en: float,
    p_title_foreign: float = 0.0,
) -> float:
    """Return the calibrated English-compatible audio probability `p_en_adjusted` in `[0.0, 1.0]`."""
    nm = raw_name or ""

    if re.search(
        r"(?i)(\bmulti\s*\([^\)]*\b(en|eng|english)\b|"
        r"\[\s*(rus|jpn|jap|ita|fre|ger|deu|spa|lat)[^\]]*\+\s*(en|eng|english)\b)",
        nm,
    ):
        return max(p_en, 0.88)

    if re.search(
        r"(?i)\b(ita|ger|deu|fre|fra|rus|spa|jpn|kor|chi|hin|dut|nl|por|pol|tur)[\.\s_-]+(en|eng|english)[\.\s_-]+sub",
        nm,
    ):
        if not re.search(
            r"(?i)\b(en|eng|english)[\.\s_-]+(ac3|aac|dts|ddp|truehd|flac|mp3|audio|dub)\b",
            nm,
        ):
            return min(p_en, 0.10)

    has_explicit_eng_audio = bool(
        re.search(r"(?i)\b(eng|english)\b", nm)
        or re.search(r"[\[\(\.\s_-]EN[\]\)\.\s_-]", nm)
        or re.search(r"英语", nm)
    )

    if not has_explicit_eng_audio:
        if re.search(
            r"(?i)(?:[\.\s_-]bg[\.\s_-]+audio\b|\bhindi[\.\s_-]+(?:dub|dubbed|web[\.\s_-]+series)\b|"
            r"\bgerman[\.\s_-]+(?:ac3ld|ld|dubbed|dts|ac3)\b|[\.\s_-]dual[\.\s_-]*lat\b|\(bnn\).*\bnl\b)",
            nm,
        ):
            return min(p_en, 0.10)
        if re.search(r"[А-Яа-яЁё]{4,}", nm):
            return min(p_en, 0.12)
        if p_title_foreign >= 0.85 and p_en < 0.75:
            return min(p_en, 0.25)

    return float(p_en)


def classify_audio_and_anime_dub(
    raw_name: str,
    p_en: float,
    p_anime: float,
    p_title_foreign: float = 0.0,
) -> dict[str, Any]:
    """Classify English/Multi audio availability and anime dub/dual-audio status."""
    p_en_adj = apply_audio_language_guard(raw_name, p_en=p_en, p_title_foreign=p_title_foreign)
    is_english_compatible = p_en_adj >= 0.50
    has_explicit_dub_cue = bool(
        re.search(r"(?i)\b(dual[\s._-]*audio|dubbed|eng[\s._-]*dub|english[\s._-]*dub|multi[\s._-]*audio)\b", raw_name or "")
    )
    is_anime = p_anime >= 0.50
    return {
        "p_english_compatible_audio": round(p_en_adj, 4),
        "is_english_compatible": is_english_compatible,
        "is_anime": is_anime,
        "is_anime_dub_or_dual": bool(is_anime and (has_explicit_dub_cue or p_en_adj >= 0.65)),
    }
