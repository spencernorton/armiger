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
     English-Dub / Dual-Audio classification (`is_anime_dub_or_dual`), including `[Dual-Audio][Multi-Subs]`
     releases (`Gap S3-01`) and data-driven JSON rule overrides from `calibration.json`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any
import unicodedata

try:
    from armiger.calibration_and_dht import safe_regex_search
except ImportError:
    from calibration_and_dht import safe_regex_search  # type: ignore[no-redef]

_FOREIGN_TITLE_TOKEN_RE = re.compile(
    r"(?i)(?:[\u0400-\u04FF\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]{2,}|"
    r"^\[(?:subsplease|erai-raws|horriblesubs)\].*\b(?:no|ni|wa|ga|wo|kara|tachi|monogatari)\b|"
    r"\b(?:sousou|hitorigoto|shingeki|kyojin|kimetsu|yaiba|jujutsu|kaisen|"
    r"shippuden|sennen|kessen|mushoku|tensei|konosuba|shokugeki|haikyuu|monogatari)\b|"
    r"\b(?:frauen|ohne|unschuld|hasslich|hässlich|alarmstufe|vendre|attendant|"
    r"diesem|noites|insonia|mord_im|in_der|een_blik|cuentos|vaaleanpunaiset)\b)"
)

_ENGLISH_TITLE_SHIELD_RE = re.compile(
    r"(?i)(?:^|_|-|\s)(?:the|of|and|in|to|for|with|on|at|from|by|adventures|story|life|world|"
    r"history|poems|tales|short|collection|book|volume|complete|works|letters|memoirs|journal|"
    r"novel|mystery|murder|detective|island|house|night|secret|woman|women|man|men|boy|girl|"
    r"king|queen|prince|princess|war|peace|love|death|time|day|days|year|years)(?:$|_|-|\s)"
)

_EXPLICIT_ENG_SUB_TAG_RE = re.compile(
    r"(?i)(?:\b(?:eng|english)[\s\._-]*subs?\b|\bsubs?[\s\._-]*(?:eng|english)\b|\besubs?\b|"
    r"\bmulti(?:ple)?[\s\._-]*sub(?:title)?s?\b|^\[(?:subsplease|erai-raws|judas|horriblesubs|ember)\])"
)

_FOREIGN_ONLY_SUB_FILE_RE = re.compile(
    r"(?i)(?:[\._-](?:es|spa|esp|spanish|castellano|lat|fr|fre|fra|french|vostfr|de|ger|deu|german|"
    r"it|ita|italian|pt|por|portuguese|pt-br|pt_br|pob|ru|rus|russian|pl|pol|polish|nl|dut|nld|dutch|"
    r"tr|tur|turkish|ar|ara|arabic|zh|chi|zho|chs|cht|cmn|chinese|ja|jpn|jap|japanese|ko|kor|korean|"
    r"ro|rom|ron|rum|romanian|cs|cze|ces|czech|hu|hun|hungarian|uk|ukr|ukrainian|fa|fas|per|persian|farsi|"
    r"el|gre|ell|greek|he|heb|hebrew|sv|swe|swedish|da|dan|danish|no|nor|nob|nno|norwegian|fi|fin|finnish|"
    r"bg|bul|bulgarian|hr|hrv|croatian|sr|srp|serbian|sk|slo|slk|slovak|sl|slv|slovenian|lt|lit|lithuanian|"
    r"lv|lav|latvian|et|est|estonian|hi|hin|hindi|vi|vie|vietnamese|th|tha|thai|id|ind|indonesian|"
    r"ms|may|msa|malay|tl|tgl|tagalog)\.(?:srt|vtt|ass|ssa|sub)$)"
)

SUB_SEGMENT_STRIP_RE = re.compile(
    r"(?i)(?:\[\s*napisy[^\]]*\]|"
    r"\[\s*(?:jpn|jap|chi|chs|cht|kor|ita|esp|spa|fre|ger)[\s._-]*[\-\+/][\s._-]*(?:en|eng|english)[\s._-]*subs?\s*\]|"
    r"\b(?:en|eng|english)[\s._-]*(?:fan[\s._-]*)?subs?\b|"
    r"\bsub(?:s|titles?|t)?[\s._:-]+(?:ita|spa|esp|fre|ger|rus|por|pol|dut|nl|eng|en|es|fr|de|it|pt|ingles|ingl[eé]s)"
    r"(?:[\s._,/-]+(?:ita|spa|esp|fre|ger|rus|por|pol|dut|nl|eng|en|es|fr|de|it|pt|ingles|ingl[eé]s))*)"
)

MULTI_ENG_AUDIO_RE = re.compile(
    r"(?i)(?:\bmulti[\s._-]*\([^\)]*\b(?:en|eng|english)\b|"
    r"\[\s*(?:rus|ukr|jpn|jap|ita|fre|fra|ger|deu|spa|esp|lat)[^\]]*[\+\-\/]\s*(?:en|eng|english)\b(?![\s._\-/]*sub)|"
    r"\[esp(?:-[a-z]+)*-eng\]|"
    r"\bcastellano[^\n\r]*\bingl[eé]s(?![\s._-]*sub)\b|"
    r"\b(?:en|eng|english)[\.\s_-]+(?:dts(?:-hd|-x)?|ac3|aac|ddp|truehd|flac)\b|"
    r"国粤英多音轨|"
    r"\b[34]audio\b|"
    r"\bENGLISH\s+XXX\b|"
    r"\b(?:ita|ger|deu|fre|fra|rus|spa|esp|jpn)[\.\s_-]+(?:ac3|aac|dts|ddp|mp3|flac|5\.1|2\.0)[\.\s_-]+(?:en|eng|english)\b)"
)

_AMBIGUOUS_FOREIGN_LANG_WORDS = (
    r"(?:cze|czech|lithuanian|latvian|estonian|finnish|hungarian|romanian|bulgarian|serbian|croatian|"
    r"slovak|slovenian|ukrainian|hebrew|persian|farsi|vietnamese|thai|indonesian|malay|tagalog)"
)

FOREIGN_AUDIO_POST_RE = re.compile(
    r"(?i)(?:[\.\s_-]bg[\.\s_-]+audio\b|"
    r"\bhindi[\.\s_-]+(?:dub|dubbed|audio|web[\.\s_-]+series)\b|"
    r"\bgerman[\.\s_-]+(?:ac3ld|ld|dubbed|dts|ac3|audio)\b|"
    r"\(bnn\).*\bnl\b|"
    r"\[esp\]|\bcastellano\b|"
    r"\b(?:divxtotal|maxitorrent|nucleohd|pctnew|descargas2020|atomohd|atomixhq|pctreload|elitetorrent)\b|"
    r"[\[\(]\s*" + _AMBIGUOUS_FOREIGN_LANG_WORDS + r"(?:[\.\s_-]+(?:audio|dub|dubbed|2\.0|5\.1))?\s*[\]\)]|"
    r"\b" + _AMBIGUOUS_FOREIGN_LANG_WORDS + r"[\.\s_-]+(?:audio|dub|dubbed|ac3|aac|dts|ddp|mp3|flac|truehd|dd5|5\.1|2\.0)\b|"
    r"(?:[\(\[\.\s_-](?:19\d\d|20\d\d)[\)\]\.\s_-]|\bS\d{1,2}(?:E\d{1,3})?\b|"
    r"\b(?:720p|1080p|2160p|480p|576p|web-dl|webrip|bluray|bdrip|brrip|dvdrip|hdtv)\b)"
    r"[^\r\n]*?\b" + _AMBIGUOUS_FOREIGN_LANG_WORDS + r"\b(?![\.\s_-]*sub))"
)

FOREIGN_TITLE_MORPH_RE = re.compile(
    r"(?i)(?:\b(?:frauen|ohne|unschuld|hasslich|hässlich|alarmstufe|fackeln\s+im\s+sturm)\b|"
    r"\b(?:a\s+vendre|à\s+vendre|en\s+attendant)\b|"
    r"\b\w*(?:aa|ee|ii|oo|uu|yy|ä|ö)\w*(?:tut|tyt|nut|nyt)\b|"
    r"\b\w*(?:sz|cz|rz|szcz)\w*\s+\w*(?:sz|cz|rz|szcz|mistrz)\w*\b)"
)


def _match_safe_token_list(text: str, tokens: list[str] | tuple[str, ...]) -> bool:
    """Safely check if any literal token from a schema-validated JSON token list appears in `text`."""
    if not tokens or not text:
        return False
    lowered = text.lower()
    for tok in tokens:
        t = str(tok).strip().lower()
        if not t:
            continue
        pat = r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])"
        if re.search(pat, lowered):
            return True
    return False


def apply_audio_language_guard_v3(
    raw_name: str,
    p_en: float,
    p_title_foreign: float = 0.0,
    plugin_rules: Mapping[str, Any] | None = None,
) -> float:
    """Return calibrated English-compatible audio probability `p_en_adjusted` in `[0.0, 1.0]`."""
    nm = raw_name or ""
    rules = plugin_rules or {}
    extra_dub_tokens = list(rules.get("extra_dub_tokens") or [])
    extra_foreign_tokens = list(rules.get("extra_foreign_audio_tokens") or [])
    foreign_title_thresh = float(rules.get("foreign_title_threshold", 0.85))
    custom_en_re = rules.get("explicit_en_regex")
    custom_non_en_re = rules.get("explicit_non_en_regex")
    custom_multi_subs_re = rules.get("multi_subs_only_regex")
    nm_nosub = SUB_SEGMENT_STRIP_RE.sub(" ", nm)

    if (
        MULTI_ENG_AUDIO_RE.search(nm_nosub)
        or _match_safe_token_list(nm_nosub, extra_dub_tokens)
        or safe_regex_search(custom_en_re, nm_nosub, field_name="audio_guard.explicit_en_regex")
    ):
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
    ) or safe_regex_search(custom_multi_subs_re, nm, field_name="audio_guard.multi_subs_only_regex"):
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
        re.search(r"(?i)(?:^|[\[\(\.\s_-])(?:eng|english)(?=$|[\]\)\.\s_-])", nm_nosub)
        or re.search(r"[\[\(\.\s_-]EN[\]\)\.\s_-]", nm_nosub)
        or re.search(r"英语", nm_nosub)
        or re.search(r"(?i)(?:^|[\[\(\.\s_-])dual[\.\s_-]*audio(?=$|[\]\)\.\s_-])", nm_nosub)
        or safe_regex_search(custom_en_re, nm_nosub, field_name="audio_guard.explicit_en_regex")
    )

    if not has_explicit_eng_audio:
        if (
            FOREIGN_AUDIO_POST_RE.search(nm)
            or FOREIGN_TITLE_MORPH_RE.search(nm)
            or _match_safe_token_list(nm, extra_foreign_tokens)
            or safe_regex_search(custom_non_en_re, nm, field_name="audio_guard.explicit_non_en_regex")
        ):
            return min(p_en, 0.10)
        if re.search(r"(?i)[\.\s_-]dual[\.\s_-]*lat\b", nm) and (p_en < 0.50 or (p_en < 0.84 and p_title_foreign >= 0.30)):
            return min(p_en, 0.10)
        if re.search(r"(?i)\bexkinoray\b", nm) and p_en < 0.85:
            return min(p_en, 0.10)
        if (
            re.search(r"[А-Яа-яЁё]{4,}", nm)
            and (p_en < 0.85 or p_title_foreign >= 0.50)
            and not re.search(
                r"(?:[А-Яа-яЁё]+\s*/\s*[A-Za-z]{3,}|[A-Za-z]{3,}[^А-Яа-яЁё]*\s-\s*[А-Яа-яЁё]{4,})",
                nm,
            )
        ):
            return min(p_en, 0.12)
        if p_title_foreign >= foreign_title_thresh and p_en < 0.75:
            return min(p_en, 0.25)

    return float(p_en)


_COMMON_EN_WORDS_SET: frozenset[str] = frozenset({
    "the", "of", "and", "in", "to", "for", "with", "on", "at", "from", "by", "about", "into", "over", "after",
    "under", "between", "through", "without", "before", "behind", "beyond", "against", "among", "around",
    "adventures", "adventure", "story", "stories", "life", "lives", "world", "worlds", "history", "histories",
    "poems", "poem", "poetry", "tales", "tale", "short", "collection", "collected", "book", "books", "volume",
    "volumes", "complete", "works", "work", "letters", "letter", "memoirs", "memoir", "journal", "journals",
    "diary", "diaries", "novel", "novels", "mystery", "mysteries", "murder", "murders", "detective", "island",
    "islands", "house", "houses", "night", "nights", "secret", "secrets", "woman", "women", "man", "men",
    "boy", "boys", "girl", "girls", "king", "kings", "queen", "queens", "prince", "princess", "war", "wars",
    "peace", "love", "death", "time", "times", "day", "days", "year", "years", "little", "great", "old", "new",
    "good", "bad", "black", "white", "red", "blue", "green", "golden", "silver", "dark", "light", "shadow",
    "shadows", "lost", "last", "first", "second", "third", "three", "two", "four", "five", "seven", "ten",
    "hundred", "thousand", "million", "uncle", "aunt", "father", "mother", "brother", "brothers", "sister",
    "sisters", "son", "sons", "daughter", "daughters", "child", "children", "family", "friend", "friends",
    "doctor", "captain", "master", "lord", "lady", "saint", "general", "colonel", "major", "professor",
    "american", "british", "modern", "ancient", "early", "late", "north", "south", "east", "west", "western", "eastern", "northern",
    "southern", "central", "city", "town", "country", "village", "castle", "palace", "tower", "garden",
    "gardens", "forest", "woods", "river", "sea", "ocean", "mountain", "mountains", "valley", "hill", "hills",
    "road", "street", "way", "journey", "voyage", "travels", "travel", "guide", "manual", "handbook",
    "introduction", "principles", "elements", "theory", "practice", "science", "art", "arts", "philosophy",
    "religion", "church", "state", "law", "laws", "government", "society", "nature", "human", "mind", "soul",
    "spirit", "heart", "body", "health", "medicine", "magic", "witch", "ghost", "ghosts", "fairy", "fairies",
    "dragon", "dragons", "monster", "monsters", "vampire", "empire", "kingdom", "republic", "revolution",
    "battle", "army", "navy", "soldier", "soldiers", "hero", "heroes", "knight", "knights", "pirate", "pirates",
    "treasure", "gold", "diamond", "diamonds", "ring", "sword", "crown", "throne", "cross", "star", "stars",
    "sun", "moon", "earth", "heaven", "hell", "paradise", "dream", "dreams", "sleep", "winter", "summer",
    "spring", "autumn", "morning", "evening", "midnight", "dawn", "twilight", "storm", "wind", "fire", "water",
    "blood", "iron", "steel", "stone", "wood", "glass", "rose", "roses", "flower", "flowers", "tree", "trees",
    "bird", "birds", "horse", "horses", "dog", "dogs", "cat", "cats", "lion", "wolf", "bear", "eagle",
    "serpent", "snake", "fish", "whale", "ship", "train", "flight", "escape", "return", "fall", "rise",
    "revenge", "fortune", "destiny", "fate", "chance", "truth", "honor", "glory", "power", "wealth", "poverty",
    "crime", "punishment", "justice", "freedom", "liberty", "slavery", "marriage", "wedding", "divorce",
    "birth", "youth", "age", "beauty", "music", "song", "songs", "dance", "play", "plays", "tragedy", "comedy",
    "drama", "theatre", "stage", "picture", "pictures", "portrait", "painted", "veil", "mirror", "window",
    "door", "gate", "key", "bridge", "wall", "room", "chamber", "hall", "school", "college", "university",
    "student", "teacher", "lesson", "lessons", "study", "studies", "essay", "essays", "speech", "speeches",
    "lecture", "lectures", "sermon", "sermons", "bible", "gospel", "psalms", "prophet", "god", "gods", "angel",
    "angels", "devil", "demon", "sin", "virtue", "grace", "faith", "hope", "charity", "wisdom", "knowledge",
    "folly", "madness", "passion", "desire", "fear", "terror", "horror", "wonder", "miracle", "vision",
    "prophecy", "legend", "legends", "myth", "myths", "fable", "fables", "folklore", "ballad", "ballads",
    "chronicle", "chronicles", "annals", "records", "account", "narrative", "sketch", "sketches", "scenes",
    "notes", "fragments", "selected", "popular", "famous", "strange", "curious", "remarkable", "true", "real",
    "wild", "free", "silent", "hidden", "forgotten", "broken", "golden", "sacred", "divine", "royal", "imperial",
    "national", "public", "private", "personal", "social", "political", "economic", "military", "naval",
    "civil", "religious", "moral", "natural", "physical", "chemical", "medical", "practical", "elementary",
    "general", "special", "common", "universal", "eternal", "immortal", "dead", "living", "young", "brave",
    "fair", "wise", "poor", "rich", "happy", "merry", "sad", "lonely", "proud", "humble", "gentle", "noble",
    "wicked", "evil", "cruel", "fierce", "bold", "strange", "mysterious", "invisible", "unknown", "adventuresof",
    "storyof", "lifeof", "historyof", "talesof", "bookof", "houseof", "worldof", "landof", "cityof",
})

_GUTENBERG_LIBRIVOX_SLUG_HINT_RE = re.compile(
    r"(?i)(?:_librivox\b|_gutenberg\b|\b\d{4}_librivox\b|_[a-z]{2,4}_librivox\b|-\d{4,6}-(?:epub|mp3|ogg|m4b|txt)\b)"
)
_SLUG_METADATA_STOP_TOKENS: frozenset[str] = frozenset({
    "librivox", "gutenberg", "audiobook", "unabridged", "abridged", "mp3", "m4b", "ogg", "flac",
    "epub", "mobi", "azw3", "pdf", "txt", "version", "ver", "vol", "volume", "part", "pt", "ed", "edition",
    "english", "french", "german", "deutsch", "spanish", "espanol", "italian", "italiano", "russian",
    "portuguese", "dutch", "nederlands", "polish", "swedish", "danish", "norwegian", "finnish", "greek",
    "latin", "chinese", "japanese", "korean", "esperanto", "catalan", "hungarian", "czech", "romanian",
    "turkish", "arabic", "hebrew", "indian", "roman",
})
_SLUG_NON_ENGLISH_LANG_MARKER_RE = re.compile(
    r"(?i)(?:^|[\W_])(?:french|german|deutsch|spanish|espanol|italian|italiano|russian|portuguese|dutch|"
    r"nederlands|polish|swedish|danish|norwegian|finnish|greek|latin|chinese|japanese|korean|esperanto|"
    r"catalan|hungarian|czech|romanian|turkish|arabic|hebrew)(?=$|[\W_])"
)
_CONCAT_ENGLISH_SLUG_RE = re.compile(r"^[a-z]{10,45}$")


def split_and_score_title_slug(raw_name: str) -> dict[str, Any]:
    """Segment concatenated lowercase English book/audiobook slugs (e.g. `thepaintedveil_2103_librivox`)."""
    nm = (raw_name or "").strip()
    is_librivox_or_gutenberg = bool(_GUTENBERG_LIBRIVOX_SLUG_HINT_RE.search(nm))
    has_non_english_marker = bool(_SLUG_NON_ENGLISH_LANG_MARKER_RE.search(nm))
    stem = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", nm)
    raw_tokens = [t.lower() for t in re.split(r"[\W_0-9]+", stem) if len(t) >= 2]
    tokens = [
        t for i, t in enumerate(raw_tokens)
        if t not in _SLUG_METADATA_STOP_TOKENS
        and not (is_librivox_or_gutenberg and i == len(raw_tokens) - 1 and 2 <= len(t) <= 4)
    ]
    segmented_words: list[str] = []
    found_concat_slug = False
    matched_chars = 0
    total_chars = sum(len(t) for t in tokens)
    for tok in tokens:
        if tok in _COMMON_EN_WORDS_SET:
            segmented_words.append(tok)
            matched_chars += len(tok)
            continue
        if _CONCAT_ENGLISH_SLUG_RE.match(tok):
            n = len(tok)
            dp: list[tuple[int, list[str]] | None] = [None] * (n + 1)
            dp[0] = (0, [])
            for i in range(n):
                if dp[i] is None:
                    continue
                cov_i, words_i = dp[i]  # type: ignore[misc]
                for j in range(i + 2, min(n, i + 15) + 1):
                    sub = tok[i:j]
                    if sub in _COMMON_EN_WORDS_SET:
                        new_cov = cov_i + (j - i)
                        if dp[j] is None or new_cov > dp[j][0] or (new_cov == dp[j][0] and len(words_i) + 1 < len(dp[j][1])):  # type: ignore[index]
                            dp[j] = (new_cov, words_i + [sub])
            best_cov, best_words = 0, []
            for state in dp:
                if state is not None and state[0] > best_cov:
                    best_cov, best_words = state
            if len(best_words) >= 2 and best_cov >= max(7, int(0.75 * n)):
                found_concat_slug = True
                segmented_words.extend(best_words)
                matched_chars += best_cov
    title_cov = (matched_chars / total_chars) if total_chars > 0 else 0.0
    is_english_slug = bool(
        not has_non_english_marker
        and len(segmented_words) >= 2
        and (found_concat_slug or (is_librivox_or_gutenberg and title_cov >= 0.60))
    )
    return {
        "is_english_slug": is_english_slug,
        "is_librivox_or_gutenberg": is_librivox_or_gutenberg,
        "found_concat_slug": found_concat_slug,
        "segmented_words": segmented_words,
    }


def apply_title_language_guard(
    raw_name: str,
    title_lang_probs: Sequence[float],
    lang_probs: Sequence[float] | None = None,
    anime_probs: Sequence[float] | None = None,
) -> list[float]:
    """Bidirectionally couple lang, anime, and title morphology into [P(english_title), P(non_english_title)]."""
    p_en_t = float(title_lang_probs[0])
    p_for_t = float(title_lang_probs[1]) if len(title_lang_probs) > 1 else (1.0 - p_en_t)
    nm = (raw_name or "").strip()

    slug_info = split_and_score_title_slug(nm)
    if slug_info["is_english_slug"] and not _FOREIGN_TITLE_TOKEN_RE.search(nm):
        p_en_t = max(p_en_t, 0.88)
        return [round(p_en_t, 6), round(1.0 - p_en_t, 6)]

    if _FOREIGN_TITLE_TOKEN_RE.search(nm):
        p_for_t = max(p_for_t, 0.88)
        return [round(1.0 - p_for_t, 6), round(p_for_t, 6)]

    p_non_en_audio = float(lang_probs[1]) if lang_probs is not None and len(lang_probs) > 1 else 0.0
    if p_non_en_audio >= 0.72 and p_for_t >= 0.08 and not _ENGLISH_TITLE_SHIELD_RE.search(nm) and not slug_info["is_english_slug"]:
        p_for_t = max(p_for_t, 0.78)
        return [round(1.0 - p_for_t, 6), round(p_for_t, 6)]

    return [round(p_en_t, 6), round(p_for_t, 6)]


def apply_english_subs_guard(
    row: Mapping[str, Any],
    eng_subs_probs: Sequence[float],
) -> list[float]:
    """Calibrate [P(no_eng_subs), P(has_eng_subs), P(unknown)] using physical subtitle sidecars & release tags."""
    p_no = max(0.0, float(eng_subs_probs[0])) if len(eng_subs_probs) > 0 else 0.0
    p_yes = max(0.0, float(eng_subs_probs[1])) if len(eng_subs_probs) > 1 else 0.0
    p_unk = max(0.0, float(eng_subs_probs[2])) if len(eng_subs_probs) > 2 else 0.0
    raw_name = str(row.get("name") or row.get("title") or "")
    files = row.get("files") or []

    if _EXPLICIT_ENG_SUB_TAG_RE.search(raw_name):
        return [0.06, 0.93, 0.01]

    has_authored_eng_sub = False
    has_only_machine_asr = False
    for f in files:
        fp = ((f.get("path") if isinstance(f, Mapping) else str(f or "")) or "").strip()
        low = fp.lower()
        if not low.endswith((".srt", ".vtt", ".ass", ".ssa", ".sub")):
            continue
        if low.endswith((
            ".asr.srt",
            ".asr.vtt",
            ".whisper.srt",
            ".whisper.vtt",
            ".auto.srt",
            ".auto.vtt",
            ".autogenerated.vtt",
            ".autogenerated.srt",
        )):
            has_only_machine_asr = True
            continue
        if _FOREIGN_ONLY_SUB_FILE_RE.search(low):
            continue
        has_authored_eng_sub = True
        break

    if has_authored_eng_sub:
        return [0.08, 0.91, 0.01]
    if has_only_machine_asr:
        return [0.94, 0.05, 0.01]

    tot = p_no + p_yes + p_unk
    if tot <= 1e-9:
        return [0.0, 0.0, 1.0] if len(eng_subs_probs) > 2 else [1.0, 0.0, 0.0]
    p0 = round(p_no / tot, 6)
    p1 = round(p_yes / tot, 6)
    p2 = round(max(0.0, 1.0 - p0 - p1), 6) if len(eng_subs_probs) > 2 else 0.0
    return [p0, p1, p2]


_LOCALIZED_FOREIGN_ALIAS_STEM_RE = re.compile(
    r"(?i)(?:^|[\s._\-\[\(])(?:"
    r"el[\s._-]+juego[\s._-]+del[\s._-]+calamar|"
    r"los[\s._-]+anillos[\s._-]+de[\s._-]+poder|el[\s._-]+senor[\s._-]+de[\s._-]+los[\s._-]+anillos|"
    r"juego[\s._-]+de[\s._-]+tronos|la[\s._-]+casa[\s._-]+del[\s._-]+dragon|"
    r"la[\s._-]+guerra[\s._-]+de[\s._-]+las[\s._-]+galaxias|piratas[\s._-]+del[\s._-]+caribe|"
    r"le[\s._-]+seigneur[\s._-]+des[\s._-]+anneaux|les[\s._-]+anneaux[\s._-]+de[\s._-]+pouvoir|"
    r"la[\s._-]+chronique[\s._-]+des[\s._-]+bridgerton|le[\s._-]+jeu[\s._-]+de[\s._-]+la[\s._-]+dame|"
    r"la[\s._-]+maison[\s._-]+du[\s._-]+dragon|la[\s._-]+guerre[\s._-]+des[\s._-]+etoiles|"
    r"pirates[\s._-]+des[\s._-]+caraibes|les[\s._-]+gardiens[\s._-]+de[\s._-]+la[\s._-]+galaxie|"
    r"haus[\s._-]+des[\s._-]+geldes|herr[\s._-]+der[\s._-]+ringe|ringe[\s._-]+der[\s._-]+macht|"
    r"krieg[\s._-]+der[\s._-]+sterne|fluch[\s._-]+der[\s._-]+karibik|"
    r"il[\s._-]+signore[\s._-]+degli[\s._-]+anelli|gli[\s._-]+anelli[\s._-]+del[\s._-]+potere|"
    r"il[\s._-]+trono[\s._-]+di[\s._-]+spade|guerre[\s._-]+stellari|"
    r"властелин[\s._-]+колец|игра[\s._-]+престолов|дом[\s._-]+дракона|игра[\s._-]+в[\s._-]+кальмара|"
    r"звездные[\s._-]+войны|пираты[\s._-]+карибского[\s._-]+моря|стражи[\s._-]+галактики|бумажный[\s._-]+дом"
    r")(?=$|[\s._\-\]\)])"
)
_FOREIGN_ORIGINAL_WORK_STEM_RE = re.compile(
    r"(?i)(?:^|[\s._\-\[\(])(?:"
    r"(?:la[\s._-]+)?casa[\s._-]+de[\s._-]+papel|"
    r"la[\s._-]+haine|le[\s._-]+samourai|les[\s._-]+miserables|la[\s._-]+dolce[\s._-]+vita|"
    r"la[\s._-]+vita[\s._-]+e[\s._-]+bella|el[\s._-]+laberinto[\s._-]+del[\s._-]+fauno|"
    r"el[\s._-]+secreto[\s._-]+de[\s._-]+sus[\s._-]+ojos|das[\s._-]+boot|"
    r"das[\s._-]+leben[\s._-]+der[\s._-]+anderen|der[\s._-]+untergang|ciudad[\s._-]+de[\s._-]+dios|"
    r"сталкер|солярис|зеркало|левиафан|брат"
    r")(?=$|[\s._\-\]\)])"
)
_NON_VIDEO_OR_ANIME_STEM_SHIELD_RE = re.compile(
    r"(?i)(?:\.(?:epub|pdf|mobi|azw3|cbz|cbr|mp3|flac|m4b|iso|exe|dmg|apk|zip|rar|7z)$|"
    r"\b(?:ost|soundtrack|audiobook|ebook|repack|fitgirl|dodi)\b)"
)


def detect_localized_foreign_title_stem(raw_name: str, p_anime: float = 0.0) -> tuple[bool, str]:
    """Detect whether a video release title uses a localized non-English alias stem (Owner Policy Rule 3)."""
    s = (raw_name or "").strip()
    if not s or p_anime >= 0.45 or _NON_VIDEO_OR_ANIME_STEM_SHIELD_RE.search(s):
        return False, "none"
    s_folded = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn" or "\u0400" <= ch <= "\u04ff")
    for cand in (s, s_folded):
        if _FOREIGN_ORIGINAL_WORK_STEM_RE.search(cand):
            return False, "foreign_original_work"
    for cand in (s, s_folded):
        if _LOCALIZED_FOREIGN_ALIAS_STEM_RE.search(cand):
            return True, "localized_romance_or_germanic_stem"
    return False, "none"


def is_localized_foreign_alias(
    raw_name: str,
    p_en: float = 0.5,
    p_title_foreign: float = 0.0,
    p_anime: float = 0.0,
) -> tuple[bool, str]:
    """Determine if a release uses a confirmed localized non-English title alias (Owner Policy Rule 3)."""
    nm = (raw_name or "").strip()
    if not nm or p_anime >= 0.45 or _NON_VIDEO_OR_ANIME_STEM_SHIELD_RE.search(nm):
        return False, "none"
    is_stem_alias, stem_reason = detect_localized_foreign_title_stem(nm, p_anime=p_anime)
    if is_stem_alias:
        return True, stem_reason
    return False, "none"


def evaluate_owner_policy_rule3(
    raw_name: str,
    p_en: float,
    p_title_foreign: float = 0.0,
    p_anime: float = 0.0,
    p_eng_subs_yes: float = 0.0,
    plugin_rules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate Owner Policy Rule 3: Foreign Release / Localized Alias with Unverified English."""
    nm = (raw_name or "").strip()
    rules = plugin_rules or {}
    nm_nosub = SUB_SEGMENT_STRIP_RE.sub(" ", nm)
    custom_en_re = rules.get("explicit_en_regex")
    has_explicit_eng_audio = bool(
        re.search(r"(?i)(?:^|[\[\(\.\s_-])(?:eng|english)(?=$|[\]\)\.\s_-])", nm_nosub)
        or re.search(r"[\[\(\.\s_-]EN[\]\)\.\s_-]", nm_nosub)
        or re.search(r"英语", nm_nosub)
        or re.search(r"(?i)(?:^|[\[\(\.\s_-])dual[\.\s_-]*audio(?=$|[\]\)\.\s_-])", nm_nosub)
        or safe_regex_search(custom_en_re, nm_nosub, field_name="audio_guard.explicit_en_regex")
    )
    has_explicit_eng_subs = bool(_EXPLICIT_ENG_SUB_TAG_RE.search(nm) or p_eng_subs_yes >= 0.60)
    has_verified_english_support = bool(has_explicit_eng_audio or has_explicit_eng_subs)
    alias_flag, alias_source = is_localized_foreign_alias(
        nm, p_en=p_en, p_title_foreign=p_title_foreign, p_anime=p_anime
    )
    rule3_disallow = bool(alias_flag and not has_verified_english_support)
    return {
        "is_localized_foreign_alias": alias_flag,
        "foreign_alias_source": alias_source,
        "has_explicit_eng_audio": has_explicit_eng_audio,
        "has_explicit_eng_subs": has_explicit_eng_subs,
        "has_verified_english_support": has_verified_english_support,
        "rule3_disallow": rule3_disallow,
        "rule3_reason": f"rule3_localized_alias_unverified_en:{alias_source}" if rule3_disallow else "allowed",
    }


def classify_audio_and_anime_dub_v3(
    raw_name: str,
    p_en: float,
    p_anime: float,
    p_title_foreign: float = 0.0,
    plugin_rules: Mapping[str, Any] | None = None,
    *,
    p_eng_subs_yes: float = 0.0,
) -> dict[str, Any]:
    """Classify general English-compatible audio and high-precision anime English-dub/dual-audio status."""
    nm = raw_name or ""
    rules = plugin_rules or {}
    en_thresh = float(rules.get("english_compatible_threshold", rules.get("en_threshold", 0.50)))
    anime_thresh = float(rules.get("anime_threshold", 0.50))
    dub_floor = float(rules.get("anime_dub_floor_prob", 0.85))
    extra_dub_tokens = list(rules.get("extra_dub_tokens") or [])
    extra_foreign_tokens = list(rules.get("extra_foreign_audio_tokens") or [])
    custom_dub_re = rules.get("dual_or_dub_regex")

    p_en_adj = apply_audio_language_guard_v3(
        nm,
        p_en=p_en,
        p_title_foreign=p_title_foreign,
        plugin_rules=rules,
    )
    rule3_eval = evaluate_owner_policy_rule3(
        nm,
        p_en=p_en_adj,
        p_title_foreign=p_title_foreign,
        p_anime=p_anime,
        p_eng_subs_yes=p_eng_subs_yes,
        plugin_rules=rules,
    )
    if rule3_eval["rule3_disallow"]:
        p_en_adj = min(p_en_adj, 0.15)
    is_english_compatible = p_en_adj >= en_thresh
    is_anime = p_anime >= anime_thresh

    clean_anime = SUB_SEGMENT_STRIP_RE.sub(
        " ",
        re.sub(r"(?i)\[(?:eng(?:lish)?[\s._-]*subs?|sub[\s._-]*eng)\]", " ", nm),
    )

    # Gap S3-01 fix: `[Multi-Subs]` is a subtitle tag, not a foreign-only audio tag, and must not
    # suppress explicit `[Dual-Audio]` or `[English Dub]` cues on anime releases (e.g. `[Dual-Audio][Multi-Subs]`).
    has_explicit_en_dub_or_dual = bool(
        re.search(
            r"(?i)(?:\b(?:english[\s._-]*dub(?:bed)?|eng[\s._-]*dub(?:bed)?|funidub|golumpa|arukoru)\b|"
            r"\bdual[\s._-]*audio\b|"
            r"\[esp(?:-[a-z]+)*-eng\]|"
            r"\[\s*(?:rus|ukr|jpn|jap|ita|fre|ger|spa)[^\]]*\+\s*(?:en|eng|english)\b(?![\s._\-/]*sub)|"
            r"\bita[\.\s_-]+eng\b(?![\s._\-/]*sub)|"
            r"\b[34]audio\b|"
            r"\b(?:cr|nf|hidive|amzn|funimation)[\s._-]+web-dl[\s._-]+dual\b|"
            r"\.dual\.(?:aac|ddp|ac3|flac|opus)|"
            r"\bmulti-varyg\b)",
            clean_anime,
        )
        or _match_safe_token_list(clean_anime, extra_dub_tokens)
        or safe_regex_search(custom_dub_re, clean_anime, field_name="audio_guard.dual_or_dub_regex")
    )

    has_foreign_only_anime_tag = bool(
        (
            re.search(r"[А-Яа-яЁё]{3,}|\b(?:vfi|vfq|vff|truefrench|dual[\s._-]*lat)\b", nm, re.I)
            or _match_safe_token_list(nm, extra_foreign_tokens)
        )
        and not re.search(r"(?i)\[\s*(?:rus|ukr)[^\]]*\+\s*(?:en|eng|english)\b", nm)
        and not re.search(r"(?i)\b(?:dual[\s._-]*audio|english[\s._-]*dub(?:bed)?|eng[\s._-]*dub(?:bed)?)\b", clean_anime)
    )

    has_verified_anime_dub_cue = bool(not has_foreign_only_anime_tag and has_explicit_en_dub_or_dual)
    if is_anime and has_verified_anime_dub_cue:
        p_en_adj = max(p_en_adj, dub_floor)
        is_english_compatible = p_en_adj >= en_thresh

    is_anime_dub = bool(is_anime and has_verified_anime_dub_cue)
    return {
        "p_english_compatible_audio": round(p_en_adj, 4),
        "is_english_compatible": is_english_compatible,
        "is_anime": is_anime,
        "is_anime_dub_or_dual": is_anime_dub,
        "audio_class": "english_compatible" if is_english_compatible else "non_english",
        "anime_audio_subtype": ("dub_or_dual" if is_anime_dub else "sub_or_raw") if is_anime else "not_anime",
        "is_localized_foreign_alias": rule3_eval["is_localized_foreign_alias"],
        "foreign_alias_source": rule3_eval["foreign_alias_source"],
        "owner_policy_rule3": rule3_eval,
    }


# Backward-compatible entrypoints used by armiger/ultra_bridge.py and existing tests
apply_audio_language_guard = apply_audio_language_guard_v3
classify_audio_and_anime_dub = classify_audio_and_anime_dub_v3
