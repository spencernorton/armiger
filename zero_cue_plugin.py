"""Zero-Cue Catalog Lookup & Structural Disambiguation Plugin for Armiger & Armiger Ultra.

Resolves bare-title ("zero-cue") torrent releases that omit explicit season/episode syntax
(e.g. `Title (2024) [1080p].mkv`) and disambiguates multi-item media bundles, interactive
software archives, and structured instructional materials before/after neural classification.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import json
import pathlib
import re
import unicodedata
from typing import Any

CATS = [
    "movie", "tv_show", "xxx", "music", "comic",
    "game", "audiobook", "software", "ebook", "course", "unknown",
]
IDX_MOVIE = 0
IDX_TV = 1
IDX_XXX = 2
IDX_MUSIC = 3
IDX_COMIC = 4
IDX_GAME = 5
IDX_AUDIOBOOK = 6
IDX_SOFTWARE = 7
IDX_EBOOK = 8
IDX_COURSE = 9

EXPLICIT_TV_RE = re.compile(
    r"(?i)(?:\bS\d{1,2}E\d{1,3}\b|\bS\d{1,2}\s*-\s*S\d{1,2}\b|\bSeason\s*(?:0?[1-9]|[1-9]\d)\b|\bStagione\s*\d+\b|\bSaison\s*\d+\b|"
    r"\bTemporada\s*\d+\b|\bStaffel\s*\d+\b|\bComplete\s+Series\b|\bTV\s*Series\b|\bMini\s*Series\b|"
    r"\bEpisode\s+[1-9]\d*\b|\bEp\.?\s*[1-9]\d*\b|\bEpiz[oó]da\b|\bЭпизод\s*\d+\b|第[一二三四五六七八九十\d]+[季集]|【[^】]*日剧】|更新至\d+[期集]|\s-\s(?:SP|NCOP|NCED)\d*\b|\b\d{3}\s*\[(?:480p|720p|1080p|2160p)\]|"
    r"\b[1-9]\d{2,3}\s*\((?:WEBRip|WEB-DL|BDRip|DVDRip|1080p|720p|480p)\b|"
    r"\bFull[\s\._]Event\b|\bPPV\b|\b1-\d{2}\s*\+\s*OVA\b|\btemp\.\d+\.completa\b|"
    r"\b\d{2}\s*-\s*\d{2,3}\s+VOSTFR\b|\s-\s\d{2,3}(?:v\d)?\s*(?:\[|\(|\b1080p\b|\b720p\b))"
)

EXPLICIT_MOVIE_PACK_RE = re.compile(
    r"(?i)(?:\bMovies?\s*\d{1,2}\s*[-~]\s*\d{1,2}\b|\bMovies?\s+Collection\b|\bTrilogy\b|\bQuadrilogy\b|\bTetralogy\b|"
    r"\bAnthology\b|\b\d+\s+Movies?\s+Pack\b|\[\d-\d\]\s*\[m?1080p\]\s*\[\d{4}-\d{4}\]|\bThe\s+Movie\b|"
    r"\(\s*Movies?\s*\)|\[\s*Movies?\s*\]|"
    r"\bMovie\s+-\s+00\b|\s-\s*Movie\s*(?:\[|\(|V\d)|\bMovie\s+\d{1,2}\s*-|\bMovie:\s|"
    r"\bGekijouban\b|\bFilm\s+Collection\b|\bIl\s+Film\b|\bLe\s+Film\b|\bDer\s+Film\b|\bKolekcja\b|"
    r"\bTheatrical\s+Cut\b|\bDirector'?s\s+Cut\b|\bExtra\s+Edition\b)"
)

GAME_ARCHIVE_RE = re.compile(
    r"(?i)(?:\bNTSC-U\b|\bNTSC-J\b|\[linux-x86\]|\[linux-x64\]|\[win64\]|"
    r"\bInteractive\s+Fiction\b|\bGOG\s+Edition\b|\bLinux\s+Native\b|\((?:Europe|USA|Japan|World)\)\.(?:zip|7z|rar|iso|chd)\b)"
)

ADULT_SITERIP_RE = re.compile(
    r"(?i)(?:\b(?:xxx|nsfw|18\+|adult[\s._-]+only)\b.*\b(?:1080p|720p|2160p|mp4|mkv|siterip)\b|"
    r"\bsiterip\b.*\b(?:xxx|18\+|adult|nsfw)\b)"
)

EXPLICIT_COMIC_RE = re.compile(
    r"(?i)(?:\b(?:cbz|cbr|cb7|fumetti)\b|^bd[\.\s_-]+fr\b)"
)

COURSE_BOOK_RE = re.compile(
    r"(?i)(?:\bTestbuch\b|\bKursbuch\b|\bArbeitsbuch\b|\bOpenCourseWare\b|\bVideo\s+Course\b|"
    r"\bData\s+Science\s+Course\b|\bLanguage\s+Course\b|\bFull\s+Tutorial\b|\bLecture\s+Series\b|\bStudy\s+Guide\b)"
)

YEAR_RE = re.compile(r"(?<![\dA-Za-z])(?:19[2-9]\d|20[0-2]\d)(?![\dA-Za-z]|\s*[xX×]\s*\d)")
NOISE_CUT_RE = re.compile(
    r"(?i)(?:\b2160p\b|\b1080p\b|\b720p\b|\b480p\b|\b576p\b|\b4k\b|\bbluray\b|\bblu-ray\b|\bweb-dl\b|\bwebdl\b|"
    r"\bwebrip\b|\bbdrip\b|\bdvdrip\b|\bdvd-remux\b|\bdvd\b|\bhdtv\b|\bremux\b|\bx264\b|\bx265\b|\bh264\b|"
    r"\bh265\b|\bhevc\b|\bav1\b|\bvostfr\b|\bmulti\b|\bdual[\s_-]*audio\b|\beng[\s_-]*sub|\bsub[\s_-]*ita|"
    r"\bita\b|\bfrench\b|\bgerman\b|\bspanish\b|\brepack\b|\bproper\b|\bremaster\b)"
)


def hash_catalog_key(key: str, hmac_key: bytes | str | None = None) -> str:
    """Compute deterministic 64-bit keyed BLAKE2b / HMAC digest for an offline catalog lookup key."""
    eff_key = hmac_key if hmac_key is not None else os.environ.get("ARMIGER_CATALOG_HMAC_KEY")
    if eff_key:
        k_bytes = eff_key.encode("utf-8") if isinstance(eff_key, str) else bytes(eff_key)
        return hmac.new(k_bytes, key.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"armiger-zc-v2").hexdigest()


def norm_title(s: str) -> str:
    """Normalize a title string to NFKC lowercase alphanumeric tokens."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


_AUTH_EPISODE_CUT_RE = re.compile(
    r"(?i)(?:\bS\d{1,2}(?:E\d{1,3}|\b)|\bSeason[\s._-]*\d+\b|\bComplete[\s._-]*Series\b|\s+-\s+\d{1,3}(?:\b|\s*[\[\(]))"
)


def extract_authoritative_title(raw_name: str) -> tuple[str, int | None]:
    """Extract the authoritative clean title and release year without relaxed sequel-number stripping."""
    s = raw_name.strip()
    if not s:
        return "", None
    s = re.sub(r"\.(?:mkv|mp4|avi|m2ts|ts|iso|zip|rar|tar\.bz2|epub|pdf|mp3|flac)$", "", s, flags=re.I)
    s_no_lead = re.sub(r"^(?:\s*[\[\(][^\]\)]{1,45}[\]\)]\s*|[_\s]+)+", "", s)
    base = s_no_lead if len(norm_title(s_no_lead)) >= 2 else s

    year_matches = [m for m in YEAR_RE.finditer(base) if m.start() >= 2]
    year = int(year_matches[0].group(0)) if year_matches else None

    cut_pos = len(base)
    if year_matches:
        cut_pos = min(cut_pos, year_matches[0].start())
    m_ns = NOISE_CUT_RE.search(base)
    if m_ns and m_ns.start() >= 2:
        cut_pos = min(cut_pos, m_ns.start())
    m_ep = _AUTH_EPISODE_CUT_RE.search(base)
    if m_ep and m_ep.start() >= 2:
        cut_pos = min(cut_pos, m_ep.start())

    stem = base[:cut_pos]
    stem = re.split(r"[\[\(\|]", stem)[0]
    clean = norm_title(stem)
    if not clean:
        clean = norm_title(re.split(r"[\[\(\|]", base)[0])
    return clean, year


def parse_clean_candidates(raw_name: str, extra_spans: list[str] | None = None) -> tuple[list[tuple[str, int | None]], int | None]:
    """Extract ordered candidate `(clean_title, year)` pairs sorted longest/most-specific first."""
    s = raw_name.strip()
    s = re.sub(r"\.(?:mkv|mp4|avi|m2ts|ts|iso|zip|rar|tar\.bz2|epub|pdf|mp3|flac)$", "", s, flags=re.I)
    s_no_lead = re.sub(r"^(?:\s*[\[\(][^\]\)]{1,45}[\]\)]\s*|[_\s]+)+", "", s)
    years = [int(m.group(0)) for m in YEAR_RE.finditer(s_no_lead)]
    year = years[0] if years else None

    raw_cands: list[tuple[str, int | None]] = []

    def add(ct: str, cy: int | None) -> None:
        ct = ct.strip()
        if len(ct) >= 4:
            raw_cands.append((ct, cy))

    for base in (s_no_lead, s):
        m_yr = YEAR_RE.search(base)
        if m_yr and m_yr.start() >= 3:
            prefix = base[:m_yr.start()]
            c1 = norm_title(prefix)
            add(c1, int(m_yr.group(0)))
            add(c1, None)
            for subp in re.split(r"(?i)\s+-\s+|\s+aka\s+", prefix):
                cs = norm_title(subp)
                if len(cs) >= 5:
                    add(cs, int(m_yr.group(0)))
        m_ns = NOISE_CUT_RE.search(base)
        if m_ns and m_ns.start() >= 3:
            c2 = norm_title(base[:m_ns.start()])
            add(c2, year)
            add(c2, None)
        c3_raw = re.split(r"[\[\(\|]", base)[0]
        c3 = norm_title(c3_raw)
        add(c3, year)
        add(c3, None)
        c4 = norm_title(re.sub(r"(?i)(?:\b(?:the\s+movie|movie|ova|oav|special)\b|\s+-\s+\d{1,3}|\s+\d{1,3})\s*$", "", c3_raw))
        add(c4, year)
        add(c4, None)

    for grp in re.findall(r"\(([^()]{5,80})\)", s_no_lead) + s_no_lead.split("|")[1:]:
        if not NOISE_CUT_RE.search(grp) and not YEAR_RE.fullmatch(grp.strip()):
            for part in re.split(r"[/｜|]", grp):
                ca = norm_title(part)
                if len(ca) >= 6:
                    add(ca, year)
                    add(ca, None)

    if extra_spans:
        for sp in sorted(extra_spans, key=lambda x: len(x or ""), reverse=True):
            nsp = norm_title(sp)
            if len(nsp) >= 5:
                add(nsp, year)
                add(nsp, None)

    raw_cands.sort(key=lambda item: (item[1] is not None, len(item[0])), reverse=True)
    ordered: list[tuple[str, int | None]] = []
    seen: set[tuple[str, int | None]] = set()
    for item in raw_cands:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered, year


def lookup_zero_cue_catalog(
    raw_name: str,
    catalog: dict[str, Any],
    is_anime: bool = True,
    extra_spans: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """Query the local validated catalog in strict specificity order:
    1. `(title, year)` exact match in `title_year_map` (472,017 entries)
    2. `title` match in `anime_map` when `is_anime=True` and stem length >= 8 (257,016 entries)
    3. `title` match in unambiguous `title_only_map` (200,367 entries)
    """
    cands, yr = parse_clean_candidates(raw_name, extra_spans=extra_spans)
    title_year_map = catalog.get("title_year_map", {})
    anime_map = catalog.get("anime_map", {})
    title_only_map = catalog.get("title_only_map", {})

    for ct, cy in cands:
        if cy is not None and len(ct) >= 4:
            key = f"{ct}|{cy}"
            hkey = hash_catalog_key(key)
            if hkey in title_year_map:
                return title_year_map[hkey], "title_year"
            if key in title_year_map:
                return title_year_map[key], "title_year"

    if is_anime:
        for ct, _ in cands:
            if len(ct) >= 8:
                hct = hash_catalog_key(ct)
                if hct in anime_map:
                    return anime_map[hct], "anime_catalog"
                if ct in anime_map:
                    return anime_map[ct], "anime_catalog"

    for ct, _ in cands:
        if len(ct) >= 14:
            hct = hash_catalog_key(ct)
            if hct in title_only_map:
                return title_only_map[hct], "title_only"
            if ct in title_only_map:
                return title_only_map[ct], "title_only"

    if cands and yr is None:
        ct0 = cands[0][0]
        if len(ct0) >= 7:
            hct0 = hash_catalog_key(ct0)
            if hct0 in title_only_map:
                return title_only_map[hct0], "title_only"
            if ct0 in title_only_map:
                return title_only_map[ct0], "title_only"

    return None, None


def apply_structural_and_zero_cue_disambiguation(
    raw_name: str,
    probs_category: list[float] | Any,
    p_adult: float,
    p_anime: float,
    catalog: dict[str, Any] | None = None,
    extra_spans: list[str] | None = None,
) -> tuple[int, str]:
    """Apply deterministic structural rules (Stage 1) and Zero-Cue catalog lookup (Stage 2)
    to an 11-class category probability vector from Armiger or Armiger Ultra.

    Returns `(predicted_category_index, resolution_source)`.
    """
    p = [float(x) for x in probs_category]
    has_tv_cue = bool(EXPLICIT_TV_RE.search(raw_name))
    has_movie_cue = bool(EXPLICIT_MOVIE_PACK_RE.search(raw_name))

    if ADULT_SITERIP_RE.search(raw_name):
        p[IDX_XXX] = max(p[IDX_XXX], 0.92)
        return IDX_XXX, "structural_adult_siterip"

    if EXPLICIT_COMIC_RE.search(raw_name) and not re.search(r"(?i)how\s+to\s+draw", raw_name):
        top_i = int(max(range(len(p)), key=p.__getitem__))
        if top_i in (IDX_EBOOK, IDX_COMIC) or p[top_i] < 0.90:
            p[IDX_COMIC] = max(p[IDX_COMIC], 0.90)
            return IDX_COMIC, "structural_comic_archive"

    if has_movie_cue and not has_tv_cue:
        if p[IDX_MOVIE] + p[IDX_TV] > 0.45:
            p[IDX_MOVIE] = max(p[IDX_MOVIE], p[IDX_TV] + 0.40)
            return int(max(range(len(p)), key=p.__getitem__)), "structural_movie_pack"
    elif has_tv_cue and not has_movie_cue:
        if p[IDX_XXX] < 0.50 and p_adult < 0.65:
            p[IDX_TV] = max(p[IDX_TV], p[IDX_MOVIE] + 0.40, 0.75)
            return int(max(range(len(p)), key=p.__getitem__)), "structural_tv_cue"

    if GAME_ARCHIVE_RE.search(raw_name) and p_adult < 0.70:
        p[IDX_GAME] = max(p[IDX_GAME], 0.88)
        return IDX_GAME, "structural_game_archive"

    if COURSE_BOOK_RE.search(raw_name):
        p[IDX_COURSE] = max(p[IDX_COURSE], 0.88)
        return IDX_COURSE, "structural_course_book"

    pred_idx = int(max(range(len(p)), key=p.__getitem__))
    if catalog is not None and pred_idx in (IDX_MOVIE, IDX_TV) and not has_tv_cue and not has_movie_cue:
        if max(p[IDX_MOVIE], p[IDX_TV]) < 0.88:
            cat_hit, src_kind = lookup_zero_cue_catalog(
                raw_name, catalog, is_anime=bool(p_anime >= 0.20), extra_spans=extra_spans
            )
            if cat_hit in ("movie", "tv_show") and src_kind is not None:
                return (IDX_MOVIE if cat_hit == "movie" else IDX_TV), f"zero_cue_{src_kind}"

    return pred_idx, "neural_backbone"


class ZeroCueCatalogPlugin:
    """High-level wrapper for loading `validated_zero_cue_catalog.json` and disambiguating predictions."""

    def __init__(self, catalog_path: str | pathlib.Path | None = None) -> None:
        if catalog_path is None:
            catalog_path = pathlib.Path(__file__).with_name("validated_zero_cue_catalog.json")
        self.catalog_path = pathlib.Path(catalog_path)
        self.catalog: dict[str, Any] = (
            json.loads(self.catalog_path.read_text()) if self.catalog_path.exists() else {}
        )

    def resolve(
        self,
        raw_name: str,
        probs_category: list[float] | Any,
        p_adult: float = 0.0,
        p_anime: float = 0.0,
        extra_spans: list[str] | None = None,
    ) -> dict[str, Any]:
        idx, source = apply_structural_and_zero_cue_disambiguation(
            raw_name=raw_name,
            probs_category=probs_category,
            p_adult=p_adult,
            p_anime=p_anime,
            catalog=self.catalog,
            extra_spans=extra_spans,
        )
        if source == "neural_backbone":
            idx_v3, src_v3, probs_v3 = resolve_category_v3(
                raw_name=raw_name,
                probs_category=probs_category,
                p_adult=p_adult,
                p_anime=p_anime,
                catalog=self.catalog,
                extra_spans=extra_spans,
            )
            if src_v3 != "v3_neural_calibrated":
                return {
                    "category_index": idx_v3,
                    "category": CATS[idx_v3],
                    "resolution_source": src_v3,
                    "calibrated_probs": probs_v3,
                }
        return {
            "category_index": idx,
            "category": CATS[idx],
            "resolution_source": source,
        }


# --- Armiger v3.0 Structural Disambiguator & Prior-Corrected Logit Calibrator ---

_V3_COMIC_RE = re.compile(
    r"(?i)(?:\.(?:cbr|cbz|cb7)$|\b(?:cbr|cbz|cb7)\b|\([0-9]{4}\)\s*\(digital\)|\bc2c\b.*\.(?:cbr|cbz|rar|zip)$)"
)
_V3_AUDIOBOOK_RE = re.compile(
    r"(?i)(?:\.m4b$|\baudiobook\b|\bunabridged\b|\babridged\b)"
)
_V3_EBOOK_RE = re.compile(
    r"(?i)(?:\.(?:epub|mobi|azw3)$|\b(?:epub|mobi|azw3)\b)"
)
_V3_GAME_STRONG_RE = re.compile(
    r"(?i)(?:\.(?:nsp|xci|cia|3ds|vpk|wbfs)$|\b(?:fitgirl|dodi[\s._-]*repack|elamigos|kaos[\s._-]*repack|"
    r"codex|skidrow|plaza|flt|rune|tenoke|razor1911|reloaded|cpy|empress|gog|rg[\s._-]*mechanics)\b|"
    r"(?:^|[\s._\-\[\(])plus[\s._-]+\d+[\s._-]+trainer(?=$|[\s._\-\]\)])|\bbuild[\s._-]+\d{5,9}\b|\bps[2345]\b.*\b(?:usa|eur|jpn|pkg|iso)\b|"
    r"\bxbox\s*360\b|\bnintendo\s+switch\b|\brpg\s+maker\b)"
)
_V3_COURSE_RE = re.compile(
    r"(?i)(?:(?:^|[\s._\-\[\(])(?:udemy|coursera|pluralsight|lynda|linkedin[\s._-]*learning|skillshare|masterclass|"
    r"the[\s._-]+great[\s._-]+courses|ttc[\s._-]+video|packt|oreilly|cbt[\s._-]+nuggets|ine[\s._-]+ccie|"
    r"laracasts|frontend[\s._-]+masters|egghead)(?=$|[\s._\-\]\)])|"
    r"(?:^|[\s._\-\[\(])(?:testbuch|kursbuch|arbeitsbuch|lehrbuch|übungsbuch)(?=$|[\s._\-\]\)])|"
    r"(?:^|[\s._\-\[\(])(?:ucberkeley|mit)[\s._-]+(?:webcast|opencourseware)(?=$|[\s._\-\]\)])|"
    r"(?:^|[\s._\-\[\(])(?:lecture[\s._-]+series|video[\s._-]+training|video[\s._-]+course|full[\s._-]+tutorial|study[\s._-]+guide|open\s*course\s*ware)(?=$|[\s._\-\]\)]))"
)
_V3_SOFTWARE_RE = re.compile(
    r"(?i)(?:\b(?:macos|macosx|mac\s+os\s+x|win32|win64|x64|x86_64|multilingual|pre-activated|"
    r"crack|keygen|patch|portable|bootable)\b.*\b(?:v?\d+\.\d+|20[12]\d)\b|"
    r"\b(?:adobe|autodesk|jetbrains|microsoft\s+office|windows\s+(?:10|11|server)|"
    r"vmware|parallels|ableton|fl\s+studio|native\s+instruments|kontakt|waves\s+complete)\b)"
)
_V3_MUSIC_RE = re.compile(
    r"(?i)(?:\.(?:flac|alac|ape|wv|dsf|dff)$|\b(?:discography|discografia|ost|soundtrack|"
    r"lossless|24bit|16bit|44\.1khz|96khz|192khz|cd-rip|vinyl[\s._-]*rip|web-flac|mp3[\s._-]*320)\b)"
)
_V3_CONCERT_OR_DOC_VIDEO_RE = re.compile(
    r"(?i)(?:^|[\s._\-\[\(])(?:philharmoniker|symphony[\s._-]+orchestra|live[\s._-]+in[\s._-]+[a-z]+|live[\s._-]+at[\s._-]+[a-z]+)(?=$|[\s._\-\]\)]).*\b(?:1080i|1080p|720p|bluray|mkv|ts)\b"
)
_V3_ADULT_STUDIO_RE = re.compile(
    r"(?i)(?:\b(?:xxx|siterip|onlyfans|manyvids|brazzers|naughtyamerica|realitykings|mofos|"
    r"bangbros|vixen|tushy|blacked|deeper|slayed|metart|sexart|hegre|twistys|playboy|penthouse|"
    r"jav|fc2[\s._-]*ppv|uncensored[\s._-]*leak|1pondo|caribbeancom|heyzo|pacopacomama)\b|"
    r"\b[a-z]{2,5}-\d{3,5}\b.*\b(?:mp4|mkv|wmv|avi|1080p|720p|fhd|hd)\b)"
)


def _boost_and_normalize_v3(p: list[float], idx: int, min_prob: float) -> list[float]:
    target = min(0.99, max(float(p[idx]), float(min_prob)))
    rem_budget = max(0.0, 1.0 - target)
    other_sum = sum(float(p[i]) for i in range(len(p)) if i != idx)
    out: list[float] = []
    for i in range(len(p)):
        if i == idx:
            out.append(round(target, 6))
        elif other_sum > 0.0:
            out.append(round(rem_budget * (float(p[i]) / other_sum), 6))
        else:
            out.append(round(rem_budget / max(1, len(p) - 1), 6))
    diff = round(1.0 - sum(out), 6)
    out[idx] = round(max(0.0, min(1.0, out[idx] + diff)), 6)
    return out


def calibrate_category_logits_v3(
    logits_or_probs: list[float] | Any,
    temperature: float = 0.85,
) -> list[float]:
    """Apply prior-shift log-odds correction and temperature scaling to 11-class category probabilities."""
    import math

    arr = [float(x) for x in logits_or_probs]
    s = sum(arr)
    is_prob = all(x >= 0.0 for x in arr) and (0.98 <= s <= 1.02)
    if is_prob:
        logits = [math.log(max(x, 1e-12)) for x in arr]
    else:
        logits = list(arr)

    prior_boost = [0.0, 0.0, 0.0, 0.05, 0.15, 0.12, 0.15, 0.12, 0.15, 0.18, -0.35]
    scaled = [(logits[i] + prior_boost[i]) / max(temperature, 0.1) for i in range(min(len(logits), 11))]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps) or 1.0
    out = [round(x / z, 6) for x in exps]
    if out:
        diff = round(1.0 - sum(out), 6)
        best_i = max(range(len(out)), key=out.__getitem__)
        out[best_i] = round(max(0.0, min(1.0, out[best_i] + diff)), 6)
    return out


def resolve_category_v3(
    raw_name: str,
    probs_category: list[float] | Any,
    p_adult: float = 0.0,
    p_anime: float = 0.0,
    catalog: dict[str, Any] | None = None,
    extra_spans: list[str] | None = None,
) -> tuple[int, str, list[float]]:
    """Armiger v3.0 11-class category resolver combining calibrated neural probabilities,
    structural/format rules, and zero-cue BLAKE2b-64 catalog lookup.

    Returns `(category_index, resolution_source, calibrated_probs)`.
    """
    p = calibrate_category_logits_v3(probs_category)
    s = raw_name.strip()

    has_tv_cue = bool(EXPLICIT_TV_RE.search(s))
    has_movie_cue = bool(EXPLICIT_MOVIE_PACK_RE.search(s))
    has_video_ext_or_codec = bool(
        re.search(
            r"(?i)(?:\.(?:mkv|mp4|avi|m2ts|ts|wmv|mov)$|\b(?:1080p|720p|2160p|480p|bluray|web-dl|webrip|bdrip|dvdrip|x264|x265|hevc|h264|h265|xvid)\b)",
            s,
        )
    )

    if ADULT_SITERIP_RE.search(s) or (_V3_ADULT_STUDIO_RE.search(s) and p_adult >= 0.25):
        return IDX_XXX, "v3_structural_adult", _boost_and_normalize_v3(p, IDX_XXX, 0.94)

    if _V3_COMIC_RE.search(s) and not re.search(r"(?i)how\s+to\s+draw", s):
        return IDX_COMIC, "v3_structural_comic", _boost_and_normalize_v3(p, IDX_COMIC, 0.92)

    if _V3_AUDIOBOOK_RE.search(s):
        return IDX_AUDIOBOOK, "v3_structural_audiobook", _boost_and_normalize_v3(p, IDX_AUDIOBOOK, 0.92)

    if _V3_COURSE_RE.search(s) or COURSE_BOOK_RE.search(s):
        return IDX_COURSE, "v3_structural_course", _boost_and_normalize_v3(p, IDX_COURSE, 0.91)

    if _V3_EBOOK_RE.search(s) and not _V3_COMIC_RE.search(s) and not _V3_AUDIOBOOK_RE.search(s):
        return IDX_EBOOK, "v3_structural_ebook", _boost_and_normalize_v3(p, IDX_EBOOK, 0.91)

    if (GAME_ARCHIVE_RE.search(s) or _V3_GAME_STRONG_RE.search(s)) and p_adult < 0.70 and not has_tv_cue:
        if not re.search(r"(?i)\b(?:ost|soundtrack|score|flac|mp3)\b", s):
            return IDX_GAME, "v3_structural_game", _boost_and_normalize_v3(p, IDX_GAME, 0.90)

    if _V3_CONCERT_OR_DOC_VIDEO_RE.search(s) and not has_tv_cue:
        return IDX_MOVIE, "v3_structural_video_override", _boost_and_normalize_v3(p, IDX_MOVIE, 0.88)

    if _V3_MUSIC_RE.search(s) and not has_video_ext_or_codec and not has_tv_cue:
        return IDX_MUSIC, "v3_structural_music", _boost_and_normalize_v3(p, IDX_MUSIC, 0.90)

    if _V3_SOFTWARE_RE.search(s) and not has_video_ext_or_codec and not has_tv_cue and p_adult < 0.50:
        return IDX_SOFTWARE, "v3_structural_software", _boost_and_normalize_v3(p, IDX_SOFTWARE, 0.89)

    if has_movie_cue and not has_tv_cue:
        if p[IDX_MOVIE] + p[IDX_TV] > 0.40:
            floor_m = min(0.99, max(p[IDX_MOVIE], p[IDX_TV] + 0.40, 0.85))
            return IDX_MOVIE, "v3_structural_movie_pack", _boost_and_normalize_v3(p, IDX_MOVIE, floor_m)
    elif has_tv_cue and not has_movie_cue:
        if p[IDX_XXX] < 0.50 and p_adult < 0.65:
            floor_t = min(0.99, max(p[IDX_TV], p[IDX_MOVIE] + 0.40, 0.85))
            return IDX_TV, "v3_structural_tv_cue", _boost_and_normalize_v3(p, IDX_TV, floor_t)

    pred_idx = int(max(range(len(p)), key=p.__getitem__))
    if catalog is not None and pred_idx in (IDX_MOVIE, IDX_TV) and not has_tv_cue and not has_movie_cue:
        if max(p[IDX_MOVIE], p[IDX_TV]) < 0.93:
            cat_hit, src_kind = lookup_zero_cue_catalog(
                s, catalog, is_anime=bool(p_anime >= 0.15), extra_spans=extra_spans
            )
            if cat_hit in ("movie", "tv_show") and src_kind is not None:
                idx_hit = IDX_MOVIE if cat_hit == "movie" else IDX_TV
                return idx_hit, f"v3_zero_cue_{src_kind}", _boost_and_normalize_v3(p, idx_hit, 0.90)

    return pred_idx, "v3_neural_calibrated", p
