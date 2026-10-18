"""Zero-Cue Catalog Lookup & Structural Disambiguation Plugin for Armiger & Armiger Ultra.

Resolves bare-title ("zero-cue") torrent releases that omit explicit season/episode syntax
(e.g. `Title (2024) [1080p].mkv`) and disambiguates multi-item media bundles, interactive
software archives, and structured instructional materials before/after neural classification.
"""
from __future__ import annotations

import hashlib
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


def hash_catalog_key(key: str) -> str:
    """Compute deterministic 64-bit keyed BLAKE2b hex digest for an offline catalog lookup key."""
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"armiger-zc-v2").hexdigest()


def norm_title(s: str) -> str:
    """Normalize a title string to NFKC lowercase alphanumeric tokens."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[\W_]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


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
                raw_name, catalog, is_anime=bool(p_anime >= 0.20)
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
    ) -> dict[str, Any]:
        idx, source = apply_structural_and_zero_cue_disambiguation(
            raw_name=raw_name,
            probs_category=probs_category,
            p_adult=p_adult,
            p_anime=p_anime,
            catalog=self.catalog,
        )
        return {
            "category_index": idx,
            "category": CATS[idx],
            "resolution_source": source,
        }
