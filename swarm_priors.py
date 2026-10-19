"""Tracker Scrape, Swarm Liveliness & Bayesian Grab-Outcome Priors Integration for Armiger.

Implements the four strictly separated signal pathways required by the BitAgent x Armiger
architecture contract:
  - Way 1: Active Teacher Sampling & Queue Priority Weighting (`compute_sampling_priority_weight`)
  - Way 2: Tracker/Source Identity & Bayesian Prior Features (`extract_way2_source_features`)
           with Mandatory Modality Dropout (`apply_modality_dropout`, p in [0.50, 0.70])
           and strict invariance to `seeders == 0` / `liveness == 'dead'`
  - Way 3: Swarm-vs-Content Anomaly Features for `junk` / Malware / Honeypot Detection
           (`extract_way3_anomaly_features`)
  - Way 4: Orthogonal Swarm Viability & Stage-2 Grab Ranking Multiplier
           (`compute_swarm_viability`), guaranteeing `seeders == 0` never flips content `category`.

Also enforces the BitAgent privacy invariant (`catalogueguard.PrivacyTagSQL` / `privacySQL`):
private-tracker names, categories (`private`, `bitgrab`), and passkey URLs are rejected/stripped
and never exported to external or unapproved stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import time
import math
import random
import re
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - stdlib fallback for bare plugin supervisor
    np = None  # type: ignore[assignment]

# Canonical top-9 public BEP-15 UDP tracker domains in the BitAgent scrape pool (ordered by corpus volume)
PUBLIC_TRACKER_DOMAINS: tuple[str, ...] = (
    "tracker.opentrackr.org",
    "open.stealth.si",
    "tracker.torrent.eu.org",
    "tracker.qu.ax",
    "open.demonii.com",
    "evan.im",
    "tracker.bittor.pw",
    "tracker.dler.org",
    "explodie.org",
)
_TRACKER_INDEX = {d: i for i, d in enumerate(PUBLIC_TRACKER_DOMAINS)}

# Empirical category/anime/language/junk prior affinities per public tracker domain
# (derived from the 79,736-record teacher-labeled training corpus; invariant to individual torrent seeders count)
TRACKER_DOMAIN_PRIORS: dict[str, tuple[float, float, float, float]] = {
    # domain: (anime_affinity, audiobook_affinity, game_sw_affinity, low_junk_shield)
    "tracker.opentrackr.org": (0.244, 0.047, 0.088, 0.983),
    "open.stealth.si":        (0.332, 0.037, 0.052, 0.992),
    "tracker.torrent.eu.org": (0.168, 0.275, 0.068, 0.990),
    "tracker.qu.ax":          (0.063, 0.025, 0.167, 0.983),
    "open.demonii.com":       (0.098, 0.031, 0.134, 0.989),
    "evan.im":                (0.015, 0.033, 0.106, 0.970),
    "tracker.bittor.pw":      (0.000, 0.025, 0.190, 0.988),
    "tracker.dler.org":       (0.032, 0.086, 0.219, 0.965),
    "explodie.org":           (0.069, 0.017, 0.103, 0.972),
}

# Privacy guard patterns matching bitagent/internal/catalogueguard and internal/seeds/admission.go
_PRIVATE_SOURCE_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:private|bitgrab|passkey|authkey|torrentleech|broadcastthenet|broadcasthenet|passthepopcorn|"
    r"redacted|orpheus|bibliotik|myanonamouse|filelist|beyondhd|blutopia|uhdbits|morethantv|alpharatio|"
    r"iptorrents|hdbits|gazellegames|animebytes|avistaz|cinemaz|exoticaz|empornium|empirnium)(?:$|[^a-z0-9])"
)
_PASSKEY_QUERY_RE = re.compile(
    r"(?i)(?:[?&]|://[^/\s@]+@|/[a-f0-9]{16,}(?:/|$)|/[a-z0-9_-]{24,}(?:/|$))"
)
_SAFE_URL_PATH_RE = re.compile(r"(?i)^(?:/(?:announce(?:\.php)?|scrape(?:\.php)?)?/?)?$")

# Feature extractors matching bitagent/internal/evidence/priors/features.go
_RG_BRACKET_RE = re.compile(r"^\[([^\[\]]{2,32})\]")
_RG_SUFFIX_RE = re.compile(r"-([A-Za-z0-9]+)(?:\.[a-z0-9]{2,5})?$")
_RES_RE = re.compile(r"(?i)\b(2160p|1080p|720p|480p|576p|4k)\b")
_QUAL_RE = re.compile(r"(?i)\b(web-dl|webdl|webrip|bluray|blu-ray|bdrip|brrip|hdtv|dvdrip|remux|hdrip|cam|ts|telesync)\b")
_CODEC_RE = re.compile(r"(?i)\b(x265|h265|hevc|x264|h264|avc|av1|xvid|divx)\b")
_EXEC_OR_TRAP_EXTS = frozenset({
    "exe", "msi", "bat", "cmd", "sh", "com", "scr", "lnk", "vbs", "vbe", "hta", "pif", "ps1", "wsf", "wsh",
    "reg", "html", "htm", "url", "mht", "dll", "cpl", "one", "xll", "xlam",
    "appref-ms", "library-ms", "search-ms", "settingcontent-ms", "diagcab",
})
_ALWAYS_TRAP_EXTS_SP = frozenset({
    "scr", "vbs", "vbe", "hta", "pif", "wsf", "wsh", "cpl",
    "appref-ms", "library-ms", "search-ms", "settingcontent-ms", "diagcab",
})
_VIDEO_EXTS_SP = frozenset({"mkv", "mp4", "avi", "mov", "wmv", "m4v", "webm", "ts", "m2ts", "mpg", "mpeg", "flv"})
_LAUNCHER_EXTS_SP = frozenset({"scr", "lnk", "hta", "vbs", "vbe", "pif", "ps1", "wsf", "wsh", "bat", "cmd", "com", "msi", "chm", "one", "xll", "xlam"})
_BENIGN_CONTAINER_EXTS_SP = frozenset({"vcdiff", "apk", "bdmv", "clpi", "mpls", "ifo", "bup", "vob", "npa", "vpk", "pck", "umd", "pak", "xp3", "rgss3a", "wolf", "nsp", "xci", "wbfs", "rpa", "rpy", "rpyb"})
_VID_OR_ANIME_CLAIM_SP_RE = re.compile(
    r"(?i)(?:\b(?:1080p|720p|2160p|480p|4k|bluray|blu-ray|bdrip|brrip|web-dl|webdl|webrip|hdtv|dvdrip|hdrip|x26[45]|h\.?26[45]|hevc|avc|s\d{1,2}[._ -]?e\d{1,3}|\d{1,2}x\d{2}|season[ ._-]*\d+)\b|"
    r"^\[(?:SubsPlease|Erai-raws|HorribleSubs|Judas|Ember|ToonsHub|ASW|Sakurato|LoliHouse|Nekomoe|ANi|B-Global|Ohys-Raws|Leopard-Raws|ReinForce|Moozzi2|Beatrice-Raws|OpenAnime)\])"
)
_SW_OR_GAME_TITLE_SP_RE = re.compile(
    r"(?i)\b(?:v?\d+\.\d+\.\d+|portable|cracked|keygen|patch|patcher|activat(?:ed|or)|"
    r"multilingual|win(?:32|64)|x64|x86|macos|android|apk|pc[ ._-]?game|fitgirl|dodi|elamigos|"
    r"codex|rune|tenoke|skidrow|gog|emulator|driver|firmware|setup|installer|downloader|"
    r"pro[ ._-]+v?\d|ce[ ._-]+v?\d)\b"
)
_ANIME_OR_EP_ZIP_SP_RE = re.compile(
    r"(?i)(?:^\[[^\[\]]{2,30}\].*?(?:\s-\s*\d{1,3}\b|\bS\d{1,2}E\d{1,3}\b).*?\b(?:1080p|720p|2160p|480p)\b|"
    r"\b(?:S\d{1,2}[._ -]?E\d{1,3}|\d{1,2}x\d{2})\b.*?\b(?:1080p|720p|2160p|480p|WEB-DL|WEBRip|BluRay|BDRip|HDTV|x264|x265|HEVC)\b|"
    r"\b(?:1080p|720p|2160p|480p|576p|4k)\b.*?\b(?:WEB-DL|WEBRip|BluRay|BDRip|BRRip|HDRip|HDTV|HDCAM|CAMRip|DVDRip|x264|x265|H\.?264|H\.?265|HEVC|AV1|XviD|DivX)\b|"
    r"\b(?:WEB-DL|WEBRip|BDRip|BRRip|HDRip|HDTV|HDCAM|CAMRip|DVDRip)\b(?:.*?\b(?:x264|x265|H\.?264|H\.?265|HEVC|AV1|XviD|DivX))?)"
)
_OPTICAL_DISC_SP_RE = re.compile(
    r"(?i)\b(?:dvd(?:[59]|r|-r|-rom)?|bd(?:25|50|66|100|mv|rom|-r)|blu-?ray[\s._-]*disc|full[\s._-]*bd|untouched|remux|pal|ntsc|iso|udf|sacd|dvd-?audio|3d[\s._-]*iso)\b"
)
_BENIGN_ZIP_SP_RE = re.compile(
    r"(?i)\b(?:batch|complete|season[ ._-]*\d+|s\d{1,2}[ ._-]*complete|e?\d{1,3}[ ._-]*[~-][ ._-]*e?\d{1,3}|시즌\s*\d+|완결|全集|合集|"
    r"ost|soundtrack|artbook|scans?|wallpapers?|fonts?|subs?|subtitles?)\b"
)
_STOP_RG = frozenset({
    "x264", "x265", "h264", "h265", "hevc", "avc", "aac", "ac3", "dts", "ddp5", "web", "dl",
    "webrip", "bluray", "1080p", "720p", "2160p", "480p", "576p", "mp4", "mkv", "avi",
    "multi", "dual", "dubbed", "subbed", "proper", "repack", "internal", "remux",
})


def normalise_source(s: str | None) -> str:
    """Exact Python parity with `bitagent/internal/evidence/priors/features.go:normaliseSource`.

    Strips Prowlarr's ` (Prowlarr)` suffix, lowercases, and collapses whitespace, `/`, and `.` to `_`.
    """
    if not s or not isinstance(s, str):
        return ""
    v = s.strip().lower()
    if v.endswith("(prowlarr)"):
        v = v[:-len("(prowlarr)")].strip()
    out: list[str] = []
    prev_underscore = False
    for ch in v:
        if ch in (" ", "\t", "/", "."):
            if not prev_underscore and out:
                out.append("_")
                prev_underscore = True
        else:
            out.append(ch)
            prev_underscore = (ch == "_")
    return "".join(out).strip("_")


def normalise_tracker_domain(url_or_host: str | None) -> str:
    """Extract normalized public hostname from a BEP-15 UDP/HTTP tracker announce URL.

    Returns `'none'` if empty or if blocked by the privacy filter.
    """
    if not url_or_host or not isinstance(url_or_host, str):
        return "none"
    raw = url_or_host.strip()
    if not raw or not is_public_safe_source(raw):
        return "none"
    u = raw.lower()
    if "://" in u:
        u = u.split("://", 1)[1]
    u = u.split("/", 1)[0].split("?", 1)[0].split(":", 1)[0].strip()
    if not u or not is_public_safe_source(u):
        return "none"
    return u


def is_public_safe_source(source_or_url: str | None) -> bool:
    """Return True iff `source_or_url` contains no private-tracker identifiers, categories, queries, or path passkeys."""
    if not source_or_url or not isinstance(source_or_url, str):
        return False
    s = source_or_url.strip()
    if not s:
        return False
    if _PASSKEY_QUERY_RE.search(s):
        return False
    if _PRIVATE_SOURCE_RE.search(s):
        return False
    if "://" in s or "/" in s:
        u = s.split("://", 1)[1] if "://" in s else s
        slash_idx = u.find("/")
        if slash_idx != -1:
            path_part = u[slash_idx:]
            if not _SAFE_URL_PATH_RE.fullmatch(path_part):
                return False
    norm = normalise_source(s)
    if norm in ("private", "bitgrab") or _PRIVATE_SOURCE_RE.search(norm):
        return False
    return True


def sanitize_sources_for_export(sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Filter a list of source observations so private-tracker names/tags are never exported."""
    if not sources:
        return []
    safe: list[dict[str, Any]] = []
    for item in sources:
        if not isinstance(item, dict):
            continue
        if item.get("private") is True:
            continue
        src = item.get("source")
        if not isinstance(src, str) or not is_public_safe_source(src):
            continue
        dom = normalise_tracker_domain(src) if "://" in src else "none"
        norm = normalise_source(dom if dom != "none" else src)
        if not norm:
            continue
        entry: dict[str, Any] = {"source": norm}
        for k in ("seeders", "leechers", "completed"):
            val = item.get(k)
            if type(val) is int and val >= 0:
                entry[k] = val
        safe.append(entry)
    return safe


def extract_release_group(name: str | None) -> str:
    """Extract normalized release group matching `bitagent/internal/evidence/priors/features.go`."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip()
    m = _RG_BRACKET_RE.match(s)
    if m:
        cand = m.group(1).strip().lower()
        if cand not in _STOP_RG and len(cand) >= 2:
            return cand
    m = _RG_SUFFIX_RE.search(s)
    if m:
        cand = m.group(1).strip().lower()
        if cand not in _STOP_RG and not cand.isdigit() and 2 <= len(cand) <= 24:
            return cand
    return ""


def extract_resolution(name: str | None) -> str:
    m = _RES_RE.search(name or "")
    if not m:
        return ""
    v = m.group(1).lower()
    return "2160p" if v == "4k" else v


def extract_quality_tag(name: str | None) -> str:
    m = _QUAL_RE.search(name or "")
    if not m:
        return ""
    v = m.group(1).lower()
    if v in ("webdl", "web-dl"):
        return "web-dl"
    if v in ("blu-ray", "bluray", "bdrip", "brrip"):
        return "bluray"
    return v


def extract_codec(name: str | None) -> str:
    m = _CODEC_RE.search(name or "")
    if not m:
        return ""
    v = m.group(1).lower()
    if v in ("h265", "hevc", "x265"):
        return "x265"
    if v in ("h264", "avc", "x264"):
        return "x264"
    return v


def normalise_extension(ext: str | None) -> str:
    if not ext or not isinstance(ext, str):
        return ""
    return ext.strip().lower().lstrip(".")


def extract_prior_features(name: str | None, ext: str | None = None, source: str | None = None) -> dict[str, str]:
    """Return the feature dictionary matching `bitagent/internal/evidence/priors.Extract`."""
    out: dict[str, str] = {}
    ns = normalise_source(source) if is_public_safe_source(source) else ""
    if ns:
        out["source"] = ns
    rg = extract_release_group(name)
    if rg:
        out["release_group"] = rg
    qt = extract_quality_tag(name)
    if qt:
        out["quality_tag"] = qt
    res = extract_resolution(name)
    if res:
        out["resolution"] = res
    cod = extract_codec(name)
    if cod:
        out["codec"] = cod
    ne = normalise_extension(ext)
    if ne:
        out["extension"] = ne
    return out


# ---------------------------------------------------------------------------
# Way 1: Active Teacher Sampling & Runtime Queue Priority Weighting
# ---------------------------------------------------------------------------

def compute_sampling_priority_weight(
    swarm: dict[str, Any] | None,
    *,
    uncertainty: float = 0.5,
    reference_epoch: float | None = None,
    dead_swarm_floor: float = 0.25,
    half_life_days: float = 30.0,
) -> float:
    """Compute Way-1 active-learning / processor-queue priority weight.

    Prioritizes live, tracker-confirmed grabbable swarms (`tracker_known=True, seeders > 0`,
    high `peak_seeders`, recent `last_positive_at`, positive velocity) while enforcing a
    strict stratified floor (`dead_swarm_floor=0.25`) for `known_zero` / `dead` torrents
    so dead-swarm and junk behavior remains continuously sampled and evaluated.
    """
    u = max(0.05, min(1.0, float(uncertainty)))
    if not swarm:
        return round(max(float(dead_swarm_floor), 0.65 * (0.5 + u)), 5)

    tr_known = bool(swarm.get("tracker_known") or swarm.get("tr_known"))
    raw_tr_s = swarm.get("tracker_seeders") if "tracker_seeders" in swarm else swarm.get("tr_seeders")
    has_tr_s = type(raw_tr_s) is int and raw_tr_s >= 0
    tr_seeders = int(raw_tr_s) if has_tr_s else 0
    peak_seeders = swarm.get("peak_seeders") if "peak_seeders" in swarm else swarm.get("tr_peak_seeders")
    peak_seeders = int(peak_seeders) if type(peak_seeders) is int and peak_seeders >= 0 else tr_seeders
    prev_seeders = swarm.get("prev_seeders") if "prev_seeders" in swarm else swarm.get("tr_prev_seeders")
    dht_seeders = swarm.get("dht_seeders")
    dht_seeders = int(dht_seeders) if type(dht_seeders) is int and dht_seeders >= 0 else 0

    last_pos = swarm.get("tr_last_pos_epoch")
    if last_pos is None and isinstance(swarm.get("last_positive_at"), str) and swarm["last_positive_at"]:
        try:
            last_pos = datetime.fromisoformat(swarm["last_positive_at"].replace("Z", "+00:00")).timestamp()
        except ValueError:
            last_pos = None

    now_ts = float(reference_epoch) if reference_epoch is not None else time.time()
    recency_bonus = 0.0
    if last_pos is not None and last_pos > 0:
        age_days = max(0.0, (now_ts - float(last_pos)) / 86400.0)
        recency_bonus = math.exp(-math.log(2.0) * age_days / max(1.0, half_life_days))

    if tr_known and has_tr_s and tr_seeders > 0:
        eff_peak = max(tr_seeders, peak_seeders)
        vel_bonus = 0.15 if (type(prev_seeders) is int and tr_seeders > prev_seeders) else 0.0
        liveliness_mult = 1.25 + 0.35 * min(5.0, math.log1p(eff_peak)) + 0.35 * recency_bonus + vel_bonus
    elif tr_known and has_tr_s and tr_seeders == 0:
        liveliness_mult = max(dead_swarm_floor, 0.25 + 0.35 * recency_bonus)
    else:
        liveliness_mult = 0.55 + 0.15 * min(3.0, math.log1p(dht_seeders)) if dht_seeders > 0 else max(dead_swarm_floor, 0.40)

    return round(max(float(dead_swarm_floor), float(liveliness_mult * (0.5 + u))), 5)


# ---------------------------------------------------------------------------
# Way 2: Tracker/Source Identity & Bayesian Prior Features (14 floats)
# ---------------------------------------------------------------------------

WAY2_FEATURE_NAMES: tuple[str, ...] = (
    "tr_dom_opentrackr",        # 0
    "tr_dom_stealth_si",        # 1
    "tr_dom_torrent_eu",        # 2
    "tr_dom_qu_ax",             # 3
    "tr_dom_demonii",           # 4
    "tr_dom_evan_im",           # 5
    "tr_dom_bittor_pw",         # 6
    "tr_dom_dler_org",          # 7
    "tr_dom_explodie",          # 8
    "tr_dom_other_public",      # 9
    "tr_prior_anime_affinity",  # 10
    "tr_prior_audiobook_aff",   # 11
    "prior_rg_success_mean",    # 12: centered (alpha/(alpha+beta) - 0.5)*2 in [-1, 1], 0 when unseen
    "prior_rg_evidence_weight", # 13: min(1.0, log1p(alpha + beta - 2) / 6.0)
)


def _lookup_prior(priors_table: dict[str, dict[str, list[int] | tuple[int, int]]] | None,
                  inline_priors: list[dict[str, Any]] | None,
                  key_type: str, key_value: str) -> tuple[int, int] | None:
    if not key_value:
        return None
    if inline_priors:
        for p in inline_priors:
            if isinstance(p, dict) and p.get("key_type") == key_type and p.get("key_value") == key_value:
                a, b = int(p.get("alpha", 1)), int(p.get("beta", 1))
                if a >= 1 and b >= 1:
                    return a, b
    if priors_table and key_type in priors_table and key_value in priors_table[key_type]:
        ab = priors_table[key_type][key_value]
        return int(ab[0]), int(ab[1])
    return None


def _as_vec(vals: list[float]):
    return np.array(vals, dtype=np.float32) if np is not None else vals


def extract_way2_source_features(
    row: dict[str, Any],
    swarm: dict[str, Any] | None = None,
    priors_table: dict[str, dict[str, Any]] | None = None,
):
    """Compute the 14 Way-2 tracker/source identity & release-group prior features.

    CAUSAL SEPARATION GUARANTEE:
    This function NEVER reads `seeders`, `leechers`, `completed`, or `liveness_status`.
    Changing `seeders` from 1000 to 0 cannot change any element of this 14-d vector.
    When `swarm` is None (cold DHT-only input), all tracker-domain slots are 0.0.
    """
    vec = [0.0] * 14
    name = row.get("name") or ""
    sw = swarm if swarm is not None else row.get("swarm")
    sources = row.get("sources")

    best_tr = None
    inline_priors = None
    if isinstance(sw, dict):
        best_tr = sw.get("best_tracker") or sw.get("tr_best_tracker")
        inline_priors = sw.get("outcome_priors")

    _norm_to_dom = {normalise_source(d): d for d in PUBLIC_TRACKER_DOMAINS}
    dom = normalise_tracker_domain(best_tr)
    if dom == "none" and isinstance(sources, list):
        for raw_item in sources:
            if isinstance(raw_item, dict) and raw_item.get("private") is not True:
                raw_src = raw_item.get("source")
                cand_dom = normalise_tracker_domain(raw_src)
                if cand_dom in _TRACKER_INDEX:
                    dom = cand_dom
                    break
        if dom == "none":
            for s_item in sanitize_sources_for_export(sources):
                src_name = s_item.get("source", "")
                if src_name not in ("", "dht", "tracker"):
                    dom = _norm_to_dom.get(src_name, src_name)
                    break

    if dom != "none":
        idx = _TRACKER_INDEX.get(dom)
        if idx is not None:
            vec[idx] = 1.0
        else:
            vec[9] = 1.0
        aff = TRACKER_DOMAIN_PRIORS.get(dom)
        if aff is not None:
            vec[10] = float(aff[0])
            vec[11] = float(aff[1])

    rg = extract_release_group(name)
    ab = _lookup_prior(priors_table, inline_priors, "release_group", rg)
    if ab is not None:
        a, b = ab
        tot = a + b
        if tot > 2:
            mean_p = a / tot
            vec[12] = float((mean_p - 0.5) * 2.0)
            vec[13] = float(min(1.0, math.log1p(max(0, tot - 2)) / 6.0))

    return _as_vec(vec)


# ---------------------------------------------------------------------------
# Way 3: Swarm-vs-Content Anomaly Features for `junk` / Malware / Trap Detection (10 floats)
# ---------------------------------------------------------------------------

WAY3_FEATURE_NAMES: tuple[str, ...] = (
    "anom_leecher_trap_severe",     # 0: tracker_known & seeders == 0 & leechers >= 50
    "anom_leecher_trap_moderate",   # 1: tracker_known & seeders == 0 & 10 <= leechers < 50
    "anom_fake_seeder_zero_comp",   # 2: tracker_known & seeders >= 50 & completed == 0
    "anom_dht_phantom_swarm",       # 3: dht_seeders >= 25 & tracker_known & tr_seeders == 0 & never positive
    "anom_tiny_or_exec_swarm_trap", # 4: (size < 15MB or exec/html/scr file) & seeders >= 20 & completed <= 1
    "anom_qb_stall_or_metadl",      # 5: liveness in (suspect, dead) & last_qb_state in (metaDL, stalledDL, error, missingFiles)
    "anom_rg_failure_prior",        # 6: release_group failure posterior max(0, (beta/(alpha+beta) - 0.5)*2)
    "anom_ext_failure_hazard",      # 7: extension failure posterior on suspicious/tiny/exec payloads
    "anom_sudden_velocity_collapse",# 8: prev_seeders >= 10 & seeders == 0 & completed == 0
    "shield_verified_high_comp",    # 9: tracker_known & seeders >= 5 & completed >= 50 & no exec/trap files
)


def extract_way3_anomaly_features(
    row: dict[str, Any],
    swarm: dict[str, Any] | None = None,
    priors_table: dict[str, dict[str, Any]] | None = None,
):
    """Compute the 10 Way-3 Swarm-vs-Content Anomaly indicators for the `junk` head."""
    vec = [0.0] * 10
    sw = swarm if swarm is not None else row.get("swarm")
    name = row.get("name") or ""
    size = int(row.get("size") or 0)
    ext = normalise_extension(row.get("ext") or row.get("extension"))
    files = row.get("files") or []

    has_exec_or_trap = ext in _EXEC_OR_TRAP_EXTS
    is_structural_malware_trap = False
    if isinstance(files, list) and files:
        f_exts: set[str] = set()
        has_real_exe = False
        has_dll = False
        has_oversized_lnk = False
        has_root_launcher = False
        vid_depths: list[int] = []

        vc_pre = bool(_VID_OR_ANIME_CLAIM_SP_RE.search(name))
        has_renpy_sp = not vc_pre and any(
            (((x.get("path") or x.get("p")) if isinstance(x, dict) else str(x or "")) or "").replace("\\", "/").lower().endswith((".rpa", ".rpy", ".rpyb"))
            for x in files[:64]
        )
        for f in files[:64]:
            p = ((f.get("path") or f.get("p")) if isinstance(f, dict) else str(f or "")) or ""
            p = p.replace("\\", "/")
            if has_renpy_sp and p.lower().endswith("/say.vbs") and "windows-" in p.lower():
                continue
            sz = int((f.get("size") or f.get("s") or 0) if isinstance(f, dict) else 0)
            if "." in p:
                f_ext = p.rsplit(".", 1)[-1].lower()
                f_exts.add(f_ext)
                is_rarbg_exe = (f_ext == "exe") and ("rarbg" in p.lower() or (0 < sz < 32768))
                if f_ext in _EXEC_OR_TRAP_EXTS and not is_rarbg_exe:
                    has_exec_or_trap = True
                depth = p.count("/")
                if f_ext == "exe" and not is_rarbg_exe and (sz >= 32768 or sz == 0):
                    has_real_exe = True
                    if depth == 0:
                        has_root_launcher = True
                elif f_ext == "dll":
                    has_dll = True
                elif f_ext in _LAUNCHER_EXTS_SP and (sz >= 4096 or sz == 0):
                    if depth == 0:
                        has_root_launcher = True
                    if f_ext == "lnk" and sz >= 32768:
                        has_oversized_lnk = True
                elif f_ext in _VIDEO_EXTS_SP:
                    vid_depths.append(depth)

        vc = bool(_VID_OR_ANIME_CLAIM_SP_RE.search(name))
        is_sw_game = bool((not vc and _SW_OR_GAME_TITLE_SP_RE.search(name)) or (f_exts & _BENIGN_CONTAINER_EXTS_SP))
        if f_exts & _ALWAYS_TRAP_EXTS_SP:
            is_structural_malware_trap = True
            has_exec_or_trap = True
        elif not is_sw_game:
            if vc and has_real_exe and has_dll:
                is_structural_malware_trap = True
            elif vc and len(vid_depths) >= 1 and (has_real_exe or has_root_launcher):
                is_structural_malware_trap = True
            elif (vc or len(vid_depths) > 0) and has_oversized_lnk:
                is_structural_malware_trap = True
            elif vc and len(vid_depths) == 0 and (has_real_exe or has_root_launcher):
                is_structural_malware_trap = True
            elif len(files) <= 2 and len(vid_depths) == 0 and (f_exts & {"zip", "zipx", "rar", "7z"}) and _ANIME_OR_EP_ZIP_SP_RE.search(name) and not _BENIGN_ZIP_SP_RE.search(name):
                is_structural_malware_trap = True
                has_exec_or_trap = True
            elif len(files) <= 2 and len(vid_depths) == 0 and (f_exts & {"iso", "img", "vhd", "vhdx", "udf"}):
                name_stem = re.sub(r"\.(?:iso|img|vhd|vhdx|udf|dmg)$", "", name.strip(), flags=re.IGNORECASE)
                if _ANIME_OR_EP_ZIP_SP_RE.search(name_stem) and not _OPTICAL_DISC_SP_RE.search(name_stem):
                    is_structural_malware_trap = True
                    has_exec_or_trap = True
    elif _VID_OR_ANIME_CLAIM_SP_RE.search(name) and ext in ("zip", "zipx", "rar", "7z", "iso", "img", "vhd", "vhdx", "udf"):
        name_stem = re.sub(r"\.(?:iso|img|vhd|vhdx|udf|dmg)$", "", name.strip(), flags=re.IGNORECASE)
        if _ANIME_OR_EP_ZIP_SP_RE.search(name_stem) and not _BENIGN_ZIP_SP_RE.search(name_stem) and not _OPTICAL_DISC_SP_RE.search(name_stem):
            is_structural_malware_trap = True
            has_exec_or_trap = True

    inline_priors = sw.get("outcome_priors") if isinstance(sw, dict) else None

    if isinstance(sw, dict):
        tr_known = bool(sw.get("tracker_known") or sw.get("tr_known"))
        tr_s = sw.get("tracker_seeders") if "tracker_seeders" in sw else sw.get("tr_seeders")
        tr_l = sw.get("tracker_leechers") if "tracker_leechers" in sw else sw.get("tr_leechers")
        tr_c = sw.get("tracker_completed") if "tracker_completed" in sw else sw.get("tr_completed")
        prev_s = sw.get("prev_seeders") if "prev_seeders" in sw else sw.get("tr_prev_seeders")
        dht_s = sw.get("dht_seeders")
        last_pos = sw.get("tr_last_pos_epoch") or sw.get("last_positive_at")
        liv_status = (sw.get("liveness_status") or sw.get("liv_status") or "").lower()
        qb_state = sw.get("last_qb_state") or sw.get("liv_qb_state") or ""

        has_tr_s = type(tr_s) is int and tr_s >= 0
        has_tr_l = type(tr_l) is int and tr_l >= 0
        has_tr_c = type(tr_c) is int and tr_c >= 0
        s_val = int(tr_s) if has_tr_s else 0
        l_val = int(tr_l) if has_tr_l else 0
        c_val = int(tr_c) if has_tr_c else 0
        d_val = int(dht_s) if type(dht_s) is int and dht_s >= 0 else 0
        p_val = int(prev_s) if type(prev_s) is int and prev_s >= 0 else 0

        if tr_known and has_tr_s and s_val == 0 and l_val >= 50:
            vec[0] = min(1.0, math.log1p(l_val) / math.log1p(200.0))
        elif tr_known and has_tr_s and s_val == 0 and 10 <= l_val < 50:
            vec[1] = float((l_val - 9) / 40.0)

        if tr_known and has_tr_s and s_val >= 50 and has_tr_c and c_val == 0:
            vec[2] = min(1.0, math.log1p(s_val) / math.log1p(500.0))

        if d_val >= 25 and tr_known and has_tr_s and s_val == 0 and (not has_tr_c or c_val == 0) and not last_pos:
            vec[3] = min(1.0, math.log1p(d_val) / math.log1p(250.0))

        max_s = max(s_val if has_tr_s else 0, d_val)
        if is_structural_malware_trap or ((0 < size < 15 * 1024 * 1024 or has_exec_or_trap) and max_s >= 20 and has_tr_c and c_val <= 1):
            vec[4] = 1.0

        if liv_status in ("suspect", "dead") and qb_state in ("metaDL", "stalledDL", "error", "missingFiles"):
            vec[5] = 1.0 if liv_status == "dead" or qb_state == "metaDL" else 0.75

        if tr_known and p_val >= 10 and has_tr_s and s_val == 0 and has_tr_c and c_val == 0:
            vec[8] = 1.0

        if tr_known and has_tr_s and s_val >= 5 and has_tr_c and c_val >= 50 and not has_exec_or_trap:
            vec[9] = min(1.0, math.log1p(c_val) / math.log1p(1000.0))

    rg = extract_release_group(name)
    ab_rg = _lookup_prior(priors_table, inline_priors, "release_group", rg)
    if ab_rg is not None:
        a, b = ab_rg
        if a + b >= 4:
            fail_rate = b / (a + b)
            if fail_rate > 0.55:
                vec[6] = float(min(1.0, (fail_rate - 0.50) * 2.0))

    ab_ext = _lookup_prior(priors_table, inline_priors, "extension", ext)
    if ab_ext is not None:
        a, b = ab_ext
        if a + b >= 10:
            fail_rate = b / (a + b)
            if fail_rate > 0.75 and (has_exec_or_trap or (0 < size < 50 * 1024 * 1024)):
                vec[7] = float(min(1.0, (fail_rate - 0.50) * 2.0))

    if is_structural_malware_trap:
        vec[4] = 1.0
        vec[7] = max(vec[7], 0.90)
        vec[9] = 0.0
    return _as_vec(vec)


def extract_swarm_side_vector(
    row: dict[str, Any],
    swarm: dict[str, Any] | None = None,
    priors_table: dict[str, dict[str, Any]] | None = None,
):
    """Concatenate Way-2 (14 floats) and Way-3 (10 floats) into the 24-d swarm side vector."""
    w2 = list(extract_way2_source_features(row, swarm=swarm, priors_table=priors_table))
    w3 = list(extract_way3_anomaly_features(row, swarm=swarm, priors_table=priors_table))
    return _as_vec(w2 + w3)


def apply_modality_dropout(
    swarm_side_matrix,
    *,
    dropout_prob: float = 0.60,
    seed: int = 0,
):
    """Randomly mask `dropout_prob` (50%–70%) of rows in `swarm_side_matrix` [N, 24] to all-zeros (cold-DHT state)."""
    if not 0.0 <= dropout_prob <= 1.0:
        raise ValueError("dropout_prob must be in [0.0, 1.0]")
    if np is not None and isinstance(swarm_side_matrix, np.ndarray):
        out = np.array(swarm_side_matrix, dtype=np.float32, copy=True)
        if len(out) == 0 or dropout_prob == 0.0:
            return out, np.zeros(len(out), dtype=bool)
        rng = np.random.default_rng(seed)
        mask = rng.random(len(out)) < dropout_prob
        out[mask, :] = 0.0
        return out, mask
    # Pure Python list-of-lists fallback
    rng_py = random.Random(seed)
    out_py = []
    mask_py = []
    for row_vec in swarm_side_matrix:
        drop = (rng_py.random() < dropout_prob)
        mask_py.append(drop)
        out_py.append([0.0] * len(row_vec) if drop else list(row_vec))
    return out_py, mask_py


# ---------------------------------------------------------------------------
# Way 4: Orthogonal Swarm Liveliness / Grab Viability & Ranking Multiplier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwarmViabilityReport:
    viability_prob: float          # Calibrated P(import_success | swarm, priors) in [0.001, 0.999]
    viability_tier: str            # alive_viable | degraded_viable | suspect_stalled | dead_known_zero | unknown_cold
    observation: str               # live | dead | suspect | unknown (for processorapi/v1 LivenessClaim)
    effective_seeders: int         # Authoritative tracker seeders when known, else conservative DHT estimate
    prior_success_prob: float      # Geometric mean of matched Beta(alpha, beta) feature priors
    ranking_multiplier: float      # E[import_success] * log(1 + max(1, effective_seeders)) for stage-2 ranking


def compute_swarm_viability(
    row: dict[str, Any],
    swarm: dict[str, Any] | None = None,
    priors_table: dict[str, dict[str, Any]] | None = None,
    *,
    wanted_prob: float | None = None,
) -> SwarmViabilityReport:
    """Compute orthogonal Swarm Viability & Grab-Outcome score (Way 4)."""
    name = row.get("name") or ""
    ext = normalise_extension(row.get("ext") or row.get("extension"))
    sw = swarm if swarm is not None else row.get("swarm")
    inline_priors = sw.get("outcome_priors") if isinstance(sw, dict) else None

    raw_source = row.get("source")
    if not raw_source and isinstance(row.get("sources"), list):
        for s_item in sanitize_sources_for_export(row.get("sources")):
            if s_item.get("source") and s_item["source"] not in ("dht", "tracker"):
                raw_source = s_item["source"]
                break
    if not raw_source and isinstance(sw, dict):
        bt = sw.get("best_tracker") or sw.get("tr_best_tracker")
        dom = normalise_tracker_domain(bt)
        if dom != "none":
            raw_source = dom
    feats = extract_prior_features(name, ext=ext, source=raw_source)
    raw_lang = row.get("language")
    if isinstance(raw_lang, str) and raw_lang.strip():
        feats["language"] = raw_lang.strip().lower()
    prior_probs: list[float] = []
    for kt, kv in feats.items():
        ab = _lookup_prior(priors_table, inline_priors, kt, kv)
        if ab is not None:
            a, b = ab
            if a + b > 2:
                prior_probs.append(max(0.01, min(0.99, a / (a + b))))
    prior_mean = math.exp(sum(math.log(p) for p in prior_probs) / len(prior_probs)) if prior_probs else 0.50

    if not isinstance(sw, dict) or not sw:
        viab = round(0.40 * 0.50 + 0.60 * prior_mean, 4)
        mult = round(viab * math.log1p(1), 4)
        return SwarmViabilityReport(
            viability_prob=viab,
            viability_tier="unknown_cold",
            observation="unknown",
            effective_seeders=0,
            prior_success_prob=round(prior_mean, 4),
            ranking_multiplier=round(mult * wanted_prob, 4) if wanted_prob is not None else mult,
        )

    tr_scraped = bool(sw.get("tracker_scraped", True) if "tracker_scraped" in sw else sw.get("tr_scraped", True))
    tr_known = bool(sw.get("tracker_known") or sw.get("tr_known"))
    tr_s = sw.get("tracker_seeders") if "tracker_seeders" in sw else sw.get("tr_seeders")
    prev_s = sw.get("prev_seeders") if "prev_seeders" in sw else sw.get("tr_prev_seeders")
    dht_s = sw.get("dht_seeders")
    liv_status = (sw.get("liveness_status") or sw.get("liv_status") or "").lower()
    qb_state = sw.get("last_qb_state") or sw.get("liv_qb_state") or ""

    has_tr_s = type(tr_s) is int and tr_s >= 0
    s_val = int(tr_s) if has_tr_s else 0
    d_val = int(dht_s) if type(dht_s) is int and dht_s >= 0 else 0
    p_val = int(prev_s) if type(prev_s) is int and prev_s >= 0 else None

    if not tr_scraped:
        eff_seeds = d_val
        base_v = 0.55 + 0.08 * min(3.0, math.log1p(d_val)) if d_val > 0 else 0.45
        tier = "alive_viable" if d_val > 0 else "unknown_cold"
        obs = "live" if d_val > 0 else "unknown"
    elif tr_known and has_tr_s and s_val > 0:
        eff_seeds = s_val
        base_v = 0.50 + 0.10 * min(4.2, math.log1p(s_val))
        if p_val is not None and s_val >= p_val:
            base_v += 0.05
        tier = "alive_viable"
        obs = "live"
    elif tr_known and has_tr_s and s_val == 0:
        eff_seeds = 0
        if d_val > 0:
            base_v = 0.12
            tier = "degraded_viable"
        else:
            base_v = 0.015
            tier = "dead_known_zero"
        obs = "dead"
    else:
        eff_seeds = d_val
        if d_val > 0:
            base_v = 0.28 + 0.06 * min(3.0, math.log1p(d_val))
            tier = "degraded_viable"
            obs = "live"
        else:
            base_v = 0.04
            tier = "unknown_cold"
            obs = "unknown"

    if liv_status == "dead" or qb_state in ("missingFiles", "error"):
        base_v *= 0.05
        tier = "dead_known_zero"
        obs = "dead"
        eff_seeds = 0
    elif liv_status == "suspect" or qb_state in ("metaDL", "stalledDL"):
        base_v *= 0.35
        tier = "suspect_stalled"
        obs = "suspect"
    elif liv_status == "alive":
        base_v = max(base_v, 0.75)
        tier = "alive_viable"
        obs = "live"

    viab = max(0.001, min(0.999, 0.58 * base_v + 0.42 * prior_mean))
    rank_mult = viab * math.log1p(max(1, eff_seeds)) if eff_seeds > 0 else viab * 0.15
    if wanted_prob is not None:
        rank_mult *= max(0.0, min(1.0, float(wanted_prob)))

    return SwarmViabilityReport(
        viability_prob=round(float(viab), 4),
        viability_tier=tier,
        observation=obs,
        effective_seeders=int(eff_seeds),
        prior_success_prob=round(float(prior_mean), 4),
        ranking_multiplier=round(float(rank_mult), 4),
    )
