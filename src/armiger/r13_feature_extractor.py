"""R13 S2 features and a read-only adapter for frozen R12 ONNX checkpoints.

There are two feature views. Category keeps the R12 enriched input; title
language sees the literal title, without file names or file statistics. S8
should train those views together, rather than transplanting a language head.

Only two semantic rules are admitted: identifier neutrality for the binary
title-language task, and a bracketed release/episode/technical-tags/video cue
for anime. ``audit_train_rules`` supplies their train-only precision gates.
Dual-audio and ASS are observable features, never standalone anime rules.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SIDE_DIM = 28
BASE_STATS_DIM = 12
STATS_DIM = 40
MIN_SUPPORT = 5
MIN_PRECISION = 0.90

# These are syntax families from the session brief, not title/group catalogs.
NUMERIC_ID_RE = re.compile(r"[0-9]+(?:-[0-9]+)?", re.ASCII)
PG_ID_RE = re.compile(r"pg[0-9]+(?:-[0-9]+)?", re.I | re.ASCII)
GUT_ID_RE = re.compile(r"[a-z0-9_.-]+gut", re.I | re.ASCII)
PACKAGE_ID_RE = re.compile(r"com(?:\.[a-z][a-z0-9_]*){2,}", re.I | re.ASCII)
FANSUB_EPISODE_RE = re.compile(
    r"^\[(?P<group>[^\[\]\r\n]{2,80})\]\s*(?P<title>.+?)\s+-\s+"
    r"(?P<episode>[0-9]{1,3})(?:v[0-9]+)?\s*"
    r"(?P<tags>(?:(?:\[[^\[\]\r\n]+\]|\([^()\r\n]+\))\s*)+)"
    r"\.(?:mkv|mp4|avi|webm)$", re.I,
)
RESOLUTION_RE = re.compile(r"\b(?:480|576|720|1080|1440|2160|4320)[pi]\b", re.I)
JAPANESE_RE = re.compile(r"(?<![a-z])(?:jpn|japanese)(?![a-z])", re.I)
ENGLISH_RE = re.compile(r"(?<![a-z])(?:eng|english)(?![a-z])", re.I)
DUAL_AUDIO_RE = re.compile(r"\bdual[ ._-]*audio\b", re.I)
VIDEO_EXTS = frozenset({"mkv", "mp4", "avi", "webm"})
RULE_IDENTIFIER = "identifier_title_neutrality"
RULE_FANSUB = "bracket_episode_resolution_video"


@lru_cache(maxsize=1)
def r12_features():
    """Load the named local baseline, without importing a trainer or dataset."""
    path = PROJECT_ROOT / "_r12_features_base.py"
    spec = importlib.util.spec_from_file_location("_s2_r12_baseline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline featurizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.SIDE_DIM, module.STATS_DIM) != (SIDE_DIM, STATS_DIM):
        raise ValueError("R12 feature contract changed; requalify S2")
    return module


def normalize_files(files: Any) -> list[dict[str, Any]]:
    """Support SDK mappings, path/size pairs, and plain relative paths."""
    result = []
    for entry in files or []:
        if isinstance(entry, Mapping):
            path = entry.get("path") or entry.get("p") or entry.get("name") or ""
            size = entry.get("size") or entry.get("s") or 0
        elif isinstance(entry, (tuple, list)):
            path = entry[0] if entry else ""
            size = entry[1] if len(entry) > 1 else 0
        else:
            path, size = entry, 0
        try:
            size = max(0, int(size))
        except (TypeError, ValueError, OverflowError):
            size = 0
        path = str(path or "").replace("\\", "/")
        if path and "/.____padding_file/" not in f"/{path}":
            result.append({"path": path, "size": size})
    return result


def _safe_nonneg_int(val: Any) -> int:
    try:
        return max(0, int(val or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def inference_input(row: Mapping[str, Any]) -> dict[str, Any]:
    """Use R12's production firewall: name/files/size/infohash, no language hint.

    Matches active_source/armiger_bench.py::sanitize_inference_input for snapshot
    mappings, while accepting SDK path/size aliases for clean external inputs.
    """
    source = row.get("inference_input", row)
    if not isinstance(source, Mapping):
        raise TypeError("inference_input must be a mapping")
    return {"name": str(source.get("name") or source.get("title") or ""),
            "files": normalize_files(source.get("files")),
            "size": _safe_nonneg_int(source.get("size")),
            "infohash": str(source.get("infohash") or "").lower()}


def _utf8_prefix(text: str, max_bytes: int) -> str:
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    return str(text).encode("utf-8", "replace")[:max_bytes].decode("utf-8", "ignore")


def build_title_language_text(name: str, files: Any = None, max_bytes: int = 256) -> str:
    """Title-only view. Files deliberately have no effect on this representation."""
    return _utf8_prefix(name, max_bytes)


def build_enriched_name_text(
    name: str, files: Any, max_bytes: int = 256, *, head: str = "category",
) -> str:
    """Keep R12 enrichment for category; expose a separate title-language view."""
    if head == "title_language":
        return build_title_language_text(name, max_bytes=max_bytes)
    text = r12_features().build_enriched_name_text(name, normalize_files(files), max_bytes=max_bytes)
    return _utf8_prefix(text, max_bytes)


def _id_stem(name: str) -> str:
    stem = name.strip()
    # Exact opaque file identifiers are also supported, but not arbitrary paths.
    if stem.lower().endswith((".txt", ".zip")):
        stem = stem.rsplit(".", 1)[0]
    return stem


def identifier_family(name: str, files: Any = None) -> str | None:
    """Return a nonlinguistic-ID family, never infer a work or spoken language."""
    stem = _id_stem(name)
    for family, pattern in (("numeric", NUMERIC_ID_RE), ("pg_numeric", PG_ID_RE),
                            ("gut_stem", GUT_ID_RE), ("ascii_package", PACKAGE_ID_RE)):
        if pattern.fullmatch(stem):
            return family
    # TXT+ZIP alone is not enough: require an opaque Gutenberg-shaped stem
    # and exactly matching relative path, rather than unrelated archive files.
    by_ext: dict[str, set[str]] = {"txt": set(), "zip": set()}
    for item in normalize_files(files):
        path = item["path"].lower()
        base, dot, ext = path.rpartition(".")
        if dot and ext in by_ext:
            leaf = base.rsplit("/", 1)[-1]
            if NUMERIC_ID_RE.fullmatch(leaf) or PG_ID_RE.fullmatch(leaf):
                by_ext[ext].add(base)
    # A readable supplied title must retain its own language, even with book files.
    opaque_name = not re.search(r"[a-z]{3,}", name, re.I | re.ASCII)
    if opaque_name and by_ext["txt"] & by_ext["zip"]:
        return "gutenberg_txt_zip"
    return None


@dataclass(frozen=True)
class StructuralCues:
    fansub_episode_keys: frozenset[int]
    fansub_video_count: int
    has_mkv: bool
    has_ass_subtitles: bool
    explicit_jpn_eng_dual_audio: bool

    @property
    def fansub_episode(self) -> bool:
        return bool(self.fansub_episode_keys)


def extract_structural_cues(name: str, files: Any = None) -> StructuralCues:
    """Parse arbitrary group tags; there are no release-group or work literals."""
    entries = normalize_files(files)
    texts = [name, *(f["path"].rsplit("/", 1)[-1] for f in entries)]
    episodes: set[int] = set()
    matched: set[str] = set()
    for text in texts:
        match = FANSUB_EPISODE_RE.fullmatch(text)
        if match and RESOLUTION_RE.search(match["tags"]):
            episodes.add(int(match["episode"]))
            matched.add(text)
    exts = {t.rsplit(".", 1)[-1].lower() for t in texts if "." in t}
    # These fields describe metadata only. They do not assert anime/sub language.
    audio_text = " ".join(texts)
    dual = bool(DUAL_AUDIO_RE.search(audio_text) and JAPANESE_RE.search(audio_text)
                and ENGLISH_RE.search(audio_text) and "mkv" in exts)
    return StructuralCues(frozenset(episodes), len(matched), "mkv" in exts, "ass" in exts, dual)


def compute_derivative_group_features(files: Any, raw_name: str = "") -> tuple[float, ...]:
    """R12 U/P/share semantics plus explicit bracketed episode/version parsing.

    Episode numbers in SxxExx and fansub paths refer to the same episodes: use
    the larger count rather than summing two parsers' possibly duplicate keys.
    S8 receives raw counts as well as the bounded side-vector representation.
    """
    normalized = normalize_files(files)
    if not normalized and raw_name.rsplit(".", 1)[-1].lower() in VIDEO_EXTS:
        normalized = [{"path": raw_name, "size": 0}]
    # The derivative view drops resolution/version tags only after a full
    # structural match. Keep parent+title+episode, so separate episodes/series
    # cannot collapse merely because their codec, group, or episode number agrees.
    derivative_view = []
    for entry in normalized:
        parent, sep, leaf = entry["path"].rpartition("/")
        match = FANSUB_EPISODE_RE.fullmatch(leaf)
        if match and RESOLUTION_RE.search(match["tags"]):
            ext = leaf.rsplit(".", 1)[-1]
            leaf = f"{match['title']}.E{int(match['episode']):03d}.{ext}"
            derivative_view.append({"path": parent + sep + leaf, "size": entry["size"]})
        else:
            derivative_view.append(entry)
    u, e, p, share = r12_features().compute_derivative_group_features(derivative_view, raw_name)
    fan_count = len(extract_structural_cues(raw_name, normalized).fansub_episode_keys)
    e = max(e, float(fan_count))
    return u, e, (0.0 if e >= 2 else p), share


def encode_torrent_features(row: Mapping[str, Any], max_name_len: int = 256,
                            max_files: int = 64) -> dict[str, Any]:
    """S8 training/inference encoder. R12 checkpoints use the adapter below.

    New keys: title_name_bytes, title_name_text, title_identifier_family,
    structural_cues. Existing stats remain 40 wide. side[13]/side[15] gain
    subtitle presence and a validated generic fansub cue. side[24:28] preserve
    their U/E/P/share meaning; no new checkpoint width is silently introduced.
    """
    clean = inference_input(row)
    name = str(clean.get("name") or clean.get("title") or "")
    encoded = r12_features().encode_torrent_features(clean, max_name_len, max_files)
    cues = extract_structural_cues(name, clean["files"])
    deriv = compute_derivative_group_features(clean["files"], name)
    stats = list(encoded["stats"])
    if cues.fansub_episode:
        stats[BASE_STATS_DIM + 15] = 1.0
    if cues.has_ass_subtitles:
        stats[BASE_STATS_DIM + 13] = 1.0
    u, e, p, share = deriv
    stats[BASE_STATS_DIM + 24:] = [min(1.0, math.log1p(u) / math.log1p(64)),
                                  min(1.0, math.log1p(e) / math.log1p(64)), p, share]
    title = build_title_language_text(name, max_bytes=max_name_len)
    encoded.update(stats=stats, deriv_features=deriv, structural_cues=cues,
                   title_name_text=title, title_name_bytes=list(title.encode("utf-8")) or [32],
                   title_identifier_family=identifier_family(name, clean["files"]))
    return encoded


def build_title_language_batch_inputs(rows: Sequence[Mapping[str, Any]],
                                      stats_dim: int = STATS_DIM) -> dict[str, np.ndarray]:
    """A fully separate name view: no file bytes, sizes, metadata, or language hint.

    Directly replacing bytes avoids R12's file fallback for dotted names. All
    non-title tensors have the same constant value for every title.
    """
    names = [str((r := inference_input(row)).get("name") or r.get("title") or "") for row in rows]
    data = [list(build_title_language_text(n).encode("utf-8")) or [32] for n in names]
    batch = len(data)
    if batch == 0:
        raise ValueError("rows must not be empty")
    name_bytes = np.zeros((batch, max(map(len, data))), dtype=np.int64)
    for i, values in enumerate(data):
        name_bytes[i, :len(values)] = values
    return {"name_bytes": name_bytes, "name_len": np.asarray(list(map(len, data)), dtype=np.int64),
            "ext": np.zeros((batch, 1), dtype=np.int32),
            "size": np.zeros((batch, 1), dtype=np.int64),
            "depth": np.zeros((batch, 1), dtype=np.int64),
            "words": np.zeros((batch, 1, 4), dtype=np.int32),
            "stats": np.zeros((batch, stats_dim), dtype=np.float32)}


def _label(row: Mapping[str, Any], head: str) -> str | None:
    target = (row.get("targets") or {}).get(head) or {}
    return target.get("value") if target.get("known_mask") else None


def audit_train_rules(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze support using train labels only; unknown masks never count as gold.

    The identifier rule is one pre-specified union from the user's brief.
    Subgroup counts are retained so absent supervision is explicit.
    """
    counts = {RULE_IDENTIFIER: Counter(), RULE_FANSUB: Counter()}
    subgroups: dict[str, Counter] = {}
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Rule admission accepts train-marked rows only")
        clean = inference_input(row)
        name = str(clean.get("name") or clean.get("title") or "")
        family = identifier_family(name, clean["files"])
        if family:
            value = _label(row, "title_language") or "unknown"
            counts[RULE_IDENTIFIER][value] += 1
            subgroups.setdefault(family, Counter())[value] += 1
        if extract_structural_cues(name, clean["files"]).fansub_episode:
            counts[RULE_FANSUB][_label(row, "anime") or "unknown"] += 1
    report: dict[str, Any] = {}
    for rule, positive in ((RULE_IDENTIFIER, "english_title"), (RULE_FANSUB, "anime")):
        c = counts[rule]
        support = sum(v for k, v in c.items() if k != "unknown")
        precision = c[positive] / support if support else 0.0
        report[rule] = {"counts": dict(c), "known_support": support, "precision": precision,
                        "admitted": support >= MIN_SUPPORT and precision >= MIN_PRECISION}
    report["identifier_subgroups"] = {k: dict(v) for k, v in subgroups.items()}
    report["dual_audio_ass_standalone_rule_admitted"] = False
    return report


def apply_head_guards(row: Mapping[str, Any], predictions: Mapping[str, str],
                      train_audit: Mapping[str, Any], *,
                      enable_identifier_guard: bool = False) -> tuple[dict[str, str], dict[str, str]]:
    """Copy predictions, modify only title_language/anime, and expose provenance.

    english_title is the existing binary task's compatibility value for an
    identifier without a readable title. It is not an assertion about book text,
    spoken language, subtitle language, country, or the title behind an ID.
    """
    clean = inference_input(row)
    name = str(clean.get("name") or clean.get("title") or "")
    result, reasons = dict(predictions), {}
    family = identifier_family(name, clean["files"])
    if (enable_identifier_guard and family
            and train_audit.get(RULE_IDENTIFIER, {}).get("admitted")):
        result["title_language"] = "english_title"
        reasons["title_language"] = f"{RULE_IDENTIFIER}:{family}"
    if (extract_structural_cues(name, clean["files"]).fansub_episode
            and train_audit.get(RULE_FANSUB, {}).get("admitted")):
        result["anime"] = "anime"
        reasons["anime"] = RULE_FANSUB
    return result, reasons


class R13FrozenR12Adapter:
    """Head-local compatibility probe, never writes/changes model weights.

    Category and all unrelated heads come from the original R12 feature feed.
    Title-only inference is used for opaque IDs or names R12 enriched; ordinary
    titles retain the frozen model's trained feature distribution. S8 can use
    title_name_bytes for every row during native joint training.

    Blanket identifier neutrality is an opt-in compatibility candidate: it
    passes the pooled train gate but regresses known validation titles. Always
    expose its ablation separately; the safe default uses title-only inference.
    """
    def __init__(self, model_path: str | Path, train_audit: Mapping[str, Any], threads: int = 2,
                 *, enable_identifier_guard: bool = False):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.stats_dim = next(i.shape[1] for i in self.session.get_inputs() if i.name == "stats")
        self.train_audit = train_audit
        self.enable_identifier_guard = enable_identifier_guard

    def predict_batch(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        baseline = r12_features()
        clean = [inference_input(r) for r in rows]
        head_names = [o.name for o in self.session.get_outputs()
                      if o.name in baseline.NEURAL_HEAD_CLASSES and len(o.shape) == 2]
        feed = baseline.build_onnx_batch_inputs(clean, stats_dim=self.stats_dim)
        logits = dict(zip(head_names, self.session.run(head_names, feed)))
        routed = []
        for i, row in enumerate(clean):
            name = str(row.get("name") or row.get("title") or "")
            if (identifier_family(name, row["files"])
                    or baseline.build_enriched_name_text(name, row["files"]) != name):
                routed.append(i)
        title_logits = None
        if routed:
            title_feed = build_title_language_batch_inputs([clean[i] for i in routed], self.stats_dim)
            title_logits = self.session.run(["title_language"], title_feed)[0]
        route_index = {index: j for j, index in enumerate(routed)}
        result = []
        for i, row in enumerate(clean):
            raw = {h: baseline.NEURAL_HEAD_CLASSES[h][int(np.argmax(values[i]))]
                   for h, values in logits.items()}
            view = dict(raw)
            if i in route_index:
                view["title_language"] = baseline.NEURAL_HEAD_CLASSES["title_language"][
                    int(np.argmax(title_logits[route_index[i]]))]
            guarded, reasons = apply_head_guards(row, view, self.train_audit,
                                                 enable_identifier_guard=self.enable_identifier_guard)
            candidate, candidate_reasons = apply_head_guards(row, view, self.train_audit,
                                                            enable_identifier_guard=True)
            result.append({"baseline": raw, "title_only": view, "predictions": guarded,
                           "guard_reasons": reasons, "title_view_routed": i in route_index,
                           "identifier_policy_candidate": candidate,
                           "identifier_policy_guard_reasons": candidate_reasons})
        return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
