"""Prediction ledger completeness lock, model-independent stratification, and recorded-pi Poisson sampling."""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .estimators import release_family_id

ABSTAIN_STATUSES = {"refused", "abstain", "timeout", "parse_error", "content_filter"}

SAMPLING_LEDGER_COLUMNS = (
    "item_id",
    "family_id",
    "stratum",
    "selector_sha256",
    "u_i",
    "pi_i",
    "R_i",
    "rng_seed_reserved_at",
)

FROZEN_SELECTOR_SPEC = "frozen_parser_disagreement_selector_v1:rtn_ptt_guessit"
FROZEN_SELECTOR_SHA256 = hashlib.sha256(FROZEN_SELECTOR_SPEC.encode("utf-8")).hexdigest()


def commit_stratum(
    model_id: str,
    preds: Dict[Any, Dict[str, Any]],
    n_stratum: int,
    receipt_path: Optional[str] = None,
    receipt_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify 100% item accounting for a stratum and record provider refusals as explicit abstentions."""
    if len(preds) != n_stratum:
        raise AssertionError(
            f"Completeness lock failed for {model_id}: {len(preds)}/{n_stratum} items present"
        )
    n_abstain = 0
    for item_pred in preds.values():
        status = str(item_pred.get("status", "ok")).lower()
        cat = item_pred.get("category")
        if status in ABSTAIN_STATUSES or cat in (None, "abstain", "⊥"):
            n_abstain += 1
    if receipt_sha256 is None:
        canonical = json.dumps(
            {str(k): preds[k] for k in sorted(preds, key=str)}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        receipt_sha256 = hashlib.sha256(canonical).hexdigest()
    out: Dict[str, Any] = {
        "model_id": model_id,
        "n_stratum": int(n_stratum),
        "n_evaluated": int(len(preds)),
        "n_abstain": int(n_abstain),
        "abstain_rate": float(n_abstain / max(1, n_stratum)),
        "receipt_sha256": receipt_sha256,
    }
    if receipt_path is not None:
        out["receipt_path"] = receipt_path
    return out


def model_independent_stratum(item: Dict[str, Any]) -> str:
    """Assign an intake item to a stratum defined strictly from model-independent metadata
    (source, time window, extension mix, size bucket) — never from candidate model output."""
    source = str(item.get("source") or "dht").lower().strip()
    epoch = str(item.get("time_window") or item.get("epoch") or "2026-Q4")
    files = item.get("files") or []
    exts = sorted({
        pathlib.Path(str(f.get("path", "") if isinstance(f, dict) else f)).suffix.lower().lstrip(".")
        for f in files
        if f
    })
    if not exts:
        ext_mix = "bare_title"
    elif any(e in ("mkv", "mp4", "avi", "ts", "m2ts") for e in exts):
        ext_mix = "video_container"
    elif any(e in ("flac", "mp3", "m4b", "opus", "wav") for e in exts):
        ext_mix = "audio_container"
    elif any(e in ("epub", "pdf", "cbz", "cbr", "mobi") for e in exts):
        ext_mix = "print_container"
    elif any(e in ("iso", "exe", "msi", "dmg", "apk", "rar", "zip", "7z") for e in exts):
        ext_mix = "archive_or_binary"
    else:
        ext_mix = "other_ext"

    size_b = int(item.get("size_bytes") or item.get("size") or 0)
    if size_b <= 0:
        size_bucket = "size_unknown"
    elif size_b < 100 * 1024 * 1024:
        size_bucket = "lt_100mb"
    elif size_b < 2 * 1024 * 1024 * 1024:
        size_bucket = "100mb_2gb"
    elif size_b < 15 * 1024 * 1024 * 1024:
        size_bucket = "2gb_15gb"
    else:
        size_bucket = "ge_15gb"
    return f"{source}|{epoch}|{ext_mix}|{size_bucket}"


def compute_poisson_inclusion_probs(
    u: Sequence[float],
    budget: float,
    pi_min: float = 0.05,
) -> np.ndarray:
    """Compute clipped Poisson inclusion probabilities pi_i = min(1, max(pi_min, c * u_i))
    with positivity floor pi_min >= 0.05 and sum(pi_i) == budget (when feasible)."""
    if pi_min < 0.05 or pi_min > 1.0:
        raise ValueError(f"pi_min={pi_min} must be in [0.05, 1.0]")
    u_arr = np.maximum(np.asarray(u, dtype=np.float64), 1e-6)
    n = len(u_arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    target = float(np.clip(budget, n * pi_min, float(n)))
    lo, hi = 0.0, float(n / np.min(u_arr)) * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = float(np.clip(mid * u_arr, pi_min, 1.0).sum())
        if s < target:
            lo = mid
        else:
            hi = mid
    return np.clip(0.5 * (lo + hi) * u_arr, pi_min, 1.0)


def reserve_poisson_tranche(
    items: Sequence[Dict[str, Any]],
    u_scores: Sequence[float],
    budget: float,
    pi_min: float = 0.05,
    seed: int = 20261019,
    reserved_at: str = "2026-10-19T00:00:00Z",
    selector_sha256: str = FROZEN_SELECTOR_SHA256,
) -> List[Dict[str, Any]]:
    """Pre-label Poisson sampling reservation writing all required B6 ledger columns before labeling.

    For each item, `u_i` is the deterministic pre-label Uniform(0,1) draw derived from
    `int.from_bytes(sha256(f"{seed}:{item_id}")[:8], "big") / 2**64`, `pi_i` is the clipped
    inclusion probability `min(1, max(pi_min, c * s_i))` from `u_scores`, and `R_i = int(u_i < pi_i)`.
    """
    if len(items) != len(u_scores):
        raise ValueError("items and u_scores must have identical length")
    pi = compute_poisson_inclusion_probs(u_scores, budget=budget, pi_min=pi_min)
    records: List[Dict[str, Any]] = []
    for idx, (it, s_val, pi_val) in enumerate(zip(items, u_scores, pi)):
        item_id = str(it.get("id") or it.get("item_id") or f"item-{idx:05d}")
        fam_id = int(it.get("family_id") or release_family_id(str(it.get("name") or item_id)))
        digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
        u_draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
        u_rounded = round(float(u_draw), 6)
        pi_rounded = round(float(pi_val), 6)
        r_val = int(u_rounded < pi_rounded)
        rec = {
            "item_id": item_id,
            "family_id": fam_id,
            "stratum": model_independent_stratum(it),
            "selector_sha256": selector_sha256,
            "difficulty_weight": round(float(s_val), 6),
            "u_i": u_rounded,
            "pi_i": pi_rounded,
            "R_i": r_val,
            "rng_seed_reserved_at": f"seed={seed}@{reserved_at}",
        }
        records.append(rec)
    return records
