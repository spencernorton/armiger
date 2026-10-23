"""Vendor-disjoint comparison keys and support-bounded scoring without fractional tie credit."""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Set, Tuple
import numpy as np


def comparison_key(
    teacher_preds: Dict[str, np.ndarray],
    vendors: Dict[str, str],
    scored: Set[str],
    supermajority: float = 2.0 / 3.0,
) -> Tuple[np.ndarray, str]:
    """Build a vendor-disjoint consensus reference key for a comparison panel.

    Raises ``AssertionError`` (Rule R7) if excluding the vendors of ``scored`` leaves
    fewer than 2 independent teacher families. Items without a supermajority among
    remaining teachers are marked ``-1`` (abstention ``⊥``), never ``0.5`` half-credit.
    """
    banned = {vendors[m] for m in scored if m in vendors}
    pool = [t for t in teacher_preds if vendors.get(t) not in banned]
    if len(pool) < 2:
        raise AssertionError(
            f"panel {sorted(scored)} leaves {len(pool)} independent teacher(s) — R7"
        )
    P = np.stack([np.asarray(teacher_preds[t]) for t in pool])
    key = np.full(P.shape[1], -1, dtype=np.int16)
    need = math.ceil(supermajority * len(pool))
    for i in range(P.shape[1]):
        vals, cnt = np.unique(P[:, i], return_counts=True)
        if int(cnt.max()) >= need:
            key[i] = int(vals[int(cnt.argmax())])
    digest = hashlib.sha256(key.tobytes() + ",".join(sorted(pool)).encode("utf-8")).hexdigest()
    return key, digest


def score_on_support(pred: np.ndarray, key: np.ndarray) -> Dict[str, float]:
    """Score integer predictions against ``key`` on its resolved support ``key >= 0``."""
    pred_arr = np.asarray(pred)
    key_arr = np.asarray(key)
    sup = key_arr >= 0
    z = (pred_arr[sup] == key_arr[sup]).astype(np.int8)
    n = int(len(key_arr))
    nbar = int((~sup).sum())
    return {
        "n_support": int(sup.sum()),
        "acc_support": float(z.mean()) if sup.any() else 0.0,
        "lower": float(z.sum() / n) if n > 0 else 0.0,
        "upper": float((z.sum() + nbar) / n) if n > 0 else 0.0,
    }
