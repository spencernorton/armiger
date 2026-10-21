"""Mondrian split-conformal threshold fitting and exact logit-temperature invariance."""
from __future__ import annotations

import math
from typing import Optional
import numpy as np


def min_feasible_n(alpha: float) -> int:
    """Minimum positive sample size ceil(1/alpha - 1) for a non-vacuous split-conformal quantile."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil(1.0 / alpha - 1.0)


def fit_mondrian_threshold(
    scores_pos: np.ndarray,
    alpha: float,
    pac_delta: Optional[float] = None,
) -> float:
    """Compute Mondrian split-conformal lower quantile threshold on final emitted scores.

    Refuses to emit a threshold when the positive sample size ``n_g`` is smaller than
    ``ceil(1 / alpha) - 1`` (where the conformal order statistic ``floor(alpha * (n_g + 1))``
    collapses to 0).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    arr = np.asarray(scores_pos, dtype=np.float64).ravel()
    n_g = int(arr.size)
    min_n = math.ceil(1.0 / alpha) - 1
    if n_g < min_n:
        raise ValueError(
            f"Infeasible Mondrian sample size n_g={n_g} < {min_n} required for alpha={alpha}"
        )
    if pac_delta is not None:
        if not (0.0 < pac_delta < 1.0):
            raise ValueError(f"pac_delta must be in (0, 1), got {pac_delta}")
        min_pac_n = math.ceil(math.log(pac_delta) / math.log(1.0 - alpha))
        if n_g < min_pac_n:
            raise ValueError(
                f"Infeasible PAC sample size n_g={n_g} < {min_pac_n} for alpha={alpha}, delta={pac_delta}"
            )
    k = math.floor(alpha * (n_g + 1))
    if k < 1:
        raise ValueError(f"Order statistic k={k} < 1 for n_g={n_g}, alpha={alpha}")
    s_sorted = np.sort(arr)
    return float(s_sorted[k - 1])


def refit_threshold_for_temperature(tau_1: float, temperature: float) -> float:
    """Transform an unscaled threshold ``tau_1`` to ``tau_T`` under temperature ``T`` so decisions are invariant."""
    if not (0.0 < tau_1 < 1.0):
        raise ValueError(f"tau_1 must be in (0, 1), got {tau_1}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    logit_tau1 = math.log(tau_1 / (1.0 - tau_1))
    return 1.0 / (1.0 + math.exp(-logit_tau1 / float(temperature)))
