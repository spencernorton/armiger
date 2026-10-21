"""Sample-size and power feasibility calculators for Stage-3 / Stage-5 gates."""
from __future__ import annotations

import math
from statistics import NormalDist


def required_human_n(
    delta: float,
    p_disc: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Return minimum i.i.d. human anchor sample size required to detect paired margin ``delta``."""
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if not (0.0 < p_disc <= 1.0):
        raise ValueError(f"p_disc must be in (0, 1], got {p_disc}")
    if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise ValueError("alpha and power must be in (0, 1)")
    nd = NormalDist()
    z_alpha = nd.inv_cdf(1.0 - alpha / 2.0)
    z_beta = nd.inv_cdf(power)
    sigma2_d = p_disc
    return int(math.ceil(((z_alpha + z_beta) ** 2) * sigma2_d / (delta ** 2)))
