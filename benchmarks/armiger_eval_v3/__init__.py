"""Armiger Evaluation Protocol v3.1 (`armiger-eval-v3.1`) statistical & cryptographic primitives."""
from .conformal import fit_mondrian_threshold, refit_threshold_for_temperature
from .estimators import hajek, ppi_plusplus_mean
from .gate import required_human_n
from .keys import comparison_key, score_on_support
from .ledger import commit_stratum

__all__ = [
    "fit_mondrian_threshold",
    "refit_threshold_for_temperature",
    "hajek",
    "ppi_plusplus_mean",
    "required_human_n",
    "comparison_key",
    "score_on_support",
    "commit_stratum",
]
