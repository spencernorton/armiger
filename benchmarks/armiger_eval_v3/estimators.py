"""Design-based Hájek IPW, Power-Tuned PPI++, and Alias-Aware Union-Find Family Ledger."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple
import numpy as np


def hajek(a: np.ndarray, pi: np.ndarray, r: np.ndarray) -> Tuple[float, float]:
    """Compute the Hájek inverse-probability-weighted mean and sandwich variance estimate."""
    a_arr = np.asarray(a, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    if np.any((pi_arr <= 0.0) | (pi_arr > 1.0)):
        raise AssertionError("positivity violated: pi must be in (0, 1]")
    w = r_arr / pi_arr
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        raise AssertionError("empty realized sample: sum(r / pi) == 0")
    est = float((w * a_arr).sum() / w_sum)
    var = float(
        ((r_arr * (1.0 - pi_arr) / (pi_arr ** 2)) * ((a_arr - est) ** 2)).sum()
        / (w_sum ** 2)
    )
    return est, var


def stratified_hajek(
    a_by_stratum: Mapping[str, np.ndarray],
    pi_by_stratum: Mapping[str, np.ndarray],
    r_by_stratum: Mapping[str, np.ndarray],
    w_fixed: Mapping[str, float],
    w_current: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compute fixed-prior (mu_strat) and current-intake (mu_current) stratified Hájek estimates."""
    per_stratum: Dict[str, Dict[str, float]] = {}
    for s, a_s in a_by_stratum.items():
        est_s, var_s = hajek(a_s, pi_by_stratum[s], r_by_stratum[s])
        per_stratum[s] = {"est": est_s, "var": var_s}
    w_f_sum = sum(w_fixed.get(s, 0.0) for s in per_stratum) or 1.0
    mu_strat = sum((w_fixed.get(s, 0.0) / w_f_sum) * per_stratum[s]["est"] for s in per_stratum)
    var_strat = sum(((w_fixed.get(s, 0.0) / w_f_sum) ** 2) * per_stratum[s]["var"] for s in per_stratum)
    w_c = w_current if w_current is not None else w_fixed
    w_c_sum = sum(w_c.get(s, 0.0) for s in per_stratum) or 1.0
    mu_curr = sum((w_c.get(s, 0.0) / w_c_sum) * per_stratum[s]["est"] for s in per_stratum)
    var_curr = sum(((w_c.get(s, 0.0) / w_c_sum) ** 2) * per_stratum[s]["var"] for s in per_stratum)
    return {
        "mu_strat": float(mu_strat),
        "var_strat": float(var_strat),
        "mu_current": float(mu_curr),
        "var_current": float(var_curr),
        "strata": per_stratum,
    }


def ppi_plusplus_mean(
    Y_lab: np.ndarray,
    f_lab: np.ndarray,
    f_unl: np.ndarray,
) -> float:
    """Power-tuned PPI++ estimator with closed-form optimal shrinkage weight lambda in [0, 1]."""
    y_l = np.asarray(Y_lab, dtype=np.float64).ravel()
    f_l = np.asarray(f_lab, dtype=np.float64).ravel()
    f_u = np.asarray(f_unl, dtype=np.float64).ravel()
    n = int(y_l.size)
    N = int(f_u.size)
    if n < 2 or N < 1:
        return float(y_l.mean())
    cov_yf = float(np.cov(y_l, f_l, ddof=1)[0, 1])
    var_f = float(np.var(f_l, ddof=1))
    denom = (1.0 + (n / float(N))) * var_f
    if denom <= 1e-12:
        lam = 0.0
    else:
        lam = float(np.clip(cov_yf / denom, 0.0, 1.0))
    return float(y_l.mean() + lam * (f_u.mean() - f_l.mean()))


def ppi_plusplus_paired_diff(
    D_lab: np.ndarray,
    f_diff_lab: np.ndarray,
    f_diff_unl: np.ndarray,
    pi_lab: Optional[np.ndarray] = None,
    family_ids_lab: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Power-tuned PPI++ on paired differences D = Z_A - Z_B with optional Hájek IPW weights
    and family-cluster sandwich variance (descriptive-only when n < required_human_n)."""
    d_l = np.asarray(D_lab, dtype=np.float64).ravel()
    fl = np.asarray(f_diff_lab, dtype=np.float64).ravel()
    fu = np.asarray(f_diff_unl, dtype=np.float64).ravel()
    n = int(d_l.size)
    N = int(fu.size)
    cov_df = float(np.cov(d_l, fl, ddof=1)[0, 1]) if n >= 2 else 0.0
    var_f = float(np.var(fl, ddof=1)) if n >= 2 else 0.0
    denom = (1.0 + (n / float(max(1, N)))) * var_f
    lam = float(np.clip(cov_df / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    resid = d_l - lam * fl
    if pi_lab is not None:
        w = 1.0 / np.asarray(pi_lab, dtype=np.float64).ravel()
        w = w / w.sum()
        rect_mean = float((w * resid).sum())
    else:
        rect_mean = float(resid.mean())
    est = float(rect_mean + lam * float(fu.mean()))
    if family_ids_lab is not None and n >= 2:
        fams = np.asarray(family_ids_lab)
        uniq = np.unique(fams)
        g_sums = np.array([resid[fams == g].sum() - (resid[fams == g].size * rect_mean) for g in uniq], dtype=np.float64)
        var_rect = float((g_sums ** 2).sum() / (max(1, n) ** 2))
    else:
        var_rect = float(np.var(resid, ddof=1) / max(1, n)) if n >= 2 else 0.0
    var_total = float(var_rect + (lam ** 2) * var_f / max(1, N))
    return {
        "estimate": est,
        "lambda_hat": lam,
        "variance": var_total,
        "se": float(np.sqrt(max(0.0, var_total))),
        "n_labeled": n,
        "n_unlabeled": N,
        "status": "descriptive_only" if n < 3925 else "powered",
    }


_FAMILY_STRIP_RE = re.compile(
    r"(?i)(?:\bS\d{1,2}(?:E\d{1,3})?\b|\bSeason\s*\d+\b|\bEpisode\s*\d+\b|\bEp\.?\s*\d+\b|"
    r"\b(?:2160p|1080p|720p|480p|576p|4k|web-dl|webdl|webrip|bluray|blu-ray|bdrip|brrip|hdtv|dvdrip|remux|"
    r"x264|x265|h264|h265|hevc|avc|av1|xvid|aac|ac3|ddp5|dts|truehd|atmos|flac|mp3|multi|dual|proper|repack)\b|"
    r"\[[^\]]*\]|\([^\)]*\)|-[A-Za-z0-9]{2,20}$)"
)


def normalize_family_stem(raw_name: str) -> str:
    s = unicodedata.normalize("NFKC", str(raw_name or "")).casefold()
    s = _FAMILY_STRIP_RE.sub(" ", s)
    toks = re.findall(r"[a-z0-9]{2,}", s)
    return " ".join(toks[:5]) if toks else s.strip()


def release_family_id(raw_name: str) -> int:
    """Deterministic 31-bit integer cluster ID for a release family stem (for cluster bootstrap)."""
    stem = normalize_family_stem(raw_name)
    return int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


class UnionFindFamilyLedger:
    """Alias-aware Union-Find family ledger over normalized stem, catalog alias ID, infohash, and file-tree hash."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._split_roots: Dict[str, Set[str]] = {}
        self._tombstoned_family_ids: Set[int] = set()

    def _find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            return x
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        curr = x
        while curr != root:
            nxt = self._parent[curr]
            self._parent[curr] = root
            curr = nxt
        return root

    def _union(self, a: str, b: str) -> str:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            if ra < rb:
                self._parent[rb] = ra
                return ra
            self._parent[ra] = rb
            return rb
        return ra

    def register_item(self, item: Mapping[str, Any], split_name: str, is_eval: bool = True) -> str:
        keys = []
        name = str(item.get("name") or item.get("title") or "")
        if name:
            keys.append(f"stem:{normalize_family_stem(name)}")
            if is_eval:
                self._tombstoned_family_ids.add(release_family_id(name))
        if item.get("catalog_alias_id"):
            keys.append(f"alias:{str(item['catalog_alias_id']).strip().lower()}")
        if item.get("info_hash") or item.get("ih"):
            keys.append(f"ih:{str(item.get('info_hash') or item.get('ih')).strip().lower()}")
        if item.get("file_tree_hash"):
            keys.append(f"fth:{str(item['file_tree_hash']).strip().lower()}")
        if not keys:
            keys.append(f"id:{item.get('id', 'unknown')}")
        first = keys[0]
        self._find(first)
        for k in keys[1:]:
            self._union(first, k)
        root = self._find(first)
        self._split_roots.setdefault(split_name, set()).add(root)
        return root

    def register_split(self, split_name: str, items: Iterable[Mapping[str, Any]], is_eval: bool = True) -> None:
        for it in items:
            self.register_item(it, split_name=split_name, is_eval=is_eval)

    def overlapping_roots(self, split_a: str, split_b: str) -> Set[str]:
        ra = {self._find(x) for x in self._split_roots.get(split_a, set())}
        rb = {self._find(x) for x in self._split_roots.get(split_b, set())}
        return ra & rb

    def assert_family_disjoint(self, split_a: str, split_b: str) -> None:
        overlap = self.overlapping_roots(split_a, split_b)
        if overlap:
            raise AssertionError(
                f"Entity-family leakage between {split_a!r} and {split_b!r}: {len(overlap)} shared family roots"
            )

    @property
    def tombstoned_family_ids(self) -> Set[int]:
        return set(self._tombstoned_family_ids)

    def filter_rsi_catalog_delta(self, candidates: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Block RSI/DHT zero-cue catalog deltas for any tombstoned evaluation family."""
        allowed = []
        for cand in candidates:
            name = str(cand.get("name") or cand.get("title") or "")
            fid = int(cand.get("family_id") or release_family_id(name))
            stem_key = f"stem:{normalize_family_stem(name)}"
            if fid in self._tombstoned_family_ids or (
                stem_key in self._parent and any(
                    self._find(stem_key) == self._find(r)
                    for s_roots in self._split_roots.values()
                    for r in s_roots
                )
            ):
                continue
            allowed.append(cand)
        return allowed
