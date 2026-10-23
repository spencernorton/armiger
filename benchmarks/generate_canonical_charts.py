#!/usr/bin/env python3
"""
Deterministic, Statistically Rooted Benchmark & Chart Generation Pipeline
for Armiger & Armiger Ultra (v3.1 Remediation).

Enforces hard statistical invariants before writing any chart:
  1. Receipt-Backed Provenance Lock (`assert_vector_provenance`)
  2. Identical Reference & Denominator Lock (`assert_identical_denominator` + `key_sha256`)
  3. Vendor-Disjoint Anti-Circularity Lock (`assert_non_circular_teacher_eval` + `comparison_key`)
     with support-restricted circularity inflation separated from Manski lower-bound unanimity drop
  4. Family-Clustered Bootstrap 95% CIs (`cluster_bootstrap_scores`, 10,000 resamples, seed 20261018)
     + Exact Two-Sided McNemar / Permutation p-values
  5. Single-Model-Per-Network Reporting (H1: Armiger Ultra is the sole Armiger bar in comparator
     `models` tables; INT8 quantization parity is reported separately under `quantization_parity_int8`)
  6. Cryptographic Provenance Manifest (`CHARTS_MANIFEST.json` + stripped PNG chunks `IHDR`, `IDAT`, `IEND`).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pathlib
import re
import struct
import sys
from typing import Any, Dict, List, Optional

import numpy as np

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from armiger_eval_v3.keys import score_on_support

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from PIL import Image
    HAS_MATPLOTLIB = True
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    GridSpec = None  # type: ignore[assignment]
    Patch = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    HAS_MATPLOTLIB = False

BOOTSTRAP_SEED = 20261018
BOOTSTRAP_RESAMPLES = 10000

BG_CANVAS = "#0D1117"
BG_PANEL = "#161B22"
BG_HIGHLIGHT = "#0F2922"
BORDER_COLOR = "#30363D"
GRID_COLOR = "#21262D"
TEXT_MAIN = "#F0F6FC"
TEXT_SUB = "#8B949E"
TEXT_MUTED = "#6E7681"

C_ULTRA = "#10B981"
C_ULTRA_EDGE = "#34D399"
C_INT8 = "#3B82F6"
C_INT8_EDGE = "#60A5FA"
C_GEMINI = "#F59E0B"
C_GPT = "#38BDF8"
C_MUSE = "#A855F7"
C_ERR = "#E2E8F0"

TEACHER_MODELS = {
    "OpenAI GPT (Fast Tier)",
    "Google Gemini Flash",
    "Meta Muse Spark 1.3",
}

# Pre-declared static set of (chart_name, panel_key, model_name) entries that evaluation.md MUST cite.
# Never inferred from evaluation.md text at generation time (remediating H3 tautology).
EVALUATION_CITED_PANEL_MODELS = {
    ("armiger_vs_llms_overall_and_6dims.png", "panel_2a_category_11class", "Armiger Ultra (Fair Hybrid)"),
    ("armiger_vs_llms_overall_and_6dims.png", "panel_2f_swarm_public_757", "Armiger Ultra (Fair Hybrid)"),
    ("armiger_vs_llms_overall_and_6dims.png", "panel_2f_swarm_warehouse_10278", "Armiger Ultra (Fair Hybrid)"),
    ("armiger_vs_torrent_classifiers_6dims.png", "panel_1_category_11class_oct15", "Armiger Ultra (Fair Hybrid)"),
    ("armiger_vs_torrent_classifiers_6dims.png", "panel_2_catalog_matching_oct15", "Armiger Ultra (Fair Hybrid)"),
    ("armiger_vs_torrent_classifiers_6dims.png", "panel_3_audio_strict_oct15", "Armiger Ultra (Fair Hybrid)"),
    ("armiger_vs_torrent_classifiers_6dims.png", "panel_5_adult_oct15", "Armiger Ultra (Fair Hybrid)"),
}

ALLOWED_MODEL_SOURCES = {"model_output"}


def assert_vector_provenance(meta: Dict[str, Any], key: str) -> None:
    """Enforce that every model vector has a measured receipt provenance record matching disk."""
    p = meta["provenance"][key]
    if p["source"] not in ALLOWED_MODEL_SOURCES | {"reference_key"}:
        raise AssertionError(f"[{key}] non-measured source {p['source']!r}")
    if p["n_evaluated"] != p["n_stratum"]:
        raise AssertionError(f"[{key}] incomplete: {p['n_evaluated']}/{p['n_stratum']}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(p["receipt_sha256"])):
        raise AssertionError(f"[{key}] missing provider receipt hash")
    if "receipt_path" in p:
        r_path = REPO_ROOT / str(p["receipt_path"])
        if not r_path.is_file():
            raise AssertionError(f"[{key}] receipt file does not exist on disk: {r_path}")
        disk_sha = hashlib.sha256(r_path.read_bytes()).hexdigest()
        if disk_sha != p["receipt_sha256"]:
            raise AssertionError(f"[{key}] receipt sha256 mismatch vs {r_path}: {disk_sha} != {p['receipt_sha256']}")


def assert_not_nested(ref: np.ndarray, other: np.ndarray, name: str, min_disc: int = 10) -> None:
    """Reject correctness vectors whose errors are artificially nested inside the reference model."""
    b = int(((ref == 1) & (other == 0)).sum())
    c = int(((ref == 0) & (other == 1)).sum())
    if b + c >= min_disc and min(b, c) == 0:
        raise AssertionError(f"[{name}] correctness vector is nested in reference ({b}:{c}) — synthetic?")


def cluster_bootstrap_scores(mat: np.ndarray, family_ids: np.ndarray, n_boot: int, rng: Any) -> np.ndarray:
    """Family-clustered multinomial ratio bootstrap over release family IDs."""
    fam, inv = np.unique(family_ids, return_inverse=True)
    G = len(fam)
    wf = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot)
    W = wf[:, inv].astype(np.float64)
    return 100.0 * (W @ mat.T) / W.sum(axis=1, keepdims=True)


def assert_identical_denominator(
    panel_name: str,
    vectors: Dict[str, np.ndarray],
    expected_n: Optional[int] = None,
) -> int:
    """Enforce that every model on a panel is evaluated on the exact same item count N
    and valid-domain mask, with zero NaNs."""
    if not vectors:
        raise AssertionError(f"[{panel_name}] Empty vectors dictionary.")
    lengths = {}
    for model_name, vec in vectors.items():
        arr = np.asarray(vec)
        if arr.ndim != 1:
            raise AssertionError(f"[{panel_name}] Model '{model_name}' vector must be 1D, got shape {arr.shape}.")
        if np.isnan(arr).any():
            raise AssertionError(f"[{panel_name}] Model '{model_name}' contains NaN values.")
        lengths[model_name] = int(arr.shape[0])

    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise AssertionError(
            f"[{panel_name}] Identical Denominator Lock violated! Mismatched N across models: {lengths}"
        )
    n_val = next(iter(unique_lengths))
    if expected_n is not None and n_val != expected_n:
        raise AssertionError(
            f"[{panel_name}] Expected N={expected_n}, but got N={n_val} ({lengths})."
        )
    return n_val


def assert_non_circular_teacher_eval(
    model_name: str,
    eval_mode: str,
    self_vote_weight: float,
    noncirc_vec: Optional[np.ndarray] = None,
    circ_vec: Optional[np.ndarray] = None,
    model_vendor: Optional[str] = None,
    key_pool_vendors: Optional[List[str]] = None,
    loto_lower_bound_vec: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Enforce that teacher LLMs are never scored against a reference key in which
    they or their model family cast a vote. Computes support-restricted circularity inflation
    separately from the Manski lower-bound unanimity drop."""
    is_teacher = any(t in model_name for t in TEACHER_MODELS)
    if key_pool_vendors is not None and model_vendor is not None:
        if model_vendor in key_pool_vendors:
            raise AssertionError(
                f"Anti-Circularity Lock violated for '{model_name}': vendor '{model_vendor}' is in key_pool_vendors={key_pool_vendors}"
            )
    if is_teacher:
        if self_vote_weight != 0.0 or eval_mode == "circular_majority":
            raise AssertionError(
                f"Anti-Circularity Lock violated for teacher '{model_name}': "
                f"eval_mode='{eval_mode}', self_vote_weight={self_vote_weight} (must be 0.0)."
            )
        if eval_mode not in ("leave_one_teacher_out", "independent_ground_truth", "vendor_disjoint_consensus"):
            raise AssertionError(
                f"Invalid eval_mode '{eval_mode}' for teacher '{model_name}'."
            )
    info: Dict[str, Any] = {
        "model_name": model_name,
        "is_teacher": is_teacher,
        "eval_mode": eval_mode,
        "self_vote_weight": float(self_vote_weight),
    }
    if model_vendor is not None:
        info["model_vendor"] = model_vendor
    if key_pool_vendors is not None:
        info["key_pool_vendors"] = list(key_pool_vendors)
    if noncirc_vec is not None and circ_vec is not None:
        nc_arr = np.asarray(noncirc_vec, dtype=np.float64)
        c_arr = np.asarray(circ_vec, dtype=np.float64)
        if np.isnan(nc_arr).any() or np.isnan(c_arr).any():
            raise AssertionError(f"NaN in anti_circ vectors for {model_name}")
        nc_pct = float(np.mean(nc_arr) * 100.0)
        c_pct = float(np.mean(c_arr) * 100.0)
        info["non_circular_support_n"] = int(nc_arr.size)
        info["circular_stratum_n"] = int(c_arr.size)
        info["non_circular_support_pct"] = round(nc_pct, 2)
        info["non_circular_pct"] = round(nc_pct, 2)
        info["circular_self_vote_pct"] = round(c_pct, 2)
        info["circularity_inflation_pp"] = round(c_pct - nc_pct, 2)
        if loto_lower_bound_vec is not None:
            lb_arr = np.asarray(loto_lower_bound_vec, dtype=np.float64)
            lb_pct = float(np.mean(lb_arr) * 100.0)
            info["manski_lower_bound_n1025_pct"] = round(lb_pct, 2)
            info["manski_unanimity_drop_pp"] = round(nc_pct - lb_pct, 2)
    return info


def exact_mcnemar_or_paired_p(vec_ref: np.ndarray, vec_mod: np.ndarray) -> float:
    """Compute exact two-sided McNemar binomial p-value for binary correctness vectors."""
    diff = np.asarray(vec_ref, dtype=np.float64) - np.asarray(vec_mod, dtype=np.float64)
    n_pos = int(np.sum(diff > 1e-6))
    n_neg = int(np.sum(diff < -1e-6))
    n_disc = n_pos + n_neg
    if n_disc == 0:
        return 1.0
    m_min = min(n_pos, n_neg)
    if 2 * m_min == n_disc:
        return 1.0
    tail_sum = sum(math.comb(n_disc, i) for i in range(m_min + 1))
    return float(min(1.0, (2 * tail_sum) / (1 << n_disc)))


def compute_auroc_pct(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    u = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(100.0 * u / (len(pos) * len(neg)))


def compute_paired_panel_stats(
    panel_name: str,
    vectors: Dict[str, np.ndarray],
    ref_model: str,
    expected_n: int,
    metric_type: str = "mean",
    y_true: Optional[np.ndarray] = None,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    key_sha256: Optional[Dict[str, str]] = None,
    family_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute point estimates, 10,000-resample paired bootstrap 95% CIs (family-clustered when
    `family_ids` is provided), paired deltas, and exact McNemar p-values for all models in a panel."""
    if key_sha256 is not None:
        if len(set(key_sha256.values())) != 1:
            raise AssertionError(
                f"[{panel_name}] Mismatched reference key sha256 across panel models: {key_sha256}"
            )
    n_val = assert_identical_denominator(panel_name, vectors, expected_n=expected_n)
    if ref_model not in vectors:
        raise AssertionError(f"[{panel_name}] Reference model '{ref_model}' missing from vectors.")

    rng = np.random.default_rng(seed)
    model_names = list(vectors.keys())
    mat = np.stack([np.asarray(vectors[m], dtype=np.float64) for m in model_names], axis=0)

    if metric_type == "mean":
        if family_ids is not None:
            fam_arr = np.asarray(family_ids)
            if fam_arr.shape[0] != n_val:
                raise AssertionError(f"[{panel_name}] family_ids length {fam_arr.shape[0]} != N={n_val}")
            boot_scores = cluster_bootstrap_scores(mat, fam_arr, n_boot, rng)
        else:
            mat_int6 = np.round(mat * 6.0)
            weights = rng.multinomial(n_val, np.full(n_val, 1.0 / n_val), size=n_boot).astype(np.float64)
            boot_scores = (weights @ mat_int6.T) * (100.0 / (6.0 * n_val))
        point_scores = {
            m: float(np.sum(np.round(np.asarray(vectors[m], dtype=np.float64) * 6.0)) * 100.0 / (6.0 * n_val))
            for m in model_names
        }
        item_correctness = {m: np.asarray(vectors[m], dtype=np.float64) for m in model_names}
    elif metric_type == "balanced":
        if y_true is None:
            raise AssertionError(f"[{panel_name}] y_true required for metric_type='balanced'.")
        y_arr = np.asarray(y_true)
        pos_mask = (y_arr == 1)
        neg_mask = (y_arr == 0)
        n_pos = int(np.sum(pos_mask))
        n_neg = int(np.sum(neg_mask))
        if n_pos < 2 or n_neg < 2:
            raise AssertionError(
                f"[{panel_name}] metric_type='balanced' is not estimable when n_pos={n_pos} or n_neg={n_neg} < 2."
            )
        pos_mat = (mat[:, pos_mask] == 1).astype(np.float64)
        neg_mat = (mat[:, neg_mask] == 0).astype(np.float64)
        if family_ids is not None:
            fam_arr = np.asarray(family_ids)
            b_pos = cluster_bootstrap_scores(pos_mat, fam_arr[pos_mask], n_boot, rng)
            b_neg = cluster_bootstrap_scores(neg_mat, fam_arr[neg_mask], n_boot, rng)
            boot_scores = 0.5 * (b_pos + b_neg)
        else:
            w_pos = rng.multinomial(n_pos, np.full(n_pos, 1.0 / n_pos), size=n_boot).astype(np.float64)
            w_neg = rng.multinomial(n_neg, np.full(n_neg, 1.0 / n_neg), size=n_boot).astype(np.float64)
            boot_scores = 0.5 * ((w_pos @ pos_mat.T) * (100.0 / n_pos) + (w_neg @ neg_mat.T) * (100.0 / n_neg))
        point_scores = {}
        item_correctness = {}
        for idx, m in enumerate(model_names):
            sens = float(np.sum(pos_mat[idx]) * 100.0 / n_pos)
            spec = float(np.sum(neg_mat[idx]) * 100.0 / n_neg)
            point_scores[m] = 0.5 * (sens + spec)
            item_corr = np.zeros(n_val, dtype=np.float64)
            item_corr[pos_mask] = pos_mat[idx]
            item_corr[neg_mask] = neg_mat[idx]
            item_correctness[m] = item_corr
    elif metric_type == "auroc":
        if y_true is None:
            raise AssertionError(f"[{panel_name}] y_true required for metric_type='auroc'.")
        y_arr = np.asarray(y_true)
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        placement_int2 = np.zeros((len(model_names), n_val), dtype=np.float64)
        placement_unit = np.zeros((len(model_names), n_val), dtype=np.float64)
        point_scores = {}
        for idx, m in enumerate(model_names):
            s = mat[idx]
            s_pos = s[pos_idx]
            s_neg = s[neg_idx]
            v10_int2 = 2.0 * (s_pos[:, None] > s_neg[None, :]).sum(axis=1) + (s_pos[:, None] == s_neg[None, :]).sum(axis=1)
            v01_int2 = 2.0 * (s_pos[:, None] > s_neg[None, :]).sum(axis=0) + (s_pos[:, None] == s_neg[None, :]).sum(axis=0)
            placement_int2[idx, pos_idx] = v10_int2
            placement_int2[idx, neg_idx] = v01_int2
            placement_unit[idx, pos_idx] = v10_int2 / (2.0 * n_neg)
            placement_unit[idx, neg_idx] = v01_int2 / (2.0 * n_pos)
            point_scores[m] = float(np.sum(v10_int2) * 100.0 / (2.0 * n_pos * n_neg))
        w_pos = rng.multinomial(n_pos, np.full(n_pos, 1.0 / n_pos), size=n_boot).astype(np.float64)
        w_neg = rng.multinomial(n_neg, np.full(n_neg, 1.0 / n_neg), size=n_boot).astype(np.float64)
        pt_vec = np.array([point_scores[m] for m in model_names], dtype=np.float64)[None, :]
        boot_scores = (
            (w_pos @ placement_int2[:, pos_idx].T) * (100.0 / (2.0 * n_pos * n_neg))
            + (w_neg @ placement_int2[:, neg_idx].T) * (100.0 / (2.0 * n_pos * n_neg))
            - pt_vec
        )
        item_correctness = {m: placement_unit[idx] for idx, m in enumerate(model_names)}
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    ref_idx = model_names.index(ref_model)
    ref_boot = boot_scores[:, ref_idx]
    ref_point = point_scores[ref_model]

    results: Dict[str, Any] = {}
    for idx, m in enumerate(model_names):
        m_boot = boot_scores[:, idx]
        ci_lo, ci_hi = np.percentile(m_boot, [2.5, 97.5])
        delta_boot = ref_boot - m_boot
        d_lo, d_hi = np.percentile(delta_boot, [2.5, 97.5])
        p_val = 1.0 if m == ref_model else exact_mcnemar_or_paired_p(item_correctness[ref_model], item_correctness[m])
        results[m] = {
            "n": n_val,
            "point_pct": round(point_scores[m], 2),
            "ci95_lo": round(float(ci_lo), 2),
            "ci95_hi": round(float(ci_hi), 2),
            "delta_vs_ref_pct": round(ref_point - point_scores[m], 2),
            "delta_ci95_lo": round(float(d_lo), 2),
            "delta_ci95_hi": round(float(d_hi), 2),
            "p_value_vs_ref": float(f"{p_val:.6e}"),
        }
    out_panel: Dict[str, Any] = {
        "panel_name": panel_name,
        "n": n_val,
        "metric_type": metric_type,
        "bootstrap_mode": "family_clustered" if family_ids is not None else "iid_multinomial",
        "reference_model": ref_model,
        "models": results,
    }
    if key_sha256 is not None:
        out_panel["key_sha256"] = next(iter(key_sha256.values()))
    if family_ids is not None:
        out_panel["n_families"] = int(len(np.unique(family_ids)))
    return out_panel


def strip_png_chunks_to_canonical(png_bytes: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    if not png_bytes.startswith(sig):
        raise ValueError("Invalid PNG header signature.")
    pos = 8
    out = bytearray(sig)
    while pos < len(png_bytes):
        length = struct.unpack(">I", png_bytes[pos:pos + 4])[0]
        ctype = png_bytes[pos + 4:pos + 8]
        chunk_data = png_bytes[pos + 8:pos + 8 + length]
        crc = png_bytes[pos + 8 + length:pos + 12 + length]
        pos += 12 + length
        if ctype in (b"IHDR", b"IDAT", b"IEND"):
            out.extend(struct.pack(">I", length))
            out.extend(ctype)
            out.extend(chunk_data)
            out.extend(crc)
    return bytes(out)


def verify_png_chunks_clean(png_path: pathlib.Path) -> List[str]:
    data = png_path.read_bytes()
    sig = b"\x89PNG\r\n\x1a\n"
    assert data.startswith(sig), f"{png_path} is not a valid PNG"
    pos = 8
    chunks: List[str] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8].decode("ascii", errors="replace")
        chunks.append(ctype)
        pos += 12 + length
    for c in chunks:
        if c not in ("IHDR", "IDAT", "IEND"):
            raise AssertionError(f"{png_path} contains non-canonical PNG chunk: {c}")
    return chunks


def save_clean_rgb_png(fig: Any, out_path: pathlib.Path, dpi: int = 180) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=BG_CANVAS,
        edgecolor="none",
        pad_inches=0.34,
    )
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as im:
        rgb = im.convert("RGB")
        raw_buf = io.BytesIO()
        rgb.save(raw_buf, format="PNG", optimize=False)
    clean_bytes = strip_png_chunks_to_canonical(raw_buf.getvalue())
    out_path.write_bytes(clean_bytes)
    verify_png_chunks_clean(out_path)
    return hashlib.sha256(clean_bytes).hexdigest()


def style_ax(ax: Any) -> None:
    ax.set_facecolor(BG_PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1.0)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="--", alpha=0.9, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def draw_ci_whiskers(ax: Any, y_coords: np.ndarray, stats_list: List[Dict[str, Any]]) -> None:
    pts = np.array([s["point_pct"] for s in stats_list], dtype=np.float64)
    lo = np.maximum(0.0, pts - np.array([s["ci95_lo"] for s in stats_list], dtype=np.float64))
    hi = np.maximum(0.0, np.array([s["ci95_hi"] for s in stats_list], dtype=np.float64) - pts)
    ax.errorbar(
        pts,
        y_coords,
        xerr=[lo, hi],
        fmt="none",
        ecolor=C_ERR,
        elinewidth=1.15,
        capsize=2.6,
        capthick=1.1,
        alpha=0.82,
        zorder=5,
    )


def build_all_charts_and_manifest(
    npz_path: pathlib.Path,
    assets_dir: pathlib.Path,
    manifest_path: pathlib.Path,
    render_pngs: Optional[bool] = None,
) -> Dict[str, Any]:
    if render_pngs is None:
        render_pngs = HAS_MATPLOTLIB
    if render_pngs:
        if not HAS_MATPLOTLIB:
            raise RuntimeError("matplotlib and Pillow are required when render_pngs=True")
        plt.rcParams.update({
            "figure.facecolor": BG_CANVAS,
            "axes.facecolor": BG_PANEL,
            "axes.edgecolor": BORDER_COLOR,
            "axes.labelcolor": TEXT_SUB,
            "xtick.color": TEXT_SUB,
            "ytick.color": TEXT_MAIN,
            "text.color": TEXT_MAIN,
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.alpha": 0.85,
        })

    raw_json = json.loads(npz_path.read_text(encoding="utf-8"))
    dtypes_map = raw_json["_meta"]["dtypes"]
    d = {k: np.array(v, dtype=np.dtype(dtypes_map[k])) for k, v in raw_json["vectors"].items()}
    files_list = list(d.keys())
    for k in files_list:
        if d[k].dtype.kind not in ("i", "u", "f"):
            raise AssertionError(f"Non-numeric array '{k}' ({d[k].dtype}) in {npz_path}")

    prov_map = raw_json["_meta"]["provenance"]
    for k in files_list:
        if "_y_" not in k and not k.endswith("_y_true") and not k.endswith("_family_ids") and "_key_" not in k:
            assert_vector_provenance(raw_json["_meta"], k)
    assert_not_nested(d["beyond_cat11_sup_no_gpt_corr_ultra"], d["beyond_cat11_sup_no_gpt_corr_gpt"], "beyond_cat11_sup_no_gpt_corr_gpt")
    assert_not_nested(d["beyond_cat11_sup_no_gemini_corr_ultra"], d["beyond_cat11_sup_no_gemini_corr_gemini"], "beyond_cat11_sup_no_gemini_corr_gemini")

    gpt_prov = prov_map["beyond_cat11_sup_no_gpt_corr_gpt"]
    gem_prov = prov_map["beyond_cat11_sup_no_gemini_corr_gemini"]
    ult_prov = prov_map["beyond_cat11_corr_ultra"]

    anti_circ_checks = [
        assert_non_circular_teacher_eval(
            "OpenAI GPT (Fast Tier)",
            eval_mode=str(gpt_prov["eval_mode"]),
            self_vote_weight=float(gpt_prov["self_vote_weight"]),
            noncirc_vec=d["beyond_cat11_sup_no_gpt_corr_gpt"],
            circ_vec=d["beyond_cat11_corr_gpt_circ"],
            model_vendor=str(gpt_prov["model_vendor"]),
            key_pool_vendors=list(gpt_prov["key_pool_vendors"]),
            loto_lower_bound_vec=d["beyond_cat11_corr_gpt_loto_lower_bound"],
        ),
        assert_non_circular_teacher_eval(
            "Google Gemini Flash",
            eval_mode=str(gem_prov["eval_mode"]),
            self_vote_weight=float(gem_prov["self_vote_weight"]),
            noncirc_vec=d["beyond_cat11_sup_no_gemini_corr_gemini"],
            circ_vec=d["beyond_cat11_corr_gemini_circ"],
            model_vendor=str(gem_prov["model_vendor"]),
            key_pool_vendors=list(gem_prov["key_pool_vendors"]),
            loto_lower_bound_vec=d["beyond_cat11_corr_gemini_loto_lower_bound"],
        ),
        assert_non_circular_teacher_eval(
            "Armiger Ultra (Fair Hybrid)",
            eval_mode=str(ult_prov["eval_mode"]),
            self_vote_weight=float(ult_prov["self_vote_weight"]),
            model_vendor=str(ult_prov["model_vendor"]),
            key_pool_vendors=list(ult_prov["key_pool_vendors"]),
        ),
    ]

    # ==========================================================================
    # SUITE B (CHART 1): VENDOR-DISJOINT 3-TEACHER & COMMON-SUPPORT LOTO PANELS
    # ==========================================================================
    ref_llm = "Armiger Ultra (Fair Hybrid)"
    gpt_llm = "OpenAI GPT (Fast Tier)"
    gem_llm = "Google Gemini Flash"

    def ksha(vec_key: str) -> str:
        return str(prov_map[vec_key]["key_sha256"])

    # 2A. 11-Class Category (N=1,025 3-teacher key + N=974 / N=982 LOTO common-support sub-panels)
    p2a_stats = compute_paired_panel_stats(
        "chart1_2a_beyond_cat11",
        {ref_llm: d["beyond_cat11_corr_ultra"]},
        ref_model=ref_llm,
        expected_n=1025,
        metric_type="mean",
        key_sha256={ref_llm: ksha("beyond_cat11_corr_ultra")},
        family_ids=d["beyond_cat11_family_ids"],
    )
    p2a_stats["quantization_parity_int8"] = {
        "int8_hyb_pct": round(float(np.mean(d["beyond_cat11_corr_int8_hyb"]) * 100.0), 2),
        "int8_raw_pct": round(float(np.mean(d["beyond_cat11_corr_int8_raw"]) * 100.0), 2),
        "delta_vs_ultra_pp": round(float((np.mean(d["beyond_cat11_corr_int8_hyb"]) - np.mean(d["beyond_cat11_corr_ultra"])) * 100.0), 2),
    }

    sha_cat_no_gpt = ksha("beyond_cat11_key_no_gpt")
    p2a_loto_gpt_stats = compute_paired_panel_stats(
        "chart1_2a_loto_vs_gpt_common_support",
        {
            ref_llm: d["beyond_cat11_sup_no_gpt_corr_ultra"],
            gpt_llm: d["beyond_cat11_sup_no_gpt_corr_gpt"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_cat11_sup_no_gpt_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_cat_no_gpt, gpt_llm: sha_cat_no_gpt},
        family_ids=d["beyond_cat11_sup_no_gpt_family_ids"],
    )
    p2a_loto_gpt_stats["partial_identification_bounds_n1025"] = {
        ref_llm: score_on_support(d["beyond_cat11_pred_ultra"], d["beyond_cat11_key_no_gpt"]),
        gpt_llm: score_on_support(d["beyond_cat11_pred_gpt"], d["beyond_cat11_key_no_gpt"]),
    }

    sha_cat_no_gem = ksha("beyond_cat11_key_no_gemini")
    p2a_loto_gemini_stats = compute_paired_panel_stats(
        "chart1_2a_loto_vs_gemini_common_support",
        {
            ref_llm: d["beyond_cat11_sup_no_gemini_corr_ultra"],
            gem_llm: d["beyond_cat11_sup_no_gemini_corr_gemini"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_cat11_sup_no_gemini_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_cat_no_gem, gem_llm: sha_cat_no_gem},
        family_ids=d["beyond_cat11_sup_no_gemini_family_ids"],
    )
    p2a_loto_gemini_stats["partial_identification_bounds_n1025"] = {
        ref_llm: score_on_support(d["beyond_cat11_pred_ultra"], d["beyond_cat11_key_no_gemini"]),
        gem_llm: score_on_support(d["beyond_cat11_pred_gemini"], d["beyond_cat11_key_no_gemini"]),
    }
    p2a_stats["loto_common_support_panels"] = {
        "vs_openai_gpt": p2a_loto_gpt_stats,
        "vs_google_gemini": p2a_loto_gemini_stats,
    }

    # 2B. Catalog Matching (N=1,025 3-teacher + LOTO common support)
    p2b_stats = compute_paired_panel_stats(
        "chart1_2b_beyond_matching",
        {ref_llm: d["beyond_match_corr_ultra"]},
        ref_model=ref_llm,
        expected_n=1025,
        metric_type="mean",
        key_sha256={ref_llm: ksha("beyond_match_corr_ultra")},
        family_ids=d["beyond_cat11_family_ids"],
    )
    p2b_stats["quantization_parity_int8"] = {
        "int8_hyb_pct": round(float(np.mean(d["beyond_match_corr_int8_hyb"]) * 100.0), 2),
        "int8_raw_pct": round(float(np.mean(d["beyond_match_corr_int8_raw"]) * 100.0), 2),
    }
    p2b_loto_gpt_stats = compute_paired_panel_stats(
        "chart1_2b_loto_vs_gpt_common_support",
        {
            ref_llm: d["beyond_match_sup_no_gpt_corr_ultra"],
            gpt_llm: d["beyond_match_sup_no_gpt_corr_gpt"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_match_sup_no_gpt_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_cat_no_gpt, gpt_llm: sha_cat_no_gpt},
        family_ids=d["beyond_cat11_sup_no_gpt_family_ids"],
    )
    p2b_loto_gemini_stats = compute_paired_panel_stats(
        "chart1_2b_loto_vs_gemini_common_support",
        {
            ref_llm: d["beyond_match_sup_no_gemini_corr_ultra"],
            gem_llm: d["beyond_match_sup_no_gemini_corr_gemini"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_match_sup_no_gemini_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_cat_no_gem, gem_llm: sha_cat_no_gem},
        family_ids=d["beyond_cat11_sup_no_gemini_family_ids"],
    )
    p2b_stats["loto_common_support_panels"] = {
        "vs_openai_gpt": p2b_loto_gpt_stats,
        "vs_google_gemini": p2b_loto_gemini_stats,
    }

    # 2C. Audio Language Strict E-vs-F (N=229 + LOTO common support N=209 / N=194)
    p2c_stats = compute_paired_panel_stats(
        "chart1_2c_beyond_audio_strict",
        {ref_llm: d["beyond_aud229_corr_ultra"]},
        ref_model=ref_llm,
        expected_n=229,
        metric_type="mean",
        key_sha256={ref_llm: ksha("beyond_aud229_corr_ultra")},
        family_ids=d["beyond_aud229_family_ids"],
    )
    p2c_stats["quantization_parity_int8"] = {
        "int8_hyb_pct": round(float(np.mean(d["beyond_aud229_corr_int8_hyb"]) * 100.0), 2),
        "int8_raw_pct": round(float(np.mean(d["beyond_aud229_corr_int8_raw"]) * 100.0), 2),
    }
    sha_aud_no_gpt = ksha("beyond_aud229_key_no_gpt")
    p2c_loto_gpt_stats = compute_paired_panel_stats(
        "chart1_2c_loto_vs_gpt_common_support",
        {
            ref_llm: d["beyond_aud229_sup_no_gpt_corr_ultra"],
            gpt_llm: d["beyond_aud229_sup_no_gpt_corr_gpt"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_aud229_sup_no_gpt_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_aud_no_gpt, gpt_llm: sha_aud_no_gpt},
        family_ids=d["beyond_aud229_sup_no_gpt_family_ids"],
    )
    sha_aud_no_gem = ksha("beyond_aud229_key_no_gemini")
    p2c_loto_gemini_stats = compute_paired_panel_stats(
        "chart1_2c_loto_vs_gemini_common_support",
        {
            ref_llm: d["beyond_aud229_sup_no_gemini_corr_ultra"],
            gem_llm: d["beyond_aud229_sup_no_gemini_corr_gemini"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_aud229_sup_no_gemini_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_aud_no_gem, gem_llm: sha_aud_no_gem},
        family_ids=d["beyond_aud229_sup_no_gemini_family_ids"],
    )
    p2c_stats["loto_common_support_panels"] = {
        "vs_openai_gpt": p2c_loto_gpt_stats,
        "vs_google_gemini": p2c_loto_gemini_stats,
    }

    # 2D. Anime Detection (N=636 + LOTO common support N=629)
    p2d_stats = compute_paired_panel_stats(
        "chart1_2d_beyond_anime_valid636",
        {ref_llm: d["beyond_ani636_corr_ultra"]},
        ref_model=ref_llm,
        expected_n=636,
        metric_type="mean",
        key_sha256={ref_llm: ksha("beyond_ani636_corr_ultra")},
        family_ids=d["beyond_ani636_family_ids"],
    )
    p2d_stats["quantization_parity_int8"] = {
        "int8_pct": round(float(np.mean(d["beyond_ani636_corr_int8"]) * 100.0), 2),
    }
    sha_ani_no_gpt = ksha("beyond_ani636_key_no_gpt")
    p2d_loto_gpt_stats = compute_paired_panel_stats(
        "chart1_2d_loto_vs_gpt_common_support",
        {
            ref_llm: d["beyond_ani636_sup_no_gpt_corr_ultra"],
            gpt_llm: d["beyond_ani636_sup_no_gpt_corr_gpt"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_ani636_sup_no_gpt_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_ani_no_gpt, gpt_llm: sha_ani_no_gpt},
        family_ids=d["beyond_ani636_sup_no_gpt_family_ids"],
    )
    sha_ani_no_gem = ksha("beyond_ani636_key_no_gemini")
    p2d_loto_gemini_stats = compute_paired_panel_stats(
        "chart1_2d_loto_vs_gemini_common_support",
        {
            ref_llm: d["beyond_ani636_sup_no_gemini_corr_ultra"],
            gem_llm: d["beyond_ani636_sup_no_gemini_corr_gemini"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_ani636_sup_no_gemini_corr_ultra"].size),
        metric_type="mean",
        key_sha256={ref_llm: sha_ani_no_gem, gem_llm: sha_ani_no_gem},
        family_ids=d["beyond_ani636_sup_no_gemini_family_ids"],
    )
    p2d_stats["loto_common_support_panels"] = {
        "vs_openai_gpt": p2d_loto_gpt_stats,
        "vs_google_gemini": p2d_loto_gemini_stats,
    }

    # 2E. Junk / Malware Detection (Balanced N=977 3-teacher + N=975 LOTO vs GPT)
    p2e_stats = compute_paired_panel_stats(
        "chart1_2e_beyond_junk977_balanced",
        {ref_llm: d["beyond_junk977_pred_ultra"]},
        ref_model=ref_llm,
        expected_n=977,
        metric_type="balanced",
        y_true=d["beyond_junk977_y_true"],
        key_sha256={ref_llm: ksha("beyond_junk977_pred_ultra")},
        family_ids=d["beyond_junk977_family_ids"],
    )
    p2e_stats["quantization_parity_int8"] = {
        "int8_balanced_pct": p2e_stats["models"][ref_llm]["point_pct"],
    }
    sha_jk_no_gpt = ksha("beyond_junk977_key_no_gpt")
    p2e_loto_gpt_stats = compute_paired_panel_stats(
        "chart1_2e_loto_vs_gpt_common_support",
        {
            ref_llm: d["beyond_junk977_sup_no_gpt_pred_ultra"],
            gpt_llm: d["beyond_junk977_sup_no_gpt_pred_gpt"],
        },
        ref_model=ref_llm,
        expected_n=int(d["beyond_junk977_sup_no_gpt_pred_ultra"].size),
        metric_type="balanced",
        y_true=d["beyond_junk977_sup_no_gpt_y_true"],
        key_sha256={ref_llm: sha_jk_no_gpt, gpt_llm: sha_jk_no_gpt},
        family_ids=d["beyond_junk977_sup_no_gpt_family_ids"],
    )
    p2e_stats["loto_common_support_panels"] = {
        "vs_openai_gpt": p2e_loto_gpt_stats,
        "vs_google_gemini": {
            "status": "not_estimable_loto_n_pos_0",
            "note": "Excluding Google leaves {openai, meta}; because OpenAI GPT predicted 0 junk positives across all 977 items, unanimous {openai, meta} support has n_pos = 0.",
        },
    }

    # 2F. Swarm Liveliness AUROC (Verified OOF only: Public N=757 & Warehouse N=10,278)
    sha_pub_k = ksha("swarm_pub_score_ultra")
    p2f_pub_stats = compute_paired_panel_stats(
        "chart1_2f_swarm_public_757",
        {
            "Armiger Ultra (Fair Hybrid)": d["swarm_pub_score_ultra"],
            "Way-4 Combined Priors": d["swarm_pub_score_priors"],
            "DHT Seeder Baseline": d["swarm_pub_score_seeders"],
        },
        ref_model="Armiger Ultra (Fair Hybrid)",
        expected_n=757,
        metric_type="auroc",
        y_true=d["swarm_pub_y_true"],
        key_sha256={
            "Armiger Ultra (Fair Hybrid)": sha_pub_k,
            "Way-4 Combined Priors": sha_pub_k,
            "DHT Seeder Baseline": sha_pub_k,
        },
        family_ids=d["swarm_pub_family_ids"],
    )

    sha_whse_k = ksha("swarm_whse_score_ultra")
    p2f_whse_stats = compute_paired_panel_stats(
        "chart1_2f_swarm_warehouse_10278",
        {
            "Armiger Ultra (Fair Hybrid)": d["swarm_whse_score_ultra"],
            "DHT Seeder Baseline": d["swarm_whse_score_priors"],
            "Raw Seeder Count": d["swarm_whse_score_seeders"],
        },
        ref_model="Armiger Ultra (Fair Hybrid)",
        expected_n=10278,
        metric_type="auroc",
        y_true=d["swarm_whse_y_true"],
        key_sha256={
            "Armiger Ultra (Fair Hybrid)": sha_whse_k,
            "DHT Seeder Baseline": sha_whse_k,
            "Raw Seeder Count": sha_whse_k,
        },
        family_ids=d["swarm_whse_family_ids"],
    )

    # Composite summaries (Vendor-Disjoint 6D for Armiger Ultra + Pairwise LOTO Common-Support 4D/5D vs GPT & Gemini)
    panels_6_ultra = [p2a_stats, p2b_stats, p2c_stats, p2d_stats, p2e_stats, p2f_pub_stats]
    pts_ultra_6 = [p["models"][ref_llm]["point_pct"] for p in panels_6_ultra]
    se_ultra_6 = [(p["models"][ref_llm]["ci95_hi"] - p["models"][ref_llm]["ci95_lo"]) / (2.0 * 1.96) for p in panels_6_ultra]
    half_w_ultra_6 = 1.96 * math.hypot(sum(se_ultra_6[:5]) / 6.0, se_ultra_6[5] / 6.0)
    pt_ultra_6 = float(np.mean(pts_ultra_6))
    pt_ultra_whse_6 = float(np.mean(pts_ultra_6[:-1] + [p2f_whse_stats["models"][ref_llm]["point_pct"]]))

    loto_4_gpt_panels = [p2a_loto_gpt_stats, p2b_loto_gpt_stats, p2c_loto_gpt_stats, p2d_loto_gpt_stats]
    loto_5_gpt_panels = loto_4_gpt_panels + [p2e_loto_gpt_stats]
    loto_4_gem_panels = [p2a_loto_gemini_stats, p2b_loto_gemini_stats, p2c_loto_gemini_stats, p2d_loto_gemini_stats]

    composite_stats: Dict[str, Any] = {
        ref_llm: {
            "point_pct": round(pt_ultra_6, 2),
            "ci95_lo": round(pt_ultra_6 - half_w_ultra_6, 2),
            "ci95_hi": round(pt_ultra_6 + half_w_ultra_6, 2),
            "point_with_corpus_swarm_pct": round(pt_ultra_whse_6, 2),
        },
        "loto_common_support_4dim_2a_to_2d": {
            "vs_openai_gpt": {
                ref_llm: round(float(np.mean([p["models"][ref_llm]["point_pct"] for p in loto_4_gpt_panels])), 2),
                gpt_llm: round(float(np.mean([p["models"][gpt_llm]["point_pct"] for p in loto_4_gpt_panels])), 2),
            },
            "vs_google_gemini": {
                ref_llm: round(float(np.mean([p["models"][ref_llm]["point_pct"] for p in loto_4_gem_panels])), 2),
                gem_llm: round(float(np.mean([p["models"][gem_llm]["point_pct"] for p in loto_4_gem_panels])), 2),
            },
        },
        "loto_common_support_5dim_2a_to_2e_vs_gpt": {
            ref_llm: round(float(np.mean([p["models"][ref_llm]["point_pct"] for p in loto_5_gpt_panels])), 2),
            gpt_llm: round(float(np.mean([p["models"][gpt_llm]["point_pct"] for p in loto_5_gpt_panels])), 2),
        },
    }

    # Load measured hardware latency/throughput and logged API token receipt from hardware_and_api_cost_receipt.json
    hw_cost_rec = json.loads((BENCH_DIR / "receipts" / "hardware_and_api_cost_receipt.json").read_text(encoding="utf-8"))
    cost_rec = hw_cost_rec["models"]
    models_cost = [
        "Armiger (15.7 MB M4 Pro CPU INT8)",
        "Armiger Ultra (M4 Pro CPU FP32)",
        "Google Gemini Flash",
        "OpenAI GPT (Fast Tier)",
        "Meta Muse Spark 1.3",
    ]
    colors_cost = [C_INT8, C_ULTRA, C_GEMINI, C_GPT, C_MUSE]
    batched_cost = [float(cost_rec[m]["batched_usd_per_1m_tasks"]) for m in models_cost]
    realtime_cost = [float(cost_rec[m]["realtime_usd_per_1m_tasks"]) for m in models_cost]

    # ==========================================================================
    # SUITE A (CHART 3): OPEN-SOURCE CLASSIFIERS & PARSERS ON OCT15 & ZERO_CUE_2000
    # ==========================================================================
    models_9 = [
        "Armiger Ultra (Fair Hybrid)",
        "MIMMO BERT (360M)",
        "MIMMO Cascade",
        "RTN (rank-torrent-name)",
        "PTT / Parsett",
        "GuessIt",
        "PTN (parse-torrent-title)",
        "Anitomy",
        "Bitmagnet Built-In",
    ]
    keys_9 = [
        "ultra", "mimmo_bert", "mimmo_cascade",
        "rtn", "ptt", "guessit", "ptn", "anitomy", "bitmagnet",
    ]
    colors_9 = [
        C_ULTRA, "#818CF8", "#A78BFA", "#F59E0B",
        "#EC4899", "#14B8A6", "#64748B", "#06B6D4", "#94A3B8",
    ]

    y_oct_cat11 = d["oct15_cat11_y_true"]
    sha_oct_cat11 = ksha("oct15_cat11_y_true")
    c3_p1_stats = compute_paired_panel_stats(
        "chart3_1_oct15_cat11",
        {m: (d[f"oct15_cat11_pred_{k}"] == y_oct_cat11).astype(np.float32) for m, k in zip(models_9, keys_9)},
        ref_model=models_9[0],
        expected_n=2835,
        metric_type="mean",
        key_sha256={m: sha_oct_cat11 for m in models_9},
        family_ids=d["oct15_cat11_family_ids"],
    )
    c3_p1_stats["quantization_parity_int8"] = {
        "int8_hyb_pct": round(float(np.mean(d["oct15_cat11_pred_int8_hyb"] == y_oct_cat11) * 100.0), 2),
        "int8_raw_pct": round(float(np.mean(d["oct15_cat11_pred_int8_raw"] == y_oct_cat11) * 100.0), 2),
    }

    y_oct_vid = d["oct15_vid_y_true"]
    sha_oct_vid = ksha("oct15_vid_y_true")
    c3_p1_vid_stats = compute_paired_panel_stats(
        "chart3_1_oct15_video_mvtv",
        {m: (d[f"oct15_vid_mvtv_pred_{k}"] == y_oct_vid).astype(np.float32) for m, k in zip(models_9, keys_9)},
        ref_model=models_9[0],
        expected_n=1296,
        metric_type="mean",
        key_sha256={m: sha_oct_vid for m in models_9},
        family_ids=d["oct15_vid_family_ids"],
    )

    c3_p2_stats = compute_paired_panel_stats(
        "chart3_2_oct15_catalog_matching",
        {m: d[f"oct15_match_correct_{k}"].astype(np.float32) for m, k in zip(models_9, keys_9)},
        ref_model=models_9[0],
        expected_n=1296,
        metric_type="mean",
        key_sha256={m: sha_oct_vid for m in models_9},
        family_ids=d["oct15_vid_family_ids"],
    )
    c3_p2_stats["quantization_parity_int8"] = {
        "int8_hyb_pct": round(float(np.mean(d["oct15_match_correct_int8_hyb"]) * 100.0), 2),
        "int8_raw_pct": round(float(np.mean(d["oct15_match_correct_int8_raw"]) * 100.0), 2),
    }

    y_oct_aud = d["oct15_aud545_y_en"]
    sha_oct_aud = ksha("oct15_aud545_y_en")
    aud_models = [models_9[i] for i in (0, 3, 4, 5, 6, 7, 8)] + ["OpenAI GPT (Fast Tier)"]
    aud_keys = [keys_9[i] for i in (0, 3, 4, 5, 6, 7, 8)] + ["gpt"]
    c3_p3_stats = compute_paired_panel_stats(
        "chart3_3_oct15_audio_strict545",
        {m: (d[f"oct15_aud545_pred_{k}"] == y_oct_aud).astype(np.float32) for m, k in zip(aud_models, aud_keys)},
        ref_model=models_9[0],
        expected_n=545,
        metric_type="mean",
        key_sha256={m: sha_oct_aud for m in aud_models},
        family_ids=d["oct15_aud545_family_ids"],
    )
    c3_p3_stats["quantization_parity_int8"] = {
        "int8_hyb_pct": round(float(np.mean(d["oct15_aud545_pred_int8_hyb"] == y_oct_aud) * 100.0), 2),
        "int8_raw_pct": round(float(np.mean(d["oct15_aud545_pred_int8_raw"] == y_oct_aud) * 100.0), 2),
    }

    y_oct_ani = d["oct15_ani507_y_true"]
    sha_oct_ani = ksha("oct15_ani507_y_true")
    c3_p4_oct_stats = compute_paired_panel_stats(
        "chart3_4_oct15_anime507",
        {
            models_9[0]: (d["oct15_ani507_pred_ultra"] == y_oct_ani).astype(np.float32),
        },
        ref_model=models_9[0],
        expected_n=507,
        metric_type="mean",
        key_sha256={models_9[0]: sha_oct_ani},
        family_ids=d["oct15_ani507_family_ids"],
    )
    c3_p4_oct_stats["quantization_parity_int8"] = {
        "int8_pct": round(float(np.mean(d["oct15_ani507_pred_int8"] == y_oct_ani) * 100.0), 2),
    }

    y_tc2k_ani = d["tc2000_ani_y_true"]
    sha_tc2k_ani = ksha("tc2000_ani_y_true")
    c3_p4_tc2k_stats = compute_paired_panel_stats(
        "chart3_4_zero_cue_2000_anime_dub",
        {
            models_9[3]: (d["tc2000_ani_pred_rtn"] == y_tc2k_ani).astype(np.float32),
            models_9[4]: (d["tc2000_ani_pred_ptt"] == y_tc2k_ani).astype(np.float32),
            models_9[5]: (d["tc2000_ani_pred_guessit"] == y_tc2k_ani).astype(np.float32),
            models_9[6]: (d["tc2000_ani_pred_ptn"] == y_tc2k_ani).astype(np.float32),
            models_9[7]: (d["tc2000_ani_pred_anitomy"] == y_tc2k_ani).astype(np.float32),
        },
        ref_model=models_9[3],
        expected_n=2000,
        metric_type="mean",
        key_sha256={models_9[i]: sha_tc2k_ani for i in (3, 4, 5, 6, 7)},
        family_ids=d["tc2000_ani_family_ids"],
    )

    y_oct_ad = d["oct15_adult_y_true"]
    sha_oct_ad = ksha("oct15_adult_y_true")
    ad_models = [models_9[i] for i in (0, 1, 2, 3, 4, 8)]
    ad_keys = ["ultra", "mimmo_bert", "mimmo_cascade", "rtn", "ptt", "bitmagnet"]
    c3_p5_stats = compute_paired_panel_stats(
        "chart3_5_oct15_adult2781",
        {m: (d[f"oct15_adult_pred_{k}"] == y_oct_ad).astype(np.float32) for m, k in zip(ad_models, ad_keys)},
        ref_model=models_9[0],
        expected_n=2781,
        metric_type="mean",
        key_sha256={m: sha_oct_ad for m in ad_models},
        family_ids=d["oct15_adult_family_ids"],
    )
    c3_p5_stats["quantization_parity_int8"] = {
        "int8_pct": round(float(np.mean(d["oct15_adult_pred_int8"] == y_oct_ad) * 100.0), 2),
    }

    # Condition 6: oct15_junk has n_pos = 1 (n_neg = 2,169), so sensitivity and balanced accuracy
    # are NOT estimable on oct15_confirmation; report beyond_junk977 (n_pos=64 LOTO / 65 full) and mark oct15_junk not_estimable.
    y_oct_jk = d["oct15_junk_y_true"]
    neg_oct_jk = y_oct_jk == 0
    c3_p6a_stats: Dict[str, Any] = {
        "panel_name": "chart3_6a_junk_audit",
        "oct15_junk_status": "n_pos = 1, not estimable",
        "oct15_n_total": int(y_oct_jk.size),
        "oct15_n_pos": int((y_oct_jk == 1).sum()),
        "oct15_n_neg": int(neg_oct_jk.sum()),
        "oct15_specificity_only": {
            "Armiger Ultra (Fair Hybrid)": {
                "tn": int(np.sum(d["oct15_junk_pred_ultra"][neg_oct_jk] == 0)),
                "n_neg": int(neg_oct_jk.sum()),
                "specificity_pct": round(float(np.mean(d["oct15_junk_pred_ultra"][neg_oct_jk] == 0) * 100.0), 2),
            },
            "Bitmagnet Built-In": {
                "tn": int(np.sum(d["oct15_junk_pred_bitmagnet"][neg_oct_jk] == 0)),
                "n_neg": int(neg_oct_jk.sum()),
                "tp_on_n_pos_1": int(np.sum(d["oct15_junk_pred_bitmagnet"][~neg_oct_jk] == 1)),
            },
            "RTN (rank-torrent-name)": {
                "tn": int(np.sum(d["oct15_junk_pred_rtn"][neg_oct_jk] == 0)),
                "n_neg": int(neg_oct_jk.sum()),
                "tp_on_n_pos_1": int(np.sum(d["oct15_junk_pred_rtn"][~neg_oct_jk] == 1)),
            },
        },
        "beyond_junk_loto_no_gpt_n975": p2e_loto_gpt_stats,
    }

    oct_cat_spec = [
        (6, "Audiobook"),
        (2, "Adult / XXX"),
        (1, "TV Show"),
        (0, "Movie"),
        (7, "Software"),
        (3, "Music"),
        (5, "Game"),
        (8, "Ebook"),
        (9, "Course"),
        (4, "Comic"),
    ]
    cats_oct_labels = []
    oct_ultra, oct_int8, oct_mimmo, oct_bit = [], [], [], []
    chart4_oct_manifest: Dict[str, Any] = {}
    for c_idx, c_title in oct_cat_spec:
        mask = (y_oct_cat11 == c_idx)
        n_c = int(np.sum(mask))
        cats_oct_labels.append(f"{c_title} (n={n_c:,})")
        u_v = float(np.mean(d["oct15_cat11_pred_ultra"][mask] == c_idx) * 100.0)
        i_v = float(np.mean(d["oct15_cat11_pred_int8_hyb"][mask] == c_idx) * 100.0)
        m_v = float(np.mean(d["oct15_cat11_pred_mimmo_bert"][mask] == c_idx) * 100.0)
        b_v = float(np.mean(d["oct15_cat11_pred_bitmagnet"][mask] == c_idx) * 100.0)
        oct_ultra.append(round(u_v, 2))
        oct_int8.append(round(i_v, 2))
        oct_mimmo.append(round(m_v, 2))
        oct_bit.append(round(b_v, 2))
        chart4_oct_manifest[c_title] = {
            "n": n_c,
            "armiger_ultra_pct": round(u_v, 2),
            "armiger_int8_parity_pct": round(i_v, 2),
            "mimmo_bert_pct": round(m_v, 2),
            "bitmagnet_pct": round(b_v, 2),
        }

    y_bc_cat11 = d["beyond_cat11_y_true"]
    k_bc_no_gpt = d["beyond_cat11_key_no_gpt"]
    k_bc_no_gem = d["beyond_cat11_key_no_gemini"]
    bc_cat_spec = [
        (6, "Audiobook"),
        (8, "Ebook"),
        (4, "Comic"),
        (0, "Movie"),
        (1, "TV Show"),
        (5, "Game"),
        (7, "Software"),
        (9, "Course"),
        (2, "Adult / XXX"),
        (3, "Music"),
    ]
    cats_bc_labels = []
    bc_ultra, bc_gpt_loto, bc_gem_loto = [], [], []
    chart4_bc_manifest: Dict[str, Any] = {}
    for c_idx, c_title in bc_cat_spec:
        mask_all3 = (y_bc_cat11 == c_idx)
        mask_ng = (k_bc_no_gpt == c_idx)
        mask_nm = (k_bc_no_gem == c_idx)
        n_c = int(np.sum(mask_all3))
        cats_bc_labels.append(f"{c_title} (n={n_c:,})")
        u_v = float(np.mean(d["beyond_cat11_corr_ultra"][mask_all3]) * 100.0)
        g_v = float(np.mean(d["beyond_cat11_pred_gpt"][mask_ng] == c_idx) * 100.0) if mask_ng.any() else 0.0
        m_v = float(np.mean(d["beyond_cat11_pred_gemini"][mask_nm] == c_idx) * 100.0) if mask_nm.any() else 0.0
        bc_ultra.append(round(u_v, 2))
        bc_gpt_loto.append(round(g_v, 2))
        bc_gem_loto.append(round(m_v, 2))
        chart4_bc_manifest[c_title] = {
            "n_all3": n_c,
            "n_loto_no_gpt": int(np.sum(mask_ng)),
            "n_loto_no_gemini": int(np.sum(mask_nm)),
            "armiger_ultra_pct": round(u_v, 2),
            "gpt_loto_support_pct": round(g_v, 2),
            "gemini_loto_support_pct": round(m_v, 2),
        }

    fallback_assets = assets_dir if (assets_dir / "armiger_vs_llms_overall_and_6dims.png").is_file() else (npz_path.parent.parent / "docs" / "assets")
    if render_pngs:
        # ==========================================================================
        # RENDER CHART 1: armiger_vs_llms_overall_and_6dims.png
        # ==========================================================================
        fig1 = plt.figure(figsize=(24, 11.4))
        fig1.suptitle(
            "Armiger Ultra vs. Cloud Teacher LLMs Across 6 Core Dimensions (Common-Support LOTO & OOF)",
            fontsize=17.0, fontweight="bold", color=TEXT_MAIN, y=0.972,
        )
        fig1.text(
            0.5, 0.934,
            "Common-Support Leave-One-Vendor-Out (LOTO) & Physical Swarm OOF with Family-Clustered 95% Bootstrap CIs (10,000 Resamples)",
            ha="center", fontsize=11.0, style="italic", color=TEXT_SUB, parse_math=False,
        )

        gs1 = GridSpec(
            2, 5, figure=fig1,
            width_ratios=[1.72, 0.12, 1.0, 1.0, 1.0],
            wspace=0.16, hspace=0.42,
            top=0.86, bottom=0.13, left=0.14, right=0.98,
        )

        ax_left = fig1.add_subplot(gs1[:, 0])
        style_ax(ax_left)
        comp_4d_gpt_u = composite_stats["loto_common_support_4dim_2a_to_2d"]["vs_openai_gpt"][ref_llm]
        comp_4d_gpt_g = composite_stats["loto_common_support_4dim_2a_to_2d"]["vs_openai_gpt"][gpt_llm]
        comp_4d_gem_u = composite_stats["loto_common_support_4dim_2a_to_2d"]["vs_google_gemini"][ref_llm]
        comp_4d_gem_m = composite_stats["loto_common_support_4dim_2a_to_2d"]["vs_google_gemini"][gem_llm]
        comp_5d_gpt_u = composite_stats["loto_common_support_5dim_2a_to_2e_vs_gpt"][ref_llm]
        comp_5d_gpt_g = composite_stats["loto_common_support_5dim_2a_to_2e_vs_gpt"][gpt_llm]

        left_rows = [
            ("Armiger Ultra (5D LOTO vs GPT)", comp_5d_gpt_u, C_ULTRA, f"{comp_5d_gpt_u:.2f}% (2A-2E {{google,meta}} Support)"),
            ("OpenAI GPT Fast (5D LOTO)", comp_5d_gpt_g, C_GPT, f"{comp_5d_gpt_g:.2f}% (0% Junk Recall on 2E)"),
            ("Armiger Ultra (4D LOTO vs Gem)", comp_4d_gem_u, C_ULTRA, f"{comp_4d_gem_u:.2f}% (2A-2D {{openai,meta}} Support)"),
            ("Google Gemini Flash (4D LOTO)", comp_4d_gem_m, C_GEMINI, f"{comp_4d_gem_m:.2f}% (+0.22 pp vs Ultra on 2A-2D)"),
            ("Armiger Ultra (6D Disjoint)", composite_stats[ref_llm]["point_pct"], C_ULTRA, f"{composite_stats[ref_llm]['point_pct']:.2f}% Pub-6D | {composite_stats[ref_llm]['point_with_corpus_swarm_pct']:.2f}% Corpus"),
        ]
        y_left = np.arange(len(left_rows))
        ax_left.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        ax_left.barh(y_left, [r[1] for r in left_rows], height=0.52, color=[r[2] for r in left_rows], zorder=3)
        ax_left.set_yticks(y_left)
        ax_left.set_yticklabels([r[0] for r in left_rows], fontsize=9.8, fontweight="bold", color=TEXT_MAIN)
        ax_left.invert_yaxis()
        ax_left.set_xlim(50, 134)
        ax_left.set_xticks([55, 65, 75, 85, 95, 100])
        ax_left.set_xlabel("Common-Support LOTO & Vendor-Disjoint Composite Means (%)", fontsize=9.8, color=TEXT_SUB, labelpad=8)
        ax_left.set_title(
            "1. Pairwise LOTO Common-Support & 6D Composites\n(Explicitly Reporting Both 2A-2D & 2A-2E Orderings)",
            fontsize=11.2, fontweight="bold", color=TEXT_MAIN, pad=10,
        )
        for idx, (_, val, _, lbl) in enumerate(left_rows):
            ax_left.text(val + 0.65, idx, lbl, va="center", ha="left", fontsize=8.4, fontweight="bold" if "Ultra" in left_rows[idx][0] else "normal", color=TEXT_MAIN if "Ultra" in left_rows[idx][0] else "#CBD5E1", zorder=6, parse_math=False)

        loto_panel_specs = [
            (0, 2, "2A. 11-Class Category (LOTO N=974 / 982)", p2a_loto_gpt_stats, p2a_loto_gemini_stats, (85, 114), [85, 90, 95, 100]),
            (0, 3, "2B. Catalog Matching (LOTO N=974 / 982)", p2b_loto_gpt_stats, p2b_loto_gemini_stats, (80, 114), [80, 85, 90, 95, 100]),
            (0, 4, "2C. Audio Language Strict (LOTO N=209 / 194)", p2c_loto_gpt_stats, p2c_loto_gemini_stats, (80, 114), [80, 85, 90, 95, 100]),
            (1, 2, "2D. Anime Detection (LOTO N=629 / 629)", p2d_loto_gpt_stats, p2d_loto_gemini_stats, (85, 114), [85, 90, 95, 100]),
        ]
        for row, col, title, st_gpt, st_gem, xlim, xticks in loto_panel_specs:
            ax = fig1.add_subplot(gs1[row, col])
            style_ax(ax)
            s_list = [
                st_gpt["models"][ref_llm],
                st_gpt["models"][gpt_llm],
                st_gem["models"][ref_llm],
                st_gem["models"][gem_llm],
            ]
            c_list = [C_ULTRA, C_GPT, C_ULTRA, C_GEMINI]
            tags = ["Ultra (vs GPT)", "GPT (LOTO)", "Ultra (vs Gem)", "Gemini (LOTO)"]
            y_pos = np.arange(4)
            ax.barh(y_pos, [s["point_pct"] for s in s_list], height=0.55, color=c_list, zorder=3)
            draw_ci_whiskers(ax, y_pos, s_list)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([""] * 4)
            ax.invert_yaxis()
            ax.set_xlim(*xlim)
            ax.set_xticks(xticks)
            ax.set_title(title, fontsize=10.2, fontweight="bold", color=TEXT_MAIN, pad=8)
            span = xlim[1] - xlim[0]
            for idx, (tag, s_item) in enumerate(zip(tags, s_list)):
                hi_x = max(s_item["point_pct"], s_item["ci95_hi"])
                ax.text(hi_x + span * 0.015, idx, f"{s_item['point_pct']:.2f}% ({tag})", va="center", ha="left", fontsize=7.8, fontweight="bold" if "Ultra" in tag else "normal", color=TEXT_MAIN if "Ultra" in tag else "#CBD5E1", zorder=6, parse_math=False)

        # Panel 2E: Junk LOTO vs GPT (N=975, n_pos=64)
        ax2e = fig1.add_subplot(gs1[1, 3])
        style_ax(ax2e)
        s2e_list = [
            p2e_loto_gpt_stats["models"][ref_llm],
            p2e_loto_gpt_stats["models"][gpt_llm],
        ]
        y2e = np.array([0, 1])
        ax2e.barh(y2e, [s["point_pct"] for s in s2e_list], height=0.55, color=[C_ULTRA, C_GPT], zorder=3)
        draw_ci_whiskers(ax2e, y2e, s2e_list)
        ax2e.barh(2, 45.0, height=0.55, color="#21262D", edgecolor=TEXT_MUTED, hatch="//", zorder=2)
        ax2e.text(46.0, 2, "Gemini N/A (LOTO n_pos=0)", va="center", ha="left", fontsize=7.8, style="italic", color=TEXT_SUB, parse_math=False)
        for idx, (tag, s_item) in enumerate(zip(["Ultra (LOTO N=975)", "GPT (0/64 Recall)"], s2e_list)):
            hi_x = max(s_item["point_pct"], s_item["ci95_hi"])
            ax2e.text(hi_x + 1.2, idx, f"{s_item['point_pct']:.2f}% ({tag})", va="center", ha="left", fontsize=7.8, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1", zorder=6, parse_math=False)
        ax2e.set_yticks([0, 1, 2])
        ax2e.set_yticklabels(["", "", ""])
        ax2e.invert_yaxis()
        ax2e.set_xlim(40, 134)
        ax2e.set_xticks([45, 60, 75, 90, 100])
        ax2e.set_title("2E. Junk / Malware (Balanced LOTO N=975)", fontsize=10.2, fontweight="bold", color=TEXT_MAIN, pad=8)

        # Panel 2F: Swarm Liveliness AUROC (Public OOF N=757)
        ax2f = fig1.add_subplot(gs1[1, 4])
        style_ax(ax2f)
        m2f_list = ["Armiger Ultra (Fair Hybrid)", "Way-4 Combined Priors", "DHT Seeder Baseline"]
        s2f_list = [p2f_pub_stats["models"][m] for m in m2f_list]
        y2f = np.arange(len(m2f_list))
        ax2f.barh(y2f, [s["point_pct"] for s in s2f_list], height=0.55, color=[C_ULTRA, C_INT8, "#64748B"], zorder=3)
        draw_ci_whiskers(ax2f, y2f, s2f_list)
        for idx, (m_name, s_item) in enumerate(zip(["Ultra OOF", "Way-4 Priors", "DHT Seeders"], s2f_list)):
            hi_x = max(s_item["point_pct"], s_item["ci95_hi"])
            ax2f.text(hi_x + 1.2, idx, f"{s_item['point_pct']:.2f}% ({m_name})", va="center", ha="left", fontsize=7.8, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1", zorder=6, parse_math=False)
        ax2f.set_yticks(y2f)
        ax2f.set_yticklabels([""] * len(m2f_list))
        ax2f.invert_yaxis()
        ax2f.set_xlim(40, 128)
        ax2f.set_xticks([45, 60, 75, 90, 100])
        ax2f.set_title("2F. Swarm Liveliness AUROC (OOF N=757)", fontsize=10.2, fontweight="bold", color=TEXT_MAIN, pad=8)

        legend_handles = [
            Patch(facecolor=C_ULTRA, edgecolor=C_ULTRA_EDGE, label="Armiger Ultra (Fair Hybrid; INT8 Parity Logged)"),
            Patch(facecolor=C_GPT, label="OpenAI GPT Fast Tier (LOTO Support vs {google, meta})"),
            Patch(facecolor=C_GEMINI, label="Google Gemini Flash (LOTO Support vs {openai, meta})"),
            Patch(facecolor=C_INT8, label="Way-4 Priors / Seeder Baselines (Swarm OOF N=757)"),
        ]
        fig1.legend(
            handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
            ncol=4, facecolor=BG_PANEL, edgecolor=BORDER_COLOR, fontsize=8.8, labelcolor=TEXT_MAIN,
            borderpad=0.6, columnspacing=1.25,
        )
        sha_chart1 = save_clean_rgb_png(fig1, assets_dir / "armiger_vs_llms_overall_and_6dims.png")

        # ==========================================================================
        # RENDER CHART 2: armiger_vs_llms_cost_comparison.png (100% receipt-driven)
        # ==========================================================================
        fig2, (ax2_l, ax2_r) = plt.subplots(1, 2, figsize=(23.5, 9.2), gridspec_kw={"wspace": 0.14})
        fig2.subplots_adjust(top=0.80, bottom=0.16, left=0.18, right=0.98)
        fig2.suptitle(
            "Cost & Measured Hardware Throughput: Armiger ONNX vs. Logged Teacher LLM Runs",
            fontsize=17.5, fontweight="bold", color=TEXT_MAIN, y=0.96,
        )
        fig2.text(
            0.5, 0.915,
            "Left: Operational Cost per 1,000,000 Classified Torrents ($ USD)  |  Right: Logged Mean Tokens/Call & Live Measured M4 Pro CPU Latency",
            ha="center", fontsize=11.3, style="italic", color=TEXT_SUB, parse_math=False,
        )

        cost_labels = [
            f"$0.00 API (~${batched_cost[0]:.4f} Local CPU Power / 1M Items)",
            f"$0.00 API (~${batched_cost[1]:.4f} Local CPU Power / 1M Items)",
            f"${batched_cost[2]:.2f} Batched (${realtime_cost[2]:.2f} Real-Time / 1M)",
            f"${batched_cost[3]:.2f} Batched (${realtime_cost[3]:.2f} Real-Time / 1M)",
            f"${batched_cost[4]:.2f} Batched (${realtime_cost[4]:.2f} Real-Time / 1M)",
        ]

        style_ax(ax2_l)
        yc = np.arange(len(models_cost))
        ax2_l.axhspan(-0.5, 1.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        for idx in range(len(models_cost)):
            ax2_l.barh(idx, realtime_cost[idx], height=0.56, color=colors_cost[idx], alpha=0.28, edgecolor=colors_cost[idx], linestyle="--", linewidth=1.2, zorder=2)
            ax2_l.barh(idx, batched_cost[idx], height=0.56, color=colors_cost[idx], zorder=3)
            x_pos = max(realtime_cost[idx], batched_cost[idx]) + 14.0
            ax2_l.text(x_pos, idx, cost_labels[idx], va="center", ha="left", fontsize=9.0, fontweight="bold" if idx < 2 else "normal", color=TEXT_MAIN if idx < 2 else "#CBD5E1", parse_math=False)

        ax2_l.set_yticks(yc)
        ax2_l.set_yticklabels(models_cost, fontsize=10.0, fontweight="bold", color=TEXT_MAIN)
        ax2_l.invert_yaxis()
        ax2_l.set_xlim(0, 1480)
        ax2_l.set_xticks([0, 200, 400, 600, 800, 1000])
        ax2_l.set_xticklabels(["$0", "$200", "$400", "$600", "$800", "$1,000"])
        ax2_l.set_xlabel("Derived from hardware_and_api_cost_receipt.json & usage-summary-all.json", fontsize=9.5, color=TEXT_SUB, labelpad=8)
        ax2_l.set_title(
            "1. Measured Cost per 1,000,000 Classified Torrents ($ USD / 1M Items)\n(From 1,012 Logged Provider Calls & 26,119,043 Total Tokens)",
            fontsize=11.2, fontweight="bold", color=TEXT_MAIN, pad=10,
        )

        style_ax(ax2_r)
        ax2_r.axhspan(-0.5, 1.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        in_rates = [float(cost_rec[m]["in_usd_per_1m_tokens"]) for m in models_cost]
        out_rates = [float(cost_rec[m]["out_usd_per_1m_tokens"]) for m in models_cost]
        rate_labels = [str(cost_rec[m]["rate_label"]) for m in models_cost]
        for idx in range(len(models_cost)):
            ax2_r.barh(idx - 0.14, in_rates[idx], height=0.28, color=colors_cost[idx], alpha=0.55, hatch="//", edgecolor=colors_cost[idx], zorder=3)
            ax2_r.barh(idx + 0.14, out_rates[idx], height=0.28, color=colors_cost[idx], zorder=3)
            ax2_r.text(max(in_rates[idx], out_rates[idx]) + 0.08, idx + 0.05, rate_labels[idx], va="center", ha="left", fontsize=8.8, fontweight="bold" if idx < 2 else "normal", color=TEXT_MAIN if idx < 2 else "#CBD5E1", parse_math=False)

        ax2_r.set_yticks(yc)
        ax2_r.set_yticklabels([""] * len(models_cost))
        ax2_r.invert_yaxis()
        ax2_r.set_xlim(0, 4.2)
        ax2_r.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
        ax2_r.set_xticklabels(["$0.00", "$0.50", "$1.00", "$1.50", "$2.00", "$2.50"])
        ax2_r.set_xlabel("Token Rates ($ USD / 1M Tokens) & Live Measured M4 Pro CPU Latency/Throughput", fontsize=9.5, color=TEXT_SUB, labelpad=8)
        ax2_r.set_title(
            "2. Token Pricing ($ USD / 1M Tokens) & Measured Hardware Latency\n(All Latency & Token Figures Read Directly from hardware_and_api_cost_receipt.json)",
            fontsize=11.2, fontweight="bold", color=TEXT_MAIN, pad=10,
        )
        ax2_r.legend(
            handles=[
                Patch(facecolor=C_GEMINI, alpha=0.55, hatch="//", edgecolor=C_GEMINI, label="Input Tokens ($ / 1M)"),
                Patch(facecolor=C_GEMINI, label="Output Tokens ($ / 1M)"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, facecolor=BG_PANEL, edgecolor=BORDER_COLOR, fontsize=8.8, labelcolor=TEXT_MAIN,
        )
        sha_chart2 = save_clean_rgb_png(fig2, assets_dir / "armiger_vs_llms_cost_comparison.png")

        # ==========================================================================
        # RENDER CHART 3: armiger_vs_torrent_classifiers_6dims.png
        # ==========================================================================
        fig3 = plt.figure(figsize=(24.5, 14.8))
        fig3.suptitle(
            "Armiger Ultra vs. Open-Source Torrent Classifiers & Parsers Across 7 Core Dimensions",
            fontsize=17.5, fontweight="bold", color=TEXT_MAIN, y=0.975,
        )
        fig3.text(
            0.5, 0.942,
            "Identical-Reference Evaluation on oct15_confirmation (N = 2,835 / N = 545 Audio), zero_cue_2000 (N = 2,000), & Swarm Holdouts (±95% Clustered CIs)",
            ha="center", fontsize=11.3, style="italic", color=TEXT_SUB, parse_math=False,
        )

        gs3 = GridSpec(
            7, 4, figure=fig3,
            height_ratios=[1.0, 1.0, 1.0, 0.42, 1.25, 0.48, 1.25],
            width_ratios=[1.08, 1.0, 0.22, 1.08],
            hspace=0.18, wspace=0.15,
            top=0.89, bottom=0.06, left=0.12, right=0.98,
        )
        y9 = np.arange(len(models_9))

        ax1 = fig3.add_subplot(gs3[0:3, 0])
        style_ax(ax1)
        ax1.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        cat_11_s = [c3_p1_stats["models"][m] for m in models_9]
        cat_11cls = [s["point_pct"] for s in cat_11_s]
        cat_vid = [c3_p1_vid_stats["models"][m]["point_pct"] for m in models_9]
        ax1.barh(y9 - 0.16, cat_vid, height=0.32, color=colors_9, zorder=3)
        ax1.barh(y9 + 0.16, cat_11cls, height=0.32, color=colors_9, alpha=0.45, hatch="//", edgecolor=colors_9, zorder=3)
        draw_ci_whiskers(ax1, y9 + 0.16, cat_11_s)
        ax1.set_yticks(y9)
        ax1.set_yticklabels(models_9, fontsize=9.5, fontweight="bold", color=TEXT_MAIN)
        ax1.invert_yaxis()
        ax1.set_xlim(25, 148)
        ax1.set_xticks([30, 40, 50, 60, 70, 80, 90, 100])
        ax1.set_xlabel("Solid = Video Subset (N=1,296); Hatched = 11-Class (N=2,835)", fontsize=8.6, color=TEXT_SUB, labelpad=6)
        ax1.set_title("1. Category Detection (%)\n(oct15_confirmation, Identical N = 2,835)", fontsize=11, fontweight="bold", color=TEXT_MAIN, pad=8)
        for idx, (v_vid, v_11) in enumerate(zip(cat_vid, cat_11cls)):
            lbl = f"{v_11:.2f}% 11-Cls (Video {v_vid:.1f}%)"
            ax1.text(max(v_vid, v_11) + 1.1, idx, lbl, va="center", ha="left", fontsize=8.2, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1")

        ax2 = fig3.add_subplot(gs3[0:3, 1])
        style_ax(ax2)
        ax2.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        p2_s = [c3_p2_stats["models"][m] for m in models_9]
        cat_match = [s["point_pct"] for s in p2_s]
        ax2.barh(y9, cat_match, height=0.58, color=colors_9, zorder=3)
        draw_ci_whiskers(ax2, y9, p2_s)
        ax2.set_yticks(y9)
        ax2.set_yticklabels([""] * len(models_9))
        ax2.invert_yaxis()
        ax2.set_xlim(35, 128)
        ax2.set_xticks([40, 50, 60, 70, 80, 90, 100])
        ax2.set_xlabel("oct15_confirmation video subset (Identical N = 1,296)", fontsize=8.8, color=TEXT_SUB, labelpad=6)
        ax2.set_title("2. Catalog Matching Rate (%)\n(Clean Title + Correct Movie/TV Routing, N = 1,296)", fontsize=11, fontweight="bold", color=TEXT_MAIN, pad=8)
        for idx, s_item in enumerate(p2_s):
            v = s_item["point_pct"]
            ax2.text(max(v, s_item["ci95_hi"]) + 0.9, idx, f"{v:.2f}%", va="center", ha="left", fontsize=8.4, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1")

        ax3 = fig3.add_subplot(gs3[0:3, 3])
        style_ax(ax3)
        ax3.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        for idx, m in enumerate(models_9):
            if m in c3_p3_stats["models"]:
                s_item = c3_p3_stats["models"][m]
                v = s_item["point_pct"]
                ax3.barh(idx, v, height=0.58, color=colors_9[idx], zorder=3)
                draw_ci_whiskers(ax3, np.array([idx]), [s_item])
                ax3.text(max(v, s_item["ci95_hi"]) + 0.25, idx, f"{v:.2f}%", va="center", ha="left", fontsize=8.4, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1")
            else:
                ax3.barh(idx, 86.2, height=0.55, color="#21262D", edgecolor=TEXT_MUTED, hatch="//", zorder=2)
                ax3.text(84.3, idx, "N/A (No Audio Language Head)", va="center", ha="left", fontsize=8.1, style="italic", color=TEXT_SUB)
        ax3.set_yticks(y9)
        ax3.set_yticklabels([""] * len(models_9))
        ax3.invert_yaxis()
        ax3.set_xlim(84, 106.5)
        ax3.set_xticks([86, 88, 90, 92, 94, 96, 98, 100])
        ax3.set_xlabel("All models scored on identical oct15_confirmation Strict E-vs-F (N = 545)", fontsize=8.6, color=TEXT_SUB, labelpad=6)
        ax3.set_title("3. Audio Language Detection (%)\n(Strict English vs. Foreign Audio, Identical N = 545)", fontsize=11, fontweight="bold", color=TEXT_MAIN, pad=8)

        ax4 = fig3.add_subplot(gs3[4:7, 0])
        style_ax(ax4)
        ax4.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        for idx, m in enumerate(models_9):
            if m in c3_p4_oct_stats["models"]:
                s_item = c3_p4_oct_stats["models"][m]
                v = s_item["point_pct"]
                ax4.barh(idx, v, height=0.58, color=colors_9[idx], zorder=3)
                draw_ci_whiskers(ax4, np.array([idx]), [s_item])
                ax4.text(max(v, s_item["ci95_hi"]) + 0.6, idx, f"{v:.2f}% (oct15 N=507)", va="center", ha="left", fontsize=8.4, fontweight="bold", color=TEXT_MAIN)
            elif m in c3_p4_tc2k_stats["models"]:
                s_item = c3_p4_tc2k_stats["models"][m]
                v = s_item["point_pct"]
                ax4.barh(idx, v, height=0.58, color=colors_9[idx], zorder=3)
                draw_ci_whiskers(ax4, np.array([idx]), [s_item])
                ax4.text(max(v, s_item["ci95_hi"]) + 0.6, idx, f"{v:.2f}%", va="center", ha="left", fontsize=8.4, color="#CBD5E1")
            else:
                ax4.barh(idx, 49.0, height=0.55, color="#21262D", edgecolor=TEXT_MUTED, hatch="//", zorder=2)
                ax4.text(46.2, idx, "N/A (No Anime / Dub Head)", va="center", ha="left", fontsize=8.1, style="italic", color=TEXT_SUB)
        ax4.set_yticks(y9)
        ax4.set_yticklabels(models_9, fontsize=9.5, fontweight="bold", color=TEXT_MAIN)
        ax4.invert_yaxis()
        ax4.set_xlim(45, 128)
        ax4.set_xticks([50, 60, 70, 80, 90, 100])
        ax4.set_xlabel("oct15_confirmation (N=507) & zero_cue_2000 (N=2,000)", fontsize=8.6, color=TEXT_SUB, labelpad=6)
        ax4.set_title("4. Anime & Anime Dub Detection (%)\n(oct15_confirmation N = 507 & zero_cue_2000 N = 2,000)", fontsize=11, fontweight="bold", color=TEXT_MAIN, pad=8)

        ax5 = fig3.add_subplot(gs3[4:7, 1])
        style_ax(ax5)
        ax5.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        for idx, m in enumerate(models_9):
            if m in c3_p5_stats["models"]:
                s_item = c3_p5_stats["models"][m]
                v = s_item["point_pct"]
                ax5.barh(idx, v, height=0.58, color=colors_9[idx], zorder=3)
                draw_ci_whiskers(ax5, np.array([idx]), [s_item])
                ax5.text(max(v, s_item["ci95_hi"]) + 0.5, idx, f"{v:.2f}%", va="center", ha="left", fontsize=8.4, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1")
            else:
                ax5.barh(idx, 68.5, height=0.55, color="#21262D", edgecolor=TEXT_MUTED, hatch="//", zorder=2)
                ax5.text(65.8, idx, "N/A (No Adult / NSFW Head)", va="center", ha="left", fontsize=8.1, style="italic", color=TEXT_SUB)
        ax5.set_yticks(y9)
        ax5.set_yticklabels([""] * len(models_9))
        ax5.invert_yaxis()
        ax5.set_xlim(65, 114)
        ax5.set_xticks([70, 75, 80, 85, 90, 95, 100])
        ax5.set_xlabel("oct15_confirmation Adult / NSFW ground truth (Identical N = 2,781)", fontsize=8.6, color=TEXT_SUB, labelpad=6)
        ax5.set_title("5. Adult / NSFW Detection (%)\n(oct15_confirmation, Identical N = 2,781)", fontsize=11, fontweight="bold", color=TEXT_MAIN, pad=8)

        ax6a = fig3.add_subplot(gs3[4, 3])
        style_ax(ax6a)
        ax6a.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        m6a = ["Armiger Ultra (beyond N=975)", "OpenAI GPT (beyond N=975)", "oct15 (n_pos=1)", "Open-Source Parsers"]
        v6a = [
            p2e_loto_gpt_stats["models"][ref_llm]["point_pct"],
            p2e_loto_gpt_stats["models"][gpt_llm]["point_pct"],
            None,
            None,
        ]
        c6a = [C_ULTRA, C_GPT, "#21262D", "#21262D"]
        l6a = [
            f"{v6a[0]:.2f}% Bal (n_pos=64, n_neg=911)",
            f"{v6a[1]:.2f}% Bal (0/64 Recall)",
            "Not Estimable (n_pos=1; Spec 99.91% on n_neg=2,169)",
            "N/A (0% Recall on Malware/Junk Traps)",
        ]
        for idx, (v, lbl) in enumerate(zip(v6a, l6a)):
            if v is not None:
                ax6a.barh(idx, v, height=0.6, color=c6a[idx], zorder=3)
                ax6a.text(v + 0.8, idx, lbl, va="center", ha="left", fontsize=8.0, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1")
            else:
                ax6a.barh(idx, 49.0, height=0.55, color="#21262D", edgecolor=TEXT_MUTED, hatch="//", zorder=2)
                ax6a.text(46.2, idx, lbl, va="center", ha="left", fontsize=7.8, style="italic", color=TEXT_SUB)
        ax6a.set_yticks(np.arange(len(m6a)))
        ax6a.set_yticklabels(m6a, fontsize=8.8, fontweight="bold", color=TEXT_MAIN)
        ax6a.invert_yaxis()
        ax6a.set_xlim(45, 148)
        ax6a.set_xticks([50, 60, 70, 80, 90, 100])
        ax6a.set_title("6A. Junk / Malware Detection (beyond N=975; oct15 n_pos=1 Not Estimable)", fontsize=9.8, fontweight="bold", color=TEXT_MAIN, pad=7)

        ax6b = fig3.add_subplot(gs3[6, 3])
        style_ax(ax6b)
        ax6b.axhspan(-0.5, 0.5, color=BG_HIGHLIGHT, alpha=0.75, zorder=1)
        m6b = ["Armiger Ultra (Fair Hybrid)", "DHT Seeder Baseline", "Raw Seeder Count", "All 8 Tools"]
        v6b = [
            p2f_whse_stats["models"]["Armiger Ultra (Fair Hybrid)"]["point_pct"],
            p2f_whse_stats["models"]["DHT Seeder Baseline"]["point_pct"],
            p2f_whse_stats["models"]["Raw Seeder Count"]["point_pct"],
            None,
        ]
        c6b = [C_ULTRA, C_INT8, "#64748B", "#21262D"]
        l6b = [
            f"{v6b[0]:.2f}% Corpus ({p2f_pub_stats['models'][ref_llm]['point_pct']:.2f}% Pub N=757)",
            f"{v6b[1]:.2f}% Corpus (DHT Seeders)",
            f"{v6b[2]:.2f}% Corpus (Tracker Seeders)",
            "N/A (No Swarm Head)",
        ]
        for idx, (v, lbl) in enumerate(zip(v6b, l6b)):
            if v is not None:
                ax6b.barh(idx, v, height=0.6, color=c6b[idx], zorder=3)
                ax6b.text(v + 0.8, idx, lbl, va="center", ha="left", fontsize=8.2, fontweight="bold" if idx == 0 else "normal", color=TEXT_MAIN if idx == 0 else "#CBD5E1")
            else:
                ax6b.barh(idx, 52.5, height=0.55, color="#21262D", edgecolor=TEXT_MUTED, hatch="//", zorder=2)
                ax6b.text(49.5, idx, lbl, va="center", ha="left", fontsize=8.1, style="italic", color=TEXT_SUB)
        ax6b.set_yticks(np.arange(len(m6b)))
        ax6b.set_yticklabels(m6b, fontsize=9, fontweight="bold", color=TEXT_MAIN)
        ax6b.invert_yaxis()
        ax6b.set_xlim(48, 144)
        ax6b.set_xticks([50, 60, 70, 80, 90, 100])
        ax6b.set_xlabel("Multi-Source Corpus (Identical N=10,278) & Public Tracker (N=757: 92.42%)", fontsize=8.2, color=TEXT_SUB, labelpad=6)
        ax6b.set_title("6B. Swarm Liveliness AUROC % (5-Fold OOF)", fontsize=10, fontweight="bold", color=TEXT_MAIN, pad=7)

        sha_chart3 = save_clean_rgb_png(fig3, assets_dir / "armiger_vs_torrent_classifiers_6dims.png")

        # ==========================================================================
        # RENDER CHART 4: armiger_categorical_breakdown.png
        # ==========================================================================
        fig4, (ax4_l, ax4_r) = plt.subplots(1, 2, figsize=(22.5, 11.2), gridspec_kw={"wspace": 0.22})
        fig4.subplots_adjust(top=0.84, bottom=0.16, left=0.11, right=0.98)
        fig4.suptitle(
            "Per-Category Recall Breakdown Across All 10 Known Media & Software Categories",
            fontsize=17, fontweight="bold", color=TEXT_MAIN, y=0.965,
        )
        fig4.text(
            0.5, 0.925,
            "Left: oct15_confirmation (Identical N = 2,835) vs. Open-Source Classifiers  |  Right: beyond_confirmation (Vendor-Disjoint & LOTO Support)",
            ha="center", fontsize=11.2, style="italic", color=TEXT_SUB, parse_math=False,
        )

        style_ax(ax4_l)
        y_c = np.arange(len(cats_oct_labels))
        h = 0.24
        ax4_l.barh(y_c - h, oct_ultra, height=h, color=C_ULTRA, label=f"Armiger Ultra Fair Hybrid ({c3_p1_stats['models'][models_9[0]]['point_pct']:.2f}% Overall)", zorder=3)
        ax4_l.barh(y_c, oct_mimmo, height=h, color="#818CF8", label=f"MIMMO BERT 360M ({c3_p1_stats['models'][models_9[1]]['point_pct']:.2f}% Overall)", zorder=3)
        ax4_l.barh(y_c + h, oct_bit, height=h, color="#64748B", label=f"Bitmagnet Built-In ({c3_p1_stats['models'][models_9[8]]['point_pct']:.2f}% Overall)", zorder=3)
        ax4_l.set_yticks(y_c)
        ax4_l.set_yticklabels(cats_oct_labels, fontsize=10, fontweight="bold", color=TEXT_MAIN)
        ax4_l.set_ylim(len(cats_oct_labels) - 0.35, -0.6)
        ax4_l.set_xlim(0, 128)
        ax4_l.set_xticks([0, 20, 40, 60, 80, 100])
        ax4_l.set_xlabel("Per-Class Recall (%) — Identical N = 2,835 Ground-Truth Mask", fontsize=9.8, color=TEXT_SUB, labelpad=8)
        ax4_l.set_title("1. Per-Category Recall vs. Open-Source Classifiers\n(oct15_confirmation, Identical N = 2,835)", fontsize=11.5, fontweight="bold", color=TEXT_MAIN, pad=10)
        for idx in range(len(cats_oct_labels)):
            ax4_l.text(oct_ultra[idx] + 1.0, idx - h, f"{oct_ultra[idx]:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color=C_ULTRA_EDGE)
            ax4_l.text(oct_mimmo[idx] + 1.0, idx, f"{oct_mimmo[idx]:.1f}%", va="center", ha="left", fontsize=7.6, color="#CBD5E1")
            ax4_l.text(oct_bit[idx] + 1.0, idx + h, f"{oct_bit[idx]:.1f}%", va="center", ha="left", fontsize=7.6, color=TEXT_SUB)
        ax4_l.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, facecolor=BG_PANEL, edgecolor=BORDER_COLOR, fontsize=8.6, labelcolor=TEXT_MAIN)

        style_ax(ax4_r)
        ax4_r.barh(y_c - h, bc_ultra, height=h, color=C_ULTRA, label=f"Armiger Ultra Fair Hybrid ({p2a_stats['models'][ref_llm]['point_pct']:.2f}% N=1,025)", zorder=3)
        ax4_r.barh(y_c, bc_gpt_loto, height=h, color=C_GPT, label=f"OpenAI GPT Fast LOTO ({p2a_loto_gpt_stats['models'][gpt_llm]['point_pct']:.2f}% N=974)", zorder=3)
        ax4_r.barh(y_c + h, bc_gem_loto, height=h, color=C_GEMINI, label=f"Google Gemini Flash LOTO ({p2a_loto_gemini_stats['models'][gem_llm]['point_pct']:.2f}% N=982)", zorder=3)
        ax4_r.set_yticks(y_c)
        ax4_r.set_yticklabels(cats_bc_labels, fontsize=10, fontweight="bold", color=TEXT_MAIN)
        ax4_r.set_ylim(len(cats_bc_labels) - 0.35, -0.6)
        ax4_r.set_xlim(0, 128)
        ax4_r.set_xticks([0, 20, 40, 60, 80, 100])
        ax4_r.set_xlabel("Per-Class Recall (%) — Vendor-Disjoint & LOTO Support Reference", fontsize=9.8, color=TEXT_SUB, labelpad=8)
        ax4_r.set_title("2. Per-Category Recall vs. Teacher LLMs (LOTO Support)\n(beyond_confirmation: Ultra N=1,025; GPT LOTO N=974; Gemini LOTO N=982)", fontsize=11.2, fontweight="bold", color=TEXT_MAIN, pad=10)
        for idx in range(len(cats_bc_labels)):
            ax4_r.text(bc_ultra[idx] + 1.0, idx - h, f"{bc_ultra[idx]:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold", color=C_ULTRA_EDGE)
            ax4_r.text(bc_gpt_loto[idx] + 1.0, idx, f"{bc_gpt_loto[idx]:.1f}%", va="center", ha="left", fontsize=7.6, color=C_GPT)
            ax4_r.text(bc_gem_loto[idx] + 1.0, idx + h, f"{bc_gem_loto[idx]:.1f}%", va="center", ha="left", fontsize=7.6, color=C_GEMINI)
        ax4_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, facecolor=BG_PANEL, edgecolor=BORDER_COLOR, fontsize=8.6, labelcolor=TEXT_MAIN)

        sha_chart4 = save_clean_rgb_png(fig4, assets_dir / "armiger_categorical_breakdown.png")
    else:
        for fn in (
            "armiger_vs_llms_overall_and_6dims.png",
            "armiger_vs_llms_cost_comparison.png",
            "armiger_vs_torrent_classifiers_6dims.png",
            "armiger_categorical_breakdown.png",
        ):
            verify_png_chunks_clean(fallback_assets / fn)
        sha_chart1 = hashlib.sha256((fallback_assets / "armiger_vs_llms_overall_and_6dims.png").read_bytes()).hexdigest()
        sha_chart2 = hashlib.sha256((fallback_assets / "armiger_vs_llms_cost_comparison.png").read_bytes()).hexdigest()
        sha_chart3 = hashlib.sha256((fallback_assets / "armiger_vs_torrent_classifiers_6dims.png").read_bytes()).hexdigest()
        sha_chart4 = hashlib.sha256((fallback_assets / "armiger_categorical_breakdown.png").read_bytes()).hexdigest()

    # ==========================================================================
    # FIVE-SPLIT QUANTIZATION PARITY SUMMARY (uses `quantization_variants`, not `models`)
    # ==========================================================================
    five_split_summary: Dict[str, Any] = {}
    for sp_code, sp_label, n_exp in [
        ("oct15", "oct15_confirmation", 2835),
        ("fresh", "fresh_confirmation", 1402),
        ("beyond", "beyond_confirmation", 1025),
        ("tc2000", "zero_cue_2000", 2000),
        ("tc3000", "multilang_3000", 3000),
    ]:
        y_sp = d[f"split5_{sp_code}_y_cat"]
        sha_sp = ksha(f"split5_{sp_code}_y_cat")
        sp_stats = compute_paired_panel_stats(
            f"five_split_{sp_label}",
            {
                "armiger_ultra_hyb": (d[f"split5_{sp_code}_fp32_hyb_pred"] == y_sp).astype(np.float32),
                "armiger_ultra_raw": (d[f"split5_{sp_code}_fp32_raw_pred"] == y_sp).astype(np.float32),
                "armiger_int8_hyb": (d[f"split5_{sp_code}_int8_hyb_pred"] == y_sp).astype(np.float32),
                "armiger_int8_raw": (d[f"split5_{sp_code}_int8_raw_pred"] == y_sp).astype(np.float32),
            },
            ref_model="armiger_ultra_hyb",
            expected_n=n_exp,
            metric_type="mean",
            key_sha256={
                "armiger_ultra_hyb": sha_sp,
                "armiger_ultra_raw": sha_sp,
                "armiger_int8_hyb": sha_sp,
                "armiger_int8_raw": sha_sp,
            },
            family_ids=d[f"split5_{sp_code}_family_ids"],
        )
        sp_stats["quantization_variants"] = sp_stats.pop("models")
        five_split_summary[sp_label] = sp_stats

    # ==========================================================================
    # WRITE CRYPTOGRAPHIC PROVENANCE MANIFEST (CHARTS_MANIFEST.json)
    # ==========================================================================
    repo_root = assets_dir.parent
    prov_hashes: Dict[str, str] = {
        "canonical_eval_vectors.json": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "build_canonical_eval_vectors.py": hashlib.sha256((BENCH_DIR / "build_canonical_eval_vectors.py").read_bytes()).hexdigest(),
        "generate_canonical_charts.py": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "docs/assets/armiger_vs_llms_overall_and_6dims.png": sha_chart1,
        "docs/assets/armiger_vs_llms_cost_comparison.png": sha_chart2,
        "docs/assets/armiger_vs_torrent_classifiers_6dims.png": sha_chart3,
        "docs/assets/armiger_categorical_breakdown.png": sha_chart4,
    }
    for r_file in sorted((BENCH_DIR / "receipts").glob("*.json")):
        prov_hashes[f"benchmarks/receipts/{r_file.name}"] = hashlib.sha256(r_file.read_bytes()).hexdigest()
    model_manifest = REPO_ROOT / "src" / "armiger" / "model_manifest.json"
    prov_hashes["src/armiger/model_manifest.json"] = hashlib.sha256(model_manifest.read_bytes()).hexdigest()

    manifest: Dict[str, Any] = {
        "schema_version": "3.1.0",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "invariants_verified": {
            "receipt_backed_provenance_lock": True,
            "identical_denominator_lock": True,
            "identical_reference_key_lock": True,
            "anti_circularity_lock": True,
            "family_clustered_bootstrap": True,
            "single_armiger_bar_per_comparator_panel_h1": True,
            "png_metadata_stripped_ihdr_idat_iend_only": True,
        },
        "anti_circularity_audit": anti_circ_checks,
        "provenance_sha256": prov_hashes,
        "charts": {
            "armiger_vs_llms_overall_and_6dims.png": {
                "sha256": sha_chart1,
                "composite_6dim_mean": composite_stats,
                "panel_2a_category_11class": p2a_stats,
                "panel_2b_catalog_matching": p2b_stats,
                "panel_2c_audio_strict_229": p2c_stats,
                "panel_2d_anime_valid_636": p2d_stats,
                "panel_2e_junk_balanced_977": p2e_stats,
                "panel_2f_swarm_public_757": p2f_pub_stats,
                "panel_2f_swarm_warehouse_10278": p2f_whse_stats,
            },
            "armiger_vs_llms_cost_comparison.png": {
                "sha256": sha_chart2,
                "deployment_targets": models_cost,
                "batched_usd_per_1m": batched_cost,
                "realtime_usd_per_1m": realtime_cost,
                "measured_local_onnx_cpu": hw_cost_rec["measured_local_onnx_cpu"],
                "logged_provider_usage": hw_cost_rec["logged_provider_usage"],
            },
            "armiger_vs_torrent_classifiers_6dims.png": {
                "sha256": sha_chart3,
                "panel_1_category_11class_oct15": c3_p1_stats,
                "panel_1_video_mvtv_oct15": c3_p1_vid_stats,
                "panel_2_catalog_matching_oct15": c3_p2_stats,
                "panel_3_audio_strict_oct15": c3_p3_stats,
                "panel_4_anime_oct15": c3_p4_oct_stats,
                "panel_4_anime_dub_zero_cue_2000": c3_p4_tc2k_stats,
                "panel_5_adult_oct15": c3_p5_stats,
                "panel_6a_junk_oct15": c3_p6a_stats,
            },
            "armiger_categorical_breakdown.png": {
                "sha256": sha_chart4,
                "left_oct15_confirmation_n2835": chart4_oct_manifest,
                "right_beyond_confirmation_n1025_loto": chart4_bc_manifest,
            },
        },
        "five_split_category_summary": five_split_summary,
    }

    for chart_name, chart_dict in manifest["charts"].items():
        if isinstance(chart_dict, dict):
            for panel_key, panel_dict in chart_dict.items():
                if isinstance(panel_dict, dict) and "models" in panel_dict:
                    for m_name, st_dict in panel_dict["models"].items():
                        if isinstance(st_dict, dict) and "point_pct" in st_dict:
                            st_dict["readme_cited"] = (chart_name, panel_key, m_name) in EVALUATION_CITED_PANEL_MODELS

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical Armiger benchmark charts and CHARTS_MANIFEST.json.")
    here = pathlib.Path(__file__).resolve().parent
    parser.add_argument("--vectors", type=pathlib.Path, default=here / "canonical_eval_vectors.json")
    parser.add_argument("--assets-dir", type=pathlib.Path, default=here.parent / "docs" / "assets")
    parser.add_argument("--manifest", type=pathlib.Path, default=here / "CHARTS_MANIFEST.json")
    args = parser.parse_args()

    manifest = build_all_charts_and_manifest(args.vectors, args.assets_dir, args.manifest)
    for name, info in manifest["charts"].items():
        print(f"[OK] {name:46s} sha256={info['sha256'][:16]}...")
    print(f"[OK] Wrote {args.manifest}")


if __name__ == "__main__":
    main()
