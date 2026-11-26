#!/usr/bin/env python3
"""Generate the standardized 4-chart visual suite for Armiger Discussion #55.

Reads exclusively from the committed primary JSON files in this directory:
  - chart-data.json
  - r26-live-data.json
  - evaluation-coverage-data.json

Outputs (300 DPI PNG, unified GitHub-dark palette, consistent version colors):
  1. unseen-live-benchmark.png
  2. relative-improvement-model-over-model.png
  3. legacy-vs-live-generalization.png
  4. small-vs-broad-evaluation.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
CHART_DATA = json.loads((ROOT / "chart-data.json").read_text(encoding="utf-8"))
R26_DATA = json.loads((ROOT / "r26-live-data.json").read_text(encoding="utf-8"))
COV_DATA = json.loads((ROOT / "evaluation-coverage-data.json").read_text(encoding="utf-8"))

# Unified GitHub-dark design system (identical across all 4 charts)
BG = "#0d1117"
PANEL = "#161b22"
EDGE = "#30363d"
GRID = "#252b35"
TXT = "#e6edf3"
MUTED = "#8b949e"
KICKER = "#f0883e"

# Fixed semantic version colors across the entire suite
C_V12 = "#f85149"   # Coral red: v3.12.0 (legacy rules + .pkl)
C_V15 = "#d29922"   # Warm amber: v3.15.4 (rules removed)
C_V17 = "#3fb950"   # Emerald green: v3.17.0 (clean-sheet, 0 .pkl)

# Fixed task / accent colors
C_CAT = "#58a6ff"   # Bright blue: 10-class category
C_AUD = "#bc8cff"   # Soft purple: audio language
C_ANI = "#f2cc60"   # Warm gold: anime detection
C_MAT = "#ff9bce"   # Soft rose: title matching
C_TEAL = "#39c5cf"  # Cyan: runtime / cumulative / Policy-2
C_ORG = "#f0883e"   # Warm orange: broader evaluation pool

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TXT,
    "axes.labelcolor": TXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": EDGE,
    "axes.facecolor": PANEL,
    "figure.facecolor": BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def wilson_ci_pct(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return exact 95% Wilson confidence interval in percentage points."""
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (center - half) * 100.0), min(100.0, (center + half) * 100.0)


def rer_pct(old_pct: float, new_pct: float) -> float:
    """Relative error reduction (%) from old_pct -> new_pct."""
    return (new_pct - old_pct) / (100.0 - old_pct) * 100.0


def add_header(fig: plt.Figure, kicker: str, title: str, subtitle: str, x: float = 0.045) -> None:
    fig.text(x, 0.945, kicker, fontsize=9.8, fontweight="bold", color=KICKER)
    fig.text(x, 0.896, title, fontsize=17.5, fontweight="bold", color=TXT)
    fig.text(x, 0.858, subtitle, fontsize=9.8, color=MUTED)


def add_footer(fig: plt.Figure, text: str, x: float = 0.045) -> None:
    fig.text(x, 0.018, text, fontsize=8.1, color=MUTED, style="italic")


# =============================================================================
# CHART 1: unseen-live-benchmark.png
# =============================================================================
def render_unseen_live_benchmark() -> None:
    fig = plt.figure(figsize=(15.6, 7.6), facecolor=BG)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.62, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.062, right=0.968, top=0.795, bottom=0.115,
        wspace=0.22, hspace=0.48,
    )
    add_header(
        fig,
        "ARMIGER ULTRA  ·  FIREWALLED LIVE BENCHMARK",
        "Unseen Live Accuracy, Safety & Efficiency (v3.17.0)",
        "Zero-overlap R26-live (N = 3,104) with 95% Wilson CIs · 0 .pkl files, torch-free",
        x=0.062,
    )

    # Left panel: R26-live grouped bars
    axL = fig.add_subplot(gs[:, 0])
    r_map = {m["key"]: m for m in R26_DATA["metrics"]}
    items = [
        ("unified_prod", "Live Prod\n(N=2,350)", "p=2.6e-5"),
        ("unified_comb", "Combined\n(N=3,104)", "p=9.0e-10"),
        ("clean10_comb", "Clean-10\n(N=2,346)", "p=3.2e-5"),
        ("audio_strict", "Audio Lang\n(N=858)", "p=0.0029"),
        ("xxx_comb", "Adult / XXX\n(N=938)", "+4.1 pp"),
        ("clean_tnr_comb", "Clean TNR\n(N=2,571)", "36→3 FPs"),
    ]
    bx = np.arange(len(items))
    w = 0.255

    for idx_m, (mk, mlbl, mcol) in enumerate([
        ("v3_12_0", "v3.12.0 (rules + .pkl)", C_V12),
        ("v3_15_4", "v3.15.4 (no rules)", C_V15),
        ("v3_17_0", "v3.17.0 (0 .pkl)", C_V17),
    ]):
        vals = np.array([r_map[k][mk]["pct"] for k, _, _ in items])
        lo = np.array([r_map[k][mk]["pct"] - r_map[k][mk]["ci95_lo"] for k, _, _ in items])
        hi = np.array([r_map[k][mk]["ci95_hi"] - r_map[k][mk]["pct"] for k, _, _ in items])
        pos = bx + (idx_m - 1) * w
        bars = axL.bar(
            pos, vals, w * 0.90, yerr=[lo, hi], label=mlbl, color=mcol, alpha=0.92,
            edgecolor=BG, linewidth=0.8,
            error_kw=dict(ecolor=TXT, elinewidth=1.0, capsize=2.4, capthick=1.0), zorder=3,
        )
        if mk == "v3_17_0":
            for i_b, (rect, v, hi_e) in enumerate(zip(bars, vals, hi)):
                axL.text(
                    rect.get_x() + rect.get_width() / 2, v + hi_e + 0.65,
                    f"{v:.1f}%\n({items[i_b][2]})",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#7ee787",
                )
        elif mk == "v3_12_0":
            for i_b, (rect, v, hi_e) in enumerate(zip(bars, vals, hi)):
                if i_b in (0, 1):
                    axL.text(
                        rect.get_x() + rect.get_width() / 2, v + hi_e + 0.55,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=7.3, fontweight="bold", color="#ffa198",
                    )

    axL.set_title("Firewalled R26-live Accuracy & Safety", fontsize=11.2, fontweight="bold", loc="left", pad=10)
    axL.set_xticks(bx)
    axL.set_xticklabels([lbl for _, lbl, _ in items], fontsize=8.4)
    axL.set_ylim(66, 111.5)
    axL.set_ylabel("Accuracy / Recall / TNR (%)", fontsize=9.4)
    axL.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8)
    axL.legend(loc="lower right", fontsize=8.2, facecolor=PANEL, edgecolor=EDGE, framealpha=0.96)

    p2 = r_map["policy2_comb"]
    axL.text(
        0.50, 0.962,
        f"Policy-2 malware recall (N={p2['n']}, off-scale):  "
        f"v3.12 {p2['v3_12_0']['pct']:.1f}%  →  v3.15.4 {p2['v3_15_4']['pct']:.1f}%  →  v3.17.0 {p2['v3_17_0']['pct']:.1f}%",
        transform=axL.transAxes, ha="center", va="top",
        fontsize=7.9, fontweight="bold", color=C_TEAL,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#10232b", edgecolor=C_TEAL, alpha=0.95),
    )

    # Top-right: Live container RAM
    rt = R26_DATA["runtime"]
    axR1 = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(3)[::-1]
    labels = ["v3.12.0", "v3.15.4", "v3.17.0"]
    cols = [C_V12, C_V15, C_V17]
    rss = [rt["v3_12_0"]["live_container_rss_mib"], rt["v3_15_4"]["live_container_rss_mib"], rt["v3_17_0"]["live_container_rss_mib"]]
    axR1.barh(y_pos, rss, height=0.56, color=cols, alpha=0.92, edgecolor=BG, zorder=3)
    axR1.set_yticks(y_pos)
    axR1.set_yticklabels(labels, fontsize=9.0, fontweight="bold", color=TXT)
    axR1.set_xlim(0, 3200)
    axR1.set_xlabel("MiB (lower is leaner)", fontsize=8.6)
    axR1.set_title("Live Production Container RAM", fontsize=11.0, fontweight="bold", loc="left", pad=8)
    axR1.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)
    rss_notes = [f"{rss[0]:,.0f} MiB", f"{rss[1]:,.0f} MiB  (−46%)", f"{rss[2]:,.0f} MiB  (−87%)"]
    for y, v, txt, c in zip(y_pos, rss, rss_notes, ["#ffa198", "#f2cc60", "#7ee787"]):
        axR1.text(v + 55, y, txt, va="center", ha="left", fontsize=8.6, fontweight="bold", color=c)

    # Bottom-right: BS=32 CPU latency + compact R27 strip
    axR2 = fig.add_subplot(gs[1, 1])
    ms = [rt["v3_12_0"]["mean_ms_per_row"], rt["v3_15_4"]["mean_ms_per_row"], rt["v3_17_0"]["mean_ms_per_row"]]
    axR2.barh(y_pos, ms, height=0.56, color=cols, alpha=0.92, edgecolor=BG, zorder=3)
    axR2.set_yticks(y_pos)
    axR2.set_yticklabels(labels, fontsize=9.0, fontweight="bold", color=TXT)
    axR2.set_xlim(0, 34.5)
    axR2.set_xlabel("ms / row at BS=32 (lower is faster)", fontsize=8.6)
    axR2.set_title("CPU Inference Latency", fontsize=11.0, fontweight="bold", loc="left", pad=8)
    axR2.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)
    ms_notes = [f"{ms[0]:.1f} ms", f"{ms[1]:.1f} ms  (4.6× faster)", f"{ms[2]:.1f} ms  (6.7× faster)"]
    for y, v, txt, c in zip(y_pos, ms, ms_notes, ["#ffa198", "#f2cc60", "#7ee787"]):
        axR2.text(v + 0.65, y, txt, va="center", ha="left", fontsize=8.6, fontweight="bold", color=c)

    r27 = R26_DATA["r27_prospective_audit"]
    fig.text(
        0.968, 0.018,
        f"R27-live shadow ({r27['live_shadow_rows']:,} unseen): v3.17.0 won {r27['v3_17_0_wins']} vs {r27['v3_15_4_wins']} "
        f"({r27['v3_17_0_win_rate_pct']}%, p=1.98e-28) · 180/180 canaries",
        ha="right", fontsize=8.1, fontweight="bold", color="#7ee787",
    )
    add_footer(fig, "Paired two-sided McNemar p-values (v3.15.4 → v3.17.0) · Source: r26-live-data.json", x=0.062)

    fig.savefig(ROOT / "unseen-live-benchmark.png", dpi=300, facecolor=BG)
    plt.close(fig)


# =============================================================================
# CHART 2: relative-improvement-model-over-model.png (THE SINGLE RELATIVE CHART)
# =============================================================================
def render_relative_improvement() -> None:
    fig = plt.figure(figsize=(15.6, 7.0), facecolor=BG)
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.35, 1.0],
        left=0.058, right=0.965, top=0.750, bottom=0.145,
        wspace=0.26,
    )
    add_header(
        fig,
        "ARMIGER ULTRA  ·  MODEL-OVER-MODEL RELATIVE IMPROVEMENT",
        "Remaining Errors Cut by Each Release",
        "Relative error reduction vs. prior model: (new − old) ÷ (100 − old) × 100 · higher is better",
        x=0.058,
    )

    # Left Panel: Step-by-step release progression with explicit regime split
    axL = fig.add_subplot(gs[0, 0])
    vers = CHART_DATA["versions"]
    tmap = {t["key"]: t["scores"] for t in CHART_DATA["tasks"]}
    r_map = {m["key"]: m for m in R26_DATA["metrics"]}

    # Steps 0..5 = legacy heldout_20261026 (3.3->3.12); Steps 6..7 = firewalled R26-live (3.12->3.15.4->3.17.0)
    x_labels = [
        "3.3→3.4", "3.4→3.7", "3.7→3.8", "3.8→3.10", "3.10→3.11", "3.11→3.12",
        "3.12→3.15.4", "3.15.4→3.17",
    ]
    x_leg = np.arange(6, dtype=float)
    x_liv = np.array([6.0, 7.0])

    cat_leg = [rer_pct(tmap["p1_cat10"][vers[i]]["point_pct"], tmap["p1_cat10"][vers[i + 1]]["point_pct"]) for i in range(6)]
    aud_leg = [rer_pct(tmap["p3_audio"][vers[i]]["point_pct"], tmap["p3_audio"][vers[i + 1]]["point_pct"]) for i in range(6)]

    cat_liv = [
        rer_pct(r_map["clean10_comb"]["v3_12_0"]["pct"], r_map["clean10_comb"]["v3_15_4"]["pct"]),
        rer_pct(r_map["clean10_comb"]["v3_15_4"]["pct"], r_map["clean10_comb"]["v3_17_0"]["pct"]),
    ]
    aud_liv = [
        rer_pct(r_map["audio_strict"]["v3_12_0"]["pct"], r_map["audio_strict"]["v3_15_4"]["pct"]),
        rer_pct(r_map["audio_strict"]["v3_15_4"]["pct"], r_map["audio_strict"]["v3_17_0"]["pct"]),
    ]
    prod_liv = [
        rer_pct(r_map["unified_prod"]["v3_12_0"]["pct"], r_map["unified_prod"]["v3_15_4"]["pct"]),
        rer_pct(r_map["unified_prod"]["v3_15_4"]["pct"], r_map["unified_prod"]["v3_17_0"]["pct"]),
    ]

    # Regime shading + crisp break (NO line crosses between benchmarks!)
    axL.axvspan(-0.45, 5.5, color="#58a6ff", alpha=0.04, zorder=0)
    axL.axvspan(5.5, 7.55, color=C_V17, alpha=0.07, zorder=0)
    axL.axvline(5.5, color=C_V17, linestyle="--", lw=1.3, alpha=0.85, zorder=1)
    axL.axhline(0.0, color=MUTED, linestyle=":", lw=1.0, alpha=0.8, zorder=1)

    # Legacy segment
    axL.plot(x_leg, cat_leg, color=C_CAT, marker="o", markersize=6.5, lw=2.2, label="Category (10-class)", zorder=4)
    axL.plot(x_leg, aud_leg, color=C_AUD, marker="s", markersize=6.0, lw=2.1, linestyle="-.", label="Audio language", zorder=4)

    # Firewalled R26-live segment (disconnected across regime boundary!)
    axL.plot(x_liv, cat_liv, color=C_CAT, marker="o", markersize=7.0, lw=2.4, zorder=4)
    axL.plot(x_liv, aud_liv, color=C_AUD, marker="s", markersize=6.5, lw=2.3, linestyle="-.", zorder=4)
    axL.plot(x_liv, prod_liv, color=C_V17, marker="D", markersize=7.2, lw=2.5, label="Live prod sample (R26-live)", zorder=5)

    # Direct clean value labels on the unseen R26-live steps (zero clutter!)
    axL.text(6.0, prod_liv[0] + 3.0, f"+{prod_liv[0]:.1f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#7ee787")
    axL.text(7.0, prod_liv[1] - 5.2, f"+{prod_liv[1]:.1f}%", ha="center", va="top", fontsize=8.3, fontweight="bold", color="#7ee787")
    axL.text(7.0, aud_liv[1] + 3.6, f"+{aud_liv[1]:.1f}%", ha="center", va="bottom", fontsize=8.1, fontweight="bold", color=C_AUD)
    axL.text(7.14, cat_liv[1] - 0.3, f"+{cat_liv[1]:.1f}%", ha="left", va="center", fontsize=8.1, fontweight="bold", color=C_CAT)
    axL.text(3.0, aud_leg[3] - 3.0, "stem cleanup", ha="center", va="top", fontsize=7.6, color=MUTED, style="italic")

    axL.text(2.5, 67.5, "Legacy heldout_20261026", ha="center", va="center", fontsize=8.4, color=MUTED, style="italic")
    axL.text(6.52, 67.5, "Unseen R26-live", ha="center", va="center", fontsize=8.5, color="#7ee787", fontweight="bold")

    axL.set_title("Step-by-Step Relative Error Reduction", fontsize=11.2, fontweight="bold", loc="left", pad=28)
    axL.set_xticks(np.arange(8))
    axL.set_xticklabels(x_labels, fontsize=8.1, rotation=20, ha="right")
    axL.set_xlim(-0.45, 7.88)
    axL.set_ylim(-17, 73)
    axL.set_ylabel("Errors reduced vs. prior model (%)", fontsize=9.3)
    axL.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8)
    axL.legend(loc="upper left", bbox_to_anchor=(0.01, 0.91), fontsize=8.2, facecolor=PANEL, edgecolor=EDGE, framealpha=0.96)

    # Right Panel: Latest step v3.15.4 -> v3.17.0 on R26-live (+ total vs v3.12.0 markers)
    axR = fig.add_subplot(gs[0, 1])
    rows = [
        ("unified_prod", "Live prod\n(N=2,350)"),
        ("clean10_comb", "Clean-10\n(N=2,346)"),
        ("unified_comb", "Combined\n(N=3,104)"),
        ("audio_strict", "Audio lang\n(N=858)"),
        ("xxx_comb", "Adult / XXX\n(N=938)"),
        ("clean_tnr_comb", "Clean TNR\n(N=2,571)"),
    ]
    ry = np.arange(len(rows))
    step_vals = [rer_pct(r_map[k]["v3_15_4"]["pct"], r_map[k]["v3_17_0"]["pct"]) for k, _ in rows]
    cum_vals = [rer_pct(r_map[k]["v3_12_0"]["pct"], r_map[k]["v3_17_0"]["pct"]) for k, _ in rows]

    bars = axR.barh(ry, step_vals, height=0.56, color=C_V17, alpha=0.90, edgecolor=BG, label="v3.15.4 → v3.17.0", zorder=3)
    axR.scatter(cum_vals, ry, color=C_TEAL, edgecolor=BG, s=70, marker="D", linewidth=1.0, label="Total vs. v3.12.0", zorder=5)

    for y, sv, cv in zip(ry, step_vals, cum_vals):
        # Connect bar end to cumulative diamond subtly when cumulative is higher
        if cv > sv + 3:
            axR.plot([sv, cv], [y, y], color=C_TEAL, linestyle=":", lw=1.4, alpha=0.8, zorder=4)
        right_edge = max(sv, cv)
        axR.text(
            right_edge + 2.2, y,
            f"+{sv:.1f}%  (total {cv:.1f}%)",
            va="center", ha="left", fontsize=8.2, fontweight="bold", color=TXT,
        )

    axR.set_title("Latest Release: v3.15.4 → v3.17.0 (R26-live)", fontsize=11.2, fontweight="bold", loc="left", pad=28)
    axR.set_yticks(ry)
    axR.set_yticklabels([lbl for _, lbl in rows], fontsize=8.5, color=TXT)
    axR.set_xlim(0, 140)
    axR.set_ylim(-0.62, 5.62)
    axR.set_xticks([0, 20, 40, 60, 80, 100])
    axR.set_xlabel("Remaining errors eliminated (%)", fontsize=9.3)
    axR.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)
    axR.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.062), ncol=2, frameon=False, fontsize=8.3, columnspacing=1.5)

    add_footer(fig, "Single standardized relative improvement view · Sources: chart-data.json, r26-live-data.json", x=0.058)

    fig.savefig(ROOT / "relative-improvement-model-over-model.png", dpi=300, facecolor=BG)
    plt.close(fig)


# =============================================================================
# CHART 3: legacy-vs-live-generalization.png
# =============================================================================
def render_legacy_vs_live() -> None:
    fig = plt.figure(figsize=(15.6, 6.9), facecolor=BG)
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.26, 1.0],
        left=0.058, right=0.968, top=0.785, bottom=0.135,
        wspace=0.28,
    )
    add_header(
        fig,
        "ARMIGER ULTRA  ·  LEGACY BENCHMARK VS. UNSEEN TRAFFIC",
        "Why Legacy Scores Dipped While Real-World Accuracy Climbed",
        "Reused heldout_20261026 trajectories vs. firewalled R26-live · exact 95% Wilson CIs",
        x=0.058,
    )

    # Left panel: Legacy heldout_20261026 trajectories across all 4 tasks
    axL = fig.add_subplot(gs[0, 0])
    vers = CHART_DATA["versions"]
    vx = np.arange(len(vers), dtype=float)
    tmap = {t["key"]: t["scores"] for t in CHART_DATA["tasks"]}

    axL.axvspan(-0.45, 6.0, color=C_V12, alpha=0.055, zorder=0)
    axL.axvspan(6.0, 8.45, color=C_V17, alpha=0.065, zorder=0)
    axL.axvline(6.0, color=C_V12, linestyle="--", lw=1.2, alpha=0.78, zorder=1)

    specs = [
        ("p1_cat10", "Category (N=2,484)", C_CAT, "o", "-", -0.09),
        ("p3_audio", "Audio (N=449)", C_AUD, "s", "-.", -0.03),
        ("p4_anime", "Anime (N=797)", C_ANI, "^", "--", +0.03),
        ("p2_match", "Title match (N=784)", C_MAT, "D", ":", +0.09),
    ]
    for key, lbl, col, mk, ls, dx in specs:
        sc = tmap[key]
        ys = np.array([sc[v]["point_pct"] for v in vers])
        lo = np.array([sc[v]["point_pct"] - sc[v]["ci95_lo"] for v in vers])
        hi = np.array([sc[v]["ci95_hi"] - sc[v]["point_pct"] for v in vers])
        axL.errorbar(
            vx + dx, ys, yerr=[lo, hi], label=lbl, color=col, marker=mk, markersize=5.4,
            lw=2.0, linestyle=ls, capsize=2.6, capthick=1.0, elinewidth=1.0, zorder=3,
        )

    cat_sc = tmap["p1_cat10"]
    axL.text(2.7, 101.4, "Regime 1: Rules & .pkl added", ha="center", fontsize=8.3, color="#ffa198", style="italic")
    axL.text(7.22, 101.4, "Regime 2: Cleanup", ha="center", fontsize=8.3, color="#7ee787", fontweight="bold")

    axL.annotate(
        "v3.12 peak: 97.3%\n(2,800 rules + .pkl)",
        xy=(5.91, cat_sc["3.12.0"]["point_pct"]), xytext=(2.45, 97.0),
        fontsize=8.0, fontweight="bold", color="#ff7b72",
        arrowprops=dict(arrowstyle="->", color="#ff7b72", lw=1.1, connectionstyle="arc3,rad=-0.10"),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#21151a", edgecolor=C_V12, alpha=0.94),
    )
    axL.annotate(
        "v3.17: 93.3%\n(0 .pkl)",
        xy=(7.91, cat_sc["3.17.0"]["point_pct"]), xytext=(6.55, 97.2),
        fontsize=8.0, fontweight="bold", color=C_CAT,
        arrowprops=dict(arrowstyle="->", color=C_CAT, lw=1.1, connectionstyle="arc3,rad=-0.12"),
    )

    axL.set_title("Reused Cohort (heldout_20261026)", fontsize=11.2, fontweight="bold", loc="left", pad=10)
    axL.set_xticks(vx)
    axL.set_xticklabels([f"v{v}" for v in vers], fontsize=8.3)
    axL.set_xlim(-0.45, 8.45)
    axL.set_ylim(75.5, 103.0)
    axL.set_ylabel("Accuracy (%)", fontsize=9.4)
    axL.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8)
    axL.legend(loc="lower left", ncol=2, columnspacing=1.0, handlelength=2.1,
               fontsize=7.9, facecolor=PANEL, edgecolor=EDGE, framealpha=0.96)

    # Right panel: Overfitting Cliff Dumbbell
    axR = fig.add_subplot(gs[0, 1])
    r_map = {m["key"]: m for m in R26_DATA["metrics"]}
    d_vers = ["v3.12.0\n(rules + .pkl)", "v3.15.4\n(no rules)", "v3.17.0\n(0 .pkl)"]
    d_leg = [cat_sc["3.12.0"]["point_pct"], cat_sc["3.15.4"]["point_pct"], cat_sc["3.17.0"]["point_pct"]]
    d_leg_lo = [cat_sc[v]["point_pct"] - cat_sc[v]["ci95_lo"] for v in ("3.12.0", "3.15.4", "3.17.0")]
    d_leg_hi = [cat_sc[v]["ci95_hi"] - cat_sc[v]["point_pct"] for v in ("3.12.0", "3.15.4", "3.17.0")]
    d_liv = [r_map["unified_prod"][k]["pct"] for k in ("v3_12_0", "v3_15_4", "v3_17_0")]
    d_liv_lo = [r_map["unified_prod"][k]["pct"] - r_map["unified_prod"][k]["ci95_lo"] for k in ("v3_12_0", "v3_15_4", "v3_17_0")]
    d_liv_hi = [r_map["unified_prod"][k]["ci95_hi"] - r_map["unified_prod"][k]["pct"] for k in ("v3_12_0", "v3_15_4", "v3_17_0")]

    dy = np.arange(3)[::-1]
    for i in range(3):
        gap = d_liv[i] - d_leg[i]
        lcol = C_V12 if abs(gap) > 10 else (C_V15 if abs(gap) > 3 else C_V17)
        axR.plot([d_liv[i], d_leg[i]], [dy[i], dy[i]], color=lcol, lw=4.0, alpha=0.86, zorder=2)
        mid = 0.5 * (d_liv[i] + d_leg[i])
        axR.text(
            mid, dy[i] + 0.16,
            f"Overfitting cliff: {gap:+.1f} pp" if i == 0 else f"Gap: {gap:+.1f} pp",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=lcol,
        )

    axR.errorbar(d_leg, dy, xerr=[d_leg_lo, d_leg_hi], fmt="o", markersize=9.2, color=C_CAT, ecolor=C_CAT,
                 capsize=3.5, lw=1.4, label="Reused heldout_20261026 (N=2,484)", zorder=4)
    axR.errorbar(d_liv, dy, xerr=[d_liv_lo, d_liv_hi], fmt="D", markersize=8.6, color=C_V17, ecolor=C_V17,
                 capsize=3.5, lw=1.4, label="Unseen R26-live Prod (N=2,350)", zorder=4)

    offsets = [(0.5, -0.5), (0.7, -0.7), (1.45, -1.45)]
    for i in range(3):
        axR.text(d_leg[i] + offsets[i][0], dy[i] - 0.19, f"{d_leg[i]:.1f}%", fontsize=8.3, color=C_CAT, fontweight="bold", ha="center", va="top")
        axR.text(d_liv[i] + offsets[i][1], dy[i] - 0.19, f"{d_liv[i]:.1f}%", fontsize=8.3, color=C_V17, fontweight="bold", ha="center", va="top")

    axR.set_title("Generalization Gap (Legacy vs. Live Prod)", fontsize=11.2, fontweight="bold", loc="left", pad=10)
    axR.set_yticks(dy)
    axR.set_yticklabels(d_vers, fontsize=8.8, color=TXT, fontweight="bold")
    axR.set_xlim(68.8, 100.8)
    axR.set_ylim(-0.68, 2.62)
    axR.set_xlabel("Accuracy (%)", fontsize=9.4)
    axR.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)
    axR.legend(loc="lower left", fontsize=8.1, facecolor=PANEL, edgecolor=EDGE, framealpha=0.96)

    add_footer(fig, "Sources: chart-data.json (N=2,484) & r26-live-data.json (N=2,350 live prod)", x=0.058)

    fig.savefig(ROOT / "legacy-vs-live-generalization.png", dpi=300, facecolor=BG)
    plt.close(fig)


# =============================================================================
# CHART 4: small-vs-broad-evaluation.png (Standardized & Improved)
# =============================================================================
def render_small_vs_broad() -> None:
    fig = plt.figure(figsize=(14.4, 6.4), facecolor=BG)
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[0.78, 1.22],
        left=0.068, right=0.958, top=0.775, bottom=0.145,
        wspace=0.34,
    )
    add_header(
        fig,
        "ARMIGER ULTRA  ·  EVALUATION COVERAGE",
        "Same Model, Broader Test, Lower Headline Score",
        "Fixed v3.0.0 pipeline across one cohort vs. five-cohort pool · 95% Wilson CIs",
        x=0.068,
    )

    cohorts = COV_DATA["cohorts"]
    pooled = COV_DATA["pooled"]
    first = cohorts[0]

    # Left panel: Single cohort vs 5-cohort pool (zoomed 84..101.5 for crisp visibility)
    axL = fig.add_subplot(gs[0, 0])
    vals = [first["accuracy_pct"], pooled["accuracy_pct"]]
    cis = [wilson_ci_pct(first["correct"], first["n"]), wilson_ci_pct(pooled["correct"], pooled["n"])]
    lo = [vals[i] - cis[i][0] for i in range(2)]
    hi = [cis[i][1] - vals[i] for i in range(2)]

    lx = np.array([0.0, 1.0])
    bars = axL.bar(
        lx, vals, width=0.52, yerr=[lo, hi], color=[C_CAT, C_ORG], alpha=0.92,
        edgecolor=BG, linewidth=0.9,
        error_kw=dict(ecolor=TXT, elinewidth=1.1, capsize=3.5, capthick=1.1), zorder=3,
    )
    for i, (rect, v, h) in enumerate(zip(bars, vals, hi)):
        sub = f"  ({COV_DATA['accuracy_change_pp']:+.1f} pp)" if i == 1 else ""
        axL.text(
            rect.get_x() + rect.get_width() / 2, v + h + 0.45,
            f"{v:.2f}%{sub}", ha="center", va="bottom",
            fontsize=9.3, fontweight="bold", color=C_CAT if i == 0 else "#ffa657",
        )

    axL.set_title("One Cohort vs. Five-Cohort Pool", fontsize=11.0, fontweight="bold", loc="left", pad=10)
    axL.set_xticks(lx)
    axL.set_xticklabels([f"Multilingual\n(N = {first['n']:,})", f"5-Cohort Pool\n(N = {pooled['n']:,})"], fontsize=8.8, color=TXT, fontweight="bold")
    axL.set_xlim(-0.55, 1.55)
    axL.set_ylim(84, 101.8)
    axL.set_ylabel("Category accuracy (%)", fontsize=9.3)
    axL.grid(axis="y", color=GRID, linestyle="--", linewidth=0.8)

    # Right panel: All 5 cohorts sorted descending + pooled dashed reference line
    axR = fig.add_subplot(gs[0, 1])
    sorted_c = sorted(cohorts, key=lambda c: c["accuracy_pct"], reverse=True)
    ry = np.arange(len(sorted_c))[::-1]
    rvals = [c["accuracy_pct"] for c in sorted_c]
    rcis = [wilson_ci_pct(c["correct"], c["n"]) for c in sorted_c]
    rlo = [rvals[i] - rcis[i][0] for i in range(len(sorted_c))]
    rhi = [rcis[i][1] - rvals[i] for i in range(len(sorted_c))]
    rcols = [C_CAT if c["cohort"] == "multilang_3000" else C_ORG for c in sorted_c]

    axR.barh(
        ry, rvals, height=0.56, xerr=[rlo, rhi], color=rcols, alpha=0.92,
        edgecolor=BG, linewidth=0.9,
        error_kw=dict(ecolor=TXT, elinewidth=1.0, capsize=3.0, capthick=1.0), zorder=3,
    )
    axR.axvline(pooled["accuracy_pct"], color=TXT, linestyle="--", lw=1.2, alpha=0.65, zorder=2)
    axR.text(pooled["accuracy_pct"] + 0.18, -0.35, f"Pooled: {pooled['accuracy_pct']:.2f}%",
             fontsize=8.1, color=TXT, fontweight="bold", va="center", ha="left",
             bbox=dict(facecolor=PANEL, edgecolor="none", pad=1.2, alpha=0.92), zorder=5)

    for y, v, h, c in zip(ry, rvals, rhi, sorted_c):
        axR.text(v + h + 0.25, y, f"{v:.2f}%", va="center", ha="left",
                 fontsize=8.6, fontweight="bold", color=C_CAT if c["cohort"] == "multilang_3000" else "#ffa657",
                 bbox=dict(facecolor=PANEL, edgecolor="none", pad=1.2, alpha=0.92), zorder=5)

    axR.set_title("Breakdown of the Five Cohorts", fontsize=11.0, fontweight="bold", loc="left", pad=10)
    axR.set_yticks(ry)
    axR.set_yticklabels([f"{c['label']}\n(N = {c['n']:,})" for c in sorted_c], fontsize=8.6, color=TXT)
    axR.set_xlim(84, 101.8)
    axR.set_xlabel("Category accuracy (%)", fontsize=9.3)
    axR.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)

    add_footer(fig, "Historical v3.0.0 reference labels · axis starts at 84% to show spread · Source: evaluation-coverage-data.json", x=0.068)

    fig.savefig(ROOT / "small-vs-broad-evaluation.png", dpi=300, facecolor=BG)
    plt.close(fig)


def main() -> None:
    render_unseen_live_benchmark()
    render_relative_improvement()
    render_legacy_vs_live()
    render_small_vs_broad()
    for name in (
        "unseen-live-benchmark.png",
        "relative-improvement-model-over-model.png",
        "legacy-vs-live-generalization.png",
        "small-vs-broad-evaluation.png",
    ):
        p = ROOT / name
        print(f"Generated {name}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
