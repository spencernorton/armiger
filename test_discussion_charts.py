"""Consistency checks for the Discussion #55 progression and R22-live charts.

Validates that chart-data.json covers every published model version,
keeps one denominator per task, derives gains, error reductions, and
velocities from the stored point estimates, and that r22-live-data.json
and all published PNG charts match their exact numeric invariants.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_VERSIONS = ["3.3.0", "3.4.0", "3.7.0", "3.8.0", "3.10.0", "3.11.0", "3.12.0", "3.15.4"]
EXPECTED_STEPS = [1, 3, 1, 2, 1, 1, 3]
NEWEST_POINTS = {
    "p1_cat10": 95.09,
    "p1_vid": 90.56,
    "p2_match": 90.56,
    "p3_audio": 91.31,
    "p4_anime": 90.84,
}


def _data() -> dict:
    return json.loads((ROOT / "chart-data.json").read_text(encoding="utf-8"))


def _r22_data() -> dict:
    return json.loads((ROOT / "r22-live-data.json").read_text(encoding="utf-8"))


def test_versions_and_sources_cover_newest_model():
    data = _data()
    assert data["versions"] == EXPECTED_VERSIONS
    assert data["minor_version_steps"] == EXPECTED_STEPS
    assert len(data["velocity_intervals"]) == len(EXPECTED_STEPS)
    assert "ad5a1e3dec56842d5b0aa6ef881151a4b87dd2a8" in data["sources"]["3.12.0"]
    assert "e47d3ee7d2f05a41c993814968b8d9a9a4042f63" in data["sources"]["3.15.4"]


def test_task_math_is_derived_from_scores():
    data = _data()
    assert len(data["tasks"]) == 5
    for task in data["tasks"]:
        scores = task["scores"]
        assert set(scores) == set(EXPECTED_VERSIONS)
        assert len({s["n"] for s in scores.values()}) == 1
        points = [scores[v]["point_pct"] for v in EXPECTED_VERSIONS]
        assert abs(task["gain_pp"] - (points[-1] - points[0])) < 1e-6
        expected_rer = (points[-1] - points[0]) / (100 - points[0]) * 100
        assert abs(task["relative_error_reduction_pct"] - expected_rer) < 1e-6
        expected_rates = [
            (points[i + 1] - points[i]) / EXPECTED_STEPS[i]
            for i in range(len(EXPECTED_STEPS))
        ]
        assert task["gain_pp_per_minor_version_step"] == expected_rates
        assert abs(scores["3.15.4"]["point_pct"] - NEWEST_POINTS[task["key"]]) < 1e-9


def test_r22_live_data_invariants():
    r22 = _r22_data()
    assert r22["cohort"] == "R22-live"
    assert r22["release_tag"] == "v3.15.4"
    assert r22["total_rows"] == 2795
    assert r22["fp32_vs_int8_agreement_rows"] == 2792
    by_key = {m["key"]: m for m in r22["metrics"]}
    assert set(by_key) == {
        "r22_prod",
        "r22_stress",
        "r22_combined",
        "struct_junk",
        "sem_junk",
        "clean_tnr",
    }
    assert by_key["r22_prod"]["n"] == 1500
    assert by_key["r22_prod"]["v3_15_4"]["pct"] == 94.13
    assert by_key["r22_stress"]["n"] == 1295
    assert by_key["r22_stress"]["v3_15_4"]["pct"] == 90.89
    assert by_key["r22_combined"]["n"] == 2795
    assert by_key["r22_combined"]["v3_15_4"]["pct"] == 92.63
    assert by_key["struct_junk"]["v3_15_4"]["pct"] == 98.99
    assert by_key["sem_junk"]["v3_15_4"]["pct"] == 91.55
    assert by_key["clean_tnr"]["v3_15_4"]["pct"] == 99.57
    for m in r22["metrics"]:
        expected_delta = round(100.0 * (m["v3_15_4"]["correct"] - m["v3_12_0"]["correct"]) / m["n"], 2)
        assert abs(m["delta_pp"] - expected_delta) < 1e-9
    rt = r22["runtime"]
    assert rt["v3_15_4_fp32"]["mean_ms_per_row"] < rt["v3_12_0_fp32"]["mean_ms_per_row"]
    assert rt["v3_15_4_int8"]["mean_ms_per_row"] < rt["v3_15_4_fp32"]["mean_ms_per_row"]
    assert rt["v3_15_4_fp32"]["peak_rss_mib"] < rt["v3_12_0_fp32"]["peak_rss_mib"]


def test_chart_images_exist():
    for name in (
        "accuracy-by-task.png",
        "relative-error-reduction.png",
        "improvement-velocity.png",
        "sealed-r22-live-v312-vs-v3154.png",
        "small-vs-broad-evaluation.png",
    ):
        blob = (ROOT / name).read_bytes()
        assert len(blob) > 10_000
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
