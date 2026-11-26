"""Consistency checks for the Discussion #55 progression, R22-live, and R26-live charts.

Validates that chart-data.json covers every published model version,
keeps one denominator per task, derives gains, error reductions, and
velocities from the stored point estimates, and that r22-live-data.json,
r26-live-data.json, and all published PNG charts match their exact numeric invariants.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_VERSIONS = ["3.3.0", "3.4.0", "3.7.0", "3.8.0", "3.10.0", "3.11.0", "3.12.0", "3.15.4", "3.17.0"]
EXPECTED_STEPS = [1, 3, 1, 2, 1, 1, 3, 2]
NEWEST_POINTS = {
    "p1_cat10": 93.32,
    "p1_vid": 88.01,
    "p2_match": 88.01,
    "p3_audio": 92.65,
    "p4_anime": 92.85,
}


def _data() -> dict:
    return json.loads((ROOT / "chart-data.json").read_text(encoding="utf-8"))


def _r22_data() -> dict:
    return json.loads((ROOT / "r22-live-data.json").read_text(encoding="utf-8"))


def _r26_data() -> dict:
    return json.loads((ROOT / "r26-live-data.json").read_text(encoding="utf-8"))


def test_versions_and_sources_cover_newest_model():
    data = _data()
    assert data["versions"] == EXPECTED_VERSIONS
    assert data["minor_version_steps"] == EXPECTED_STEPS
    assert len(data["velocity_intervals"]) == len(EXPECTED_STEPS)
    assert "ad5a1e3dec56842d5b0aa6ef881151a4b87dd2a8" in data["sources"]["3.12.0"]
    assert "e47d3ee7d2f05a41c993814968b8d9a9a4042f63" in data["sources"]["3.15.4"]
    assert "1a1e224914913092451b77ff1d19b1ed472b0276" in data["sources"]["3.17.0"]


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
        assert abs(scores["3.17.0"]["point_pct"] - NEWEST_POINTS[task["key"]]) < 1e-9


def test_r26_live_data_invariants():
    r26 = _r26_data()
    assert r26["cohort"] == "R26-live"
    assert r26["release_tag"] == "v3.17.0"
    assert r26["total_rows"] == 3104
    by_key = {m["key"]: m for m in r26["metrics"]}
    assert by_key["unified_prod"]["v3_17_0"]["pct"] == 91.53
    assert by_key["unified_comb"]["v3_17_0"]["pct"] == 92.91
    assert by_key["clean10_comb"]["v3_17_0"]["pct"] == 95.10
    assert by_key["audio_strict"]["v3_17_0"]["pct"] == 90.68
    assert by_key["xxx_comb"]["v3_17_0"]["pct"] == 96.38
    assert by_key["policy2_comb"]["v3_17_0"]["pct"] == 99.80
    assert by_key["clean_tnr_comb"]["v3_17_0"]["pct"] == 99.88
    for m in r26["metrics"]:
        assert m["v3_17_0"]["pct"] > m["v3_15_4"]["pct"]


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
