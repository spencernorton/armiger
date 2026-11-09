"""Consistency checks for the Discussion #55 progression charts.

Validates that chart-data.json covers every published model version,
keeps one denominator per task, and derives gains, error reductions,
and velocities from the stored point estimates.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_VERSIONS = ["3.3.0", "3.4.0", "3.7.0", "3.8.0", "3.10.0", "3.11.0", "3.12.0"]
EXPECTED_STEPS = [1, 3, 1, 2, 1, 1]
NEWEST_POINTS = {
    "p1_cat10": 97.30,
    "p1_vid": 95.54,
    "p2_match": 95.54,
    "p3_audio": 96.88,
    "p4_anime": 95.73,
}


def _data() -> dict:
    return json.loads((ROOT / "chart-data.json").read_text())


def test_versions_and_sources_cover_newest_model():
    data = _data()
    assert data["versions"] == EXPECTED_VERSIONS
    assert data["minor_version_steps"] == EXPECTED_STEPS
    assert "ad5a1e3dec56842d5b0aa6ef881151a4b87dd2a8" in data["sources"]["3.12.0"]


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
        assert abs(scores["3.12.0"]["point_pct"] - NEWEST_POINTS[task["key"]]) < 1e-9


def test_chart_images_exist():
    for name in ("accuracy-by-task.png", "relative-error-reduction.png", "improvement-velocity.png"):
        blob = (ROOT / name).read_bytes()
        assert len(blob) > 10_000
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
