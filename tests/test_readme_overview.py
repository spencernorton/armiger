"""Keep the public overview tied to measured results and honest score units."""
from __future__ import annotations

import copy
import importlib.util
import json
import struct
from pathlib import Path

import pytest

from generate_readme_overview import (
    IMAGE, RECORD, ROOT, SOURCE, ULTRA, build_data, load_source, render, sha256,
)


def test_category_values_match_individual_predictions():
    data = build_data(load_source())
    vectors = json.loads((ROOT / "benchmarks/canonical_eval_vectors.json").read_text())["vectors"]
    truth = vectors["oct15_cat11_y_true"]
    labels = {
        "Movies": 0, "TV shows": 1, "Adult content": 2, "Music": 3,
        "Comics": 4, "Games": 5, "Audiobooks": 6, "Software": 7,
        "eBooks": 8, "Courses": 9,
    }
    predictions = {
        ULTRA: vectors["oct15_cat11_pred_ultra"],
        "MIMMO BERT (360M)": vectors["oct15_cat11_pred_mimmo_bert"],
        "Bitmagnet Built-In": vectors["oct15_cat11_pred_bitmagnet"],
    }
    assert {row["label"] for row in data["categories"]} == set(labels)
    for row in data["categories"]:
        category = labels[row["label"]]
        indices = [i for i, value in enumerate(truth) if value == category]
        assert row["n"] == len(indices)
        for model, predicted in predictions.items():
            assert row["scores"][model] == round(100 * sum(predicted[i] == category for i in indices) / len(indices), 2)
    for model, predicted in predictions.items():
        assert data["overall"]["n"] == len(truth)
        assert data["overall"]["scores"][model] == round(100 * sum(a == b for a, b in zip(predicted, truth)) / len(truth), 2)
    assert data["other_category_n"] == sum(value == 10 for value in truth)


def test_checks_keep_their_original_metrics_and_cohorts():
    source = load_source()
    data = build_data(source)
    assert (data["title_matching"]["metric"], data["title_matching"]["n"]) == ("mean", 1296)
    assert (data["audio_language"]["metric"], data["audio_language"]["n"]) == ("mean", 545)
    assert (data["junk"]["metric"], data["junk"]["n"]) == ("balanced", 977)
    assert data["junk"]["scores"] == {ULTRA: 99.23}
    assert (data["seeds"]["metric"], data["seeds"]["n"]) == ("auroc", 10278)
    assert data["seeds"]["scores"] == {ULTRA: 96.83, "Raw Seeder Count": 85.17}
    wrong = copy.deepcopy(source)
    wrong["charts"]["armiger_vs_llms_overall_and_6dims.png"]["panel_2f_swarm_warehouse_10278"]["metric_type"] = "mean"
    with pytest.raises(ValueError, match="Unexpected metric"):
        build_data(wrong)
    wrong = copy.deepcopy(source)
    wrong["charts"]["armiger_vs_torrent_classifiers_6dims.png"]["panel_2_catalog_matching_oct15"]["models"]["GuessIt"]["n"] -= 1
    with pytest.raises(ValueError, match="denominators differ"):
        build_data(wrong)


def test_committed_overview_matches_its_evidence_and_renderer():
    record = json.loads((ROOT / RECORD).read_text())
    assert record["data"] == build_data(load_source())
    assert record["source_manifest_sha256"] == sha256(ROOT / SOURCE)
    assert record["renderer_sha256"] == sha256(ROOT / record["renderer"])
    assert record["image_sha256"] == sha256(ROOT / IMAGE)


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="Optional chart renderer not installed")
def test_renderer_writes_clean_rgb_png(tmp_path: Path):
    image = tmp_path / "overview.png"
    render(build_data(load_source()), image)
    raw = image.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, colour, *_ = struct.unpack(">IIBBBBB", raw[16:29])
    assert (width, height, depth, colour) == (1920, 1200, 8, 2)
    offset, chunks = 8, []
    while offset < len(raw):
        size = struct.unpack(">I", raw[offset:offset + 4])[0]
        chunks.append(raw[offset + 4:offset + 8])
        offset += size + 12
    assert set(chunks) <= {b"IHDR", b"IDAT", b"IEND"}
