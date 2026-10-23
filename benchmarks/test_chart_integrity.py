"""
Automated integrity and statistical invariant test suite for the canonical
Armiger & Armiger Ultra chart and benchmark pipeline.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pytest

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))


def _load_vectors(path: pathlib.Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    dtypes = raw["_meta"]["dtypes"]
    return {k: np.array(v, dtype=np.dtype(dtypes[k])) for k, v in raw["vectors"].items()}

from generate_canonical_charts import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    HAS_MATPLOTLIB,
    assert_identical_denominator,
    assert_non_circular_teacher_eval,
    build_all_charts_and_manifest,
    compute_paired_panel_stats,
    verify_png_chunks_clean,
)


def test_identical_denominator_lock_enforced():
    """Verify that any denominator mismatch across models on a panel raises AssertionError."""
    v_1025 = np.ones(1025, dtype=np.float32)
    v_1009 = np.ones(1009, dtype=np.float32)  # Dropped 16 'unknown' items
    v_2835 = np.ones(2835, dtype=np.float32)
    v_2825 = np.ones(2825, dtype=np.float32)  # Dropped 10 'unknown' items
    v_545 = np.ones(545, dtype=np.float32)
    v_849 = np.ones(849, dtype=np.float32)    # Mismatched audio language split

    assert assert_identical_denominator("ok_1025", {"ultra": v_1025, "gpt": v_1025}, expected_n=1025) == 1025

    with pytest.raises(AssertionError, match="Identical Denominator Lock violated"):
        assert_identical_denominator("bad_beyond_unknown_drop", {"ultra": v_1025, "gpt": v_1009})

    with pytest.raises(AssertionError, match="Identical Denominator Lock violated"):
        assert_identical_denominator("bad_oct15_unknown_drop", {"ultra": v_2835, "mimmo": v_2825})

    with pytest.raises(AssertionError, match="Identical Denominator Lock violated"):
        assert_identical_denominator("bad_audio_split_mix", {"ultra": v_545, "gpt": v_849})

    with pytest.raises(AssertionError, match="Expected N=636"):
        assert_identical_denominator("bad_expected_n", {"ultra": np.ones(650), "gpt": np.ones(650)}, expected_n=636)

    v_nan = v_1025.copy()
    v_nan[0] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        assert_identical_denominator("bad_nan", {"ultra": v_1025, "gpt": v_nan})


def test_anti_circularity_lock_enforced():
    """Verify that scoring a teacher model against its own circular majority vote raises AssertionError,
    and that support-restricted circularity inflation is separated from Manski lower-bound unanimity drop."""
    d = _load_vectors(BENCH_DIR / "canonical_eval_vectors.json")

    with pytest.raises(AssertionError, match="Anti-Circularity Lock violated"):
        assert_non_circular_teacher_eval("OpenAI GPT (Fast Tier)", eval_mode="circular_majority", self_vote_weight=0.333)

    with pytest.raises(AssertionError, match="Anti-Circularity Lock violated"):
        assert_non_circular_teacher_eval("Google Gemini Flash", eval_mode="leave_one_teacher_out", self_vote_weight=0.5)

    gpt_audit = assert_non_circular_teacher_eval(
        "OpenAI GPT (Fast Tier)",
        eval_mode="leave_one_teacher_out",
        self_vote_weight=0.0,
        noncirc_vec=d["beyond_cat11_sup_no_gpt_corr_gpt"],
        circ_vec=d["beyond_cat11_corr_gpt_circ"],
        loto_lower_bound_vec=d["beyond_cat11_corr_gpt_loto_lower_bound"],
    )
    assert gpt_audit["non_circular_support_n"] == 974
    assert gpt_audit["non_circular_support_pct"] == 97.84
    assert gpt_audit["circular_self_vote_pct"] == 97.95
    expected_gpt_inf = round(float(np.mean(d["beyond_cat11_corr_gpt_circ"]) * 100.0 - np.mean(d["beyond_cat11_sup_no_gpt_corr_gpt"]) * 100.0), 2)
    assert gpt_audit["circularity_inflation_pp"] == expected_gpt_inf
    assert gpt_audit["manski_lower_bound_n1025_pct"] == 92.98
    assert gpt_audit["manski_unanimity_drop_pp"] == 4.87

    gem_audit = assert_non_circular_teacher_eval(
        "Google Gemini Flash",
        eval_mode="leave_one_teacher_out",
        self_vote_weight=0.0,
        noncirc_vec=d["beyond_cat11_sup_no_gemini_corr_gemini"],
        circ_vec=d["beyond_cat11_corr_gemini_circ"],
        loto_lower_bound_vec=d["beyond_cat11_corr_gemini_loto_lower_bound"],
    )
    assert gem_audit["non_circular_support_n"] == 982
    assert gem_audit["non_circular_support_pct"] == 97.05
    assert gem_audit["circular_self_vote_pct"] == 97.17
    expected_gem_inf = round(float(np.mean(d["beyond_cat11_corr_gemini_circ"]) * 100.0 - np.mean(d["beyond_cat11_sup_no_gemini_corr_gemini"]) * 100.0), 2)
    assert gem_audit["circularity_inflation_pp"] == expected_gem_inf
    assert gem_audit["manski_lower_bound_n1025_pct"] == 92.98
    assert gem_audit["manski_unanimity_drop_pp"] == 4.07


def test_eval_vectors_privacy_and_dtypes():
    """Verify that canonical_eval_vectors.json contains only numeric int16/float32 arrays and zero strings."""
    npz_path = BENCH_DIR / "canonical_eval_vectors.json"
    assert npz_path.is_file()
    d = _load_vectors(npz_path)
    assert len(list(d.keys())) >= 140
    for k in list(d.keys()):
        arr = d[k]
        assert arr.dtype in (np.int16, np.float32), f"Array {k} has disallowed dtype {arr.dtype}"
        assert arr.ndim == 1, f"Array {k} must be 1D, got shape {arr.shape}"
        assert not np.isnan(arr).any(), f"Array {k} contains NaNs"


def test_paired_bootstrap_and_mcnemar_reproducibility():
    """Verify that paired bootstrap 95% CIs and McNemar p-values are 100% deterministic."""
    d = _load_vectors(BENCH_DIR / "canonical_eval_vectors.json")
    vecs = {
        "Armiger Ultra": (d["oct15_aud545_pred_ultra"] == d["oct15_aud545_y_en"]).astype(np.float32),
        "Armiger INT8": (d["oct15_aud545_pred_int8_hyb"] == d["oct15_aud545_y_en"]).astype(np.float32),
        "GuessIt": (d["oct15_aud545_pred_guessit"] == d["oct15_aud545_y_en"]).astype(np.float32),
        "OpenAI GPT (Fast Tier)": (d["oct15_aud545_pred_gpt"] == d["oct15_aud545_y_en"]).astype(np.float32),
    }
    res1 = compute_paired_panel_stats("test_aud545", vecs, ref_model="Armiger Ultra", expected_n=545)
    res2 = compute_paired_panel_stats("test_aud545", vecs, ref_model="Armiger Ultra", expected_n=545)
    assert res1 == res2
    for m, st in res1["models"].items():
        assert st["n"] == 545
        assert st["ci95_lo"] <= st["point_pct"] <= st["ci95_hi"]
        assert 0.0 <= st["p_value_vs_ref"] <= 1.0
    assert res1["models"]["Armiger Ultra"]["point_pct"] == 96.7
    assert res1["models"]["GuessIt"]["point_pct"] == 94.86
    assert res1["models"]["OpenAI GPT (Fast Tier)"]["point_pct"] == 94.68
    assert res1["models"]["OpenAI GPT (Fast Tier)"]["p_value_vs_ref"] == pytest.approx(0.1262895, abs=1e-5)


def test_manifest_sha256_and_png_chunks(tmp_path: pathlib.Path):
    """Verify CHARTS_MANIFEST.json provenance hashes, PNG chunk hygiene, and byte-exact determinism."""
    manifest_path = BENCH_DIR / "CHARTS_MANIFEST.json"
    assert manifest_path.is_file(), "CHARTS_MANIFEST.json missing"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["bootstrap_seed"] == BOOTSTRAP_SEED
    assert manifest["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
    for k, v in manifest["invariants_verified"].items():
        assert v is True, f"Invariant {k} not verified"

    # Verify all provenance SHA-256 hashes match disk
    for rel_path, expected_sha in manifest["provenance_sha256"].items():
        target = (BENCH_DIR / rel_path) if (BENCH_DIR / rel_path).is_file() else (REPO_ROOT / rel_path)
        assert target.is_file(), f"Provenance file missing: {rel_path}"
        actual_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"SHA-256 mismatch for {rel_path}: {actual_sha} != {expected_sha}"

    # Verify all 4 PNGs have only IHDR, IDAT, IEND chunks and RGB mode
    for chart_name, chart_meta in manifest["charts"].items():
        png_path = REPO_ROOT / "docs" / "assets" / chart_name
        assert png_path.is_file(), f"Chart missing: {png_path}"
        chunks = verify_png_chunks_clean(png_path)
        assert chunks[0] == "IHDR" and chunks[-1] == "IEND"
        assert set(chunks).issubset({"IHDR", "IDAT", "IEND"})

    # Re-run pipeline into tmp_path and verify statistical + PNG determinism
    tmp_assets = tmp_path / "assets"
    tmp_manifest = tmp_path / "CHARTS_MANIFEST.json"
    render_mode = bool(HAS_MATPLOTLIB and sys.platform == "darwin")
    re_manifest = build_all_charts_and_manifest(
        BENCH_DIR / "canonical_eval_vectors.json",
        tmp_assets,
        tmp_manifest,
        render_pngs=render_mode,
    )
    assert re_manifest["invariants_verified"] == manifest["invariants_verified"]
    assert re_manifest["anti_circularity_audit"] == manifest["anti_circularity_audit"]
    assert re_manifest["five_split_category_summary"] == manifest["five_split_category_summary"]
    for chart_name, chart_meta in manifest["charts"].items():
        assert re_manifest["charts"][chart_name] == chart_meta, (
            f"Non-deterministic manifest or PNG output for {chart_name}"
        )


def test_evaluation_claims_match_charts_manifest():
    """Verify evaluation.md contains canonical CHARTS_MANIFEST.json metrics and zero stale numbers."""
    readme_path = REPO_ROOT / "docs" / "evaluation.md"
    assert readme_path.is_file()
    text = readme_path.read_text(encoding="utf-8")

    # Canonical numbers that must appear in evaluation guide
    for required_token in ("94.85%", "95.83%", "95.06%", "96.70%", "96.83%", "93.07%", "97.84%", "97.05%"):
        assert required_token in text, f"Missing canonical metric {required_token} in {readme_path}"

    # Fresh non-teacher LLM numbers are allowed only because they are backed by an
    # on-disk receipt with full coverage (982/982 records per model).
    if "Claude Haiku" in text:
        receipt = REPO_ROOT / "benchmarks" / "receipts" / "llm_beyond_fresh_receipt.json"
        assert receipt.is_file(), "Claude Haiku claim in evaluation guide without llm_beyond_fresh_receipt.json"
        for required_token in ("92.45%", "47.67%", "llm_beyond_fresh_receipt.json"):
            assert required_token in text, f"Missing receipt-backed LLM metric {required_token} in {readme_path}"

    # Stale pre-audit numbers that must never reappear in evaluation guide
    for stale_token in ("99.95%", "98.48%", "97.13%", "72.35%", "35.29%", "97.55%", "Vinci-MLE", "true" + "spec", "tru" + "spec", "tc" + "2000", "multilang" + "3000"):
        assert stale_token not in text, f"Stale metric {stale_token} found in {readme_path}"
