#!/usr/bin/env python3
"""
Build `benchmarks/canonical_eval_vectors.json` deterministically and exclusively
from the verified on-disk evaluation receipts in `benchmarks/receipts/*.json`:

  1. `armiger_v23_eval_receipt.json`
  2. `beyond_confirmation_teachers_receipt.json`
  3. `suite_a_parsers_receipt.json`
  4. `swarm_oof_receipt.json`
  5. `hardware_and_api_cost_receipt.json`

Enforces:
  - Zero synthetic vectors and zero unverifiable LLM receipts (`Claude`, `Vinci`, and `swarm_llm_757`
    removed per Round-3 Condition 4; `swarm_pub_score_int8` removed per Condition 6).
  - Every `receipt_sha256` in `_meta.provenance` is the actual SHA-256 of the on-disk receipt file
    at `_meta.provenance[k]["receipt_path"]`.
  - Vendor-disjoint comparison keys (`comparison_key`) and support bounds (`score_on_support`)
    computed via `armiger_eval_v3.keys`; ties stay `-1` (`⊥`) in `*_key_*` arrays and are never
    assigned fractional or coin-flip credit.
  - Lower-bound vectors on full strata are explicitly named `*_loto_lower_bound` (never `_noncirc`),
    while support-restricted LOTO vectors are stored under `*_sup_no_gpt_*` and `*_sup_no_gemini_*`.
  - Dense `int16` release-family cluster IDs (`*_family_ids`) shipped for every cohort to drive
    family-clustered bootstrap resampling in `generate_canonical_charts.py`.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Set

import numpy as np

BENCH_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RECEIPTS_DIR = BENCH_DIR / "receipts"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from armiger_eval_v3.estimators import release_family_id
from armiger_eval_v3.keys import comparison_key, score_on_support
from armiger_eval_v3.ledger import commit_stratum

CATS = [
    "movie", "tv_show", "xxx", "music", "comic",
    "game", "audiobook", "software", "ebook", "course", "unknown",
]
CAT_TO_IDX = {c: i for i, c in enumerate(CATS)}

PROVIDER_TO_VENDOR = {
    "muse-spark-1.3": "meta",
    "gpt-6-luna": "openai",
    "gemini-3.8-flash-low": "google",
}

MODEL_VENDOR = {
    "ultra": "armiger",
    "int8_hyb": "armiger",
    "int8_raw": "armiger",
    "int8": "armiger",
    "gpt": "openai",
    "gemini": "google",
    "muse": "meta",
    "mimmo_bert": "mimmo",
    "mimmo_cascade": "mimmo",
    "rtn": "open_source_parser",
    "ptt": "open_source_parser",
    "guessit": "open_source_parser",
    "ptn": "open_source_parser",
    "anitomy": "open_source_parser",
    "bitmagnet": "open_source_parser",
    "priors": "armiger",
    "seeders": "telemetry_baseline",
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_family_ids(raw_fams: List[int] | np.ndarray) -> np.ndarray:
    _, inv = np.unique(np.asarray(raw_fams, dtype=np.int64), return_inverse=True)
    return inv.astype(np.int16)


def key_digest(arr: np.ndarray, label: str) -> str:
    return hashlib.sha256(np.asarray(arr, dtype=np.int16).tobytes() + label.encode("utf-8")).hexdigest()


def build_vectors() -> Dict[str, Any]:
    r_files = {
        "v23": RECEIPTS_DIR / "armiger_v23_eval_receipt.json",
        "tchr": RECEIPTS_DIR / "beyond_confirmation_teachers_receipt.json",
        "sa": RECEIPTS_DIR / "suite_a_parsers_receipt.json",
        "sw_oof": RECEIPTS_DIR / "swarm_oof_receipt.json",
        "cost": RECEIPTS_DIR / "hardware_and_api_cost_receipt.json",
    }
    r_shas = {k: sha256_file(p) for k, p in r_files.items()}
    r_rels = {k: f"benchmarks/receipts/{p.name}" for k, p in r_files.items()}

    v23 = json.loads(r_files["v23"].read_text(encoding="utf-8"))
    tchr = json.loads(r_files["tchr"].read_text(encoding="utf-8"))
    sa = json.loads(r_files["sa"].read_text(encoding="utf-8"))
    sw_oof = json.loads(r_files["sw_oof"].read_text(encoding="utf-8"))

    bc_v23 = v23["splits"]["beyond_confirmation"]
    bc_cids = bc_v23["case_ids"]
    bc_fams_1046 = np.array(bc_v23["family_ids"], dtype=np.int64)

    gpt_map = tchr["teachers"]["gpt"]
    gem_map = tchr["teachers"]["google"]
    muse_map = tchr["teachers"]["muse"]

    # Verify completeness lock on teacher receipts via commit_stratum (recording bc-00825 abstention in gpt-6-luna)
    gpt_norm_1046 = {
        cid: gpt_map[cid] if cid in gpt_map else {"status": "abstain", "category": "⊥"}
        for cid in bc_cids
    }
    commit_stratum("gpt-6-luna", gpt_norm_1046, n_stratum=1046, receipt_path=r_rels["tchr"], receipt_sha256=r_shas["tchr"])
    commit_stratum("gemini-3.8-flash-low", gem_map, n_stratum=1046, receipt_path=r_rels["tchr"], receipt_sha256=r_shas["tchr"])
    commit_stratum("muse-spark-1.3", muse_map, n_stratum=1046, receipt_path=r_rels["tchr"], receipt_sha256=r_shas["tchr"])

    vectors: Dict[str, List[Any]] = {}
    dtypes: Dict[str, str] = {}
    provenance: Dict[str, Dict[str, Any]] = {}

    def add_vec(
        name: str,
        arr: np.ndarray,
        dtype_str: str,
        r_key: str,
        source: str,
        model_tag: str,
        key_pool_vendors: Set[str],
        ref_key_sha: str,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        np_dt = np.int16 if dtype_str == "int16" else np.float32
        cast_arr = np.asarray(arr, dtype=np_dt)
        if cast_arr.ndim != 1 or np.isnan(cast_arr).any():
            raise AssertionError(f"Invalid vector {name}: shape={cast_arr.shape}")
        if dtype_str == "int16":
            vectors[name] = [int(x) for x in cast_arr.tolist()]
        else:
            vectors[name] = [round(float(x), 6) for x in cast_arr.tolist()]
        dtypes[name] = dtype_str

        m_vendor = MODEL_VENDOR.get(model_tag, "reference")
        pool_sorted = sorted(key_pool_vendors)
        if source == "reference_key":
            self_vote_w = 0.0
            eval_mode = "reference_key"
        elif m_vendor in key_pool_vendors and len(key_pool_vendors) > 0:
            self_vote_w = round(1.0 / len(key_pool_vendors), 6)
            eval_mode = "circular_majority"
        elif m_vendor in {"openai", "google", "meta"}:
            self_vote_w = 0.0
            eval_mode = "leave_one_teacher_out"
        else:
            self_vote_w = 0.0
            eval_mode = "independent_ground_truth"

        prov: Dict[str, Any] = {
            "source": source,
            "receipt_path": r_rels[r_key],
            "receipt_sha256": r_shas[r_key],
            "n_stratum": int(cast_arr.size),
            "n_evaluated": int(cast_arr.size),
            "n_abstain": 0,
            "model_vendor": m_vendor,
            "key_pool_vendors": pool_sorted,
            "eval_mode": eval_mode,
            "self_vote_weight": self_vote_w,
            "key_sha256": ref_key_sha,
        }
        if extra:
            prov.update(extra)
        provenance[name] = prov

    # ==========================================================================
    # 1. BEYOND_CONFIRMATION (N=1046 raw -> N=1025 cat11, N=229 aud, N=636 ani, N=977 junk)
    # ==========================================================================
    ref_providers = set()
    for r_obj in tchr["refs"]:
        for p_name in r_obj.get("provenance", {}).get("providers", []):
            if p_name in PROVIDER_TO_VENDOR:
                ref_providers.add(PROVIDER_TO_VENDOR[p_name])

    y_bc_cat_1046 = np.array(bc_v23["y"]["category"], dtype=np.int16)
    y_bc_lang_1046 = np.array(bc_v23["y"]["lang"], dtype=np.int16)
    y_bc_ani_1046 = np.array(bc_v23["y"]["anime"], dtype=np.int16)
    y_bc_jk_1046 = np.array(bc_v23["y"]["junk"], dtype=np.int16)

    m_bc_cat = y_bc_cat_1046 >= 0
    m_bc_aud = (y_bc_lang_1046 == 0) | (y_bc_lang_1046 == 1)
    m_bc_ani = y_bc_ani_1046 >= 0
    m_bc_jk = y_bc_jk_1046 >= 0

    def extract_cat(by_id: Dict[str, Any], apply_adult_override: bool = True) -> np.ndarray:
        out = []
        for cid in bc_cids:
            r = by_id.get(cid)
            if not r:
                out.append(-1)
                continue
            c = str(r.get("category") or "unknown").lower().strip()
            if apply_adult_override and r.get("adult") in ("yes", True, "adult", "xxx"):
                c = "xxx"
            out.append(CAT_TO_IDX.get(c, 10))
        return np.array(out, dtype=np.int16)

    def extract_tv(by_id: Dict[str, Any]) -> np.ndarray:
        return np.array([1 if by_id.get(cid, {}).get("title_valid", False) else 0 for cid in bc_cids], dtype=np.int16)

    def extract_aud_en(by_id: Dict[str, Any]) -> np.ndarray:
        out = []
        for cid in bc_cids:
            r = by_id.get(cid, {})
            ac = str(r.get("audio_claim", "")).lower().strip()
            tl = str(r.get("title_language", "")).lower().strip()
            out.append(0 if (ac == "foreign" or tl == "foreign") else 1)
        return np.array(out, dtype=np.int16)

    def extract_ani(by_id: Dict[str, Any]) -> np.ndarray:
        out = []
        for cid in bc_cids:
            r = by_id.get(cid, {})
            v = r.get("anime_heur", r.get("anime", False))
            out.append(1 if v in (True, "yes", 1, "anime") else 0)
        return np.array(out, dtype=np.int16)

    def extract_jk(by_id: Dict[str, Any]) -> np.ndarray:
        out = []
        for cid in bc_cids:
            r = by_id.get(cid, {})
            v = str(r.get("junk", "no")).lower().strip()
            out.append(1 if v in ("yes", "true", "1", "junk") else 0)
        return np.array(out, dtype=np.int16)

    u_cat_1046 = np.array(bc_v23["fp32"]["hyb_cat"], dtype=np.int16)
    i_hyb_1046 = np.array(bc_v23["int8"]["hyb_cat"], dtype=np.int16)
    i_raw_1046 = np.array(bc_v23["int8"]["raw_cat"], dtype=np.int16)
    u_tv_1046 = np.array(bc_v23["fp32"]["title_valid"], dtype=np.int16)
    i_tv_1046 = np.array(bc_v23["int8"]["title_valid"], dtype=np.int16)

    gpt_cat_1025 = extract_cat(gpt_map, apply_adult_override=True)[m_bc_cat]
    gem_cat_1025 = extract_cat(gem_map, apply_adult_override=True)[m_bc_cat]
    muse_cat_1025 = extract_cat(muse_map, apply_adult_override=True)[m_bc_cat]
    u_cat_1025 = u_cat_1046[m_bc_cat]
    i_hyb_1025 = i_hyb_1046[m_bc_cat]
    i_raw_1025 = i_raw_1046[m_bc_cat]

    t_preds_cat = {"gpt": gpt_cat_1025, "gemini": gem_cat_1025, "muse": muse_cat_1025}
    vendors_3t = {"gpt": "openai", "gemini": "google", "muse": "meta"}

    k_cat_all3, sha_cat_all3 = comparison_key(t_preds_cat, vendors_3t, scored={"ultra", "int8"})
    k_cat_no_gpt, sha_cat_no_gpt = comparison_key(t_preds_cat, vendors_3t, scored={"ultra", "int8", "gpt"})
    k_cat_no_gem, sha_cat_no_gem = comparison_key(t_preds_cat, vendors_3t, scored={"ultra", "int8", "gemini"})

    assert np.array_equal(k_cat_all3, y_bc_cat_1046[m_bc_cat]), "3-teacher consensus key must match beyond_cat11_y_true"

    fams_cat_1025 = dense_family_ids(bc_fams_1046[m_bc_cat])
    add_vec("beyond_cat11_y_true", k_cat_all3, "int16", "tchr", "reference_key", "ref", ref_providers, sha_cat_all3)
    add_vec("beyond_cat11_key_no_gpt", k_cat_no_gpt, "int16", "tchr", "reference_key", "ref", {"google", "meta"}, sha_cat_no_gpt)
    add_vec("beyond_cat11_key_no_gemini", k_cat_no_gem, "int16", "tchr", "reference_key", "ref", {"openai", "meta"}, sha_cat_no_gem)
    add_vec("beyond_cat11_family_ids", fams_cat_1025, "int16", "v23", "reference_key", "ref", set(), sha_cat_all3)

    add_vec("beyond_cat11_pred_ultra", u_cat_1025, "int16", "v23", "model_output", "ultra", ref_providers, sha_cat_all3)
    add_vec("beyond_cat11_pred_int8_hyb", i_hyb_1025, "int16", "v23", "model_output", "int8_hyb", ref_providers, sha_cat_all3)
    add_vec("beyond_cat11_pred_int8_raw", i_raw_1025, "int16", "v23", "model_output", "int8_raw", ref_providers, sha_cat_all3)
    add_vec("beyond_cat11_pred_gpt", gpt_cat_1025, "int16", "tchr", "model_output", "gpt", {"google", "meta"}, sha_cat_no_gpt, {"n_abstain": int((gpt_cat_1025 < 0).sum())})
    add_vec("beyond_cat11_pred_gemini", gem_cat_1025, "int16", "tchr", "model_output", "gemini", {"openai", "meta"}, sha_cat_no_gem)

    add_vec("beyond_cat11_corr_ultra", (u_cat_1025 == k_cat_all3).astype(np.float32), "float32", "v23", "model_output", "ultra", ref_providers, sha_cat_all3, score_on_support(u_cat_1025, k_cat_all3))
    add_vec("beyond_cat11_corr_int8_hyb", (i_hyb_1025 == k_cat_all3).astype(np.float32), "float32", "v23", "model_output", "int8_hyb", ref_providers, sha_cat_all3, score_on_support(i_hyb_1025, k_cat_all3))
    add_vec("beyond_cat11_corr_int8_raw", (i_raw_1025 == k_cat_all3).astype(np.float32), "float32", "v23", "model_output", "int8_raw", ref_providers, sha_cat_all3, score_on_support(i_raw_1025, k_cat_all3))

    # Support-restricted LOTO vectors on N=974 (vs GPT) and N=982 (vs Gemini)
    sup_cat_ng = k_cat_no_gpt >= 0
    sup_cat_nm = k_cat_no_gem >= 0
    add_vec("beyond_cat11_sup_no_gpt_family_ids", dense_family_ids(bc_fams_1046[m_bc_cat][sup_cat_ng]), "int16", "v23", "reference_key", "ref", {"google", "meta"}, sha_cat_no_gpt)
    add_vec("beyond_cat11_sup_no_gpt_corr_ultra", (u_cat_1025[sup_cat_ng] == k_cat_no_gpt[sup_cat_ng]).astype(np.float32), "float32", "v23", "model_output", "ultra", {"google", "meta"}, sha_cat_no_gpt)
    add_vec("beyond_cat11_sup_no_gpt_corr_gpt", (gpt_cat_1025[sup_cat_ng] == k_cat_no_gpt[sup_cat_ng]).astype(np.float32), "float32", "tchr", "model_output", "gpt", {"google", "meta"}, sha_cat_no_gpt, {"n_abstain": int((gpt_cat_1025[sup_cat_ng] < 0).sum())})

    add_vec("beyond_cat11_sup_no_gemini_family_ids", dense_family_ids(bc_fams_1046[m_bc_cat][sup_cat_nm]), "int16", "v23", "reference_key", "ref", {"openai", "meta"}, sha_cat_no_gem)
    add_vec("beyond_cat11_sup_no_gemini_corr_ultra", (u_cat_1025[sup_cat_nm] == k_cat_no_gem[sup_cat_nm]).astype(np.float32), "float32", "v23", "model_output", "ultra", {"openai", "meta"}, sha_cat_no_gem)
    add_vec("beyond_cat11_sup_no_gemini_corr_gemini", (gem_cat_1025[sup_cat_nm] == k_cat_no_gem[sup_cat_nm]).astype(np.float32), "float32", "tchr", "model_output", "gemini", {"openai", "meta"}, sha_cat_no_gem)

    # Manski lower-bound vectors on N=1025 (explicitly named *_loto_lower_bound, never _noncirc)
    add_vec(
        "beyond_cat11_corr_gpt_loto_lower_bound",
        ((k_cat_no_gpt >= 0) & (gpt_cat_1025 == k_cat_no_gpt)).astype(np.float32),
        "float32", "tchr", "model_output", "gpt", {"google", "meta"}, sha_cat_no_gpt,
        {**score_on_support(gpt_cat_1025, k_cat_no_gpt), "n_abstain": int((gpt_cat_1025 < 0).sum())},
    )
    add_vec(
        "beyond_cat11_corr_gpt_circ",
        (gpt_cat_1025 == k_cat_all3).astype(np.float32),
        "float32", "tchr", "model_output", "gpt", ref_providers, sha_cat_all3,
        {"n_abstain": int((gpt_cat_1025 < 0).sum())},
    )
    add_vec(
        "beyond_cat11_corr_gemini_loto_lower_bound",
        ((k_cat_no_gem >= 0) & (gem_cat_1025 == k_cat_no_gem)).astype(np.float32),
        "float32", "tchr", "model_output", "gemini", {"openai", "meta"}, sha_cat_no_gem,
        score_on_support(gem_cat_1025, k_cat_no_gem),
    )
    add_vec(
        "beyond_cat11_corr_gemini_circ",
        (gem_cat_1025 == k_cat_all3).astype(np.float32),
        "float32", "tchr", "model_output", "gemini", ref_providers, sha_cat_all3,
    )

    # 2B. Catalog Matching (category correct AND title_valid == 1)
    u_tv_1025 = u_tv_1046[m_bc_cat]
    i_tv_1025 = i_tv_1046[m_bc_cat]
    gpt_tv_1025 = extract_tv(gpt_map)[m_bc_cat]
    gem_tv_1025 = extract_tv(gem_map)[m_bc_cat]

    add_vec("beyond_match_corr_ultra", ((u_cat_1025 == k_cat_all3) & (u_tv_1025 == 1)).astype(np.float32), "float32", "v23", "model_output", "ultra", ref_providers, sha_cat_all3)
    add_vec("beyond_match_corr_int8_hyb", ((i_hyb_1025 == k_cat_all3) & (i_tv_1025 == 1)).astype(np.float32), "float32", "v23", "model_output", "int8_hyb", ref_providers, sha_cat_all3)
    add_vec("beyond_match_corr_int8_raw", ((i_raw_1025 == k_cat_all3) & (i_tv_1025 == 1)).astype(np.float32), "float32", "v23", "model_output", "int8_raw", ref_providers, sha_cat_all3)
    add_vec("beyond_match_sup_no_gpt_corr_ultra", ((u_cat_1025[sup_cat_ng] == k_cat_no_gpt[sup_cat_ng]) & (u_tv_1025[sup_cat_ng] == 1)).astype(np.float32), "float32", "v23", "model_output", "ultra", {"google", "meta"}, sha_cat_no_gpt)
    add_vec("beyond_match_sup_no_gpt_corr_gpt", ((gpt_cat_1025[sup_cat_ng] == k_cat_no_gpt[sup_cat_ng]) & (gpt_tv_1025[sup_cat_ng] == 1)).astype(np.float32), "float32", "tchr", "model_output", "gpt", {"google", "meta"}, sha_cat_no_gpt, {"n_abstain": int((gpt_cat_1025[sup_cat_ng] < 0).sum())})
    add_vec("beyond_match_sup_no_gemini_corr_ultra", ((u_cat_1025[sup_cat_nm] == k_cat_no_gem[sup_cat_nm]) & (u_tv_1025[sup_cat_nm] == 1)).astype(np.float32), "float32", "v23", "model_output", "ultra", {"openai", "meta"}, sha_cat_no_gem)
    add_vec("beyond_match_sup_no_gemini_corr_gemini", ((gem_cat_1025[sup_cat_nm] == k_cat_no_gem[sup_cat_nm]) & (gem_tv_1025[sup_cat_nm] == 1)).astype(np.float32), "float32", "tchr", "model_output", "gemini", {"openai", "meta"}, sha_cat_no_gem)

    # 2C. Audio Language Strict E-vs-F (N=229)
    y_aud_229 = (y_bc_lang_1046[m_bc_aud] == 0).astype(np.int16)
    u_aud_229 = (np.array(bc_v23["fp32"]["aud_guard_en_prob"])[m_bc_aud] >= 0.5).astype(np.int16)
    i_aud_hyb_229 = (np.array(bc_v23["int8"]["aud_guard_en_prob"])[m_bc_aud] >= 0.5).astype(np.int16)
    i_aud_raw_229 = (np.array(bc_v23["int8"]["en_prob"])[m_bc_aud] >= 0.5).astype(np.int16)
    gpt_aud_229 = extract_aud_en(gpt_map)[m_bc_aud]
    gem_aud_229 = extract_aud_en(gem_map)[m_bc_aud]
    muse_aud_229 = extract_aud_en(muse_map)[m_bc_aud]

    t_preds_aud = {"gpt": gpt_aud_229, "gemini": gem_aud_229, "muse": muse_aud_229}
    k_aud_no_gpt, sha_aud_no_gpt = comparison_key(t_preds_aud, vendors_3t, scored={"ultra", "int8", "gpt"})
    k_aud_no_gem, sha_aud_no_gem = comparison_key(t_preds_aud, vendors_3t, scored={"ultra", "int8", "gemini"})
    sha_aud_ref = key_digest(y_aud_229, "beyond_aud229_y_en")
    sup_aud_ng = k_aud_no_gpt >= 0
    sup_aud_nm = k_aud_no_gem >= 0

    add_vec("beyond_aud229_y_en", y_aud_229, "int16", "v23", "reference_key", "ref", ref_providers, sha_aud_ref)
    add_vec("beyond_aud229_key_no_gpt", k_aud_no_gpt, "int16", "tchr", "reference_key", "ref", {"google", "meta"}, sha_aud_no_gpt)
    add_vec("beyond_aud229_key_no_gemini", k_aud_no_gem, "int16", "tchr", "reference_key", "ref", {"openai", "meta"}, sha_aud_no_gem)
    add_vec("beyond_aud229_family_ids", dense_family_ids(bc_fams_1046[m_bc_aud]), "int16", "v23", "reference_key", "ref", set(), sha_aud_ref)

    add_vec("beyond_aud229_corr_ultra", (u_aud_229 == y_aud_229).astype(np.float32), "float32", "v23", "model_output", "ultra", ref_providers, sha_aud_ref)
    add_vec("beyond_aud229_corr_int8_hyb", (i_aud_hyb_229 == y_aud_229).astype(np.float32), "float32", "v23", "model_output", "int8_hyb", ref_providers, sha_aud_ref)
    add_vec("beyond_aud229_corr_int8_raw", (i_aud_raw_229 == y_aud_229).astype(np.float32), "float32", "v23", "model_output", "int8_raw", ref_providers, sha_aud_ref)
    add_vec("beyond_aud229_sup_no_gpt_family_ids", dense_family_ids(bc_fams_1046[m_bc_aud][sup_aud_ng]), "int16", "v23", "reference_key", "ref", {"google", "meta"}, sha_aud_no_gpt)
    add_vec("beyond_aud229_sup_no_gpt_corr_ultra", (u_aud_229[sup_aud_ng] == k_aud_no_gpt[sup_aud_ng]).astype(np.float32), "float32", "v23", "model_output", "ultra", {"google", "meta"}, sha_aud_no_gpt)
    add_vec("beyond_aud229_sup_no_gpt_corr_gpt", (gpt_aud_229[sup_aud_ng] == k_aud_no_gpt[sup_aud_ng]).astype(np.float32), "float32", "tchr", "model_output", "gpt", {"google", "meta"}, sha_aud_no_gpt, score_on_support(gpt_aud_229, k_aud_no_gpt))
    add_vec("beyond_aud229_sup_no_gemini_family_ids", dense_family_ids(bc_fams_1046[m_bc_aud][sup_aud_nm]), "int16", "v23", "reference_key", "ref", {"openai", "meta"}, sha_aud_no_gem)
    add_vec("beyond_aud229_sup_no_gemini_corr_ultra", (u_aud_229[sup_aud_nm] == k_aud_no_gem[sup_aud_nm]).astype(np.float32), "float32", "v23", "model_output", "ultra", {"openai", "meta"}, sha_aud_no_gem)
    add_vec("beyond_aud229_sup_no_gemini_corr_gemini", (gem_aud_229[sup_aud_nm] == k_aud_no_gem[sup_aud_nm]).astype(np.float32), "float32", "tchr", "model_output", "gemini", {"openai", "meta"}, sha_aud_no_gem, score_on_support(gem_aud_229, k_aud_no_gem))

    # 2D. Anime Detection (N=636)
    y_ani_636 = y_bc_ani_1046[m_bc_ani]
    u_ani_636 = (np.array(bc_v23["fp32"]["anime_prob"])[m_bc_ani] >= 0.5).astype(np.int16)
    i_ani_636 = (np.array(bc_v23["int8"]["anime_prob"])[m_bc_ani] >= 0.5).astype(np.int16)
    gpt_ani_636 = extract_ani(gpt_map)[m_bc_ani]
    gem_ani_636 = extract_ani(gem_map)[m_bc_ani]
    muse_ani_636 = extract_ani(muse_map)[m_bc_ani]

    t_preds_ani = {"gpt": gpt_ani_636, "gemini": gem_ani_636, "muse": muse_ani_636}
    k_ani_no_gpt, sha_ani_no_gpt = comparison_key(t_preds_ani, vendors_3t, scored={"ultra", "int8", "gpt"})
    k_ani_no_gem, sha_ani_no_gem = comparison_key(t_preds_ani, vendors_3t, scored={"ultra", "int8", "gemini"})
    sha_ani_ref = key_digest(y_ani_636, "beyond_ani636_y_true")
    sup_ani_ng = k_ani_no_gpt >= 0
    sup_ani_nm = k_ani_no_gem >= 0

    add_vec("beyond_ani636_y_true", y_ani_636, "int16", "v23", "reference_key", "ref", ref_providers, sha_ani_ref)
    add_vec("beyond_ani636_key_no_gpt", k_ani_no_gpt, "int16", "tchr", "reference_key", "ref", {"google", "meta"}, sha_ani_no_gpt)
    add_vec("beyond_ani636_key_no_gemini", k_ani_no_gem, "int16", "tchr", "reference_key", "ref", {"openai", "meta"}, sha_ani_no_gem)
    add_vec("beyond_ani636_family_ids", dense_family_ids(bc_fams_1046[m_bc_ani]), "int16", "v23", "reference_key", "ref", set(), sha_ani_ref)
    add_vec("beyond_ani636_corr_ultra", (u_ani_636 == y_ani_636).astype(np.float32), "float32", "v23", "model_output", "ultra", ref_providers, sha_ani_ref)
    add_vec("beyond_ani636_corr_int8", (i_ani_636 == y_ani_636).astype(np.float32), "float32", "v23", "model_output", "int8", ref_providers, sha_ani_ref)
    add_vec("beyond_ani636_sup_no_gpt_family_ids", dense_family_ids(bc_fams_1046[m_bc_ani][sup_ani_ng]), "int16", "v23", "reference_key", "ref", {"google", "meta"}, sha_ani_no_gpt)
    add_vec("beyond_ani636_sup_no_gpt_corr_ultra", (u_ani_636[sup_ani_ng] == k_ani_no_gpt[sup_ani_ng]).astype(np.float32), "float32", "v23", "model_output", "ultra", {"google", "meta"}, sha_ani_no_gpt)
    add_vec("beyond_ani636_sup_no_gpt_corr_gpt", (gpt_ani_636[sup_ani_ng] == k_ani_no_gpt[sup_ani_ng]).astype(np.float32), "float32", "tchr", "model_output", "gpt", {"google", "meta"}, sha_ani_no_gpt, score_on_support(gpt_ani_636, k_ani_no_gpt))
    add_vec("beyond_ani636_sup_no_gemini_family_ids", dense_family_ids(bc_fams_1046[m_bc_ani][sup_ani_nm]), "int16", "v23", "reference_key", "ref", {"openai", "meta"}, sha_ani_no_gem)
    add_vec("beyond_ani636_sup_no_gemini_corr_ultra", (u_ani_636[sup_ani_nm] == k_ani_no_gem[sup_ani_nm]).astype(np.float32), "float32", "v23", "model_output", "ultra", {"openai", "meta"}, sha_ani_no_gem)
    add_vec("beyond_ani636_sup_no_gemini_corr_gemini", (gem_ani_636[sup_ani_nm] == k_ani_no_gem[sup_ani_nm]).astype(np.float32), "float32", "tchr", "model_output", "gemini", {"openai", "meta"}, sha_ani_no_gem, score_on_support(gem_ani_636, k_ani_no_gem))

    # 2E. Junk / Malware Detection (N=977 3-teacher key + N=975 LOTO key_no_gpt)
    y_jk_977 = y_bc_jk_1046[m_bc_jk]
    u_jk_977 = (np.array(bc_v23["fp32"]["junk_prob"])[m_bc_jk] >= 0.5).astype(np.int16)
    i_jk_977 = (np.array(bc_v23["int8"]["junk_prob"])[m_bc_jk] >= 0.5).astype(np.int16)
    gpt_jk_977 = extract_jk(gpt_map)[m_bc_jk]
    gem_jk_977 = extract_jk(gem_map)[m_bc_jk]
    muse_jk_977 = extract_jk(muse_map)[m_bc_jk]

    t_preds_jk = {"gpt": gpt_jk_977, "gemini": gem_jk_977, "muse": muse_jk_977}
    k_jk_no_gpt, sha_jk_no_gpt = comparison_key(t_preds_jk, vendors_3t, scored={"ultra", "int8", "gpt"})
    sup_jk_ng = k_jk_no_gpt >= 0
    sha_jk_ref = key_digest(y_jk_977, "beyond_junk977_y_true")

    add_vec("beyond_junk977_y_true", y_jk_977, "int16", "v23", "reference_key", "ref", ref_providers, sha_jk_ref)
    add_vec("beyond_junk977_key_no_gpt", k_jk_no_gpt, "int16", "tchr", "reference_key", "ref", {"google", "meta"}, sha_jk_no_gpt)
    add_vec("beyond_junk977_family_ids", dense_family_ids(bc_fams_1046[m_bc_jk]), "int16", "v23", "reference_key", "ref", set(), sha_jk_ref)
    add_vec("beyond_junk977_pred_ultra", u_jk_977, "int16", "v23", "model_output", "ultra", ref_providers, sha_jk_ref)
    add_vec("beyond_junk977_pred_int8", i_jk_977, "int16", "v23", "model_output", "int8", ref_providers, sha_jk_ref)
    # Full-977 teacher junk predictions are circular against the 3-teacher key (Condition 5/6 fix)
    add_vec("beyond_junk977_pred_gpt_circ", gpt_jk_977, "int16", "tchr", "model_output", "gpt", ref_providers, sha_jk_ref)
    add_vec("beyond_junk977_pred_gemini_circ", gem_jk_977, "int16", "tchr", "model_output", "gemini", ref_providers, sha_jk_ref)
    # Non-circular LOTO junk comparison on {google, meta} support (N=975, n_pos=64, n_neg=911)
    add_vec("beyond_junk977_sup_no_gpt_y_true", k_jk_no_gpt[sup_jk_ng], "int16", "tchr", "reference_key", "ref", {"google", "meta"}, sha_jk_no_gpt)
    add_vec("beyond_junk977_sup_no_gpt_family_ids", dense_family_ids(bc_fams_1046[m_bc_jk][sup_jk_ng]), "int16", "v23", "reference_key", "ref", {"google", "meta"}, sha_jk_no_gpt)
    add_vec("beyond_junk977_sup_no_gpt_pred_ultra", u_jk_977[sup_jk_ng], "int16", "v23", "model_output", "ultra", {"google", "meta"}, sha_jk_no_gpt)
    add_vec("beyond_junk977_sup_no_gpt_pred_gpt", gpt_jk_977[sup_jk_ng], "int16", "tchr", "model_output", "gpt", {"google", "meta"}, sha_jk_no_gpt)

    # ==========================================================================
    # 2. OCT15_CONFIRMATION & TC_TRUESPEC_2000 (SUITE A)
    # ==========================================================================
    oct_v23 = v23["splits"]["oct15_confirmation"]
    oct_cids = oct_v23["case_ids"]
    oct_fams_all = np.array(oct_v23["family_ids"], dtype=np.int64)
    sa_oct_items = sa["oct15_confirmation"]["items"]

    y_oct_cat_all = np.array(oct_v23["y"]["category"], dtype=np.int16)
    y_oct_ad_all = np.array(oct_v23["y"]["adult"], dtype=np.int16)
    y_oct_ani_all = np.array(oct_v23["y"]["anime"], dtype=np.int16)
    y_oct_lang_all = np.array(oct_v23["y"]["lang"], dtype=np.int16)
    y_oct_jk_all = np.array(oct_v23["y"]["junk"], dtype=np.int16)

    m_oct_cat = y_oct_cat_all >= 0                        # 2835
    m_oct_vid = (y_oct_cat_all == 0) | (y_oct_cat_all == 1)  # 1296
    m_oct_aud = (y_oct_lang_all == 0) | (y_oct_lang_all == 1)  # 545
    m_oct_ani = y_oct_ani_all >= 0                        # 507
    m_oct_ad = y_oct_ad_all >= 0                          # 2781
    m_oct_jk = y_oct_jk_all >= 0                          # 2170 (n_pos=1, n_neg=2169)

    u_oct_hyb = np.array(oct_v23["fp32"]["hyb_cat"], dtype=np.int16)
    i_oct_hyb = np.array(oct_v23["int8"]["hyb_cat"], dtype=np.int16)
    i_oct_raw = np.array(oct_v23["int8"]["raw_cat"], dtype=np.int16)

    # 2.1 oct15_cat11 (N=2835)
    y_oct_cat = y_oct_cat_all[m_oct_cat]
    sha_oct_cat = key_digest(y_oct_cat, "oct15_cat11_y_true")
    add_vec("oct15_cat11_y_true", y_oct_cat, "int16", "v23", "reference_key", "ref", set(), sha_oct_cat)
    add_vec("oct15_cat11_family_ids", dense_family_ids(oct_fams_all[m_oct_cat]), "int16", "v23", "reference_key", "ref", set(), sha_oct_cat)
    add_vec("oct15_cat11_pred_ultra", u_oct_hyb[m_oct_cat], "int16", "v23", "model_output", "ultra", set(), sha_oct_cat)
    add_vec("oct15_cat11_pred_int8_hyb", i_oct_hyb[m_oct_cat], "int16", "v23", "model_output", "int8_hyb", set(), sha_oct_cat)
    add_vec("oct15_cat11_pred_int8_raw", i_oct_raw[m_oct_cat], "int16", "v23", "model_output", "int8_raw", set(), sha_oct_cat)

    for p_tag in ["mimmo_bert", "mimmo_cascade", "rtn", "ptt", "guessit", "ptn", "anitomy", "bitmagnet", "gpt"]:
        arr_p = np.array([int(sa_oct_items[cid]["cat11"][p_tag]) for cid in oct_cids], dtype=np.int16)[m_oct_cat]
        add_vec(f"oct15_cat11_pred_{p_tag}", arr_p, "int16", "sa", "model_output", p_tag, set(), sha_oct_cat)

    # 2.2 oct15_vid (N=1296) & Catalog Matching
    y_oct_vid = y_oct_cat_all[m_oct_vid]
    sha_oct_vid = key_digest(y_oct_vid, "oct15_vid_y_true")
    add_vec("oct15_vid_y_true", y_oct_vid, "int16", "v23", "reference_key", "ref", set(), sha_oct_vid)
    add_vec("oct15_vid_family_ids", dense_family_ids(oct_fams_all[m_oct_vid]), "int16", "v23", "reference_key", "ref", set(), sha_oct_vid)
    add_vec("oct15_vid_mvtv_pred_ultra", u_oct_hyb[m_oct_vid], "int16", "v23", "model_output", "ultra", set(), sha_oct_vid)
    add_vec("oct15_vid_mvtv_pred_int8_hyb", i_oct_hyb[m_oct_vid], "int16", "v23", "model_output", "int8_hyb", set(), sha_oct_vid)
    add_vec("oct15_vid_mvtv_pred_int8_raw", i_oct_raw[m_oct_vid], "int16", "v23", "model_output", "int8_raw", set(), sha_oct_vid)

    arm_tv_vid = np.array([int(sa_oct_items[cid]["title_valid"]["armiger"]) for cid in oct_cids], dtype=np.int16)[m_oct_vid]
    add_vec("oct15_match_correct_ultra", ((u_oct_hyb[m_oct_vid] == y_oct_vid) & (arm_tv_vid == 1)).astype(np.int16), "int16", "v23", "model_output", "ultra", set(), sha_oct_vid)
    add_vec("oct15_match_correct_int8_hyb", ((i_oct_hyb[m_oct_vid] == y_oct_vid) & (arm_tv_vid == 1)).astype(np.int16), "int16", "v23", "model_output", "int8_hyb", set(), sha_oct_vid)
    add_vec("oct15_match_correct_int8_raw", ((i_oct_raw[m_oct_vid] == y_oct_vid) & (arm_tv_vid == 1)).astype(np.int16), "int16", "v23", "model_output", "int8_raw", set(), sha_oct_vid)

    tv_key_map = {
        "mimmo_bert": "mimmo_bert",
        "mimmo_cascade": "mimmo_cascade",
        "rtn": "rtn",
        "ptt": "ptt",
        "guessit": "guessit",
        "ptn": "ptn",
        "anitomy": "anitomy",
        "bitmagnet": "bitmagnet",
    }
    for p_tag, tv_k in tv_key_map.items():
        p_vid = np.array([int(sa_oct_items[cid].get("mvtv", sa_oct_items[cid]["cat11"])[p_tag]) for cid in oct_cids], dtype=np.int16)[m_oct_vid]
        tv_vid = np.array([int(sa_oct_items[cid]["title_valid"][tv_k]) for cid in oct_cids], dtype=np.int16)[m_oct_vid]
        add_vec(f"oct15_vid_mvtv_pred_{p_tag}", p_vid, "int16", "sa", "model_output", p_tag, set(), sha_oct_vid)
        add_vec(f"oct15_match_correct_{p_tag}", ((p_vid == y_oct_vid) & (tv_vid == 1)).astype(np.int16), "int16", "sa", "model_output", p_tag, set(), sha_oct_vid)

    # 2.3 oct15_aud545 (Strict English vs Foreign, N=545)
    y_oct_aud = (y_oct_lang_all[m_oct_aud] == 0).astype(np.int16)
    sha_oct_aud = key_digest(y_oct_aud, "oct15_aud545_y_en")
    u_oct_aud = (np.array(oct_v23["fp32"]["aud_guard_en_prob"])[m_oct_aud] >= 0.5).astype(np.int16)
    i_oct_aud_hyb = (np.array(oct_v23["int8"]["aud_guard_en_prob"])[m_oct_aud] >= 0.5).astype(np.int16)
    i_oct_aud_raw = (np.array(oct_v23["int8"]["en_prob"])[m_oct_aud] >= 0.5).astype(np.int16)

    add_vec("oct15_aud545_y_en", y_oct_aud, "int16", "v23", "reference_key", "ref", set(), sha_oct_aud)
    add_vec("oct15_aud545_family_ids", dense_family_ids(oct_fams_all[m_oct_aud]), "int16", "v23", "reference_key", "ref", set(), sha_oct_aud)
    add_vec("oct15_aud545_pred_ultra", u_oct_aud, "int16", "v23", "model_output", "ultra", set(), sha_oct_aud)
    add_vec("oct15_aud545_pred_int8_hyb", i_oct_aud_hyb, "int16", "v23", "model_output", "int8_hyb", set(), sha_oct_aud)
    add_vec("oct15_aud545_pred_int8_raw", i_oct_aud_raw, "int16", "v23", "model_output", "int8_raw", set(), sha_oct_aud)
    for p_tag in ["rtn", "ptt", "guessit", "ptn", "anitomy", "bitmagnet", "gpt"]:
        arr_a = np.array([int(sa_oct_items[cid]["aud_en"][p_tag]) for cid in oct_cids], dtype=np.int16)[m_oct_aud]
        add_vec(f"oct15_aud545_pred_{p_tag}", arr_a, "int16", "sa", "model_output", p_tag, set(), sha_oct_aud)

    # 2.4 oct15_ani507 (N=507)
    y_oct_ani = y_oct_ani_all[m_oct_ani]
    sha_oct_ani = key_digest(y_oct_ani, "oct15_ani507_y_true")
    u_oct_ani = (np.array(oct_v23["fp32"]["anime_prob"])[m_oct_ani] >= 0.5).astype(np.int16)
    i_oct_ani = (np.array(oct_v23["int8"]["anime_prob"])[m_oct_ani] >= 0.5).astype(np.int16)
    add_vec("oct15_ani507_y_true", y_oct_ani, "int16", "v23", "reference_key", "ref", set(), sha_oct_ani)
    add_vec("oct15_ani507_family_ids", dense_family_ids(oct_fams_all[m_oct_ani]), "int16", "v23", "reference_key", "ref", set(), sha_oct_ani)
    add_vec("oct15_ani507_pred_ultra", u_oct_ani, "int16", "v23", "model_output", "ultra", set(), sha_oct_ani)
    add_vec("oct15_ani507_pred_int8", i_oct_ani, "int16", "v23", "model_output", "int8", set(), sha_oct_ani)

    # 2.5 oct15_adult (N=2781)
    y_oct_ad = y_oct_ad_all[m_oct_ad]
    sha_oct_ad = key_digest(y_oct_ad, "oct15_adult_y_true")
    u_oct_ad = (np.array(oct_v23["fp32"]["adult_prob"])[m_oct_ad] >= 0.5).astype(np.int16)
    i_oct_ad = (np.array(oct_v23["int8"]["adult_prob"])[m_oct_ad] >= 0.5).astype(np.int16)
    add_vec("oct15_adult_y_true", y_oct_ad, "int16", "v23", "reference_key", "ref", set(), sha_oct_ad)
    add_vec("oct15_adult_family_ids", dense_family_ids(oct_fams_all[m_oct_ad]), "int16", "v23", "reference_key", "ref", set(), sha_oct_ad)
    add_vec("oct15_adult_pred_ultra", u_oct_ad, "int16", "v23", "model_output", "ultra", set(), sha_oct_ad)
    add_vec("oct15_adult_pred_int8", i_oct_ad, "int16", "v23", "model_output", "int8", set(), sha_oct_ad)
    for p_tag in ["mimmo_bert", "mimmo_cascade", "rtn", "ptt", "bitmagnet"]:
        arr_ad = np.array([int(sa_oct_items[cid]["adult"][p_tag]) for cid in oct_cids], dtype=np.int16)[m_oct_ad]
        add_vec(f"oct15_adult_pred_{p_tag}", arr_ad, "int16", "sa", "model_output", p_tag, set(), sha_oct_ad)

    # 2.6 oct15_junk (N=2170, n_pos=1, n_neg=2169: sensitivity not estimable; retained for specificity audit)
    y_oct_jk = y_oct_jk_all[m_oct_jk]
    sha_oct_jk = key_digest(y_oct_jk, "oct15_junk_y_true")
    u_oct_jk = (np.array(oct_v23["fp32"]["junk_prob"])[m_oct_jk] >= 0.5).astype(np.int16)
    i_oct_jk = (np.array(oct_v23["int8"]["junk_prob"])[m_oct_jk] >= 0.5).astype(np.int16)
    add_vec("oct15_junk_y_true", y_oct_jk, "int16", "v23", "reference_key", "ref", set(), sha_oct_jk)
    add_vec("oct15_junk_family_ids", dense_family_ids(oct_fams_all[m_oct_jk]), "int16", "v23", "reference_key", "ref", set(), sha_oct_jk)
    add_vec("oct15_junk_pred_ultra", u_oct_jk, "int16", "v23", "model_output", "ultra", set(), sha_oct_jk)
    add_vec("oct15_junk_pred_int8", i_oct_jk, "int16", "v23", "model_output", "int8", set(), sha_oct_jk)
    for p_tag in ["rtn", "bitmagnet"]:
        arr_jk = np.array([int(sa_oct_items[cid]["junk"][p_tag]) for cid in oct_cids], dtype=np.int16)[m_oct_jk]
        add_vec(f"oct15_junk_pred_{p_tag}", arr_jk, "int16", "sa", "model_output", p_tag, set(), sha_oct_jk)

    # 2.7 tc_truespec_2000 (zero_cue_2000) anime & dub
    tc2k_v23 = v23["splits"]["tc_truespec_2000"]
    tc2k_fams = np.array(tc2k_v23["family_ids"], dtype=np.int64)
    y_tc2k_dub = (np.array(tc2k_v23["y"]["english_audio"], dtype=np.int16) == 0).astype(np.int16)
    sha_tc2k_dub = key_digest(y_tc2k_dub, "tc2000_ani_y_true")
    add_vec("tc2000_ani_y_true", y_tc2k_dub, "int16", "v23", "reference_key", "ref", set(), sha_tc2k_dub)
    add_vec("tc2000_ani_family_ids", dense_family_ids(tc2k_fams), "int16", "v23", "reference_key", "ref", set(), sha_tc2k_dub)
    for p_tag in ["rtn", "ptt", "guessit", "ptn", "anitomy"]:
        arr_tc = np.array(sa["tc_truespec_2000"]["ani_dub_preds"][p_tag], dtype=np.int16)
        add_vec(f"tc2000_ani_pred_{p_tag}", arr_tc, "int16", "sa", "model_output", p_tag, set(), sha_tc2k_dub)

    y_tc2k_ani_all = np.array(tc2k_v23["y"]["anime"], dtype=np.int16)
    m_tc2k_ani = y_tc2k_ani_all >= 0  # 1998
    y_tc2k_ani = y_tc2k_ani_all[m_tc2k_ani]
    sha_tc2k_ani = key_digest(y_tc2k_ani, "tc2000_anime_head_y_true")
    add_vec("tc2000_anime_head_y_true", y_tc2k_ani, "int16", "v23", "reference_key", "ref", set(), sha_tc2k_ani)
    add_vec("tc2000_anime_head_pred_ultra", (np.array(tc2k_v23["fp32"]["anime_prob"])[m_tc2k_ani] >= 0.5).astype(np.int16), "int16", "v23", "model_output", "ultra", set(), sha_tc2k_ani)
    add_vec("tc2000_anime_head_pred_int8", (np.array(tc2k_v23["int8"]["anime_prob"])[m_tc2k_ani] >= 0.5).astype(np.int16), "int16", "v23", "model_output", "int8", set(), sha_tc2k_ani)

    # ==========================================================================
    # 3. SWARM SURVIVAL (PUBLIC N=757 & WAREHOUSE N=10,278 — VERIFIED OOF ONLY)
    # ==========================================================================
    pub_757 = sw_oof["public_757"]
    y_pub = np.array(pub_757["y_true"], dtype=np.int16)
    sha_pub = key_digest(y_pub, "swarm_pub_y_true")
    pub_fams = dense_family_ids([release_family_id(t) for t in pub_757["titles"]])
    add_vec("swarm_pub_y_true", y_pub, "int16", "sw_oof", "reference_key", "ref", set(), sha_pub)
    add_vec("swarm_pub_family_ids", pub_fams, "int16", "sw_oof", "reference_key", "ref", set(), sha_pub)
    add_vec("swarm_pub_score_ultra", np.array(pub_757["scores"]["armiger_ultra"], dtype=np.float32), "float32", "sw_oof", "model_output", "ultra", set(), sha_pub)
    add_vec("swarm_pub_score_priors", np.array(pub_757["scores"]["way4_combined_formula"], dtype=np.float32), "float32", "sw_oof", "model_output", "priors", set(), sha_pub)
    add_vec("swarm_pub_score_seeders", np.array(pub_757["scores"]["dht_seeders"], dtype=np.float32), "float32", "sw_oof", "model_output", "seeders", set(), sha_pub)

    whse = sw_oof["warehouse_10278"]
    y_whse = np.array(whse["y_true"], dtype=np.int16)
    sha_whse = key_digest(y_whse, "swarm_whse_y_true")
    whse_fams = dense_family_ids([release_family_id(t) for t in whse["titles"]])
    add_vec("swarm_whse_y_true", y_whse, "int16", "sw_oof", "reference_key", "ref", set(), sha_whse)
    add_vec("swarm_whse_family_ids", whse_fams, "int16", "sw_oof", "reference_key", "ref", set(), sha_whse)
    add_vec("swarm_whse_score_ultra", np.array(whse["scores"]["armiger_ultra"], dtype=np.float32), "float32", "sw_oof", "model_output", "ultra", set(), sha_whse)
    add_vec("swarm_whse_score_priors", np.array(whse["scores"]["dht_seeders"], dtype=np.float32), "float32", "sw_oof", "model_output", "priors", set(), sha_whse)
    add_vec("swarm_whse_score_seeders", np.array(whse["scores"]["tr_seeders"], dtype=np.float32), "float32", "sw_oof", "model_output", "seeders", set(), sha_whse)

    # ==========================================================================
    # 4. FIVE-SPLIT CATEGORY SUMMARY ARRAYS
    # ==========================================================================
    for sp_code, sp_name in [
        ("oct15", "oct15_confirmation"),
        ("fresh", "fresh_confirmation"),
        ("beyond", "beyond_confirmation"),
        ("tc2000", "tc_truespec_2000"),
        ("tc3000", "tc_truespec_multilang_3000"),
    ]:
        sp_d = v23["splits"][sp_name]
        y_sp_all = np.array(sp_d["y"]["category"], dtype=np.int16)
        m_sp = y_sp_all >= 0
        y_sp = y_sp_all[m_sp]
        sha_sp = key_digest(y_sp, f"split5_{sp_code}_y_cat")
        fams_sp = dense_family_ids(np.array(sp_d["family_ids"], dtype=np.int64)[m_sp])
        add_vec(f"split5_{sp_code}_y_cat", y_sp, "int16", "v23", "reference_key", "ref", set(), sha_sp)
        add_vec(f"split5_{sp_code}_family_ids", fams_sp, "int16", "v23", "reference_key", "ref", set(), sha_sp)
        add_vec(f"split5_{sp_code}_fp32_hyb_pred", np.array(sp_d["fp32"]["hyb_cat"], dtype=np.int16)[m_sp], "int16", "v23", "model_output", "ultra", set(), sha_sp)
        add_vec(f"split5_{sp_code}_fp32_raw_pred", np.array(sp_d["fp32"]["raw_cat"], dtype=np.int16)[m_sp], "int16", "v23", "model_output", "ultra", set(), sha_sp)
        add_vec(f"split5_{sp_code}_int8_hyb_pred", np.array(sp_d["int8"]["hyb_cat"], dtype=np.int16)[m_sp], "int16", "v23", "model_output", "int8_hyb", set(), sha_sp)
        add_vec(f"split5_{sp_code}_int8_raw_pred", np.array(sp_d["int8"]["raw_cat"], dtype=np.int16)[m_sp], "int16", "v23", "model_output", "int8_raw", set(), sha_sp)

    payload = {
        "_meta": {
            "schema": "armiger.canonical_eval_vectors.v3.1",
            "receipt_files_sha256": {r_rels[k]: r_shas[k] for k in sorted(r_files)},
            "dtypes": dtypes,
            "provenance": provenance,
        },
        "vectors": vectors,
    }
    out_path = BENCH_DIR / "canonical_eval_vectors.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"[OK] Built {out_path.name} with {len(vectors)} vectors from {len(r_files)} verified on-disk receipts.")
    return payload


if __name__ == "__main__":
    build_vectors()
