# Canonical Statistical Chart & Benchmark Pipeline (`v2.1.0`)

Every benchmark figure and headline accuracy table across the **Armiger** and **Armiger Ultra** repositories is generated deterministically from anonymized evaluation vectors (`benchmarks/canonical_eval_vectors.json`) and locked by a cryptographic provenance manifest (`benchmarks/CHARTS_MANIFEST.json`).

---

## 1. Pipeline Architecture & Artifacts

| Artifact | SHA-256 Digest | Description |
| :--- | :--- | :--- |
| `benchmarks/canonical_eval_vectors.json` | `058473f4ba21bc07db2c2b6981d5e62f3c584311fc134d92cb523de7f261243a` | Plain-text JSON containing `>140` 1D numeric `int16`/`float32` prediction and ground-truth label arrays across all 5 evaluation splits (`N=10,262`). Contains **zero** strings, release titles, file paths, or infohashes. |
| `benchmarks/generate_canonical_charts.py` | `dfe6262c4815ec8fa0ab47569b80faebe7a4d29fb85824c4cc05c1b921e32270` | Deterministic chart generator and statistical test runner (`BOOTSTRAP_SEED = 20261018`, `BOOTSTRAP_RESAMPLES = 10000`). |
| `benchmarks/CHARTS_MANIFEST.json` | *(Tracked by `test_chart_integrity.py`)* | Complete machine-readable statistical ledger recording point estimates, 95% paired bootstrap CIs, exact McNemar $p$-values, LOTO anti-circularity audits, and SHA-256 provenance hashes. |
| `assets/armiger_vs_llms_overall_and_6dims.png` | `2367dbf6ae84cc2f75b0b13f377f7053f8f0e63ebe56cf42e1093b26aa18c082` | 6-dimension accuracy & composite mean comparison against frontier cloud LLMs and local GPU LLMs. |
| `assets/armiger_vs_llms_cost_comparison.png` | `ff33da1aa78dc14be919d34d5914db5efabe8c30d1698bda378962a3c92c5dbc` | Operational cost per 1M torrents and token/hardware rate comparison. |
| `assets/armiger_vs_torrent_classifiers_6dims.png` | `a4ebfcb1565ed94c830846845ecf16e46061424180ea01a5852b8d4d1e614a6a` | 7-dimension comparison against open-source torrent classifiers and regex parsers (`MIMMO BERT`, `RTN`, `Parsett`, `GuessIt`, `PTN`, `Anitomy`, `Bitmagnet`). |
| `assets/armiger_categorical_breakdown.png` | `b137523c9ef1e5e26463f06db2ca477b9acfbae83a3f07869d74c8ce915a4513` | Per-category breakdown across all 10 media & software categories on `oct15_confirmation` (`N=2,835`) and `beyond_confirmation` (`N=1,025`). |

---

## 2. Four Machine-Enforced Statistical Invariants

`benchmarks/test_chart_integrity.py` enforces four strict invariants on every CI run:

### Invariant 1 — Identical Denominator Lock (`assert_identical_denominator`)
Every model compared on a benchmark panel must be evaluated on the **exact same item mask and denominator $N$**. Dropping abstentions (`unknown`), mixing evaluation splits, or evaluating on a subset raises an immediate `AssertionError`:
- `oct15_confirmation` 11-Class Category: $N = 2,835$
- `beyond_confirmation` 11-Class Category & Catalog Matching: $N = 1,025$
- `oct15_confirmation` Movie-vs-TV & Catalog Matching: $N = 1,296$
- `oct15_confirmation` Binary English-vs-Foreign Audio: $N = 545$ (`beyond_confirmation` strict audio: $N = 229$)
- `oct15_confirmation` Anime Detection: $N = 507$ (`beyond_confirmation` valid media: $N = 636$)
- `oct15_confirmation` Junk / Malware Gate: $N = 2,170$ (`beyond_confirmation`: $N = 977$)

### Invariant 2 — Anti-Circularity Lock (`assert_non_circular_teacher_eval`)
On `beyond_confirmation` ($N = 1,025$), where multi-teacher consensus contributed to reference curation, teacher models (`OpenAI GPT gpt-6-luna` and `Google Gemini gemini-3.8-flash-low`) are strictly evaluated against **Leave-One-Teacher-Out (LOTO)** reference keys (`self_vote_weight = 0.0`):

| Teacher Model | Non-Circular LOTO Accuracy (`self_vote_weight = 0.0`) | Circular Self-Voted Accuracy | Removed Circularity Inflation |
| :--- | :---: | :---: | :---: |
| `OpenAI GPT (gpt-6-luna)` | **`95.12%`** | `96.59%` | `+1.46 pp` |
| `Google Gemini (gemini-3.8-flash-low)` | **`93.80%`** | `95.61%` | `+1.80 pp` |

Scoring any teacher model with `self_vote_weight > 0.0` or `eval_mode != "leave_one_teacher_out"` raises an `AssertionError`.

### Invariant 3 — Paired Bootstrap 95% CIs & Exact McNemar $p$-Values (`compute_paired_panel_stats`)
All uncertainty intervals and significance tests are paired by item index across `10,000` bootstrap resamples (`seed = 20261018`) using integer-scaled `float64` dot products (eliminating BLAS thread non-determinism across macOS ARM64 and Linux x86_64) and exact binomial `math.comb` McNemar two-sided $p$-values.

### Invariant 4 — Cryptographic PNG Hygiene (`verify_png_chunks_clean`)
Every generated PNG is stripped of non-deterministic metadata (`tEXt`, `iTXt`, `zTXt`, `tIME`, `pHYs`) so only `IHDR`, `IDAT`, and `IEND` chunks remain, guaranteeing byte-exact reproducibility.

---

## 3. Running Verification Locally

```bash
pytest benchmarks/test_chart_integrity.py -v
```
