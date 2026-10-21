# Canonical Statistical Chart & Benchmark Pipeline (`v3.0.0`)

Every benchmark figure and headline accuracy table across the **Armiger** and **Armiger Ultra** repositories is generated deterministically from anonymized evaluation vectors (`benchmarks/canonical_eval_vectors.json`) and locked by a cryptographic provenance manifest (`benchmarks/CHARTS_MANIFEST.json`).

---

## 1. Pipeline Architecture & Artifacts

| Artifact | SHA-256 Digest | Description |
| :--- | :--- | :--- |
| `benchmarks/canonical_eval_vectors.json` | `62992cc441d321938446c54d52732a65b36da2d83c41d837d34b5c7674156aa9` | Plain-text JSON containing `184` 1D prediction, correctness, score, ground-truth, and family-cluster arrays across all 5 evaluation splits (`N=10,262`), built by `benchmarks/build_canonical_eval_vectors.py` from the verified receipts in `benchmarks/receipts/`. Contains **zero** release titles, file paths, or raw infohashes (swarm identifiers are hashed). |
| `benchmarks/generate_canonical_charts.py` | `4ffca978472527d879c0d547dfadbbcd9b2ee60467945eff1976dac429f119a8` | Deterministic chart generator and statistical test runner (`BOOTSTRAP_SEED = 20261018`, `BOOTSTRAP_RESAMPLES = 10000`). |
| `benchmarks/CHARTS_MANIFEST.json` | *(Tracked by `test_chart_integrity.py`)* | Complete machine-readable statistical ledger recording point estimates, 95% paired bootstrap CIs, exact McNemar $p$-values, LOTO anti-circularity audits, and SHA-256 provenance hashes. |
| `assets/armiger_vs_llms_overall_and_6dims.png` | `8f8b952ff9f0eb68911aac1b1d50b2cda5394a5312f0051b9335703dc89b8312` | 6-dimension accuracy & composite mean comparison against cloud teacher LLMs on identical evaluation cohorts (with `OpenAI GPT` scored across its 5 prompted dimensions and marked `N/A` on `beyond_confirmation` sample/junk). |
| `assets/armiger_vs_llms_cost_comparison.png` | `447f0e9767b36fe1f5bc16623e249e0b897f65b147a7b8abbaebce4865c934e3` | Operational cost per 1M torrents and token/hardware rate comparison. |
| `assets/armiger_vs_torrent_classifiers_6dims.png` | `7e9704d0c0fdb77029d826fe450e19ec277e0723e76bd27281deec8b5acedc55` | 7-dimension comparison against open-source torrent classifiers and title parsers (`MIMMO BERT`, `RTN`, `Parsett`, `GuessIt`, `PTN`, `Anitomy`). |
| `assets/armiger_categorical_breakdown.png` | `9246ce3c6eea024383ee21a44608f632c08e12cbb2027fac4b2f19352163e0df` | Per-category recall breakdown across all 10 media & software categories on `oct15_confirmation` (`N=2,835`) and `beyond_confirmation` (`N=1,025`). |

---

## 2. Four Machine-Enforced Statistical Invariants

`benchmarks/test_chart_integrity.py` enforces four strict invariants on every CI run:

### Invariant 1 — Identical Denominator Lock (`assert_identical_denominator`)
Every model evaluated on a benchmark panel is scored on the **exact same item mask and denominator $N$**. Dropping abstentions (`unknown`), mixing evaluation splits, or evaluating on a subset raises an immediate `AssertionError`:
- `oct15_confirmation` 11-Class Category: $N = 2,835$
- `beyond_confirmation` 11-Class Category & Catalog Matching: $N = 1,025$
- `oct15_confirmation` Movie-vs-TV & Catalog Matching: $N = 1,296$
- `oct15_confirmation` Binary English-vs-Foreign Audio: $N = 545$ (`beyond_confirmation` strict audio: $N = 229$)
- `oct15_confirmation` Anime Detection: $N = 507$ (`beyond_confirmation` valid media: $N = 636$)
- `beyond_confirmation` Junk / Sample Gate: $N = 977$ (`65` positive items, `912` negative items; `OpenAI GPT` is explicitly marked `N/A` in Panel 2E because its prompt schema omitted the sample/extra junk definition)
- `Chart 1 Panel 2F` Swarm Liveliness AUROC: $N = 757$ Public-Tracker Cohort across all 6 compared systems (`Chart 3 Panel 6B`: $N = 10,278$ Multi-Source Corpus across all 3 compared methods)

### Invariant 2 — Anti-Circularity Lock (`assert_non_circular_teacher_eval`)
On `beyond_confirmation` ($N = 1,025$), where multi-teacher consensus contributed to reference curation, teacher models (`OpenAI GPT (Fast Tier)` and `Google Gemini Flash`) are strictly evaluated against **Leave-One-Teacher-Out (LOTO)** reference keys (`self_vote_weight = 0.0`):

| Teacher Model | Non-Circular LOTO Lower Bound (full cohort, `self_vote_weight = 0.0`) | Circular Self-Voted Accuracy | Common-Support LOTO Panel Accuracy |
| :--- | :---: | :---: | :---: |
| `OpenAI GPT (Fast Tier)` | **`92.98%`** (`N=1,025`) | `97.95%` | `97.84%` (`N=974`) |
| `Google Gemini Flash` | **`92.98%`** (`N=1,025`) | `97.17%` | `97.05%` (`N=982`) |

Scoring any teacher model with `self_vote_weight > 0.0` or `eval_mode != "leave_one_teacher_out"` raises an `AssertionError`.

### Invariant 3 — Paired Bootstrap 95% CIs, Hájek/DeLong AUROC CIs & Exact McNemar $p$-Values (`compute_paired_panel_stats`)
All accuracy uncertainty intervals and significance tests are paired by item index across `10,000` bootstrap resamples (`seed = 20261018`) using integer-scaled `float64` dot products (eliminating BLAS thread non-determinism across macOS ARM64 and Linux x86_64), two-sample Hájek/DeLong rank-placement variance for AUROC panels, and exact binomial `math.comb` McNemar two-sided $p$-values.

### Invariant 4 — Cryptographic PNG Hygiene (`verify_png_chunks_clean`)
Every generated PNG is stripped of non-deterministic metadata (`tEXt`, `iTXt`, `zTXt`, `tIME`, `pHYs`) so only `IHDR`, `IDAT`, and `IEND` chunks remain, guaranteeing byte-exact reproducibility.

---

## 3. Running Verification Locally

```bash
python benchmarks/build_canonical_eval_vectors.py
python benchmarks/generate_canonical_charts.py
pytest benchmarks/test_chart_integrity.py -v
```
