# Charts for Armiger discussion 55

These assets illustrate [the release cadence discussion](https://github.com/spencernorton/armiger/discussions/55).

## Legacy cohort progression (`heldout_20261026`)

- `accuracy-by-task.png`: v3.3.0, v3.8.0, v3.10.0, v3.11.0, v3.12.0, and v3.15.4 point estimates and 95% intervals across the five `heldout_20261026` tasks.
- `relative-error-reduction.png`: two-panel view comparing legacy `heldout_20261026` error reduction (`v3.3.0 -> v3.15.4`, with `v3.12.0` peak markers) alongside fresh sealed zero-overlap `R22-live` error reduction (`v3.12.0 -> v3.15.4`).
- `improvement-velocity.png`: accuracy-point change per minor-version step on `heldout_20261026`, using gaps 1, 3, 1, 2, 1, 1, and 3 across v3.3.0, v3.4.0, v3.7.0, v3.8.0, v3.10.0, v3.11.0, v3.12.0, and v3.15.4. Both shaded intervals highlight deliberate rule removal (`v3.10.0` removed 68 curated title stems; `v3.15.4` retired the 2,800-line legacy heuristic cascade in favor of clean regularized Stage-2 combiners).
- `chart-data.json`: exact `heldout_20261026` chart values, task denominators, Wilson 95% intervals for `v3.15.4`, and immutable public source links.

All `heldout_20261026` values use the named cohort denominators. The smaller task slices are separate denominators, not disjoint datasets to add together. These charts report the Ultra pipeline, including enrichment components.

## Fresh sealed prospective holdout (`R22-live`)

- `sealed-r22-live-v312-vs-v3154.png`: paired `v3.12.0` vs `v3.15.4` (`v5.4-pareto`) comparison on `R22-live` (`N = 2,795` freshly sampled torrents with zero training overlap and dual-provider consensus labels), covering live production accuracy (`85.93% -> 94.13%`, `+8.20 pp`, McNemar `p = 3.05e-25`), hard stress accuracy (`55.98% -> 90.89%`, `+34.90 pp`, `p = 7.29e-115`), combined accuracy (`72.06% -> 92.63%`, `+20.57 pp`, `p = 1.19e-136`), structural junk recall (`15.33% -> 98.99%`), semantic junk recall (`2.82% -> 91.55%`), clean content TNR (`99.61% -> 99.57%`), CPU latency (`23.96 -> 8.71 / 5.26 ms/row`), and peak RSS memory (`2.70 -> 1.53 / 1.50 GiB`).
- `r22-live-data.json`: exact `R22-live` counts, 95% Wilson intervals, exact two-sided paired McNemar p-values, FP32/INT8 agreement (`99.89%`, `2,792/2,795`), runtime latency/RSS figures, and immutable release links.

## Broader evaluation coverage example

- `small-vs-broad-evaluation.png` compares the same published v3.0.0 Ultra FP32 fair-hybrid pipeline on `multilang_3000` (2,950/3,000 = 98.33%) and the combined five-cohort pool (9,711/10,262 = 94.63%). The second panel shows each component cohort.
- `evaluation-coverage-data.json` records exact counts, immutable public source links, source hash, and interpretation limits.

The larger pool includes the original 3,000 cases plus 7,262 from four other cohorts. This illustrates differences in evaluation coverage and case mix, not a causal effect of test-set size, a chronological experiment, or a new benchmark. Historical label provenance and teacher involvement remain as documented in the original public sources.
