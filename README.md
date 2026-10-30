# Charts for Armiger discussion 55

These assets illustrate [the release cadence discussion](https://github.com/spencernorton/armiger/discussions/55).

- `accuracy-by-task.png`: v3.3.0, v3.8.0, and v3.10.0 point estimates and published 95% intervals.
- `relative-error-reduction.png`: (new accuracy - old accuracy) / (100 - old accuracy) * 100 from v3.3.0 to v3.10.0.
- `improvement-velocity.png`: accuracy-point change per minor-version step, using gaps 1, 3, 1, and 2 across v3.3.0, v3.4.0, v3.7.0, v3.8.0, and v3.10.0. These are interval averages, not per-day rates; intermediate scores are not inferred.
- `chart-data.json`: exact chart values, task denominators, and immutable source links.

All values use the published `heldout_20261026` cohort. The smaller task slices are separate denominators, not disjoint datasets to add together. These charts report the Ultra pipeline, including enrichment components. Individual changes are point estimates and do not establish statistical significance or attribute gains solely to training-data growth.

## Broader evaluation coverage example

- `small-vs-broad-evaluation.png` compares the same published v3.0.0 Ultra FP32 fair-hybrid pipeline on `multilang_3000` (2,950/3,000 = 98.33%) and the combined five-cohort pool (9,711/10,262 = 94.63%). The second panel shows each component cohort.
- `evaluation-coverage-data.json` records exact counts, immutable public source links, source hash, and interpretation limits.

The larger pool includes the original 3,000 cases plus 7,262 from four other cohorts. This illustrates differences in evaluation coverage and case mix, not a causal effect of test-set size, a chronological experiment, or a new benchmark. Historical label provenance and teacher involvement remain as documented in the original public sources.
