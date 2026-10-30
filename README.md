# Charts for Armiger discussion 55

These assets illustrate [the release cadence discussion](https://github.com/spencernorton/armiger/discussions/55).

- `accuracy-by-task.png`: v3.3.0 versus v3.8.0 point estimates and published 95% intervals.
- `relative-error-reduction.png`: (new accuracy - old accuracy) / (100 - old accuracy) * 100.
- `improvement-velocity.png`: accuracy-point change per minor-version step, using gaps 1, 3, and 1 across v3.3.0, v3.4.0, v3.7.0, and v3.8.0. These are interval averages, not per-day rates; intermediate scores are not inferred.
- `chart-data.json`: exact chart values, task denominators, and immutable source links.

All values use the published `heldout_20261026` cohort. The smaller task slices are separate denominators, not disjoint datasets to add together. These charts report the Ultra pipeline, including enrichment components. Individual changes are point estimates and do not establish statistical significance or attribute gains solely to training-data growth.
