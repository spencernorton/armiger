# Charts for Armiger Discussion #55

Standardized 4-chart visual suite for [Discussion #55](https://github.com/spencernorton/armiger/discussions/55), generated directly from the committed JSON data via `generate_discussion_charts.py`.

## Charts

- `unseen-live-benchmark.png`: Firewalled zero-overlap `R26-live` (`N = 3,104`) accuracy & safety with 95% Wilson CIs, paired McNemar p-values, live container RAM, BS=32 CPU latency, and `R27-live` (`44,672` unseen rows) blind audit summary.
- `relative-improvement-model-over-model.png`: Single standardized model-over-model relative error reduction (`%` fewer errors vs. prior release) across `v3.3.0 → v3.17.0` with an explicit regime split between legacy `heldout_20261026` and unseen `R26-live`.
- `legacy-vs-live-generalization.png`: Legacy `heldout_20261026` trajectories (`v3.3.0 → v3.17.0`, 95% Wilson CIs) paired with the legacy-vs-live generalization cliff (`−23.6 pp → −1.8 pp`).
- `small-vs-broad-evaluation.png`: Fixed `v3.0.0` pipeline on `multilang_3000` (`98.33%`) vs. the broader five-cohort pool (`94.63%`, `N = 10,262`) with 95% Wilson CIs.

## Data & Verification

- `chart-data.json`: `heldout_20261026` scores, 95% intervals, and immutable source links.
- `r26-live-data.json` & `r22-live-data.json`: Firewalled prospective holdout & `R27-live` shadow metrics and runtime numbers.
- `evaluation-coverage-data.json`: Five-cohort `v3.0.0` coverage counts and provenance.
- `test_discussion_charts.py`: Automated consistency tests for all JSON invariants and PNG assets.
