# Charts for Armiger discussion 55

These assets illustrate [the release cadence discussion](https://github.com/spencernorton/armiger/discussions/55).

## Legacy cohort progression (`heldout_20261026`)

- `accuracy-by-task.png`: v3.3.0, v3.8.0, v3.10.0, v3.11.0, v3.12.0, v3.15.4, and v3.17.0 point estimates and 95% intervals across the five `heldout_20261026` tasks.
- `relative-error-reduction.png`: two-panel view comparing legacy `heldout_20261026` error reduction (`v3.3.0 -> v3.17.0`, with `v3.15.4` dashed markers) alongside firewalled `R26-live` error reduction (`v3.15.4 -> v3.17.0` and vs `v3.12.0`).
- `improvement-velocity.png`: accuracy-point change per minor-version step on `heldout_20261026`, using gaps 1, 3, 1, 2, 1, 1, 3, and 2 across v3.3.0, v3.4.0, v3.7.0, v3.8.0, v3.10.0, v3.11.0, v3.12.0, v3.15.4, and v3.17.0.
- `chart-data.json`: exact `heldout_20261026` chart values, task denominators, Wilson 95% intervals for `v3.15.4` and `v3.17.0`, and immutable public source links.

## Sealed firewalled prospective holdouts (`R22-live`, `R26-live`, and `R27-live`)

- `sealed-r22-live-v312-vs-v3154.png`: paired `v3.12.0` vs `v3.15.4` vs `v3.17.0` comparison on firewalled `R26-live` (`N = 3,104`) and live production container latency/memory (`3.57 ms/row`, `283.1 MiB` live container RSS, `-76.6%` vs `v3.15.4`).
- `r26-live-data.json` & `r22-live-data.json`: exact `R26-live` (`N = 3,104`), `R27-live` (`44,672` prospective shadow rows, `194 vs 12` blind audit win rate `94.17%`, `180/180` red-team canaries), and `R22-live` (`N = 2,795`) metrics and runtime figures.

## Broader evaluation coverage example

- `small-vs-broad-evaluation.png` compares the same published v3.0.0 Ultra FP32 fair-hybrid pipeline on `multilang_3000` (2,950/3,000 = 98.33%) and the combined five-cohort pool (9,711/10,262 = 94.63%).
- `evaluation-coverage-data.json` records exact counts, immutable public source links, source hash, and interpretation limits.
