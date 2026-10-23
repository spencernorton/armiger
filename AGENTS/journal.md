## 2026-10-23 — codex — package and setup cleanup

- **Branch:** codex/package-and-setup
- **What changed:** Installable source package, unified tests, pinned model data downloads, four focused guides, and prominent Bitmagnet/Rats Search Compose examples. Flat public imports remain compatibility aliases.
- **Validation:** 267 tests passed, one optional ONNX test skipped before installing that dependency; installed-wheel INT8/Ultra inference, cold model download, offline checks and both Compose startup/classification/database-write checks passed. Final full-dependency run and restart checks pending.
- **Deployment:** None; source and local verification only.
- **Next step:** Complete local Compose verification and required PR checks.
- **Status file updated:** Pending final handoff.
