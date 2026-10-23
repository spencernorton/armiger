## 2026-10-23 — codex — setup verification checkpoint

- **Branch:** codex/package-and-setup
- **What changed:** Finished the cleanup and retained the Ultra, Bitmagnet and Rats Search setup paths; pinned the merged connector loader.
- **Validation:** 267 tests passed with one internal-only integration skipped; installed-wheel INT8/Ultra inference, cold verified downloads, offline cache checks, both Compose startups and synthetic database writes passed. Restart/recreation verification and required GitHub checks finish the release gate.
- **Deployment:** No production deployment or Git history rewrite.
- **Next step:** Complete required checks for the cleanup PR.
- **Status file updated:** Final private handoff follows this checkpoint.

## 2026-10-23 — codex — package and setup cleanup

- **Branch:** codex/package-and-setup
- **What changed:** Installable source package, unified tests, pinned model data downloads, four focused guides, and prominent Bitmagnet/Rats Search Compose examples. Flat public imports remain compatibility aliases.
- **Validation:** Runtime, benchmark integrity, downloader and installed-wheel checks; Compose startup verification in progress.
- **Deployment:** None; source and local verification only.
- **Next step:** Complete local Compose verification and required PR checks.
- **Status file updated:** Pending final handoff.
