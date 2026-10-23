## 2026-10-23 — codex — plain setup guides

- **Branch:** codex/plain-setup-guides
- **What changed:** Prioritize Bitmagnet, Rats Search, then standalone; link to Armiger's NorviTech page; rewrite setup around prerequisites, expected results and saved data. Add one runnable standalone example.
- **Validation:** Standalone example prints `Content type: movie`; wheel metadata, local links, setup order, Compose configuration and patch whitespace pass. Compose runtime is unchanged from the verified cleanup.
- **Deployment:** No production changes. GitHub's website link is updated directly.
- **Next step:** Complete protected PR checks and merge the guide update.
- **Status file updated:** Tracker and private handoff record the guide pass.

## 2026-10-23 — codex — Compose verification complete

- **Branch:** codex/package-and-setup
- **Validation:** Full suite: 267 passed, one internal-only bridge integration skipped. Both supported Compose stacks pass fresh download, real classification, synthetic database writes and data/cache persistence after container recreation. Installed-wheel INT8 and Ultra inference pass outside the checkout.
- **Deployment:** None. Required GitHub checks govern merging. Concurrent security work remains in its own PR; file moves retain compatibility aliases.
- **Next step:** Merge the cleanup after its required checks pass; subsequent runtime hardening targets `src/armiger/` and `tests/`.
- **Status file updated:** Private status handoff records the PRs and measurements.

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
