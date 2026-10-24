## 2026-10-24 — Distributed updates explained

- **Branch:** codex/distributed-updates-guide
- **What changed:** Lead the distributed update guide with a plain-language callout, practical benefits and a compact flow diagram. Explain receiving updates separately from optional contributions; move implementation detail into an expandable reference.
- **Evidence:** Check the sharing resolver, signed update verification, Ultra staging and rollback, and the keyed catalog guard against current source. Distinguish contribution blocking from network silence, payload protections from sender anonymity, and regression checks from accuracy qualification.
- **Validation:** Local links, anchors, JSON, code fences and patch whitespace pass. The six-node Mermaid diagram renders and has been visually checked. Required package tests and builds run through protected CI.
- **Deployment:** Documentation only; package, model, runtime and production settings are unchanged.
- **Status file updated:** Record the public PR and verification in the private project handoff; the separate benchmark objective remains active.

## 2026-10-24 — Bitmagnet Ultra configuration reference

- **Branch:** codex/readme-ultra-configuration
- **What changed:** Add a complete reference for the shipped Bitmagnet connector environment settings, a copyable Compose override and configuration links from the setup guide, README and Python reference. Remove the measured-results caption below the overview.
- **Evidence:** The connector source matches the image's pinned fd5c3fa checksum. Document effective Compose defaults, filtering versus index removal, lazy catalog loading, listener scopes and compatibility settings that do not apply to the prebuilt image.
- **Validation:** All 22 environment names read by the pinned connector are covered. The actual published OCI setup accepts the documented automatic override and `.env` interpolation; default-off pruning/dead-swarm rejection remain intact. Local links and patch whitespace checked.
- **Deployment:** Documentation and an optional example only; package, model, runtime and production settings are unchanged.
- **Status file updated:** Record the public PR and remaining sample-size clarification in the private project handoff.

## 2026-10-23 — plain-language comparison overview

- **Branch:** codex/readme-comparison-overview
- **What changed:** Show a dark category comparison with MIMMO and Bitmagnet, plus readable Ultra cards for title matching, audio language, junk detection and seed availability. Keep full methodology in the evaluation guide.
- **Evidence:** Derive values from the canonical benchmark manifest; the overview record retains full precision and source, renderer and image hashes. Seed ranking and balanced junk scores retain their distinct meanings and cohorts.
- **Validation:** Compare category values with individual prediction vectors, reject mixed metrics/denominators, verify provenance and inspect the rendered figure and README.
- **Deployment:** Presentation only; package, model and runtime versions remain unchanged.
- **Status file updated:** Record the public PR and verification in the private project handoff.

## 2026-10-23 — dark headline chart below the description

- **Branch:** codex/readme-headline-chart
- **What changed:** Place the existing dark headline benchmark chart directly below the README introduction, before the setup guides. Keep the model version and evaluation link beside it.
- **Validation:** The chart bytes match the canonical manifest; its source, data and labels are unchanged. Verify the rendered placement and required CI before merging.
- **Deployment:** Documentation only; package, model and runtime versions remain unchanged.
- **Status file updated:** Record the public pull request and verification in the private project handoff.

## 2026-10-23 — consistent README badges

- **Branch:** codex/readme-badges
- **What changed:** Centre the product title and add the NorviTech Armiger page, main CI, latest release, Hugging Face, Apache-2.0 licence and Stripe donation badges. Stripe remains last.
- **Validation:** All six badge SVGs load, all external destinations return HTTP 200, the Armiger anchor exists, the licence target exists and Stripe is last. Required CI governs merging.
- **Deployment:** Documentation only; the package and model versions remain unchanged.
- **Status file updated:** Record the public pull request and validation in the private project handoff.

## 2026-10-23 — pin Torznab numeric file-count connector fix (3.2.1)

- **Branch:** fix/pin-torznab-files-connector
- **What changed:** Pin merged `Armiger-Connectors` `fd5c3faae6f5bd915a66010b7d1ed67ed8dd96f8` (`v1.1.1`, PR #3) in `examples/Dockerfile` with updated SHA-256 checksum and bump release version to `3.2.1` so CI publishes updated container images and Compose packages.
- **Why:** Prowlarr and standard Torznab feeds send `<torznab:attr name="files" value="36"/>` as a numeric file count rather than a semicolon-delimited path list.
- **Validation:** Connector suite (14 passed) and Armiger runtime suite pass.
- **Deployment:** Published via protected `main` CI to GHCR (`3.2.1`).
- **Next step:** Merge after green required checks and tag `v3.2.1`.
- **Status file updated:** Recorded in `shared-ops/projects/status/bitagent.md`.

## 2026-10-23 — codex — packaged setup acceptance and release provenance

- **Branch:** codex/prebuilt-containers
- **Validation:** Both two-line OCI setups start healthy from an isolated registry and complete real Bitmagnet tag/Rats Search category writes. The loaded network/database names follow the caller's project; processor containers need no model mounts.
- **Release:** Serialize main CI/publication without cancelling active releases. Native build tags bind the exact source SHA and architecture; release manifests combine only that SHA's builds and record its revision. Existing version tags stay unchanged.
- **Why:** Prevent a superseding main run or partial retry from mixing architecture builds from different source commits.
- **Next step:** Complete protected checks and public anonymous acceptance for the two images and two Compose packages.
- **Status file updated:** Final private checkpoint records source, package digests and verification.

## 2026-10-23 — codex — two-line packaged integration setups

- **Branch:** codex/prebuilt-containers
- **What changed:** Publish the full Bitmagnet and Rats Search Compose setups as versioned OCI artifacts after their images. Main guides use two-line `include` files (Compose 2.34+). Full files remain available for custom settings and older Compose.
- **Implementation:** Normalize defaults to JSON for portable publication, then remove generated project/network/volume names so each user's folder gets its own resources. Published tags are preserved.
- **Validation:** Isolated TLS registry publication and two-line loading pass; the loaded Bitmagnet network/database names follow the caller's project. Full runtime verification completes before public acceptance.
- **Next step:** Verify both packaged setups, protected checks, public images/artifacts and anonymous pulls.
- **Status file updated:** Final private checkpoint records the packaged setup.

## 2026-10-23 — codex — keep bundled-model containers stateless

- **Branch:** codex/prebuilt-containers
- **Decision:** Remove redundant model-cache volumes. The pinned model ships in the image; each processor uses that version directly. Only Bitmagnet's database and Rats Search's index require persistent volumes in the default setups.
- **Why:** Fewer novice settings, no duplicate model copies, and image updates always select the matching bundled model. Python callers retain the verified-download cache API.
- **Validation:** Recheck the simplified Compose definitions and fresh startup; prior source, wheel, offline inference and backend-write checks remain applicable.
- **Next step:** Complete protected publication and anonymous-pull acceptance.
- **Status file updated:** Final private checkpoint records the image and data-volume model.

## 2026-10-23 — codex — bundled container acceptance

- **Branch:** codex/prebuilt-containers
- **Validation:** All three fresh pull-only Compose stacks reach healthy with the bundled model. Standalone HTTP classification works with networking disabled. Source suite: 272 passed, one internal-only skip; installed-wheel INT8/Ultra inference passed. The model is fetched by checksummed build inputs and verified offline against the package manifest during the build.
- **Fix:** Pin merged Armiger-Connectors `8e808e481d2470420f5f0cd29fa02e39f4213180`, which fixes stock Bitmagnet's 30-character tag limit. Its 13 tests and protected check pass. Include the bundled librats license alongside Rats Search and Manticore source/notices.
- **Deployment:** Local disposable stacks only. Protected GitHub merge and public image verification remain before release completion.
- **Next step:** Publish tested native images and verify anonymous fresh pulls.
- **Status file updated:** Tracker records the acceptance fix and remaining publication gate.

## 2026-10-23 — codex — bundle verified model in the image

- **Branch:** codex/prebuilt-containers
- **Decision:** Include the pinned, checksummed Ultra model in the image. Fresh Docker volumes are initialized from it, avoiding a second download provider during novice setup. Existing compatible model volumes remain reusable.
- **Why:** A separate first-start Hugging Face download stalled during disposable setup verification. Docker should deliver the complete initial installation.
- **Validation:** Rebuild and offline/fresh-stack verification in progress; prior source and installed-wheel checks remain green.
- **Next step:** Verify the bundled image and complete publication.
- **Status file updated:** Tracker will record final release results.

## 2026-10-23 — codex — container source verification

- **Branch:** codex/prebuilt-containers
- **Validation:** 272 tests passed, one internal-only integration skipped. Wheel built and classified with INT8 and Ultra outside the checkout. ARM64 standalone container completed cold model download and real HTTP classification. Fresh Bitmagnet stack wrote a movie tag and retained it after recreation; the pinned AMD64 Rats backend built successfully.
- **Decision:** Use native AMD64/ARM64 GitHub runners and the repository's existing GitHub-owned action policy; no action allowlist expansion or emulation is required for publishing.
- **Next step:** Finish Rats persistence and protected checks, then publish and verify anonymous pulls.
- **Status file updated:** Tracker records source verification.

## 2026-10-23 — codex — ready-made containers

- **Branch:** codex/prebuilt-containers
- **What changed:** Publish versioned Armiger and pinned Rats Search images after protected main checks; fresh Compose setups need no build, clone or required environment file. Add standalone Ultra HTTP service and reduce setup to download, start, open.
- **Validation:** HTTP contract tests, Compose configuration and native ARM64 image startup/classification pass. Complete suite, Rats backend build and public registry verification are release gates in progress.
- **Deployment:** Local disposable verification only; publication follows protected GitHub checks.
- **Next step:** Finish image checks, publish, verify anonymous pulls and fresh setups.
- **Status file updated:** Tracker records the release work and remaining gates.

## 2026-10-23 — codex — fresh all-in-one setup scope

- **Branch:** codex/plain-setup-guides
- **Decision:** The main Bitmagnet and Rats Search guides cover fresh all-in-one setups. Adding Armiger to existing installations is deferred.
- **What changed:** Remove existing-installation detours from the setup flow; retain the supported fresh-stack commands and expected results.
- **Validation:** Guide order, links, example output and Compose configuration verified in this pass.
- **Deployment:** None; protected PR checks govern merging.
- **Next step:** Merge the guide pass after required checks.
- **Status file updated:** Tracker and private handoff record the scope.

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
