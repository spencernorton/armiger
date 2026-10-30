## 2026-10-30 — Explain the classification pipeline with an architecture diagram

The README explains the product with a real interaction recording or a source-grounded
architecture diagram. Recordings retain their source attribution and offer still
images for reduced-motion viewing. No runtime behaviour is changed.

## 2026-10-30 — Explain the five published packages

- **Task:** Say what each GitHub Container Registry package is, so `armiger-rats-search` and `armiger-rats-search-stack` no longer look like duplicates.
- **What changed:** Add "Published packages" under Docker settings in `docs/reference.md`: `armiger` and its `armiger-ultra` alias, the Rats Search backend image, the two Compose `-stack` artifacts, and which tags move. `README.md` is unchanged because it is an image input in `ci/release_contract.py` and would force a new release.
- **Validation:** `python3 -m unittest discover -s ci/tests -v` and `python3 -m ci.release_contract` pass; no image input changed.
- **Release:** None. 3.8.1 stays current.

## 2026-10-29 — Release Armiger 3.8.1 with Armiger-Connectors v1.2.7 and runtime acceptance gates

- **Task:** Pin `spencernorton/Armiger-Connectors` `v1.2.7` (`283f4461c18f2b873460c4cde8f43d03ac4e3c03`), harden single-file extension and variant alias handling, and add release contract + container acceptance checks for Bitmagnet `contentType` writeback and Torznab category promotion/rescue.
- **What changed:** Update `examples/Dockerfile` and `docs/bitmagnet-configuration.md` to `Armiger-Connectors` `v1.2.7` (`283f4461c18f2b873460c4cde8f43d03ac4e3c03`). Add `VARIANT_ALIASES` in `src/armiger/artifacts.py` and CLI variant aliases in `src/armiger/__main__.py`. Preserve single-file `extension`/`ext` hints in `src/armiger/armiger_inference.py` and `src/armiger/armiger_ultra_pipeline.py`. Enforce connector pin consistency in `ci/release_contract.py` and `ci/pin_connectors.py`, and verify Bitmagnet `/import` `contentType` writeback and Torznab `tvsearch`/`movie` rescue against live `bitmagnet:v0.10.0` in `ci/runtime_acceptance.py`. Bump software version to `3.8.1`.
- **Validation:** `python3 -m unittest discover -s ci/tests -v` and `pytest -q` pass.
- **Release:** Ready for `3.8.1` publication via CI.

## 2026-10-29 — Promote Armiger 3.8.0 and v3.0-gold-r13.8 model bundle

- **Task:** Release Armiger software and model bundle `3.8.0` (`v3.0-gold-r13.8`, DHT `seq=12` / `seq=3012`).
- **What changed:** Pin `NorviTech/armiger-ultra` revision `7de69d7a94def2425e7d873aff7df1c041ab03da` (`v3.8.0`) across `src/armiger/model_manifest.json`, `src/armiger/artifacts.py`, `src/armiger/oci_vpn_guard.py`, `src/armiger/calibration.json`, `src/armiger/oci_sqlite_catalog.py`, `src/armiger/armiger_ultra_pipeline.py`, and `examples/Dockerfile`. Refresh `benchmarks/CHARTS_MANIFEST.json`, `benchmarks/README_OVERVIEW.json`, and `docs/assets/*.png`, and bump package and Compose quickstarts to `3.8.0`.
- **Validation:** Verify `ci` release contract and promotion unit tests, external guide verification, and full `pytest` suite against the `3.8.0` model artifacts.
- **Release:** Ready for `3.8.0` publication via CI.

## 2026-10-28 — Validate release source in every native image config (3.4.3)

- **Branch:** codex/native-release-provenance
- **What changed:** Support Docker manifest lists as well as OCI indexes by validating every native config's source/version labels, complete supported platforms and any index annotation that is present. Tested native manifest/config digests remain required for promotion and immutable retries. Bump software and local quickstarts to 3.4.3 because the stopped 3.4.2 publication created an incomplete immutable image tag.
- **Validation:** Main 3.4.2 CI passed both native runtime/push jobs, Python suites and the aggregate before its final metadata assertion caught the Docker-list annotation mismatch. Independent receipt reads prove all three pushed config digests equal the tested local image IDs. New failure cases reject missing platforms, mixed/missing/invalid source labels, wrong versions and contradictory annotations. Real 3.4.1 OCI and partial 3.4.2 Docker-list source readback are checked before merging. Final 3.4.3 native CI and full publication remain required.
- **Release:** 3.4.2 remains an incomplete registry tag with no release announcement or mutable-alias promotion. Model bundle 3.4.0 is unchanged.

## 2026-10-28 — Compare equivalent legacy Docker and OCI configuration defaults

- **Branch:** codex/registry-config-defaults
- **What changed:** Accept omitted versus false values for the six legacy Docker container flags returned by older image-inspect APIs. Keep filesystem, user, command, environment and meaningful zero-valued settings subject to the existing strict comparison.
- **Validation:** Reproduce the publication failure using Docker API 1.48 inspection of the released 3.4.1 image and its independently fetched GHCR configuration. Add regression cases for both omission directions, enabled flags, unknown flags and explicit zero settings. The first 3.4.2 main run passed both native runtime suites and stopped before release promotion at this representation difference.
- **Release:** The 3.4.2 version and mutable aliases remain unpublished pending a complete passing publication run.

## 2026-10-28 — Prove upstream writes, filtered search and failed-release detection

- **Branch:** codex/runtime-acceptance
- **What changed:** Extend real-service acceptance to actual Torznab search admission and Rats HTTP-to-Manticore category writes with persisted readback and collision refusal. Keep already-published main runs subject to immutable-image/OCI completeness checks; a same-source retry cannot skip partial publication. Advance mutable aliases only after OCI verification. Use GitHub-owned actions and the installed Docker builder to honor the existing Actions policy.
- **Validation:** All three3.4.2 CPU-image entrypoints pass locally on native arm64, with the actual x86 Rats backend under emulation. Stock Bitmagnet processes350 initial records (2 unsafe deletions),50 later arrivals,6 canonical tier inputs and3 legacy repairs; real filtered search returns fixtures. Actual Rats category1/1 persistence and collision-preserved2/3 pass. The acceptance harness rejects an injected non-searchable connector directory and captures Errno13. Eight release/promotion failure-case tests, actionlint and read-only validation of the existing3.4.1 images/platforms/alias/OCI stacks pass. Earlier required native amd64/arm64 and Python3.10/3.12 jobs pass in CI40791280858; final expanded checks require a fresh run.
- **Release:**3.4.2 publication and final native CI remain pending. Model bundle3.4.0 and production deployments are unchanged.

## 2026-10-28 — Require native runtime and stock-upstream release acceptance (3.4.2)

- **Branch:** codex/runtime-acceptance
- **What changed:** Make release/version contracts, both native runtime checks, existing Python/model tests and installed-wheel smoke prerequisites of the required test check. Exercise actual documented standalone/Bitmagnet/Rats commands as UID10001. Add disposable stock Bitmagnet/Postgres fixtures for350-record pagination, deletion shifts, later arrivals, exact canonical tag persistence and legacy repair. Qualify the actual x86 Rats/Manticore backend. Publish only the tested filesystem/configuration, promote native manifest digests and read back OCI stack commands. Expose the same acceptance workflow to exact Connectors revisions. Use CPU PyTorch in the CPU image and bump software/quickstarts to3.4.2; model bundle stays3.4.0.
- **Validation:** Work in progress. Initial real3.4.1 native-arm64 startup/inference and stock Bitmagnet pagination/deletion/repair/new-wave readback pass. Seven release/promotion failure-case tests and actionlint pass. Candidate3.4.2 CPU image builds; full candidate and native CI acceptance remain in progress.
- **Release:** Pending required checks and post-merge publication proof; no production stack deployment.

## 2026-10-28 — Fix non-root connector startup permissions (3.4.1)

- **Branch:** fix/connector-permissions
- **What changed:** Restore directory traversal for `/opt/connectors`, both connector subdirectories and `/models` after remote `ADD --chmod=0444`. Keep downloaded files read-only, and verify both connector scripts and offline Ultra model integrity after switching to UID 10001. Bump the library and setup images to 3.4.1; retain model bundle 3.4.0.
- **Validation:** Reproduce the original permission error using the pinned connector ADD instructions; build the corrected image and check both connector entrypoints and offline model access as its default user. Run the unit suite, installed-wheel smoke, and all three Compose configuration checks.
- **Follow-ups:** Merge after required checks and verify the published 3.4.1 image.

## 2026-10-27 — NorviTechGH — Align .torrent bundle names and metadata with v3.4.0 (seq=10)

- **Branch:** fix/v340-torrent-bundle-names
- **What changed:** Update `.torrent` bundle names (`armiger-weights-{variant}-v3.4.0`, `armiger-zero-cue-db-v3.4.0`, `armiger-v3.4.0-gold-r13-4`) and embedded metainfo `comment` / `created by` strings from `v2.1.0` to `v3.4.0`, and pin `NorviTech/armiger-ultra` commit `aa760eeea91fba887124e535dcd1df7496591cea`.

## 2026-10-27 — NorviTechGH — Release 3.4.0: v3.0-gold-r13.4 model bundle, DHT seq=9 pointers, and updated benchmark charts

- **Branch:** release/v3.4.0
- **What changed:** Pin `NorviTech/armiger-ultra` revision `43c4a795e1a2e55826478b4b57ff7c95e3c77730` (`v3.4.0`, `v3.0-gold-r13.4`, DHT `seq=9` / `seq=3009`) across `src/armiger/model_manifest.json`, `src/armiger/artifacts.py`, `src/armiger/oci_vpn_guard.py`, and `examples/Dockerfile`. Sync `calibration.json` with the `v3.4.0` bundle parameters, refresh `benchmarks/updated_std_charts_data.json` and all 5 canonical evaluation and README overview charts (`91.87%` 10-class category accuracy, `85.08%` video slice, `84.44%` title matching, `90.20%` byte-verified speech), and bump package and container tags to `3.4.0`.
- **Validation:** `pytest -q` (`benchmarks/test_chart_integrity.py`, `tests/test_readme_overview.py`, and full unit suite), live Mainline DHT + BitTorrent P2P + BEP-19 webseed verification (`seq=9`), and `git diff --check`.
- **Follow-ups:** Merge PR, tag `v3.4.0`, and publish GitHub Release with `.torrent` assets.

## 2026-10-27 — NorviTechGH — Pin Armiger-Connectors v1.2.5 for canonical Bitmagnet viability tags and automatic legacy tag repair (v3.3.4)

- **Branch:** release/v3.3.4-bitmagnet-viability-tags
- **What changed:** Pin `spencernorton/Armiger-Connectors` `9b00089a06df24497e45d2f2144eb14a1bb03ac2` (`v1.2.5`) in `examples/Dockerfile`, update `examples/bitmagnet/README.md` and `docs/bitmagnet-configuration.md` with canonical swarm viability tags (`armiger-viability-alive-viable`, `armiger-viability-degraded`, `armiger-viability-stalled`, `armiger-viability-dead`, `armiger-viability-unknown-cold`) and automatic migration of legacy hash-truncated tags (`#42`), and bump version to `3.3.4`.
- **Validation:** `pytest -q`, `docker compose config --quiet` across all 3 example stacks, and `git diff --check`.
- **Follow-ups:** Merge PR and tag `v3.3.4` after `Armiger-Connectors` `v1.2.5` merges.

## 2026-10-27 — Refresh 2026-10-26 held-out chart data and validate finite overview scores

- **Branch:** claude/fix-adult-chart-nan
- **What changed:** Update `benchmarks/updated_std_charts_data.json` to the complete 2026-10-26 held-out evaluation output (`N = 2,484` across all 10 categories, including `n = 75` and `100.0%` recall on `Adult / XXX` and `95.65%` on `Comic`). Validate positive sample counts and finite `[0, 100]` scores via `score(...)` inside `benchmarks/generate_readme_overview.py` (`render()`) so `NaN` or empty category strata fail fast, and regenerate all 5 evaluation PNGs (`docs/assets/armiger_overview.png`, `docs/assets/armiger_categorical_breakdown.png`, `docs/assets/armiger_vs_llms_overall_and_6dims.png`, `docs/assets/armiger_vs_llms_cost_comparison.png`, `docs/assets/armiger_vs_torrent_classifiers_6dims.png`) and their provenance manifests (`benchmarks/CHARTS_MANIFEST.json`, `benchmarks/README_OVERVIEW.json`).
- **Validation:** Verified `pytest -q` (`benchmarks/test_chart_integrity.py`, `tests/test_readme_overview.py`, and full unit suite) and `git diff --check`.
- **Follow-ups:** None.

## 2026-10-27 — Release 3.3.3: BEP 46 webseed metadata alignment, KRPC interop, and HF repo consolidation

- **Branch:** release/v3.3.3-dht-webseed-and-hf-cleanup
- **What changed:** Pin `NorviTech/armiger-ultra` revision `18ef5418b3858ee4f70c137540febb5343b8754e` (with canonical `<infohash>.torrent` / `bundle.torrent` metadata, `zero_cue_catalog.sqlite.zlib`, and cleaned `config.json`) across `src/armiger/model_manifest.json`, `src/armiger/artifacts.py`, `src/armiger/oci_vpn_guard.py`, and `examples/Dockerfile`. Fix package-prefixed import (`from armiger.oci_sqlite_catalog import _https_get_with_deadline`) and live webseed payload budget in `src/armiger/verify_dht.py`. Allow unsorted KRPC framing keys in `bdecode_krpc` (`allow_unsorted_keys=True`, while preserving strict duplicate-key rejection and strict canonical key ordering in `bdecode`) so `libtorrent` Mainline DHT routers (`dht.libtorrent.org:25401`) are not dropped. Reorder `DEFAULT_DHT_BOOTSTRAP_NODES` in `src/armiger/oci_dht_seeder.py` and prioritize bootstrap entry nodes in `iterative_find_closest`. Bump package and image version to `3.3.3`.
- **Validation:** Verified full pytest suite, `git diff --check`, and live Mainline DHT + BEP 19 webseed verification.
- **Follow-ups:** None.

## 2026-10-26 — Document all Bitmagnet + Armiger environment variables, setup options, and usage workflows

- **Branch:** docs/bitmagnet-env-and-usage-guide
- **What changed:** Expand `examples/bitmagnet/README.md` and `docs/bitmagnet-configuration.md` to document every environment variable across the Bitmagnet Compose stack, `bitmagnet_connector.py`, and `ArmigerUltraPipeline` (including stack ports/credentials, `DHT_CRAWLER_SCALING_FACTOR` + 60-second `persistTorrents` polling math, Torznab pre-grab filter flags, index pruning, model/catalog settings, and all `v3.3.0+` BEP 44/46 auto-update and opt-in contribution variables). Add clear step-by-step instructions for Option A (two-line OCI `include`), Option B (full `docker-compose.yml` / Portainer Stacks / `.env`), and Option C (attaching `armiger-ultra` to an existing Bitmagnet installation), plus usage instructions for `armiger-*` tags in Bitmagnet (`:3333`), Prowlarr/`*Arr` Torznab filtering (`:8097/torznab`), and HTTP health/classification endpoints (`:8096`). Wire `${VAR:-default}` interpolation in `examples/bitmagnet/docker-compose.yml` and expand `examples/bitmagnet/compose.override.example.yaml`.
- **Validation:** Verified all environment variable names and defaults against `examples/bitmagnet/docker-compose.yml`, `bitmagnet_connector.py` (`v1.2.3`), and `src/armiger/` (`v3.3.2`), ran `docker compose config --quiet`, and ran `pytest -q`.
- **Follow-ups:** None.

## 2026-10-26 — Clarify Bitmagnet env variable options, pre-populated defaults, and literal Compose values

### What changed
- Replaced `${VAR:-default}` variable interpolation in `examples/bitmagnet/docker-compose.yml` with literal default values (`ghcr.io/spencernorton/armiger:3.3.2`) and embedded the full copy-pasteable `docker-compose.yml` directly into `examples/bitmagnet/README.md`.
- Added explicit `Required / Optional` and `Valid options / range` columns to all environment variable tables in `examples/bitmagnet/README.md` and `docs/bitmagnet-configuration.md`, plus a clear section explaining which settings are required vs. optional.
- Simplified the distributed model updates section so the 5 user-facing toggles are presented clearly and the 4 pre-populated release-channel/mirror defaults are placed in an expandable advanced section with their actual built-in values.

### Why
- Make `examples/bitmagnet/docker-compose.yml` directly copy-pasteable without `.env` files or variable interpolation, show the valid options/ranges for every setting, and clarify that distributed updates are pre-configured out of the box.

## 2026-10-26 — Release model bundle 3.3.0 (R13.3), updated evaluation charts, and configurable DHT auto-update

- **Branch:** release/v3.3.2-r13-3
- **What changed:** Pin the `3.3.0` (`v3.0-gold-r13.3`) `NorviTech/armiger-ultra` revision (`cd8a5fbed4245557999ce3aafd168a568edd7fdd`) in `src/armiger/model_manifest.json`, `src/armiger/artifacts.py`, `src/armiger/oci_vpn_guard.py`, and `examples/Dockerfile`; add R13.3 41-stat feature support (`r13_features.py`) and updated `calibration.json`; regenerate the 5 canonical and overview evaluation charts from `benchmarks/updated_std_charts_data.json`; expose `ARMIGER_DHT_AUTO_UPDATE` across `examples/Dockerfile` and Compose files; and publish pre-built `ghcr.io/spencernorton/armiger-ultra` container tags in `.github/workflows/ci.yml`.
- **Validation:** Verified full pytest suite (`290 passed`), `tests/smoke_installed.py`, `docker compose config --quiet` on all 3 example manifests, and `git diff --check`.
- **Deployment:** Public release `3.3.2` publishes via GitHub Actions on merge.
- **Status file updated:** Yes.

## 2026-10-26 — Pin Bitmagnet poller pagination fix and clarify Portainer and polling docs

- **Branch:** fix/bitmagnet-poller-pagination-and-docs
- **What changed:** Pin `spencernorton/Armiger-Connectors` `58679a4f0fd28de4ebee9133a23a6a10bb6adaf0` (`v1.2.3`) in `examples/Dockerfile` so `BitmagnetConnector.sync_once` paginates past items tagged earlier in an active 60-second Bitmagnet crawl flush (`#37`). Expose `BITMAGNET_POLL_SECONDS` and `BITMAGNET_BATCH_SIZE` in `examples/bitmagnet/docker-compose.yml`, align override examples with the `5`s / `100`-item defaults, document how Bitmagnet's 60-second / 1,000-item `persistTorrents` batching channel interacts with polling and `DHT_CRAWLER_SCALING_FACTOR`, and point Portainer users directly to the self-contained `docker-compose.yml`. Bump version to `3.3.1` to publish updated container images and OCI Compose stacks.
- **Validation:** Verified Compose manifests (`docker compose config --quiet`), `git diff --check`, and unit test suite.
- **Deployment:** None; public release `3.3.1` publishes via GitHub Actions on merge.
- **Status file updated:** Yes.

## 2026-10-26 — Wire BEP-44/46 auto-update poller, writable overlay hot-swap, and opt-in contribution dispatcher

- **Branch:** feat/dht-autoupdate-and-contribution-loop
- **What changed:** Pin canonical release Ed25519 public key (`3599362bd52dad8b89398b904d3081c2cb483d13376f7ffb49f3748208731e76`) and default salt (`armiger:weights:ultra`) in `verify_dht.py` and `oci_vpn_guard.py`. Implement background BEP-44/46 auto-update poller (`ARMIGER_DHT_AUTO_UPDATE`, `ARMIGER_DHT_POLL_INTERVAL_SEC`) and atomic writable-overlay hot-swap (`ARMIGER_RUNTIME_CACHE_DIR`) in `ArmigerUltraPipeline`, plus `/healthz` observability fields (`active_seq`, `active_bundle_merkle_root`, `model_source`, `last_dht_check_ts`, `sharing_status()`). Wire opt-in UDP contribution dispatcher (`ARMIGER_DHT_SHARE_OPT_IN`, strictly defaulting to `false` / `pull_only`) in `ArmigerDHTRSIEngine` (`dispatch_contribution_udp`) and collector ingestion (`ingest_signed_delta_bundle`). Pin merged `Armiger-Connectors` `a18596f8a371afcd7df339f3fe6fb288749f55fe` (`v1.2.1`) in `examples/Dockerfile` and bump release version to `3.3.0`.
- **Validation:** Verified unit and adversarial fault tests (`tests/plugin/test_dht_autoupdate_and_contribution_loop.py`, `tests/test_server.py`, `tests/test_artifacts.py`, `tests/test_armiger_ultra_pipeline.py`, `tests/test_distributed_knowledge_transfer.py`) and Compose manifests.
- **Follow-ups:** None.

## 2026-10-24 — Show concrete contributing, attesting and receiving node roles in diagram

- **Branch:** docs/peer-validation-diagram
- **What changed:** Replace the single `Subset of nodes confirm & sign` box in `docs/distributed-updates.md` with explicit node groups matching the implementation: `Nodes A, B, C` (contribute verified imports/deltas) -> `Aggregator` (builds candidate hash or weight delta) -> `Nodes D, E, F` (peer-test against local ledger + 500-item canary across distinct ASNs and sign `PeerAttestation`s) -> `Promote seq` (`M-of-N` federation signers + BEP 44/46 announcement) -> `Nodes G, H` (verify quorum, run local canary check, adopt `seq` or roll back).
- **Validation:** Verified all local/external links and anchors, JSON block syntax, Mermaid rendering, and `git diff --check`.
- **Follow-ups:** None; documentation only.

## 2026-10-24 — Restructure distributed updates guide around Problem, Solution and peer quorum promotion

- **Branch:** docs/problem-solution-how-it-works
- **What changed:** Restructure `docs/distributed-updates.md` to open with `## The Problem`, `## The Solution` and `## How it works`, replacing the single-file loop diagram with a multi-peer fan-in diagram showing peers contributing feedback on plugin hash files and weights, confirmation across a subset of independent nodes, and promotion to a new `seq` version via BEP 44/46. Explicitly note the federated/quorum-gated boundary on version promotion alongside the existing technical reference.
- **Validation:** Verified all local/external links and anchors, JSON block syntax, Mermaid rendering on GitHub, and `git diff --check`.
- **Follow-ups:** None; documentation only.

## 2026-10-24 — Refine distributed updates guide and technical reference

- **Branch:** docs/distributed-updates-precision
- **What changed:** Clarify in the opening that ordinary classification never sends feedback or fetches updates automatically, and state what requires explicit configuration for contributing versus receiving. Keep the overview and four-stage Mermaid loop approachable while consolidating comparison semantics into the four workflow steps. Expand the technical reference to distinguish `armiger-catalog-v1` unkeyed catalog hashes (`export_rsi_telemetry_bundle` / `zero_cue_catalog_delta`) from `armiger-catalog-v2` keyed hashes (`AnonymityGuard`), document `RSIUpdateAggregator` versus `ByzantineResilientAggregator` thresholds and 3-tensor `SwarmResidualAdapter` training behavior, contrast peer attestation ledger confirmation (`require_ledger_confirmation=True`) with receiver baseline comparison, note the single-process `_GLOBAL_PINNED_PUBKEY_ASN` contributor registration behavior, and detail how `sign_manifest()`, the `seq=3006` release receipt and `tests/bep44_udp_wire_harness.py` differ from BEP 44/46.
- **Validation:** Verified all local file/anchor links and external BEP links, validated embedded JSON and Mermaid syntax, ran `git diff --check`, and executed `pytest -q tests/test_anonymous_opt_in_dht_sharing.py tests/test_distributed_knowledge_transfer.py`.
- **Follow-ups:** None; documentation only.

## 2026-10-24 — Verify the protocol reference presentation

- **Branch:** codex/distributed-bep-reference
- **What changed:** Use a textual reference to the expandable implementation note; GitHub does not automatically open its details block when following a fragment link to the hidden heading.
- **Validation:** Fourteen local links/anchors and both primary specification links remain. JSON, code fences, whitespace and the diagram pass; the visible BEP section and the diagram's BEP 44/46 label render on GitHub.
- **Deployment:** Documentation only; no runtime or model changes.

## 2026-10-24 — Explain BEP 44 and BEP 46

- **Branch:** codex/distributed-bep-reference
- **What changed:** Name BEP 44/46 on the discovery link and explain signed mutable announcements, torrent-update pointers and separate bundle retrieval. Link both primary specifications and keep the four-stage learning loop intact.
- **Evidence:** Compare the official BEPs with the mutable-item harness, custom UDP bundle messages, release pointer receipts and update bridge contract. Distinguish Armiger's application manifests and local test transport from full BEP 46 client interoperability.
- **Validation:** Fifteen local links/anchors, both primary specification pages, JSON, code fences and patch whitespace pass. The revised Mermaid loop renders and has been visually inspected. Runtime and build verification remain required CI gates.
- **Deployment:** Documentation only; no model, runtime or production setting changed.
- **Status file updated:** Record the protocol clarification and publication evidence in the private project handoff.

## 2026-10-24 — Show the shared-learning feedback loop

- **Branch:** codex/distributed-learning-loop
- **What changed:** Replace the delivery-focused diagram with the contribution, aggregation, peer review, publication and receiver feedback loop. Explain that verified feedback and candidate changes can contribute without an instance winning an accuracy ranking.
- **Evidence:** Check RSI feedback export, catalog consensus, calibration fitting, adapter training, continuous-delta aggregation and peer/receiver evaluation. Link the multi-instance knowledge-transfer tests. Distinguish encrypted individual feedback from the guarded aggregate export format.
- **Validation:** Twelve local links/anchors, the JSON example, code fences and patch whitespace pass. The four-stage Mermaid loop renders and has been visually inspected. Runtime and build verification remain required CI gates.
- **Deployment:** Documentation only; no model, runtime or production settings changed.
- **Status file updated:** Record the correction and publication evidence in the private project handoff.

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
