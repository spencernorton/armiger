# Changelog

## 3.13.2

- Pin `spencernorton/Armiger-Connectors` `v1.2.10` (`99dcc59b8e733b4063bf09f61bcb80b73e802bd1`) in `examples/Dockerfile`. The Bitmagnet `/torznab` proxy now runs unclassified release rescue on ID-based searches (`imdbid`, `tmdbid`, `tvdbid`) by resolving canonical titles through Bitmagnet's GraphQL `content.search` with a bounded 24-hour cache, keeping linked items first and enforcing requested `season`/`ep` filters ([Armiger-Connectors#26](https://github.com/spencernorton/Armiger-Connectors/pull/26), [Armiger-Connectors#20](https://github.com/spencernorton/Armiger-Connectors/issues/20)). Model bundle 3.12.0 is unchanged.

## 3.13.1

- Pin `spencernorton/Armiger-Connectors` `v1.2.9` (`eadc6c40b62473e2ee7ce0def542ee539b0a5f60`) in `examples/Dockerfile`. The Bitmagnet poller's `contentType` repair now queries one `armiger-*` tag per request instead of one OR across all ten, which never returned on very large catalogs ([Armiger-Connectors#20](https://github.com/spencernorton/Armiger-Connectors/issues/20)). A failed repair no longer skips classification in the same cycle.
- Add `BITMAGNET_GRAPHQL_TIMEOUT_SECONDS` (default `10`) for catalogs where Postgres needs longer than 10 seconds per GraphQL query. Model bundle 3.12.0 is unchanged.

## 3.13.0

- Update the pinned model bundle to `3.12.0` (`v3.0-gold-r13.13` on `NorviTech/armiger-ultra` at `41cb0db4c3dfce2e3866fb67d414f0ca7026af2a`, DHT `seq=15` / `seq=3015`), refreshing `config.json` (`sha256=e3133bf37b9bafb34bbf8ff240c8562c2a1063aabadf6df6460c80d8046aae06`) and regenerating the benchmark and README overview figures.

Historical entries below preserve earlier announcements. Their numerical, timing, privacy and safety wording is qualified by the current [evaluation status](docs/evaluation.md) and [resource/prediction limits](docs/resources.md); it is not current-release validation.

## 3.12.2

- Clarify model-file size versus working RAM and the distinct runtime paths. Describe audio-language, junk and availability outputs as metadata estimates, with no payload-safety guarantee.
- Withdraw current benchmark promotion pending reconciliation of displayed model/cohort values and their receipts. Preserve historical data and graphics; this change introduces no new evaluation or model qualification.
- Correct stale model-file sizes and require an explicit update-channel choice in setup instructions. Model bundle3.11.0 and inference behavior are unchanged.

## 3.12.1

- Model bundle remains `3.11.0` (`v3.0-gold-r13.11`, DHT `seq=14` / `seq=3014`) unchanged. Require an explicit model update channel: set `ARMIGER_UPDATE_CHANNEL` to `dht` (DHT peer-wire first, Hugging Face webseed fallback), `hf` (Hugging Face webseed), or `off`. Background updating no longer starts without a channel; health output reports `update_channel` (`not_configured` when unset). The legacy `ARMIGER_DHT_AUTO_UPDATE` switch still maps to `dht`/`off`, and Docker Compose examples now require the variable to be set.

## 3.11.0

- Update the pinned model bundle to `3.11.0` (`v3.0-gold-r13.11` on `NorviTech/armiger-ultra` at `1ac1b682f36a6d371aef2a9183aef032a19be98f`, DHT `seq=14` / `seq=3014`), enforcing zero train-split regex-only tokens (`0` zero-train tokens across `3,810` regex tokens), eliminating post-hoc evaluation overrides by integrating runtime post-catalog resolution into `r13_8_combiner_runtime.py`, and retraining the GPU 255-dim `ResidualCombinerNet` (`r13_8_combiner.pkl` `sha256=7d055151ee403fa766cc0e4b72812870ca11dde6e357cc98636989c484e6e848`) on the conserved Gold neural backbone (`97.29%` Clean v3 9-category Macro-F1, `92.70%` strictly disjoint Fresh Balanced 1,000 Macro-F1, `88.78%` Gold Val Macro-F1, and `81.86%` Gold Test Macro-F1).
- Sync `calibration.json` (`sha256=27d04c6e0832b7f8bce3d9564497d85f6cca7bf4c90895c07bc8417c243f64ef`) and `config.json` (`sha256=87d3255383f800023bde1d596dbd85dd4d23098c30945213623ee24d4d71911d`) bundle manifests, and refresh the canonical evaluation and overview charts.

## 3.10.0

- Update the pinned model bundle to `3.10.0` (`v3.0-gold-r13.10` on `NorviTech/armiger-ultra` at `b9f70a0448fc98f4113126552f46bbd42eaecf0e`, DHT `seq=13` / `seq=3013`), eliminating curated validation stems (`CURATED_CANONICAL_STEMS = {}`), recovering `Suite 2 (W5A)` title span extraction (`71.89%` exact-row, `0.00pp` drift vs `R13.7`), and achieving `97.29%` Clean v3 9-category Macro-F1 (`N=1,355`, 9/9 categories `>= 95.45%` F1).
- Sync `calibration.json` and `config.json` bundle manifests (`r13_6_lang_llr.json` + `r13_2_audio_language_guard.py` audio-guard bundle parity), and refresh the canonical evaluation and overview charts (`93.32%` 10-class category accuracy, `87.50%` video slice, `86.86%` title matching, `91.54%` byte-verified speech / `95.00%` file-tree).

## 3.8.2

- Explain the local classifier, optional Ultra pipeline and integration outputs with a README architecture diagram. Model bundle 3.8.0 is unchanged.

## 3.8.1

- Pin `spencernorton/Armiger-Connectors` `v1.2.7` (`283f4461c18f2b873460c4cde8f43d03ac4e3c03`) in `examples/Dockerfile` so published `ghcr.io/spencernorton/armiger:3.8.1` and `:latest` images include Bitmagnet `POST /import` `contentType` hint writeback, Torznab `Unknown` (`8000`) category promotion and `tvsearch`/`movie` category rescue, hardened `repair_unknown_content_types` pagination, and bounded JSON request validation.
- Support `"int8"`, `"armiger"`, and `"fp32"` variant aliases in `armiger.artifacts.download_model` and `python -m armiger` CLI commands, and preserve single-file `extension`/`ext` hints in `encode_torrent_features` and `_file_tree_structural_override`.
- Verify `examples/Dockerfile` and `docs/bitmagnet-configuration.md` connector pins in `ci/release_contract.py`, and exercise live Bitmagnet `/import` `contentType` writeback and Torznab category rescue in `ci/runtime_acceptance.py`.

## 3.8.0

- Update the pinned model bundle to `3.8.0` (`v3.0-gold-r13.8` on `NorviTech/armiger-ultra` at `7de69d7a94def2425e7d873aff7df1c041ab03da`, DHT `seq=12` / `seq=3012`), sync the `v3.8.0` `calibration.json` parameters and bundle plugin manifests, and refresh the canonical evaluation and overview charts (`91.87%` 10-class category accuracy, `85.08%` video slice, `84.44%` title matching, `90.20%` byte-verified speech).

## 3.4.1

- Fix `Permission denied` on fresh Bitmagnet and Rats Search connector startup by restoring searchable permissions on bundled connector and model directories while keeping downloaded files read-only (#48).
- Verify both connector scripts and every bundled Ultra model file as the non-root runtime user during image builds. Update all supported Compose setups to the corrected image.

## 3.4.0

- Update the pinned model bundle to `3.4.0` (`v3.0-gold-r13.4` on `NorviTech/armiger-ultra` at `43c4a795e1a2e55826478b4b57ff7c95e3c77730`, DHT `seq=9` / `seq=3009`), sync the `v3.4.0` `calibration.json` parameters and bundle plugin manifests, and refresh the canonical evaluation and overview charts (`91.87%` 10-class category accuracy, `85.08%` video slice, `84.44%` title matching, `90.20%` byte-verified speech).

## 3.3.4

- Pin `spencernorton/Armiger-Connectors` `9b00089a06df24497e45d2f2144eb14a1bb03ac2` (`v1.2.5`) in `examples/Dockerfile` to emit canonical human-readable Bitmagnet swarm viability tags (`armiger-viability-alive-viable`, `armiger-viability-degraded`, `armiger-viability-stalled`, `armiger-viability-dead`, `armiger-viability-unknown-cold`) and automatically migrate and remove legacy hash-truncated viability tags (`armiger-viability-dea-c24281f7`, `armiger-viability-deg-7c170f8b`, `armiger-viability-sus-cdc9d2c6`) on startup (`#42`).

## 3.3.3

- Pin `NorviTech/armiger-ultra` revision `18ef5418b3858ee4f70c137540febb5343b8754e` with canonical BEP 46 `.torrent` metadata and `zero_cue_catalog.sqlite.zlib` webseed assets, fix package-qualified import and webseed transfer budget in `verify_dht.py`, accept unsorted KRPC framing keys (`bdecode_krpc`) from Mainline `libtorrent` routers, and prioritize responsive Mainline DHT bootstrap routers.

## 3.3.2

- Update the pinned model bundle to `3.3.0` (`v3.0-gold-r13.3` on `NorviTech/armiger-ultra`), refresh the canonical evaluation and overview charts, publish `ghcr.io/spencernorton/armiger-ultra` tags, and expose `ARMIGER_DHT_AUTO_UPDATE` in the Docker and Compose setups.

## 3.3.1

- Paginate the Bitmagnet poller across multi-batch 60-second crawl flushes (`#37`), expose `BITMAGNET_POLL_SECONDS` and `BITMAGNET_BATCH_SIZE` in the full Bitmagnet Compose file, and document Portainer stack setup and crawler scaling.

## 3.3.0

- Background BEP 44/46 auto-update polling, verified runtime overlay hot-swap, and opt-in contribution push.

## 3.2.1

- Fix Torznab XML filtering when upstream indexers such as Prowlarr send numeric file counts in the `files` attribute.

## 3.2.0

- Prebuilt Armiger images with the verified Ultra model for x86-64 and ARM64, plus a pinned Rats Search backend image.
- Ready-to-run fresh Compose setups without cloning, building or required environment files.
- Standalone Ultra HTTP classification service.

## 3.1.0

- Installable Python package with verified, pinned model downloads and offline loading.
- Consolidated documentation and prominent Ultra, Bitmagnet and Rats Search setup paths.
- Unified tests and installed-wheel verification.

Model weights remain at version 3.0.0. Historical details are in [GitHub Releases](https://github.com/spencernorton/armiger/releases).
