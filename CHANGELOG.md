# Changelog

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
