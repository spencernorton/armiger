# Changelog

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
