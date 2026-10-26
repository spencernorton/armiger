# Changelog

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
