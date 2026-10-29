# Armiger

Local ONNX classification of torrent and media metadata, with an optional Ultra enrichment pipeline.

- Install: `python -m pip install -e '.[ultra,dev]'`.
- Models: `python -m armiger download --variant all`; `ARMIGER_MODEL_DIR` selects a cache.
- Verify: `python -m pytest -q` and `python -m build`; run the installed-wheel smoke check outside the checkout.
- Image, connector or setup changes also require the native runtime and stock-upstream checks described in [release validation](docs/release-validation.md). The required `test` check includes these; Compose syntax alone is insufficient.
- Bump the software version for image-affecting changes. After publication is verified, explain the fix and link its release notes in the issue before closing it.
- Keep the three setup paths clear: Bitmagnet Compose, Rats Search Compose, standalone Ultra. Benchmark detail belongs in `docs/evaluation.md`.
- Keep model assets pinned and checksummed; do not import code from model downloads.
- Use `codex/`, `claude/` or `spencer/` branches. See [the journal](AGENTS/journal.md). Private operating context lives in `shared-ops/projects/status/bitagent.md`.
