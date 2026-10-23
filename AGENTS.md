# Armiger

Local ONNX classification of torrent and media metadata, with an optional Ultra enrichment pipeline.

- Install: `python -m pip install -e '.[ultra,dev]'`.
- Models: `python -m armiger download --variant all`; `ARMIGER_MODEL_DIR` selects a cache.
- Verify: `python -m pytest -q` and `python -m build`; run the installed-wheel smoke check outside the checkout.
- Keep the three setup paths clear: Bitmagnet Compose, Rats Search Compose, standalone Ultra. Benchmark detail belongs in `docs/evaluation.md`.
- Keep model assets pinned and checksummed; do not import code from model downloads.
- Use `codex/`, `claude/` or `spencer/` branches. See [the journal](AGENTS/journal.md). Private operating context lives in `shared-ops/projects/status/bitagent.md`.
