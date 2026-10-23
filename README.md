# Armiger

Armiger helps Bitmagnet and Rats Search label torrents by type and language, using names and file lists. **Armiger Ultra** also adds title matching, audio-language checks and estimates of download availability. It runs on a CPU; no GPU or paid API is required.

## Choose your setup

1. **[Bitmagnet + Armiger Ultra](examples/bitmagnet/README.md)** — classify items and add searchable tags in Bitmagnet.
2. **[Rats Search + Armiger Ultra](examples/rats-search/README.md)** — classify items in Rats Search.
3. **[Standalone Armiger Ultra](docs/quickstart.md)** — run Armiger on its own or use it from Python.

Choose one guide. The Docker setups include Armiger Ultra, so you do **not** need to install the standalone version first. Each Docker guide explains what it starts and what to do if you already use that indexer.

## What Armiger does

Armiger reads names, file paths, sizes and optional swarm counters. It predicts content type, language, subtitle availability and other metadata. Ultra combines those predictions with its catalog and additional checks.

It does not inspect media payloads. Metadata predictions can be wrong and do not prove that a download is safe. See [evaluation and limitations](docs/evaluation.md) for measured results and label coverage. Sharing is off by default.

## More information

- [Armiger on NorviTech](https://norvitech.com/ai-models/#armiger)
- [Python API and component reference](docs/reference.md)
- [Optional distributed updates](docs/distributed-updates.md)
- [Other integrations](https://github.com/spencernorton/Armiger-Connectors) — DH-Waypoint, Torznab and Arr.
- [Help and bug reports](https://github.com/spencernorton/armiger/issues) · [Contributing](.github/CONTRIBUTING.md) · [Security reporting](.github/SECURITY.md)

Library **3.1.0** uses model bundle **3.0.0**. Model files are downloaded from a fixed [Hugging Face](https://huggingface.co/NorviTech/armiger-ultra) revision and verified before use. Apache-2.0: [license](LICENSE), [notices](NOTICE), [use policy](.github/LEGAL.md).
