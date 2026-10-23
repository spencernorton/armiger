# Armiger

Fast local classification of torrent and media metadata, with ONNX inference and optional enrichment. Armiger reads names, file paths, sizes and optional swarm counters; it does not inspect media payloads.

**Armiger Ultra** combines the model with catalog matching, audio-language rules, swarm viability estimates, calibration and metadata safety checks. It runs on a CPU; a GPU is not required.

## Get started

| Setup | Guide |
| --- | --- |
| **Armiger Ultra** — use the complete pipeline locally | [Install and run](docs/quickstart.md) |
| **Bitmagnet + Armiger Ultra** — start the indexer, database and classifier with Docker Compose | [Bitmagnet setup](examples/bitmagnet/README.md) |
| **Rats Search + Armiger Ultra** — run the classifier alongside Rats Search | [Rats Search setup](examples/rats-search/README.md) |

For local Python use, install from this checkout and download the pinned model bundle:

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[ultra]'
python -m armiger download --directory ./models
```

```python
from armiger import ArmigerUltraPipeline

pipeline = ArmigerUltraPipeline.from_directory("./models", preload_catalog=True)
result = pipeline.classify({
    "name": "Tears.of.Steel.2012.1080p.BluRay.x264-OPEN",
    "size": 2_400_000_000,
    "files": [{"path": "Tears.of.Steel.2012.mkv", "size": 2_400_000_000}],
})
print(result["category"])  # movie
```

The [Ultra guide](docs/quickstart.md) covers prerequisites, model storage, CPU-only inference and a command-line check. Model downloads are checksummed and cached; inference can run offline after setup. Sharing is disabled unless explicitly enabled.

## What it returns

The model predicts content category, adult/junk indicators, anime, audio/title language, subtitle availability, catalog scope and title/year/episode spans. Ultra adds catalog disambiguation and swarm estimates. See the [API and component reference](docs/reference.md) for inputs, outputs and configuration.

Metadata-derived language, swarm and safety predictions can be wrong. They are not payload inspection or proof that a download is safe. [Evaluation](docs/evaluation.md) documents measured results, label coverage and current release-gate limitations.

## Documentation and support

- [Distributed updates and sharing](docs/distributed-updates.md) — optional advanced features.
- [Additional connectors](https://github.com/spencernorton/Armiger-Connectors) — DH-Waypoint, Torznab and Arr integrations.
- [Contributing](.github/CONTRIBUTING.md) · [Issues](https://github.com/spencernorton/armiger/issues) · [Security reporting](.github/SECURITY.md).

Library **3.1.0** uses the pinned **3.0.0** model bundle. Weights are hosted on [Hugging Face](https://huggingface.co/NorviTech/armiger-ultra); source and reproducible evaluation tools live here. Apache-2.0: [license](LICENSE), [notices](NOTICE), [use policy](.github/LEGAL.md).
