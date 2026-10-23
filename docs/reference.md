# Python reference

## Input and output

`ArmigerUltraPipeline.classify(item)` takes a dictionary containing `name`, optional total `size`, `files` (`path` and `size` per entry), and optional `swarm`/source counters. It returns an `ArmigerUltraPrediction` supporting dictionary-style access and `.to_dict()`.

The nine neural heads are `category`, `adult`, `junk`, `anime`, `english_subs`, `lang`, `title_language`, `owner_scope`, and byte-level `span_bio`. Category labels are `movie`, `tv_show`, `xxx`, `music`, `comic`, `game`, `audiobook`, `software`, `ebook`, `course`, and `unknown`. Spans use a seven-tag BIO decoder for title, year and episode boundaries.

Ultra additionally returns `swarm_viability`, `honeypot_trap`, `csam_safety`, `mondrian_gates` and `plugin_trace`. A passed gate is a model/policy decision, not a guarantee about payload contents. See [evaluation and limitations](evaluation.md).

## Load the full pipeline

```python
from armiger import ArmigerUltraPipeline

pipeline = ArmigerUltraPipeline.from_directory(
    "./models", variant="ultra", preload_catalog=True,
    rsi_mode="pull_only", opt_in_sharing=False,
)
```

Use `variant="int8"` to run the pipeline with the INT8 model (download `--variant all` first). `from_directory` loads local data; it does not fetch missing models. `ARMIGER_MODEL_DIR` sets the default directory for the constructor. Download and check data with `python -m armiger download`; the model manifest records its immutable upstream revision and file hashes.

## Model storage

`python -m armiger download` and the default Python constructor share a versioned cache under `~/.cache/armiger/`. Set `ARMIGER_MODEL_DIR` to choose a different folder. Alternatively, download with `--directory ./models` and load with `ArmigerUltraPipeline.from_directory("./models")`. The CLI uses `--model-dir ./models` for classification. Keep the same location for downloading and loading.

`python -m armiger download --offline` verifies cached Ultra files without accessing the network. Model data is reused between sessions; download again if a new library release selects a different model revision.

## Standalone ONNX

For only the lightweight model, install the package without the Ultra extra and download its CPU files:

```bash
python -m pip install .
python -m armiger download --directory ./models-cpu --variant cpu
```

This path needs NumPy and ONNX Runtime, but no PyTorch, catalog or swarm adapter.

```python
from armiger import ArmigerStandaloneONNX

model = ArmigerStandaloneONNX(
    onnx_path="./models-cpu/armiger.int8.onnx",
    vocab_path="./models-cpu/vocab.json",
    calibration_path="./models-cpu/calibration.json",
)
result = model.predict({"name": "Sintel.2010.1080p.WEBM", "files": []})
```

`predict` and `predict_batch` return dictionaries with `heads`, probabilities, extracted identity and gate results. The standalone path omits the Ultra enrichment components.

## Ultra components

| Component | Module | Role |
| --- | --- | --- |
| Swarm survival | `armiger.swarm_survival` | Residual adapter estimating swarm viability; needs `swarm_adapter.pt` and PyTorch |
| Swarm priors | `armiger.swarm_priors` | Tracker/source normalization and download-outcome priors |
| Catalog | `armiger.zero_cue_plugin` | Hashed offline title lookup and file-tree disambiguation |
| Audio guard | `armiger.audio_language_guard` | Audio/title/subtitle rules and anime dub interpretation |
| Calibration | `armiger.calibration_and_dht` | Per-head calibration and conformal gates |
| Distributed sync | `armiger.armiger_dht_db_sync` | Optional signed metadata updates and pre-ingest filtering |

The complete pipeline composes these components. Most callers need only `ArmigerUltraPipeline`, not separate component installs. Swarm counters do not change content-category probabilities.

## Compatibility and versions

Library version 3.1.0 and model version 3.0.0 are separate. Existing flat imports such as `armiger_inference` and `armiger_ultra_pipeline` remain compatibility aliases for this release; new code should import from `armiger`. `ArmigerPipeline` aliases `ArmigerUltraPipeline`.

Model weights, vocabulary, calibration, adapter and catalog must be from a compatible bundle. Do not mix assets from different releases. The complete `all` download also includes the PyTorch checkpoint for tooling that needs it.

## Remote Docker server

The Docker examples keep their web and status ports local to the server. To open them from your own computer, open a terminal **on your own computer** and use one of these commands. Replace `user@server` with your server's SSH login.

For Bitmagnet:

```bash
ssh -N -L 3333:127.0.0.1:3333 -L 8096:127.0.0.1:8096 user@server
```

For Rats Search:

```bash
ssh -N -L 8095:127.0.0.1:8095 -L 8098:127.0.0.1:8098 user@server
```

Leave that terminal open and use the `localhost` links in the setup guide. If you changed ports in `.env`, adjust the matching port numbers here. Close the terminal when finished.
