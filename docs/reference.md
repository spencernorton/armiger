# Python reference

## Python installation

For Python callers, install Python 3.12 and Git, then clone the repository and create a virtual environment:

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows, use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1` instead of the last two commands. On Linux, install CPU-only PyTorch first: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.

```bash
python -m pip install '.[ultra]'
python -m armiger download
python examples/classify.py
```

The example prints `Content type: movie`. On subsequent runs, activate the virtual environment and run your script; no reinstall is needed. `python -m armiger serve` runs the same local HTTP interface as the container. Add `--offline` to require cached models without downloading.

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
| DHT verifier CLI | `armiger.verify_dht` | Standalone BEP 44/46 + BEP 9/10/19 peer-wire and webseed bundle verification (`python -m armiger.verify_dht`) |

The complete pipeline composes these components. Most callers need only `ArmigerUltraPipeline`, not separate component installs. Swarm counters do not change content-category probabilities.

## Compatibility and versions

Library version 3.3.4 uses model version 3.3.0. Existing flat imports such as `armiger_inference` and `armiger_ultra_pipeline` remain compatibility aliases for this release; new code should import from `armiger`. `ArmigerPipeline` aliases `ArmigerUltraPipeline`.

Model weights, vocabulary, calibration, adapter and catalog must be from a compatible bundle. Do not mix assets from different releases. The complete `all` download also includes the PyTorch checkpoint for tooling that needs it.

## Docker settings and updates

The two-line integration setups use packaged defaults and require Docker Compose 2.34 or newer. For Bitmagnet, keep the two-line file and use a local `compose.override.yaml` to configure Armiger. See the [complete Armiger Ultra environment-variable reference](bitmagnet-configuration.md) for defaults, examples and the effects of each setting.

For Portainer Stacks, older Compose versions, or direct single-file customization, save the full [Bitmagnet file](https://raw.githubusercontent.com/spencernorton/armiger/main/examples/bitmagnet/docker-compose.yml) or [Rats Search file](https://raw.githubusercontent.com/spencernorton/armiger/main/examples/rats-search/docker-compose.yml) as `docker-compose.yml` **in place of** the two-line file. The Bitmagnet file has all default environment values and port mappings written directly in the file (no `.env` file or `${VAR}` placeholders required) so you can copy-paste it into a folder or Portainer Stack, edit any setting in place, and run `docker compose up -d`.

| Setup | Pre-populated settings in `docker-compose.yml` (defaults) |
| --- | --- |
| Bitmagnet | Ports `127.0.0.1:3333:3333`, `3334:3334`, `127.0.0.1:8096:8096`, `127.0.0.1:8097:8097`; `POSTGRES_PASSWORD=armiger-local`, `TMDB_API_KEY=''`, `DHT_CRAWLER_SCALING_FACTOR='10'`, `BITMAGNET_POLL_SECONDS='5'`, `BITMAGNET_BATCH_SIZE='100'`, `ARMIGER_REQUIRE_ENGLISH_AUDIO='false'`, `ARMIGER_ALLOW_ENGLISH_SUBS='true'`, `ARMIGER_REJECT_DEAD_SWARMS='false'`, `ARMIGER_PRUNE_MALWARE='false'`, `ARMIGER_PRUNE_DEAD_SWARMS='false'`, `ARMIGER_DHT_AUTO_UPDATE='true'`, `ARMIGER_DHT_POLL_INTERVAL_SEC='14400'`, `ARMIGER_DHT_SHARE_OPT_IN='false'`, `ARMIGER_DHT_RSI_MODE=pull_only` (see [all Bitmagnet variables, required vs. optional status, and valid ranges](bitmagnet-configuration.md)) |
| Rats Search | `RATS_HTTP_PORT=8095`, `ARMIGER_HTTP_PORT=8098` |

Bitmagnet's database has no published port and uses an internal password. To choose your own, set `POSTGRES_PASSWORD` **before the first start**; keep it unchanged while reusing that database. TMDB enrichment is optional. Automatic deletion of flagged items is disabled.

Images are pinned to a release, so restarting does not silently upgrade software. To update, back up your data and change the version on the `include` line to the new release. If using the full file, download its new version and preserve local edits and `.env`. Then run:

```bash
docker compose pull
docker compose up -d
```

Keep the same folder name so Compose reuses your existing volumes. Armiger supports Linux x86-64 and ARM64. The Rats Search backend currently supports x86-64 (with emulation on ARM).

The companion Rats Search image builds upstream [v2.3.5](https://github.com/librats/rats-search/tree/8415a208025d3827f2f380f3bfe530557a10eaba), including its pinned submodules. Rats Search's MIT license and Manticore's GPL license, component licenses and corresponding source archive are included under `/usr/share/doc/`. Manticore's source is also [available directly](https://github.com/manticoresoftware/manticoresearch/tree/2c59803fefb5512dd868fbf42aa759d78e617e74).

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
