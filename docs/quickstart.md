# Set up Armiger Ultra

Ultra runs the complete classification and enrichment pipeline on a CPU. Use Python 3.10–3.12, internet access for the initial installation/download, and roughly 2 GB of free disk for dependencies and cached assets. No GPU, API key or account is required for the public model.

For an indexer deployment, use [Bitmagnet Compose](../examples/bitmagnet/README.md) or [Rats Search Compose](../examples/rats-search/README.md).

## Install

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[ultra]'
python -m armiger download --directory ./models
```

On Windows, activate with `.venv\Scripts\activate`.

The download installs the FP32 ONNX model, swarm adapter, hashed lookup catalog, vocabulary, calibration and model config. Every file is verified against the packaged SHA-256 manifest at a fixed Hugging Face revision. Model code comes from the installed Python package, not the download. The Ultra data is approximately 59 MB.

## Check the installation

From the checkout (or anywhere with the virtual environment activated), run:

```bash
printf '%s\n' '{"name":"Tears.of.Steel.2012.1080p.BluRay.x264-OPEN","size":2400000000,"files":[{"path":"Tears.of.Steel.2012.mkv","size":2400000000}]}' \
  | python -m armiger classify --model-dir ./models
```

The JSON response should include `"category": "movie"`, `heads`, `mondrian_gates`, `plugin_trace` and `swarm_viability`. Missing swarm counters produce an unknown viability estimate; they do not mean the swarm is dead. The [README](../README.md) shows the equivalent Python call.

## Storage and offline use

`--directory` chooses the download location; `--model-dir` chooses the inference location. Set `ARMIGER_MODEL_DIR` to use one location across commands and Python clients. Without either, Armiger uses `~/.cache/armiger/<model-revision>/`.

```bash
export ARMIGER_MODEL_DIR="$PWD/models"
python -m armiger download --offline
```

The offline check verifies the complete cached bundle. Inference never downloads files implicitly. Keep the model directory when updating the library; new library releases may pin a new model revision. Download again when the offline verification reports missing or mismatched files.

## Lightweight ONNX only

This omits the Ultra catalog and swarm adapter and requires no PyTorch:

```bash
python -m pip install .
python -m armiger download --directory ./models-cpu --variant cpu
printf '%s\n' '{"name":"Tears.of.Steel.2012.1080p.WEB-DL","files":[{"path":"Tears.of.Steel.2012.mkv","size":2400000000}]}' \
  | python -m armiger classify --model-dir ./models-cpu --variant cpu
```

## Troubleshooting

- **Missing model:** run the download command for the same directory and variant used for inference.
- **No PyTorch:** install `'.[ultra]'` from the checkout for the full swarm adapter.
- **Checksum/download error:** retry the download; an incomplete download does not replace an existing verified file.
- **Unexpected result:** include synthetic metadata, library/model versions and the output in a [bug report](https://github.com/spencernorton/armiger/issues/new/choose).

DHT sharing is off by default. [Distributed updates](distributed-updates.md) explains explicit opt-in and its limits.
