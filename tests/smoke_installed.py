"""Run with an installed wheel outside the checkout, using real pinned models."""
import importlib
import json
from pathlib import Path

import armiger
from armiger import ArmigerStandaloneONNX, ArmigerUltraPipeline
from armiger.artifacts import download_model

assert "site-packages" in str(Path(armiger.__file__).resolve()), armiger.__file__
for name in (
    "armiger_dht_rsi", "armiger_dht_db_sync", "byte_span_extractor", "zero_cue_binary",
    "pq_xmss_merkle", "ultra_pipeline_update_bridge", "dht_head_delta", "oci_dht_node", "verify_dht",
):
    importlib.import_module(f"armiger.{name}")
for name in ("armiger_inference", "armiger_ultra_pipeline", "armiger_ultra"):
    importlib.import_module(name)

directory = download_model(variant="all", offline=True)
item = {"name": "Tears.of.Steel.2012.1080p.BluRay.x264-OPEN", "size": 2400000000,
        "files": [{"path": "Tears.of.Steel.2012.mkv", "size": 2400000000}]}
standalone = ArmigerStandaloneONNX(directory / "armiger.int8.onnx", directory / "vocab.json", directory / "calibration.json")
assert standalone.predict(item)["category"] == "movie"
pipeline = ArmigerUltraPipeline.from_directory(directory, preload_catalog=True, opt_in_sharing=False)
result = pipeline.classify(item)
assert result["category"] == "movie"
assert not pipeline.opt_in_sharing
print(json.dumps({"library": armiger.__version__, "category": result["category"], "installed": armiger.__file__}))
