"""Download data files from an immutable model revision, verifying SHA-256."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

MANIFEST = json.loads(Path(__file__).with_name("model_manifest.json").read_text())
_COMMON = ("config.json", "vocab.json", "calibration.json")
PRESETS = {
    "cpu": (*_COMMON, "armiger.int8.onnx"),
    "ultra": (*_COMMON, "armiger_ultra.fp32.onnx", "swarm_adapter.pt", "validated_zero_cue_catalog.json"),
    "all": tuple(MANIFEST["files"]),
}


def model_directory(directory: str | Path | None = None) -> Path:
    """Resolve an explicit path, ARMIGER_MODEL_DIR, or the versioned user cache."""
    return Path(directory or os.environ.get("ARMIGER_MODEL_DIR") or
                Path.home() / ".cache" / "armiger" / MANIFEST["revision"]).expanduser().resolve()


def _matches(path: Path, record: dict) -> bool:
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == record["sha256"]


def download_model(directory: str | Path | None = None, *, variant: str = "ultra", offline: bool = False) -> Path:
    """Fetch and atomically install verified data. Never download or execute Python code."""
    names = PRESETS[variant]
    target = model_directory(directory)
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        record = MANIFEST["files"][name]
        path = target / name
        if _matches(path, record):
            continue
        if offline:
            raise FileNotFoundError(f"Missing or invalid model file: {path}. Run 'python -m armiger download --directory {target}'.")
        url = f"https://huggingface.co/{MANIFEST['repository']}/resolve/{MANIFEST['revision']}/{name}"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target, prefix=f".{name}.", delete=False) as output:
                temporary = Path(output.name)
                with urlopen(url, timeout=60) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            if not _matches(temporary, record):
                raise ValueError(f"Checksum or size mismatch for {name}; existing file was preserved")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return target
