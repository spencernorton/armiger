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


# Allow both the baked HF revision digests and DHT-promoted v3.0-gold-r13.3 digests
KNOWN_RELEASE_DIGESTS = {
    "armiger_ultra.fp32.onnx": {
        ("51de6ff771bd10d4db75cf38d8d0fd9aa8a4bd52208fb46308335b7fb7f458fa", 25390473),
        ("d092b49428dbca550d4099b168e231c35b0cc60c7632a60e55d0be63646317dc", 25387489),
    },
    "armiger.int8.onnx": {
        ("8a8be5343f8e2fbdf62f8c7a7c2ca40b61fd239d111a18c448f31d0c6d39ca45", 6667891),
        ("aa509ff08a38d9f8187250860516cc0f218c604bbac8d3726c215bb558bc5588", 6664718),
    },
    "swarm_adapter.pt": {
        ("4d615c0e8efaa75be252f83910099c454a188b76e11653c8da20a4b95c48c996", 316931),
        ("74aa36c7701092d198a8238462c0b7a44a55f238ba26f6c506b984a83b1c0ca1", 316931),
    },
    "calibration.json": {
        ("300387779dc16292dd9e6be038ce9dd60e5bdb0ee929b383dcd4e51715f8b5f8", 76612),
        ("3eb5d460bf8408fd0bc104f1cf84ddcd1e71cb8f3d2cbbfab69f38be3eb6776c", 74060),
    },
    "config.json": {
        ("f2dfa7c8901585c6718391c60efa8c327f00168ded250a581565675fed42a985", 5561),
        ("3de681533f672940f4c55b0b666111bf4a653db750c1d184188022905dc771d8", 5228),
    },
    "vocab.json": {
        ("4368a1ebfe2fb3bf5fd7258d7fb8e7bf21134ff698ee90f6dc66c85f891e1a05", 3741),
    },
    "validated_zero_cue_catalog.json": {
        ("77a018fb4bd85c1b8dd4794bb21a4ab1a4c42cb75890f023c627e93fd0b45dc6", 44249032),
    },
}


def _matches(path: Path, record: dict) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    allowed = set(KNOWN_RELEASE_DIGESTS.get(path.name, set()))
    allowed.add((record["sha256"], record["size"]))
    if size not in {sz for _, sz in allowed}:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return (digest.hexdigest(), size) in allowed


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
