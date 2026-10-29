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
        ("3125e70b22303f1a548e7b890d2b1c078858eed1b987cc9d63c52188fbc630c0", 33260312),
        ("ebed96dfef6f78842768ecdad646f6e5d838681feed9a4919d4b95aa0f912ea4", 33258907),
        ("6f94aef676e5bec4968378437769cfdc768fefcc6a21854b8319b33456ebd2ec", 33254585),
        ("ed27945bb6cb024d4486392c2c714df4558c1fc581f86568ab53ace7d40f19f1", 32626152),
        ("9d8a556ddba4f66052db27bc19cf2a303067a5c537e6a8d084145f665301f536", 33254585),
        ("51de6ff771bd10d4db75cf38d8d0fd9aa8a4bd52208fb46308335b7fb7f458fa", 25390473),
        ("d092b49428dbca550d4099b168e231c35b0cc60c7632a60e55d0be63646317dc", 25387489),
    },
    "armiger.int8.onnx": {
        ("7ee3dfdaae4283faecabf06bc7d7644d1747d959264506b07065a3035fe6ffd0", 17606389),
        ("02d185e673e8552a6b09d921b1bf8bc6e64cdb4ba984ffacdbe804b487132271", 17604743),
        ("c0e96936038988c7465b7fe041c4dc3c92df5a8cb4f56c2666271cd4d3e79420", 17599636),
        ("93ed3d6f0ce8af8f0b788fa75575a07deaeb3cfe23ca2df4275b74eca2237c9f", 16967970),
        ("116e3a291a036c667794a028cf5b06441371a76d03edba04da741f11d87882f9", 17599636),
        ("8a8be5343f8e2fbdf62f8c7a7c2ca40b61fd239d111a18c448f31d0c6d39ca45", 6667891),
        ("aa509ff08a38d9f8187250860516cc0f218c604bbac8d3726c215bb558bc5588", 6664718),
    },
    "swarm_adapter.pt": {
        ("4e4c7a2dd59fb91a234e6627e7ade35062f99268d3f05370432bb9f2a2f9b1c9", 144144),
        ("4d615c0e8efaa75be252f83910099c454a188b76e11653c8da20a4b95c48c996", 316931),
        ("74aa36c7701092d198a8238462c0b7a44a55f238ba26f6c506b984a83b1c0ca1", 316931),
    },
    "calibration.json": {
        ("ac2fe3b3de8cda9d3e7bca3b334157ac0d90e052e2ff03c3787900b991b13b34", 76612),
        ("ea7a4df70747160b41ae0cbca0292562f097bbbea7b5d549c5348f743e30b33c", 76612),
        ("300387779dc16292dd9e6be038ce9dd60e5bdb0ee929b383dcd4e51715f8b5f8", 76612),
        ("eb201c12539367aa7f3af8ab55b71113a680ad3f4c57a41a589621dccc957de9", 76610),
        ("3eb5d460bf8408fd0bc104f1cf84ddcd1e71cb8f3d2cbbfab69f38be3eb6776c", 74060),
    },
    "config.json": {
        ("6263a1f6bc82bbcafdfd2d98d327d53b07b8f2d6385cbb5a0cb8c1c9f42dff46", 5201),
        ("caf3c30c7f6380b84af4f61ebc0afa80707b9137ec3be4e2d03e09c3e05a0ad4", 5201),
        ("da965d165bc45b5dafb2ae2fcb45781b1c37cebd9fc3026604c12d060d6e523f", 5201),
        ("047286ecbccd7360a597cf9eff21a688f9f842a427c1e1f877b367d3ce42e7eb", 5629),
        ("bacf41ee0a941200229108785bd9207deca265e83d1daf95b4fb49aef0071db1", 5562),
        ("f2dfa7c8901585c6718391c60efa8c327f00168ded250a581565675fed42a985", 5561),
        ("210760e3802332bd8c3987a2c1e82790c40ecad566034321b501f04e069fee94", 8503),
        ("3de681533f672940f4c55b0b666111bf4a653db750c1d184188022905dc771d8", 5228),
    },
    "vocab.json": {
        ("c10afea430188e0207752e93bf0b340fbb585858684037c5134a477cbc0dd7bb", 39996),
        ("4368a1ebfe2fb3bf5fd7258d7fb8e7bf21134ff698ee90f6dc66c85f891e1a05", 3741),
    },
    "validated_zero_cue_catalog.json": {
        ("8a8b7a4ba97cb22c3aa1f75ab33762052deb09580df464b1eabfe622a2837d26", 26039500),
        ("930942f45bbcf4a04c696abddbed2348851bea4c8435d9e48de3209b8a0bc81f", 21298000),
        ("3f42219f7e9ac1b38e7aa0f1301ae96298dd58e7d71960c371a4cdaf86c3d0b2", 25990894),
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
