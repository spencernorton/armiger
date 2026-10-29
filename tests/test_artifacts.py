"""Integrity and offline behavior of downloadable model assets."""
import hashlib
import io

import pytest

from armiger import artifacts


@pytest.fixture
def model(monkeypatch):
    payload = b"verified model bytes"
    monkeypatch.setattr(artifacts, "MANIFEST", {
        "repository": "NorviTech/armiger-ultra", "revision": "a" * 40,
        "files": {"model.onnx": {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}},
    })
    monkeypatch.setitem(artifacts.PRESETS, "ultra", ("model.onnx",))
    return payload


def test_pinned_download_then_offline_reuse(tmp_path, monkeypatch, model):
    urls = []

    def fetch(url, **kwargs):
        urls.append(url)
        return io.BytesIO(model)

    monkeypatch.setattr(artifacts, "urlopen", fetch)
    artifacts.download_model(tmp_path)
    assert urls == [f"https://huggingface.co/NorviTech/armiger-ultra/resolve/{'a' * 40}/model.onnx"]
    monkeypatch.setattr(artifacts, "urlopen", lambda *a, **k: pytest.fail("offline contacted network"))
    artifacts.download_model(tmp_path, offline=True)
    (tmp_path / "model.onnx").write_bytes(b"corrupt")
    with pytest.raises(FileNotFoundError, match="invalid model"):
        artifacts.download_model(tmp_path, offline=True)


def test_corrupt_response_preserves_previous_file(tmp_path, monkeypatch, model):
    previous = tmp_path / "model.onnx"
    previous.write_bytes(b"previous model")
    monkeypatch.setattr(artifacts, "urlopen", lambda *a, **k: io.BytesIO(b"bad response"))
    with pytest.raises(ValueError, match="mismatch"):
        artifacts.download_model(tmp_path)
    assert previous.read_bytes() == b"previous model"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.onnx"]


def test_variant_aliases_fp32_and_int8(tmp_path, monkeypatch):
    payload_int8 = b"int8 bytes"
    payload_fp32 = b"fp32 bytes"
    monkeypatch.setattr(artifacts, "MANIFEST", {
        "repository": "NorviTech/armiger-ultra",
        "revision": "b" * 40,
        "files": {
            "armiger.int8.onnx": {"sha256": hashlib.sha256(payload_int8).hexdigest(), "size": len(payload_int8)},
            "armiger_ultra.fp32.onnx": {"sha256": hashlib.sha256(payload_fp32).hexdigest(), "size": len(payload_fp32)},
        },
    })
    monkeypatch.setitem(artifacts.PRESETS, "cpu", ("armiger.int8.onnx",))
    monkeypatch.setitem(artifacts.PRESETS, "ultra", ("armiger_ultra.fp32.onnx",))

    def fetch(url, **kwargs):
        if url.endswith("armiger.int8.onnx"):
            return io.BytesIO(payload_int8)
        return io.BytesIO(payload_fp32)

    monkeypatch.setattr(artifacts, "urlopen", fetch)
    artifacts.download_model(tmp_path / "i8", variant="int8")
    assert (tmp_path / "i8" / "armiger.int8.onnx").read_bytes() == payload_int8
    artifacts.download_model(tmp_path / "fp", variant="fp32")
    assert (tmp_path / "fp" / "armiger_ultra.fp32.onnx").read_bytes() == payload_fp32
