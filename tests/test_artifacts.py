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
