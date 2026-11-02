"""ARMIGER_UPDATE_CHANNEL resolution and gating.

The update channel must be specified explicitly (env ARMIGER_UPDATE_CHANNEL
or constructor argument): dht = DHT peer-wire first with Hugging Face webseed
fallback, hf = Hugging Face webseed transport, off = no updates. When no
channel is configured, updates stay off and health reports
``update_channel_not_configured`` instead of silently polling.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from armiger.armiger_ultra_pipeline import ArmigerUltraPipeline, _resolve_update_channel


def _make_model_dir(root: Path) -> Path:
    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    cal = {"schema": "armiger.calibration.v2", "heads": {}, "mondrian_thresholds": {}}
    cfg = {"version": "v3.0-test", "dht_seq": 3, "mondrian_thresholds": {}}
    vocab = {"<pad>": 0, "<unk>": 1}
    (model_dir / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (model_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    return model_dir


class ResolveUpdateChannelTests(unittest.TestCase):
    def test_env_values(self) -> None:
        for raw, want in [("dht", "dht"), ("HF", "hf"), (" off ", "off")]:
            with mock.patch.dict(os.environ, {"ARMIGER_UPDATE_CHANNEL": raw}, clear=False):
                os.environ.pop("ARMIGER_DHT_AUTO_UPDATE", None)
                self.assertEqual(_resolve_update_channel(None), want)

    def test_explicit_argument_wins_over_env(self) -> None:
        with mock.patch.dict(os.environ, {"ARMIGER_UPDATE_CHANNEL": "off"}, clear=False):
            self.assertEqual(_resolve_update_channel("dht"), "dht")

    def test_invalid_value_raises(self) -> None:
        with mock.patch.dict(os.environ, {"ARMIGER_UPDATE_CHANNEL": "carrier-pigeon"}, clear=False):
            with self.assertRaises(ValueError):
                _resolve_update_channel(None)

    def test_unset_is_not_configured(self) -> None:
        env = {k: v for k, v in os.environ.items()
               if k not in ("ARMIGER_UPDATE_CHANNEL", "ARMIGER_DHT_AUTO_UPDATE")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(_resolve_update_channel(None))

    def test_legacy_auto_update_maps_to_channel(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "ARMIGER_UPDATE_CHANNEL"}
        with mock.patch.dict(os.environ, {**env, "ARMIGER_DHT_AUTO_UPDATE": "true"}, clear=True):
            self.assertEqual(_resolve_update_channel(None), "dht")
        with mock.patch.dict(os.environ, {**env, "ARMIGER_DHT_AUTO_UPDATE": "false"}, clear=True):
            self.assertEqual(_resolve_update_channel(None), "off")


class PipelineChannelGateTests(unittest.TestCase):
    def _pipeline(self, root: Path, **kwargs: object) -> ArmigerUltraPipeline:
        return ArmigerUltraPipeline(
            model_dir=_make_model_dir(root),
            variant="ultra",
            allow_synthetic_fallback=True,
            runtime_cache_dir=root / "overlay",
            initial_seq=3,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_unconfigured_channel_disables_updates(self) -> None:
        env = {k: v for k, v in os.environ.items()
               if k not in ("ARMIGER_UPDATE_CHANNEL", "ARMIGER_DHT_AUTO_UPDATE",
                            "ARMIGER_DHT_POLL_INTERVAL_SEC")}
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, env, clear=True):
            pipe = self._pipeline(Path(td))
            try:
                self.assertIsNone(pipe.update_channel)
                self.assertFalse(pipe.auto_update_enabled)
                self.assertEqual(pipe.health_status()["update_channel"], "not_configured")
                res = pipe.check_and_apply_dht_release_update(from_poller=True)
                self.assertFalse(res["updated"])
                self.assertEqual(res["reason"], "update_channel_not_configured")
            finally:
                pipe.close()

    def test_off_channel_never_touches_network(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = self._pipeline(Path(td), update_channel="off")
            try:
                self.assertEqual(pipe.update_channel, "off")
                self.assertFalse(pipe.auto_update_enabled)
                res = pipe.check_and_apply_dht_release_update()
                self.assertFalse(res["updated"])
                self.assertEqual(res["reason"], "update_channel_off")
            finally:
                pipe.close()

    def test_dht_channel_enables_poller(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pipe = self._pipeline(Path(td), update_channel="dht", poll_interval_sec=3600.0)
            try:
                self.assertEqual(pipe.update_channel, "dht")
                self.assertTrue(pipe.auto_update_enabled)
                self.assertEqual(pipe.health_status()["update_channel"], "dht")
            finally:
                pipe.close()


if __name__ == "__main__":
    unittest.main()
