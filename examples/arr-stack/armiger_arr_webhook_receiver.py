"""Armiger Post-Import Webhook & ffprobe Ground-Truth Receiver for Sonarr, Radarr, Lidarr, Readarr & qBittorrent.

Wires completion and import events from the *Arr media automation stack and qBittorrent directly into
Armiger's `SeriesWebhookAdapter`, `MovieWebhookAdapter`, `AudioWebhookAdapter`, `PublicationWebhookAdapter`,
`QBittorrentAdapter`, and `PhysicalGroundTruthVerifier`:
  - `POST /webhooks/sonarr`       -> `SeriesWebhookAdapter` (`adapter="series_webhook"`)
  - `POST /webhooks/radarr`       -> `MovieWebhookAdapter` (`adapter="movie_webhook"`)
  - `POST /webhooks/lidarr`       -> `AudioWebhookAdapter` (`adapter="audio_webhook"`)
  - `POST /webhooks/readarr`      -> `PublicationWebhookAdapter` (`adapter="publication_webhook"`)
  - `POST /webhooks/qbittorrent`  -> `QBittorrentAdapter` (`adapter="qbittorrent"`)

When `ARMIGER_RUN_FFPROBE=true` and the imported media path is accessible on disk, the receiver runs
`ffprobe -v quiet -print_format json -show_format -show_streams <path>` to physically verify video/audio
codecs, channel counts, audio languages, and subtitle tracks before passing the observation to
`ArmigerUltraPipeline.ingest_rsi_feedback(...)`.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, Mapping, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from armiger_dht_rsi import DHTRSIMode
from armiger_ultra_pipeline import ArmigerUltraPipeline

ROUTE_TO_ADAPTER: Dict[str, str] = {
    "/webhooks/sonarr": "series_webhook",
    "/webhooks/radarr": "movie_webhook",
    "/webhooks/lidarr": "audio_webhook",
    "/webhooks/readarr": "publication_webhook",
    "/webhooks/qbittorrent": "qbittorrent",
}


def run_ffprobe_json(media_path: str, timeout_sec: float = 10.0) -> Optional[Dict[str, Any]]:
    """Run ffprobe on a local media file and return parsed JSON stream/format metadata."""
    p = Path(media_path)
    if not p.is_file():
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(p),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
        return json.loads(proc.stdout)
    except Exception:
        return None


class ArmigerArrWebhookService:
    """Processes *Arr and qBittorrent completion webhooks into Armiger RSI ground-truth observations."""

    def __init__(
        self,
        pipeline: ArmigerUltraPipeline,
        *,
        enable_ffprobe: bool = True,
        ffprobe_runner: Callable[[str], Optional[Dict[str, Any]]] = run_ffprobe_json,
    ) -> None:
        self.pipeline = pipeline
        self.enable_ffprobe = enable_ffprobe
        self.ffprobe_runner = ffprobe_runner

    @staticmethod
    def _extract_media_path(payload: Mapping[str, Any]) -> Optional[str]:
        for section_key in ("movieFile", "episodeFile"):
            sec = payload.get(section_key)
            if isinstance(sec, Mapping):
                full_path = sec.get("path") or sec.get("relativePath")
                if full_path:
                    return str(full_path)
        for list_key in ("trackFiles", "bookFiles"):
            items = payload.get(list_key)
            if isinstance(items, list) and items and isinstance(items[0], Mapping):
                full_path = items[0].get("path")
                if full_path:
                    return str(full_path)
        if payload.get("content_path") or payload.get("save_path"):
            return str(payload.get("content_path") or payload.get("save_path"))
        return None

    def process_webhook(
        self,
        route_path: str,
        payload: Mapping[str, Any],
        *,
        explicit_ffprobe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a webhook payload from Sonarr, Radarr, Lidarr, Readarr, or qBittorrent."""
        adapter_name = ROUTE_TO_ADAPTER.get(route_path)
        if not adapter_name:
            return {"accepted": False, "reason": f"unknown_route:{route_path}"}

        event_type = str(payload.get("eventType") or "")
        if event_type.lower() == "test":
            return {
                "accepted": True,
                "verified": False,
                "adapter": adapter_name,
                "reason": "test_webhook_acknowledged",
            }

        ffprobe_data = explicit_ffprobe
        if ffprobe_data is None and self.enable_ffprobe:
            media_path = self._extract_media_path(payload)
            if media_path:
                ffprobe_data = self.ffprobe_runner(media_path)

        ok, reason, record = self.pipeline.ingest_rsi_feedback(
            payload,
            adapter=adapter_name,
            ffprobe_output=ffprobe_data,
        )
        return {
            "accepted": True,
            "verified": bool(ok),
            "adapter": adapter_name,
            "reason": reason,
            "blinded_hash": record.blinded_stem.blinded_hash if record is not None else None,
            "category": record.category if record is not None else None,
            "lang": record.lang if record is not None else None,
            "junk": record.junk if record is not None else None,
        }


def create_webhook_server(host: str = "localhost", port: int = 8098) -> ThreadingHTTPServer:
    model_dir = os.environ.get("ARMIGER_MODEL_DIR", str(ROOT_DIR))
    variant = os.environ.get("ARMIGER_VARIANT", "int8")
    rsi_mode_str = os.environ.get("ARMIGER_DHT_RSI_MODE", "pull_only")
    pipeline = ArmigerUltraPipeline.from_directory(
        model_dir,
        variant=variant,
        rsi_mode=DHTRSIMode(rsi_mode_str),
        preload_catalog=False,
    )
    service = ArmigerArrWebhookService(
        pipeline,
        enable_ffprobe=os.environ.get("ARMIGER_RUN_FFPROBE", "true").lower() == "true",
    )

    class WebhookHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok", "service": "armiger-arr-webhook-receiver", "rsi_mode": rsi_mode_str})
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if self.path in ROUTE_TO_ADAPTER:
                res = service.process_webhook(self.path, payload)
                self._send_json(200, res)
            else:
                self._send_json(404, {"error": "not_found"})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), WebhookHandler)


if __name__ == "__main__":
    listen_host = os.environ.get("ARMIGER_WEBHOOK_HOST", "localhost")
    listen_port = int(os.environ.get("ARMIGER_WEBHOOK_PORT", "8098"))
    srv = create_webhook_server(listen_host, listen_port)
    print(f"Armiger *Arr & qBittorrent webhook receiver listening on {listen_host}:{listen_port}")
    srv.serve_forever()
