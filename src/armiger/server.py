"""Small local HTTP interface for the standalone Ultra container."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os


def create_server(pipeline, host="127.0.0.1", port=8096):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/healthz"):
                health_extra = pipeline.health_status() if hasattr(pipeline, "health_status") else {
                    "active_seq": int(getattr(pipeline, "active_seq", 0)),
                    "active_bundle_merkle_root": getattr(pipeline, "active_bundle_merkle_root", None),
                    "model_source": str(getattr(pipeline, "model_source", "baked_image")),
                    "last_dht_check_ts": getattr(pipeline, "last_dht_check_ts", None),
                    "sharing_status": pipeline.sharing_status() if hasattr(pipeline, "sharing_status") else None,
                }
                self.respond(200, {
                    "status": "ok",
                    "service": "armiger-ultra",
                    "classify": "/v1/classify",
                    **health_extra,
                })
            else:
                self.respond(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/v1/classify":
                self.respond(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    self.respond(413, {"error": "request_must_be_between_1_byte_and_1_MiB"})
                    return
                self.connection.settimeout(15)
                item = json.loads(self.rfile.read(length))
                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                    raise ValueError("Supply a JSON object with a nonempty name")
                result = pipeline.classify(item)
                self.respond(200, result.to_dict() if hasattr(result, "to_dict") else result)
            except (ValueError, TypeError, KeyError, TimeoutError) as error:
                self.respond(400, {"error": "invalid_metadata", "detail": str(error)})
            except Exception:
                self.respond(500, {"error": "classification_failed"})

        def log_message(self, fmt, *args):
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(directory=None, *, host="127.0.0.1", port=8096, offline=False):
    from . import ArmigerUltraPipeline
    from .artifacts import download_model

    print("Preparing Armiger Ultra and verifying model files.", flush=True)
    model_dir = download_model(directory, offline=offline)
    pipeline = ArmigerUltraPipeline.from_directory(
        model_dir,
        preload_catalog=True,
        rsi_mode=os.environ.get("ARMIGER_DHT_RSI_MODE") or None,
        opt_in_sharing=None,
        offline=offline,
    )
    server = create_server(pipeline, host, port)
    print(f"Armiger Ultra ready on {host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        if hasattr(pipeline, "stop_auto_update_poller"):
            pipeline.stop_auto_update_poller()
        server.server_close()
