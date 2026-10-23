"""Small local HTTP interface for the standalone Ultra container."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json


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
                self.respond(200, {"status": "ok", "service": "armiger-ultra", "classify": "/v1/classify"})
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

    print("Preparing Armiger Ultra; model files are downloaded once and cached.", flush=True)
    model_dir = download_model(directory, offline=offline)
    pipeline = ArmigerUltraPipeline.from_directory(model_dir, preload_catalog=True, rsi_mode="pull_only", opt_in_sharing=False)
    server = create_server(pipeline, host, port)
    print(f"Armiger Ultra ready on {host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
