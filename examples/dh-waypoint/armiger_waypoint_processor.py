"""DH-Waypoint processorapi/v1 HTTP Service for Armiger & Armiger Ultra.

Implements all three DH-Waypoint external processor contracts:
  - `POST /v1/classify` (`dh-waypoint.processor.v1`)
  - `POST /v1/admit` (`dh-waypoint.processors.admission.v1`)
  - `POST /v1/feedback` (`dh-waypoint.processors.feedback.v1`)
  - `GET /healthz`
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from armiger_dht_rsi import CandidateObservation, DHTRSIMode, resolve_opt_in_sharing
from armiger_ultra_pipeline import ArmigerUltraPipeline, ArmigerUltraPrediction

# Map Armiger 11-class labels to DH-Waypoint canonical content types
WAYPOINT_CONTENT_TYPE_MAP: Dict[str, str] = {
    "movie": "movie",
    "tv_show": "tv_show",
    "music": "music",
    "ebook": "ebook",
    "audiobook": "audiobook",
    "comic": "comic",
    "software": "software",
    "game": "game",
    "course": "ebook",
    "xxx": "xxx",
    "unknown": "unknown",
}


class ArmigerWaypointAdapter:
    """Translates between DH-Waypoint processorapi/v1 envelopes and ArmigerUltraPipeline."""

    def __init__(
        self,
        pipeline: Optional[ArmigerUltraPipeline] = None,
        *,
        use_zero_cue_catalog: bool = True,
        reject_dead_swarms: bool = False,
        opt_in_sharing: Optional[bool] = None,
    ) -> None:
        self.opt_in_sharing: bool = resolve_opt_in_sharing(opt_in_sharing)
        self.pipeline = pipeline or ArmigerUltraPipeline.from_directory(
            ROOT_DIR, variant="int8", preload_catalog=False, opt_in_sharing=self.opt_in_sharing
        )
        if opt_in_sharing is None and pipeline is not None:
            self.opt_in_sharing = bool(getattr(pipeline, "opt_in_sharing", False))
        self.use_zero_cue_catalog = use_zero_cue_catalog
        self.reject_dead_swarms = reject_dead_swarms

    def sharing_status(self) -> Dict[str, Any]:
        st = self.pipeline.sharing_status()
        st["opt_in_sharing_enabled"] = bool(self.opt_in_sharing)
        return st

    @staticmethod
    def _torrent_to_row(t: Mapping[str, Any]) -> Dict[str, Any]:
        swarm_in = t.get("swarm") if isinstance(t.get("swarm"), Mapping) else {}
        seeders = int(swarm_in.get("seeders", t.get("seeders", 0)) or 0)
        leechers = int(swarm_in.get("leechers", t.get("leechers", 0)) or 0)
        files = t.get("files") if isinstance(t.get("files"), list) else []
        return {
            "infohash": str(t.get("info_hash") or t.get("infohash") or ""),
            "name": str(t.get("name") or t.get("title") or ""),
            "size": int(t.get("size") or 0),
            "files": files,
            "file_count": int(t.get("files_count") or len(files) or 1),
            "sources": t.get("sources") if isinstance(t.get("sources"), list) else [],
            "swarm": {
                "tracker_known": bool(seeders > 0 or leechers > 0 or swarm_in),
                "tracker_seeders": seeders,
                "tracker_leechers": leechers,
                "tracker_completed": int(swarm_in["completed"]) if "completed" in swarm_in and swarm_in["completed"] is not None else None,
            },
        }

    def handle_classify_batch(self, request_body: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle `POST /v1/classify` (`dh-waypoint.processor.v1`)."""
        torrents = request_body.get("torrents") or []
        rows = [self._torrent_to_row(t) for t in torrents]
        predictions: List[ArmigerUltraPrediction] = self.pipeline.classify_batch(
            rows,
            use_zero_cue_catalog=self.use_zero_cue_catalog,
        )

        decisions: List[Dict[str, Any]] = []
        for pred in predictions:
            gates_passed = bool(pred.mondrian_gates["passed_all"])
            viability_tier = str(pred.swarm_viability["viability_tier"])
            if not gates_passed:
                reason = (
                    pred.csam_safety.get("reason")
                    or pred.mondrian_gates.get("safety_gate_reason")
                    or ("honeypot_trap_blocked" if pred.honeypot_trap["is_honeypot_trap"] else "safety_gate_rejected")
                )
                decisions.append({
                    "info_hash": pred.infohash,
                    "action": "delete",
                    "reason": str(reason),
                })
                continue

            content_type = WAYPOINT_CONTENT_TYPE_MAP.get(pred.category, "unknown")
            tags = [
                f"armiger:category:{pred.category}",
                f"armiger:lang:{pred.lang}",
                f"armiger:viability:{viability_tier}",
            ]
            if pred.anime == "anime":
                tags.append("armiger:anime")
                dub_status = pred.heads["anime"].get("anime_dub_status")
                if dub_status:
                    tags.append(f"armiger:anime_dub:{dub_status}")
            if pred.english_subs == "has_eng_subs":
                tags.append("armiger:eng_subs")

            decisions.append({
                "info_hash": pred.infohash,
                "action": "enrich",
                "content_type": content_type,
                "languages": ["en"] if pred.lang == "english" else [],
                "tags": tags,
                "attributes": {
                    "armiger.category": pred.category,
                    "armiger.category_confidence": f"{pred.heads['category']['confidence']:.4f}",
                    "armiger.lang": pred.lang,
                    "armiger.viability_tier": viability_tier,
                    "armiger.viability_prob": f"{pred.swarm_viability['viability_prob']:.4f}",
                    "armiger.survival_net_prob": f"{pred.swarm_viability['survival_net_prob']:.4f}",
                    "armiger.catalog_source": str(pred.plugin_trace["plugin_3_zero_cue_catalog"]["resolution_source"]),
                },
            })

        return {
            "schema": "dh-waypoint.processor.v1",
            "decisions": decisions,
        }

    def handle_admission_batch(self, request_body: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle `POST /v1/admit` (`dh-waypoint.processors.admission.v1`)."""
        candidates = request_body.get("candidates") or []
        rows = [self._torrent_to_row(c) for c in candidates]
        predictions = self.pipeline.classify_batch(
            rows,
            use_zero_cue_catalog=self.use_zero_cue_catalog,
        )

        decisions: List[Dict[str, Any]] = []
        for cand, pred in zip(candidates, predictions):
            info_hash = str(cand.get("info_hash") or pred.infohash)
            gates_passed = bool(pred.mondrian_gates["passed_all"])
            viability_tier = str(pred.swarm_viability["viability_tier"])
            admit = gates_passed and (not self.reject_dead_swarms or viability_tier != "dead")
            reason = "passed_mondrian_conformal_gates"
            if not gates_passed:
                reason = str(
                    pred.csam_safety.get("reason")
                    or pred.mondrian_gates.get("safety_gate_reason")
                    or "honeypot_or_safety_gate_blocked"
                )
            elif not admit:
                reason = "dead_swarm_rejected"

            decisions.append({
                "info_hash": info_hash,
                "admit": admit,
                "cohort": pred.category if admit else "rejected",
                "score": float(pred.swarm_viability["viability_prob"] if admit else 0.0),
                "reason": reason,
            })

        return {
            "schema": "dh-waypoint.processors.admission.v1",
            "decisions": decisions,
        }

    def handle_feedback_batch(self, request_body: Mapping[str, Any]) -> Dict[str, Any]:
        """Handle `POST /v1/feedback` (`dh-waypoint.processors.feedback.v1`)."""
        events = request_body.get("events") or []
        accepted = 0
        verified = 0
        for ev in events:
            accepted += 1
            tier = str(ev.get("tier") or "")
            outcome = str(ev.get("outcome") or "")
            if tier in ("t3_files", "t4_playback") and outcome in ("verified", "imported", "completed"):
                cand = CandidateObservation(
                    source_adapter="ffprobe",
                    raw_title=str(ev.get("name") or ev.get("info_hash") or "verified_item"),
                    claimed_category=str(ev.get("content_type") or "movie"),
                    download_completed=True,
                    bytes_downloaded=int(ev.get("size") or 100_000_000),
                    total_bytes=int(ev.get("size") or 100_000_000),
                    catalog_id_type="movie_catalog" if ev.get("content_type") == "movie" else "series_catalog",
                    catalog_id_value=str(ev.get("info_hash") or "verified"),
                    stream_metadata={
                        "video_codec": str((ev.get("attributes") or {}).get("video_codec", "hevc")),
                        "audio_codec": str((ev.get("attributes") or {}).get("audio_codec", "eac3")),
                        "audio_languages": [str((ev.get("attributes") or {}).get("audio_lang", "eng"))],
                    },
                )
                ok, _, _ = self.pipeline.ingest_rsi_feedback(cand)
                if ok:
                    verified += 1

        return {
            "schema": "dh-waypoint.processors.feedback.v1",
            "accepted": accepted,
            "verified": verified,
        }


def create_server(host: str = "localhost", port: int = 8095) -> ThreadingHTTPServer:
    model_dir = os.environ.get("ARMIGER_MODEL_DIR", str(ROOT_DIR))
    variant = os.environ.get("ARMIGER_VARIANT", "fp32")
    rsi_mode_str = os.environ.get("ARMIGER_DHT_RSI_MODE", "pull_only")
    pipeline = ArmigerUltraPipeline.from_directory(
        model_dir,
        variant=variant,
        adult_threshold=float(os.environ.get("ARMIGER_ADULT_THRESHOLD", "0.35")),
        junk_threshold=float(os.environ.get("ARMIGER_JUNK_THRESHOLD", "0.25")),
        scope_threshold=float(os.environ.get("ARMIGER_SCOPE_THRESHOLD", "0.35")),
        rsi_mode=DHTRSIMode(rsi_mode_str),
        preload_catalog=os.environ.get("ARMIGER_USE_ZERO_CUE_CATALOG", "true").lower() == "true",
    )
    adapter = ArmigerWaypointAdapter(
        pipeline,
        use_zero_cue_catalog=os.environ.get("ARMIGER_USE_ZERO_CUE_CATALOG", "true").lower() == "true",
        reject_dead_swarms=os.environ.get("ARMIGER_REJECT_DEAD_SWARMS", "false").lower() == "true",
    )

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok", "engine": "armiger-ultra", "variant": variant})
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if self.path == "/v1/classify":
                self._send_json(200, adapter.handle_classify_batch(payload))
            elif self.path == "/v1/admit":
                self._send_json(200, adapter.handle_admission_batch(payload))
            elif self.path == "/v1/feedback":
                self._send_json(200, adapter.handle_feedback_batch(payload))
            else:
                self._send_json(404, {"error": "not_found"})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    listen_host = os.environ.get("ARMIGER_HOST", "localhost")
    listen_port = int(os.environ.get("ARMIGER_PORT", "8095"))
    srv = create_server(listen_host, listen_port)
    print(f"Armiger DH-Waypoint processor listening on {listen_host}:{listen_port}")
    srv.serve_forever()
