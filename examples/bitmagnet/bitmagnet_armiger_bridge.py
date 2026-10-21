"""Bitmagnet <-> Armiger Sidecar Batch Classifier & GraphQL Tag/Prune Bridge.

Runs Armiger as a local HTTP sidecar alongside a Bitmagnet deployment:
  1. HTTP Batch Classification Endpoint (`POST /v1/bitmagnet/classify`):
     Accepts batches of newly discovered Bitmagnet torrents, runs `ArmigerUltraPipeline.classify_batch`,
     and returns Bitmagnet-compatible content types (`movie`, `tv_show`, `music`, `ebook`, `audiobook`,
     `comic`, `software`, `game`, `xxx`), structured tags (`armiger:<category>`, `armiger:lang:<lang>`,
     `armiger:viability:<tier>`), and `delete` actions for disguised executable lures (`.mkv.exe`,
     `.scr`, `.lnk`) or safety-gate failures.
  2. Optional Bitmagnet GraphQL Helpers (`build_graphql_delete_mutation`, `build_graphql_put_tags_mutation`):
     Builds and optionally dispatches Bitmagnet GraphQL `torrent.delete` and `torrentContent.putTags`
     mutations to complement Bitmagnet's built-in `classifier.yml` workflow and TMDB matching.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence
import urllib.request

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from armiger_dht_rsi import DHTRSIMode, resolve_opt_in_sharing
from armiger_ultra_pipeline import ArmigerUltraPipeline, ArmigerUltraPrediction

# Map Armiger 11-class labels to Bitmagnet ContentType enum values
BITMAGNET_CONTENT_TYPE_MAP: Dict[str, Optional[str]] = {
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
    "unknown": None,
}


class BitmagnetArmigerBridge:
    """Classifies, filters, and tags Bitmagnet torrent items with Armiger."""

    def __init__(
        self,
        pipeline: Optional[ArmigerUltraPipeline] = None,
        *,
        use_zero_cue_catalog: bool = True,
        prune_honeypots_and_malware: bool = True,
        prune_dead_swarms: bool = False,
        isolate_adult_namespace: bool = True,
        opt_in_sharing: Optional[bool] = None,
    ) -> None:
        self.opt_in_sharing: bool = resolve_opt_in_sharing(opt_in_sharing)
        self.pipeline = pipeline or ArmigerUltraPipeline.from_directory(
            ROOT_DIR, variant="int8", preload_catalog=False, opt_in_sharing=self.opt_in_sharing
        )
        if opt_in_sharing is None and pipeline is not None:
            self.opt_in_sharing = bool(getattr(pipeline, "opt_in_sharing", False))
        self.use_zero_cue_catalog = use_zero_cue_catalog
        self.prune_honeypots_and_malware = prune_honeypots_and_malware
        self.prune_dead_swarms = prune_dead_swarms
        self.isolate_adult_namespace = isolate_adult_namespace

    def sharing_status(self) -> Dict[str, Any]:
        st = self.pipeline.sharing_status()
        st["opt_in_sharing_enabled"] = bool(self.opt_in_sharing)
        return st

    @staticmethod
    def _bitmagnet_item_to_row(item: Mapping[str, Any]) -> Dict[str, Any]:
        files_in = item.get("files") if isinstance(item.get("files"), list) else []
        seeders = int(item.get("seeders", item.get("tracker_seeders", 0)) or 0)
        leechers = int(item.get("leechers", item.get("peers", item.get("tracker_leechers", 0))) or 0)
        return {
            "infohash": str(item.get("infoHash") or item.get("info_hash") or item.get("infohash") or ""),
            "name": str(item.get("name") or item.get("title") or ""),
            "size": int(item.get("size") or 0),
            "files": [
                {"path": str(f.get("path") or f.get("name") or ""), "size": int(f.get("size") or 0)}
                if isinstance(f, Mapping)
                else {"path": str(f), "size": 0}
                for f in files_in
            ],
            "file_count": int(item.get("filesCount") or item.get("file_count") or len(files_in) or 1),
            "swarm": {
                "tracker_known": bool(seeders > 0 or leechers > 0),
                "tracker_seeders": seeders,
                "tracker_leechers": leechers,
                "tracker_completed": int(item["completed"]) if "completed" in item and item["completed"] is not None else None,
            },
        }

    def evaluate_batch(self, items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Run Armiger on a batch of Bitmagnet torrents and return tag/prune actions."""
        rows = [self._bitmagnet_item_to_row(it) for it in items]
        predictions: List[ArmigerUltraPrediction] = self.pipeline.classify_batch(
            rows,
            use_zero_cue_catalog=self.use_zero_cue_catalog,
        )

        actions: List[Dict[str, Any]] = []
        delete_infohashes: List[str] = []
        tag_mutations: List[Dict[str, Any]] = []

        for pred in predictions:
            is_malware_or_unsafe = not pred.csam_safety["is_safe"] or pred.junk == "junk"
            is_honeypot = bool(pred.honeypot_trap["is_honeypot_trap"])
            viability_tier = str(pred.swarm_viability["viability_tier"])
            is_dead = viability_tier == "dead"

            if self.prune_honeypots_and_malware and (is_malware_or_unsafe or is_honeypot):
                reason = (
                    pred.csam_safety.get("reason")
                    or ("honeypot_trap" if is_honeypot else "junk_or_malware_gate")
                )
                delete_infohashes.append(pred.infohash)
                actions.append({
                    "infoHash": pred.infohash,
                    "action": "delete",
                    "reason": str(reason),
                    "category": pred.category,
                    "honeypot_trap": is_honeypot,
                    "junk": pred.junk,
                })
                continue

            if self.prune_dead_swarms and is_dead:
                delete_infohashes.append(pred.infohash)
                actions.append({
                    "infoHash": pred.infohash,
                    "action": "delete",
                    "reason": "dead_swarm",
                    "category": pred.category,
                    "viability_tier": viability_tier,
                })
                continue

            if self.isolate_adult_namespace and pred.adult == "adult" and pred.category != "xxx":
                bitmagnet_type: Optional[str] = "xxx"
            else:
                bitmagnet_type = BITMAGNET_CONTENT_TYPE_MAP.get(pred.category)

            tags = [
                f"armiger:{pred.category}",
                f"armiger:lang:{pred.lang}",
                f"armiger:viability:{viability_tier}",
            ]
            if pred.anime == "anime":
                tags.append("armiger:anime")
                dub_status = pred.heads["anime"].get("anime_dub_status")
                if dub_status:
                    tags.append(f"armiger:dub:{dub_status}")
            if pred.english_subs == "has_eng_subs":
                tags.append("armiger:eng_subs")

            if pred.infohash:
                tag_mut = self.build_graphql_put_tags_mutation([pred.infohash], tags)
                if tag_mut is not None:
                    tag_mutations.append(tag_mut)

            actions.append({
                "infoHash": pred.infohash,
                "action": "update",
                "contentType": bitmagnet_type,
                "armigerCategory": pred.category,
                "languages": ["en"] if pred.lang == "english" else [],
                "tags": tags,
                "viabilityTier": viability_tier,
                "viabilityProb": float(pred.swarm_viability["viability_prob"]),
                "mondrianPassed": bool(pred.mondrian_gates["passed_all"]),
            })

        return {
            "total": len(actions),
            "deletedCount": len(delete_infohashes),
            "deleteInfoHashes": delete_infohashes,
            "actions": actions,
            "graphqlDeleteMutation": self.build_graphql_delete_mutation(delete_infohashes),
            "graphqlPutTagsMutations": tag_mutations,
        }

    @staticmethod
    def build_graphql_delete_mutation(infohashes: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Build a Bitmagnet GraphQL `torrent.delete` mutation payload for blocked infohashes."""
        clean = [h.strip().lower() for h in infohashes if h and h.strip()]
        if not clean:
            return None
        return {
            "query": (
                "mutation ArmigerDeleteTorrents($infoHashes: [Hash20!]!) { "
                "torrent { delete(infoHashes: $infoHashes) } "
                "}"
            ),
            "variables": {"infoHashes": clean},
        }

    @staticmethod
    def build_graphql_put_tags_mutation(
        infohashes: Sequence[str],
        tag_names: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        """Build a Bitmagnet GraphQL `torrentContent.putTags` mutation payload."""
        clean_hashes = [h.strip().lower() for h in infohashes if h and h.strip()]
        clean_tags = [t.strip() for t in tag_names if t and t.strip()]
        if not clean_hashes or not clean_tags:
            return None
        return {
            "query": (
                "mutation ArmigerPutTags($input: TorrentContentPutTagsInput!) { "
                "torrentContent { putTags(input: $input) } "
                "}"
            ),
            "variables": {
                "input": {
                    "infoHashes": clean_hashes,
                    "tagNames": clean_tags,
                }
            },
        }

    @staticmethod
    def push_graphql_delete(
        graphql_endpoint: str,
        infohashes: Sequence[str],
        timeout: float = 5.0,
    ) -> Optional[Dict[str, Any]]:
        """Execute the `torrent.delete` GraphQL mutation against a running Bitmagnet endpoint."""
        mutation = BitmagnetArmigerBridge.build_graphql_delete_mutation(infohashes)
        if mutation is None:
            return None
        req = urllib.request.Request(
            graphql_endpoint,
            data=json.dumps(mutation).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


def create_server(host: str = "localhost", port: int = 8096) -> ThreadingHTTPServer:
    model_dir = os.environ.get("ARMIGER_MODEL_DIR", str(ROOT_DIR))
    variant = os.environ.get("ARMIGER_VARIANT", "int8")
    pipeline = ArmigerUltraPipeline.from_directory(
        model_dir,
        variant=variant,
        rsi_mode=DHTRSIMode(os.environ.get("ARMIGER_DHT_RSI_MODE", "pull_only")),
        preload_catalog=os.environ.get("ARMIGER_USE_ZERO_CUE_CATALOG", "true").lower() == "true",
    )
    bridge = BitmagnetArmigerBridge(
        pipeline,
        use_zero_cue_catalog=os.environ.get("ARMIGER_USE_ZERO_CUE_CATALOG", "true").lower() == "true",
        prune_honeypots_and_malware=os.environ.get("ARMIGER_PRUNE_MALWARE", "true").lower() == "true",
        prune_dead_swarms=os.environ.get("ARMIGER_PRUNE_DEAD_SWARMS", "false").lower() == "true",
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
                self._send_json(200, {"status": "ok", "service": "bitmagnet-armiger-bridge", "variant": variant})
            else:
                self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if self.path == "/v1/bitmagnet/classify":
                items = payload.get("torrents") or payload.get("items") or []
                result = bridge.evaluate_batch(items)
                gql_url = os.environ.get("BITMAGNET_GRAPHQL_URL", "").strip()
                if gql_url and result["deleteInfoHashes"]:
                    try:
                        bridge.push_graphql_delete(gql_url, result["deleteInfoHashes"])
                    except Exception as exc:
                        result["graphqlPushError"] = str(exc)
                self._send_json(200, result)
            else:
                self._send_json(404, {"error": "not_found"})

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    listen_host = os.environ.get("ARMIGER_BRIDGE_HOST", "localhost")
    listen_port = int(os.environ.get("ARMIGER_BRIDGE_PORT", "8096"))
    srv = create_server(listen_host, listen_port)
    print(f"Bitmagnet <-> Armiger sidecar bridge listening on {listen_host}:{listen_port}")
    srv.serve_forever()
