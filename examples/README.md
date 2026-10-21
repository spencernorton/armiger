# Armiger & Armiger Ultra Integration Examples

Turnkey reference adapters, HTTP services, and Docker Compose stacks for integrating **Armiger** (`15.7 MB INT8`) and **Armiger Ultra** (`32.5 MB FP32` + 6 plugins) across the self-hosted indexer and media automation ecosystem:

| Stack | Directory | Description |
| :--- | :--- | :--- |
| **[DH-Waypoint](https://github.com/spencernorton/DH-Waypoint)** | [`examples/dh-waypoint/`](dh-waypoint) | Native `processorapi/v1` HTTP processor (`dh-waypoint.processor.v1`), pre-ingest cohort admission gate (`dh-waypoint.processors.admission.v1`), and 4-tier `dh-waypoint.processors.feedback.v1` closed-loop RSI telemetry bridge. |
| **Bitmagnet** | [`examples/bitmagnet/`](bitmagnet) | Fast pre-ingest / post-crawl batch classifier, malware/honeypot pruner, PostgreSQL/GraphQL enrichment bridge (`bitmagnet_armiger_bridge.py`), and Torznab proxy stack. |
| **Torznab Proxy & Webhook Suite** | [`examples/arr-stack/`](arr-stack) | Pre-grab Torznab filtering proxy (`armiger_torznab_proxy.py`) compatible with `Prowlarr`, `Bitmagnet`, `DH-Waypoint`, `Sonarr`, `Radarr`, `Lidarr`, and `Readarr`, plus a post-import webhook & `ffprobe` ground-truth receiver (`armiger_arr_webhook_receiver.py`). |

See [`docs/INTEGRATION_GUIDE.md`](../docs/INTEGRATION_GUIDE.md) for the complete architecture and step-by-step setup guide.
