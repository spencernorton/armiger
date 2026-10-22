# Turnkey Docker Compose Stacks (`Armiger-Connectors`)

Each `docker-compose.yml` in this directory is **100% turnkey** and automatically bootstraps [`NorviTech/armiger-ultra`](https://huggingface.co/NorviTech/armiger-ultra) (`v3.0.0`, 9-head neural core + 6 plugins) and the corresponding sidecar connector from [`spencernorton/Armiger-Connectors`](https://github.com/spencernorton/Armiger-Connectors):

- **[`bitmagnet/`](bitmagnet/docker-compose.yml)** — PostgreSQL 16 + stock `bitmagnet` (`:3333`) + `armiger-ultra` GraphQL poller, HTTP bridge (`:8096`), and Torznab pre-grab XML filter proxy (`:8097`).
- **[`rats-search/`](rats-search/docker-compose.yml)** — Turnkey REST/WebSocket/Manticore `mysql41` sidecar for stock [`librats/rats-search`](https://github.com/librats/rats-search).
- **[`dh-waypoint/`](dh-waypoint/docker-compose.yml)** — PostgreSQL 16 + `dh-waypoint` + `armiger-ultra` `processorapi/v1` service (`:8095`).
- **[`arr-stack/`](arr-stack/docker-compose.yml)** — Standalone Torznab pre-grab filter proxy (`:8097`) + `*Arr`/qBittorrent post-import `ffprobe` webhook receiver (`:8098`).
