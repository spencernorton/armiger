# Bitmagnet + Armiger Ultra

Run [Bitmagnet](https://github.com/bitmagnet-io/bitmagnet) with **Armiger Ultra** automatically classifying newly indexed torrents, tagging them in Bitmagnet's database, and providing a pre-grab filtered Torznab endpoint for Prowlarr and the `*Arr` stack.

The packaged stack runs three services:

- **`postgres` (`postgres:16-alpine`)** — Bitmagnet's PostgreSQL database.
- **`bitmagnet` (`ghcr.io/bitmagnet-io/bitmagnet:v0.10.0`)** — Bitmagnet Web UI, GraphQL API, and raw Torznab feed on `:3333`, plus DHT crawler on `:3334/tcp` and `:3334/udp`.
- **`armiger-ultra` (`ghcr.io/spencernorton/armiger:3.3.1`)** — Background Bitmagnet GraphQL poller, HTTP status/classification API on `:8096`, and filtered Torznab proxy on `:8097` (with the verified Ultra model bundled inside the image).

---

## 1. Set up the stack

Choose **Option A** (quick two-line Compose), **Option B** (Portainer Stacks, `.env` files, or full `docker-compose.yml`), or **Option C** (attach to an existing Bitmagnet installation).

### Option A: Two-line `compose.yaml` (Docker Compose 2.34+)

Create a folder called `armiger-bitmagnet`, save this as **`compose.yaml`** inside it, and start the stack:

```yaml
include:
  - oci://ghcr.io/spencernorton/armiger-bitmagnet-stack:3.3.1
```

```bash
docker compose up -d
```

Docker downloads the ready-made containers, including Armiger's verified model. If Docker asks whether you trust the included OCI setup, answer `y`. To customize environment variables with this two-line file, create a [`compose.override.yaml`](#changing-settings) in the same folder.

### Option B: Full `docker-compose.yml` (Portainer Stacks, `.env` files, or older Compose)

If you use **Portainer Stacks**, want to configure settings through a `.env` file or Portainer's **Environment variables** UI, or run an older Docker Compose version without `include: oci://...`, use the self-contained [`docker-compose.yml`](docker-compose.yml):

1. In Portainer (**Stacks → Add stack**), paste the contents of [`docker-compose.yml`](docker-compose.yml) into the Web editor, or save it locally as `compose.yaml`.
2. Add any optional environment variables (see the [complete environment variable tables](#3-environment-variables-reference) below) in Portainer's **Environment variables** section or in a `.env` file beside `compose.yaml`.
3. Deploy the stack (or run `docker compose up -d`).

> **Note for headless homelab / NAS servers:** By default, [`docker-compose.yml`](docker-compose.yml) binds the web ports to `127.0.0.1` (`127.0.0.1:${BITMAGNET_HTTP_PORT:-3333}:3333`, `127.0.0.1:${ARMIGER_HTTP_PORT:-8096}:8096`, `127.0.0.1:${ARMIGER_PROXY_PORT:-8097}:8097`) so they are not exposed to untrusted networks. If you access Portainer or Docker from another machine on your private LAN and are not using an [SSH tunnel](../../docs/reference.md#remote-docker-server) or reverse proxy, remove the `127.0.0.1:` prefix on the `ports:` lines (for example, `'${BITMAGNET_HTTP_PORT:-3333}:3333'`).

### Option C: Add Armiger Ultra to an existing Bitmagnet installation

If you already run Bitmagnet and want to keep your existing PostgreSQL database and index, add only the `armiger-ultra` service to your existing Bitmagnet Compose stack (or run it on the same Docker network):

```yaml
services:
  armiger-ultra:
    image: ghcr.io/spencernorton/armiger:3.3.1
    restart: unless-stopped
    command: [python, /opt/connectors/bitmagnet/bitmagnet_connector.py]
    ports:
      - '127.0.0.1:8096:8096'
      - '127.0.0.1:8097:8097'
    environment:
      BITMAGNET_GRAPHQL_URL: http://bitmagnet:3333/graphql
      UPSTREAM_TORZNAB_URL: http://bitmagnet:3333/torznab
      BITMAGNET_POLL_SECONDS: "5"
      BITMAGNET_BATCH_SIZE: "100"
      ARMIGER_REQUIRE_ENGLISH_AUDIO: "false"
      ARMIGER_ALLOW_ENGLISH_SUBS: "true"
      ARMIGER_REJECT_DEAD_SWARMS: "false"
      ARMIGER_PRUNE_MALWARE: "false"
      ARMIGER_PRUNE_DEAD_SWARMS: "false"
```

Replace `http://bitmagnet:3333` with your Bitmagnet container name or host address if it differs.

---

## 2. How to use Armiger with Bitmagnet

### Browse and filter by Armiger tags in Bitmagnet (`:3333`)

Open **[http://localhost:3333](http://localhost:3333)**. Every `BITMAGNET_POLL_SECONDS` (default `5` seconds), Armiger queries Bitmagnet's GraphQL API for untagged torrents, fetches their file lists (`torrent.files`), evaluates all 9 neural heads and Ultra plugins, and writes kebab-case tags via `torrent.putTags`:

| Tag family | Tags written in Bitmagnet | Meaning |
| --- | --- | --- |
| **Content category** | `armiger-movie`, `armiger-tv-show`, `armiger-music`, `armiger-game`, `armiger-software`, `armiger-ebook`, `armiger-audiobook`, `armiger-comic`, `armiger-course`, `armiger-xxx`, `armiger-unknown` | Predicted primary media or software type. |
| **Audio language** | `armiger-lang-english`, `armiger-lang-non-english`, `armiger-lang-multi-or-unknown` | Predicted audio track language class. |
| **Release title script** | `armiger-title-english`, `armiger-title-non-english` | Whether the release title is English/Latin vs. localized non-English script. |
| **Swarm viability** | `armiger-viability-alive-viable`, `armiger-viability-unknown-cold`, plus 30-char hash-suffixed tags for `degraded_viable`, `suspect_stalled`, and `dead_known_zero` | Estimated swarm health tier from `SwarmSurvivalNet` and tracker priors (bounded to Bitmagnet's 30-character tag limit). |
| **Anime & dub status** | `armiger-anime`, plus `armiger-dub-dual-audio-or-dub`, `armiger-dub-subbed-only`, or `armiger-dub-raw-or-vernacular` | Added when the item is classified as anime, along with its dub/subtitle classification. |
| **English subtitles** | `armiger-eng-subs` | Added when English subtitles (`has_eng_subs`) are detected in the release metadata or file list. |

In Bitmagnet's web UI or GraphQL API, filter by these tags to isolate specific categories, languages, or healthy swarms.

### Use the filtered Torznab proxy in Prowlarr, Sonarr, Radarr, Lidarr, or Readarr (`:8097`)

Instead of pointing your indexer client directly at Bitmagnet (`:3333/torznab`), point it at **Armiger's Torznab pre-grab proxy**:

- **From the same host:** `http://localhost:8097/torznab`
- **From another container on the same Docker network:** `http://armiger-ultra:8097/torznab`

When Prowlarr or an `*Arr` app searches through `:8097/torznab`, Armiger classifies every returned candidate before the client sees it:

1. **Always blocked automatically:** disguised executable lures (`.mkv.exe`, `.scr`, `.lnk`, `.vbs`, `.bat`, `.ps1`, etc.), safety-gate blocks (`csam_safe=False`), fake-seeder honeypot traps (`honeypot_trap=True`), and `junk` / failed Mondrian conformal gates.
2. **Optional policy filters:** set `ARMIGER_REQUIRE_ENGLISH_AUDIO="true"` (with `ARMIGER_ALLOW_ENGLISH_SUBS="true"` or `"false"`) to filter out foreign-audio releases, or `ARMIGER_REJECT_DEAD_SWARMS="true"` to drop dead swarms from search responses.
3. **Annotated XML & headers:** admitted items receive `<torznab:attr>` annotations (`armiger_category`, `armiger_lang`, `armiger_title_lang`, `armiger_viability`, `armiger_survival_prob`), and HTTP responses include `X-Armiger-Total`, `X-Armiger-Admitted`, and `X-Armiger-Blocked` headers.

### Check health, logs, and on-demand HTTP classification (`:8096`)

- **Health & model status (`GET http://localhost:8096/healthz`):** returns `"status": "ok"`, active model variant (`ultra` or `int8`), catalog/adapter readiness, active DHT sequence (`active_seq`), bundle Merkle root (`active_bundle_merkle_root`), `model_source` (`baked_image` or `dht_live_overlay`), `last_dht_check_ts`, and `sharing_status`.
- **Container logs (`docker compose logs -f armiger-ultra`):** shows startup configuration (`variant`, `interval`, `batch`) and polling errors if Bitmagnet is still starting.
- **On-demand batch classification (`POST http://localhost:8096/v1/bitmagnet/classify`):** accepts `{"torrents": [{"infoHash": "...", "name": "...", "size": 123, "files": [{"path": "...", "size": 123}]}], "applyMutations": false}` and returns normalized predictions plus the prepared GraphQL tag/delete mutations.

---

## 3. Environment variables reference

### Changing settings

- **If using Option A (two-line `compose.yaml`):** save a **`compose.override.yaml`** file in the same folder (or copy [`compose.override.example.yaml`](compose.override.example.yaml)) and run `docker compose config --quiet && docker compose up -d`:
  ```yaml
  services:
    armiger-ultra:
      environment:
        BITMAGNET_POLL_SECONDS: "5"
        BITMAGNET_BATCH_SIZE: "100"
        ARMIGER_REQUIRE_ENGLISH_AUDIO: "false"
        ARMIGER_ALLOW_ENGLISH_SUBS: "true"
  ```
- **If using Option B (full `docker-compose.yml` or Portainer Stacks):** set variables directly in a `.env` file beside `compose.yaml` or in Portainer's **Environment variables** UI, then redeploy. Keep boolean values in YAML quoted as `"true"` or `"false"`.

### Stack, host ports, and Bitmagnet settings

These variables configure host port bindings, PostgreSQL credentials, and Bitmagnet's crawler in [`docker-compose.yml`](docker-compose.yml):

| Variable | Default | What it does |
| --- | --- | --- |
| `BITMAGNET_HTTP_PORT` | `3333` | Host port mapped to Bitmagnet's Web UI, GraphQL API, and raw Torznab endpoint (`127.0.0.1:${BITMAGNET_HTTP_PORT:-3333}:3333`). |
| `BITMAGNET_DHT_PORT` | `3334` | Host TCP and UDP port mapped to Bitmagnet's DHT crawler (`${BITMAGNET_DHT_PORT:-3334}:3334/tcp` and `/udp`). |
| `ARMIGER_HTTP_PORT` | `8096` | Host port mapped to Armiger's HTTP status and classification service (`127.0.0.1:${ARMIGER_HTTP_PORT:-8096}:8096`). |
| `ARMIGER_PROXY_PORT` | `8097` | In `.env` / host port mapping, sets the host port mapped to Armiger's filtered Torznab proxy (`127.0.0.1:${ARMIGER_PROXY_PORT:-8097}:8097`). Inside `armiger-ultra.environment`, sets the container proxy port (`8097`). |
| `POSTGRES_PASSWORD` | `armiger-local` | PostgreSQL password shared by `postgres` and `bitmagnet`. Set before first start; keep unchanged when reusing an existing database volume. |
| `TMDB_API_KEY` | `""` (empty) | Optional TMDB API key passed to `bitmagnet` for Bitmagnet's built-in TMDB matching. Armiger's own offline catalog does not require a TMDB key. |
| `DHT_CRAWLER_SCALING_FACTOR` | `10` | Bitmagnet DHT crawler concurrency multiplier (`bitmagnet` service). Higher values discover more torrents per minute; scale `BITMAGNET_BATCH_SIZE` proportionally when raising this above `30`. |

### Polling and Bitmagnet connection (`armiger-ultra`)

| Variable | Default | What it does |
| --- | --- | --- |
| `ARMIGER_CONNECTOR_MODE` | `both` | `both` runs the background Bitmagnet GraphQL poller and HTTP/Torznab servers. `http` runs only the HTTP/Torznab servers without polling. `poll` runs only the background poller (note: container healthcheck expects `both` or `http`). |
| `BITMAGNET_POLL_SECONDS` | `5` (`5.0`) | Seconds between background GraphQL polling cycles. |
| `BITMAGNET_BATCH_SIZE` | `100` | Maximum untagged Bitmagnet torrents classified per polling cycle. When a 60-second Bitmagnet crawl flush exceeds one batch, consecutive cycles paginate via `offset`/`hasNextPage` past items already tagged earlier in the active flush. |
| `BITMAGNET_GRAPHQL_URL` | `http://bitmagnet:3333/graphql` | Bitmagnet GraphQL endpoint queried for untagged items (`torrentContent.search`), file lists (`torrent.files`), tag writes (`torrent.putTags`), and optional deletions (`torrent.delete`). Defaults to `http://localhost:3333/graphql` when running the script outside Compose. |
| `UPSTREAM_TORZNAB_URL` | `http://bitmagnet:3333/torznab` | Upstream Bitmagnet Torznab endpoint proxied and filtered by Armiger. Defaults to `http://localhost:3333/torznab` outside Compose. |

> **How polling, batch size, and `DHT_CRAWLER_SCALING_FACTOR` interact:** Stock Bitmagnet buffers newly crawled DHT torrents in a batching channel (`persistTorrents`) that flushes to PostgreSQL at **1,000 items or every 60 seconds**, whichever comes first. With `BITMAGNET_POLL_SECONDS=5` and `BITMAGNET_BATCH_SIZE=100`, Armiger paginates across each 60-second flush in 100-item slices—processing up to **1,200 torrents/minute** while keeping individual ONNX inference and `torrent.files` queries small. If you raise `DHT_CRAWLER_SCALING_FACTOR` above `30` or use a longer poll interval (such as `BITMAGNET_POLL_SECONDS=15` or `30`), increase `BITMAGNET_BATCH_SIZE` so `(60 / BITMAGNET_POLL_SECONDS) * BITMAGNET_BATCH_SIZE` exceeds your crawler's per-minute intake.

### Torznab pre-grab filtering (`armiger-ultra`)

These variables control how Armiger filters search results requested through `http://localhost:8097/torznab` (or `:8096/torznab`). They do not delete items from Bitmagnet's database.

| Variable | Default | What it does |
| --- | --- | --- |
| `ARMIGER_REQUIRE_ENGLISH_AUDIO` | `false` | When `true`, blocks Torznab search results classified as `lang == "non_english"`, unless permitted by `ARMIGER_ALLOW_ENGLISH_SUBS` below. Items with `english` or `multi_or_unknown` audio are admitted. |
| `ARMIGER_ALLOW_ENGLISH_SUBS` | `true` | When `ARMIGER_REQUIRE_ENGLISH_AUDIO=true`, still admits `non_english` audio items if English subtitles (`english_subs == "has_eng_subs"`) are detected. Set to `false` for strict English audio only. |
| `ARMIGER_REJECT_DEAD_SWARMS` | `false` (Compose) / `true` (CLI script) | When `true`, excludes results classified as `dead` from Torznab responses without deleting them from Bitmagnet. |
| `ARMIGER_ENABLE_TORZNAB_PROXY` | `true` | Starts the dedicated Torznab proxy listener on `ARMIGER_PROXY_PORT` (`8097`) in `both` or `http` mode. When `false`, `/torznab` and `/api` remain served on `ARMIGER_CONNECTOR_PORT` (`8096`). |

### Index pruning / deleting flagged items from Bitmagnet (`armiger-ultra`)

Both deletion flags default to **`false` in the packaged Compose setup** so no indexed records are deleted unless you opt in:

| Variable | Default | What it does |
| --- | --- | --- |
| `ARMIGER_PRUNE_MALWARE` | `false` (Compose) / `true` (CLI script) | When `true`, the poller deletes torrents flagged as unsafe (`csam_safe=False`), disguised executables (`.mkv.exe`, `.scr`, `.lnk`, etc.), `junk`, or honeypot traps from Bitmagnet's index via `mutation TorrentDelete`. |
| `ARMIGER_PRUNE_DEAD_SWARMS` | `false` | When `true`, the poller deletes torrents classified as `dead` from Bitmagnet's index via `mutation TorrentDelete`. |

### Model, catalog, and container listeners (`armiger-ultra`)

| Variable | Default | What it does |
| --- | --- | --- |
| `ARMIGER_MODEL_DIR` | `/models` (image) / `.` (CLI script) | Directory containing the verified model bundle. The container image pre-bundles the Ultra files in `/models`. |
| `ARMIGER_VARIANT` | `ultra` | Selects `ultra` (31.9 MB FP32 ONNX model + Ultra plugins, bundled in the image) or `int8` / `cpu` / `armiger` (8.2 MB INT8 ONNX model + Ultra plugins; downloads the pinned `all` bundle when `ARMIGER_AUTO_DOWNLOAD=true`). |
| `ARMIGER_AUTO_DOWNLOAD` | `true` | When `true`, downloads and SHA-256-verifies any missing model files against the pinned manifest. When `false`, runs strictly offline and requires all files to exist locally. |
| `ARMIGER_USE_ZERO_CUE_CATALOG` | `true` | When `true`, preloads the hashed offline title catalog (`validated_zero_cue_catalog.json`) at startup. When `false`, defers loading until first lookup. |
| `ARMIGER_CONNECTOR_HOST` | `0.0.0.0` | Bind address inside the container/process for HTTP and Torznab listeners (fallback alias: `ARMIGER_BRIDGE_HOST`). |
| `ARMIGER_CONNECTOR_PORT` | `8096` | Container/process port for `/healthz`, `/v1/bitmagnet/classify`, and `/torznab` (fallback alias: `ARMIGER_BRIDGE_PORT`). |
| `ARMIGER_HF_REPO` | `NorviTech/armiger-ultra` | Hugging Face repository ID used only by the legacy standalone connector loader when the `armiger` Python package is not installed. |

### Distributed model updates (BEP 44/46) and opt-in sharing (`armiger-ultra`)

Armiger Ultra can check for signed model/catalog updates over Mainline DHT (BEP 44/46) and hot-swap them into a writable overlay after passing a 500-item canary regression check. Outbound feedback contribution is **strictly off by default** (`ARMIGER_DHT_SHARE_OPT_IN=false`), and ordinary classification never transmits torrent names or unverified predictions. See [distributed updates](../../docs/distributed-updates.md) for architecture details.

| Variable | Default | What it does |
| --- | --- | --- |
| `ARMIGER_DHT_AUTO_UPDATE` | `true` | Enables background BEP 44/46 polling for newer signed model/catalog releases (`seq > active_seq`), verified piece download, 500-item canary validation, and atomic hot-swap. Set to `false` (or `0`/`off`) to disable background update checks. |
| `ARMIGER_DHT_POLL_INTERVAL_SEC` | `14400` (4 hours) | Interval in seconds between background BEP 44/46 DHT release checks (minimum `0.2`). |
| `ARMIGER_RUNTIME_CACHE_DIR` | `/tmp/armiger-live-models-<uid>` | Private (`0700`) writable directory where verified live-update overlays are staged, leaving `/models` read-only. Mount a volume here to persist hot-swapped updates across container recreation. |
| `ARMIGER_DHT_RSI_MODE` | `pull_only` | Distributed update policy: `pull_only` (receive verified updates only), `off`, or `anonymous_contribute` (when `ARMIGER_DHT_SHARE_OPT_IN=true`). |
| `ARMIGER_DHT_SHARE_OPT_IN` | `false` (`0`) | Strict opt-in switch (`true`/`1`/`yes` to enable; `false`/`0` by default). When `false`, all outbound contribution packets are blocked. |
| `ARMIGER_RELEASE_PUBKEY_HEX` | `3599362bd52dad8b89398b904d3081c2cb483d13376f7ffb49f3748208731e76` | 64-hex-character Ed25519 public key verified on BEP 44/46 mutable release pointers. |
| `ARMIGER_RELEASE_SALT` | `armiger:weights:ultra` (or `armiger:weights:int8`) | BEP 44 salt used with `ARMIGER_RELEASE_PUBKEY_HEX` to derive the mutable DHT lookup target `SHA-1(pubkey \|\| salt)`. |
| `ARMIGER_LOCAL_SEQ` | Auto-detected (`3` or `4`) | Optional integer override for the initial local release sequence number (`active_seq`) before polling for newer DHT releases. |
| `ARMIGER_DHT_BOOTSTRAP_NODES` | `router.bittorrent.com:6881,dht.transmissionbt.com:6881,router.utorrent.com:6881` | Comma-separated `host:port` list of Mainline DHT bootstrap nodes queried during BEP 44 release lookup. |
| `ARMIGER_DHT_PEERS` | `""` (empty) | Optional comma-separated `host:port` list of explicit BitTorrent peer-wire seeders (`:6882`) tried alongside peers announced in the BEP 44 pointer. |
| `ARMIGER_WEBSEED_URLS` | `https://huggingface.co/NorviTech/armiger-ultra/resolve/main` | Comma-separated BEP 19 HTTPS webseed base URLs used as fallback if direct TCP peer-wire piece transfer is unreachable; pieces are still verified against the signed infohash (`ih`) and Merkle root (`mr`). |
| `ARMIGER_DHT_COLLECTOR_NODES` | `""` (empty) | Optional comma-separated `host:port` list of collector nodes for direct UDP contribution delivery when `ARMIGER_DHT_SHARE_OPT_IN=true`. |
| `ARMIGER_CATALOG_HMAC_KEY` | Random 32-byte key | Optional secret key (`>= 16` chars or `hex:<64-hex-chars>`) used by `AnonymityGuard` to key `armiger-catalog-v2` standalone catalog hashes across restarts. |
| `ARMIGER_PUBLIC_IPV4` | `""` (empty) | Optional public IPv4 address used by `python -m armiger.verify_dht` when generating a BEP 42 CRC32C-compliant DHT node ID behind NAT. |

---

## Stop, start, and update

From your stack folder, run `docker compose down` to stop or `docker compose up -d` to start again. Your PostgreSQL index stays saved in `postgres_data` (do not pass `-v` unless you want to erase the database).

[Detailed Bitmagnet configuration guide](../../docs/bitmagnet-configuration.md) · [Docker settings and updates](../../docs/reference.md#docker-settings-and-updates) · [Using a remote server](../../docs/reference.md#remote-docker-server)
