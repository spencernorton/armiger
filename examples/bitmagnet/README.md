# Bitmagnet + Armiger Ultra

Run [Bitmagnet](https://github.com/bitmagnet-io/bitmagnet) with **Armiger Ultra** automatically classifying newly indexed torrents, tagging them in Bitmagnet's database, and providing a pre-grab filtered Torznab endpoint for Prowlarr and the `*Arr` stack.

The packaged stack runs three services:

- **`postgres` (`postgres:16-alpine`)** — Bitmagnet's PostgreSQL database.
- **`bitmagnet` (`ghcr.io/bitmagnet-io/bitmagnet:v0.10.0`)** — Bitmagnet Web UI, GraphQL API, and raw Torznab feed on `:3333`, plus DHT crawler on `:3334/tcp` and `:3334/udp`.
- **`armiger-ultra` (`ghcr.io/spencernorton/armiger:3.4.1`)** — Background Bitmagnet GraphQL poller, HTTP status/classification API on `:8096`, and filtered Torznab proxy on `:8097` (with the verified Ultra model bundled inside the image).

---

## 1. Set up the stack

Choose **Option A** (copy-paste `docker-compose.yml` with literal defaults), **Option B** (two-line OCI Compose include), or **Option C** (attach to an existing Bitmagnet installation).

### Option A: Copy-paste `docker-compose.yml` (Standalone folder or Portainer Stacks)

All environment variables in [`docker-compose.yml`](docker-compose.yml) are **pre-filled with working literal defaults** (no `.env` file or `${VAR}` placeholders required).

1. Create a directory for your stack (or open **Portainer → Stacks → Add stack**), save the following as **`docker-compose.yml`**, edit any setting directly in the file if desired, and run `docker compose up -d`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: bitmagnet
      POSTGRES_USER: bitmagnet
      POSTGRES_PASSWORD: armiger-local
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: [CMD-SHELL, 'pg_isready -U bitmagnet -d bitmagnet']
      interval: 5s
      timeout: 5s
      retries: 20

  bitmagnet:
    image: ghcr.io/bitmagnet-io/bitmagnet:v0.10.0
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - '127.0.0.1:3333:3333'
      - '3334:3334/tcp'
      - '3334:3334/udp'
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_NAME: bitmagnet
      POSTGRES_USER: bitmagnet
      POSTGRES_PASSWORD: armiger-local
      TMDB_API_KEY: ''
      DHT_CRAWLER_SCALING_FACTOR: '10'
    command: [worker, run, --keys=http_server, --keys=queue_server, --keys=dht_crawler]

  armiger-ultra:
    image: ghcr.io/spencernorton/armiger:3.4.1
    restart: unless-stopped
    command: [python, /opt/connectors/bitmagnet/bitmagnet_connector.py]
    depends_on:
      bitmagnet:
        condition: service_started
    ports:
      - '127.0.0.1:8096:8096'
      - '127.0.0.1:8097:8097'
    environment:
      ARMIGER_CONNECTOR_MODE: both
      ARMIGER_CONNECTOR_HOST: 0.0.0.0
      ARMIGER_CONNECTOR_PORT: '8096'
      ARMIGER_PROXY_PORT: '8097'
      ARMIGER_ENABLE_TORZNAB_PROXY: 'true'
      BITMAGNET_GRAPHQL_URL: http://bitmagnet:3333/graphql
      UPSTREAM_TORZNAB_URL: http://bitmagnet:3333/torznab
      BITMAGNET_POLL_SECONDS: '5'
      BITMAGNET_BATCH_SIZE: '100'
      ARMIGER_VARIANT: ultra
      ARMIGER_USE_ZERO_CUE_CATALOG: 'true'
      ARMIGER_REQUIRE_ENGLISH_AUDIO: 'false'
      ARMIGER_ALLOW_ENGLISH_SUBS: 'true'
      ARMIGER_REJECT_DEAD_SWARMS: 'false'
      ARMIGER_PRUNE_MALWARE: 'false'
      ARMIGER_PRUNE_DEAD_SWARMS: 'false'
      ARMIGER_DHT_AUTO_UPDATE: 'true'
      ARMIGER_DHT_POLL_INTERVAL_SEC: '14400'
      ARMIGER_DHT_SHARE_OPT_IN: 'false'
      ARMIGER_DHT_RSI_MODE: pull_only

volumes:
  postgres_data:
```

```bash
docker compose up -d
```

> **Note for headless homelab / NAS servers:** By default, [`docker-compose.yml`](docker-compose.yml) binds the web ports to `127.0.0.1` (`127.0.0.1:3333:3333`, `127.0.0.1:8096:8096`, `127.0.0.1:8097:8097`) so they are not exposed to untrusted networks. If you access Portainer or Docker from another machine on your private LAN and are not using an [SSH tunnel](../../docs/reference.md#remote-docker-server) or reverse proxy, remove the `127.0.0.1:` prefix on those `ports:` lines (for example, `'3333:3333'`).

### Option B: Two-line `compose.yaml` (Docker Compose 2.34+)

If you use Docker Compose 2.34 or newer, you can also include the published stack directly from GHCR:

```yaml
include:
  - oci://ghcr.io/spencernorton/armiger-bitmagnet-stack:3.4.1
```

```bash
docker compose up -d
```

If Docker asks whether you trust the included OCI setup, answer `y`. To customize environment variables with this two-line file, create a [`compose.override.yaml`](#changing-settings) in the same folder.

### Option C: Add Armiger Ultra to an existing Bitmagnet installation

If you already run Bitmagnet and want to keep your existing PostgreSQL database and index, add only the `armiger-ultra` service to your existing Bitmagnet Compose stack (or run it on the same Docker network):

```yaml
services:
  armiger-ultra:
    image: ghcr.io/spencernorton/armiger:3.4.1
    restart: unless-stopped
    command: [python, /opt/connectors/bitmagnet/bitmagnet_connector.py]
    ports:
      - '127.0.0.1:8096:8096'
      - '127.0.0.1:8097:8097'
    environment:
      BITMAGNET_GRAPHQL_URL: http://bitmagnet:3333/graphql
      UPSTREAM_TORZNAB_URL: http://bitmagnet:3333/torznab
      BITMAGNET_POLL_SECONDS: '5'
      BITMAGNET_BATCH_SIZE: '100'
      ARMIGER_REQUIRE_ENGLISH_AUDIO: 'false'
      ARMIGER_ALLOW_ENGLISH_SUBS: 'true'
      ARMIGER_REJECT_DEAD_SWARMS: 'false'
      ARMIGER_PRUNE_MALWARE: 'false'
      ARMIGER_PRUNE_DEAD_SWARMS: 'false'
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
| **Swarm viability** | `armiger-viability-alive-viable`, `armiger-viability-degraded`, `armiger-viability-stalled`, `armiger-viability-dead`, `armiger-viability-unknown-cold` | Estimated swarm health tier from `SwarmSurvivalNet` and tracker priors. Legacy hash-truncated viability tags (`armiger-viability-dea-c24281f7`, `armiger-viability-deg-7c170f8b`, `armiger-viability-sus-cdc9d2c6`) are automatically replaced and removed on startup. |
| **Anime & dub status** | `armiger-anime`, plus `armiger-dub-dual-audio-or-dub`, `armiger-dub-subbed-only`, or `armiger-dub-raw-or-vernacular` | Added when the item is classified as anime, along with its dub/subtitle classification. |
| **English subtitles** | `armiger-eng-subs` | Added when English subtitles (`has_eng_subs`) are detected in the release metadata or file list. |

In Bitmagnet's web UI or GraphQL API, filter by these tags to isolate specific categories, languages, or healthy swarms.

### Use the filtered Torznab proxy in Prowlarr, Sonarr, Radarr, Lidarr, or Readarr (`:8097`)

Instead of pointing your indexer client directly at Bitmagnet (`:3333/torznab`), point it at **Armiger's Torznab pre-grab proxy**:

- **From the same host:** `http://localhost:8097/torznab`
- **From another container on the same Docker network:** `http://armiger-ultra:8097/torznab`

When Prowlarr or an `*Arr` app searches through `:8097/torznab`, Armiger classifies every returned candidate before the client sees it:

1. **Always blocked automatically:** disguised executable lures (`.mkv.exe`, `.scr`, `.lnk`, `.vbs`, `.bat`, `.ps1`, etc.), safety-gate blocks (`csam_safe=False`), fake-seeder honeypot traps (`honeypot_trap=True`), and `junk` / failed Mondrian conformal gates.
2. **Optional policy filters:** set `ARMIGER_REQUIRE_ENGLISH_AUDIO: 'true'` (with `ARMIGER_ALLOW_ENGLISH_SUBS: 'true'` or `'false'`) to filter out foreign-audio releases, or `ARMIGER_REJECT_DEAD_SWARMS: 'true'` to drop dead swarms from search responses.
3. **Annotated XML & headers:** admitted items receive `<torznab:attr>` annotations (`armiger_category`, `armiger_lang`, `armiger_title_lang`, `armiger_viability`, `armiger_survival_prob`), and HTTP responses include `X-Armiger-Total`, `X-Armiger-Admitted`, and `X-Armiger-Blocked` headers.

### Check health, logs, and on-demand HTTP classification (`:8096`)

- **Health & model status (`GET http://localhost:8096/healthz`):** returns `"status": "ok"`, active model variant (`ultra` or `int8`), catalog/adapter readiness, active DHT sequence (`active_seq`), bundle Merkle root (`active_bundle_merkle_root`), `model_source` (`baked_image` or `dht_live_overlay`), `last_dht_check_ts`, and `sharing_status`.
- **Container logs (`docker compose logs -f armiger-ultra`):** shows startup configuration (`variant`, `interval`, `batch`) and polling errors if Bitmagnet is still starting.
- **On-demand batch classification (`POST http://localhost:8096/v1/bitmagnet/classify`):** accepts `{"torrents": [{"infoHash": "...", "name": "...", "size": 123, "files": [{"path": "...", "size": 123}]}], "applyMutations": false}` and returns normalized predictions plus the prepared GraphQL tag/delete mutations.
- **Legacy viability tag repair (`POST http://localhost:8096/v1/bitmagnet/repair-tags`):** scans Bitmagnet for legacy hash-truncated viability tags (`armiger-viability-dea-c24281f7`, `armiger-viability-deg-7c170f8b`, `armiger-viability-sus-cdc9d2c6`), writes the canonical replacement tags via `torrent.putTags`, and removes the legacy tags via `torrent.deleteTags`.

---

## 3. Environment variables reference

### Required vs. optional variables

- **When using the turnkey [`docker-compose.yml`](docker-compose.yml) (Option A) or two-line OCI stack (Option B):** **every setting is already pre-populated with a working default** — you do not need to set any required variables before running `docker compose up -d`.
- **When connecting `armiger-ultra` to an existing/external Bitmagnet stack (Option C):** only **`BITMAGNET_GRAPHQL_URL`** (for polling/tagging) and **`UPSTREAM_TORZNAB_URL`** (for the Torznab proxy) are **required** if your Bitmagnet service is not reachable at `http://bitmagnet:3333`. All other variables are optional.

### Changing settings

- **If using Option A (full [`docker-compose.yml`](docker-compose.yml) or Portainer Stacks):** edit the values directly under `environment:` or `ports:` in `docker-compose.yml` and run `docker compose up -d`. Keep boolean and numeric values in YAML quoted as strings (for example `'true'`, `'false'`, `'5'`, `'100'`).
- **If using Option B (two-line `compose.yaml`):** save a **`compose.override.yaml`** file in the same folder (or copy [`compose.override.example.yaml`](compose.override.example.yaml)) and run `docker compose config --quiet && docker compose up -d`:
  ```yaml
  services:
    armiger-ultra:
      environment:
        BITMAGNET_POLL_SECONDS: "5"
        BITMAGNET_BATCH_SIZE: "100"
        ARMIGER_REQUIRE_ENGLISH_AUDIO: "false"
        ARMIGER_ALLOW_ENGLISH_SUBS: "true"
  ```

### Stack, host ports, and Bitmagnet settings (`postgres` & `bitmagnet`)

These settings configure PostgreSQL credentials, Bitmagnet's DHT crawler, and host port bindings in [`docker-compose.yml`](docker-compose.yml):

| Variable / Compose setting | Required / Optional | Default in Compose | Valid options / range | What it does |
| --- | --- | --- | --- | --- |
| `POSTGRES_PASSWORD` | **Required** *(pre-filled in Compose)* | `armiger-local` | Any non-empty string | PostgreSQL password shared by `postgres` and `bitmagnet`. Change before first start if desired; keep unchanged when reusing an existing `postgres_data` volume. |
| `POSTGRES_DB` / `POSTGRES_NAME` | **Required** *(pre-filled in Compose)* | `bitmagnet` | Any valid PostgreSQL database name | PostgreSQL database name created by `postgres` (`POSTGRES_DB`) and connected to by `bitmagnet` (`POSTGRES_NAME`). |
| `POSTGRES_USER` | **Required** *(pre-filled in Compose)* | `bitmagnet` | Any valid PostgreSQL username | PostgreSQL username shared by `postgres` and `bitmagnet`. |
| `DHT_CRAWLER_SCALING_FACTOR` | Optional | `'10'` | Integer `'1'` – `'100'` *(recommended `'5'` – `'30'`)* | Bitmagnet DHT crawler concurrency multiplier (`bitmagnet` service). Higher values discover more torrents per minute; scale `BITMAGNET_BATCH_SIZE` proportionally when raising this above `'30'`. |
| `TMDB_API_KEY` | Optional | `''` *(empty)* | Valid TMDB v3 API key string, or `''` *(disabled)* | Optional TMDB API key passed to `bitmagnet` for Bitmagnet's built-in TMDB matching. Armiger's own offline catalog does not require a TMDB key. |
| `bitmagnet` HTTP port (`ports:`) | Optional | `'127.0.0.1:3333:3333'` | `'<host_ip>:<host_port>:3333'` or `'<host_port>:3333'` *(ports `1`–`65535`)* | Host port mapped to Bitmagnet's Web UI, GraphQL API, and raw Torznab endpoint. |
| `bitmagnet` DHT port (`ports:`) | Optional | `'3334:3334/tcp'` & `'3334:3334/udp'` | `'<host_port>:3334/tcp'` & `/udp` *(ports `1`–`65535`)* | Host TCP and UDP port mapped to Bitmagnet's DHT crawler. |
| `armiger-ultra` HTTP port (`ports:`) | Optional | `'127.0.0.1:8096:8096'` | `'<host_ip>:<host_port>:8096'` or `'<host_port>:8096'` *(ports `1`–`65535`)* | Host port mapped to Armiger's HTTP status and classification service (`/healthz`, `/v1/bitmagnet/classify`). |
| `armiger-ultra` Torznab port (`ports:`) | Optional | `'127.0.0.1:8097:8097'` | `'<host_ip>:<host_port>:8097'` or `'<host_port>:8097'` *(ports `1`–`65535`)* | Host port mapped to Armiger's filtered Torznab proxy (`/torznab`, `/api`). |

### Polling and Bitmagnet connection (`armiger-ultra`)

| Variable | Required / Optional | Default | Valid options / range | What it does |
| --- | --- | --- | --- | --- |
| `BITMAGNET_GRAPHQL_URL` | **Required** *(for `both`/`poll`; pre-filled in Compose)* | `http://bitmagnet:3333/graphql` | Any valid `http://` or `https://` URL ending in `/graphql` | Bitmagnet GraphQL endpoint queried for untagged items (`torrentContent.search`), file lists (`torrent.files`), tag writes (`torrent.putTags`), and optional deletions (`torrent.delete`). Defaults to `http://localhost:3333/graphql` when running the CLI script outside Compose. |
| `UPSTREAM_TORZNAB_URL` | **Required** *(for `both`/`http`; pre-filled in Compose)* | `http://bitmagnet:3333/torznab` | Any valid `http://` or `https://` URL ending in `/torznab` | Upstream Bitmagnet Torznab endpoint proxied and filtered by Armiger. Defaults to `http://localhost:3333/torznab` outside Compose. |
| `ARMIGER_CONNECTOR_MODE` | Optional | `both` | `'both'`, `'http'`, `'poll'` | `'both'` runs the background Bitmagnet GraphQL poller and HTTP/Torznab servers. `'http'` runs only the HTTP/Torznab servers without polling. `'poll'` runs only the background poller *(note: container healthcheck expects `'both'` or `'http'`)*. |
| `BITMAGNET_POLL_SECONDS` | Optional | `'5'` | Number `>= 1` *(recommended `'2'` – `'60'` seconds)* | Seconds between background GraphQL polling cycles. |
| `BITMAGNET_BATCH_SIZE` | Optional | `'100'` | Integer `'10'` – `'1000'` *(recommended `'50'` – `'500'` items)* | Maximum untagged Bitmagnet torrents classified per polling cycle. When a 60-second Bitmagnet crawl flush exceeds one batch, consecutive cycles paginate via `offset`/`hasNextPage` past items already tagged earlier in the active flush. |

> **How polling, batch size, and `DHT_CRAWLER_SCALING_FACTOR` interact:** Stock Bitmagnet buffers newly crawled DHT torrents in a batching channel (`persistTorrents`) that flushes to PostgreSQL at **1,000 items or every 60 seconds**, whichever comes first. With `BITMAGNET_POLL_SECONDS: '5'` and `BITMAGNET_BATCH_SIZE: '100'`, Armiger paginates across each 60-second flush in 100-item slices—processing up to **1,200 torrents/minute** while keeping individual ONNX inference and `torrent.files` queries small. If you raise `DHT_CRAWLER_SCALING_FACTOR` above `'30'` or use a longer poll interval (such as `BITMAGNET_POLL_SECONDS: '15'` or `'30'`), increase `BITMAGNET_BATCH_SIZE` so `(60 / BITMAGNET_POLL_SECONDS) * BITMAGNET_BATCH_SIZE` exceeds your crawler's per-minute intake.

### Torznab pre-grab filtering (`armiger-ultra`)

These variables control how Armiger filters search results requested through `http://localhost:8097/torznab` (or `:8096/torznab`). They do not delete items from Bitmagnet's database.

| Variable | Required / Optional | Default | Valid options / range | What it does |
| --- | --- | --- | --- | --- |
| `ARMIGER_REQUIRE_ENGLISH_AUDIO` | Optional | `'false'` | `'true'`, `'false'` | When `'true'`, blocks Torznab search results classified as `lang == "non_english"`, unless permitted by `ARMIGER_ALLOW_ENGLISH_SUBS` below. Items with `english` or `multi_or_unknown` audio are admitted. |
| `ARMIGER_ALLOW_ENGLISH_SUBS` | Optional | `'true'` | `'true'`, `'false'` | When `ARMIGER_REQUIRE_ENGLISH_AUDIO='true'`, still admits `non_english` audio items if English subtitles (`english_subs == "has_eng_subs"`) are detected. Set to `'false'` for strict English audio only. |
| `ARMIGER_REJECT_DEAD_SWARMS` | Optional | `'false'` *(Compose)* / `'true'` *(CLI script)* | `'true'`, `'false'` | When `'true'`, excludes results classified as `dead` from Torznab responses without deleting them from Bitmagnet. |
| `ARMIGER_ENABLE_TORZNAB_PROXY` | Optional | `'true'` | `'true'`, `'false'` | Starts the dedicated Torznab proxy listener on `ARMIGER_PROXY_PORT` (`8097`) in `'both'` or `'http'` mode. When `'false'`, `/torznab` and `/api` remain served on `ARMIGER_CONNECTOR_PORT` (`8096`). |

### Index pruning / deleting flagged items from Bitmagnet (`armiger-ultra`)

Both deletion flags default to **`'false'` in the packaged Compose setup** so no indexed records are deleted unless you opt in:

| Variable | Required / Optional | Default | Valid options / range | What it does |
| --- | --- | --- | --- | --- |
| `ARMIGER_PRUNE_MALWARE` | Optional | `'false'` *(Compose)* / `'true'` *(CLI script)* | `'true'`, `'false'` | When `'true'`, the poller deletes torrents flagged as unsafe (`csam_safe=False`), disguised executables (`.mkv.exe`, `.scr`, `.lnk`, etc.), `junk`, or honeypot traps from Bitmagnet's index via `mutation TorrentDelete`. |
| `ARMIGER_PRUNE_DEAD_SWARMS` | Optional | `'false'` | `'true'`, `'false'` | When `'true'`, the poller deletes torrents classified as `dead` from Bitmagnet's index via `mutation TorrentDelete`. |

### Model, catalog, and container listeners (`armiger-ultra`)

| Variable | Required / Optional | Default | Valid options / range | What it does |
| --- | --- | --- | --- | --- |
| `ARMIGER_VARIANT` | Optional | `ultra` | `'ultra'`, `'int8'` *(aliases for `'int8'`: `'cpu'`, `'armiger'`)* | Selects `'ultra'` (31.9 MB FP32 ONNX model + Ultra plugins, pre-bundled in the image) or `'int8'` (8.2 MB INT8 ONNX model + Ultra plugins; downloads the pinned `all` bundle when `ARMIGER_AUTO_DOWNLOAD='true'`). |
| `ARMIGER_USE_ZERO_CUE_CATALOG` | Optional | `'true'` | `'true'`, `'false'` | When `'true'`, preloads the hashed offline title catalog (`validated_zero_cue_catalog.json`) at startup. When `'false'`, defers loading until first lookup. |
| `ARMIGER_MODEL_DIR` | Optional | `/models` *(image)* / `.` *(CLI script)* | Valid container/host directory path | Directory containing the verified model bundle. The container image pre-bundles the Ultra files in `/models`. |
| `ARMIGER_AUTO_DOWNLOAD` | Optional | `'true'` | `'true'`, `'false'` | When `'true'`, downloads and SHA-256-verifies any missing model files against the pinned manifest. When `'false'`, runs strictly offline and requires all files to exist locally. |
| `ARMIGER_CONNECTOR_HOST` | Optional | `0.0.0.0` | `'0.0.0.0'`, `'127.0.0.1'`, or valid interface IP | Bind address inside the container/process for HTTP and Torznab listeners (fallback alias: `ARMIGER_BRIDGE_HOST`). |
| `ARMIGER_CONNECTOR_PORT` | Optional | `'8096'` | Integer `'1'` – `'65535'` | Container/process port for `/healthz`, `/v1/bitmagnet/classify`, and `/torznab` (fallback alias: `ARMIGER_BRIDGE_PORT`). |
| `ARMIGER_PROXY_PORT` | Optional | `'8097'` | Integer `'1'` – `'65535'` | Container/process port for the dedicated Torznab proxy listener (`/torznab` and `/api`). |

### Automatic model updates (BEP 44/46) and optional telemetry (`armiger-ultra`)

Armiger Ultra automatically checks for newer signed model and catalog releases in the background, verifies their cryptographic signatures and 500-item canary regression suite, and hot-swaps them without restarting the container.

**Zero extra configuration is required:** the official Ed25519 release verification key, release channel salt, Mainline DHT bootstrap routers, and Hugging Face HTTPS fallback mirror are **already built in**, and seeder/collector endpoints are discovered automatically from the signed release pointer. You only ever need the 5 settings below:

| Variable | Required / Optional | Default | Valid options / range | What it does |
| --- | --- | --- | --- | --- |
| `ARMIGER_DHT_AUTO_UPDATE` | Optional | `'true'` | `'true'`, `'false'` *(also accepts `'1'`, `'0'`, `'off'`)* | Enables automatic background checks for newer signed model/catalog releases (`seq > active_seq`), verified download, 500-item canary check, and atomic hot-swap. Set to `'false'` to disable background update checks and stay pinned to the image's baked-in model. |
| `ARMIGER_DHT_POLL_INTERVAL_SEC` | Optional | `'14400'` *(4 hours)* | Number `>= 60` *(recommended `'3600'` – `'86400'` seconds)* | How often (in seconds) Armiger checks for a newer signed model release (`'3600'` = 1 hour, `'14400'` = 4 hours, `'86400'` = 24 hours). |
| `ARMIGER_DHT_SHARE_OPT_IN` | Optional | `'false'` | `'true'`, `'false'` *(also accepts `'1'`, `'0'`, `'yes'`, `'no'`)* | Opt-in anonymous contribution switch. **Off (`'false'`) by default.** When `'false'`, all outbound contribution packets are blocked at the socket guard. Even when enabled, Armiger never sends torrent names, infohashes, or unverified predictions. |
| `ARMIGER_DHT_RSI_MODE` | Optional | `pull_only` | `'pull_only'`, `'off'`, `'anonymous_contribute'` | Distributed update policy:<br>• `'pull_only'` *(default)* — downloads verified model updates only.<br>• `'off'` — disables the distributed update engine.<br>• `'anonymous_contribute'` — enables anonymized differential-privacy telemetry when `ARMIGER_DHT_SHARE_OPT_IN='true'`. |
| `ARMIGER_RUNTIME_CACHE_DIR` | Optional | `/tmp/armiger-live-models-<uid>` | Any writable container directory path *(e.g. `/data/armiger/live-models`)* | Writable directory where verified live-update overlays are staged, leaving `/models` inside the image read-only. Mount a Docker volume here if you want hot-swapped updates to persist across container recreation. |

<details>
<summary><strong>Advanced: Pre-populated release channel &amp; mirror overrides (only for custom forks or private mirrors — no changes needed)</strong></summary>

These variables are **already pre-populated inside Armiger** so automatic updates work out of the box. Override them only if you run a custom-signed model channel or private mirror:

| Variable | Required / Optional | Built-in Default | Valid options / range | What it does |
| --- | --- | --- | --- | --- |
| `ARMIGER_RELEASE_PUBKEY_HEX` | Optional *(pre-populated)* | `3599362bd52dad8b89398b904d3081c2cb483d13376f7ffb49f3748208731e76` | 64-character hex Ed25519 public key | Official Ed25519 public key used to verify signed BEP 44/46 release pointers. |
| `ARMIGER_RELEASE_SALT` | Optional *(pre-populated)* | `armiger:weights:ultra` *(or `armiger:weights:int8`)* | Non-empty string (`'armiger:weights:ultra'`, `'armiger:weights:int8'`) | BEP 44 channel salt used with `ARMIGER_RELEASE_PUBKEY_HEX` to derive the DHT lookup target. |
| `ARMIGER_DHT_BOOTSTRAP_NODES` | Optional *(pre-populated)* | `router.bittorrent.com:6881,dht.transmissionbt.com:6881,router.utorrent.com:6881` | Comma-separated `host:port` list | Public Mainline DHT bootstrap routers queried during release discovery. |
| `ARMIGER_WEBSEED_URLS` | Optional *(pre-populated)* | `https://huggingface.co/NorviTech/armiger-ultra/resolve/main` | Comma-separated `https://` base URLs | Official BEP 19 HTTPS fallback mirror used when direct BitTorrent peer-wire transfer is blocked by a firewall/NAT; downloaded pieces are still verified against the signed infohash and Merkle root. |

</details>

See [distributed updates](../../docs/distributed-updates.md) for protocol and cryptographic verification details.

---

## Stop, start, and update

From your stack folder, run `docker compose down` to stop or `docker compose up -d` to start again. Your PostgreSQL index stays saved in `postgres_data` (do not pass `-v` unless you want to erase the database).

If startup reports `Permission denied` opening `/opt/connectors/bitmagnet/bitmagnet_connector.py`, set the `armiger-ultra` image to `ghcr.io/spencernorton/armiger:3.4.1` or later, then pull and recreate that service:

```bash
docker compose pull armiger-ultra
docker compose up -d --no-deps armiger-ultra
```

Version 3.4.1 fixes directory permissions inside the image. These commands preserve the existing Bitmagnet service and PostgreSQL index.

[Detailed Bitmagnet configuration guide](../../docs/bitmagnet-configuration.md) · [Docker settings and updates](../../docs/reference.md#docker-settings-and-updates) · [Using a remote server](../../docs/reference.md#remote-docker-server)
