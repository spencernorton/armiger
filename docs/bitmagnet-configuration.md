# Armiger Ultra settings for Bitmagnet

These settings apply to the [Bitmagnet + Armiger Ultra setup](../examples/bitmagnet/README.md), using `ghcr.io/spencernorton/armiger:3.4.3`. The image bundles the verified Ultra model and the [pinned Bitmagnet connector](https://github.com/spencernorton/Armiger-Connectors/blob/9b00089a06df24497e45d2f2144eb14a1bb03ac2/bitmagnet/bitmagnet_connector.py). Defaults below are the effective defaults of that packaged setup.

## Required vs. optional variables

- **With the packaged [`examples/bitmagnet/docker-compose.yml`](../examples/bitmagnet/docker-compose.yml) or two-line OCI stack:** **all settings are pre-filled with working defaults** — no additional environment variables are required before running `docker compose up -d`.
- **With an external/existing Bitmagnet deployment:** only **`BITMAGNET_GRAPHQL_URL`** (for polling/tagging) and **`UPSTREAM_TORZNAB_URL`** (for the filtered Torznab proxy) are **required** if Bitmagnet is not at `http://bitmagnet:3333`. Every other variable is optional.

## Change a setting

### With the full `docker-compose.yml` or Portainer Stacks

[`examples/bitmagnet/docker-compose.yml`](../examples/bitmagnet/docker-compose.yml) has all default values written directly in the file (no `${VAR}` placeholders or `.env` file needed). Copy-paste the file into your stack directory or Portainer Stack editor, edit any value under `environment:` or `ports:` directly in place, and run:

```bash
docker compose up -d
```

Keep boolean and numeric environment values in YAML quoted as strings (for example `'true'`, `'false'`, `'5'`, `'100'`).

### With the two-line `compose.yaml` (`include: oci://...`)

Keep the existing two-line `compose.yaml`. Create **`compose.override.yaml`** in the same folder:

```yaml
services:
  armiger-ultra:
    environment:
      BITMAGNET_POLL_SECONDS: "5"
      BITMAGNET_BATCH_SIZE: "100"
      ARMIGER_REQUIRE_ENGLISH_AUDIO: "false"
      ARMIGER_ALLOW_ENGLISH_SUBS: "true"
```

Then run these commands from that folder:

```bash
docker compose config --quiet
docker compose up -d
docker compose logs --tail=50 armiger-ultra
```

Compose automatically merges the override into the included setup. The startup log reports the active interval and batch size. For more common settings, save the [commented example](../examples/bitmagnet/compose.override.example.yaml) as `compose.override.yaml`. Docker documents [overrides for included Compose files](https://docs.docker.com/compose/how-tos/multiple-compose-files/include/#using-overrides-with-included-compose-files).

## Stack, host ports and Bitmagnet service

These settings configure PostgreSQL credentials, Bitmagnet's DHT crawler, and host port bindings in [`examples/bitmagnet/docker-compose.yml`](../examples/bitmagnet/docker-compose.yml):

| Variable / Compose setting | Required / Optional | Packaged default | Valid options / range | Effect |
| --- | --- | --- | --- | --- |
| `POSTGRES_PASSWORD` | **Required** *(pre-filled in Compose)* | `armiger-local` | Any non-empty string | PostgreSQL password shared by the `postgres` and `bitmagnet` services. Set before first start; keep unchanged while reusing the `postgres_data` volume. |
| `POSTGRES_DB` / `POSTGRES_NAME` | **Required** *(pre-filled in Compose)* | `bitmagnet` | Valid PostgreSQL database name | PostgreSQL database name created on `postgres` (`POSTGRES_DB`) and connected to by `bitmagnet` (`POSTGRES_NAME`). |
| `POSTGRES_USER` | **Required** *(pre-filled in Compose)* | `bitmagnet` | Valid PostgreSQL username | PostgreSQL user shared by `postgres` and `bitmagnet`. |
| `DHT_CRAWLER_SCALING_FACTOR` | Optional | `'10'` | Integer `'1'` – `'100'` *(recommended `'5'` – `'30'`)* | Bitmagnet's DHT crawler concurrency multiplier. Higher values increase crawled torrents per minute; scale `BITMAGNET_BATCH_SIZE` proportionally if raising this above `'30'`. |
| `TMDB_API_KEY` | Optional | `''` *(empty)* | Valid TMDB v3 API key string, or `''` *(disabled)* | Optional TMDB API key passed to `bitmagnet` for Bitmagnet's built-in TMDB lookup. Armiger's offline catalog does not require a TMDB key. |
| `bitmagnet` HTTP port (`ports:`) | Optional | `'127.0.0.1:3333:3333'` | `'<host_ip>:<host_port>:3333'` or `'<host_port>:3333'` *(ports `1`–`65535`)* | Host port mapped to Bitmagnet's Web UI, GraphQL API, and raw Torznab endpoint. |
| `bitmagnet` DHT port (`ports:`) | Optional | `'3334:3334/tcp'` & `'3334:3334/udp'` | `'<host_port>:3334/tcp'` & `/udp` *(ports `1`–`65535`)* | Host TCP and UDP port mapped to Bitmagnet's DHT crawler. |
| `armiger-ultra` HTTP port (`ports:`) | Optional | `'127.0.0.1:8096:8096'` | `'<host_ip>:<host_port>:8096'` or `'<host_port>:8096'` *(ports `1`–`65535`)* | Host port mapped to Armiger's HTTP status and classification endpoint (`/healthz`, `/v1/bitmagnet/classify`). |
| `armiger-ultra` Torznab port (`ports:`) | Optional | `'127.0.0.1:8097:8097'` | `'<host_ip>:<host_port>:8097'` or `'<host_port>:8097'` *(ports `1`–`65535`)* | Host port mapped to Armiger's filtered Torznab proxy (`/torznab`, `/api`). |

## Polling and operation

| Variable | Required / Optional | Packaged default | Valid options / range | Effect |
| --- | --- | --- | --- | --- |
| `BITMAGNET_GRAPHQL_URL` | **Required** *(for `both`/`poll`; pre-filled in Compose)* | `http://bitmagnet:3333/graphql` | Valid `http://` or `https://` URL ending in `/graphql` | Internal Bitmagnet endpoint used to read items and write tags or enabled prune mutations. Use the service name inside Compose (defaults to `http://localhost:3333/graphql` in the standalone CLI script). |
| `ARMIGER_CONNECTOR_MODE` | Optional | `both` | `'both'`, `'http'`, `'poll'` | `'both'` runs the poller and HTTP services. `'http'` serves requests without background polling. `'poll'` runs only the poller. Keep `'both'` with the supplied health check; `'poll'` has no HTTP health endpoint. |
| `BITMAGNET_POLL_SECONDS` | Optional | `'5'` (`5.0`) | Number `>= 1` *(recommended `'2'` – `'60'` seconds)* | Seconds between polling cycles. Higher values reduce how often new items are checked. |
| `BITMAGNET_BATCH_SIZE` | Optional | `'100'` | Integer `'10'` – `'1000'` *(recommended `'50'` – `'500'` items)* | Maximum untagged Bitmagnet records classified in each cycle. When a Bitmagnet crawl flush exceeds one batch, the poller pages past items already tagged earlier in the active flush on subsequent cycles. |

Stock Bitmagnet buffers newly crawled DHT torrents in a batching channel (`persistTorrents`) that flushes at 1,000 items or every 60 seconds. With the default `5.0`-second poll interval and `100`-item batch size, the poller paginates through each 60-second flush across consecutive cycles (up to 1,200 torrents per minute) while keeping individual ONNX inference and `torrent.files` requests small. If you increase Bitmagnet's `DHT_CRAWLER_SCALING_FACTOR` above 30, or prefer less frequent polling (such as `BITMAGNET_POLL_SECONDS=15` or `30`), increase `BITMAGNET_BATCH_SIZE` proportionally so `(60 / BITMAGNET_POLL_SECONDS) * BITMAGNET_BATCH_SIZE` exceeds your crawler's per-minute intake.

The poller labels recent, untagged entries. Changing model or filtering settings does not automatically reclassify entries that already have an `armiger-` or `armiger:` tag, except for legacy hash-truncated viability tags (`armiger-viability-dea-c24281f7`, `armiger-viability-deg-7c170f8b`, and `armiger-viability-sus-cdc9d2c6`), which are automatically migrated to `armiger-viability-dead`, `armiger-viability-degraded`, and `armiger-viability-stalled` and removed on startup (or on demand via `POST /v1/bitmagnet/repair-tags`).

## Models and catalog

| Variable | Required / Optional | Packaged default | Valid options / range | Effect |
| --- | --- | --- | --- | --- |
| `ARMIGER_VARIANT` | Optional | `ultra` | `'ultra'`, `'int8'` *(aliases for `'int8'`: `'cpu'`, `'armiger'`)* | `'ultra'` uses the FP32 Ultra model (`31.9 MB`). `'int8'` selects the INT8 model (`8.2 MB`) while retaining the Ultra plugins. INT8 requests the additional pinned `all` bundle if not already present. |
| `ARMIGER_USE_ZERO_CUE_CATALOG` | Optional | `'true'` | `'true'`, `'false'` | Preloads the hashed offline title catalog (`validated_zero_cue_catalog.json`) during startup. `'false'` defers loading until first lookup; it does not disable catalog lookups. |
| `ARMIGER_MODEL_DIR` | Optional | `/models` | Valid container/host directory path | Folder containing the compatible, verified model bundle. The image already includes its Ultra files here. A different folder must contain the needed files or allow verified downloads. |
| `ARMIGER_AUTO_DOWNLOAD` | Optional | `'true'` | `'true'`, `'false'` | Allows missing model files to be downloaded and verified against the package's pinned manifest. `'false'` requires the selected files to be present and valid already, and disables background network updates. |
| `ARMIGER_HF_REPO` | Optional | `NorviTech/armiger-ultra` | Hugging Face `<org>/<repo>` ID | Used only by the connector's legacy fallback loader when the `armiger` Python package is not installed. The prebuilt image uses the installed package and its pinned model manifest. |

All variants in this connector use the Ultra pipeline. For the separate lightweight standalone path, see the [Python reference](reference.md#standalone-onnx).

## Distributed model updates (BEP 44/46) and opt-in sharing

Armiger Ultra includes a background BEP 44/46 update poller that checks for newer signed model/catalog bundles (`seq > active_seq`), verifies piece hashes and Merkle roots, runs the 500-item canary regression gate, and atomically hot-swaps the active session into a private writable overlay directory (`ARMIGER_RUNTIME_CACHE_DIR`).

**Automatic updates work out of the box with zero extra configuration:** the official Ed25519 release signing key, release channel salt, Mainline DHT bootstrap routers, and Hugging Face HTTPS fallback mirror are **already built into the image**, and seeder/collector nodes are discovered automatically from the signed DHT pointer. Outbound telemetry/contribution sharing is **off by default** (`ARMIGER_DHT_SHARE_OPT_IN='false'`), and ordinary classification never sends torrent names or unverified predictions. See [distributed updates](distributed-updates.md) for full details.

| Variable | Required / Optional | Packaged default | Valid options / range | Effect |
| --- | --- | --- | --- | --- |
| `ARMIGER_DHT_AUTO_UPDATE` | Optional | `'true'` | `'true'`, `'false'` *(also accepts `'1'`, `'0'`, `'off'`)* | Enables background BEP 44/46 polling for newer signed model/catalog releases (`seq > active_seq`), piece verification, canary testing, and atomic overlay hot-swap. Set to `'false'` to disable background update polling. |
| `ARMIGER_DHT_POLL_INTERVAL_SEC` | Optional | `'14400'` | Number `>= 60` *(recommended `'3600'` – `'86400'` seconds; `'14400'` = 4 hours)* | Seconds between background BEP 44/46 release checks (`'3600'` = 1 hour, `'14400'` = 4 hours, `'86400'` = 24 hours). |
| `ARMIGER_DHT_SHARE_OPT_IN` | Optional | `'false'` | `'true'`, `'false'` *(also accepts `'1'`, `'0'`, `'yes'`, `'no'`)* | Strict opt-in switch (`'true'` to enable; `'false'` by default). When `'false'`, all outbound contribution packets are blocked at the socket guard. |
| `ARMIGER_DHT_RSI_MODE` | Optional | `pull_only` | `'pull_only'`, `'off'`, `'anonymous_contribute'` | Selects the update engine's policy: `'pull_only'` (receive verified updates only), `'off'` (disable update engine), or `'anonymous_contribute'` (when `ARMIGER_DHT_SHARE_OPT_IN='true'`). |
| `ARMIGER_RUNTIME_CACHE_DIR` | Optional | `/tmp/armiger-live-models-<uid>` | Any writable container directory path *(e.g. `/data/armiger/live-models`)* | Private (`0700`) writable directory where verified DHT live-update overlays and `dht_overlay_state_*.json` are staged (since `/models` inside the image is read-only `0444`). Mount a persistent volume here to keep hot-swapped updates across container recreation. |

<details>
<summary><strong>Advanced: Pre-populated release channel &amp; mirror defaults (only for custom forks or private mirrors — no changes needed)</strong></summary>

These variables are **already pre-populated inside Armiger** so you do not need to set them unless you operate a custom release key or private mirror:

| Variable | Required / Optional | Built-in Default | Valid options / range | Effect |
| --- | --- | --- | --- | --- |
| `ARMIGER_RELEASE_PUBKEY_HEX` | Optional *(pre-populated)* | `3599362bd52dad8b89398b904d3081c2cb483d13376f7ffb49f3748208731e76` | 64-hex-character Ed25519 public key | Official Ed25519 public key verified on BEP 44/46 mutable release pointers. |
| `ARMIGER_RELEASE_SALT` | Optional *(pre-populated)* | `armiger:weights:ultra` *(or `armiger:weights:int8`)* | Non-empty string (`'armiger:weights:ultra'`, `'armiger:weights:int8'`) | BEP 44 salt used with `ARMIGER_RELEASE_PUBKEY_HEX` to derive the mutable DHT lookup target `SHA-1(pubkey || salt)`. |
| `ARMIGER_DHT_BOOTSTRAP_NODES` | Optional *(pre-populated)* | `router.bittorrent.com:6881,dht.transmissionbt.com:6881,router.utorrent.com:6881` | Comma-separated `host:port` list | Mainline DHT bootstrap nodes queried during BEP 44 release discovery and contribution dispatch. |
| `ARMIGER_WEBSEED_URLS` | Optional *(pre-populated)* | `https://huggingface.co/NorviTech/armiger-ultra/resolve/main` | Comma-separated `https://` base URLs | BEP 19 HTTPS webseed base URLs used as fallback if direct TCP peer-wire transfer is unreachable; pieces are still verified against the signed infohash (`ih`) and Merkle root (`mr`). |

</details>

## Torznab filtering

These flags affect results requested through Armiger's Torznab endpoint, normally `http://localhost:8097/torznab` for a client on the Docker host. A client on the same Compose network uses `http://armiger-ultra:8097/torznab`. A client that queries Bitmagnet directly (`:3333/torznab`) bypasses these checks.

| Variable | Required / Optional | Packaged default | Valid options / range | Effect |
| --- | --- | --- | --- | --- |
| `ARMIGER_REQUIRE_ENGLISH_AUDIO` | Optional | `'false'` | `'true'`, `'false'` | When `'true'`, rejects results explicitly classified as `non_english`, subject to the subtitle exception below. Unknown or multi-audio language (`multi_or_unknown`) is not automatically rejected. |
| `ARMIGER_ALLOW_ENGLISH_SUBS` | Optional | `'true'` | `'true'`, `'false'` | With `ARMIGER_REQUIRE_ENGLISH_AUDIO='true'`, allows a non-English result when English subtitles (`has_eng_subs`) are detected. This is metadata inference. |
| `ARMIGER_REJECT_DEAD_SWARMS` | Optional | `'false'` | `'true'`, `'false'` | When `'true'`, excludes results classified as `dead` from the proxy response. It does not remove those records from Bitmagnet. |

The proxy always screens safety blocks (`csam_safe=False`), disguised executables (`.mkv.exe`, `.scr`, `.lnk`, `.vbs`, `.bat`, `.ps1`, etc.), honeypot/junk predictions, and failed Mondrian admission gates. `ARMIGER_PRUNE_MALWARE='false'` does not turn off that proxy screening; it controls index removal below.

## Removing entries from Bitmagnet

Both flags below are **off (`'false'`) in the supplied Compose setup**. Turning one on allows the poller to remove matching records from Bitmagnet's index via `mutation TorrentDelete`. These are index mutations, not operations on downloaded media files.

| Variable | Required / Optional | Packaged default | Valid options / range | Effect when `'true'` |
| --- | --- | --- | --- | --- |
| `ARMIGER_PRUNE_MALWARE` | Optional | `'false'` | `'true'`, `'false'` | Removes entries flagged as unsafe, disguised executables, junk or honeypot traps. |
| `ARMIGER_PRUNE_DEAD_SWARMS` | Optional | `'false'` | `'true'`, `'false'` | Removes entries classified as `dead`. This is separate from filtering a Torznab response. |

The bare connector script defaults `ARMIGER_PRUNE_MALWARE` and `ARMIGER_REJECT_DEAD_SWARMS` to `true`; the packaged Compose file explicitly sets them to `'false'`. Keep those overrides when adapting the script outside this setup.

## Endpoints and listeners

| Variable | Required / Optional | Packaged default | Valid options / range | Effect |
| --- | --- | --- | --- | --- |
| `UPSTREAM_TORZNAB_URL` | **Required** *(for `both`/`http`; pre-filled in Compose)* | `http://bitmagnet:3333/torznab` | Valid `http://` or `https://` URL ending in `/torznab` | Internal upstream feed that Armiger proxies. |
| `ARMIGER_CONNECTOR_HOST` | Optional | `0.0.0.0` | `'0.0.0.0'`, `'127.0.0.1'`, or valid interface IP | Listening address inside the container. The supplied Compose file publishes web ports only on the host's loopback address (`127.0.0.1`) by default. |
| `ARMIGER_CONNECTOR_PORT` | Optional | `'8096'` | Integer `'1'` – `'65535'` | Main HTTP listener for `/healthz`, `/v1/bitmagnet/classify` and `/torznab`. |
| `ARMIGER_PROXY_PORT` | Optional | `'8097'` | Integer `'1'` – `'65535'` | Additional listener used for Torznab requests (`/torznab` and `/api`). |
| `ARMIGER_ENABLE_TORZNAB_PROXY` | Optional | `'true'` | `'true'`, `'false'` | Starts the additional proxy listener in `'both'` or `'http'` mode. `'false'` leaves `/torznab` available on the main listener (`8096`). |

Keep the container listener ports unchanged with the supplied port mappings and image health check. To resolve a host-port clash, change the left-hand host port on the `ports:` lines in `docker-compose.yml` (for example `'127.0.0.1:18096:8096'`).

## Compatibility notes

- `ARMIGER_BRIDGE_HOST` and `ARMIGER_BRIDGE_PORT` are fallback aliases. `ARMIGER_CONNECTOR_HOST` and `ARMIGER_CONNECTOR_PORT` take precedence and are already set by the supplied Compose file.
- `ARMIGER_HF_REPO` belongs to the connector's legacy loader. The prebuilt image uses the installed package and its pinned model manifest, so this variable does not select a different download repository here.
- The standalone `python -m armiger serve` command has its own CLI options (`--host`, `--port`, `--model-dir`, `--variant`, `--offline`). Connector-only polling and proxy variables (`BITMAGNET_*`, `UPSTREAM_TORZNAB_URL`, `ARMIGER_PRUNE_*`) apply to `bitmagnet_connector.py`, whereas model and DHT update variables (`ARMIGER_MODEL_DIR`, `ARMIGER_DHT_*`, `ARMIGER_RUNTIME_CACHE_DIR`) apply to both `ArmigerUltraPipeline` callers.
