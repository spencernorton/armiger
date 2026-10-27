# Armiger Ultra settings for Bitmagnet

These settings apply to the [Bitmagnet + Armiger Ultra setup](../examples/bitmagnet/README.md), using `ghcr.io/spencernorton/armiger:3.3.1`. The image bundles the verified Ultra model and the [pinned Bitmagnet connector](https://github.com/spencernorton/Armiger-Connectors/blob/58679a4f0fd28de4ebee9133a23a6a10bb6adaf0/bitmagnet/bitmagnet_connector.py). Defaults below are the effective defaults of that packaged setup.

## Change a setting

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

Keep boolean values quoted as `"true"` or `"false"`. The connector's boolean flags recognise `true`, ignoring case; use the quoted forms shown here. Use positive numbers for polling and batch size.

### With the full `docker-compose.yml`, Portainer Stacks, or a `.env` file

When using the full [`examples/bitmagnet/docker-compose.yml`](../examples/bitmagnet/docker-compose.yml) (recommended for **Portainer Stacks** or older Docker Compose versions), all common stack and connector variables are already wired for `${VAR:-default}` interpolation. You can set them directly in Portainer's **Environment variables** UI or in a `.env` file beside `compose.yaml`:

```dotenv
BITMAGNET_HTTP_PORT=3333
BITMAGNET_DHT_PORT=3334
ARMIGER_HTTP_PORT=8096
ARMIGER_PROXY_PORT=8097
POSTGRES_PASSWORD=armiger-local
TMDB_API_KEY=
DHT_CRAWLER_SCALING_FACTOR=10
BITMAGNET_POLL_SECONDS=5
BITMAGNET_BATCH_SIZE=100
ARMIGER_REQUIRE_ENGLISH_AUDIO=false
ARMIGER_ALLOW_ENGLISH_SUBS=true
ARMIGER_REJECT_DEAD_SWARMS=false
ARMIGER_PRUNE_MALWARE=false
ARMIGER_PRUNE_DEAD_SWARMS=false
```

If you use the two-line `include:` file and want to read from `.env`, reference the variable in `compose.override.yaml` (for example `BITMAGNET_POLL_SECONDS: ${BITMAGNET_POLL_SECONDS:-5}`) because Compose only interpolates `.env` values that are referenced in a loaded Compose file.

## Stack, host ports and Bitmagnet service

These variables configure host port bindings, PostgreSQL credentials, and Bitmagnet's DHT crawler in [`examples/bitmagnet/docker-compose.yml`](../examples/bitmagnet/docker-compose.yml):

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `BITMAGNET_HTTP_PORT` | `3333` | Host port mapped to Bitmagnet's Web UI, GraphQL API, and raw Torznab endpoint (`127.0.0.1:${BITMAGNET_HTTP_PORT:-3333}:3333`). |
| `BITMAGNET_DHT_PORT` | `3334` | Host TCP and UDP port mapped to Bitmagnet's DHT crawler (`${BITMAGNET_DHT_PORT:-3334}:3334/tcp` and `/udp`). |
| `ARMIGER_HTTP_PORT` | `8096` | Host port mapped to Armiger's HTTP status and classification endpoint (`127.0.0.1:${ARMIGER_HTTP_PORT:-8096}:8096`). |
| `ARMIGER_PROXY_PORT` | `8097` | In `.env` / host port mappings, sets the host port mapped to Armiger's filtered Torznab proxy (`127.0.0.1:${ARMIGER_PROXY_PORT:-8097}:8097`). Inside `armiger-ultra.environment`, sets the container listener port (`8097`). |
| `POSTGRES_PASSWORD` | `armiger-local` | PostgreSQL password shared by the `postgres` and `bitmagnet` services. Set before first start; keep unchanged while reusing the `postgres_data` volume. |
| `TMDB_API_KEY` | `""` (empty) | Optional TMDB API key passed to `bitmagnet` for Bitmagnet's built-in TMDB lookup. Armiger's offline catalog does not require a TMDB key. |
| `DHT_CRAWLER_SCALING_FACTOR` | `10` | Bitmagnet's DHT crawler concurrency multiplier. Higher values increase crawled torrents per minute; scale `BITMAGNET_BATCH_SIZE` proportionally if raising this above `30`. |

## Polling and operation

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `ARMIGER_CONNECTOR_MODE` | `both` | `both` runs the poller and HTTP services. `http` serves requests without background polling. `poll` runs only the poller. Keep `both` with the supplied health check; `poll` has no HTTP health endpoint. |
| `BITMAGNET_POLL_SECONDS` | `5.0` | Seconds between polling cycles. Higher values reduce how often new items are checked. |
| `BITMAGNET_BATCH_SIZE` | `100` | Maximum untagged Bitmagnet records classified in each cycle. When a Bitmagnet crawl flush exceeds one batch, the poller pages past items already tagged earlier in the active flush on subsequent cycles. |
| `BITMAGNET_GRAPHQL_URL` | `http://bitmagnet:3333/graphql` | Internal Bitmagnet endpoint used to read items and write tags or enabled prune mutations. Use the service name inside Compose (defaults to `http://localhost:3333/graphql` in the standalone CLI script). |

Stock Bitmagnet buffers newly crawled DHT torrents in a batching channel (`persistTorrents`) that flushes at 1,000 items or every 60 seconds. With the default `5.0`-second poll interval and `100`-item batch size, the poller paginates through each 60-second flush across consecutive cycles (up to 1,200 torrents per minute) while keeping individual ONNX inference and `torrent.files` requests small. If you increase Bitmagnet's `DHT_CRAWLER_SCALING_FACTOR` above 30, or prefer less frequent polling (such as `BITMAGNET_POLL_SECONDS=15` or `30`), increase `BITMAGNET_BATCH_SIZE` proportionally so `(60 / BITMAGNET_POLL_SECONDS) * BITMAGNET_BATCH_SIZE` exceeds your crawler's per-minute intake.

The poller labels recent, untagged entries. Changing model or filtering settings does not automatically reclassify entries that already have an `armiger-` or `armiger:` tag. There is no force-relabel environment flag in this connector.

## Models and catalog

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `ARMIGER_MODEL_DIR` | `/models` | Folder containing the compatible, verified model bundle. The image already includes its Ultra files here. A different folder must contain the needed files or allow verified downloads. |
| `ARMIGER_VARIANT` | `ultra` | Uses the FP32 Ultra model (`31.9 MB`). `int8` selects the INT8 model (`8.2 MB`) while retaining the Ultra plugins; `cpu` and `armiger` are aliases for that selection. INT8 requests the additional pinned `all` bundle if not already present. |
| `ARMIGER_AUTO_DOWNLOAD` | `true` | Allows missing model files to be downloaded and verified against the package's pinned manifest. `false` requires the selected files to be present and valid already, and disables background network updates. |
| `ARMIGER_USE_ZERO_CUE_CATALOG` | `true` | Preloads the hashed offline title catalog (`validated_zero_cue_catalog.json`) during startup. `false` defers loading until first lookup; it does not disable catalog lookups. |
| `ARMIGER_HF_REPO` | `NorviTech/armiger-ultra` | Used only by the connector's legacy fallback loader when the `armiger` Python package is not installed. The prebuilt image uses the installed package and its pinned model manifest. |

All variants in this connector use the Ultra pipeline. For the separate lightweight standalone path, see the [Python reference](reference.md#standalone-onnx).

## Distributed model updates (BEP 44/46) and opt-in sharing

Armiger Ultra includes a background BEP 44/46 update poller that checks for newer signed model/catalog bundles (`seq > active_seq`), verifies piece hashes and Merkle roots, runs the 500-item canary regression gate, and atomically hot-swaps the active session into a private writable overlay directory (`ARMIGER_RUNTIME_CACHE_DIR`). Outbound telemetry/contribution sharing is **off by default** (`ARMIGER_DHT_SHARE_OPT_IN=false`), and ordinary classification never sends torrent names or unverified predictions. See [distributed updates](distributed-updates.md) for full details.

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `ARMIGER_DHT_AUTO_UPDATE` | `true` | Enables background BEP 44/46 polling for newer signed model/catalog releases (`seq > active_seq`), piece verification, canary testing, and atomic overlay hot-swap. Set to `false` (or `0`/`off`) to disable background update polling. |
| `ARMIGER_DHT_POLL_INTERVAL_SEC` | `14400` | Seconds between background BEP 44/46 release checks (default 4 hours; minimum `0.2`). |
| `ARMIGER_RUNTIME_CACHE_DIR` | `/tmp/armiger-live-models-<uid>` | Private (`0700`) writable directory where verified DHT live-update overlays and `dht_overlay_state_*.json` are staged (since `/models` inside the image is read-only `0444`). Mount a persistent volume here to keep hot-swapped updates across container recreation. |
| `ARMIGER_DHT_RSI_MODE` | `pull_only` | Selects the update engine's policy: `pull_only` (receive verified updates only), `off`, or `anonymous_contribute` (when `ARMIGER_DHT_SHARE_OPT_IN=true`). |
| `ARMIGER_DHT_SHARE_OPT_IN` | `false` (`0`) | Strict opt-in switch (`true`, `1`, or `yes` to enable; `false` by default). When `false`, all outbound contribution packets are blocked at the socket guard. |
| `ARMIGER_RELEASE_PUBKEY_HEX` | `3599362bd52dad8b89398b904d3081c2cb483d13376f7ffb49f3748208731e76` | 64-hex-character Ed25519 public key verified on BEP 44/46 mutable release pointers. |
| `ARMIGER_RELEASE_SALT` | `armiger:weights:ultra` (or `armiger:weights:int8`) | BEP 44 salt used with `ARMIGER_RELEASE_PUBKEY_HEX` to derive the mutable DHT lookup target `SHA-1(pubkey \|\| salt)`. |
| `ARMIGER_LOCAL_SEQ` | Auto-detected (`3` or `4`) | Optional integer override for the initial local release sequence number (`active_seq`) before polling for newer DHT releases. |
| `ARMIGER_DHT_BOOTSTRAP_NODES` | `router.bittorrent.com:6881,dht.transmissionbt.com:6881,router.utorrent.com:6881` | Comma-separated `host:port` list of Mainline DHT bootstrap nodes queried during BEP 44 release discovery and contribution dispatch. |
| `ARMIGER_DHT_PEERS` | `""` (empty) | Optional comma-separated `host:port` list of explicit BitTorrent peer-wire seeders (`:6882`) tried alongside peers announced in the BEP 44 pointer. |
| `ARMIGER_WEBSEED_URLS` | `https://huggingface.co/NorviTech/armiger-ultra/resolve/main` | Comma-separated BEP 19 HTTPS webseed base URLs used as fallback if direct TCP peer-wire transfer is unreachable; pieces are still verified against the signed infohash (`ih`) and Merkle root (`mr`). |
| `ARMIGER_DHT_COLLECTOR_NODES` | `""` (empty) | Optional comma-separated `host:port` list of collector nodes for direct UDP contribution delivery when `ARMIGER_DHT_SHARE_OPT_IN=true`. |
| `ARMIGER_CATALOG_HMAC_KEY` | Random 32-byte key | Optional secret key (`>= 16` chars or `hex:<64-hex-chars>`) used by `AnonymityGuard` to key `armiger-catalog-v2` standalone catalog hashes across restarts. |
| `ARMIGER_PUBLIC_IPV4` | `""` (empty) | Optional public IPv4 address used by `python -m armiger.verify_dht` when generating a BEP 42 CRC32C-compliant DHT node ID behind NAT. |

## Torznab filtering

These flags affect results requested through Armiger's Torznab endpoint, normally `http://localhost:8097/torznab` for a client on the Docker host. A client on the same Compose network uses `http://armiger-ultra:8097/torznab`. A client that queries Bitmagnet directly (`:3333/torznab`) bypasses these checks.

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `ARMIGER_REQUIRE_ENGLISH_AUDIO` | `false` | When enabled, rejects results explicitly classified as `non_english`, subject to the subtitle exception below. Unknown or multi-audio language (`multi_or_unknown`) is not automatically rejected. |
| `ARMIGER_ALLOW_ENGLISH_SUBS` | `true` | With the English-audio rule enabled, allows a non-English result when English subtitles (`has_eng_subs`) are detected. This is metadata inference. |
| `ARMIGER_REJECT_DEAD_SWARMS` | `false` | When enabled, excludes results classified as `dead` from the proxy response. It does not remove those records from Bitmagnet. |

The proxy always screens safety blocks (`csam_safe=False`), disguised executables (`.mkv.exe`, `.scr`, `.lnk`, `.vbs`, `.bat`, `.ps1`, etc.), honeypot/junk predictions, and failed Mondrian admission gates. `ARMIGER_PRUNE_MALWARE=false` does not turn off that proxy screening; it controls index removal below.

## Removing entries from Bitmagnet

Both flags below are **off in the supplied Compose setup**. Turning one on allows the poller to remove matching records from Bitmagnet's index via `mutation TorrentDelete`. These are index mutations, not operations on downloaded media files.

| Variable | Packaged default | Effect when `true` |
| --- | --- | --- |
| `ARMIGER_PRUNE_MALWARE` | `false` | Removes entries flagged as unsafe, disguised executables, junk or honeypot traps. |
| `ARMIGER_PRUNE_DEAD_SWARMS` | `false` | Removes entries classified as `dead`. This is separate from filtering a Torznab response. |

The bare connector script defaults `ARMIGER_PRUNE_MALWARE` and `ARMIGER_REJECT_DEAD_SWARMS` to `true`; the packaged Compose file explicitly sets them to `false`. Keep those overrides when adapting the script outside this setup.

## Endpoints and listeners

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `UPSTREAM_TORZNAB_URL` | `http://bitmagnet:3333/torznab` | Internal upstream feed that Armiger proxies. |
| `ARMIGER_CONNECTOR_HOST` | `0.0.0.0` | Listening address inside the container. The supplied Compose file publishes web ports only on the host's loopback address (`127.0.0.1`) by default. |
| `ARMIGER_CONNECTOR_PORT` | `8096` | Main HTTP listener for `/healthz`, `/v1/bitmagnet/classify` and `/torznab`. |
| `ARMIGER_PROXY_PORT` | `8097` | Additional listener used for Torznab requests (`/torznab` and `/api`). |
| `ARMIGER_ENABLE_TORZNAB_PROXY` | `true` | Starts the additional proxy listener in `both` or `http` mode. `false` leaves `/torznab` available on the main listener (`8096`). |

Keep the container listener ports unchanged with the supplied port mappings and image health check. To resolve a host-port clash, change the host-side binding (`BITMAGNET_HTTP_PORT`, `BITMAGNET_DHT_PORT`, `ARMIGER_HTTP_PORT`, `ARMIGER_PROXY_PORT`).

`ARMIGER_PROXY_PORT` has two scopes: in the full file's `.env` it changes the host binding, while its value inside the service's `environment` changes the container listener. The shipped file keeps the container listener at `8097`.

## Compatibility notes

- `ARMIGER_BRIDGE_HOST` and `ARMIGER_BRIDGE_PORT` are fallback aliases. `ARMIGER_CONNECTOR_HOST` and `ARMIGER_CONNECTOR_PORT` take precedence and are already set by the supplied Compose file.
- `ARMIGER_HF_REPO` belongs to the connector's legacy loader. The prebuilt image uses the installed package and its pinned model manifest, so this variable does not select a different download repository here.
- The standalone `python -m armiger serve` command has its own CLI options (`--host`, `--port`, `--model-dir`, `--variant`, `--offline`). Connector-only polling and proxy variables (`BITMAGNET_*`, `UPSTREAM_TORZNAB_URL`, `ARMIGER_PRUNE_*`) apply to `bitmagnet_connector.py`, whereas model and DHT update variables (`ARMIGER_MODEL_DIR`, `ARMIGER_DHT_*`, `ARMIGER_RUNTIME_CACHE_DIR`) apply to both `ArmigerUltraPipeline` callers.
