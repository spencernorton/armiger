# Armiger Ultra settings for Bitmagnet

These settings apply to the `armiger-ultra` service in the [Bitmagnet setup](../examples/bitmagnet/README.md), using `ghcr.io/spencernorton/armiger:3.3.0`. The image bundles the verified Ultra model and the [pinned Bitmagnet connector](https://github.com/spencernorton/Armiger-Connectors/blob/a18596f8a371afcd7df339f3fe6fb288749f55fe/bitmagnet/bitmagnet_connector.py). Defaults below are the effective defaults of that packaged setup.

## Change a setting

Keep the existing two-line `compose.yaml`. Create **`compose.override.yaml`** in the same folder:

```yaml
services:
  armiger-ultra:
    environment:
      BITMAGNET_POLL_SECONDS: "10"
      BITMAGNET_BATCH_SIZE: "50"
      ARMIGER_REQUIRE_ENGLISH_AUDIO: "false"
      ARMIGER_ALLOW_ENGLISH_SUBS: "true"
```

Then run these commands from that folder:

```bash
docker compose config --quiet
docker compose up -d
docker compose logs --tail=50 armiger-ultra
```

Compose automatically merges the override into the included setup. The example changes the poll interval from five to ten seconds and the maximum batch from one hundred to fifty. The startup log reports the active interval and batch size. For more common settings, save the [commented example](../examples/bitmagnet/compose.override.example.yaml) as `compose.override.yaml`. Docker documents [overrides for included Compose files](https://docs.docker.com/compose/how-tos/multiple-compose-files/include/#using-overrides-with-included-compose-files).

Keep boolean values quoted as `"true"` or `"false"`. The connector's boolean flags recognise `true`, ignoring case; use the quoted forms shown here. Use positive numbers for polling and batch size.

### Using a `.env` file

A `.env` file supplies Compose interpolation values. It does not automatically add arbitrary variables to a container. To use one, explicitly reference the setting in `compose.override.yaml`:

```yaml
services:
  armiger-ultra:
    environment:
      BITMAGNET_POLL_SECONDS: ${BITMAGNET_POLL_SECONDS:-5}
      BITMAGNET_BATCH_SIZE: ${BITMAGNET_BATCH_SIZE:-100}
```

Then save `.env` beside the two Compose files:

```dotenv
BITMAGNET_POLL_SECONDS=10
BITMAGNET_BATCH_SIZE=50
```

Apply with `docker compose up -d`. An explicit `environment` entry takes precedence over the same variable in a container `env_file`. If you use `-f` explicitly, include both Compose files in the command.

## Polling and operation

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `ARMIGER_CONNECTOR_MODE` | `both` | `both` runs the poller and HTTP services. `http` serves requests without background polling. `poll` runs only the poller. Keep `both` with the supplied health check; `poll` has no HTTP health endpoint. |
| `BITMAGNET_POLL_SECONDS` | `5.0` | Seconds between polling cycles. Higher values reduce how often new items are checked. |
| `BITMAGNET_BATCH_SIZE` | `100` | Maximum recent Bitmagnet records requested in each cycle. Already tagged or recently seen items are skipped. |
| `BITMAGNET_GRAPHQL_URL` | `http://bitmagnet:3333/graphql` | Internal Bitmagnet endpoint used to read items and write tags or enabled prune mutations. Use the service name inside Compose. |

The poller labels recent, untagged entries. Changing model or filtering settings does not automatically reclassify entries that already have an `armiger-` or `armiger:` tag. There is no force-relabel environment flag in this connector.

## Models and catalog

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `ARMIGER_MODEL_DIR` | `/models` | Folder containing the compatible, verified model bundle. The image already includes its Ultra files here. A different folder must contain the needed files or allow verified downloads. |
| `ARMIGER_VARIANT` | `ultra` | Uses the FP32 Ultra model. `int8` selects the INT8 model while retaining the Ultra plugins; `cpu` and `armiger` are aliases for that selection. INT8 requests the additional pinned `all` bundle, which is not fully bundled in this image. |
| `ARMIGER_AUTO_DOWNLOAD` | `true` | Allows missing model files to be downloaded and verified against the package's pinned manifest. `false` requires the selected files to be present and valid already. |
| `ARMIGER_USE_ZERO_CUE_CATALOG` | `true` | Preloads the title catalog during startup. `false` defers loading until it is first used; it does not disable catalog lookups. |
| `ARMIGER_DHT_RSI_MODE` | `pull_only` | Selects the update engine's policy: `off`, `pull_only` or `anonymous_contribute`. The packaged connector explicitly keeps outbound sharing disabled. This variable does not configure trusted publishers or start a peer-network service. See [distributed updates](distributed-updates.md). |

All variants in this connector use the Ultra pipeline. For the separate lightweight standalone path, see the [Python reference](reference.md#standalone-onnx).

## Torznab filtering

These flags affect results requested through Armiger's Torznab endpoint, normally `http://localhost:8097/torznab` for a client on the Docker host. A client on the same Compose network uses `http://armiger-ultra:8097/torznab`. A client that queries Bitmagnet directly bypasses these checks.

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `ARMIGER_REQUIRE_ENGLISH_AUDIO` | `false` | When enabled, rejects results explicitly classified as `non_english`, subject to the subtitle exception below. Unknown language is not automatically rejected. |
| `ARMIGER_ALLOW_ENGLISH_SUBS` | `true` | With the English-audio rule enabled, allows a non-English result when English subtitles are detected. This is metadata inference. |
| `ARMIGER_REJECT_DEAD_SWARMS` | `false` | When enabled, excludes results classified as `dead` from the proxy response. It does not remove those records from Bitmagnet. |

The proxy also screens safety blocks, disguised executables, honeypot/junk predictions and failed admission gates. `ARMIGER_PRUNE_MALWARE=false` does not turn off that proxy screening; it controls index removal below.

## Removing entries from Bitmagnet

Both flags below are **off in the supplied Compose setup**. Turning one on allows the poller to remove matching records from Bitmagnet's index. These are index mutations, not operations on downloaded media files.

| Variable | Packaged default | Effect when `true` |
| --- | --- | --- |
| `ARMIGER_PRUNE_MALWARE` | `false` | Removes entries flagged as unsafe, disguised executables, junk or honeypot traps. |
| `ARMIGER_PRUNE_DEAD_SWARMS` | `false` | Removes entries classified as `dead`. This is separate from filtering a Torznab response. |

The bare connector script defaults `ARMIGER_PRUNE_MALWARE` and `ARMIGER_REJECT_DEAD_SWARMS` to `true`; the packaged Compose file explicitly sets them to `false`. Keep those overrides when adapting the script outside this setup.

## Endpoints and listeners

| Variable | Packaged default | Effect |
| --- | --- | --- |
| `UPSTREAM_TORZNAB_URL` | `http://bitmagnet:3333/torznab` | Internal upstream feed that Armiger proxies. |
| `ARMIGER_CONNECTOR_HOST` | `0.0.0.0` | Listening address inside the container. The supplied Compose file publishes web ports only on the host's loopback address. |
| `ARMIGER_CONNECTOR_PORT` | `8096` | Main HTTP listener for `/healthz`, `/v1/bitmagnet/classify` and `/torznab`. |
| `ARMIGER_PROXY_PORT` | `8097` | Additional listener used for Torznab requests. |
| `ARMIGER_ENABLE_TORZNAB_PROXY` | `true` | Starts the additional proxy listener in `both` or `http` mode. `false` leaves `/torznab` available on the main listener. |

Keep the container listener ports unchanged with the supplied port mappings and image health check. To resolve a host-port clash, change the host-side binding. The [full Compose file settings](reference.md#docker-settings-and-updates) document the host variables `BITMAGNET_HTTP_PORT`, `BITMAGNET_DHT_PORT`, `ARMIGER_HTTP_PORT`, `ARMIGER_PROXY_PORT`, `TMDB_API_KEY` and `POSTGRES_PASSWORD`.

`ARMIGER_PROXY_PORT` has two scopes: in the full file's `.env` it changes the host binding, while its value inside the service's `environment` changes the container listener. The shipped file keeps the container listener at `8097`.

## Compatibility and limits

- `ARMIGER_BRIDGE_HOST` and `ARMIGER_BRIDGE_PORT` are fallback aliases. `ARMIGER_CONNECTOR_HOST` and `ARMIGER_CONNECTOR_PORT` take precedence and are already set by the supplied Compose file.
- `ARMIGER_HF_REPO` belongs to the connector's legacy loader. The prebuilt image uses the installed package and its pinned model manifest, so this variable does not select a different download repository here.
- `ARMIGER_DHT_SHARE_OPT_IN` does not enable sharing in this packaged connector, which passes `opt_in_sharing=False` explicitly. Enabling a distributed publisher requires a separately configured application and trusted authority setup.
- The standalone `python -m armiger serve` command has its own CLI options. Connector-only polling and proxy variables do not configure that command.
