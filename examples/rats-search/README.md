# Rats Search + Armiger Ultra

Starts Rats Search in console mode and an Armiger Ultra classifier. The connector polls recently indexed metadata and writes content categories to Rats Search's Manticore index.

## Start

Install Docker with Compose v2 and Git. The pinned upstream Rats Search image builds from source and includes an x86-64 Manticore binary; ARM machines use Docker's x86-64 emulation. Its first build is slower than Bitmagnet's and needs several GB of disk and build memory.

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger/examples/rats-search
cp .env.example .env
docker compose up -d --build
docker compose ps
```

The defaults enable the Rats REST API and indexer on first initialization. Armiger then downloads roughly 59 MB of verified model data. No GPU, API key or manual model installation is needed.

- **Rats Search REST API:** <http://localhost:8095> (console backend, not a browser UI).
- **Armiger health:** <http://localhost:8098/healthz>.

Both HTTP ports bind to localhost and can be changed in `.env`. Rats data and Armiger models live in separate named volumes. The connector shares Rats Search's container network to reach its loopback-only Manticore service; it does not require host networking.

## Verify classification

```bash
curl -fsS http://localhost:8098/healthz
curl -fsS http://localhost:8098/v1/rats-search/classify \
  -H 'Content-Type: application/json' \
  -d '{"torrents":[{"name":"Tears.of.Steel.2012.1080p.BluRay.x264-OPEN","size":2400000000,"files":[{"path":"Tears.of.Steel.2012.mkv","size":2400000000}]}]}'
```

Expect the example to be classified as `movie`. This request evaluates metadata without adding a torrent to the index. Newly indexed torrents are categorized by the background poller; a fresh crawler may initially have no items. Check connector logs for REST or Manticore errors.

Automatic deletion is off (`ARMIGER_PRUNE_MALWARE=false`). Set it to `true` only if you want flagged items removed. Predictions are metadata checks, not payload inspection.

## Stop, update and troubleshoot

```bash
docker compose logs --tail=50 rats-search armiger-ultra
docker compose down
git pull --ff-only
docker compose up -d --build
```

`down` preserves data; `down -v` deletes this stack's index and model volumes. If the upstream build runs out of memory, increase Docker's memory allocation and rebuild. If the REST health check fails with an existing volume, ensure `/data/rats.json` contains `"restApi": true` and `"httpPort": 8095`; initial configuration is not overwritten on restart.

To connect Armiger to an existing desktop/server installation instead, see [the connector repository](https://github.com/spencernorton/Armiger-Connectors/tree/main/rats-search).
