# Bitmagnet + Armiger Ultra

Starts PostgreSQL, stock Bitmagnet and an Armiger Ultra sidecar. Armiger classifies crawled metadata and adds searchable tags. CPU inference is the default; no model account or GPU is needed.

## Start

Install Docker with Compose v2, then:

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger/examples/bitmagnet
cp .env.example .env
```

Set `POSTGRES_PASSWORD` in `.env` to your own password. Other values can stay at their defaults. Then:

```bash
docker compose up -d --build
docker compose ps
```

The first build installs Python dependencies; the first start downloads roughly 59 MB of verified model data. Allow several minutes. PostgreSQL and the models use named volumes and survive container restarts.

- **Bitmagnet:** <http://localhost:3333>
- **Armiger health:** <http://localhost:8096/healthz>
- **Filtered Torznab:** <http://localhost:8097/torznab> for clients on this host.

HTTP ports bind to localhost. `BITMAGNET_DHT_PORT` exposes the BitTorrent crawler's TCP/UDP port. Set `TMDB_API_KEY` in `.env` only if you want Bitmagnet's own TMDB enrichment.

## Verify classification

```bash
curl -fsS http://localhost:8096/healthz
curl -fsS http://localhost:8096/v1/bitmagnet/classify \
  -H 'Content-Type: application/json' \
  -d '{"torrents":[{"name":"Tears.of.Steel.2012.1080p.BluRay.x264-OPEN","size":2400000000,"files":[{"path":"Tears.of.Steel.2012.mkv","size":2400000000}]}]}'
```

The response should classify the example as `movie`. This request only evaluates the supplied metadata. As Bitmagnet indexes real items, the poller writes `armiger-*` tags to its database; an empty new index can take time to populate. Check logs for GraphQL errors if tags do not appear.

Automatic deletion is disabled initially (`ARMIGER_PRUNE_MALWARE=false`). Enable it only if you want flagged torrents removed from the index. Armiger's safety predictions describe metadata, not inspected payloads.

## Stop, update and troubleshoot

```bash
docker compose logs --tail=50 armiger-ultra bitmagnet
docker compose down
# From the same checkout after pulling a reviewed release:
git pull --ff-only
docker compose up -d --build
```

`down` keeps data; `down -v` deletes this stack's database and model volumes. Change ports in `.env` if already occupied. Keep the original database password when reusing its volume. If first-start downloads fail, inspect logs and restart `armiger-ultra`; verified files are reused.

For an existing indexer or additional integrations, see [Armiger-Connectors](https://github.com/spencernorton/Armiger-Connectors).
