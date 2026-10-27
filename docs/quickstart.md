# Standalone Armiger Ultra

Run Armiger Ultra as a local classification service. The [Bitmagnet](../examples/bitmagnet/README.md) and [Rats Search](../examples/rats-search/README.md) setups already include it; use this guide when you want Armiger on its own.

## 1. Save the setup

With [Docker](https://docs.docker.com/get-started/get-docker/) running and Docker Compose available, create a folder called `armiger`. Save this as **`compose.yaml`** inside it:

```yaml
services:
  armiger-ultra:
    image: ghcr.io/spencernorton/armiger:3.3.1
    restart: unless-stopped
    ports:
      - '127.0.0.1:8096:8096'
```

## 2. Start it

Open a terminal in that folder and run:

```bash
docker compose up -d
```

Docker downloads the container, including the verified Ultra model. Allow several minutes for the first download. You do not need Python, a GPU or an API key.

## 3. Try it

Open **[localhost:8096](http://localhost:8096)**. When ready, it shows `"status": "ok"`. This is a service for other software to call, not a search interface.

To classify a name, run this in a terminal on macOS or Linux:

```bash
curl -H 'Content-Type: application/json' -d '{"name":"Tears.of.Steel.2012.1080p.BluRay.x264-OPEN"}' http://localhost:8096/v1/classify
```

The JSON response includes `"category": "movie"`. This classifies metadata; it does not download the film. Other software can send the same JSON to `/v1/classify`; see [input and output](reference.md#input-and-output) for more fields.

## Stop and start

From the same folder, run `docker compose down` to stop or `docker compose up -d` to start again. The model is part of the image; you do not need a separate model folder. For startup errors, run `docker compose logs --tail=50`.

[Python installation](reference.md#python-installation) · [Updates](reference.md#docker-settings-and-updates) · [Remote server](reference.md#remote-docker-server)
