# Rats Search + Armiger Ultra

Start a new Rats Search server with Armiger Ultra already connected. This is a background server with an API; it does **not** include the Rats Search desktop interface or a web search page.

You need [Docker](https://docs.docker.com/get-started/get-docker/) running, with Docker Compose available. Rats Search runs as an x86-64 container; Docker Desktop uses emulation on Apple Silicon.

## 1. Download the setup

Create a folder called `armiger-rats`. Save [this Compose file](https://raw.githubusercontent.com/spencernorton/armiger/main/examples/rats-search/docker-compose.yml) inside it as **`compose.yaml`**. If it opens in your browser, use **Save As**.

The file is ready to use. You do not need to edit it, install Python or compile Rats Search.

## 2. Start it

Open a terminal in that folder and run:

```bash
docker compose up -d
```

Docker downloads the ready-made containers, including Armiger’s verified model. Allow several minutes for the first download.

## 3. Check it is ready

Open **[Armiger status](http://localhost:8098/healthz)** on the computer running Docker. When ready, it shows `"status": "ok"`. Armiger automatically categorizes newly indexed items; a fresh index may initially be empty.

Rats Search's API is at `http://localhost:8095`. This is a connection address for software, not a search page. If Armiger is still starting, run `docker compose logs --tail=50 armiger-ultra`.

## Stop and start

From the same folder, run `docker compose down` to stop or `docker compose up -d` to start again. Your index stays saved. Adding `-v` to the stop command erases them. Automatic deletion of flagged items is off.

[Optional settings and updates](../../docs/reference.md#docker-settings-and-updates) · [Using a remote server](../../docs/reference.md#remote-docker-server)
