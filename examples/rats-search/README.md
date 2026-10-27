# Rats Search + Armiger Ultra

Start a new Rats Search server with Armiger Ultra already connected. This is a background server with an API; it does **not** include the Rats Search desktop interface or a web search page.

You need [Docker](https://docs.docker.com/get-started/get-docker/) running, with **Docker Compose 2.34 or newer** (included in current Docker Desktop). Rats Search runs as an x86-64 container; Docker Desktop uses emulation on Apple Silicon.

## 1. Save the setup

Create a folder called `armiger-rats`. Save this as **`compose.yaml`** inside it:

```yaml
include:
  - oci://ghcr.io/spencernorton/armiger-rats-search-stack:3.3.2
```

This loads the complete setup from our versioned package. All connections are already configured.

## 2. Start it

Open a terminal in that folder and run:

```bash
docker compose up -d
```

Docker downloads the ready-made containers, including Armiger’s verified model. Allow several minutes for the first download. If Docker asks whether you trust this setup, answer `y`.

## 3. Check it is ready

Open **[Armiger status](http://localhost:8098/healthz)** on the computer running Docker. When ready, it shows `"status": "ok"`. Armiger automatically categorizes newly indexed items; a fresh index may initially be empty.

Rats Search's API is at `http://localhost:8095`. This is a connection address for software, not a search page. If Armiger is still starting, run `docker compose logs --tail=50 armiger-ultra`.

## Stop and start

From the same folder, run `docker compose down` to stop or `docker compose up -d` to start again. Your index stays saved. Adding `-v` to the stop command erases that saved data. Automatic deletion of flagged items is off.

[Custom settings, older Docker versions and updates](../../docs/reference.md#docker-settings-and-updates) · [Using a remote server](../../docs/reference.md#remote-docker-server)
