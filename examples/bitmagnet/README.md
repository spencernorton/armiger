# Bitmagnet + Armiger Ultra

Start a new Bitmagnet installation with Armiger Ultra already connected. You need [Docker](https://docs.docker.com/get-started/get-docker/) running, with Docker Compose available.

## 1. Download the setup

Create a folder called `armiger-bitmagnet`. Save [this Compose file](https://raw.githubusercontent.com/spencernorton/armiger/main/examples/bitmagnet/docker-compose.yml) inside it as **`compose.yaml`**. If it opens in your browser, use **Save As**.

The file is ready to use. You do not need to edit it, create a password file, install Python or build anything.

## 2. Start it

Open a terminal in that folder and run:

```bash
docker compose up -d
```

Docker downloads the ready-made containers, including Armiger’s verified model. Allow several minutes for the first download.

## 3. Open Bitmagnet

Open **[localhost:3333](http://localhost:3333)** on the computer running Docker. Armiger automatically labels newly indexed items; look for tags such as `armiger-movie`. A new index starts empty and needs time to find items.

To check Armiger itself, open [its status page](http://localhost:8096/healthz). When ready, it shows `"status": "ok"`. If it is still starting, run `docker compose logs --tail=50 armiger-ultra`.

## Stop and start

From the same folder, run `docker compose down` to stop or `docker compose up -d` to start again. Your index stays saved. Adding `-v` to the stop command erases them.

The setup contains Bitmagnet, its private database and Armiger. Their connection settings are already filled in. Automatic deletion of flagged items is off.

[Optional settings and updates](../../docs/reference.md#docker-settings-and-updates) · [Using a remote server](../../docs/reference.md#remote-docker-server)
