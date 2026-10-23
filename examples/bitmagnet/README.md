# Bitmagnet + Armiger Ultra

Armiger adds content-type and language tags to items in Bitmagnet. This example starts a **new Bitmagnet installation**, its database and Armiger Ultra together.

## 1. Get the files

Install [Docker](https://docs.docker.com/get-started/get-docker/) and [Git](https://git-scm.com/downloads), and make sure Docker is running. Docker Desktop includes Docker Compose; Linux users also need the [Compose plugin](https://docs.docker.com/compose/install/linux/).

If you already downloaded Armiger, use its `examples/bitmagnet` folder and skip the first two commands below. Keep an existing `.env` instead of copying over it.

For a new download, open a terminal and run:

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger/examples/bitmagnet
cp .env.example .env
```

## 2. Start

Open `.env` in a text editor. Replace `replace-with-a-password` with a password of your choice, save it, then run:

```bash
docker compose up -d --build
```

The first start builds Armiger and downloads its model. Allow several minutes. You do not need Python, a GPU or a separate Armiger install. A TMDB API key is optional; you can leave it blank.

## 3. Check it works

```bash
docker compose ps
```

`postgres` and `armiger-ultra` should say `healthy`; `bitmagnet` should be running. While the model downloads, Armiger may show `starting`.

On the computer running Docker, open:

- **[Bitmagnet](http://localhost:3333)** — your index and search page.
- **[Armiger status](http://localhost:8096/healthz)** — should show `"status": "ok"`.

Armiger labels newly indexed items automatically. Look for `armiger-movie`, `armiger-tv-show` and language tags in Bitmagnet. A new index starts empty; no results yet does not mean Armiger failed.

## Everyday use

Run these from the same folder:

| Task | Command |
| --- | --- |
| See Armiger's logs | `docker compose logs --tail=50 armiger-ultra` |
| Stop | `docker compose down` |
| Start again | `docker compose up -d` |
| Update | `git pull --ff-only`, then `docker compose up -d --build` |

Stopping keeps the database and downloaded model. Keep the same database password when starting again. Do not add `-v` to the stop command unless you intend to erase this example's saved data.

If a port is already in use, change its value in `.env` and start again; use that new port in your browser. For a remote server, use the [remote access instructions](../../docs/reference.md#remote-docker-server) to open these links from your own computer. Automatic removal of flagged items is off; normal setup only adds labels.
