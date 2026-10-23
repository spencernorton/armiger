# Rats Search + Armiger Ultra

Armiger automatically assigns content categories to items found by Rats Search. This example starts a **new Rats Search server** and Armiger Ultra together. It runs in the background and **does not include the Rats Search desktop interface or a web search page**.

## 1. Get the files

Install [Docker](https://docs.docker.com/get-started/get-docker/) and [Git](https://git-scm.com/downloads), and make sure Docker is running. Docker Desktop includes Docker Compose; Linux users also need the [Compose plugin](https://docs.docker.com/compose/install/linux/).

```bash
git clone https://github.com/spencernorton/armiger.git
cd armiger/examples/rats-search
cp .env.example .env
```

If you already cloned this repository, use its `examples/rats-search` folder. Keep an existing `.env`; otherwise copy `.env.example` as shown. The default settings are ready to use.

## 2. Start

```bash
docker compose up -d --build
```

The first build compiles Rats Search and can take a while. Allow several GB of free disk space. On Apple Silicon and other ARM computers, Docker must be able to run x86-64 containers; this example needs emulation for Rats Search's bundled search engine.

Armiger downloads its model automatically. You do not need Python, a GPU, an API key or a separate standalone Armiger install.

## 3. Check it works

```bash
docker compose ps
```

Both services should eventually say `healthy`. On the computer running Docker, open **[Armiger status](http://localhost:8098/healthz)** and check for `"status": "ok"`.

Rats Search's API is available at `http://localhost:8095`. That is a software connection endpoint, not a search page to open in your browser. Armiger categorizes newly indexed items automatically; a fresh index may initially be empty.

## Everyday use

Run these from the same folder:

| Task | Command |
| --- | --- |
| See Armiger's logs | `docker compose logs --tail=50 armiger-ultra` |
| See Rats Search's logs | `docker compose logs --tail=50 rats-search` |
| Stop | `docker compose down` |
| Start again | `docker compose up -d` |
| Update | `git pull --ff-only`, then `docker compose up -d --build` |

Stopping keeps your index and downloaded model. Do not add `-v` unless you intend to erase this example's saved data. If the build runs out of memory, increase Docker's memory allocation and try again. Change ports in `.env` if they are already in use. On a remote server, use SSH port forwarding to reach the local status/API ports. Automatic removal of flagged items is off.
