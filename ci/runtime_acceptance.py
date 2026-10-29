"""Exercise the documented containers with real models and disposable upstreams.

Run with a locally built Armiger image. Supplying --rats-image also qualifies the
actual x86 Rats/Manticore backend; the other native architecture checks its
connector's HTTP entrypoint. Every Compose project gets fresh, isolated volumes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ITEM = {"name": "Tears.of.Steel.2012.1080p.mkv", "size": 2400000000,
        "infoHash": "f" * 40, "files": [{"path": "Tears.of.Steel.2012.mkv", "size": 2400000000}]}
OFFLINE = {"ARMIGER_AUTO_DOWNLOAD": "false", "ARMIGER_DHT_AUTO_UPDATE": "false",
           "ARMIGER_DHT_SHARE_OPT_IN": "false", "OMP_NUM_THREADS": "2"}


def run(*args, **kwargs):
    try:
        return subprocess.run([str(x) for x in args], check=True, text=True,
                              capture_output=True, timeout=kwargs.pop("timeout", 120), **kwargs).stdout.strip()
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        raise


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def eventually(check, description, timeout=120):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = check()
            if last:
                return last
        except (urllib.error.URLError, TimeoutError, ConnectionError, subprocess.CalledProcessError) as error:
            last = str(error)
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {description}: {last}")


class Stack:
    def __init__(self, setup, image, evidence, rats_image=None, http_only=False):
        self.setup = setup
        self.project = f"armiger-ci-{setup}-{uuid.uuid4().hex[:10]}"
        self.evidence = evidence
        self.path = evidence / f"{setup}-compose.json"
        model = json.loads(run("docker", "compose", "-f", ROOT / f"examples/{setup}/docker-compose.yml",
                               "config", "--format", "json"))
        model.pop("name", None)
        for kind in ("volumes", "networks"):
            for resource in model.get(kind, {}).values():
                resource.pop("name", None)
        model["networks"] = {"default": {}}
        for service in model["services"].values():
            service["restart"] = "no"
            service["ports"] = [{"target": p["target"], "published": "0", "host_ip": "127.0.0.1",
                                 "protocol": p.get("protocol", "tcp")} for p in service.get("ports", [])
                                if p.get("protocol", "tcp") == "tcp"]
        app = model["services"]["armiger-ultra"]
        app["image"] = image
        app.setdefault("environment", {}).update(OFFLINE)
        if setup == "bitmagnet":
            # Use the stock HTTP/queue workers; synthetic fixtures need no public DHT/TMDB traffic.
            model["services"]["bitmagnet"]["command"] = ["worker", "run", "--keys=http_server", "--keys=queue_server"]
            model["services"]["bitmagnet"].setdefault("environment", {})["TMDB_ENABLED"] = "false"
            app["environment"]["ARMIGER_PRUNE_MALWARE"] = "true"
        if setup == "rats-search":
            if http_only:
                model["services"].pop("rats-search")
                app.pop("network_mode", None)
                app.pop("depends_on", None)
                app["ports"] = [{"target": 8097, "published": "0", "host_ip": "127.0.0.1"}]
                app["environment"]["ARMIGER_CONNECTOR_MODE"] = "http"
            else:
                model["services"]["rats-search"]["image"] = rats_image
                model["services"]["rats-search"].setdefault("environment", {})["RATS_INDEXER"] = "false"
        self.path.write_text(json.dumps(model, indent=2))
        self.model = model

    def compose(self, *args, **kwargs):
        return run("docker", "compose", "-p", self.project, "-f", self.path, *args, **kwargs)

    def url(self, port, service="armiger-ultra"):
        address = self.compose("port", service, str(port)).splitlines()[0]
        return "http://" + address

    def sql(self, sql):
        return self.compose("exec", "-T", "postgres", "psql", "-X", "-qAt", "-v", "ON_ERROR_STOP=1",
                            "-U", "bitmagnet", "-d", "bitmagnet", input=sql)

    def image_is_running(self, expected, service="armiger-ultra"):
        container = self.compose("ps", "--all", "-q", service)
        details = json.loads(run("docker", "inspect", container))[0]
        assert details["State"]["Running"], f"{service} exited: {details['State']}"
        image_id = json.loads(run("docker", "image", "inspect", expected))[0]["Id"]
        assert details["Image"] == image_id, (details["Image"], image_id)
        if service == "armiger-ultra":
            uid = self.compose("exec", "-T", service, "python", "-c", "import os; print(os.geteuid())")
            assert uid == "10001", uid
        return image_id

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            (self.evidence / f"{self.setup}.log").write_text(self.compose("logs", "--no-color", timeout=30))
            (self.evidence / f"{self.setup}-ps.json").write_text(self.compose("ps", "--all", "--format", "json"))
        finally:
            self.compose("down", "--volumes", "--remove-orphans", "--timeout", "5", timeout=60)


def graphql(url, query, variables=None):
    response = request(url + "/graphql", {"query": query, "variables": variables or {}})
    assert not response.get("errors"), response
    return response["data"]


SEARCH = """query($input: TorrentContentSearchQueryInput!) {
 torrentContent {search(input:$input) {items {infoHash torrent {tagNames}}}}} """


def rows(url):
    found = graphql(url, SEARCH, {"input": {"limit": 1000}})
    return {x["infoHash"]: x["torrent"]["tagNames"] or [] for x in found["torrentContent"]["search"]["items"]}


def seed(stack, start, stop, *, historical=False):
    # Only fabricate metadata in a fresh, local upstream database. No .torrent or media downloads.
    base = "2026-01-01 00:00:00+00" if historical else "2026-02-01 00:00:00+00"
    stack.sql(f"""
    begin;
    insert into torrents(info_hash,name,size,private,files_status,files_count,created_at,updated_at)
      select decode(lpad(to_hex(i),40,'0'),'hex'),'Tears.of.Steel.2012.1080p.mkv',2400000000,false,
        'multi',1,'{base}'::timestamptz+i*interval '1 second','{base}'::timestamptz+i*interval '1 second'
      from generate_series({start},{stop}) i;
    insert into torrent_contents(info_hash,size,files_count,seeders,leechers,created_at,updated_at,tsv)
      select info_hash,size,files_count,20,2,created_at,updated_at,
        to_tsvector('simple','Tears of Steel 2012 1080p') from torrents
      where info_hash between decode(lpad(to_hex({start}),40,'0'),'hex') and decode(lpad(to_hex({stop}),40,'0'),'hex');
    insert into torrent_files(info_hash,index,path,size,created_at,updated_at)
      select info_hash,0,case when encode(info_hash,'hex') in ('{325:040x}','{340:040x}')
        then 'Tears.of.Steel.2012.mkv.exe' else 'Tears.of.Steel.2012.mkv' end,size,created_at,updated_at from torrents
      where info_hash between decode(lpad(to_hex({start}),40,'0'),'hex') and decode(lpad(to_hex({stop}),40,'0'),'hex');
    commit;
    """)


def classify(stack, connector=None):
    port = 8097 if stack.setup == "rats-search" else 8096
    service = "rats-search" if stack.setup == "rats-search" and "rats-search" in stack.model["services"] else "armiger-ultra"
    url = stack.url(port, service)
    def healthy():
        container = stack.compose("ps", "--all", "-q", "armiger-ultra")
        state = json.loads(run("docker", "inspect", container))[0]["State"]
        assert state["Running"], f"{stack.setup} exited: {state}"
        return request(url + "/healthz")
    health = eventually(healthy, f"{stack.setup} health")
    assert health["status"] == "ok", health
    if connector:
        result = request(url + f"/v1/{connector}/classify", {"items": [ITEM]})
        assert result["total"] == 1, result
        action = result["actions"][0]
        assert action.get("armigerCategory", action.get("category")) == "movie", result
    else:
        result = request(url + "/v1/classify", ITEM)
        assert result["category"] == "movie", result
    return url, {"health": health, "classification": result}


def rats_sql(stack, sql):
    code = """import json,sys
sys.path.insert(0,'/opt/connectors/rats-search')
from rats_search_connector import execute_manticore_mysql41
print(json.dumps(execute_manticore_mysql41('127.0.0.1',9306,sys.argv[1])))
"""
    return json.loads(stack.compose("exec", "-T", "armiger-ultra", "python", "-c", code, sql))


def bitmagnet(image, evidence):
    with Stack("bitmagnet", image, evidence) as stack:
        stack.compose("up", "-d", "postgres", "bitmagnet")
        url = stack.url(3333, "bitmagnet")
        eventually(lambda: graphql(url, SEARCH, {"input": {"limit": 1}}), "stock Bitmagnet migrations/API")
        assert stack.sql("select count(*) from torrents;") == "0", "Acceptance requires an empty database"
        seed(stack, 1, 350)
        seed(stack, 1001, 1050, historical=True)
        graphql(url, "mutation($h:[Hash20!]!,$t:[String!]!){torrent{putTags(infoHashes:$h,tagNames:$t)}}",
                {"h": [f"{i:040x}" for i in range(1001, 1051)], "t": ["armiger-movie"]})
        # Put real legacy tags into the upstream API before the default command starts.
        legacy = {"armiger-viability-dea-c24281f7": "armiger-viability-dead",
                  "armiger-viability-deg-7c170f8b": "armiger-viability-degraded",
                  "armiger-viability-sus-cdc9d2c6": "armiger-viability-stalled"}
        for i, tag in enumerate(legacy, 1001):
            graphql(url, "mutation($h:[Hash20!]!,$t:[String!]!){torrent{putTags(infoHashes:$h,tagNames:$t)}}",
                    {"h": [f"{i:040x}"], "t": [tag]})
        stack.compose("up", "-d", "armiger-ultra")
        stack.image_is_running(image)
        _, api = classify(stack, "bitmagnet")
        expected = {f"{i:040x}" for i in range(1, 351)} - {f"{i:040x}" for i in (325, 340)}
        def complete():
            state = rows(url)
            return state if all("armiger-movie" in state.get(h, []) for h in expected) else None
        drained = eventually(complete, "all 350 burst records processed across 100-item polls", timeout=150)
        assert all(f"{i:040x}" not in drained for i in (325, 340)), "Unsafe fixture rows must be deleted"
        for i, (old, new) in enumerate(legacy.items(), 1001):
            assert new in drained[f"{i:040x}"] and old not in drained[f"{i:040x}"], drained[f"{i:040x}"]
        expected_tags = {"alive_viable": "armiger-viability-alive-viable",
                         "degraded_viable": "armiger-viability-degraded",
                         "suspect_stalled": "armiger-viability-stalled",
                         "dead_known_zero": "armiger-viability-dead", "dead": "armiger-viability-dead",
                         "unknown_cold": "armiger-viability-unknown-cold"}
        # Exercise the bundled production serializer, then let stock GraphQL/PG validate
        # and persist every tier. These are protocol contract cases, not fake predictions.
        code = """import json,sys
sys.path.insert(0,'/opt/connectors/bitmagnet')
from bitmagnet_connector import BitmagnetConnector, BITMAGNET_VIABILITY_TAG_MAP
expected=json.loads(sys.argv[1])
assert BITMAGNET_VIABILITY_TAG_MAP==expected
print(json.dumps([BitmagnetConnector.build_put_tags_mutation([f'{i:040x}'],['armiger-viability-'+tier])
 for i,tier in enumerate(expected,1004)]))
"""
        mutations = json.loads(stack.compose("exec", "-T", "armiger-ultra", "python", "-c", code, json.dumps(expected_tags)))
        for mutation in mutations:
            graphql(url, mutation["query"], mutation["variables"])
        canonical = rows(url)
        for i, tag in enumerate(expected_tags.values(), 1004):
            actual = {t for t in canonical[f"{i:040x}"] if t.startswith("armiger-viability-")}
            assert actual == {tag}, (i, actual, tag)
        # A later flush must be noticed after the prior wave reaches its tagged frontier.
        seed(stack, 351, 400)
        expected.update(f"{i:040x}" for i in range(351, 401))
        drained = eventually(complete, "new arrivals after an exhausted crawl wave", timeout=60)
        for tags in drained.values():
            assert all(len(t) <= 30 and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", t) for t in tags), tags
            assert not set(tags).intersection(legacy), tags
        # The proxy must actually reach stock upstream and produce parseable capabilities.
        with urllib.request.urlopen(stack.url(8097) + "/torznab?t=caps", timeout=15) as response:
            assert ET.fromstring(response.read()).tag == "caps"
        with urllib.request.urlopen(url + "/torznab?t=search&q=Tears&limit=10", timeout=30) as response:
            upstream_xml = response.read()
            (evidence / "bitmagnet-upstream.xml").write_bytes(upstream_xml)
            assert ET.fromstring(upstream_xml).findall("./channel/item"), "Stock search cannot find seeded metadata"
        with urllib.request.urlopen(stack.url(8097) + "/torznab?t=search&q=Tears&limit=10", timeout=30) as response:
            filtered_xml = response.read()
            (evidence / "bitmagnet-filtered.xml").write_bytes(filtered_xml)
            items = ET.fromstring(filtered_xml).findall("./channel/item")
            assert items, "Real Torznab search returned no admitted fixtures"
            assert items[0].findtext("category") == "Movie", ET.tostring(items[0], encoding="unicode")
            assert int(response.headers["X-Armiger-Admitted"]) > 0
        # Seed a sourced Unknown TV show (to verify POST /import contentType hint writeback) and a sourceless
        # Unknown Movie (to verify Torznab t=movie&cat=2000 fallback rescue + 8000->2000 category promotion).
        stack.sql(f"""
        begin;
        insert into torrent_sources(key,name,created_at,updated_at)
          values ('dht','DHT',now(),now()) on conflict do nothing;
        insert into torrents(info_hash,name,size,private,files_status,files_count,created_at,updated_at) values
          (decode('{2001:040x}','hex'),'Caminandes.Season.01.Complete.1080p.WEBRip',3200000000,false,'multi',1,'2026-03-01 00:00:01+00','2026-03-01 00:00:01+00'),
          (decode('{2002:040x}','hex'),'Sintel.Open.Movie.1080p.BluRay',4100000000,false,'multi',1,'2026-03-01 00:00:02+00','2026-03-01 00:00:02+00');
        insert into torrents_torrent_sources(source,info_hash,seeders,leechers,created_at,updated_at) values
          ('dht',decode('{2001:040x}','hex'),18,2,'2026-03-01 00:00:01+00','2026-03-01 00:00:01+00');
        insert into torrent_contents(info_hash,size,files_count,seeders,leechers,created_at,updated_at,tsv) values
          (decode('{2001:040x}','hex'),3200000000,1,18,2,'2026-03-01 00:00:01+00','2026-03-01 00:00:01+00',to_tsvector('simple','Caminandes Season 01 Complete 1080p WEBRip')),
          (decode('{2002:040x}','hex'),4100000000,1,25,3,'2026-03-01 00:00:02+00','2026-03-01 00:00:02+00',to_tsvector('simple','Sintel Open Movie 1080p BluRay'));
        insert into torrent_files(info_hash,index,path,size,created_at,updated_at) values
          (decode('{2001:040x}','hex'),0,'Caminandes.S01E01.1080p.mkv',3200000000,'2026-03-01 00:00:01+00','2026-03-01 00:00:01+00'),
          (decode('{2002:040x}','hex'),0,'Sintel.Open.Movie.1080p.mkv',4100000000,'2026-03-01 00:00:02+00','2026-03-01 00:00:02+00');
        commit;
        """)
        eventually(
            lambda: stack.sql(f"select content_type from torrent_contents where info_hash=decode('{2001:040x}','hex');") == "tv_show",
            "Bitmagnet /import contentType hint writeback for sourced Unknown TV show",
            timeout=60,
        )
        with urllib.request.urlopen(stack.url(8097) + "/torznab?t=tvsearch&cat=5000&q=Caminandes&limit=10", timeout=30) as response:
            tv_items = ET.fromstring(response.read()).findall("./channel/item")
            assert any(it.findtext("category") in ("tv_show", "TV show") for it in tv_items), "Torznab tvsearch did not return tv_show item"
        with urllib.request.urlopen(stack.url(8097) + "/torznab?t=movie&cat=2000&q=Sintel&limit=10", timeout=30) as response:
            movie_items = ET.fromstring(response.read()).findall("./channel/item")
            assert any(it.findtext("category") == "Movie" for it in movie_items), "Torznab movie rescue did not promote Unknown movie to 2000"
        return {"burst": 350, "batch_size": 100, "deleted": 2, "new_arrivals": 50,
                "processed": len(expected), "legacy_repaired": 3, "canonical_tiers": len(expected_tags),
                "content_type_hint_verified": True, "torznab_category_rescue_verified": True, "api": api}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--rats-image")
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    image = json.loads(run("docker", "image", "inspect", args.image))[0]
    result = {"image": args.image, "image_id": image["Id"], "architecture": image["Architecture"],
              "source": image["Config"].get("Labels", {}).get("org.opencontainers.image.revision")}
    with Stack("standalone", args.image, args.evidence) as stack:
        stack.compose("up", "-d")
        stack.image_is_running(args.image)
        _, result["standalone"] = classify(stack)
    result["bitmagnet"] = bitmagnet(args.image, args.evidence)
    with Stack("rats-search", args.image, args.evidence, args.rats_image, http_only=not args.rats_image) as stack:
        stack.compose("up", "-d", timeout=240)
        stack.image_is_running(args.image)
        rats_url, result["rats-search"] = classify(stack, "rats-search")
        result["rats-search"]["upstream"] = "real Rats/Manticore" if args.rats_image else "HTTP entrypoint only; full stack required on amd64"
        if args.rats_image:
            stack.image_is_running(args.rats_image, "rats-search")
            backend = json.loads(run("docker", "image", "inspect", args.rats_image))[0]
            result["rats_image"] = args.rats_image
            result["rats_image_id"] = backend["Id"]
            # Reach the actual REST API as well as the connector's real-model HTTP API.
            recent = request(stack.url(8095, "rats-search") + "/api/search.recent?limit=1")
            assert recent.get("success") is True and isinstance(recent.get("data"), list), recent
            result["rats-search"]["rest"] = {"recent_items": len(recent["data"])}
            result["rats-search"]["manticore"] = eventually(
                lambda: rats_sql(stack, "SELECT COUNT(*) FROM torrents"),
                "real Manticore SQL handshake and torrents table")
            row_id = (1 << 63) - 1  # Production rowIdFromHash for the fixture's all-f hash.
            rats_sql(stack, f"INSERT INTO torrents(id,hash,name,size,files,added,contentType,contentCategory) VALUES "
                     f"({row_id},'{ITEM['infoHash']}','Tears.of.Steel.2012.1080p.mkv',2400000000,1,1,2,3)")
            applied = request(rats_url + "/v1/rats-search/classify", {"items": [ITEM], "applyChanges": True})
            assert applied["applied"] == {"removed": 0, "updated": 1, "collisionsSkipped": 0}, applied
            persisted = rats_sql(stack, f"SELECT contentType,contentCategory FROM torrents WHERE id={row_id}")[0]
            assert {k.lower(): int(v) for k, v in persisted.items()} == {"contenttype": 1, "contentcategory": 1}, persisted
            # Distinct hashes sharing the first63bits must never overwrite each other's metadata.
            collision_hash = "f" * 16 + "e" * 24
            rats_sql(stack, f"REPLACE INTO torrents(id,hash,name,size,files,added,contentType,contentCategory) VALUES "
                     f"({row_id},'{collision_hash}','Tears.of.Steel.2012.1080p.mkv',2400000000,1,1,2,3)")
            collision = request(rats_url + "/v1/rats-search/classify", {"items": [ITEM], "applyChanges": True})
            assert collision["applied"] == {"removed": 0, "updated": 0, "collisionsSkipped": 1}, collision
            untouched = rats_sql(stack, f"SELECT contentType,contentCategory FROM torrents WHERE id={row_id}")[0]
            assert {k.lower(): int(v) for k, v in untouched.items()} == {"contenttype": 2, "contentcategory": 3}, untouched
            result["rats-search"]["mutation_readback"] = persisted
            result["rats-search"]["collision_preserved"] = untouched
    (args.evidence / f"runtime-{image['Architecture']}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"image_id": result["image_id"], "architecture": result["architecture"],
                      "checks": ["standalone", "bitmagnet", "rats-search"],
                      "bitmagnet": {k: v for k, v in result["bitmagnet"].items() if k != "api"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
