"""Promote only the native manifests proved by this run's acceptance receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from ci.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {("spencernorton/armiger", "amd64"), ("spencernorton/armiger", "arm64"),
            ("spencernorton/armiger-rats-search", "amd64")}


def check_receipts(receipts, version, source):
    actual = [(r["repository"], r["architecture"]) for r in receipts]
    assert len(actual) == len(set(actual)) and set(actual) == EXPECTED, f"Missing or duplicate native acceptance: {actual}"
    for receipt in receipts:
        assert receipt["version"] == version and receipt["source"] == source, "Receipt is from a different revision/version"
        for key in ("manifest_digest", "config_digest", "tested_image_id"):
            assert receipt[key].startswith("sha256:") and len(receipt[key]) == 71, (key, receipt[key])


def normalized_compose(model):
    model.pop("name", None)
    for kind in ("volumes", "networks"):
        for value in model.get(kind, {}).values():
            value.pop("name", None)
    return model


def command(*args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()


def check_published(registry, version, source):
    releases = {}
    for repo in sorted({repo for repo, _ in EXPECTED}):
        actual_source, digest = registry.release_identity(repo, version, [arch for name, arch in EXPECTED if name == repo])
        assert actual_source == source, "Published source differs from acceptance"
        releases[repo] = digest
    assert registry.manifest("spencernorton/armiger-ultra", version)[1] == releases["spencernorton/armiger"]
    for setup in ("bitmagnet", "rats-search"):
        expected = normalized_compose(json.loads(command("docker", "compose", "-f", ROOT / f"examples/{setup}/docker-compose.yml",
                                                         "config", "--format", "json")))
        ref = f"ghcr.io/spencernorton/armiger-{setup}-stack:{version}"
        actual = normalized_compose(json.loads(command("docker", "compose", "-f", "oci://" + ref, "config", "--format", "json")))
        assert actual == expected, "Published stack is incomplete or differs from the supported setup"
        releases[f"spencernorton/armiger-{setup}-stack"] = registry.manifest(f"spencernorton/armiger-{setup}-stack", version)[1]
    return releases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--verify-only", action="store_true", help="Read-only completeness check for an unchanged existing release")
    args = parser.parse_args()
    registry = Registry()
    if args.verify_only:
        releases = check_published(registry, args.version, args.source)
        args.evidence.mkdir(parents=True, exist_ok=True)
        (args.evidence / "published.json").write_text(json.dumps(releases, indent=2))
        print(json.dumps(releases, indent=2))
        return
    receipts = [r for path in args.evidence.glob("candidates-*.json") for r in json.loads(path.read_text())]
    check_receipts(receipts, args.version, args.source)
    # Check every candidate before moving any user-facing tag. Promote by digest, never by mutable candidate tag.
    for receipt in receipts:
        native = registry.native(receipt["repository"], receipt["manifest_digest"], receipt["architecture"])
        assert native["config_digest"] == receipt["config_digest"], "Candidate changed after acceptance"
        labels = native["config"]["config"]["Labels"]
        assert labels["org.opencontainers.image.revision"] == args.source
        assert labels["org.opencontainers.image.version"] == args.version
    releases = {}
    for repo in sorted({r["repository"] for r in receipts}):
        existing = registry.manifest(repo, args.version, missing_ok=True)
        if existing:
            actual_source, _ = registry.release_identity(repo, args.version, [arch for name, arch in EXPECTED if name == repo])
            assert actual_source == args.source, "Refusing to replace an existing release"
            for receipt in [r for r in receipts if r["repository"] == repo]:
                actual = registry.native(repo, args.version, receipt["architecture"])
                assert actual["manifest_digest"] == receipt["manifest_digest"], "Rerun produced a different release payload; choose a new version"
            releases[repo] = existing[1]
            continue
        tags = [f"ghcr.io/{repo}:{args.version}"]
        sources = [f"ghcr.io/{repo}@{r['manifest_digest']}" for r in receipts if r["repository"] == repo]
        command("docker", "buildx", "imagetools", "create", "--annotation",
                f"index:org.opencontainers.image.revision={args.source}",
                *[part for tag in tags for part in ("--tag", tag)], *sources)
        actual_source, digest = registry.release_identity(repo, args.version, [arch for name, arch in EXPECTED if name == repo])
        assert actual_source == args.source, "Published source differs from acceptance"
        for receipt in [r for r in receipts if r["repository"] == repo]:
            actual = registry.native(repo, args.version, receipt["architecture"])
            assert actual["manifest_digest"] == receipt["manifest_digest"]
            assert actual["config_digest"] == receipt["config_digest"]
        for tag in tags:
            target_repo, target_tag = tag.removeprefix("ghcr.io/").split(":")
            assert registry.manifest(target_repo, target_tag)[1] == digest, f"Alias differs: {tag}"
        releases[repo] = digest
    for setup in ("bitmagnet", "rats-search"):
        model = normalized_compose(json.loads(command("docker", "compose", "-f", ROOT / f"examples/{setup}/docker-compose.yml",
                                                      "config", "--format", "json")))
        for service in model["services"].values():
            if service.get("image", "").startswith("ghcr.io/spencernorton/armiger"):
                assert service["image"].endswith(":" + args.version), service
        path = args.evidence / f"release-{setup}.json"
        path.write_text(json.dumps(model))
        repo = f"spencernorton/armiger-{setup}-stack"
        ref = f"ghcr.io/{repo}:{args.version}"
        if registry.manifest(repo, args.version, missing_ok=True) is None:
            command("docker", "compose", "-f", path, "publish", "--yes", ref)
        published = normalized_compose(json.loads(command("docker", "compose", "-f", "oci://" + ref, "config", "--format", "json")))
        assert published == model, "Published OCI stack differs from the documented/tested commands and images"
        releases[repo] = registry.manifest(repo, args.version)[1]
    # Advance mutable entrypoints only after both immutable images and OCI
    # stacks have passed readback. The versioned Ultra alias is immutable too.
    digest = releases["spencernorton/armiger"]
    ultra = registry.manifest("spencernorton/armiger-ultra", args.version, missing_ok=True)
    if ultra is not None:
        assert ultra[1] == digest, "Refusing to replace an existing Ultra release"
    aliases = ["ghcr.io/spencernorton/armiger:latest", "ghcr.io/spencernorton/armiger-ultra:latest",
               "ghcr.io/spencernorton/armiger-ultra:3.3.0"]
    if ultra is None:
        aliases.append(f"ghcr.io/spencernorton/armiger-ultra:{args.version}")
    command("docker", "buildx", "imagetools", "create", *[part for tag in aliases for part in ("--tag", tag)],
            f"ghcr.io/spencernorton/armiger@{digest}")
    for tag in aliases:
        repo, version = tag.removeprefix("ghcr.io/").split(":")
        assert registry.manifest(repo, version)[1] == digest, f"Alias differs from accepted release: {tag}"
    check_published(registry, args.version, args.source)
    (args.evidence / "published.json").write_text(json.dumps(releases, indent=2))
    print(json.dumps(releases, indent=2))


if __name__ == "__main__":
    main()
