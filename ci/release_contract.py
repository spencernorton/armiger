"""Fail ambiguous releases and stale runnable quickstarts before expensive builds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

from ci.registry import Registry, fetch

ROOT = Path(__file__).resolve().parents[1]
IMAGE_REF = re.compile(r"ghcr\.io/spencernorton/(armiger(?:-ultra|-rats-search|-bitmagnet-stack|-rats-search-stack)?):(\d+\.\d+\.\d+)")
EXTERNAL_GUIDES = (
    "https://raw.githubusercontent.com/spencernorton/Armiger-Connectors/main/bitmagnet/README.md",
    "https://huggingface.co/NorviTech/armiger/raw/main/README.md",
    "https://huggingface.co/NorviTech/armiger-ultra/raw/main/README.md",
)


def image_affecting(path):
    return path.startswith("src/") or path in {
        "examples/Dockerfile", "examples/rats-search/Dockerfile", ".dockerignore",
        "pyproject.toml", "README.md", "LICENSE", "NOTICE",
    }


def release_needed(version, published, changed, *, same_source=False):
    if published is None:
        return True
    affected = sorted(p for p in changed if image_affecting(p))
    if affected:
        raise AssertionError(f"Software {version} is already published; bump the version for image-affecting changes: {affected}")
    # A retry at the owning commit must finish/verify publication, even when a
    # prior attempt created the version index before failing later in promotion.
    return same_source


def local_guides(root, version):
    paths = [root / "README.md", root / "docs/bitmagnet-configuration.md"]
    paths += list((root / "examples").rglob("*.md")) + list((root / "examples").rglob("*.yml"))
    references = 0
    for path in paths:
        for name, tag in IMAGE_REF.findall(path.read_text()):
            assert tag == version, f"{path}: {name}:{tag} disagrees with software {version}"
            references += 1
    assert references >= 8, "Missing supported quickstart image references"
    init = (root / "src/armiger/__init__.py").read_text()
    assert f'__version__ = "{version}"' in init, "Package and release versions disagree"
    assert f"Library **{version}**" in (root / "README.md").read_text(), "README software version is stale"
    return references


def external_guides(registry):
    checked = {}
    for url in EXTERNAL_GUIDES:
        content = fetch(url).decode()
        assert "spencernorton/armiger" in content, f"Missing canonical setup link in {url}"
        for name, tag in IMAGE_REF.findall(content):
            if name.endswith("-stack"):
                assert registry.manifest(f"spencernorton/{name}", tag), (url, name, tag)
                continue
            key = f"{name}:{tag}"
            if key not in checked:
                native = registry.native(f"spencernorton/{name}", tag, "amd64")
                labels = native["config"]["config"].get("Labels", {})
                actual = labels.get("org.opencontainers.image.version", "")
                assert re.fullmatch(r"\d+\.\d+\.\d+", actual), (url, key, labels)
                assert tuple(map(int, actual.split("."))) >= (3, 4, 1), f"{url} points at the known broken image {key} ({actual})"
                checked[key] = {"actual_version": actual, "digest": native["manifest_digest"]}
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer", action="store_true", help="Validate a connector candidate without requesting a release")
    parser.add_argument("--external-only", action="store_true", help="Check published guide references independently of native builds")
    args = parser.parse_args()
    if args.external_only:
        report = {"external_images": external_guides(Registry())}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), version
    references = local_guides(ROOT, version)
    registry = Registry()
    published = registry.manifest("spencernorton/armiger", version, missing_ok=True)
    changed, revision = [], ""
    if published is not None:
        revision = published[0].get("annotations", {}).get("org.opencontainers.image.revision", "")
        assert re.fullmatch(r"[a-f0-9]{40}", revision), "Existing release lacks verifiable source provenance"
        changed = subprocess.check_output(["git", "diff", "--no-renames", "--name-only", revision, "HEAD"], cwd=ROOT, text=True).splitlines()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    needed = release_needed(version, published, changed, same_source=revision == head)
    report = {"version": version, "release_needed": needed and not args.consumer,
              "local_image_references": references, "published_source": revision}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"version={version}\nrelease_needed={str(report['release_needed']).lower()}\n")
            output.write(f"published_source={revision}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
