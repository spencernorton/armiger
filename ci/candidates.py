"""Bind pushed candidate manifests to the local images exercised by acceptance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from ci.registry import Registry


# Docker API 1.48 includes these legacy container defaults in image inspect;
# newer APIs and OCI image configs can omit them without changing the image.
FALSE_DEFAULTS = frozenset({"AttachStdin", "AttachStdout", "AttachStderr", "Tty", "OpenStdin", "StdinOnce"})


def inspect(image):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]


def same_payload(local, remote):
    assert local["Architecture"] == remote["architecture"]
    assert local["Os"] == remote["os"] == "linux"
    assert local["RootFS"]["Layers"] == remote["rootfs"]["diff_ids"], "Pushed filesystem differs from tested image"
    left, right = local["Config"], remote["config"]
    for key in left.keys() | right.keys():
        a, b = left.get(key), right.get(key)
        if a != b:
            if key in FALSE_DEFAULTS and ((a is False and b is None) or (a is None and b is False)):
                continue
            assert a in (None, "", [], {}) and b in (None, "", [], {}), f"Pushed {key} differs from tested image"


def candidate_tag(arch):
    return f"ci-{os.environ['GITHUB_SHA']}-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-{arch}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    records = list(args.evidence.glob("runtime-*.json"))
    assert len(records) == 1, records
    runtime = json.loads(records[0].read_text())
    arch = runtime["architecture"]
    images = [("armiger", runtime["image"], runtime["image_id"])]
    if "rats_image_id" in runtime:
        images.append(("armiger-rats-search", runtime["rats_image"], runtime["rats_image_id"]))
    registry, receipt = Registry(), []
    for name, image, tested_id in images:
        local = inspect(image)
        assert local["Id"] == tested_id, "Local tag changed after acceptance"
        repo = f"spencernorton/{name}"
        tag = candidate_tag(arch)
        ref = f"ghcr.io/{repo}:{tag}"
        subprocess.run(["docker", "tag", image, ref], check=True)
        subprocess.run(["docker", "push", ref], check=True)
        native = registry.native(repo, tag, arch)
        same_payload(local, native["config"])
        labels = native["config"]["config"]["Labels"]
        assert labels["org.opencontainers.image.revision"] == args.source
        assert labels["org.opencontainers.image.version"] == args.version
        receipt.append({"repository": repo, "architecture": arch, "version": args.version,
                        "source": args.source, "tested_image_id": tested_id,
                        "manifest_digest": native["manifest_digest"], "config_digest": native["config_digest"]})
    (args.evidence / f"candidates-{arch}.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
