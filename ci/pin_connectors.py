"""Test a Connectors PR in the same image/harness used to qualify Armiger releases."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from ci.registry import fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository)
    assert re.fullmatch(r"[a-f0-9]{40}", args.revision)
    path = Path(__file__).resolve().parents[1] / "examples/Dockerfile"
    dockerfile = path.read_text()
    for relative in ("bitmagnet/bitmagnet_connector.py", "rats-search/rats_search_connector.py"):
        url = f"https://raw.githubusercontent.com/{args.repository}/{args.revision}/{relative}"
        checksum = hashlib.sha256(fetch(url)).hexdigest()
        destination = "/opt/connectors/" + relative
        pattern = r"(?m)^ADD --chmod=0444 --checksum=sha256:[a-f0-9]{64} https://raw\.githubusercontent\.com/\S+ " + re.escape(destination) + r"$"
        dockerfile, count = re.subn(pattern, f"ADD --chmod=0444 --checksum=sha256:{checksum} {url} {destination}", dockerfile)
        assert count == 1, f"Missing unique pinned connector ADD: {relative}"
    path.write_text(dockerfile)
    cfg_doc = Path(__file__).resolve().parents[1] / "docs/bitmagnet-configuration.md"
    if cfg_doc.is_file():
        doc_text = cfg_doc.read_text()
        doc_text = re.sub(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/blob/[a-f0-9]{40}/bitmagnet/bitmagnet_connector\.py",
            f"https://github.com/{args.repository}/blob/{args.revision}/bitmagnet/bitmagnet_connector.py",
            doc_text,
        )
        cfg_doc.write_text(doc_text)
    print(json.dumps({"connector_repository": args.repository, "connector_revision": args.revision}))


if __name__ == "__main__":
    main()
