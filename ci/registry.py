"""Small anonymous GHCR reader shared by release checks and promotion."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

ACCEPT = ", ".join(("application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json"))


def fetch(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=45) as response:
        return response.read()


class Registry:
    def __init__(self):
        self.tokens = {}

    def headers(self, repo):
        if repo not in self.tokens:
            query = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{repo}:pull"})
            self.tokens[repo] = json.loads(fetch("https://ghcr.io/token?" + query))["token"]
        return {"Authorization": f"Bearer {self.tokens[repo]}", "Accept": ACCEPT}

    def manifest(self, repo, ref, *, missing_ok=False):
        try:
            raw = fetch(f"https://ghcr.io/v2/{repo}/manifests/{ref}", self.headers(repo))
        except urllib.error.HTTPError as error:
            if missing_ok and error.code == 404:
                return None
            raise
        return json.loads(raw), "sha256:" + hashlib.sha256(raw).hexdigest()

    def native(self, repo, ref, arch):
        manifest, digest = self.manifest(repo, ref)
        if "manifests" in manifest:
            candidates = [m for m in manifest["manifests"] if m.get("platform") == {"architecture": arch, "os": "linux"}
                          or (m.get("platform", {}).get("architecture") == arch
                              and m.get("platform", {}).get("os") == "linux")]
            assert len(candidates) == 1, (repo, ref, arch, candidates)
            manifest, digest = self.manifest(repo, candidates[0]["digest"])
        config_ref = manifest["config"]["digest"]
        config_raw = fetch(f"https://ghcr.io/v2/{repo}/blobs/{config_ref}", self.headers(repo))
        assert "sha256:" + hashlib.sha256(config_raw).hexdigest() == config_ref
        config = json.loads(config_raw)
        assert config["architecture"] == arch and config["os"] == "linux", config
        return {"manifest_digest": digest, "config_digest": config_ref,
                "config": config, "manifest": manifest}
