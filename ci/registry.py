"""Small anonymous GHCR reader shared by release checks and promotion."""
from __future__ import annotations

import hashlib
import json
import re
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

    def release_identity(self, repo, version, architectures):
        """Bind a Docker manifest list or OCI index to every native image config."""
        index, digest = self.manifest(repo, version)
        assert index["mediaType"] in ("application/vnd.oci.image.index.v1+json",
                                      "application/vnd.docker.distribution.manifest.list.v2+json")
        platforms = {(x.get("platform", {}).get("os"), x.get("platform", {}).get("architecture"))
                     for x in index["manifests"]
                     if x.get("annotations", {}).get("vnd.docker.reference.type") != "attestation-manifest"}
        assert platforms == {("linux", arch) for arch in architectures}, platforms
        sources = set()
        for arch in architectures:
            labels = self.native(repo, version, arch)["config"]["config"]["Labels"]
            assert labels["org.opencontainers.image.version"] == version, "Native image version differs"
            source = labels["org.opencontainers.image.revision"]
            assert re.fullmatch(r"[a-f0-9]{40}", source), "Native image lacks source provenance"
            sources.add(source)
        assert len(sources) == 1, "Native images identify different source revisions"
        source = sources.pop()
        annotation = index.get("annotations", {}).get("org.opencontainers.image.revision")
        assert annotation in (None, source), "Index and native source provenance disagree"
        return source, digest
