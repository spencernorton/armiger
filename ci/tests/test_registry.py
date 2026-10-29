"""Source provenance must hold in every native image, in either index format."""
from copy import deepcopy
import unittest

from ci.registry import Registry


class FixtureRegistry(Registry):
    def __init__(self, media_type):
        self.index = {"mediaType": media_type, "manifests": [
            {"platform": {"os": "linux", "architecture": arch}} for arch in ("amd64", "arm64")]}
        self.images = {arch: {"config": {"config": {"Labels": {
            "org.opencontainers.image.revision": "a" * 40,
            "org.opencontainers.image.version": "3.4.3"}}}} for arch in ("amd64", "arm64")}

    def manifest(self, repo, ref):
        return self.index, "sha256:" + "1" * 64

    def native(self, repo, ref, arch):
        return self.images[arch]


class RegistryIdentityTests(unittest.TestCase):
    def test_oci_and_docker_indexes_bind_every_native_source(self):
        for media in ("application/vnd.oci.image.index.v1+json",
                      "application/vnd.docker.distribution.manifest.list.v2+json"):
            with self.subTest(media=media):
                registry = FixtureRegistry(media)
                self.assertEqual(registry.release_identity("test", "3.4.3", ("amd64", "arm64")),
                                 ("a" * 40, "sha256:" + "1" * 64))

    def test_missing_platform_wrong_source_or_version_is_rejected(self):
        original = FixtureRegistry("application/vnd.docker.distribution.manifest.list.v2+json")
        cases = []
        missing = deepcopy(original)
        missing.index["manifests"].pop()
        cases.append(missing)
        for key, value in (("org.opencontainers.image.revision", "b" * 40),
                           ("org.opencontainers.image.revision", "main"),
                           ("org.opencontainers.image.version", "3.4.2")):
            altered = deepcopy(original)
            altered.images["arm64"]["config"]["config"]["Labels"][key] = value
            cases.append(altered)
        missing = deepcopy(original)
        del missing.images["arm64"]["config"]["config"]["Labels"]["org.opencontainers.image.revision"]
        cases.append(missing)
        misleading = deepcopy(original)
        misleading.index["annotations"] = {"org.opencontainers.image.revision": "b" * 40}
        cases.append(misleading)
        for registry in cases:
            with self.subTest(registry=registry.index), self.assertRaises((AssertionError, KeyError)):
                registry.release_identity("test", "3.4.3", ("amd64", "arm64"))


if __name__ == "__main__":
    unittest.main()
