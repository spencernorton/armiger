"""Regression tests for ambiguous green releases and misleading quickstarts."""
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from ci.registry import Registry
from ci.release_contract import local_guides, release_needed


class ReleaseContractTests(unittest.TestCase):
    def test_unversioned_runtime_or_packaging_change_cannot_report_success(self):
        for path in ("src/armiger/server.py", "examples/Dockerfile", "examples/rats-search/Dockerfile",
                     "pyproject.toml", ".dockerignore", "README.md"):
            with self.subTest(path=path), self.assertRaisesRegex(AssertionError, "bump the version"):
                release_needed("3.4.0", ({}, "sha256:existing"), [path])

    def test_docs_and_test_changes_do_not_republish_existing_version(self):
        self.assertFalse(release_needed("3.4.1", ({}, "sha256:existing"),
                                        ["docs/evaluation.md", "tests/test_server.py", ".github/workflows/ci.yml"]))

    def test_unpublished_version_requests_a_release(self):
        self.assertTrue(release_needed("3.4.2", None, ["examples/Dockerfile"]))

    def test_same_source_retry_cannot_skip_partial_publication(self):
        self.assertTrue(release_needed("3.4.2", ({}, "sha256:existing"), [], same_source=True))

    def test_registry_failures_are_not_treated_as_unpublished_versions(self):
        registry = Registry()
        registry.tokens["spencernorton/armiger"] = "anonymous-test-token"
        for status in (401, 403, 429, 500, 503):
            error = HTTPError("https://ghcr.io/test", status, "unavailable", {}, io.BytesIO())
            with self.subTest(status=status), patch("ci.registry.fetch", side_effect=error), self.assertRaises(HTTPError):
                registry.manifest("spencernorton/armiger", "3.4.2", missing_ok=True)
            error.close()
        missing = HTTPError("https://ghcr.io/test", 404, "missing", {}, io.BytesIO())
        with patch("ci.registry.fetch", side_effect=missing):
            self.assertIsNone(registry.manifest("spencernorton/armiger", "3.4.2", missing_ok=True))
        missing.close()

    def test_stale_runnable_example_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs").mkdir()
            (root / "docs/bitmagnet-configuration.md").write_text("ghcr.io/spencernorton/armiger:3.3.4")
            (root / "README.md").write_text("Library **3.4.2**")
            with self.assertRaisesRegex(AssertionError, "disagrees"):
                local_guides(root, "3.4.2")


if __name__ == "__main__":
    unittest.main()
