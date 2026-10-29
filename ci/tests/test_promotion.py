"""A successful job cannot promote receipts from missing/older/different artifacts."""
from copy import deepcopy
import unittest

from ci.candidates import FALSE_DEFAULTS, same_payload
from ci.promote import check_receipts, EXPECTED


class PromotionTests(unittest.TestCase):
    def receipts(self):
        return [{"repository": repo, "architecture": arch, "version": "3.4.2", "source": "a" * 40,
                 "manifest_digest": "sha256:" + "1" * 64, "config_digest": "sha256:" + "2" * 64,
                 "tested_image_id": "sha256:" + "3" * 64} for repo, arch in EXPECTED]

    def test_all_native_receipts_required_and_bound_to_source(self):
        receipts = self.receipts()
        check_receipts(receipts, "3.4.2", "a" * 40)
        cases = [receipts[:-1], receipts + receipts[:1]]
        for key, value in (("source", "b" * 40), ("version", "3.4.1")):
            altered = deepcopy(receipts)
            altered[0][key] = value
            cases.append(altered)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(AssertionError):
                check_receipts(case, "3.4.2", "a" * 40)

    def test_changed_filesystem_command_user_or_environment_is_rejected(self):
        local = {"Architecture": "amd64", "Os": "linux", "RootFS": {"Layers": ["sha256:original"]},
                 "Config": {"User": "10001:10001", "Cmd": ["python", "-m", "armiger", "serve"], "Env": ["MODE=offline"]}}
        remote = {"architecture": "amd64", "os": "linux", "rootfs": {"diff_ids": ["sha256:original"]},
                  "config": deepcopy(local["Config"])}
        same_payload(local, remote)
        altered = deepcopy(remote)
        altered["rootfs"]["diff_ids"] = ["sha256:different"]
        cases = [altered]
        for key, value in (("User", "0"), ("Cmd", ["sleep", "infinity"]), ("Env", ["MODE=online"])):
            altered = deepcopy(remote)
            altered["config"][key] = value
            cases.append(altered)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(AssertionError):
                same_payload(local, case)

    def test_legacy_api_false_defaults_match_omitted_oci_fields(self):
        local = {"Architecture": "arm64", "Os": "linux", "RootFS": {"Layers": ["sha256:original"]},
                 "Config": {key: False for key in FALSE_DEFAULTS}}
        remote = {"architecture": "arm64", "os": "linux", "rootfs": {"diff_ids": ["sha256:original"]},
                  "config": {}}
        same_payload(local, remote)
        local["Config"], remote["config"] = remote["config"], local["Config"]
        same_payload(local, remote)
        for key, value in [(key, True) for key in FALSE_DEFAULTS] + [
            ("StopTimeout", 0), ("UnknownFlag", False), ("OpenStdin", 0)
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                altered = deepcopy(remote)
                altered["config"] = {key: value}
                same_payload(local, altered)


if __name__ == "__main__":
    unittest.main()
