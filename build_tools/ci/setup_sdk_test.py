# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from build_tools.ci import setup_sdk


class SetupSDKTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "build_tools/ci").mkdir(parents=True)
        self.output = self.root / "output"
        self.wheel = "hrx_loom-1.2.3rc1-py3-none-manylinux_2_39_x86_64.whl"
        self.contents = b"qualified wheel"
        self.digest = hashlib.sha256(self.contents).hexdigest()
        self.write_pin()
        (self.root / setup_sdk.MODULE).write_text(
            'git_override(module_name = "hrx", commit = "' + "a" * 40 + '")\n'
        )

    def write_pin(self, **changes):
        document = {
            "repository": "fixture/sdk",
            "revision": "a" * 40,
            "run_id": 123,
            "artifact_id": 456,
            "release_tag": "v1.2.3rc1",
            "wheel": self.wheel,
            "sha256": self.digest,
            "catalog": "github:fixture/library@v1.2.3rc1",
        } | changes
        (self.root / setup_sdk.SDK_PIN).write_text(json.dumps(document))

    def release(self, **changes):
        return {
            "tag_name": "v1.2.3rc1",
            "draft": False,
            "immutable": True,
            "assets": [
                {
                    "id": 789,
                    "name": self.wheel,
                    "state": "uploaded",
                    "digest": "sha256:" + self.digest,
                }
            ],
        } | changes

    def candidate_download(self, endpoint, output, _token, **_):
        self.assertIn("/actions/artifacts/456/zip", endpoint)
        with ZipFile(output, "w") as archive:
            archive.writestr(self.wheel, self.contents)

    def release_download(self, endpoint, output, _token, **arguments):
        self.assertIn("/releases/assets/789", endpoint)
        self.assertEqual(arguments["accept"], "application/octet-stream")
        output.write_bytes(self.contents)

    def test_missing_release_uses_the_exact_candidate_artifact(self):
        with (
            patch.object(setup_sdk, "_get_optional_release", return_value=None),
            patch.object(setup_sdk, "_download", side_effect=self.candidate_download),
        ):
            wheel, source = setup_sdk.download(self.root, self.output, "token")
        self.assertEqual(source, "candidate")
        self.assertEqual(wheel.read_bytes(), self.contents)

    def test_public_release_replaces_the_expiring_candidate_source(self):
        with (
            patch.object(
                setup_sdk, "_get_optional_release", return_value=self.release()
            ),
            patch.object(setup_sdk, "_download", side_effect=self.release_download),
        ):
            wheel, source = setup_sdk.download(self.root, self.output, "token")
        self.assertEqual(source, "release")
        self.assertEqual(wheel.read_bytes(), self.contents)

    def test_conflicting_public_release_never_falls_back(self):
        cases = (
            {"tag_name": "different"},
            {"draft": True},
            {"immutable": False},
            {"assets": []},
            {
                "assets": [
                    self.release()["assets"][0] | {"digest": "sha256:" + "0" * 64}
                ]
            },
        )
        for change in cases:
            with (
                self.subTest(change=change),
                patch.object(
                    setup_sdk,
                    "_get_optional_release",
                    return_value=self.release(**change),
                ),
                patch.object(setup_sdk, "_download") as download,
            ):
                with self.assertRaises(ValueError):
                    setup_sdk.download(self.root, self.output, "token")
                download.assert_not_called()

    def test_downloaded_bytes_must_match_the_reviewed_digest(self):
        def corrupt_download(_, output, _token, **__):
            output.write_bytes(b"different")

        with (
            patch.object(
                setup_sdk, "_get_optional_release", return_value=self.release()
            ),
            patch.object(setup_sdk, "_download", side_effect=corrupt_download),
        ):
            with self.assertRaisesRegex(ValueError, "differs from the pin"):
                setup_sdk.download(self.root, self.output, "token")
        self.assertFalse((self.output / self.wheel).exists())

    def test_development_pin_has_no_public_release_probe(self):
        development_wheel = "hrx_loom-1.2.3.dev0-py3-none-manylinux_2_39_x86_64.whl"
        self.write_pin(release_tag=None, wheel=development_wheel)

        def candidate_download(endpoint, output, _token, **_):
            self.assertIn("/actions/artifacts/456/zip", endpoint)
            with ZipFile(output, "w") as archive:
                archive.writestr(development_wheel, self.contents)

        with (
            patch.object(setup_sdk, "_get_optional_release") as get_release,
            patch.object(setup_sdk, "_download", side_effect=candidate_download),
        ):
            _, source = setup_sdk.download(self.root, self.output, "token")
        self.assertEqual(source, "candidate")
        get_release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
