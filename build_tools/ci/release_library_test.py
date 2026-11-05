# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Library publication transitions at the GitHub boundary."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from build_tools.ci import release_library


class ReleaseLibraryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "build_tools/ci").mkdir(parents=True)
        self.commit = "a" * 40
        self.write_pin("1.2.3rc1")
        self.identity = release_library.read_identity(self.root)
        self.catalog = {
            "schema_version": 1,
            "models": {
                "fixture/model": {
                    "name": "Fixture",
                    "card": "model/fixture/model.json",
                    "services": {},
                    "variants": ["default"],
                    "presets": ["default"],
                }
            },
        }
        (self.root / release_library.CATALOG).write_text(json.dumps(self.catalog))
        self.loom_fetch = self.root / "loom-fetch"
        self.loom_fetch.touch()

    def write_pin(self, version):
        (self.root / release_library.SDK_PIN).write_text(
            json.dumps(
                {
                    "repository": release_library.HRX_REPOSITORY,
                    "revision": "b" * 40,
                    "run_id": 123,
                    "artifact_id": 456,
                    "wheel": (
                        f"hrx_loom-{version}-py3-none-"
                        "manylinux_2_39_x86_64.whl"
                    ),
                    "sha256": "c" * 64,
                    "catalog": f"github:ROCm/hrx-loom-library@v{version}",
                }
            )
        )

    def catalog_listing(self):
        return self.catalog | {
            "snapshot": {
                "source": self.identity.catalog,
                "location": (
                    "https://raw.githubusercontent.com/ROCm/hrx-loom-library/"
                    f"{self.commit}/catalog.json"
                ),
                "revision": self.commit,
            }
        }

    def tag_document(self, commit=None):
        return {"object": {"type": "commit", "sha": commit or self.commit}}

    def library_release_document(self, *, draft=False, immutable=True):
        return {
            "id": 789,
            "tag_name": self.identity.tag,
            "name": f"Loom Library {self.identity.tag}",
            "body": release_library.release_body(self.identity),
            "draft": draft,
            "prerelease": self.identity.prerelease,
            "immutable": immutable,
            "assets": [],
        }

    def publish(self):
        release_library.publish(
            self.root,
            "fixture/library",
            self.commit,
            self.identity,
            "token",
            self.loom_fetch,
        )

    def test_identity_is_derived_from_the_qualified_release_wheel(self):
        self.assertEqual(self.identity.version, "1.2.3rc1")
        self.assertEqual(self.identity.tag, "v1.2.3rc1")
        self.assertTrue(self.identity.prerelease)
        self.assertEqual(self.identity.wheel, (
            "hrx_loom-1.2.3rc1-py3-none-manylinux_2_39_x86_64.whl"
        ))
        self.assertEqual(self.identity.digest, "c" * 64)
        self.assertEqual(
            self.identity.catalog,
            "github:ROCm/hrx-loom-library@v1.2.3rc1",
        )
        self.write_pin("1.2.3")
        self.assertFalse(release_library.read_identity(self.root).prerelease)
        for version in ("1.2.3.dev0", "01.2.3", "unknown"):
            with self.subTest(version=version):
                self.write_pin(version)
                with self.assertRaisesRegex(ValueError, "release SDK wheel"):
                    release_library.read_identity(self.root)

        self.write_pin("1.2.3")
        path = self.root / release_library.SDK_PIN
        pin = json.loads(path.read_text())
        pin["catalog"] = "github:ROCm/hrx-loom-library@different"
        path.write_text(json.dumps(pin))
        with self.assertRaisesRegex(ValueError, "coordinated library tag"):
            release_library.read_identity(self.root)

    def test_new_tag_and_release_publish_from_a_verified_draft(self):
        with (
            patch.object(
                release_library,
                "_get_optional",
                side_effect=[
                    None,
                    self.tag_document(),
                    None,
                    self.library_release_document(draft=True, immutable=False),
                    self.library_release_document(),
                ],
            ),
            patch.object(release_library, "_gh") as gh,
            patch.object(release_library, "_read_catalog") as read_catalog,
        ):
            read_catalog.side_effect = lambda _: (
                self.assertEqual(gh.call_count, 2) or self.catalog_listing()
            )
            self.publish()
        self.assertEqual(gh.call_count, 3)
        tag = gh.call_args_list[0].args
        self.assertEqual(tag[:3], ("api", "--method", "POST"))
        self.assertIn(f"ref=refs/tags/{self.identity.tag}", tag)
        self.assertIn(f"sha={self.commit}", tag)
        create = gh.call_args_list[1].args
        self.assertEqual(create[:3], ("release", "create", self.identity.tag))
        self.assertIn("--draft", create)
        self.assertIn("--verify-tag", create)
        self.assertIn("--prerelease", create)
        self.assertIn("--latest=false", create)
        self.assertEqual(gh.call_args_list[2].args[1], "edit")

    def test_retry_publishes_an_exact_existing_draft(self):
        with (
            patch.object(
                release_library,
                "_get_optional",
                side_effect=[
                    self.tag_document(),
                    self.library_release_document(draft=True, immutable=False),
                    self.library_release_document(),
                ],
            ),
            patch.object(release_library, "_gh") as gh,
            patch.object(
                release_library, "_read_catalog", return_value=self.catalog_listing()
            ),
        ):
            self.publish()
        gh.assert_called_once_with(
            "release",
            "edit",
            self.identity.tag,
            "--repo",
            "fixture/library",
            "--draft=false",
        )

    def test_complete_public_release_is_read_only(self):
        with (
            patch.object(
                release_library,
                "_get_optional",
                side_effect=[
                    self.tag_document(),
                    self.library_release_document(),
                ],
            ),
            patch.object(release_library, "_gh") as gh,
            patch.object(
                release_library, "_read_catalog", return_value=self.catalog_listing()
            ),
        ):
            self.publish()
        gh.assert_not_called()

    def test_catalog_proof_blocks_immutable_publication(self):
        listing = self.catalog_listing()
        listing["models"] = {}
        with (
            patch.object(
                release_library,
                "_get_optional",
                side_effect=[
                    None,
                    self.tag_document(),
                    None,
                    self.library_release_document(draft=True, immutable=False),
                ],
            ),
            patch.object(release_library, "_gh") as gh,
            patch.object(release_library, "_read_catalog", return_value=listing),
        ):
            with self.assertRaisesRegex(ValueError, "admitted catalog"):
                self.publish()
        self.assertEqual(gh.call_count, 2)
        self.assertNotIn("edit", [call.args[1] for call in gh.call_args_list])

    def test_catalog_listing_uses_the_installed_sdk_default(self):
        listing = self.catalog_listing()
        completed = CompletedProcess([], 0, stdout=json.dumps(listing))
        with patch.object(
            release_library.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(
                release_library._read_catalog(self.loom_fetch), listing
            )
        arguments = run.call_args.args[0]
        self.assertEqual(arguments[0], str(self.loom_fetch))
        self.assertIn("--list", arguments)
        self.assertIn("--json", arguments)
        self.assertIn("--no-input", arguments)
        self.assertTrue(any(value.startswith("--cache=") for value in arguments))
        self.assertFalse(any(value.startswith("--catalog") for value in arguments))

    def test_conflicting_library_state_never_mutates(self):
        conflicts = [
            [self.tag_document("d" * 40)],
            [
                self.tag_document(),
                self.library_release_document() | {"name": "Different"},
            ],
            [
                self.tag_document(),
                self.library_release_document()
                | {"assets": [{"name": "unexpected"}]},
            ],
            [
                self.tag_document(),
                self.library_release_document(immutable=False),
            ],
        ]
        for responses in conflicts:
            with (
                self.subTest(responses=responses),
                patch.object(
                    release_library, "_get_optional", side_effect=responses
                ),
                patch.object(release_library, "_gh") as gh,
                patch.object(release_library, "_read_catalog") as read_catalog,
            ):
                with self.assertRaises(ValueError):
                    self.publish()
                gh.assert_not_called()
                read_catalog.assert_not_called()

    def test_only_http_404_means_absence(self):
        for code in (404, 401, 403, 429, 500):
            with (
                self.subTest(code=code),
                patch(
                    "urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError(
                        "https://api.github.com", code, "Failure", {}, None
                    ),
                ),
            ):
                if code == 404:
                    self.assertIsNone(
                        release_library._get_optional(
                            "https://api.github.com/fixture", "token"
                        )
                    )
                else:
                    with self.assertRaises(urllib.error.HTTPError):
                        release_library._get_optional(
                            "https://api.github.com/fixture", "token"
                        )

    def test_http_response_is_decoded_as_json(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b'{"ok":true}')):
            self.assertEqual(
                release_library._get_optional(
                    "https://api.github.com/fixture", "token"
                ),
                {"ok": True},
            )


if __name__ == "__main__":
    unittest.main()
