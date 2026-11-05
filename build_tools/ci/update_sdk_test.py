# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from build_tools.ci import update_sdk


class UpdateSDKTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.revision = "1" * 40
        self.run = {
            "id": 123,
            "path": update_sdk.SDK_WORKFLOW,
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
            "head_sha": self.revision,
            "head_repository": {
                "id": 456,
                "full_name": update_sdk.HRX_REPOSITORY,
            },
        }
        self.artifact = {
            "id": 789,
            "name": update_sdk.SDK_ARTIFACT,
            "expired": False,
            "workflow_run": {
                "id": 123,
                "head_sha": self.revision,
                "head_repository_id": 456,
            },
        }

    def write_artifact(
        self,
        *,
        version: str = "0.2.0.dev0",
        package: str = "hrx-loom",
        metadata_version: str | None = None,
        artifact_name: str | None = None,
        catalog: str | None = ("github:ROCm/hrx-loom-library@" + "4" * 40),
    ) -> tuple[Path, bytes, str]:
        filename = (
            artifact_name or f"hrx_loom-{version}-py3-none-manylinux_2_39_x86_64.whl"
        )
        wheel_path = self.root / "wheel.zip"
        prefix = f"hrx_loom-{version}.dist-info/"
        with ZipFile(wheel_path, "w") as wheel:
            wheel.writestr(
                prefix + "METADATA",
                f"Name: {package}\nVersion: {metadata_version or version}\n",
            )
            wheel.writestr(prefix + "loom.json", json.dumps({"catalog": catalog}))
        contents = wheel_path.read_bytes()
        artifact_path = self.root / "artifact.zip"
        with ZipFile(artifact_path, "w") as artifact:
            artifact.writestr(filename, contents)
        return artifact_path, contents, filename

    def write_pin(self, revision: str = "2" * 40) -> None:
        (self.root / "build_tools/ci").mkdir(parents=True)
        (self.root / update_sdk.SDK_PIN).write_text(
            json.dumps(
                {
                    "repository": update_sdk.HRX_REPOSITORY,
                    "revision": revision,
                    "run_id": 1,
                    "artifact_id": 2,
                    "release_tag": None,
                    "wheel": "old.whl",
                    "sha256": "3" * 64,
                    "catalog": "github:ROCm/hrx-loom-library@" + "4" * 40,
                }
            )
        )
        (self.root / update_sdk.MODULE).write_text(
            f'git_override(\n    module_name = "hrx",\n    commit = "{revision}",\n)\n'
        )

    def test_run_id_accepts_the_github_run_and_job_forms(self):
        self.assertEqual(update_sdk.parse_run_id("123"), 123)
        self.assertEqual(
            update_sdk.parse_run_id(
                "https://github.com/ROCm/hrx-system/actions/runs/123/job/456?pr=7"
            ),
            123,
        )
        for value in (
            "0",
            "main",
            "http://github.com/ROCm/hrx-system/actions/runs/123",
            "https://github.com/other/repository/actions/runs/123",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                update_sdk.parse_run_id(value)

    def test_candidate_binds_a_successful_manual_sdk_run_to_its_artifact(self):
        self.assertEqual(
            update_sdk.select_candidate(123, self.run, [self.artifact]),
            update_sdk.Candidate(123, self.revision, 789),
        )
        for field, value in (
            ("id", 124),
            ("path", ".github/workflows/ci_iree.yml"),
            ("event", "pull_request"),
            ("status", "in_progress"),
            ("conclusion", "failure"),
        ):
            with self.subTest(field=field):
                run = dict(self.run, **{field: value})
                with self.assertRaisesRegex(ValueError, "successful manually"):
                    update_sdk.select_candidate(123, run, [self.artifact])

    def test_candidate_rejects_an_unusable_or_unrelated_artifact(self):
        cases = (
            [],
            [self.artifact, self.artifact],
            [self.artifact | {"expired": True}],
            [
                self.artifact
                | {
                    "workflow_run": self.artifact["workflow_run"]
                    | {"head_sha": "2" * 40}
                }
            ],
        )
        for artifacts in cases:
            with self.subTest(artifacts=artifacts), self.assertRaises(ValueError):
                update_sdk.select_candidate(123, self.run, artifacts)

    def test_artifact_supplies_the_exact_wheel_name_version_and_digest(self):
        path, contents, filename = self.write_artifact()
        self.assertEqual(
            update_sdk.inspect_artifact(path),
            update_sdk.Wheel(
                filename,
                "0.2.0.dev0",
                hashlib.sha256(contents).hexdigest(),
                "github:ROCm/hrx-loom-library@" + "4" * 40,
            ),
        )

    def test_artifact_rejects_ambiguous_or_inconsistent_wheels(self):
        cases = (
            {"package": "other"},
            {"metadata_version": "0.2.0"},
            {"artifact_name": "other.whl"},
            {"catalog": "not-a-catalog"},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                path, _, _ = self.write_artifact(**arguments)
                with self.assertRaises(ValueError):
                    update_sdk.inspect_artifact(path)
                path.unlink()

        path, contents, filename = self.write_artifact()
        with ZipFile(path, "a") as artifact:
            artifact.writestr("second.whl", contents)
        with self.assertRaisesRegex(ValueError, "one root-level wheel"):
            update_sdk.inspect_artifact(path)

    def test_update_changes_the_installed_and_source_pins_together(self):
        old_revision = "2" * 40
        self.write_pin(old_revision)
        candidate = update_sdk.Candidate(123, self.revision, 789)
        catalog = "github:ROCm/hrx-loom-library@v0.2.0rc1"
        wheel = update_sdk.Wheel("candidate.whl", "0.2.0rc1", "4" * 64, catalog)
        update_sdk.update_pin(self.root, candidate, wheel)
        self.assertEqual(
            json.loads((self.root / update_sdk.SDK_PIN).read_text()),
            {
                "repository": update_sdk.HRX_REPOSITORY,
                "revision": self.revision,
                "run_id": 123,
                "artifact_id": 789,
                "release_tag": "v0.2.0rc1",
                "wheel": "candidate.whl",
                "sha256": "4" * 64,
                "catalog": catalog,
            },
        )
        module = (self.root / update_sdk.MODULE).read_text()
        self.assertIn(f'commit = "{self.revision}"', module)
        self.assertNotIn(old_revision, module)

    def test_update_rejects_mismatched_existing_pins_before_writing(self):
        self.write_pin()
        module_path = self.root / update_sdk.MODULE
        module_path.write_text(module_path.read_text().replace("2" * 40, "3" * 40))
        sdk_path = self.root / update_sdk.SDK_PIN
        before = (sdk_path.read_bytes(), module_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "must select the same"):
            update_sdk.update_pin(
                self.root,
                update_sdk.Candidate(123, self.revision, 789),
                update_sdk.Wheel("candidate.whl", "0.2.0.dev0", "4" * 64, None),
            )
        self.assertEqual(before, (sdk_path.read_bytes(), module_path.read_bytes()))


if __name__ == "__main__":
    unittest.main()
