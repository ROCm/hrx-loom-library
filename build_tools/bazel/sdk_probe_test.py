# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile
import unittest
from pathlib import Path

from build_tools.bazel import sdk_probe


class InstalledSdkTest(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.packages = self.root / "packages"
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        self.metadata = self.packages / "hrx_loom-1.2.3.dist-info"
        self.metadata.mkdir(parents=True)
        (self.metadata / "METADATA").write_text("Name: hrx-loom\nVersion: 1.2.3\n")
        (self.metadata / "loom.json").write_text('{"catalog":null}')
        python = self.packages / "loom" / "__init__.py"
        python.parent.mkdir()
        python.write_text("")
        extension = ".exe" if sys.platform == "win32" else ""
        self.commands = [
            self.scripts / (tool + extension) for tool in sdk_probe.NATIVE_TOOLS
        ]
        for command in self.commands:
            command.write_bytes(b"native program")
        self.library = self.scripts / ("loomc.dll" if extension else "libloomc.so")
        self.library.write_bytes(b"shared library")
        self.record = self.metadata / "RECORD"
        files = [
            python,
            self.metadata / "METADATA",
            self.metadata / "loom.json",
            self.record,
            self.library,
            *self.commands,
        ]
        self.record.write_text(
            "".join(
                Path(os.path.relpath(file, self.packages)).as_posix() + ",,\n"
                for file in files
            )
        )
        self.distribution = importlib.metadata.Distribution.at(self.metadata)

    def test_inventory_keeps_native_closure_and_python_package(self):
        sdk = sdk_probe.describe_distribution(self.distribution, self.scripts)
        self.assertEqual(sdk["version"], "1.2.3")
        self.assertEqual(sdk["record"], str(self.record.resolve()))
        self.assertEqual(
            sdk["files"]["native/" + self.library.name], str(self.library.resolve())
        )
        self.assertIn("python/loom/__init__.py", sdk["files"])
        self.assertIn("python/hrx_loom-1.2.3.dist-info/loom.json", sdk["files"])
        self.assertNotIn("python/hrx_loom-1.2.3.dist-info/RECORD", sdk["files"])

    def test_missing_installed_file_is_an_error(self):
        self.commands[0].unlink()
        with self.assertRaisesRegex(ValueError, "Installed SDK file is missing"):
            sdk_probe.describe_distribution(self.distribution, self.scripts)

    def test_incomplete_wheel_does_not_select_source_tools(self):
        self.record.write_text(
            "loom/__init__.py,,\nhrx_loom-1.2.3.dist-info/RECORD,,\n"
        )
        with self.assertRaisesRegex(ValueError, "missing native command"):
            sdk_probe.describe_distribution(self.distribution, self.scripts)


if __name__ == "__main__":
    unittest.main()
