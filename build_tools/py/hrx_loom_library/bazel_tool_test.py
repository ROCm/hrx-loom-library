# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from build_tools.py.hrx_loom_library.bazel_tool import resolve_tool
from build_tools.py.hrx_loom_library.context import UserError


class BazelToolTest(unittest.TestCase):
    def test_identity_tracks_runtime_contents_not_install_or_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {
                "executable": "bin/tool",
                "files": {
                    "bin/tool": ["bin/tool", False],
                    "bin/library": ["bin/library", False],
                    "python/report.py": ["external/sdk/report.py", False],
                    "python/runtime": ["external/python/runtime", False],
                },
            }
            for filename, _ in record["files"].values():
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(filename)

            def bazel(arguments, *, capture_output):
                if arguments[0] == "info":
                    return str(root)
                return json.dumps(record)

            link = root / "venv/python"
            link.parent.mkdir()
            link.symlink_to("../../python/runtime")
            record["files"]["venv/python"] = ["venv/python", True]

            first = resolve_tool("@sdk//:tool", [], bazel)
            (root / "unrelated.py").write_text("not a runtime input")
            self.assertEqual(first, resolve_tool("@sdk//:tool", [], bazel))

            # Bazel's configuration/output directory is not implementation.
            (root / "bin").rename(root / "other-bin")
            record["executable"] = "other-bin/tool"
            record["files"]["bin/tool"] = ["other-bin/tool", False]
            record["files"]["bin/library"] = ["other-bin/library", False]
            relocated = resolve_tool("@sdk//:tool", [], bazel)
            self.assertEqual(first.sha256, relocated.sha256)

            (root / "other-bin/library").write_text("changed library")
            changed = resolve_tool("@sdk//:tool", [], bazel)
            self.assertNotEqual(first.sha256, changed.sha256)
            (root / "external/sdk/report.py").write_text("changed Python")
            self.assertNotEqual(
                changed.sha256, resolve_tool("@sdk//:tool", [], bazel).sha256
            )

    def test_missing_declared_input_fails(self):
        with tempfile.TemporaryDirectory() as directory:

            def bazel(arguments, *, capture_output):
                if arguments[0] == "info":
                    return directory
                return json.dumps(
                    {
                        "executable": "tool",
                        "files": {
                            "tool": ["tool", False],
                            "library": ["library", False],
                        },
                    }
                )

            with self.assertRaisesRegex(UserError, "did not materialize"):
                resolve_tool("@sdk//:tool", [], bazel)


if __name__ == "__main__":
    unittest.main()
