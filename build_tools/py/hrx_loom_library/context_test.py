# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from build_tools.py.hrx_loom_library import context


class BazelSelectionTest(unittest.TestCase):
    def test_selected_launcher_receives_arguments_in_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = context.RepositoryContext(Path(directory))
            with mock.patch.dict(context.os.environ, {"BAZEL": sys.executable}):
                output = repository.bazel(
                    [
                        "-c",
                        "import os, sys; print(os.getcwd()); print(sys.argv[1])",
                        "argument with spaces; $unchanged",
                    ],
                    capture_output=True,
                )
            self.assertEqual(
                output.splitlines(),
                [str(Path(directory).resolve()), "argument with spaces; $unchanged"],
            )

    def test_invalid_selection_fails_without_using_path_default(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = context.RepositoryContext(Path(directory))
            for selection in ("", str(Path(directory) / "missing-bazel")):
                with self.subTest(selection=selection):
                    with mock.patch.dict(context.os.environ, {"BAZEL": selection}):
                        with self.assertRaisesRegex(
                            context.UserError, "BAZEL does not name an executable"
                        ):
                            repository.bazel_executable()

    def test_unset_selection_prefers_bazelisk(self):
        with mock.patch.dict(context.os.environ):
            context.os.environ.pop("BAZEL", None)
            with mock.patch.object(
                context.shutil,
                "which",
                side_effect=lambda name: {"bazelisk": "/tools/bazelisk"}.get(name),
            ):
                repository = context.RepositoryContext(Path.cwd())
                self.assertEqual(repository.bazel_executable(), "/tools/bazelisk")


class CommandFormattingTest(unittest.TestCase):
    def test_uses_windows_argument_rendering_on_windows(self):
        command = ["C:/Program Files/Bazel/bazel.exe", "build", "//kernel/..."]

        with mock.patch.object(context.os, "name", "nt"):
            rendered = context.format_command(command)

        self.assertEqual(rendered, subprocess.list2cmdline(command))

    def test_uses_shell_argument_rendering_on_posix(self):
        command = ["bazel", "build", "//kernel/..."]

        with mock.patch.object(context.os, "name", "posix"):
            rendered = context.format_command(command)

        self.assertEqual(rendered, shlex.join(command))


class ProcessDiagnosticsTest(unittest.TestCase):
    def test_success_preserves_captured_output_without_echoing_by_default(self):
        command = [sys.executable, "-c", "print('query result')"]
        with tempfile.TemporaryDirectory() as directory:
            for verbose in (False, True):
                with self.subTest(verbose=verbose):
                    repository = context.RepositoryContext(Path(directory), verbose)
                    output, diagnostics = io.StringIO(), io.StringIO()
                    with redirect_stdout(output), redirect_stderr(diagnostics):
                        result = repository.run(command, capture_output=True)
                    self.assertEqual(result, "query result\n")
                    self.assertEqual(output.getvalue(), "")
                    expected = "+ " + context.format_command(command) + "\n"
                    self.assertEqual(
                        diagnostics.getvalue(), expected if verbose else ""
                    )

    def test_failure_reports_command_and_both_streams_once(self):
        command = [
            sys.executable,
            "-c",
            "import sys; print('partial result'); "
            "print('failure detail', file=sys.stderr); sys.exit(7)",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for verbose in (False, True):
                with self.subTest(verbose=verbose):
                    repository = context.RepositoryContext(Path(directory), verbose)
                    output, diagnostics = io.StringIO(), io.StringIO()
                    with redirect_stdout(output), redirect_stderr(diagnostics):
                        with self.assertRaises(subprocess.CalledProcessError) as error:
                            repository.run(command, capture_output=True)
                    self.assertEqual(error.exception.returncode, 7)
                    self.assertEqual(error.exception.cmd, command)
                    self.assertEqual(output.getvalue(), "partial result\n")
                    self.assertEqual(
                        diagnostics.getvalue(),
                        "+ " + context.format_command(command) + "\nfailure detail\n",
                    )


if __name__ == "__main__":
    unittest.main()
