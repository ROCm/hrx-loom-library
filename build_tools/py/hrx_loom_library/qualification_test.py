# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.py.hrx_loom_library import qualification
from build_tools.py.hrx_loom_library.context import RepositoryContext, UserError


class QualificationTest(unittest.TestCase):
    def test_benchmark_query_uses_only_populated_authoring_roots(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            for package in (
                "experimental/model",
                "kernel/example",
                "motif/test",
                "program/text/test",
            ):
                package_root = repository_root / package
                package_root.mkdir(parents=True)
                (package_root / "BUILD.bazel").write_text(
                    "# package\n", encoding="utf-8"
                )
            (repository_root / "model").mkdir()

            expression = qualification.benchmark_query_expression(repository_root)

        self.assertEqual(
            expression,
            'attr(tags, "loom-benchmark-module", '
            "//experimental/... union //kernel/... union //motif/... union //program/...)",
        )

    def test_library_admission_uses_repository_tags_and_dependency_edges(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            for root in ("kernel", "motif"):
                package_root = repository_root / root / "example"
                package_root.mkdir(parents=True)
                (package_root / "BUILD.bazel").write_text(
                    "# package\n", encoding="utf-8"
                )

            expression = qualification._library_admission_query_expression(
                repository_root
            )

        self.assertEqual(
            expression,
            'labels(srcs, attr(tags, "loom-benchmark-admission", '
            '//kernel/... union //motif/...)) '
            "except "
            'deps(attr(tags, "loom-benchmark-module", '
            "//kernel/... union //motif/...))",
        )

    def test_library_admission_accepts_complete_qualification_closure(self):
        context = mock.Mock(spec=RepositoryContext)
        context.repository_root = Path("/repository")
        context.bazel.return_value = ""
        with mock.patch.object(
            qualification,
            "_library_admission_query_expression",
            return_value="qualification expression",
        ):
            qualification._check_library_qualification(context)

        context.bazel.assert_called_once_with(
            [
                "cquery",
                "qualification expression",
                "--output=label",
                "--noimplicit_deps",
                "--notool_deps",
                "--noinclude_aspects",
            ],
            capture_output=True,
        )

    def test_library_admission_reports_every_unqualified_library(self):
        context = mock.Mock(spec=RepositoryContext)
        context.repository_root = Path("/repository")
        context.bazel.return_value = (
            "//kernel/linear:projection\n"
            "//motif/format/other:q3\n//motif/format/ggml:q5_k\n"
        )
        with (
            mock.patch.object(
                qualification,
                "_library_admission_query_expression",
                return_value="qualification expression",
            ),
            self.assertRaisesRegex(
                UserError,
                "(?s)//kernel/linear:projection.*"
                "//motif/format/ggml:q5_k.*//motif/format/other:q3",
            ),
        ):
            qualification._check_library_qualification(context)


if __name__ == "__main__":
    unittest.main()
