# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Repository-wide source and benchmark qualification policy."""

from __future__ import annotations

import sys
from pathlib import Path

from .context import RepositoryContext, UserError


LOOM_BENCHMARK_ROOTS = ("experimental", "kernel", "model", "motif", "program")
LOOM_BENCHMARK_TAG = "loom-benchmark-module"
LOOM_BENCHMARK_ADMISSION_TAG = "loom-benchmark-admission"


def _authoring_scope(repository_root: Path) -> str:
    """Returns the populated packages participating in authoring policy."""
    scopes = [
        f"//{root}/..."
        for root in LOOM_BENCHMARK_ROOTS
        if any((repository_root / root).rglob("BUILD.bazel"))
    ]
    if not scopes:
        raise UserError("No benchmark library packages exist in this checkout")
    return " union ".join(scopes)


def benchmark_query_expression(repository_root: Path) -> str:
    """Returns the repository's test-owned benchmark modules."""
    return f'attr(tags, "{LOOM_BENCHMARK_TAG}", {_authoring_scope(repository_root)})'


def _library_admission_query_expression(repository_root: Path) -> str:
    scope = _authoring_scope(repository_root)
    benchmark_targets = f'attr(tags, "{LOOM_BENCHMARK_TAG}", {scope})'
    source_libraries = (
        f'labels(srcs, attr(tags, "{LOOM_BENCHMARK_ADMISSION_TAG}", {scope}))'
    )
    return f"{source_libraries} except deps({benchmark_targets})"


def _check_library_qualification(context: RepositoryContext) -> None:
    output = context.bazel(
        [
            "cquery",
            _library_admission_query_expression(context.repository_root),
            "--output=label",
            "--noimplicit_deps",
            "--notool_deps",
            "--noinclude_aspects",
        ],
        capture_output=True,
    )
    unqualified_libraries = sorted(
        line.strip() for line in output.splitlines() if line.strip()
    )
    if unqualified_libraries:
        labels = "\n  ".join(unqualified_libraries)
        raise UserError(
            "Source-bearing motifs and kernels require a benchmarkable "
            f"test dependency:\n  {labels}"
        )


def lint_repository(context: RepositoryContext) -> None:
    """Runs all repository-owned source and dependency policy checks."""
    source_policy_tool = (
        context.repository_root / "build_tools" / "bazel" / "source_policy.py"
    )
    context.run(
        [
            sys.executable,
            "-B",
            str(source_policy_tool),
            f"--repository-root={context.repository_root}",
        ]
    )
    _check_library_qualification(context)
