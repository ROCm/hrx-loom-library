#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Checks whether Loom sources match their repository architecture layer."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SOURCE_NON_CODE_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"|//[^\r\n]*')
LAYER_OPERATION_PATTERN = re.compile(
    r"(?<![\w.$%@#^-])"
    r"(?:config\.[A-Za-z_][A-Za-z0-9_.]*|kernel\.def|"
    r"[A-Za-z_][A-Za-z0-9_.]*\.target|target\.generic)"
    r"(?![\w.-])"
)
LOCAL_RULES_LABEL = '"//build_tools/bazel:defs.bzl"'
UPSTREAM_RULES_LABEL = '"@hrx//loom/build_tools/bazel:defs.bzl"'


def _layer_operations(source: str) -> set[str]:
    # The compiler owns grammar validation. These lexical layer checks ignore
    # literals, comments, and sigil-prefixed references to operation-like names.
    code = SOURCE_NON_CODE_PATTERN.sub(" ", source)
    return set(LAYER_OPERATION_PATTERN.findall(code))


def _starlark_without_line_comments(source: str) -> str:
    return "\n".join(line.partition("#")[0] for line in source.splitlines())


def check_sources(layer: str, sources: Iterable[tuple[str, str]]) -> list[str]:
    """Returns policy violations for named Loom source strings."""
    source_definitions = [
        (
            name,
            _layer_operations(text),
        )
        for name, text in sources
    ]
    if layer == "motif":
        violations = []
        for name, operations in source_definitions:
            if "kernel.def" in operations:
                violations.append(f"{name}: motif sources cannot declare kernel.def")
            for operation in sorted(operations - {"kernel.def"}):
                if operation.startswith("config."):
                    violations.append(
                        f"{name}: motif sources cannot use {operation}; "
                        "pass configuration values from a composition leaf"
                    )
                else:
                    violations.append(
                        f"{name}: motif sources cannot declare {operation}; "
                        "define concrete targets at a composition leaf"
                    )
        return violations
    if layer == "kernel" and not any(
        "kernel.def" in operations for _, operations in source_definitions
    ):
        names = ", ".join(name for name, _ in source_definitions)
        return [f"kernel package has no kernel.def declaration: {names}"]
    return []


def _package_policy(layer: str, relative_directory: Path) -> tuple[str, str]:
    if "test" in relative_directory.parts:
        return "test", "loom_test"
    if layer == "kernel":
        return "kernel", "loom_kernel_library"
    if layer == "model":
        return "model", "loom_library"
    return "motif", "loom_motif_library"


def _check_scoped_readmes(repository_root: Path) -> list[str]:
    violations: list[str] = []
    checked_directories: set[Path] = set()
    for root_name in ("kernel", "model", "motif", "target"):
        source_root = repository_root / root_name
        if not source_root.is_dir():
            continue
        source_directories = sorted(
            {source_path.parent for source_path in source_root.rglob("*.loom")}
        )
        for source_directory in source_directories:
            documentation_directory = source_directory
            while documentation_directory != repository_root:
                if documentation_directory in checked_directories:
                    break
                checked_directories.add(documentation_directory)
                if not (documentation_directory / "README.md").is_file():
                    violations.append(
                        f"{documentation_directory.relative_to(repository_root)}: "
                        "missing scoped README.md"
                    )
                if documentation_directory == source_root:
                    break
                documentation_directory = documentation_directory.parent
    return violations


def check_repository(repository_root: Path) -> list[str]:
    """Returns architecture-policy violations in a library checkout."""
    violations = _check_scoped_readmes(repository_root)
    for source_path in sorted((repository_root / "target").rglob("*.loom")):
        violations.append(
            f"{source_path.relative_to(repository_root)}: source-bearing "
            "target packages require a repository admission rule"
        )
    for layer in ("kernel", "model", "motif"):
        layer_root = repository_root / layer
        if not layer_root.is_dir():
            continue
        sources_by_directory: dict[Path, list[Path]] = defaultdict(list)
        for source_path in layer_root.rglob("*.loom"):
            sources_by_directory[source_path.parent].append(source_path)
        for source_directory, source_paths in sorted(sources_by_directory.items()):
            relative_directory = source_directory.relative_to(repository_root)
            source_layer, rule_name = _package_policy(layer, relative_directory)
            named_sources = [
                (str(path.relative_to(repository_root)), path.read_text())
                for path in sorted(source_paths)
            ]
            violations.extend(check_sources(source_layer, named_sources))

            build_path = source_directory / "BUILD.bazel"
            if not build_path.is_file():
                violations.append(f"{relative_directory}: missing adjacent BUILD.bazel")
                continue
            build_text = _starlark_without_line_comments(build_path.read_text())
            if LOCAL_RULES_LABEL not in build_text:
                violations.append(
                    f"{build_path.relative_to(repository_root)}: load repository "
                    "rules from //build_tools/bazel:defs.bzl"
                )
            if UPSTREAM_RULES_LABEL in build_text:
                violations.append(
                    f"{build_path.relative_to(repository_root)}: library "
                    "packages cannot bypass repository policy wrappers"
                )
            if not re.search(rf"\b{rule_name}\s*\(", build_text):
                violations.append(
                    f"{build_path.relative_to(repository_root)}: "
                    f"{source_layer} packages "
                    f"must declare {rule_name}"
                )
            for source_path in sorted(source_paths):
                source_literal = f'"{source_path.name}"'
                if source_literal not in build_text:
                    violations.append(
                        f"{build_path.relative_to(repository_root)}: "
                        f"{source_path.name} is not explicitly declared"
                    )
    for card_path in sorted((repository_root / "model").rglob("model.json")):
        relative_directory = card_path.parent.relative_to(repository_root)
        build_path = card_path.with_name("BUILD.bazel")
        if not build_path.is_file():
            violations.append(
                f"{relative_directory}: model card requires adjacent BUILD.bazel"
            )
            continue
        build_text = _starlark_without_line_comments(build_path.read_text())
        if LOCAL_RULES_LABEL not in build_text or UPSTREAM_RULES_LABEL in build_text:
            violations.append(
                f"{build_path.relative_to(repository_root)}: model cards use "
                "repository rules from //build_tools/bazel:defs.bzl"
            )
        has_model = re.search(r"\bloom_model\s*\(", build_text)
        has_reference = re.search(r"\bloom_model_documentation\s*\(", build_text)
        if not (has_model or has_reference) or '"model.json"' not in build_text:
            violations.append(
                f"{build_path.relative_to(repository_root)}: model card must be "
                "explicitly declared by loom_model or loom_model_documentation"
            )
        has_providers = any(
            re.search(
                r"\bloom_library\s*\(",
                _starlark_without_line_comments(provider_build.read_text()),
            )
            for provider_build in card_path.parent.rglob("BUILD.bazel")
        )
        if not has_model and has_providers:
            violations.append(
                f"{build_path.relative_to(repository_root)}: model programs require "
                "loom_model; a generated reference does not qualify executable providers"
            )
    return violations


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=["kernel", "model", "motif", "test"])
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("sources", metavar="SOURCE", nargs="*")
    return parser


def main() -> int:
    parser = _create_parser()
    args = parser.parse_args()
    if args.repository_root:
        if args.layer or args.sources:
            parser.error("--repository-root cannot be combined with source checks")
        violations = check_repository(args.repository_root.resolve())
    else:
        if not args.layer or not args.sources:
            parser.error("--layer and at least one SOURCE are required")
        source_paths = [Path(path) for path in args.sources]
        violations = check_sources(
            args.layer,
            [(str(path), path.read_text()) for path in source_paths],
        )
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
