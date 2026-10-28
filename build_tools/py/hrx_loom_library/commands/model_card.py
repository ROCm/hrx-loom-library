# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Synchronizes model-card compiler file lists from Bazel provider analysis."""

from __future__ import annotations

import argparse
import copy
import difflib
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ..context import RepositoryContext, UserError


_MODEL_QUERY = (
    '"\\n".join([json.encode({"target": str(target.label), '
    '"card": provider.card, "programs": provider.programs}) '
    "for key, provider in providers(target).items() "
    'if key.endswith("%LoomModelInfo")])'
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise UserError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(text: str, name: str) -> dict:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except ValueError as error:
        raise UserError(f"Invalid JSON in {name}: {error}") from error
    if not isinstance(value, dict):
        raise UserError(f"Expected a JSON object in {name}")
    return value


def _card_path(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise UserError(f"Expected a repository-relative model card: {name!r}")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise UserError(f"Model card resolves outside the repository: {name}")
    return result


def _file_records(program: dict) -> dict[str, dict]:
    records = {}
    for role in ("inputs", "libraries"):
        for value in program[role]:
            provider = {"path": value} if isinstance(value, str) else value
            for item in [provider, *provider.get("data", [])]:
                record = {"path": item} if isinstance(item, str) else item
                path = record["path"]
                if path in records and any(
                    records[path].get(field) != record.get(field)
                    for field in ("origin", "transform")
                ):
                    raise UserError(f"Conflicting acquisition declarations for {path}")
                records[path] = record
    return records


def _merge_provider(value, authored: dict[str, dict]):
    record = {"path": value} if isinstance(value, str) else copy.deepcopy(value)
    previous = authored.get(record["path"], {})
    # Acquisition remains authored even when a file changes linker role or
    # auxiliary owner. The analysis result owns only compiler declarations.
    for field in ("origin", "transform"):
        if field in previous:
            record[field] = copy.deepcopy(previous[field])
    if "data" in record:
        record["data"] = [_merge_provider(item, authored) for item in record["data"]]
    return record if len(record) > 1 else record["path"]


def _staged_inputs_ready(context: RepositoryContext) -> bool:
    def relevant(status: str, name: str) -> bool:
        path = PurePosixPath(name)
        # File additions/removals can change a declared glob without a BUILD
        # edit. Content-only edits outside build/card declarations cannot.
        return (
            status != "M"
            or path.name in ("BUILD", "WORKSPACE", "MODULE.bazel.lock", "model.json")
            or path.name.startswith(".bazelrc")
            or path.suffix in (".bzl", ".bazel")
        )

    def changed_inputs(*arguments: str) -> list[str]:
        output = context.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-renames",
                "--name-status",
                "-z",
                *arguments,
            ],
            capture_output=True,
        )
        fields = output.rstrip("\0").split("\0") if output else []
        return [
            name
            for status, name in zip(fields[::2], fields[1::2], strict=True)
            if relevant(status, name)
        ]

    if not changed_inputs("--cached"):
        return False
    unstaged = changed_inputs()
    untracked = context.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], capture_output=True
    )
    conflicts = sorted(set(unstaged) | {name for name in untracked.split("\0") if name})
    if conflicts:
        raise UserError(
            "Model-card analysis requires complete staged build/card files; "
            "stage or isolate these working-tree changes first: " + ", ".join(conflicts)
        )
    return True


def synchronize(
    context: RepositoryContext, *, targets: Sequence[str] = (), fix: bool = False
) -> None:
    patterns = targets or ["//model/..."]
    if any(
        not pattern.startswith("//")
        or any(character.isspace() or character in "'\"()" for character in pattern)
        for pattern in patterns
    ):
        raise UserError("Model-card targets must be local Bazel target patterns")
    output = context.bazel(
        [
            "cquery",
            "kind('loom_model rule', set(" + " ".join(patterns) + "))",
            "--output=starlark",
            "--starlark:expr=" + _MODEL_QUERY,
        ],
        capture_output=True,
    )
    cards = {}
    owners = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        result = _load_json(line, "Bazel model analysis")
        card = result["card"]
        programs = cards.setdefault(card, {})
        for name, declarations in result["programs"].items():
            key = (card, name)
            if name in programs and programs[name] != declarations:
                raise UserError(
                    f"{card}: programs.{name} differs between "
                    f"{owners[key]} and {result['target']}"
                )
            programs[name] = declarations
            owners[key] = result["target"]
    changes = []
    for name, programs in sorted(cards.items()):
        path = _card_path(context.repository_root, name)
        original = path.read_text(encoding="utf-8")
        document = _load_json(original, name)
        changed = False
        for key, declarations in sorted(programs.items()):
            if key not in document.get("programs", {}):
                raise UserError(f"{name}: Bazel names undeclared program set {key!r}")
            program = document["programs"][key]
            try:
                authored = _file_records(program)
            except (AttributeError, KeyError, TypeError) as error:
                raise UserError(
                    f"{name}: malformed programs.{key} file declarations: {error}"
                ) from error
            for role in ("inputs", "libraries"):
                generated = [
                    _merge_provider(item, authored) for item in declarations[role]
                ]
                changed |= program[role] != generated
                program[role] = generated
        if changed:
            replacement = json.dumps(document, indent=2) + "\n"
            changes.append((path, original, replacement))
            print(
                "".join(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        replacement.splitlines(keepends=True),
                        fromfile="a/" + name,
                        tofile="b/" + name,
                    )
                ),
                end="",
            )
    if fix:
        # Validate every selected card before mutating any of them.
        for path, original, _ in changes:
            if path.read_text(encoding="utf-8") != original:
                raise UserError(f"Model card changed during analysis: {path}")
        for path, _, replacement in changes:
            path.write_text(replacement, encoding="utf-8")
    count = sum(len(programs) for programs in cards.values())
    print(
        f"Model cards: {len(cards)} checked, {count} program sets, {len(changes)} {'updated' if fix else 'stale'}."
    )
    if changes and not fix:
        raise UserError(
            "Model cards differ from Bazel; run python dev.py model-card --fix and review the diff"
        )


def check_repository(context: RepositoryContext) -> None:
    """Checks every model composition admitted to the repository."""
    synchronize(context)


def _run(args: argparse.Namespace, context: RepositoryContext) -> None:
    if args.staged:
        if args.mode == "fix":
            raise UserError(
                "Use --fix outside the commit hook, review, and stage the result"
            )
        if not _staged_inputs_ready(context):
            return
    synchronize(context, targets=args.targets, fix=args.mode == "fix")


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "model-card",
        help="Check or synchronize compiler file lists from Bazel.",
        description=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", dest="mode", action="store_const", const="check")
    mode.add_argument("--fix", dest="mode", action="store_const", const="fix")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        dest="targets",
        metavar="PATTERN",
        help="Limit model declarations to a Bazel target pattern (repeatable).",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Check complete staged build/card changes for the commit hook.",
    )
    parser.set_defaults(handler=_run, mode="check")
