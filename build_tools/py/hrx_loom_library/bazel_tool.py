# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Resolves a built Bazel executable and the content identity of its runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .context import UserError


@dataclass(frozen=True)
class BazelTool:
    """A materialized executable and its declared files, including shared libraries."""

    # Executable in the active Bazel execution root.
    path: Path
    # Content digest of the executable and its declared runtime file closure.
    sha256: str


def runfiles_sha256(label: str, execution_root: Path, files: dict) -> str:
    """Hashes declared runtime files by logical path and content."""
    identity = []
    for name, (filename, is_symlink) in sorted(files.items()):
        path = execution_root / filename
        # Unresolved symlinks (such as a Python venv interpreter) are relative
        # to their runfiles layout, not their location in the execution root.
        # Their target bytes are declared separately in the same closure.
        if is_symlink:
            identity.append((name, "symlink", str(path.readlink())))
            continue
        if not path.is_file():
            raise UserError(f"Bazel did not materialize {label} runtime input: {path}")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        identity.append((name, "file", digest))
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def resolve_tool(
    label: str, options: list[str], bazel: Callable[..., str]
) -> BazelTool:
    """Resolves a previously built tool without assuming a single-file target."""
    output = bazel(
        [
            "cquery",
            label,
            *options,
            "--output=starlark",
            "--starlark:expr=json.encode({"
            '"executable": target.files_to_run.executable.path, '
            '"files": {file.short_path: [file.path, file.is_symlink] for file in '
            "target.files.to_list() + target.default_runfiles.files.to_list()}})",
        ],
        capture_output=True,
    )
    record = json.loads(output)
    execution_root = Path(
        bazel(["info", *options, "execution_root"], capture_output=True).strip()
    )
    sha256 = runfiles_sha256(label, execution_root, record["files"])
    executable = execution_root / record["executable"]
    if not executable.is_file():
        raise UserError(f"Bazel did not materialize {label} executable: {executable}")
    return BazelTool(
        path=executable,
        sha256=sha256,
    )
