# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Describes the files owned by an installed hrx-loom distribution."""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import sys
import sysconfig
from pathlib import Path


NATIVE_TOOLS = (
    "loom-benchmark",
    "loom-compile",
    "loom-format",
    "loom-link",
    "loom-serve",
    "loom-test",
)


def describe_distribution(
    distribution: importlib.metadata.Distribution, scripts: Path
) -> dict[str, object]:
    """Returns declared Python and native inputs without importing SDK code."""
    # Distribution.files filters out missing paths. Read the wheel's inventory
    # directly so an incomplete installation cannot silently lose a library.
    records = distribution.read_text("RECORD")
    if records is None:
        raise ValueError("hrx-loom has no installed file record; install its wheel")
    scripts = scripts.resolve()
    files: dict[str, str] = {}
    metadata = None
    for filename, _hash, _size in csv.reader(io.StringIO(records)):
        record = importlib.metadata.PackagePath(filename)
        source = Path(distribution.locate_file(record)).resolve()
        name = source.name
        if record.parts[0] == "loom" and source.suffix != ".pyc":
            destination = "python/" + record.as_posix()
        elif len(record.parts) == 2 and record.parts[0].endswith(".dist-info"):
            if name == "RECORD":
                metadata = str(source)
            if name not in ("METADATA", "loom.json"):
                continue
            destination = "python/" + record.as_posix()
        elif source.parent == scripts and (
            name.removesuffix(".exe") in NATIVE_TOOLS
            or name.endswith((".dll", ".so", ".dylib"))
            or ".so." in name
        ):
            destination = "native/" + name
        else:
            continue
        if not source.is_file():
            raise ValueError(f"Installed SDK file is missing: {source}")
        files[destination] = str(source)
    if metadata is None:
        raise ValueError("hrx-loom has no RECORD; install its wheel")
    extension = ".exe" if sys.platform == "win32" else ""
    for tool in NATIVE_TOOLS:
        if f"native/{tool}{extension}" not in files:
            raise ValueError(f"Installed SDK is missing native command {tool}")
    return {
        "version": distribution.version,
        "record": metadata,
        "executable_suffix": extension,
        "files": dict(sorted(files.items())),
    }


def main() -> int:
    try:
        distribution = importlib.metadata.distribution("hrx-loom")
        document = describe_distribution(
            distribution, Path(sysconfig.get_path("scripts"))
        )
    except (importlib.metadata.PackageNotFoundError, ValueError) as error:
        print(
            f"Cannot select the installed Loom SDK: {error}. "
            "Install hrx-loom in this Python environment, or select "
            "dev.py setup --hrx-source=PATH for compiler development.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
