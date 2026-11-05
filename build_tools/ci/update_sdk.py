#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Selects one qualified HRX SDK workflow artifact for library CI."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


HRX_REPOSITORY = "ROCm/hrx-system"
SDK_WORKFLOW = ".github/workflows/build_loom_sdk.yml"
SDK_ARTIFACT = "loom-sdk-wheel-linux-Release"
SDK_PIN = Path("build_tools/ci/sdk.json")
MODULE = Path("MODULE.bazel")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+){2}(?:(?:a|b|rc)[0-9]+|\.dev[0-9]+)?")
_WHEEL = re.compile(
    r"hrx_loom-(?P<version>[^/]+)-py3-none-manylinux_2_39_x86_64\.whl"
)
_CATALOG = re.compile(r"github:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)@[^\s]+")


@dataclass(frozen=True)
class Candidate:
    run_id: int
    revision: str
    artifact_id: int


@dataclass(frozen=True)
class Wheel:
    filename: str
    version: str
    digest: str
    catalog: str | None


def parse_run_id(value: str) -> int:
    """Parses a workflow run ID or an HRX workflow run URL."""
    if value.isdecimal():
        run_id = int(value)
    else:
        url = urllib.parse.urlsplit(value)
        parts = url.path.strip("/").split("/")
        if (
            url.scheme != "https"
            or url.netloc.casefold() != "github.com"
            or parts[:4] != ["ROCm", "hrx-system", "actions", "runs"]
            or len(parts) < 5
            or not parts[4].isdecimal()
        ):
            raise ValueError("run must be an ROCm/hrx-system workflow run ID or URL")
        run_id = int(parts[4])
    if run_id <= 0:
        raise ValueError("workflow run ID must be positive")
    return run_id


def _gh_json(endpoint: str) -> Any:
    result = subprocess.run(
        ["gh", "api", endpoint], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def _download_artifact(artifact_id: int, output: Path) -> None:
    with output.open("wb") as stream:
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/{HRX_REPOSITORY}/actions/artifacts/{artifact_id}/zip",
            ],
            check=True,
            stdout=stream,
        )


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"GitHub returned an invalid {description}")
    return value


def select_candidate(
    run_id: int, run_value: Any, artifact_values: list[Any]
) -> Candidate:
    """Validates the producer run and selects its sole Linux Release wheel."""
    run = _require_mapping(run_value, "workflow run")
    repository = _require_mapping(run.get("head_repository"), "head repository")
    revision = run.get("head_sha")
    expected = {
        "id": run_id,
        "path": SDK_WORKFLOW,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "run must be a successful manually dispatched Loom / SDK workflow"
        )
    if repository.get("full_name") != HRX_REPOSITORY:
        raise ValueError(f"workflow source must be {HRX_REPOSITORY}")
    if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
        raise ValueError("workflow run has an invalid source revision")

    matching = []
    for value in artifact_values:
        artifact = _require_mapping(value, "artifact")
        if artifact.get("name") == SDK_ARTIFACT:
            matching.append(artifact)
    if len(matching) != 1:
        raise ValueError(f"workflow must contain exactly one {SDK_ARTIFACT} artifact")
    artifact = matching[0]
    artifact_run = _require_mapping(artifact.get("workflow_run"), "artifact run")
    artifact_id = artifact.get("id")
    if artifact.get("expired") is not False:
        raise ValueError("selected SDK artifact has expired")
    if not isinstance(artifact_id, int) or artifact_id <= 0:
        raise ValueError("selected SDK artifact has an invalid ID")
    if (
        artifact_run.get("id") != run_id
        or artifact_run.get("head_sha") != revision
        or artifact_run.get("head_repository_id") != repository.get("id")
    ):
        raise ValueError("selected SDK artifact does not belong to the workflow run")
    return Candidate(run_id, revision, artifact_id)


def fetch_candidate(run_id: int) -> Candidate:
    run = _gh_json(f"repos/{HRX_REPOSITORY}/actions/runs/{run_id}")
    artifacts = []
    page = 1
    while True:
        document = _require_mapping(
            _gh_json(
                f"repos/{HRX_REPOSITORY}/actions/runs/{run_id}/artifacts"
                f"?per_page=100&page={page}"
            ),
            "artifact listing",
        )
        records = document.get("artifacts")
        if not isinstance(records, list):
            raise ValueError("GitHub returned an invalid artifact listing")
        artifacts.extend(records)
        if len(records) < 100:
            break
        page += 1
    return select_candidate(run_id, run, artifacts)


def inspect_artifact(path: Path) -> Wheel:
    """Returns the exact qualified wheel identity carried by an artifact ZIP."""
    try:
        with ZipFile(path) as artifact:
            members = [member for member in artifact.infolist() if not member.is_dir()]
            if len(members) != 1 or Path(members[0].filename).name != members[0].filename:
                raise ValueError("SDK artifact must contain one root-level wheel")
            filename = members[0].filename
            match = _WHEEL.fullmatch(filename)
            if match is None:
                raise ValueError("SDK artifact does not contain a Linux Release wheel")
            version = match.group("version")
            if _VERSION.fullmatch(version) is None:
                raise ValueError("SDK wheel has an invalid release version")
            contents = artifact.read(members[0])

        prefix = f"hrx_loom-{version}.dist-info/"
        with ZipFile(io.BytesIO(contents)) as wheel:
            names = wheel.namelist()
            metadata_name = prefix + "METADATA"
            if names.count(metadata_name) != 1:
                raise ValueError("SDK wheel has an invalid metadata identity")
            metadata = email.parser.Parser().parsestr(
                wheel.read(metadata_name).decode("utf-8")
            )
            if metadata.get_all("Name") != ["hrx-loom"]:
                raise ValueError("SDK wheel has an invalid package name")
            if metadata.get_all("Version") != [version]:
                raise ValueError("SDK wheel filename and metadata versions differ")
            catalog_name = prefix + "loom.json"
            if names.count(catalog_name) != 1:
                raise ValueError("SDK wheel has an invalid catalog identity")
            catalog_document = _require_mapping(
                json.loads(wheel.read(catalog_name)), "SDK catalog identity"
            )
            if set(catalog_document) != {"catalog"}:
                raise ValueError("SDK wheel has an invalid catalog identity")
            catalog = catalog_document["catalog"]
            if catalog is not None:
                match = _CATALOG.fullmatch(catalog) if isinstance(catalog, str) else None
                if match is None or any(part in (".", "..") for part in match.groups()):
                    raise ValueError("SDK wheel has an invalid catalog association")
    except BadZipFile as error:
        raise ValueError("SDK artifact or wheel is not a valid ZIP archive") from error
    return Wheel(filename, version, hashlib.sha256(contents).hexdigest(), catalog)


def update_pin(root: Path, candidate: Candidate, wheel: Wheel) -> None:
    """Updates the installed SDK and public Bazel source pins together."""
    sdk_path = root / SDK_PIN
    module_path = root / MODULE
    current = _require_mapping(
        json.loads(sdk_path.read_text(encoding="utf-8")), "SDK pin"
    )
    expected_keys = {
        "repository",
        "revision",
        "run_id",
        "artifact_id",
        "wheel",
        "sha256",
        "catalog",
    }
    if set(current) != expected_keys or current.get("repository") != HRX_REPOSITORY:
        raise ValueError("sdk.json has an unexpected schema or repository")
    old_revision = current.get("revision")
    if not isinstance(old_revision, str) or _COMMIT.fullmatch(old_revision) is None:
        raise ValueError("sdk.json has an invalid source revision")

    module = module_path.read_text(encoding="utf-8")
    module_revisions = re.findall(r'commit\s*=\s*"([0-9a-f]{40})"', module)
    if module_revisions != [old_revision]:
        raise ValueError("MODULE.bazel and sdk.json must select the same HRX revision")
    old_assignment = f'commit = "{old_revision}"'
    new_assignment = f'commit = "{candidate.revision}"'
    if module.count(old_assignment) != 1:
        raise ValueError("MODULE.bazel has an unexpected HRX override")

    updated = {
        "repository": HRX_REPOSITORY,
        "revision": candidate.revision,
        "run_id": candidate.run_id,
        "artifact_id": candidate.artifact_id,
        "wheel": wheel.filename,
        "sha256": wheel.digest,
        "catalog": wheel.catalog,
    }
    sdk_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    module_path.write_text(
        module.replace(old_assignment, new_assignment), encoding="utf-8"
    )


def _write_github_outputs(candidate: Candidate, wheel: Wheel) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(f"run_id={candidate.run_id}\n")
        stream.write(f"revision={candidate.revision}\n")
        stream.write(f"version={wheel.version}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", help="successful Loom / SDK workflow run ID or GitHub URL"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        candidate = fetch_candidate(parse_run_id(args.run))
        with tempfile.TemporaryDirectory(prefix="loom-sdk-candidate-") as temporary:
            artifact_path = Path(temporary) / "artifact.zip"
            _download_artifact(candidate.artifact_id, artifact_path)
            wheel = inspect_artifact(artifact_path)
        update_pin(args.root, candidate, wheel)
        _write_github_outputs(candidate, wheel)
    except (
        BadZipFile,
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        parser.exit(1, f"SDK candidate selection failed: {error}\n")
    print(
        f"Selected hrx-loom {wheel.version} from workflow run {candidate.run_id}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
