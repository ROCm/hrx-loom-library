#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Downloads the exact pinned SDK from its candidate or immutable release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


SDK_PIN = Path("build_tools/ci/sdk.json")
MODULE = Path("MODULE.bazel")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_WHEEL = re.compile(r"hrx_loom-(?P<version>[^/]+)-py3-none-manylinux_2_39_x86_64\.whl")


@dataclass(frozen=True)
class Pin:
    repository: str
    revision: str
    run_id: int
    artifact_id: int
    release_tag: str | None
    wheel: str
    digest: str


def read_pin(root: Path) -> Pin:
    document = json.loads((root / SDK_PIN).read_text(encoding="utf-8"))
    expected = {
        "repository",
        "revision",
        "run_id",
        "artifact_id",
        "release_tag",
        "wheel",
        "sha256",
        "catalog",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("sdk.json has an unexpected schema")
    repository = document["repository"]
    revision = document["revision"]
    run_id = document["run_id"]
    artifact_id = document["artifact_id"]
    release_tag = document["release_tag"]
    wheel = document["wheel"]
    digest = document["sha256"]
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise ValueError("SDK repository is invalid")
    if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
        raise ValueError("SDK revision is invalid")
    if type(run_id) is not int or run_id <= 0:
        raise ValueError("SDK workflow run is invalid")
    if type(artifact_id) is not int or artifact_id <= 0:
        raise ValueError("SDK workflow artifact is invalid")
    wheel_match = _WHEEL.fullmatch(wheel) if isinstance(wheel, str) else None
    if wheel_match is None:
        raise ValueError("SDK wheel name is invalid")
    version = wheel_match.group("version")
    expected_release_tag = None if ".dev" in version else "v" + version
    if release_tag != expected_release_tag:
        raise ValueError("SDK release tag differs from its wheel version")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError("SDK wheel digest is invalid")
    module = (root / MODULE).read_text(encoding="utf-8")
    if re.findall(r'commit\s*=\s*"([0-9a-f]{40})"', module) != [revision]:
        raise ValueError("SDK and MODULE.bazel must select the same HRX revision")
    return Pin(repository, revision, run_id, artifact_id, release_tag, wheel, digest)


def _get_optional_release(pin: Pin, token: str) -> Any | None:
    tag = urllib.parse.quote(pin.release_tag or "", safe="")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{pin.repository}/releases/tags/{tag}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def _url_origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.casefold()
    return (
        scheme,
        parsed.hostname.casefold() if parsed.hostname else None,
        parsed.port or {"http": 80, "https": 443}.get(scheme),
    )


class _GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, url):
        redirected = super().redirect_request(
            request, file_pointer, code, message, headers, url
        )
        if redirected is not None and _url_origin(request.full_url) != _url_origin(url):
            redirected.remove_header("Authorization")
        return redirected


def _download(
    endpoint: str,
    output: Path,
    token: str,
    *,
    accept: str = "application/vnd.github+json",
) -> None:
    request = urllib.request.Request(
        "https://api.github.com/" + endpoint,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    opener = urllib.request.build_opener(_GitHubRedirectHandler())
    with opener.open(request, timeout=60) as response, output.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)


def _release_asset(pin: Pin, document: Any) -> int:
    if (
        not isinstance(document, dict)
        or document.get("tag_name") != pin.release_tag
        or document.get("draft") is not False
        or document.get("immutable") is not True
        or not isinstance(document.get("assets"), list)
    ):
        raise ValueError("SDK release is not an exact immutable public release")
    matches = [
        asset
        for asset in document["assets"]
        if isinstance(asset, dict) and asset.get("name") == pin.wheel
    ]
    if len(matches) != 1:
        raise ValueError("SDK release does not contain the pinned wheel")
    asset = matches[0]
    asset_id = asset.get("id")
    if (
        type(asset_id) is not int
        or asset_id <= 0
        or asset.get("state") != "uploaded"
        or asset.get("digest") != "sha256:" + pin.digest
    ):
        raise ValueError("SDK release wheel differs from the pin")
    return asset_id


def download(root: Path, output: Path, token: str) -> tuple[Path, str]:
    """Downloads and verifies the selected wheel, returning its source kind."""
    if not token:
        raise ValueError("SDK download requires GH_TOKEN")
    pin = read_pin(root)
    release_document = (
        _get_optional_release(pin, token) if pin.release_tag is not None else None
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="loom-sdk-", dir=output) as temporary:
        temporary_root = Path(temporary)
        candidate = temporary_root / pin.wheel
        if release_document is not None:
            asset_id = _release_asset(pin, release_document)
            _download(
                f"repos/{pin.repository}/releases/assets/{asset_id}",
                candidate,
                token,
                accept="application/octet-stream",
            )
            source = "release"
        else:
            artifact = temporary_root / "artifact.zip"
            _download(
                f"repos/{pin.repository}/actions/artifacts/{pin.artifact_id}/zip",
                artifact,
                token,
            )
            with ZipFile(artifact) as archive:
                members = [
                    member for member in archive.infolist() if not member.is_dir()
                ]
                if (
                    len(members) != 1
                    or members[0].filename != pin.wheel
                    or Path(members[0].filename).name != members[0].filename
                ):
                    raise ValueError("SDK artifact does not contain the pinned wheel")
                candidate.write_bytes(archive.read(members[0]))
            source = "candidate"
        with candidate.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != pin.digest:
            raise ValueError(f"{source} SDK wheel differs from the pin")
        destination = output / pin.wheel
        candidate.replace(destination)
    return destination, source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        wheel, source = download(args.root, args.output, os.environ.get("GH_TOKEN", ""))
    except (
        BadZipFile,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        parser.exit(1, f"SDK download failed: {error}\n")
    values = f"wheel={wheel.name}\nsource={source}\n"
    print(f"Installed SDK source: {source}")
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write(values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
