#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Publishes a library tree through its qualified SDK's catalog tag."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


API_VERSION = "2026-03-10"
HRX_REPOSITORY = "ROCm/hrx-system"
SDK_PIN = Path("build_tools/ci/sdk.json")
CATALOG = Path("catalog.json")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = re.compile(rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}(?:(?:a|b|rc){_NUMBER})?")
_WHEEL = re.compile(r"hrx_loom-(?P<version>[^/]+)-py3-none-manylinux_2_39_x86_64\.whl")


@dataclass(frozen=True)
class Identity:
    # SDK version qualified by this exact library tree.
    version: str
    # Compatible library source and release tag.
    tag: str
    # Whether GitHub presents the SDK and library as prereleases.
    prerelease: bool
    # Exact qualified Linux Release wheel.
    wheel: str
    # SHA-256 of the complete qualified wheel archive.
    digest: str
    # Default catalog association authored into the qualified wheel.
    catalog: str


def read_identity(root: Path) -> Identity:
    """Derives the library release identity from its reviewed SDK pin."""
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
    if document["repository"] != HRX_REPOSITORY:
        raise ValueError(f"SDK source must be {HRX_REPOSITORY}")
    if (
        not isinstance(document["revision"], str)
        or _COMMIT.fullmatch(document["revision"]) is None
    ):
        raise ValueError("SDK source revision must be a Git commit")
    if (
        type(document["run_id"]) is not int
        or document["run_id"] <= 0
        or type(document["artifact_id"]) is not int
        or document["artifact_id"] <= 0
    ):
        raise ValueError("SDK workflow and artifact IDs must be positive integers")
    digest = document["sha256"]
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError("SDK wheel digest must be SHA-256")
    wheel = document["wheel"]
    match = _WHEEL.fullmatch(wheel) if isinstance(wheel, str) else None
    version = match.group("version") if match is not None else ""
    if _VERSION.fullmatch(version) is None:
        raise ValueError("library publication requires a release SDK wheel")
    tag = "v" + version
    if document["release_tag"] != tag:
        raise ValueError("qualified SDK does not name its matching release tag")
    catalog = document["catalog"]
    if catalog != f"github:ROCm/hrx-loom-library@{tag}":
        raise ValueError("qualified SDK does not name this coordinated library tag")
    return Identity(
        version,
        tag,
        bool(re.search(r"(?:a|b|rc)[0-9]+$", version)),
        wheel,
        digest,
        catalog,
    )


def release_body(identity: Identity) -> str:
    return (
        f"Loom Library {identity.version} contains the admitted model cards, "
        "programs, and reusable components qualified with the exact Loom SDK "
        f"{identity.version} candidate. The matching SDK release uses this tag "
        "for default model discovery."
    )


def _get_optional(url: str, token: str) -> Any | None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def _gh(*arguments: str) -> None:
    subprocess.run(["gh", *arguments], check=True)


def _validate_tag(document: Any, commit: str) -> None:
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("object"), dict)
        or document["object"].get("type") != "commit"
        or document["object"].get("sha") != commit
    ):
        raise ValueError("existing library tag does not select this commit")


def _validate_library_release(document: Any, identity: Identity) -> tuple[int, bool]:
    expected = {
        "tag_name": identity.tag,
        "name": f"Loom Library {identity.tag}",
        "body": release_body(identity),
        "prerelease": identity.prerelease,
    }
    if not isinstance(document, dict) or any(
        document.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("existing library release has conflicting metadata")
    release_id = document.get("id")
    draft = document.get("draft")
    if (
        type(release_id) is not int
        or release_id <= 0
        or type(draft) is not bool
        or document.get("assets") != []
    ):
        raise ValueError("GitHub returned an invalid library release state")
    if not draft and document.get("immutable") is not True:
        raise ValueError(
            "published library release is mutable; use a new version after "
            "enabling repository release immutability"
        )
    return release_id, draft


def _read_catalog(loom_fetch: Path) -> Any:
    """Lists the installed SDK's implicit catalog from a fresh content cache."""
    with tempfile.TemporaryDirectory(prefix="loom-release-catalog-") as temporary:
        result = subprocess.run(
            [
                str(loom_fetch),
                "--list",
                "--json",
                "--no-input",
                f"--cache={temporary}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    return json.loads(result.stdout)


def _validate_catalog(
    root: Path, commit: str, identity: Identity, document: Any
) -> None:
    """Requires default discovery to reproduce the tagged admitted catalog."""
    if not isinstance(document, dict):
        raise ValueError("installed SDK returned an invalid catalog listing")
    listing = dict(document)
    snapshot = listing.pop("snapshot", None)
    expected = json.loads((root / CATALOG).read_text(encoding="utf-8"))
    if listing != expected:
        raise ValueError("installed SDK catalog differs from the admitted catalog")
    expected_snapshot = {
        "source": identity.catalog,
        "location": (
            "https://raw.githubusercontent.com/ROCm/hrx-loom-library/"
            f"{commit}/catalog.json"
        ),
        "revision": commit,
    }
    if snapshot != expected_snapshot:
        raise ValueError("installed SDK did not resolve the coordinated library tag")


def publish(
    root: Path,
    repository: str,
    commit: str,
    identity: Identity,
    token: str,
    loom_fetch: Path,
) -> None:
    """Proves and publishes one tagged library tree with its candidate SDK."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or any(
        part in (".", "..") for part in repository.split("/")
    ):
        raise ValueError("GitHub repository must be OWNER/REPOSITORY")
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("library release commit must be a full Git commit")
    if not token:
        raise ValueError("library publication requires GH_TOKEN")
    if not loom_fetch.is_file():
        raise ValueError("library publication requires the installed loom-fetch")

    tag = urllib.parse.quote(identity.tag, safe="")
    api = f"https://api.github.com/repos/{repository}"
    tag_url = f"{api}/git/ref/tags/{tag}"
    tag_document = _get_optional(tag_url, token)
    if tag_document is None:
        _gh(
            "api",
            "--method",
            "POST",
            f"repos/{repository}/git/refs",
            "--field",
            f"ref=refs/tags/{identity.tag}",
            "--field",
            f"sha={commit}",
        )
        tag_document = _get_optional(tag_url, token)
    _validate_tag(tag_document, commit)

    release_url = f"{api}/releases/tags/{tag}"
    release_document = _get_optional(release_url, token)
    created = release_document is None
    if created:
        options = ["--prerelease", "--latest=false"] if identity.prerelease else []
        _gh(
            "release",
            "create",
            identity.tag,
            "--repo",
            repository,
            "--verify-tag",
            "--draft",
            "--title",
            f"Loom Library {identity.tag}",
            "--notes",
            release_body(identity),
            *options,
        )
        release_document = _get_optional(release_url, token)

    release_id, draft = _validate_library_release(release_document, identity)
    if created and not draft:
        raise ValueError("new library release did not remain a draft")

    _validate_catalog(root, commit, identity, _read_catalog(loom_fetch))

    if draft:
        _gh("release", "edit", identity.tag, "--repo", repository, "--draft=false")
        published = _get_optional(release_url, token)
        published_id, published_draft = _validate_library_release(published, identity)
        if published_id != release_id or published_draft:
            raise ValueError("library release did not reach its immutable public state")
        print(f"Published immutable library release {identity.tag}.")
    else:
        print(f"Library release {identity.tag} is already immutable.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--loom-fetch", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.repository or not args.commit:
        parser.error("repository and commit require GitHub Actions or explicit values")
    try:
        publish(
            args.root,
            args.repository,
            args.commit,
            read_identity(args.root),
            os.environ.get("GH_TOKEN", ""),
            args.loom_fetch,
        )
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        parser.exit(1, f"Library publication failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
