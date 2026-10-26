#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Checks published discovery metadata against its admitted model cards."""

import argparse
from pathlib import Path

from loom.model.catalog import ModelCatalog, generate
from loom.model.manifest import ModelCard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--card", action="append", required=True)
    args = parser.parse_args()
    cards = {}
    for value in args.card:
        published, separator, local = value.partition("=")
        if not separator or not local or published in cards:
            parser.error("--card requires a unique PUBLISHED_PATH=LOCAL_JSON")
        cards[published] = ModelCard.parse(Path(local).read_text(encoding="utf-8"))
    actual = ModelCatalog.parse(args.catalog.read_text(encoding="utf-8"))
    expected = generate(cards)
    if actual.document != expected.document:
        raise AssertionError(
            "catalog.json differs from its admitted cards; regenerate it with "
            "python -m loom.model.catalog and the declared --card arguments"
        )


if __name__ == "__main__":
    main()
