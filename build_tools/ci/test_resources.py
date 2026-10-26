#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Produces a Bazel test tag filter from query XML and available resources."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET


REQUIREMENT_PREFIX = "iree-run-requirement="


def test_tag_filters(query: ET.Element, available: set[str]) -> str:
    """Selects resource-free tests, or hardware tests needing only available resources."""
    if query.tag != "query":
        raise ValueError("Expected the XML output of bazel query 'tests(//...)'")

    requirements = []
    for rule in query.findall("rule"):
        tags = {
            entry.attrib["value"]
            for entry in rule.findall("list[@name='tags']/string")
        }
        if "manual" not in tags:
            requirements.append(
                {
                    tag.removeprefix(REQUIREMENT_PREFIX)
                    for tag in tags
                    if tag.startswith(REQUIREMENT_PREFIX)
                }
            )

    declared = set().union(*requirements)
    unknown = available - declared
    if unknown:
        raise ValueError(
            "Resources absent from the test graph: " + ", ".join(sorted(unknown))
        )
    if not any(
        required <= available and (not available or required)
        for required in requirements
    ):
        raise ValueError("No tests match the available resources")

    # Bazel ORs positive tags and rejects a test matching any negative tag.
    # Exclude every unavailable resource so combined requirements remain conjunctive.
    filters = ["-manual"]
    filters.extend(REQUIREMENT_PREFIX + resource for resource in sorted(available))
    filters.extend(
        "-" + REQUIREMENT_PREFIX + resource
        for resource in sorted(declared - available)
    )
    return "--test_tag_filters=" + ",".join(filters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="XML from bazel query 'tests(//...)'.")
    parser.add_argument(
        "--available",
        nargs="*",
        default=[],
        help="Run requirement IDs available on this runner.",
    )
    args = parser.parse_args()
    try:
        filters = test_tag_filters(ET.parse(args.query).getroot(), set(args.available))
    except (OSError, ET.ParseError, ValueError) as error:
        parser.error(str(error))
    print(filters)


if __name__ == "__main__":
    main()
