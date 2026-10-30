#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Produces compact NPY observations of CPython's Unicode 15.0 behavior."""

import argparse
import struct
import unicodedata
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if unicodedata.unidata_version != "15.0.0":
        raise SystemExit("The oracle requires Unicode 15.0.0 (CPython 3.12)")
    points = set()
    for edge in (0, 0x7F, 0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF):
        points.update((edge - 1, edge, edge + 1))
    for point in range(0x110000):
        character = chr(point)
        if character.upper() != character or character.isspace() or character.isdigit():
            points.update((point - 1, point, point + 1))
    rows = []
    for point in sorted(points):
        if not 0 <= point <= 0x10FFFF or 0xD800 <= point <= 0xDFFF:
            continue
        character = chr(point)
        uppercase = list(map(ord, character.upper()))
        rows.append(
            (
                point,
                *uppercase,
                *([0] * (3 - len(uppercase))),
                int(character.isspace()),
                int(character.isdigit()),
            )
        )
    uppercase = sorted({(a - point, b, c) for point, a, b, c, _, _ in rows})
    indices = {record: index for index, record in enumerate(uppercase)}
    records = [
        (point | (space << 21) | (digit << 22), indices[a - point, b, c])
        for point, a, b, c, space, digit in rows
    ]
    for name, values, width in (
        ("unicode_records.npy", records, 2),
        ("unicode_uppercase.npy", uppercase, 3),
    ):
        data = b"".join(struct.pack("<" + "i" * width, *row) for row in values)
        header = str(
            {"descr": "<i4", "fortran_order": False, "shape": (len(values), width)}
        ).encode("ascii")
        header += b" " * ((-len(header) - 11) % 64) + b"\n"
        (args.output_directory / name).write_bytes(
            b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data
        )


if __name__ == "__main__":
    main()
