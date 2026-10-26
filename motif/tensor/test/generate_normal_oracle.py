#!/usr/bin/env python3
"""Refresh binary32 Box-Muller observations using double-precision libm.

The uniform input conversion follows the declared F32 half-bin contract.
Logarithm, square root, and trigonometry are evaluated independently of Loom's
approximate target operations, then rounded once to the F32 output format.
"""

import math
import struct
from pathlib import Path


def round_f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def main() -> None:
    radial_words = (
        0, 1, 0x00FFFFFF, 0x01000000, 0x7FFFFFFF, 0x80000000,
        0xFFFFFF7F, 0xFFFFFFFF,
    )
    angular_words = (
        0, 0x20000000, 0x40000000, 0x60000000, 0x80000000,
        0xA0000000, 0xC0000000, 0xFFFFFFFF,
    )
    words = [(radial, angle) for radial in radial_words for angle in angular_words]
    results = []
    for radial, angular in words:
        uniform = min(round_f32(round_f32(radial) + 0.5) * 2**-32, 1 - 2**-24)
        turns = round_f32(round_f32(angular) + 0.5) * 2**-32
        radius = math.sqrt(-2 * math.log(uniform))
        results.append(
            (radius * math.cos(math.tau * turns), radius * math.sin(math.tau * turns))
        )
    inputs = b"".join(struct.pack("<2I", *row) for row in words)
    outputs = b"".join(struct.pack("<2f", *row) for row in results)
    declarations = (
        f'global.rodata.def @normal_words = align(4) bytes("{inputs.hex()}")\n\n'
        f'global.rodata.def @normal_values = align(4) bytes("{outputs.hex()}")\n\n'
    )
    filename = Path(__file__).with_name("random_gpu.loom")
    prefix, rest = filename.read_text().split("// BEGIN GENERATED NORMAL ORACLE\n", 1)
    _, suffix = rest.split("// END GENERATED NORMAL ORACLE", 1)
    filename.write_text(
        prefix + "// BEGIN GENERATED NORMAL ORACLE\n" + declarations
        + "// END GENERATED NORMAL ORACLE" + suffix
    )


if __name__ == "__main__":
    main()
