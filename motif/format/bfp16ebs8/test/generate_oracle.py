#!/usr/bin/env python3
"""Refresh exact BFP16EBS8 records without using a native encoder.

Inputs are explicit BF16 bit patterns. Rational arithmetic selects the shared
exponent and rounds signed mantissas to nearest, ties to even. The residual is
the exact difference from the represented high value; no model output supplies
an expected byte. Run this script from any working directory.
"""

from fractions import Fraction
from pathlib import Path
import struct


def power_of_two(exponent: int) -> Fraction:
    return Fraction(2**exponent) if exponent >= 0 else Fraction(1, 2**-exponent)


def bf16_value(bits: int) -> Fraction:
    exponent = (bits >> 7) & 255
    if exponent == 0:
        return Fraction(0)
    if exponent == 255:
        raise ValueError("Nonfinite operands are outside the packing contract")
    sign = -1 if bits & 0x8000 else 1
    return sign * (128 + (bits & 127)) * power_of_two(exponent - 134)


def exponent_bits(value: Fraction) -> int:
    if not value:
        return 0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < power_of_two(exponent):
        exponent -= 1
    return exponent + 127


def encode_group(values: list[Fraction]) -> tuple[bytes, list[Fraction]]:
    # The native floating path flushes subnormal operands/intermediates.
    values = [
        value if abs(value) >= power_of_two(-126) else Fraction(0)
        for value in values
    ]
    exponent = max(map(exponent_bits, values))
    scale = power_of_two(exponent - 133)
    mantissas = [round(value / scale) for value in values]
    if max(mantissas) > 127:
        exponent += 1
        scale *= 2
        mantissas = [round(value / scale) for value in values]
    if not all(-128 <= value <= 127 for value in mantissas):
        raise ValueError("Mantissa overflow")
    encoded = bytes([exponent, *(value & 255 for value in mantissas)])
    return encoded, [value * scale for value in mantissas]


def encode_block(words: list[int]) -> tuple[bytes, bytes]:
    high = bytearray()
    residual = bytearray()
    for position in range(0, 64, 8):
        values = [bf16_value(word) for word in words[position : position + 8]]
        packed, represented = encode_group(values)
        high.extend(packed)
        low, _ = encode_group(
            [value - approximation for value, approximation in zip(values, represented)]
        )
        residual.extend(low)
    return bytes(high), bytes(residual)


def input_blocks() -> list[list[int]]:
    groups = [
        [0, 0x8000, 1, 0x8001, 0x7F, 0x807F, 0, 0],
        [0x3F80, 0xBF80, 0x4000, 0xC000, 0x3F00, 0xBF00, 0, 0x8000],
        [0x3F81, 0x3F83, 0x3F85, 0x3F87, 0xBF81, 0xBF83, 0xBF85, 0xBF87],
        [0x3FFF, 0x3FFE, 0x3FFD, 0x3FFC, 0xBFFF, 0xBFFE, 0xBFFD, 0xBFFC],
        [0xBFFF, 0xBFFE, 0xBFFD, 0xBFFC, 0xBF81, 0xBF83, 0xBF85, 0xBF87],
        [0x3F80, 0x3F00, 0x3E80, 0x3C80, 0x3B80, 0xBC80, 0xBB80, 0xBB00],
        [0x7F00, 0x7F01, 0x7F7C, 0x7F7E, 0xFF00, 0xFF01, 0xFF7C, 0xFF7E],
        [0x7EFF, 0xFEFF, 0x0080, 0x8080, 0x3F83, 0xBF83, 0x007F, 0x807F],
    ]
    for exponent in (1, 7, 31, 63, 95, 126, 127, 128, 159, 191, 223, 247, 253):
        groups.append(
            [
                (exponent << 7) | mantissa | sign
                for mantissa, sign in (
                    (0, 0), (1, 0), (63, 0), (127, 0),
                    (0, 0x8000), (1, 0x8000), (63, 0x8000), (127, 0x8000),
                )
            ]
        )
    blocks = [[0] * 64, [0x3F80] * 64, [0xBF80] * 64]
    for start in range(len(groups)):
        blocks.append(
            [word for lane in range(8) for word in groups[(start + lane) % len(groups)]]
        )
    return blocks


def main() -> None:
    # Analytically known records establish exponent, parity, and signed carry.
    assert encode_group([Fraction(1)] * 8)[0] == bytes([127] + [64] * 8)
    assert encode_group([Fraction(255, 128)] * 8)[0] == bytes([128] + [64] * 8)
    assert encode_group([Fraction(-255, 128)] * 8)[0] == bytes([127] + [128] * 8)
    midpoint = [Fraction(value, 128) for value in (129, 131, 133, 135, -129, -131, -133, -135)]
    assert encode_group(midpoint)[0] == bytes([127, 64, 66, 66, 68, 192, 190, 190, 188])
    blocks = input_blocks()
    assert len(blocks) == 24
    inputs = b"".join(struct.pack("<64H", *block) for block in blocks)
    encoded = [encode_block(block) for block in blocks]
    high = b"".join(record[0] for record in encoded)
    residual = b"".join(record[1] for record in encoded)
    path = Path(__file__).with_name("aie2p.loom")
    source = path.read_text()
    start = source.index("// BEGIN GENERATED ORACLE")
    end = source.index("// END GENERATED ORACLE", start) + len("// END GENERATED ORACLE")
    generated = "\n".join([
        "// BEGIN GENERATED ORACLE",
        f"// {len(blocks)} complete blocks; refresh with generate_oracle.py.",
        f'global.rodata.def @input_words = align(64) bytes("{inputs.hex()}")',
        f'global.rodata.def @high_records = align(64) bytes("{high.hex()}")',
        f'global.rodata.def @residual_records = align(64) bytes("{residual.hex()}")',
        "// END GENERATED ORACLE",
    ])
    path.write_text(source[:start] + generated + source[end:])
    print(f"Generated {len(blocks)} input blocks and {len(encoded) * 2} exact records")


if __name__ == "__main__":
    main()
