#!/usr/bin/env python3
"""Refresh BF16 activation and rounded-update observations.

Activations use binary64 libm formulas, independently of native F32 recipes.
Residual and Euler observations use exact rational arithmetic and integer
round-to-nearest-even. The Euler contract rounds its product first to F32,
then to BF16; rounding the time step itself is a different operation.
"""

import math
import random
import struct
from fractions import Fraction
from pathlib import Path

from binary_reference import f32, round_binary


def bf16(value: float) -> float:
    return float(round_binary(Fraction(value), 8))


def adjacent_f32(value: float, displacement: int) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<f", struct.pack("<I", bits + displacement))[0]


def rounded_update(row: tuple[float, float, float]) -> float:
    sample, velocity, delta = map(Fraction, row)
    product = round_binary(velocity * delta, 24)
    update = round_binary(product, 8)
    addition = round_binary(sample + update, 24)
    return float(round_binary(addition, 8))


def fused_update(row: tuple[float, float, float]) -> float:
    sample, velocity, delta = map(Fraction, row)
    return float(round_binary(sample + velocity * delta, 8))


def main() -> None:
    # Check the oracle's sign, midpoint parity, and carry independently.
    assert round_binary(Fraction(257, 256), 8) == 1
    assert round_binary(Fraction(-257, 256), 8) == -1
    assert round_binary(Fraction(259, 256), 8) == Fraction(130, 128)
    assert round_binary(Fraction(511, 256), 8) == 2

    activations = [value / 8 for value in range(-128, 129)]
    activations += [-2**-20, 2**-20, -2**-8, 2**-8]
    # Adjacent BF16 values around the negative minima of both curves.
    activations += [-1.28125, -1.2734375, -0.75390625, -0.75]
    assert len(activations) == 265
    activation_results = []
    for value in activations:
        assert bf16(value) == value
        silu = value / (1 + math.exp(-value))
        argument = math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)
        gelu = value * (1 + math.tanh(argument)) / 2
        activation_results.extend((silu, gelu))

    residuals = []
    for mantissa in (1, 3, 5, 7, 15, 31, 63, 127):
        for sign in (-1, 1):
            update = sign * (1 + mantissa / 128)
            gate = 1.5
            for sample in (-bf16(update * gate), -0.5, 0.0, 0.5, 1.0):
                residuals.append((sample, update, gate))

    euler = []
    for fraction in (1, 3, 5, 7):
        for sign in (-1, 1):
            delta = sign * (1 + fraction / 1024)
            for sample in (-1.5, -0.5, 0.0, 0.5, 1.5):
                for velocity in (-1.5, 1.5):
                    euler.append((sample, velocity, delta))

    generator = random.Random(0xBF16)
    for _ in range(48):
        sample = bf16(generator.uniform(-8, 8))
        velocity = bf16(generator.uniform(-8, 8))
        residuals.append((sample, velocity, bf16(generator.uniform(-2, 2))))
        euler.append((sample, velocity, f32(generator.uniform(-2, 2))))
    # F32 products immediately below and above a BF16 midpoint can round
    # onto the midpoint before the second rounding, unlike exact products.
    for target in (1 + 1 / 256, 1 + 3 / 256):
        for displacement in (-1, 1):
            euler.append((-1.0, 1.5, adjacent_f32(target / 1.5, displacement)))
    assert len(residuals) == 128 and len(euler) == 132
    assert any(rounded_update(row) != fused_update(row) for row in residuals)
    assert any(rounded_update(row) != fused_update(row) for row in euler)
    assert any(
        rounded_update(row) != rounded_update((row[0], row[1], bf16(row[2])))
        for row in euler
    )
    assert any(
        round_binary(round_binary(Fraction(row[1]) * Fraction(row[2]), 24), 8)
        != round_binary(Fraction(row[1]) * Fraction(row[2]), 8)
        for row in euler
    )

    data = {
        "activation_inputs": activations,
        "activation_expected": activation_results,
        "residual_inputs": [value for row in residuals for value in row],
        "residual_expected": [rounded_update(row) for row in residuals],
        "euler_inputs": [value for row in euler for value in row],
        "euler_expected": [rounded_update(row) for row in euler],
    }
    declarations = []
    for name, values in data.items():
        encoded = struct.pack(f"<{len(values)}f", *values).hex()
        declarations.append(
            f'global.rodata.def @{name} = align(4) bytes("{encoded}")\n\n'
        )
    filename = Path(__file__).with_name("pointwise.loom")
    begin = "// BEGIN GENERATED POINTWISE ORACLE\n"
    end = "// END GENERATED POINTWISE ORACLE"
    prefix, remainder = filename.read_text().split(begin, 1)
    _, suffix = remainder.split(end, 1)
    filename.write_text(prefix + begin + "".join(declarations) + end + suffix)


if __name__ == "__main__":
    main()
