# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Exact significand rounding for finite numerical fixture generation."""

import math
import struct
from fractions import Fraction


def round_binary(
    value: Fraction, precision: int, minimum_exponent: int | None = None
) -> Fraction:
    """Round a finite rational to a binary significand, ties to even.

    With a minimum normal exponent, subnormals use fixed spacing down to zero.
    Otherwise the exponent is unbounded. This oracle expresses significand
    rounding, rather than simulating host instructions; overflow is excluded
    by the authored corpora.
    """
    if not value:
        return value
    sign = 1 if value > 0 else -1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = Fraction(2) ** exponent
    if value < power:
        exponent -= 1
    if minimum_exponent is not None:
        exponent = max(exponent, minimum_exponent)
    spacing = Fraction(2) ** (exponent - precision + 1)
    scaled = value / spacing
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = remainder * 2
    if twice > scaled.denominator or (
        twice == scaled.denominator and quotient % 2
    ):
        quotient += 1
    return sign * quotient * spacing


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def rounded_float(value: Fraction | float, precision: int) -> float:
    """Round finite F32/BF16 arithmetic, preserving zero's sign and subnormals."""
    result = float(round_binary(Fraction(value), precision, minimum_exponent=-126))
    return math.copysign(0.0, float(value)) if result == 0 else result


def add_f32(left: float, right: float) -> float:
    exact = Fraction(left) + Fraction(right)
    return rounded_float(exact if exact else left + right, 24)


def multiply_f32(left: float, right: float) -> float:
    exact = Fraction(left) * Fraction(right)
    return rounded_float(exact if exact else left * right, 24)


def round_binary_sqrt(value: Fraction, precision: int) -> Fraction:
    """Round a nonnegative rational's normal square root, ties to even.

    Comparing squared midpoints keeps the rounding decision exact without
    depending on a host square-root instruction or floating-point library.
    """
    if not value:
        return value
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    spacing = Fraction(2) ** (exponent // 2 - precision + 1)
    scaled_square = value / (spacing * spacing)
    lower = math.isqrt(scaled_square.numerator // scaled_square.denominator)
    midpoint_square = Fraction((2 * lower + 1) ** 2, 4)
    if scaled_square > midpoint_square or (
        scaled_square == midpoint_square and lower % 2
    ):
        lower += 1
    return lower * spacing
