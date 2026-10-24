# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Exact significand rounding for finite numerical fixture generation."""

import struct
from fractions import Fraction


def round_binary(value: Fraction, precision: int) -> Fraction:
    """Round a finite rational to a normal binary significand, ties to even.

    The authored corpora stay far from overflow and underflow. This oracle
    expresses significand rounding, rather than simulating host instructions.
    """
    if not value:
        return value
    sign = 1 if value > 0 else -1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = Fraction(2) ** exponent
    if value < power:
        exponent -= 1
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
