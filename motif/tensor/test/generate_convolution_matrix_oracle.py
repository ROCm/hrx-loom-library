#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author BF16/F32 convolution checks against exact, scale-bounded references."""

import argparse
import random
from fractions import Fraction
from pathlib import Path

from binary_reference import round_binary
from convolution_fixture import Case, render_fixture


CASES = (
    Case("minimum", 1, 1, 32, 32, extent=3, threads=128),
    Case("spatial_tail", 3, 5, 32, 64, extent=3, planes=3, threads=128, residual=True),
    Case("complete_tile", 4, 8, 64, 32, extent=3, threads=128, residual=True, alias=True),
    Case("nearest_double", 3, 5, 32, 32, extent=3, upsample=2, threads=128),
    Case("nearest_triple", 1, 2, 64, 64, extent=3, planes=2,
         upsample=3, threads=128, residual=True, alias=True),
    Case("wide_reduction", 1, 1, 384, 32, extent=3, threads=128),
    Case("operand_rounding", 1, 5, 32, 32, extent=3, threads=128),
    Case("f32_accumulation", 1, 1, 32, 32, extent=3, threads=128),
    Case("residual_after_bias", 1, 1, 32, 32, extent=3, threads=128, residual=True, alias=True),
)


def fixtures(case):
    source = random.Random(0xC03 + case.height * 37 + case.inputs * 13 + case.outputs)
    values = [source.randrange(-7, 8) / 16
              for _ in range(case.inputs * case.height * case.width)]
    active = [source.randrange(-7, 8) / 16
              for _ in range(case.outputs * case.inputs * 9)]
    bias = [source.randrange(-7, 8) / 8 for _ in range(case.outputs)]
    pixels = case.height * case.width * case.upsample**2
    residual = [source.randrange(-7, 8) / 8 for _ in range(case.outputs * pixels)]
    if case.name in ("operand_rounding", "f32_accumulation", "residual_after_bias"):
        active = [0.0] * len(active)
        bias = [0.0] * len(bias)
        if case.name == "operand_rounding":
            # Both midpoint parities and signs exercise each BF16 conversion.
            midpoints = (1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8)
            values = [midpoints[i % 4] for i in range(len(values))]
            for output in range(case.outputs):
                channel = output % case.inputs
                active[(output * case.inputs + channel) * 9 + 4] = midpoints[(output + 1) % 4]
        elif case.name == "f32_accumulation":
            values = [1.0] + [2**-12] * (len(values) - 1)
            for output in range(case.outputs):
                for channel in range(case.inputs):
                    active[(output * case.inputs + channel) * 9 + 4] = 1.0
        else:
            values = [1.0] + [0.0] * (len(values) - 1)
            bias = [1.0] * len(bias)
            residual = [float(-2**24)] * len(residual)
            for output in range(case.outputs):
                active[output * case.inputs * 9 + 4] = float(2**24)

    # Expand physical input first. The oracle indexes the padded logical
    # image directly, independently of the production coordinate transform.
    converted = [round_binary(Fraction(value), 8) for value in values]
    coefficients = [round_binary(Fraction(value), 8) for value in active]
    expanded_width = case.width * case.upsample
    padded = []
    for channel in range(case.inputs):
        rows = [[Fraction(0)] * (expanded_width + 2)]
        for y in range(case.height):
            row = [Fraction(0)]
            for x in range(case.width):
                row.extend([converted[(channel * case.height + y) * case.width + x]] * case.upsample)
            row.append(Fraction(0))
            rows.extend([row] * case.upsample)
        rows.append([Fraction(0)] * (expanded_width + 2))
        padded.append(rows)

    weight = []
    for output in range(case.outputs):
        for channel in range(case.inputs):
            begin = (output * case.inputs + channel) * 9
            for plane in range(case.planes):
                weight.extend(active[begin:begin + 9] if plane == case.planes - 1
                              else [float(1000 + plane + channel)] * 9)

    expected = []
    maximum_contribution_scale = Fraction(0)
    for output in range(case.outputs):
        for y in range(case.height * case.upsample):
            for x in range(expanded_width):
                total = Fraction(0)
                magnitude = Fraction(0)
                for channel in range(case.inputs):
                    begin = (output * case.inputs + channel) * 9
                    for ky in range(3):
                        for kx in range(3):
                            value = padded[channel][y + ky][x + kx]
                            coefficient = coefficients[begin + ky * 3 + kx]
                            if value and coefficient:
                                total += value * coefficient
                                magnitude += abs(value * coefficient)
                # Ordinary values are multiples of 1/16, so even the largest
                # partial sum is bounded by 3456*49/256, exactly within F32.
                # Native WMMA can still differ from this exact evaluation.
                assert round_binary(total, 24) == total
                total = round_binary(total + Fraction(bias[output]), 24)
                magnitude += abs(Fraction(bias[output]))
                if case.residual:
                    total = round_binary(total + Fraction(residual[len(expected)]), 24)
                    magnitude += abs(Fraction(residual[len(expected)]))
                expected.append(float(total))
                maximum_contribution_scale = max(maximum_contribution_scale, magnitude)
    if case.name == "f32_accumulation":
        assert expected == [1.0 + 31 * 2**-12] * case.outputs
        assert round_binary(Fraction(expected[0]), 11) != Fraction(expected[0])
    elif case.name == "residual_after_bias":
        assert expected == [0.0] * case.outputs
    # Demand twenty significant binary bits relative to the sum of absolute
    # contributions, twelve beyond the eight-bit BF16 operands. This is an
    # accuracy requirement on the complete contraction, not an emulation of
    # native dot-product ordering or a bound inferred from device output.
    # Using an input scale keeps cancellation from making the budget vanish.
    tolerance = maximum_contribution_scale / 2**20
    if case.name in ("f32_accumulation", "residual_after_bias"):
        tolerance = None
    else:
        # Ordinary nonzero products are multiples of 1/256. Even the largest
        # budget is below half that unit, so a lost minimal product cannot pass.
        assert tolerance < Fraction(1, 512)
        tolerance = float(tolerance)
    return {"input": values, "weight": weight, "bias": bias, "residual": residual}, expected, tolerance


def render_kernel(case):
    pixels = case.height * case.width * case.upsample**2
    groups = (pixels + 31) // 32
    channel_groups = case.outputs // 32
    output_argument = "" if case.alias else ", %output: buffer"
    output = "%residual" if case.alias else "%output"
    return f"""kernel.def target(@convolution_rdna3_wave32) @convolve_{case.name}() {{
  %groups = index.constant {groups} : index
  %channels = index.constant {channel_groups} : index
  %threads = index.constant 128 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %channels, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer, %bias: buffer, %residual: buffer{output_argument}) {{
  %height = index.constant {case.height} : index
  %width = index.constant {case.width} : index
  %inputs = index.constant {case.inputs} : index
  %outputs = index.constant {case.outputs} : index
  %planes = index.constant {case.planes} : index
  %upsample = index.constant {case.upsample} : index
  %add_residual = scalar.constant {str(case.residual).lower()} : i1
  template.apply<@image.convolution_3x3_matrix_f32>(%height, %width, %inputs, %outputs, %planes, %upsample, %add_residual, %input, %weight, %bias, %residual, {output}) : (index, index, index, index, index, index, i1, buffer, buffer, buffer, buffer, buffer)
  kernel.return
}}"""


def render():
    header = """// Generated by generate_convolution_matrix_oracle.py.
// Logical padded-image references quantize operands with exact BF16 rounding.
// Native results have a twenty-bit input-scaled error budget; precision and
// bias/residual rounding constructions, guards, and immutable inputs are exact.

amdgpu.target<gfx11-generic> @tensor_convolution_rdna3

amdgpu.target<gfx11-generic> @convolution_rdna3_wave32 {subgroup_size = 32}

template.decl @image.convolution_3x3_matrix_f32(%height: index, %width: index, %inputs: index, %outputs: index, %planes: index, %upsample: index, %add_residual: i1, %input: buffer, %weight: buffer, %bias: buffer, %residual: buffer, %output: buffer)
"""
    parts = []
    for case in CASES:
        arrays, expected, tolerance = fixtures(case)
        parts.append(render_fixture(case, arrays, expected, render_kernel(case),
                                    absolute_tolerance=tolerance))
    return header + "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("convolution_matrix_test.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its authored oracle")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
