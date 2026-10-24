#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author exact F32 convolution checks from expanded, padded logical images."""

import argparse
import random
import struct
from fractions import Fraction
from pathlib import Path

from binary_reference import round_binary
from convolution_fixture import Case, render_fixture


CASES = (
    Case("single", 1, 1, 1, 1),
    Case("padded_single", 1, 1, 3, 2, extent=3, planes=3, residual=True),
    Case("rectangle_tail", 3, 5, 3, 4, extent=3, planes=3),
    Case("nearest_double", 2, 3, 2, 3, extent=3, planes=2,
         upsample=2, threads=64, residual=True, alias=True),
    Case("nearest_triple", 2, 1, 2, 2, extent=5, planes=2, upsample=3),
    Case("exact_group", 4, 8, 2, 2, residual=True),
    Case("multiple_groups", 9, 17, 2, 3, threads=128, residual=True, alias=True),
    Case("wide_input", 1, 3, 384, 2, threads=64),
    Case("wide_output", 1, 2, 3, 384, threads=256),
    Case("decoder_output", 2, 2, 96, 3, extent=3, planes=3, threads=128),
    Case("fused_product", 1, 1, 2, 1),
    Case("bias_after_sum", 1, 1, 2, 1),
    Case("residual_after_bias", 1, 1, 2, 1, residual=True, alias=True),
)


def random_float(source):
    bits = (
        (source.getrandbits(1) << 31)
        | ((126 + source.randrange(-2, 3)) << 23)
        | source.getrandbits(23)
    )
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fixtures(case):
    source = random.Random(0xC07 + case.height * 37 + case.inputs * 13 + case.outputs)
    values = [random_float(source) for _ in range(case.inputs * case.height * case.width)]
    coefficient_count = case.outputs * case.inputs * case.extent * case.extent
    active = [random_float(source) for _ in range(coefficient_count)]
    bias = [random_float(source) for _ in range(case.outputs)]
    pixels = case.height * case.width * case.upsample**2
    residual = [random_float(source) for _ in range(case.outputs * pixels)]
    if case.name == "fused_product":
        values, active, bias = [-1.0, 1.0 + 2**-23], [1.0, 1.0 - 2**-23], [0.0]
    elif case.name == "bias_after_sum":
        values, active, bias = [float(2**24), float(-2**24)], [1.0, 1.0], [1.0]
    elif case.name == "residual_after_bias":
        values, active, bias = [float(2**24), 0.0], [1.0, 1.0], [1.0]
        residual = [float(-2**24)]

    # The oracle sees an explicitly expanded and padded image. It never uses
    # the production modulo/divide expression to recover source coordinates.
    padding = case.extent // 2
    expanded_width = case.width * case.upsample
    padded = []
    for channel in range(case.inputs):
        rows = [[0.0] * (expanded_width + 2 * padding) for _ in range(padding)]
        for y in range(case.height):
            row = [0.0] * padding
            for x in range(case.width):
                row.extend([values[(channel * case.height + y) * case.width + x]] * case.upsample)
            row.extend([0.0] * padding)
            rows.extend([row] * case.upsample)
        rows.extend([[0.0] * (expanded_width + 2 * padding) for _ in range(padding)])
        padded.append(rows)

    # Author logical coefficients separately, then embed them in the final
    # temporal plane. Earlier planes contain distinct finite values.
    weight = []
    for output in range(case.outputs):
        for channel in range(case.inputs):
            begin = (output * case.inputs + channel) * case.extent**2
            for plane in range(case.planes):
                if plane == case.planes - 1:
                    weight.extend(active[begin:begin + case.extent**2])
                else:
                    weight.extend([float(1000 + plane + channel)] * case.extent**2)

    expected = []
    for output in range(case.outputs):
        for y in range(case.height * case.upsample):
            for x in range(expanded_width):
                total = Fraction(0)
                for channel in range(case.inputs):
                    begin = (output * case.inputs + channel) * case.extent**2
                    for ky in range(case.extent):
                        for kx in range(case.extent):
                            product = Fraction(padded[channel][y + ky][x + kx])
                            product *= Fraction(active[begin + ky * case.extent + kx])
                            total = round_binary(total + product, 24)
                total = round_binary(total + Fraction(bias[output]), 24)
                if case.residual:
                    total = round_binary(total + Fraction(residual[len(expected)]), 24)
                expected.append(float(total))
    if case.name == "fused_product":
        assert expected == [-2**-46]
    elif case.name == "bias_after_sum":
        assert expected == [1.0]
    elif case.name == "residual_after_bias":
        assert expected == [0.0]
    return {"input": values, "weight": weight, "bias": bias, "residual": residual}, expected


def render_kernel(case):
    name = case.name
    pixels = case.height * case.width * case.upsample**2
    groups = (pixels + case.threads - 1) // case.threads
    output_argument = "" if case.alias else ", %output: buffer"
    output = "%residual" if case.alias else "%output"
    residual = f"""    %residual_view = buffer.view %residual[%origin] : buffer -> view<[%outputs]x[%pixels]xf32>
    %skip = view.load %residual_view[%channel, %pixel] : view<[%outputs]x[%pixels]xf32> -> f32
    %combined = scalar.addf %value, %skip : f32
""" if case.residual else ""
    result = "%combined" if case.residual else "%value"
    return f"""kernel.def target(@convolution_wave32) @convolve_{name}() {{
  %groups = index.constant {groups} : index
  %channels = index.constant {case.outputs} : index
  %threads = index.constant {case.threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %channels, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer, %bias: buffer, %residual: buffer{output_argument}) {{
  %height = index.constant {case.height} : index
  %width = index.constant {case.width} : index
  %inputs = index.constant {case.inputs} : index
  %outputs = index.constant {case.outputs} : index
  %extent = index.constant {case.extent} : index
  %planes = index.constant {case.planes} : index
  %upsample = index.constant {case.upsample} : index
  %pixels = index.constant {pixels} : index
  %threads = index.constant {case.threads} : index
  %origin = index.constant 0 : offset
  %group = kernel.workgroup.id<x> : index
  %channel = kernel.workgroup.id<y> : index
  %lane = kernel.workitem.id<x> : index
  %pixel = index.madd %group, %threads, %lane : index
  %live = index.cmp ult, %pixel, %pixels : index
  scf.if %live {{
    %value = template.apply<@image.convolution_nchw_element_f32>(%height, %width, %inputs, %outputs, %extent, %planes, %upsample, %channel, %pixel, %input, %weight, %bias) : (index, index, index, index, index, index, index, index, index, buffer, buffer, buffer) -> (f32)
{residual}    %output_view = buffer.view {output}[%origin] : buffer -> view<[%outputs]x[%pixels]xf32>
    view.store {result}, %output_view[%channel, %pixel] : f32, view<[%outputs]x[%pixels]xf32>
  }}
  kernel.return
}}"""


def render_case(case):
    arrays, expected = fixtures(case)
    return render_fixture(case, arrays, expected, render_kernel(case))


def render():
    header = """// Generated by generate_convolution_oracle.py.
// The reference explicitly expands/pads images and embeds active coefficients
// in their temporal storage separately. FMA, bias, and residual rounding are exact.

amdgpu.target<gfx11-generic> @convolution_wave32 {subgroup_size = 32}

template.decl @image.convolution_nchw_element_f32(%height: index, %width: index, %inputs: index, %outputs: index, %extent: index, %planes: index, %upsample: index, %channel: index, %pixel: index, %input: buffer, %weight: buffer, %bias: buffer) -> (f32)
"""
    return header + "".join(render_case(case) for case in CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("convolution.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its authored oracle")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
