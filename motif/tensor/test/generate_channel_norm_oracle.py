#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author channel normalization references and native-root accuracy checks."""

import argparse
import math
import random
import struct
from fractions import Fraction
from pathlib import Path

from binary_reference import f32, round_binary, round_binary_sqrt


CASES = (
    ("affine_single", 1, 1, 32, False, False, 1e-12),
    ("affine_tail", 35, 3, 32, False, False, 1e-12),
    ("affine_exact_group", 64, 16, 64, False, False, 1e-12),
    ("affine_in_place", 7, 384, 64, False, True, 1e-12),
    ("silu_tail", 65, 5, 64, True, False, 1e-12),
    ("silu_384", 3, 384, 128, True, False, 1e-12),
    ("silu_192_in_place", 5, 192, 64, True, True, 1e-12),
    ("silu_96_in_place", 9, 96, 256, True, True, 1e-12),
    ("clamp_affine", 9, 1, 32, False, True, 1e-12),
    ("clamp_exact", 9, 1, 32, False, True, 0.25),
    ("clamp_silu", 7, 4, 32, True, False, 0.25),
    ("multiple_groups", 259, 7, 128, False, False, 1e-12),
)


def float_bytes(values):
    return b"".join(struct.pack("<f", value) for value in values)


def product(left, right):
    if left == 0 or right == 0:
        return math.copysign(0.0, left * right)
    return float(round_binary(Fraction(left) * Fraction(right), 24))


def observations(values, weights, pixels, channels, minimum_norm, silu):
    scale = float(round_binary_sqrt(Fraction(channels), 24))
    result = [0.0] * len(values)
    for pixel in range(pixels):
        squared = Fraction(0)
        for channel in range(channels):
            value = Fraction(values[channel * pixels + pixel])
            squared = round_binary(squared + value * value, 24)
        length = float(round_binary_sqrt(squared, 24))
        denominator = max(length, minimum_norm)
        for channel in range(channels):
            position = channel * pixels + pixel
            value = values[position]
            normalized = (
                float(round_binary(Fraction(value) / Fraction(denominator), 24))
                if value
                else value
            )
            affine = product(product(normalized, scale), weights[channel])
            # The afn exponential is compared with an independent binary64
            # SiLU expression, rather than a copy of the native approximation.
            result[position] = affine / (1 + math.exp(-affine)) if silu else affine
    return result


def fixtures(name, pixels, channels, threads, silu, in_place, minimum_norm):
    del threads, in_place
    minimum_norm = f32(minimum_norm)
    random_source = random.Random(0xC4A + pixels * 37 + channels * 11)
    values = []
    for _ in range(pixels * channels):
        bits = (
            (random_source.getrandbits(1) << 31)
            | ((126 + random_source.randrange(-2, 3)) << 23)
            | random_source.getrandbits(23)
        )
        values.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    weights = [
        (1.0, -0.5, 0.0, 1.25, -1.75, 0.125, 2.0)[channel % 7]
        for channel in range(channels)
    ]
    if name == "affine_single":
        values = [-2.0]
    elif name == "clamp_affine":
        values = [f32(minimum_norm * factor) for factor in (-2, -1, -0.5)]
        values += [-0.0, 0.0]
        values += [f32(minimum_norm * factor) for factor in (0.5, 1, 2, 3)]
    elif name == "clamp_exact":
        values = [
            minimum_norm * factor
            for factor in (-0.5, -0.25, -0.125, -0.0, 0.0, 0.125, 0.25, 0.5, 0.75)
        ]
    else:
        # Exercise zero, a clamped norm, and a norm above the clamp without
        # making the whole corpus depend on these unusually small values.
        for channel in range(channels):
            values[channel * pixels] = -0.0 if channel % 2 else 0.0
            values[channel * pixels + 1] = f32(minimum_norm / (2 * channels))
            values[channel * pixels + 2] = f32(minimum_norm * 2)
    expected = observations(values, weights, pixels, channels, minimum_norm, silu)
    return values, weights, expected


def render_case(name, pixels, channels, threads, silu, in_place, minimum_norm):
    values, weights, expected = fixtures(
        name, pixels, channels, threads, silu, in_place, minimum_norm
    )
    count = pixels * channels
    storage_count = count + 32
    weight_count = channels + 32
    inputs = float_bytes([123.25] * 16 + values + [123.25] * 16)
    gamma = float_bytes([333.25] * 16 + weights + [333.25] * 16)
    expected_bytes = float_bytes(expected)
    groups = (pixels + threads - 1) // threads
    output_argument = "" if in_place else ", %output: buffer"
    output = "%input" if in_place else "%output"
    arguments = "%input, %weight" + ("" if in_place else ", %output")
    types = f"tensor<{count}xf32>, tensor<{channels}xf32>"
    if not in_place:
        types += f", tensor<{count}xf32>"
    output_setup = "" if in_place else f"""    %output_storage = check.generate.fill value(123.25) : tensor<{storage_count}xf32>
    %output = check.tensor.view %output_storage offset(64) : tensor<{storage_count}xf32> -> tensor<{count}xf32>
"""
    storage = "%input_storage" if in_place else "%output_storage"
    exact = name in ("affine_single", "clamp_exact")
    if exact:
        comparison = f"check.expect.bitwise actual({output}) expected({output}) : tensor<{count}xf32>"
    else:
        absolute = "4.9999999999999998e-07" if silu else "0.0"
        relative = 2e-6 if silu else 2**-20
        comparison = f"check.expect.close actual({output}) expected({output}) atol({absolute}) rtol({relative:.17g}) nan(different) : tensor<{count}xf32>"
    unchanged_input = "" if in_place else f"""      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{storage_count}xf32>
"""
    return f"""
global.rodata.def @{name}_inputs = align(4) bytes("{inputs.hex()}")

global.rodata.def @{name}_gamma = align(4) bytes("{gamma.hex()}")

global.rodata.def @{name}_expected = align(4) bytes("{expected_bytes.hex()}")

kernel.def target(@channel_norm_wave32) @normalize_{name}() {{
  %groups = index.constant {groups} : index
  %threads = index.constant {threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer{output_argument}) {{
  %pixels = index.constant {pixels} : index
  %channels = index.constant {channels} : index
  %threads = index.constant {threads} : index
  %group = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %pixel = index.madd %group, %threads, %lane : index
  %live = index.cmp ult, %pixel, %pixels : index
  scf.if %live {{
    %silu = scalar.constant {str(silu).lower()} : i1
    %minimum_norm = scalar.constant {minimum_norm:.17g} : f32
    template.apply<@image.channel_norm_f32>(%pixels, %channels, %pixel, %silu, %minimum_norm, %input, %weight, {output}) : (index, index, index, i1, f32, buffer, buffer, buffer)
  }}
  kernel.return
}}

func.def @initialize_{name}(%input_storage: buffer, %weight_storage: buffer) {{
  %origin = index.constant 0 : offset
  %input_bytes = index.constant {len(inputs)} : offset
  %weight_bytes = index.constant {len(gamma)} : offset
  %input_values = global.load @{name}_inputs : buffer
  %weight_values = global.load @{name}_gamma : buffer
  buffer.copy %input_values[%origin], %input_storage[%origin], %input_bytes
  buffer.copy %weight_values[%origin], %weight_storage[%origin], %weight_bytes
  func.return
}}

func.def @reference_{name}(%input: buffer, %weight: buffer{output_argument}) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {len(expected_bytes)} : offset
  %values = global.load @{name}_expected : buffer
  buffer.copy %values[%origin], {output}[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(123.25) : tensor<{storage_count}xf32>
    %weight_storage = check.generate.fill value(333.25) : tensor<{weight_count}xf32>
    check.generate<@initialize_{name}>(%input_storage, %weight_storage) : (tensor<{storage_count}xf32>, tensor<{weight_count}xf32>) -> ()
    %input = check.tensor.view %input_storage offset(64) : tensor<{storage_count}xf32> -> tensor<{count}xf32>
    %weight = check.tensor.view %weight_storage offset(64) : tensor<{weight_count}xf32> -> tensor<{channels}xf32>
{output_setup}    %prefix = check.tensor.view {storage} offset(0) : tensor<{storage_count}xf32> -> tensor<16xf32>
    %suffix = check.tensor.view {storage} offset({(count + 16) * 4}) : tensor<{storage_count}xf32> -> tensor<16xf32>
    check.compare<@normalize_{name}, @reference_{name}>({arguments}) : ({types}) -> () {{
      {comparison}
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xf32>
      check.expect.bitwise actual(%weight_storage) expected(%weight_storage) : tensor<{weight_count}xf32>
{unchanged_input}    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_normalize
"""


def verify_reference():
    for lower in (1 << 23, (1 << 23) + 1, (1 << 24) - 1):
        spacing = Fraction(1, 1 << 23)
        midpoint = (lower + Fraction(1, 2)) * spacing
        expected = (lower + lower % 2) * spacing
        assert round_binary_sqrt(midpoint * midpoint, 24) == expected
    # A native root within one ULP differs from nearest-even by at most
    # 3u/(1-u), u=2^-24. Two roots and three rounded arithmetic steps
    # propagate less than 16u relative error for the normal finite corpus.
    unit_roundoff = Fraction(1, 1 << 24)
    root_error = 3 * unit_roundoff / (1 - unit_roundoff)
    ratio = (1 + root_error) / (1 - root_error)
    ratio *= ((1 + unit_roundoff) / (1 - unit_roundoff)) ** 3
    assert ratio - 1 < Fraction(1, 1 << 20)


def render():
    verify_reference()
    header = """// Generated by generate_channel_norm_oracle.py.
// Exact rational FMA and midpoint-square comparisons define the reference.
// General affine cases allow the derived 2^-20 relative native-root error.
// SiLU uses binary64; directed clamp arithmetic and guards remain bitwise.

amdgpu.target<gfx11-generic> @channel_norm_wave32 {subgroup_size = 32}

template.decl @image.channel_norm_f32(%pixels: index, %channels: index, %pixel: index, %silu: i1, %minimum_norm: f32, %input: buffer, %weight: buffer, %output: buffer)
"""
    return header + "".join(render_case(*case) for case in CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("channel_norm_test.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its authored oracle")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
