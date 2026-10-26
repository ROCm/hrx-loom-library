#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author packed BF16 inputs and independently rounded channel-first pixels."""

import argparse
import random
import struct
from fractions import Fraction
from pathlib import Path

from binary_reference import f32, round_binary


CASES = (
    ("one_patch", 2, 2, 1, 32),
    ("portrait", 10, 2, 3, 64),
    ("landscape", 2, 22, 7, 128),
    ("exact_group", 8, 16, 16, 128),
    ("tail", 6, 22, 5, 64),
    ("multiple_groups", 18, 18, 16, 256),
    ("division_rounding", 2, 2, 2, 32),
)


def float_bytes(values):
    return b"".join(struct.pack("<f", value) for value in values)


def fixtures(name, height, width, channels):
    """Start from logical channel-first pixels, then pack whole patches."""
    random_source = random.Random(0x517A + height * 37 + width * 11 + channels)
    pixels = height * width
    words = [
        (random_source.getrandbits(1) << 15)
        | ((127 + random_source.randrange(-4, 5)) << 7)
        | random_source.getrandbits(7)
        for _ in range(channels * pixels)
    ]
    inverses = [
        (1.0, 0.25, 3.0, 10.0, 5.5, 0.125, 2.0)[channel % 7]
        for channel in range(channels)
    ]
    means = [f32((channel - channels // 2) / 32) for channel in range(channels)]
    if name == "division_rounding":
        words = [0x3F80] * pixels + [0xBF80] * pixels
        inverses = [3.0, 10.0]
        means = [-f32(1 / 3), f32(1 / 10)]

    expected = [123.25] * 16
    for channel in range(channels):
        for pixel in range(pixels):
            word = words[channel * pixels + pixel]
            value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
            quotient = Fraction(value) / Fraction(inverses[channel])
            rounded = round_binary(quotient, 24)
            result = round_binary(rounded + Fraction(means[channel]), 24)
            if name == "division_rounding":
                assert result == 0
                assert round_binary(quotient + Fraction(means[channel]), 24) != 0
            expected.append(float(result))
    expected += [123.25] * 16

    # This traversal follows the stored patch layout. Expected values above
    # remain in logical channel-first order and never invert a packed index.
    packed = []
    for patch_y in range(0, height, 2):
        for patch_x in range(0, width, 2):
            for channel in range(channels):
                for row in range(patch_y, patch_y + 2):
                    for column in range(patch_x, patch_x + 2):
                        packed.append(words[channel * pixels + row * width + column])
    packed += [0x7F7F] * 16
    affine = inverses + means + [777.0] * 16
    return (
        b"".join(struct.pack("<H", word) for word in packed),
        float_bytes(affine),
        float_bytes(expected),
    )


def render_case(name, height, width, channels, threads):
    inputs, affine, expected = fixtures(name, height, width, channels)
    pixels = height * width
    values = pixels * channels
    groups = (pixels + threads - 1) // threads
    return f"""
global.rodata.def @{name}_inputs = align(4) bytes("{inputs.hex()}")

global.rodata.def @{name}_affine = align(4) bytes("{affine.hex()}")

global.rodata.def @{name}_expected = align(4) bytes("{expected.hex()}")

kernel.def target(@image_unpack_wave32) @unpack_{name}() {{
  %groups = index.constant {groups} : index
  %channels = index.constant {channels} : index
  %threads = index.constant {threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %channels, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%inputs: buffer, %affine: buffer, %outputs: buffer) {{
  %height = index.constant {height} : index
  %width = index.constant {width} : index
  %channels = index.constant {channels} : index
  %pixels = index.constant {pixels} : index
  %threads = index.constant {threads} : index
  %group = kernel.workgroup.id<x> : index
  %channel_value = kernel.workgroup.id<y> : index
  %channel = index.assume %channel_value [ge(%channel_value, 0), lt(%channel_value, %channels)] : index
  %lane = kernel.workitem.id<x> : index
  %pixel = index.madd %group, %threads, %lane : index
  %live = index.cmp ult, %pixel, %pixels : index
  scf.if %live {{
    %value = template.apply<@image.unpack_affine_element_bf16_f32>(%height, %width, %channels, %channel, %pixel, %inputs, %affine) : (index, index, index, index, index, buffer, buffer) -> (f32)
    %output_origin = index.constant 64 : offset
    %output = buffer.view %outputs[%output_origin] : buffer -> view<{channels}x{pixels}xf32>
    view.store %value, %output[%channel, %pixel] : f32, view<{channels}x{pixels}xf32>
  }}
  kernel.return
}}

func.def @initialize_{name}(%inputs: buffer, %affine: buffer) {{
  %origin = index.constant 0 : offset
  %input_bytes = index.constant {len(inputs)} : offset
  %affine_bytes = index.constant {len(affine)} : offset
  %input_values = global.load @{name}_inputs : buffer
  %affine_values = global.load @{name}_affine : buffer
  buffer.copy %input_values[%origin], %inputs[%origin], %input_bytes
  buffer.copy %affine_values[%origin], %affine[%origin], %affine_bytes
  func.return
}}

func.def @reference_{name}(%inputs: buffer, %affine: buffer, %outputs: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {len(expected)} : offset
  %values = global.load @{name}_expected : buffer
  buffer.copy %values[%origin], %outputs[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %inputs = check.generate.fill value(0.0) : tensor<{values + 16}xbf16>
    %affine = check.generate.fill value(0.0) : tensor<{channels * 2 + 16}xf32>
    %outputs = check.generate.fill value(123.25) : tensor<{values + 32}xf32>
    check.generate<@initialize_{name}>(%inputs, %affine) : (tensor<{values + 16}xbf16>, tensor<{channels * 2 + 16}xf32>) -> ()
    check.compare<@unpack_{name}, @reference_{name}>(%inputs, %affine, %outputs) : (tensor<{values + 16}xbf16>, tensor<{channels * 2 + 16}xf32>, tensor<{values + 32}xf32>) -> () {{
      check.expect.bitwise actual(%outputs) expected(%outputs) : tensor<{values + 32}xf32>
      check.expect.bitwise actual(%inputs) expected(%inputs) : tensor<{values + 16}xbf16>
      check.expect.bitwise actual(%affine) expected(%affine) : tensor<{channels * 2 + 16}xf32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_unpack
"""


def render():
    header = """// Generated by generate_image_unpack_oracle.py.
// Logical channel-first expectations are authored before packing the input.
// Exact rational rounding keeps division and affine addition separate.
// Whole-storage comparisons include untouched inputs and output guards.

amdgpu.target<gfx11-generic> @image_unpack_wave32 {subgroup_size = 32}

template.decl @image.unpack_affine_element_bf16_f32(%height: index, %width: index, %channels: index, %channel: index, %pixel: index, %input: buffer, %affine: buffer) -> (f32)
"""
    return header + "".join(render_case(*case) for case in CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("image_unpack.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its authored oracle")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
