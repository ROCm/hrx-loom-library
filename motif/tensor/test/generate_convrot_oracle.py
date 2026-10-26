#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author ConvRot observations using an explicit dense Kronecker matrix."""

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import struct

from binary_reference import f32, rounded_float


H4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


def rotation_matrix():
    matrix = ((1,),)
    for _ in range(4):
        matrix = tuple(tuple(outer * inner for outer in row4 for inner in row)
                       for row4 in H4 for row in matrix)
    assert len(matrix) == 256 and all(sum(row) == 16 for row in matrix)
    return matrix


H256 = rotation_matrix()


def bf16(value):
    return rounded_float(value, 8)


def encode(values, dtype):
    if dtype == "f32":
        return struct.pack(f"<{len(values)}f", *values).hex()
    if dtype == "i8":
        return struct.pack(f"<{len(values)}b", *values).hex()
    words = []
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        assert bits & 65535 == 0
        words.append(bits >> 16)
    return struct.pack(f"<{len(words)}H", *words).hex()


def rotate(values):
    rotated = []
    for offset in range(0, len(values), 256):
        block = values[offset:offset + 256]
        # These dyadic sums are exact in binary64 and in every F32 stage.
        # Matrix rows encode the transform independently of lane shuffles.
        outputs = [sum(sign * value for sign, value in zip(row, block)) / 16
                   for row in H256]
        assert all(f32(value) == value for value in outputs)
        rotated.extend(outputs)
    return rotated


def quantize(rotated, omitted=""):
    values = rotated if omitted == "rotation" else [bf16(value) for value in rotated]
    maximum = max(abs(value) for value in values)
    scale = max(rounded_float(Fraction(maximum) / 127, 24), f32(1e-30))
    divisor = scale if omitted == "divisor" else bf16(scale)
    quotients = [rounded_float(Fraction(value) / Fraction(divisor), 24) for value in values]
    if omitted != "quotient":
        quotients = [bf16(value) for value in quotients]
    integers = [max(-128, min(127, round(value))) for value in quotients]
    return integers, scale, quotients


@dataclass(frozen=True)
class Shape:
    # Stable observation name.
    name: str
    # Independent activation rows.
    rows: int
    # Columns sharing one maximum and scale per row.
    columns: int
    # Authored exact BF16 input family.
    pattern: str = "pattern"


SHAPES = (
    Shape("basis", 256, 256, "basis"),
    Shape("one_block", 4, 256),
    Shape("two_blocks", 3, 512),
    Shape("three_blocks", 2, 768),
    Shape("five_blocks", 3, 1280),
    Shape("hidden", 2, 6144),
    Shape("feed_forward", 1, 16384),
    Shape("zero", 2, 256, "zero"),
    Shape("tiny", 2, 256, "tiny"),
    Shape("large", 2, 256, "large"),
    Shape("saturation", 2, 256, "saturation"),
)


def observations(shape):
    inputs, output, scales, quotients = [], [], [], []
    differences = {name: 0 for name in ("rotation", "divisor", "quotient")}
    for row in range(shape.rows):
        values = []
        for column in range(shape.columns):
            value = ((row * 43 + column * 31 + column // 256 * 17) % 255 - 127) / 16
            if shape.pattern == "basis":
                value = 16.0 if column == row else 0.0
            elif shape.pattern == "zero":
                value = -0.0 if column % 2 else 0.0
            elif shape.pattern == "tiny":
                value = (1 if row == 0 else -1) * 2.0**-110
            elif shape.pattern == "large":
                value *= 2.0**96
            elif shape.pattern == "saturation":
                value = (1 if row == 0 else -1) * 1.328125
            assert bf16(value) == value
            values.append(value)
        rotated = rotate(values)
        integers, scale, row_quotients = quantize(rotated)
        for omitted in differences:
            other, other_scale, _ = quantize(rotated, omitted)
            differences[omitted] += sum(a != b for a, b in zip(integers, other))
            differences[omitted] += encode([scale], "f32") != encode([other_scale], "f32")
        inputs.extend(values)
        output.extend(integers)
        scales.append(scale)
        quotients.extend(row_quotients)
    return inputs, output, scales, quotients, differences


def render_row(shape, values):
    inputs, outputs, scales, _, _ = values
    name = "convrot_" + shape.name
    count = shape.rows * shape.columns
    return f'''
global.rodata.def @{name}_input = align(2) bytes("{encode(inputs, 'bf16')}")

global.rodata.def @{name}_expected = align(1) bytes("{encode(outputs, 'i8')}")

global.rodata.def @{name}_scales = align(4) bytes("{encode(scales, 'f32')}")

func.def @initialize_{name}(%input: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 2} : offset
  %values = global.load @{name}_input : buffer
  buffer.copy %values[%origin], %input[%origin], %bytes
  func.return
}}

kernel.def target(@convrot_wave32) @apply_{name}() {{
  %rows = index.constant {shape.rows} : index
  %threads = index.constant 128 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %output: buffer, %scales: buffer) {{
  %rows = index.constant {shape.rows} : index
  %columns = index.constant {shape.columns} : index
  template.apply<@quantization.convrot256_i8>(%rows, %columns, %input, %output, %scales) : (index, index, buffer, buffer, buffer)
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %output: buffer, %scales: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count} : offset
  %scale_bytes = index.constant {shape.rows * 4} : offset
  %expected = global.load @{name}_expected : buffer
  %expected_scales = global.load @{name}_scales : buffer
  buffer.copy %expected[%origin], %output[%origin], %bytes
  buffer.copy %expected_scales[%origin], %scales[%origin], %scale_bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(-12345) : tensor<{count + 32}xi16>
    %input = check.tensor.view %input_storage offset(32) : tensor<{count + 32}xi16> -> tensor<{count}xbf16>
    check.generate<@initialize_{name}>(%input) : (tensor<{count}xbf16>) -> ()
    %output_storage = check.generate.fill value(99) : tensor<{count + 32}xi8>
    %output = check.tensor.view %output_storage offset(16) : tensor<{count + 32}xi8> -> tensor<{count}xi8>
    %output_prefix = check.tensor.view %output_storage offset(0) : tensor<{count + 32}xi8> -> tensor<16xi8>
    %output_suffix = check.tensor.view %output_storage offset({count + 16}) : tensor<{count + 32}xi8> -> tensor<16xi8>
    %scale_storage = check.generate.fill value(123.25) : tensor<{shape.rows + 32}xf32>
    %scales = check.tensor.view %scale_storage offset(64) : tensor<{shape.rows + 32}xf32> -> tensor<{shape.rows}xf32>
    %scale_prefix = check.tensor.view %scale_storage offset(0) : tensor<{shape.rows + 32}xf32> -> tensor<16xf32>
    %scale_suffix = check.tensor.view %scale_storage offset({(shape.rows + 16) * 4}) : tensor<{shape.rows + 32}xf32> -> tensor<16xf32>
    check.compare<@apply_{name}, @reference_{name}>(%input, %output, %scales) : (tensor<{count}xbf16>, tensor<{count}xi8>, tensor<{shape.rows}xf32>) -> () {{
      check.expect.bitwise actual(%output) expected(%output) : tensor<{count}xi8>
      check.expect.bitwise actual(%scales) expected(%scales) : tensor<{shape.rows}xf32>
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{count + 32}xi16>
      check.expect.bitwise actual(%output_prefix) expected(%output_prefix) : tensor<16xi8>
      check.expect.bitwise actual(%output_suffix) expected(%output_suffix) : tensor<16xi8>
      check.expect.bitwise actual(%scale_prefix) expected(%scale_prefix) : tensor<16xf32>
      check.expect.bitwise actual(%scale_suffix) expected(%scale_suffix) : tensor<16xf32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render_rows():
    header = '''// Generated by generate_convrot_oracle.py.
// A dense regular H4 Kronecker product and exact rational rounding author outputs.

amdgpu.target<gfx11-generic> @convrot_wave32 {subgroup_size = 32}

template.decl @quantization.convrot256_i8(%rows: index, %columns: index, %input: buffer, %output: buffer, %scales: buffer)
'''
    cases = [(shape, observations(shape)) for shape in SHAPES]
    assert all(sum(values[4][name] for _, values in cases) > 0
               for name in ("rotation", "divisor", "quotient"))
    all_outputs = [value for _, values in cases for value in values[1]]
    assert min(all_outputs) == -128 and max(all_outputs) == 127
    midpoints = [value for _, values in cases for value in values[3]
                 if abs(value) % 1 == 0.5]
    assert {int(abs(value)) % 2 for value in midpoints if value > 0} == {0, 1}
    assert {int(abs(value)) % 2 for value in midpoints if value < 0} == {0, 1}
    return header + "".join(render_row(shape, values) for shape, values in cases)


def digit_observations(distance):
    inputs = [((lane * 17 + component * 13) % 127 - 63) / 8
              for lane in range(32) for component in range(8)]
    outputs = []
    for lane in range(32):
        digit = (lane // distance) % 4
        origin = lane - digit * distance
        for component in range(8):
            outputs.append(sum(sign * inputs[(origin + index * distance) * 8 + component]
                               for index, sign in enumerate(H4[digit])))
    return inputs, outputs


def render_digit(distance):
    inputs, outputs = digit_observations(distance)
    name = f"digit_{distance}"
    return f'''
global.rodata.def @{name}_input = align(4) bytes("{encode(inputs, 'f32')}")

global.rodata.def @{name}_expected = align(4) bytes("{encode(outputs, 'f32')}")

func.def @initialize_{name}(%input: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant 1024 : offset
  %values = global.load @{name}_input : buffer
  buffer.copy %values[%origin], %input[%origin], %bytes
  func.return
}}

kernel.def target(@convrot_digit_wave32) @apply_{name}() {{
  %unit = index.constant 1 : index
  %threads = index.constant 32 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %lane = kernel.subgroup.lane.id : index
  %distance = scalar.constant {distance} : i32
  %inputs = buffer.view %input[%origin] : buffer -> view<32x8xf32>
  %outputs = buffer.view %output[%origin] : buffer -> view<32x8xf32>
  %values = vector.load %inputs[%lane, 0] : view<32x8xf32> -> vector<8xf32>
  %transformed = template.apply<@quantization.convrot_digit>(%values, %distance) : (vector<8xf32>, i32) -> (vector<8xf32>)
  vector.store %transformed, %outputs[%lane, 0] : vector<8xf32>, view<32x8xf32>
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant 1024 : offset
  %values = global.load @{name}_expected : buffer
  buffer.copy %values[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(123.25) : tensor<288xf32>
    %input = check.tensor.view %input_storage offset(64) : tensor<288xf32> -> tensor<256xf32>
    check.generate<@initialize_{name}>(%input) : (tensor<256xf32>) -> ()
    %output_storage = check.generate.fill value(123.25) : tensor<288xf32>
    %output = check.tensor.view %output_storage offset(64) : tensor<288xf32> -> tensor<256xf32>
    %prefix = check.tensor.view %output_storage offset(0) : tensor<288xf32> -> tensor<16xf32>
    %suffix = check.tensor.view %output_storage offset(1088) : tensor<288xf32> -> tensor<16xf32>
    check.compare<@apply_{name}, @reference_{name}>(%input, %output) : (tensor<256xf32>, tensor<256xf32>) -> () {{
      check.expect.bitwise actual(%output) expected(%output) : tensor<256xf32>
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<288xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xf32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render_digits():
    return '''// Generated by generate_convrot_oracle.py.
// Dense H4 rows select each lane digit independently of the shuffle network.

amdgpu.target<gfx11-generic> @convrot_digit_wave32 {subgroup_size = 32}

template.decl @quantization.convrot_digit(%value: vector<8xf32>, %distance: i32) -> (vector<8xf32>)
''' + "".join(render_digit(distance) for distance in (1, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, render in (("convrot_rows", render_rows), ("convrot_digits", render_digits)):
        path = Path(__file__).with_name(name + ".loom")
        content = render()
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"{path} differs from its dense matrix reference")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
