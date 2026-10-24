#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author exact BF16 modulation observations with independent rational rounding."""

import argparse
import math
import random
import struct
from fractions import Fraction
from pathlib import Path

from binary_reference import f32, round_binary


def rounded(value, precision):
    result = float(round_binary(Fraction(value), precision, minimum_exponent=-126))
    return math.copysign(0.0, float(value)) if result == 0 else result


def add(left, right):
    exact = Fraction(left) + Fraction(right)
    return rounded(exact if exact else left + right, 24)


def multiply(left, right):
    exact = Fraction(left) * Fraction(right)
    return rounded(exact if exact else left * right, 24)


def evaluate(row, omitted=""):
    value, scale, scale_table, shift, shift_table = row

    def boundary(value, name):
        return value if name == omitted else rounded(value, 8)

    scale_sum = boundary(add(scale, boundary(scale_table, "scale_table")), "scale_sum")
    shift_sum = boundary(add(shift, boundary(shift_table, "shift_table")), "shift_sum")
    affine_scale = boundary(add(scale_sum, 1.0), "affine_scale")
    scaled = boundary(multiply(value, affine_scale), "scaled")
    result = rounded(add(scaled, shift_sum), 8)
    return rounded(scale_sum, 8), rounded(shift_sum, 8), result


def encode_bf16(values):
    words = []
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        assert bits & 65535 == 0
        words.append(bits >> 16)
    return struct.pack(f"<{len(words)}H", *words).hex()


def cases():
    assert rounded(Fraction(257, 256), 8) == 1.0
    assert rounded(Fraction(259, 256), 8) == 130 / 128
    assert rounded(Fraction(2) ** -134, 8) == 0.0
    assert rounded(3 * Fraction(2) ** -134, 8) == 2.0**-132
    assert rounded(Fraction(2) ** -150, 24) == 0.0
    assert rounded(3 * Fraction(2) ** -150, 24) == 2.0**-148

    rounding = []
    for sign in (-1, 1):
        for mantissa in (1, 3, 7, 15, 63, 127):
            midpoint = sign * (1 + mantissa / 256)
            bits = struct.unpack("<I", struct.pack("<f", midpoint))[0]
            for displacement in (-1, 0, 1):
                table = struct.unpack("<f", struct.pack("<I", bits + displacement))[0]
                for value in (-1.5, 1.5):
                    rounding.append((value, 0.5, table, -0.5, -table))
    generator = random.Random(0xAF16)
    for _ in range(128):
        value, scale, shift = (rounded(generator.uniform(-8, 8), 8) for _ in range(3))
        tables = [f32(generator.uniform(-4, 4)) for _ in range(2)]
        rounding.append((value, scale, tables[0], shift, tables[1]))
    for omitted in ("scale_table", "scale_sum", "shift_table", "shift_sum", "affine_scale", "scaled"):
        assert any(encode_bf16(evaluate(row)) != encode_bf16(evaluate(row, omitted))
                   for row in rounding), omitted

    extremes = [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.0, 0.0, 0.0, -0.0, -0.0),
        (-0.0, -1.0, 0.0, -0.0, -0.0),
        (1.5, -1.0, 0.0, -1.0, 1.0),
    ]
    for exponent in (-133, -132, -127, -126, -100, -60, -20, 0, 20, 60, 100):
        for sign in (-1, 1):
            value = sign * 2.0**exponent
            extremes.append((value, -0.5, 0.0, 0.0, 0.0))
            extremes.append((value, 0.5, 0.0, -value, 0.0))
    for exponent in (-149, -134, -133, -126):
        for sign in (-1, 1):
            extremes.append((1.0, 0.0, sign * 2.0**exponent, 0.0, sign * 2.0**exponent))
    widened = [
        (value, scale, rounded(scale_table, 8), shift, rounded(shift_table, 8))
        for value, scale, scale_table, shift, shift_table in rounding[-63:]
    ]
    return (("rounding", rounding, 128), ("extremes", extremes, 32),
            ("widened_tables", widened, 256))


def render_case(name, rows, threads):
    count = len(rows)
    inputs = [element for value, scale, _, shift, _ in rows for element in (value, scale, shift)]
    tables = [element for _, _, scale_table, _, shift_table in rows for element in (scale_table, shift_table)]
    expected = [element for row in rows for element in evaluate(row)]
    input_hex = encode_bf16(inputs)
    table_hex = struct.pack(f"<{len(tables)}f", *tables).hex()
    expected_hex = encode_bf16(expected)
    return f'''
global.rodata.def @{name}_inputs = align(2) bytes("{input_hex}")

global.rodata.def @{name}_tables = align(4) bytes("{table_hex}")

global.rodata.def @{name}_expected = align(2) bytes("{expected_hex}")

func.def @initialize_{name}(%input: buffer, %table: buffer) {{
  %origin = index.constant 0 : offset
  %input_bytes = index.constant {count * 6} : offset
  %table_bytes = index.constant {count * 8} : offset
  %inputs = global.load @{name}_inputs : buffer
  %tables = global.load @{name}_tables : buffer
  buffer.copy %inputs[%origin], %input[%origin], %input_bytes
  buffer.copy %tables[%origin], %table[%origin], %table_bytes
  func.return
}}

kernel.def target(@modulation_wave32) @apply_{name}() {{
  %groups = index.constant {(count + threads - 1) // threads} : index
  %threads = index.constant {threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %table: buffer, %output: buffer) {{
  %count = index.constant {count} : index
  %threads = index.constant {threads} : index
  %group = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %row = index.madd %group, %threads, %lane : index
  %live = index.cmp ult, %row, %count : index
  scf.if %live {{
    %origin = index.constant 0 : offset
    %inputs = buffer.view %input[%origin] : buffer -> view<[%count]x3xbf16>
    %tables = buffer.view %table[%origin] : buffer -> view<[%count]x2xf32>
    %outputs = buffer.view %output[%origin] : buffer -> view<[%count]x3xbf16>
    %value = view.load %inputs[%row, 0] : view<[%count]x3xbf16> -> bf16
    %scale = view.load %inputs[%row, 1] : view<[%count]x3xbf16> -> bf16
    %shift = view.load %inputs[%row, 2] : view<[%count]x3xbf16> -> bf16
    %scale_table = view.load %tables[%row, 0] : view<[%count]x2xf32> -> f32
    %shift_table = view.load %tables[%row, 1] : view<[%count]x2xf32> -> f32
    %scale_sum = template.apply<@modulation.combine_bf16>(%scale, %scale_table) : (bf16, f32) -> (bf16)
    %shift_sum = template.apply<@modulation.combine_bf16>(%shift, %shift_table) : (bf16, f32) -> (bf16)
    %affine = template.apply<@modulation.affine_bf16>(%value, %scale, %scale_table, %shift, %shift_table) : (bf16, bf16, f32, bf16, f32) -> (bf16)
    view.store %scale_sum, %outputs[%row, 0] : bf16, view<[%count]x3xbf16>
    view.store %shift_sum, %outputs[%row, 1] : bf16, view<[%count]x3xbf16>
    view.store %affine, %outputs[%row, 2] : bf16, view<[%count]x3xbf16>
  }}
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %table: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 6} : offset
  %expected = global.load @{name}_expected : buffer
  buffer.copy %expected[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(-12345) : tensor<{count * 3 + 32}xi16>
    %input = check.tensor.view %input_storage offset(32) : tensor<{count * 3 + 32}xi16> -> tensor<{count * 3}xbf16>
    %table_storage = check.generate.fill value(-99.0) : tensor<{count * 2 + 32}xf32>
    %table = check.tensor.view %table_storage offset(64) : tensor<{count * 2 + 32}xf32> -> tensor<{count * 2}xf32>
    check.generate<@initialize_{name}>(%input, %table) : (tensor<{count * 3}xbf16>, tensor<{count * 2}xf32>) -> ()
    %output_storage = check.generate.fill value(-12345) : tensor<{count * 3 + 32}xi16>
    %output = check.tensor.view %output_storage offset(32) : tensor<{count * 3 + 32}xi16> -> tensor<{count * 3}xbf16>
    %prefix = check.tensor.view %output_storage offset(0) : tensor<{count * 3 + 32}xi16> -> tensor<16xi16>
    %suffix = check.tensor.view %output_storage offset({count * 6 + 32}) : tensor<{count * 3 + 32}xi16> -> tensor<16xi16>
    check.compare<@apply_{name}, @reference_{name}>(%input, %table, %output) : (tensor<{count * 3}xbf16>, tensor<{count * 2}xf32>, tensor<{count * 3}xbf16>) -> () {{
      check.expect.bitwise actual(%output) expected(%output) : tensor<{count * 3}xbf16>
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{count * 3 + 32}xi16>
      check.expect.bitwise actual(%table_storage) expected(%table_storage) : tensor<{count * 2 + 32}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xi16>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xi16>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_modulation
'''


def render():
    header = '''// Generated by generate_modulation_oracle.py.
// Exact rational expressions round at each F32/BF16 tensor boundary.

amdgpu.target<gfx11-generic> @modulation_wave32 {subgroup_size = 32}

template.decl @modulation.combine_bf16(%value: bf16, %table: f32) -> (bf16)

template.decl @modulation.affine_bf16(%value: bf16, %scale: bf16, %scale_table: f32, %shift: bf16, %shift_table: f32) -> (bf16)
'''
    return header + ''.join(render_case(*case) for case in cases())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(__file__).with_name('modulation.loom')
    content = render()
    if args.check:
        if path.read_text(encoding='utf-8') != content:
            raise SystemExit(f'{path} differs from its rational reference')
    else:
        path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
