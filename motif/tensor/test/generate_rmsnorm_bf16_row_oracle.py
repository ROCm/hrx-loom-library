#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author BF16 RMS reductions and row-storage cases from independent whole rows."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import struct

from binary_reference import add_f32, f32, rounded_float


@dataclass(frozen=True)
class Shape:
    # Stable observation name.
    name: str
    # Independently normalized rows.
    rows: int
    # Elements in each row.
    width: int
    # Complete participants in each row's workgroup.
    threads: int
    # Caller-owned F32 addition to the mean square.
    epsilon: float
    # Authored input distribution.
    pattern: str = "dyadic"
    # Whether the output occupies exactly the input allocation.
    inplace: bool = False


SHAPES = (
    Shape("head_pairs", 3, 128, 64, 1e-6),
    Shape("head_lanes", 2, 128, 128, 1e-5),
    Shape("hidden", 2, 2560, 256, 1e-6),
    Shape("dit", 2, 6144, 256, 1e-5),
    Shape("zero", 1, 128, 64, 1e-5, "zero"),
    Shape("epsilon_dominated", 2, 64, 32, 1e-6, "tiny"),
    Shape("unit", 1, 32, 32, 0.0, "unit"),
    Shape("inplace", 3, 128, 64, 1e-6, inplace=True),
)


def literal(value):
    text = format(value, '.17g')
    return text if '.' in text or 'e' in text else text + '.0'


def encode(values, dtype):
    if dtype == 'f32':
        return struct.pack(f'<{len(values)}f', *values).hex()
    words = []
    for value in values:
        bits = struct.unpack('<I', struct.pack('<f', rounded_float(value, 8)))[0]
        assert bits & 65535 == 0
        words.append(bits >> 16)
    return struct.pack(f'<{len(words)}H', *words).hex()


def observations(shape, kind):
    values = []
    for row in range(shape.rows):
        for column in range(shape.width):
            if shape.pattern == 'zero':
                value = math.copysign(0.0, -1 if column % 2 else 1)
            elif shape.pattern == 'unit':
                value = -1.0 if column % 2 else 1.0
            else:
                value = ((row * 31 + column * 17) % 127 - 63) / 32
                if shape.pattern == 'tiny':
                    value *= 2.0**-20
            assert rounded_float(value, 8) == value
            values.append(value)
    if kind == 'scaled':
        weights = [((column * 19) % 63 - 31) / 16 for column in range(shape.width)]
        scales = weights
    else:
        weights = [f32(((column * 13) % 257 - 128) / 64 + 1 / 512)
                   for column in range(shape.width)]
        scales = [rounded_float(add_f32(rounded_float(value, 8), 1.0), 8)
                  for value in weights]
    expected = []
    for row in range(shape.rows):
        inputs = values[row * shape.width:(row + 1) * shape.width]
        mean = math.fsum(value * value for value in inputs) / shape.width
        inverse = 1.0 / math.sqrt(mean + f32(shape.epsilon))
        if kind == 'inverse':
            expected.append(inverse)
        else:
            # Whole-row accuracy is measured against ideal normalization.
            # The separate scalar corpus checks each explicit BF16 boundary.
            expected.extend(value * inverse * scale for value, scale in zip(inputs, scales))
    return values, weights, expected


def render_case(shape, kind):
    name = f'{kind}_{shape.name}'
    inputs, weights, expected = observations(shape, kind)
    count = shape.rows * shape.width
    output_count = len(expected)
    dtype = 'f32' if kind == 'inverse' else 'bf16'
    weight_type = 'bf16' if kind == 'scaled' else 'f32'
    weight_bytes = 2 if weight_type == 'bf16' else 4
    output_bytes = 4 if kind == 'inverse' else 2
    guard_type = 'f32' if kind == 'inverse' else 'i16'
    guard_value = '123.25' if kind == 'inverse' else '-12345'
    output = '%input' if shape.inplace else '%output'
    storage = '%input_storage' if shape.inplace else '%output_storage'
    output_allocation = '' if shape.inplace else f'''    %output_storage = check.generate.fill value({guard_value}) : tensor<{output_count + 32}x{guard_type}>
    %output = check.tensor.view %output_storage offset({16 * output_bytes}) : tensor<{output_count + 32}x{guard_type}> -> tensor<{output_count}x{dtype}>
'''
    immutable_input = '' if shape.inplace else f'      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{count + 32}xi16>\n'
    if kind == 'inverse':
        body = '''  %row = kernel.workgroup.id<x> : index
  %lane = kernel.workitem.id<x> : index
  %inverse = template.apply<@normalization.rms_inverse_bf16>(%rows, %width, %threads, %row, %epsilon, %input) : (index, index, index, index, f32, buffer) -> (f32)
  %begin = index.constant 0 : index
  %leader = index.cmp eq, %lane, %begin : index
  scf.if %leader {
    %origin = index.constant 0 : offset
    %outputs = buffer.view %output[%origin] : buffer -> view<[%rows]xf32>
    view.store %inverse, %outputs[%row] : f32, view<[%rows]xf32>
  }
'''
        tolerance = 2.0**-16
    else:
        body = f'  template.apply<@normalization.{kind}_rows_bf16>(%rows, %width, %threads, %epsilon, %input, %weight, %output) : (index, index, index, f32, buffer, buffer, buffer)\n'
        boundaries = 1 if kind == 'zero_centered' else 2
        tolerance = (1 + 2.0**-16) * (1 + 2.0**-24)**2 * (1 + 2.0**-8)**boundaries / (1 - 2.0**-8) - 1
    if shape.pattern in ('zero', 'unit') and kind != 'inverse':
        comparison = f'check.expect.bitwise actual({output}) expected({output}) : tensor<{output_count}x{dtype}>'
    else:
        comparison = f'check.expect.close actual({output}) expected({output}) atol(0.0) rtol({literal(tolerance)}) nan(different) : tensor<{output_count}x{dtype}>'
    return f'''
global.rodata.def @{name}_input = align(2) bytes("{encode(inputs, 'bf16')}")

global.rodata.def @{name}_weight = align({weight_bytes}) bytes("{encode(weights, weight_type)}")

global.rodata.def @{name}_expected = align({output_bytes}) bytes("{encode(expected, dtype)}")

func.def @initialize_{name}(%input: buffer, %weight: buffer) {{
  %origin = index.constant 0 : offset
  %input_bytes = index.constant {count * 2} : offset
  %weight_bytes = index.constant {shape.width * weight_bytes} : offset
  %inputs = global.load @{name}_input : buffer
  %weights = global.load @{name}_weight : buffer
  buffer.copy %inputs[%origin], %input[%origin], %input_bytes
  buffer.copy %weights[%origin], %weight[%origin], %weight_bytes
  func.return
}}

kernel.def target(@rms_rows_wave32) @apply_{name}() {{
  %rows = index.constant {shape.rows} : index
  %threads = index.constant {shape.threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer, %output: buffer) {{
  %rows = index.constant {shape.rows} : index
  %width = index.constant {shape.width} : index
  %threads = index.constant {shape.threads} : index
  %epsilon = scalar.constant {literal(f32(shape.epsilon))} : f32
{body}  kernel.return
}}

func.def @reference_{name}(%input: buffer, %weight: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {output_count * output_bytes} : offset
  %expected = global.load @{name}_expected : buffer
  buffer.copy %expected[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(-12345) : tensor<{count + 32}xi16>
    %input = check.tensor.view %input_storage offset(32) : tensor<{count + 32}xi16> -> tensor<{count}xbf16>
    %weight_storage = check.generate.fill value(123.0) : tensor<{shape.width + 32}x{weight_type}>
    %weight = check.tensor.view %weight_storage offset({weight_bytes * 16}) : tensor<{shape.width + 32}x{weight_type}> -> tensor<{shape.width}x{weight_type}>
    check.generate<@initialize_{name}>(%input, %weight) : (tensor<{count}xbf16>, tensor<{shape.width}x{weight_type}>) -> ()
{output_allocation}    %prefix = check.tensor.view {storage} offset(0) : tensor<{output_count + 32}x{guard_type}> -> tensor<16x{guard_type}>
    %suffix = check.tensor.view {storage} offset({(output_count + 16) * output_bytes}) : tensor<{output_count + 32}x{guard_type}> -> tensor<16x{guard_type}>
    check.compare<@apply_{name}, @reference_{name}>(%input, %weight, {output}) : (tensor<{count}xbf16>, tensor<{shape.width}x{weight_type}>, tensor<{output_count}x{dtype}>) -> () {{
      {comparison}
{immutable_input}      check.expect.bitwise actual(%weight_storage) expected(%weight_storage) : tensor<{shape.width + 32}x{weight_type}>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16x{guard_type}>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16x{guard_type}>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render(kinds):
    header = '''// Generated by generate_rmsnorm_bf16_row_oracle.py.
// Binary64 whole-row expressions separate reduction accuracy from BF16 publication.

amdgpu.target<gfx11-generic> @rms_rows_wave32 {subgroup_size = 32}

template.decl @normalization.rms_inverse_bf16(%rows: index, %width: index, %threads: index, %row: index, %epsilon: f32, %input: buffer) -> (f32)

template.decl @normalization.zero_centered_rows_bf16(%rows: index, %width: index, %threads: index, %epsilon: f32, %input: buffer, %weight: buffer, %output: buffer)

template.decl @normalization.scaled_rows_bf16(%rows: index, %width: index, %threads: index, %epsilon: f32, %input: buffer, %weight: buffer, %output: buffer)
'''
    return header + ''.join(render_case(shape, kind) for shape in SHAPES for kind in kinds
                            if not (shape.inplace and kind == 'inverse'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for suffix, kinds in (('inverse', ('inverse',)), ('rows', ('zero_centered', 'scaled'))):
        path = Path(__file__).with_name(f'rmsnorm_bf16_{suffix}_test.loom')
        content = render(kinds)
        if args.check:
            if path.read_text(encoding='utf-8') != content:
                raise SystemExit(f'{path} differs from its whole-row reference')
        else:
            path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
