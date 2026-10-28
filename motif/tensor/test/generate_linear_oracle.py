#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author BF16 linear references independently of native recurrence order."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct

from binary_reference import f32, rounded_float


@dataclass(frozen=True)
class Shape:
    # Stable scenario name.
    name: str
    # Complete activation rows.
    rows: int
    # Reduction extent in each activation/weight row.
    inputs: int
    # Complete output columns.
    outputs: int
    # Checkpoint storage representation.
    weight_type: str = "bf16"
    # Whether to use the four-chain native BF16 provider.
    four_chains: bool = False
    # Physical output layout; raw F32 observations stay dense.
    layout: str = "dense"
    # Whether launch-grid coordinates visit logical tiles in reverse order.
    reverse: bool = False
    # Whether weights straddle BF16 conversion midpoints.
    midpoints: bool = False
    # Whether small terms distinguish F32 from BF16 accumulation.
    precision: bool = False


SHAPES = (
    Shape("storage_minimum", 16, 16, 16),
    Shape("storage_short_tail", 32, 48, 48, layout="padded", reverse=True),
    Shape("storage_block_tail", 32, 272, 32, "f32", layout="column", reverse=True),
    Shape("storage_midpoints", 16, 16, 32, "f32", midpoints=True),
    Shape("storage_model", 16, 6144, 16, "f32"),
    Shape("storage_precision", 16, 256, 16, precision=True),
    Shape("four_minimum", 16, 64, 16, four_chains=True),
    Shape("four_block_tail", 32, 320, 32, four_chains=True, reverse=True),
    Shape("four_blocks", 16, 512, 48, four_chains=True, layout="column"),
    Shape("four_model", 16, 2560, 16, four_chains=True),
    Shape("four_precision", 16, 256, 16, four_chains=True, precision=True),
)


def encode(values, dtype):
    if dtype == "f32":
        return struct.pack(f"<{len(values)}f", *values).hex()
    words = []
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        assert bits & 65535 == 0
        words.append(bits >> 16)
    return struct.pack(f"<{len(words)}H", *words).hex()


def output_geometry(shape):
    if shape.layout == "column":
        return 1, shape.rows, shape.rows * shape.outputs
    stride = shape.outputs + (16 if shape.layout == "padded" else 0)
    return stride, 1, shape.rows * stride


def observations(shape):
    inputs = [((row * 3 + column * 11) % 15 - 7) / 8
              for row in range(shape.rows) for column in range(shape.inputs)]
    weights = [((row * 5 + column * 7) % 15 - 7) / 8
               for row in range(shape.outputs) for column in range(shape.inputs)]
    if shape.midpoints:
        choices = (1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, -1 - 3 / 256)
        weights = [choices[(row + column) % 4]
                   for row in range(shape.outputs) for column in range(shape.inputs)]
    if shape.precision:
        inputs = [1.0] * len(inputs)
        weights = [1.0 if column == 0 else 2.0**-13
                   for _ in range(shape.outputs) for column in range(shape.inputs)]
    converted = [rounded_float(value, 8) for value in weights]
    wide = []
    contribution_sums = []
    minimum_product = float("inf")
    unrounded_differences = 0
    for row in range(shape.rows):
        values = inputs[row * shape.inputs:(row + 1) * shape.inputs]
        for output in range(shape.outputs):
            begin = output * shape.inputs
            products = [left * right for left, right in zip(values, converted[begin:begin + shape.inputs])]
            total = sum(products)
            # Finite dyadic products and the sum of their magnitudes fit F32.
            # No target's matrix reduction order is used to author this result.
            assert all(f32(product) == product for product in products)
            assert f32(sum(abs(product) for product in products)) == sum(abs(product) for product in products)
            assert f32(total) == total
            wide.append(total)
            if shape.precision:
                assert rounded_float(total, 8) != total
            contribution_sums.append(sum(abs(product) for product in products))
            minimum_product = min(minimum_product, min(abs(product) for product in products if product))
            if shape.midpoints:
                raw = sum(left * right for left, right in zip(values, weights[begin:begin + shape.inputs]))
                unrounded_differences += raw != total
    if shape.midpoints:
        assert unrounded_differences > 0
    # The target owns native matrix arithmetic. Require twenty significant
    # binary bits relative to input contributions, independently of observed
    # GPU error. This remains smaller than half one ordinary nonzero term.
    tolerance = max(contribution_sums) * 2.0**-20
    assert tolerance < minimum_product / 2
    row_stride, column_stride, count = output_geometry(shape)
    output = [123.0] * count
    for row in range(shape.rows):
        for column in range(shape.outputs):
            output[row * row_stride + column * column_stride] = rounded_float(wide[row * shape.outputs + column], 8)
    return inputs, weights, wide, output, tolerance


def render_case(shape):
    name = shape.name
    inputs, weights, wide, output, tolerance = observations(shape)
    input_count, weight_count, wide_count, output_count = map(len, (inputs, weights, wide, output))
    weight_bytes = 4 if shape.weight_type == "f32" else 2
    weight_guard_type = "f32" if shape.weight_type == "f32" else "i16"
    weight_guard_value = "123.25" if shape.weight_type == "f32" else "-12345"
    row_stride, column_stride, _ = output_geometry(shape)
    arrays = (("input", inputs, "bf16"), ("weight", weights, shape.weight_type),
              ("wide", wide, "f32"), ("expected", output, "bf16"))
    rodata = "\n\n".join(f'global.rodata.def @{name}_{key} = align({4 if dtype == "f32" else 2}) bytes("{encode(values, dtype)}")'
                         for key, values, dtype in arrays)
    if shape.reverse:
        coordinates = f'''  %last_row = index.constant {shape.rows // 16 - 1} : index
  %last_output = index.constant {shape.outputs // 16 - 1} : index
  %row_tile = index.sub %last_row, %grid_row : index
  %output_tile = index.sub %last_output, %grid_output : index
'''
    else:
        coordinates = ""
    row_tile = "%row_tile" if shape.reverse else "%grid_row"
    output_tile = "%output_tile" if shape.reverse else "%grid_output"
    if shape.four_chains:
        call = "  %result = template.apply<@projection.linear_bf16>(%rows, %inputs, %outputs, %row_origin, %output_origin, %input, %weight) : (index, index, index, index, index, buffer, buffer) -> (vector<8xf32>)\n"
    else:
        call = f"  %weight_is_bf16 = scalar.constant {'true' if shape.weight_type == 'bf16' else 'false'} : i1\n  %result = template.apply<@projection.linear_storage_bf16>(%rows, %inputs, %outputs, %weight_is_bf16, %row_origin, %output_origin, %input, %weight) : (index, index, index, i1, index, index, buffer, buffer) -> (vector<8xf32>)\n"
    if shape.precision:
        expectation = f"check.expect.bitwise actual(%wide) expected(%wide) : tensor<{wide_count}xf32>"
    else:
        expectation = f"check.expect.close actual(%wide) expected(%wide) atol({format(tolerance, '.17g')}) rtol(0.0) nan(different) : tensor<{wide_count}xf32>"
    return f'''
{rodata}

func.def @initialize_{name}(%input: buffer, %weight: buffer) {{
  %origin = index.constant 0 : offset
  %input_bytes = index.constant {input_count * 2} : offset
  %weight_bytes = index.constant {weight_count * weight_bytes} : offset
  %inputs = global.load @{name}_input : buffer
  %weights = global.load @{name}_weight : buffer
  buffer.copy %inputs[%origin], %input[%origin], %input_bytes
  buffer.copy %weights[%origin], %weight[%origin], %weight_bytes
  func.return
}}

kernel.def target(@projection_rdna3_wave32) @apply_{name}() {{
  %row_tiles = index.constant {shape.rows // 16} : index
  %output_tiles = index.constant {shape.outputs // 16} : index
  %threads = index.constant 32 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%row_tiles, %output_tiles, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer, %wide: buffer, %output: buffer) {{
  %rows = index.constant {shape.rows} : index
  %inputs = index.constant {shape.inputs} : index
  %outputs = index.constant {shape.outputs} : index
  %tile = index.constant 16 : index
  %grid_row = kernel.workgroup.id<x> : index
  %grid_output = kernel.workgroup.id<y> : index
{coordinates}  %row_origin = index.mul {row_tile}, %tile : index
  %output_origin = index.mul {output_tile}, %tile : index
{call}  %origin = index.constant 0 : offset
  %wide_view = buffer.view %wide[%origin] : buffer -> view<[%rows]x[%outputs]xf32>
  vector.fragment.store<result> %result, %wide_view[%row_origin, %output_origin] shape [%tile, %tile] : vector<8xf32>, view<[%rows]x[%outputs]xf32>
  %row_stride = index.constant {row_stride} : index
  %column_stride = index.constant {column_stride} : index
  %layout = encoding.layout.strided [%row_stride, %column_stride] : encoding<layout>
  template.apply<@projection.store_bf16>(%rows, %outputs, %row_origin, %output_origin, %layout, %result, %output) : (index, index, index, index, encoding<layout>, vector<8xf32>, buffer)
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %weight: buffer, %wide: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %wide_bytes = index.constant {wide_count * 4} : offset
  %output_bytes = index.constant {output_count * 2} : offset
  %wide_values = global.load @{name}_wide : buffer
  %output_values = global.load @{name}_expected : buffer
  buffer.copy %wide_values[%origin], %wide[%origin], %wide_bytes
  buffer.copy %output_values[%origin], %output[%origin], %output_bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(-12345) : tensor<{input_count + 32}xi16>
    %input = check.tensor.view %input_storage offset(32) : tensor<{input_count + 32}xi16> -> tensor<{input_count}xbf16>
    %weight_storage = check.generate.fill value({weight_guard_value}) : tensor<{weight_count + 32}x{weight_guard_type}>
    %weight = check.tensor.view %weight_storage offset({weight_bytes * 16}) : tensor<{weight_count + 32}x{weight_guard_type}> -> tensor<{weight_count}x{shape.weight_type}>
    check.generate<@initialize_{name}>(%input, %weight) : (tensor<{input_count}xbf16>, tensor<{weight_count}x{shape.weight_type}>) -> ()
    %wide_storage = check.generate.fill value(123.25) : tensor<{wide_count + 32}xf32>
    %wide = check.tensor.view %wide_storage offset(64) : tensor<{wide_count + 32}xf32> -> tensor<{wide_count}xf32>
    %wide_prefix = check.tensor.view %wide_storage offset(0) : tensor<{wide_count + 32}xf32> -> tensor<16xf32>
    %wide_suffix = check.tensor.view %wide_storage offset({(wide_count + 16) * 4}) : tensor<{wide_count + 32}xf32> -> tensor<16xf32>
    %output_storage = check.generate.fill value(17142) : tensor<{output_count + 32}xi16>
    %output = check.tensor.view %output_storage offset(32) : tensor<{output_count + 32}xi16> -> tensor<{output_count}xbf16>
    check.compare<@apply_{name}, @reference_{name}>(%input, %weight, %wide, %output) : (tensor<{input_count}xbf16>, tensor<{weight_count}x{shape.weight_type}>, tensor<{wide_count}xf32>, tensor<{output_count}xbf16>) -> () {{
      {expectation}
      check.expect.bitwise actual(%wide_prefix) expected(%wide_prefix) : tensor<16xf32>
      check.expect.bitwise actual(%wide_suffix) expected(%wide_suffix) : tensor<16xf32>
      check.expect.bitwise actual(%output_storage) expected(%output_storage) : tensor<{output_count + 32}xi16>
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{input_count + 32}xi16>
      check.expect.bitwise actual(%weight_storage) expected(%weight_storage) : tensor<{weight_count + 32}x{weight_guard_type}>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    header = '''// Generated by generate_linear_oracle.py.
// Exact dyadic references have a twenty-bit input-scaled F32 accuracy budget.
// Accumulator-precision cases, BF16 publication, and storage guards are exact.

amdgpu.target<gfx11-generic> @tensor_projection_rdna3

amdgpu.target<gfx11-generic> @projection_rdna3_wave32 {subgroup_size = 32}

template.decl @projection.linear_storage_bf16(%rows: index, %inputs: index, %outputs: index, %weight_is_bf16: i1, %row_origin: index, %output_origin: index, %input: buffer, %weight: buffer) -> (vector<8xf32>)

template.decl @projection.linear_bf16(%rows: index, %inputs: index, %outputs: index, %row_origin: index, %output_origin: index, %input: buffer, %weight: buffer) -> (vector<8xf32>)

template.decl @projection.store_bf16(%rows: index, %outputs: index, %row_origin: index, %output_origin: index, %output_layout: encoding<layout>, %result: vector<8xf32>, %output: buffer)
'''
    return header + "".join(render_case(shape) for shape in SHAPES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("linear_test.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its whole-matrix reference")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
