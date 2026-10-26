#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author independent temporal-bank contraction and traversal observations."""

import argparse
from pathlib import Path

from generate_linear_oracle import Shape, encode, observations
from generate_linear_tiled_oracle import GROUP_REFERENCE


SHAPES = (
    *((Shape(f"tail{row}", row, 256, column), 1, ordinal % 2 == 0)
      for ordinal, (row, column) in enumerate(zip(range(16, 128, 16),
                                                 (16, 32, 48, 64, 80, 96, 112)))),
    (Shape("full_pairwise", 128, 512, 96), 1, False),
    (Shape("full_read_ahead", 128, 512, 96), 1, True),
    (Shape("interior_tail_pairwise", 144, 768, 208), 1, False),
    (Shape("interior_tail_read_ahead", 144, 768, 208), 1, True),
    (Shape("partial_group", 416, 256, 112), 2, True),
    (Shape("oversized_group", 144, 256, 112), 8, False),
    (Shape("model6144", 16, 6144, 112), 1, True),
    (Shape("model16384", 16, 16384, 96), 1, False),
    (Shape("precision_pairwise", 16, 256, 16, precision=True), 1, False),
    (Shape("precision_read_ahead", 16, 256, 16, precision=True), 1, True),
)

# Logical dimensions cover physical rounding independently on both axes.
GRIDS = (
    (16, 16, 1), (128, 96, 1), (144, 112, 1), (272, 208, 8),
    *((128 * 8 + row, 96 + column, 8)
      for row, column in zip(range(16, 129, 16), (16, 32, 48, 64, 80, 96, 16, 32))),
    *((128 * (8 + remainder), 288, 8) for remainder in range(8)),
    (4224, 6144, 8), (4224, 16384, 8), (4608, 16384, 4),
    (4608, 6144, 4), (65536, 16384, 8),
)


def render_matrix(shape, group, read_ahead):
    inputs, weights, _, output, _ = observations(shape)
    name = shape.name
    input_count, weight_count, output_count = map(len, (inputs, weights, output))
    rodata = "\n\n".join(f'global.rodata.def @{name}_{key} = align(2) bytes("{encode(values, "bf16")}")'
                         for key, values in (("input", inputs), ("weight", weights), ("expected", output)))
    return f'''
{rodata}

func.def @initialize_{name}(%input: buffer, %weight: buffer) {{
  %origin = index.constant 0 : offset
  %input_bytes = index.constant {input_count * 2} : offset
  %weight_bytes = index.constant {weight_count * 2} : offset
  %input_values = global.load @{name}_input : buffer
  %weight_values = global.load @{name}_weight : buffer
  buffer.copy %input_values[%origin], %input[%origin], %input_bytes
  buffer.copy %weight_values[%origin], %weight[%origin], %weight_bytes
  func.return
}}

kernel.def target(@temporal_projection_wave32) @apply_{name}() {{
  %row_tiles = index.constant {(shape.rows + 127) // 128} : index
  %output_tiles = index.constant {(shape.outputs + 95) // 96} : index
  %threads = index.constant 128 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%row_tiles, %output_tiles, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer, %output: buffer) {{
  %rows = index.constant {shape.rows} : index
  %inputs = index.constant {shape.inputs} : index
  %outputs = index.constant {shape.outputs} : index
  %group = index.constant {group} : index
  %read_ahead = scalar.constant {str(read_ahead).lower()} : i1
  template.apply<@projection.linear_temporal_bf16>(%rows, %inputs, %outputs, %group, %read_ahead, %input, %weight, %output) : (index, index, index, index, i1, buffer, buffer, buffer)
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %weight: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {output_count * 2} : offset
  %expected = global.load @{name}_expected : buffer
  buffer.copy %expected[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(-12345) : tensor<{input_count + 32}xi16>
    %input = check.tensor.view %input_storage offset(32) : tensor<{input_count + 32}xi16> -> tensor<{input_count}xbf16>
    %weight_storage = check.generate.fill value(-12345) : tensor<{weight_count + 32}xi16>
    %weight = check.tensor.view %weight_storage offset(32) : tensor<{weight_count + 32}xi16> -> tensor<{weight_count}xbf16>
    %output_storage = check.generate.fill value(17142) : tensor<{output_count + 32}xi16>
    %output = check.tensor.view %output_storage offset(32) : tensor<{output_count + 32}xi16> -> tensor<{output_count}xbf16>
    check.generate<@initialize_{name}>(%input, %weight) : (tensor<{input_count}xbf16>, tensor<{weight_count}xbf16>) -> ()
    check.compare<@apply_{name}, @reference_{name}>(%input, %weight, %output) : (tensor<{input_count}xbf16>, tensor<{weight_count}xbf16>, tensor<{output_count}xbf16>) -> () {{
      check.expect.bitwise actual(%output_storage) expected(%output_storage) : tensor<{output_count + 32}xi16>
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{input_count + 32}xi16>
      check.expect.bitwise actual(%weight_storage) expected(%weight_storage) : tensor<{weight_count + 32}xi16>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render_grid(rows, columns, group):
    name = f"grid_{rows}_{columns}_{group}"
    row_tiles, column_tiles = (rows + 127) // 128, (columns + 95) // 96
    count = row_tiles * column_tiles * 2
    return f'''
kernel.def target(@coordinates_wave32) @apply_{name}() {{
  %rows = index.constant {row_tiles} : index
  %columns = index.constant {column_tiles} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %columns, %unit) workgroup_size(%unit, %unit, %unit) : index
}} launch(%output: buffer) {{
  %rows = index.constant {rows} : index
  %columns = index.constant {columns} : index
  %row_tiles = index.constant {row_tiles} : index
  %group = index.constant {group} : index
  %row, %column = template.apply<@projection.temporal_coordinates>(%rows, %columns, %group) : (index, index, index) -> (index, index)
  %origin = index.constant 0 : offset
  %physical_row = kernel.workgroup.id<x> : index
  %physical_column = kernel.workgroup.id<y> : index
  %position = index.madd %physical_column, %row_tiles, %physical_row : index
  %row_value = index.cast %row : index to i32
  %column_value = index.cast %column : index to i32
  %view = buffer.view %output[%origin] : buffer -> view<{count // 2}x2xi32>
  view.store %row_value, %view[%position, 0] : i32, view<{count // 2}x2xi32>
  view.store %column_value, %view[%position, 1] : i32, view<{count // 2}x2xi32>
  kernel.return
}}

func.def @reference_{name}(%output: buffer) {{
  %rows = index.constant {row_tiles} : index
  %columns = index.constant {column_tiles} : index
  %group = index.constant {group} : index
  func.call @enumerate_groups(%rows, %columns, %group, %output) : (index, index, index, buffer)
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %storage = check.generate.fill value(-77) : tensor<{count + 32}xi32>
    %output = check.tensor.view %storage offset(64) : tensor<{count + 32}xi32> -> tensor<{count}xi32>
    check.compare<@apply_{name}, @reference_{name}>(%output) : (tensor<{count}xi32>) -> () {{
      check.expect.bitwise actual(%storage) expected(%storage) : tensor<{count + 32}xi32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    matrix_header = '''// Generated by generate_linear_temporal_oracle.py.
// Independent dyadic whole-matrix products, complete guards, and immutable inputs.

amdgpu.target<gfx11-generic> @tensor_projection_temporal_rdna3

amdgpu.target<gfx11-generic> @temporal_projection_wave32 {subgroup_size = 32}

template.decl @projection.linear_temporal_bf16(%rows: index, %inputs: index, %outputs: index, %group_size: index, %rhs_read_ahead: i1, %input: buffer, %weight: buffer, %output: buffer)
'''
    coordinate_header = '''// Generated by generate_linear_temporal_oracle.py.
// Group enumeration is independent of production rounding and division.

amdgpu.target<gfx11-generic> @coordinates_wave32 {subgroup_size = 32}

template.decl @projection.temporal_coordinates(%rows: index, %outputs: index, %group_size: index) -> (index, index)

''' + GROUP_REFERENCE
    return {
        "linear_temporal.loom": matrix_header + "".join(render_matrix(*shape) for shape in SHAPES),
        "linear_temporal_coordinates.loom": coordinate_header + "".join(render_grid(*grid) for grid in GRIDS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in render().items():
        path = Path(__file__).with_name(name)
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"{path} differs from its independent reference")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
