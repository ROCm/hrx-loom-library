#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author cooperative projection references and independent traversal checks."""

import argparse
from pathlib import Path

from generate_linear_oracle import Shape, encode, observations, output_geometry


SHAPES = (
    (Shape("minimum", 16, 256, 64), 1),
    (Shape("tail32", 32, 256, 64), 1),
    (Shape("tail48", 48, 256, 64), 1),
    (Shape("full", 64, 512, 128), 1),
    (Shape("column_tail", 80, 512, 128, layout="column"), 2),
    (Shape("partial_group", 144, 256, 192, layout="padded"), 2),
    (Shape("multiple_groups", 272, 768, 128), 3),
    (Shape("oversized_group", 80, 256, 64), 8),
    (Shape("model6144", 16, 6144, 64), 1),
    (Shape("model16384", 16, 16384, 64), 1),
)

GRIDS = (
    (1, 1, 1), (1, 3, 8), (2, 1, 1), (3, 5, 2), (5, 3, 3),
    *((8 + remainder, 3, 8) for remainder in range(8)),
    (17, 2, 8), (33, 4, 8), (66, 24, 8), (72, 96, 8), (1024, 256, 8),
)


def render_matrix(shape, group):
    inputs, weights, _, output, _ = observations(shape)
    name = shape.name
    row_stride, column_stride, output_count = output_geometry(shape)
    input_count, weight_count = len(inputs), len(weights)
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

kernel.def target(@tiled_projection_wave32) @apply_{name}() {{
  %row_tiles = index.constant {(shape.rows + 63) // 64} : index
  %output_tiles = index.constant {shape.outputs // 64} : index
  %threads = index.constant 256 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%row_tiles, %output_tiles, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer, %output: buffer) {{
  %rows = index.constant {shape.rows} : index
  %inputs = index.constant {shape.inputs} : index
  %outputs = index.constant {shape.outputs} : index
  %group = index.constant {group} : index
  %row_stride = index.constant {row_stride} : index
  %column_stride = index.constant {column_stride} : index
  %layout = encoding.layout.strided [%row_stride, %column_stride] : encoding<layout>
  template.apply<@projection.linear_tiled_bf16>(%rows, %inputs, %outputs, %group, %layout, %input, %weight, %output) : (index, index, index, index, encoding<layout>, buffer, buffer, buffer)
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
    count = rows * columns * 2
    return f'''
kernel.def target(@coordinates_wave32) @apply_{name}() {{
  %rows = index.constant {rows} : index
  %columns = index.constant {columns} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %columns, %unit) workgroup_size(%unit, %unit, %unit) : index
}} launch(%output: buffer) {{
  %rows = index.constant {rows} : index
  %columns = index.constant {columns} : index
  %group = index.constant {group} : index
  %row, %column = template.apply<@projection.linear_tiled_coordinates>(%rows, %columns, %group) : (index, index, index) -> (index, index)
  %origin = index.constant 0 : offset
  %physical_row = kernel.workgroup.id<x> : index
  %physical_column = kernel.workgroup.id<y> : index
  %position = index.madd %physical_column, %rows, %physical_row : index
  %row_value = index.cast %row : index to i32
  %column_value = index.cast %column : index to i32
  %view = buffer.view %output[%origin] : buffer -> view<{rows * columns}x2xi32>
  view.store %row_value, %view[%position, 0] : i32, view<{rows * columns}x2xi32>
  view.store %column_value, %view[%position, 1] : i32, view<{rows * columns}x2xi32>
  kernel.return
}}

func.def @reference_{name}(%output: buffer) {{
  %rows = index.constant {rows} : index
  %columns = index.constant {columns} : index
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
    matrix_header = '''// Generated by generate_linear_tiled_oracle.py.
// Independent dyadic whole-matrix products, complete guards, and immutable inputs.

amdgpu.target<gfx11-generic> @tensor_projection_tiled_rdna3

amdgpu.target<gfx11-generic> @tiled_projection_wave32 {subgroup_size = 32}

template.decl @projection.linear_tiled_bf16(%rows: index, %inputs: index, %outputs: index, %row_tile_group_size: index, %output_layout: encoding<layout>, %input: buffer, %weight: buffer, %output: buffer)
'''
    coordinate_header = '''// Generated by generate_linear_tiled_oracle.py.
// Group enumeration is independent of the production division/remainder mapping.

amdgpu.target<gfx11-generic> @coordinates_wave32 {subgroup_size = 32}

template.decl @projection.linear_tiled_coordinates(%row_tiles: index, %output_tiles: index, %group_size: index) -> (index, index)

func.def @enumerate_groups(%rows: index, %columns: index, %group: index, %output: buffer) {
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %count = index.mul %rows, %columns : index
  %view = buffer.view %output[%origin] : buffer -> view<[%count]x2xi32>
  %end = scf.for %first_row = [%c0 to %rows step %group](%group_position = %c0 : index) -> (index) {
    %limit = index.add %first_row, %group : index
    %last_row = index.min %limit, %rows : index
    %next_group = scf.for %column = [%c0 to %columns step %c1](%column_position = %group_position : index) -> (index) {
      %next_column = scf.for %row = [%first_row to %last_row step %c1](%position = %column_position : index) -> (index) {
        // Group/column/row enumeration visits every pair exactly once.
        %ordinal = index.assume %position [ge(%position, 0), lt(%position, %count)] : index
        %row_value = index.cast %row : index to i32
        %column_value = index.cast %column : index to i32
        view.store %row_value, %view[%ordinal, 0] : i32, view<[%count]x2xi32>
        view.store %column_value, %view[%ordinal, 1] : i32, view<[%count]x2xi32>
        %next = index.add %position, %c1 : index
        scf.yield %next : index
      }
      scf.yield %next_column : index
    }
    scf.yield %next_group : index
  }
  func.return
}
'''
    return {
        "linear_tiled.loom": matrix_header + "".join(render_matrix(shape, group) for shape, group in SHAPES),
        "linear_coordinates.loom": coordinate_header + "".join(render_grid(*grid) for grid in GRIDS),
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
