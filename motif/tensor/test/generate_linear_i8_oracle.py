#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author signed integer projections with independent scalar rounding."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct

from binary_reference import f32, multiply_f32, rounded_float


@dataclass(frozen=True)
class Shape:
    # Stable scenario name.
    name: str
    # Activation rows, a positive multiple of sixteen.
    rows: int
    # Integer reduction width, a positive multiple of sixty-four.
    inputs: int
    # Output channels, a positive multiple of sixty-four.
    outputs: int
    # Logical row tiles visited together before advancing output columns.
    group: int = 1
    # Physical output layout.
    layout: str = "dense"
    # Operand construction for exact conversion boundaries.
    pattern: str = "mixed"


SHAPES = (
    Shape("minimum_k64", 16, 64, 64),
    Shape("tail32_k128", 32, 128, 64),
    Shape("tail48_k192", 48, 192, 128),
    Shape("tail64", 64, 256, 64),
    Shape("tail80_column", 80, 128, 192, 2, "column"),
    Shape("tail96", 96, 192, 64),
    Shape("tail112", 112, 128, 64),
    Shape("full", 128, 256, 128),
    Shape("partial_group", 144, 192, 192, 2, "padded"),
    Shape("multiple_groups", 272, 128, 64, 2),
    Shape("oversized_group", 16, 128, 64, 8),
    Shape("midpoints", 16, 64, 64, pattern="midpoints"),
    Shape("model6144", 16, 6144, 64),
    Shape("model16384", 16, 16384, 64, pattern="large"),
)


def encode(values, dtype):
    if dtype == "i8":
        return struct.pack(f"<{len(values)}b", *values).hex()
    if dtype == "f32":
        return struct.pack(f"<{len(values)}f", *values).hex()
    words = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
    assert all(word & 65535 == 0 for word in words)
    return struct.pack(f"<{len(words)}H", *(word >> 16 for word in words)).hex()


def observations(shape):
    inputs = [((row * 53 + column * 31) % 256) - 128
              for row in range(shape.rows) for column in range(shape.inputs)]
    weights = [((channel * 41 + column * 17) % 256) - 128
               for channel in range(shape.outputs) for column in range(shape.inputs)]
    row_choices = (0.0, 0.125, -0.25, f32(0.1), f32(0.001), 1.0, 0.03125)
    channel_choices = (0.0625, f32(0.3), -0.125, 1.0, f32(0.009), 0.5, 0.0)
    row_scales = [row_choices[row % len(row_choices)] for row in range(shape.rows)]
    channel_scales = [channel_choices[column % len(channel_choices)]
                      for column in range(shape.outputs)]
    if shape.pattern == "midpoints":
        inputs = ([127, 1] + [0] * (shape.inputs - 2)) * shape.rows
        weights = [value for channel in range(shape.outputs)
                   for value in ([2 if channel % 4 < 2 else -2,
                                  (3 if channel % 2 == 0 else 5)
                                  * (1 if channel % 4 < 2 else -1)]
                                 + [0] * (shape.inputs - 2))]
        row_scales = [1.0] * shape.rows
        channel_scales = [1.0] * shape.outputs
    elif shape.pattern == "large":
        inputs = [127 if column else 126
                  for _ in range(shape.rows) for column in range(shape.inputs)]
        weights = [(127 if channel % 2 == 0 else -128)
                   for channel in range(shape.outputs) for _ in range(shape.inputs)]
    if shape.layout == "column":
        row_stride, column_stride, count = 1, shape.rows, shape.rows * shape.outputs
    else:
        row_stride = shape.outputs + (16 if shape.layout == "padded" else 0)
        column_stride, count = 1, shape.rows * row_stride
    expected = [123.0] * count
    for row in range(shape.rows):
        activation = inputs[row * shape.inputs:(row + 1) * shape.inputs]
        for channel in range(shape.outputs):
            weight = weights[channel * shape.inputs:(channel + 1) * shape.inputs]
            total = sum(left * right for left, right in zip(activation, weight))
            assert -(1 << 31) <= total < (1 << 31)
            scale = multiply_f32(row_scales[row], channel_scales[channel])
            value = multiply_f32(rounded_float(total, 24), scale)
            expected[row * row_stride + channel * column_stride] = rounded_float(value, 8)
    return inputs, row_scales, weights, channel_scales, expected, row_stride, column_stride


def render_case(shape):
    inputs, row_scales, weights, channel_scales, output, row_stride, column_stride = observations(shape)
    name = shape.name
    arguments = (("input", inputs, "i8", 1),
                 ("input_scales", row_scales, "f32", 4),
                 ("weight", weights, "i8", 1),
                 ("weight_scales", channel_scales, "f32", 4))
    rodata = "\n\n".join(
        f'global.rodata.def @{name}_{key} = align({width}) bytes("{encode(values, dtype)}")'
        for key, values, dtype, width in (*arguments, ("expected", output, "bf16", 2)))
    initialize = []
    storage = []
    expectations = []
    for key, values, dtype, width in (*arguments, ("output", output, "bf16", 2)):
        count = len(values)
        storage_type = {"i8": "i8", "f32": "i32", "bf16": "i16"}[dtype]
        sentinel = 17142 if key == "output" else -77
        storage.extend([
            f"    %{key}_storage = check.generate.fill value({sentinel}) : tensor<{count + 32}x{storage_type}>",
            f"    %{key} = check.tensor.view %{key}_storage offset({16 * width}) : tensor<{count + 32}x{storage_type}> -> tensor<{count}x{dtype}>",
        ])
        expectations.append(f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{count + 32}x{storage_type}>")
        if key != "output":
            initialize.extend([
                f"  %{key}_bytes = index.constant {count * width} : offset",
                f"  %{key}_values = global.load @{name}_{key} : buffer",
                f"  buffer.copy %{key}_values[%origin], %{key}[%origin], %{key}_bytes",
            ])
    names = ", ".join("%" + item[0] for item in arguments)
    declarations = ", ".join("%" + item[0] + ": buffer" for item in arguments)
    tensors = ", ".join(f"tensor<{len(values)}x{dtype}>" for _, values, dtype, _ in arguments)
    return f'''
{rodata}

func.def @initialize_{name}({declarations}) {{
  %origin = index.constant 0 : offset
{chr(10).join(initialize)}
  func.return
}}

kernel.def target(@i8_projection_wave32) @apply_{name}() {{
  %row_tiles = index.constant {(shape.rows + 127) // 128} : index
  %output_tiles = index.constant {(shape.outputs + 127) // 128} : index
  %threads = index.constant 256 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%row_tiles, %output_tiles, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch({declarations}, %output: buffer) {{
  %rows = index.constant {shape.rows} : index
  %inputs = index.constant {shape.inputs} : index
  %outputs = index.constant {shape.outputs} : index
  %group = index.constant {shape.group} : index
  %row_stride = index.constant {row_stride} : index
  %column_stride = index.constant {column_stride} : index
  %layout = encoding.layout.strided [%row_stride, %column_stride] : encoding<layout>
  template.apply<@projection.linear_tiled_i8>(%rows, %inputs, %outputs, %group, %layout, {names}, %output) : (index, index, index, index, encoding<layout>, buffer, buffer, buffer, buffer, buffer)
  kernel.return
}}

func.def @reference_{name}({declarations}, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {len(output) * 2} : offset
  %expected = global.load @{name}_expected : buffer
  buffer.copy %expected[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
{chr(10).join(storage)}
    check.generate<@initialize_{name}>({names}) : ({tensors}) -> ()
    check.compare<@apply_{name}, @reference_{name}>({names}, %output) : ({tensors}, tensor<{len(output)}xbf16>) -> () {{
{chr(10).join(expectations)}
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    header = '''// Generated by generate_linear_i8_oracle.py.
// Integer dot products, scalar F32/BF16 rounding, and complete storage guards.

amdgpu.target<gfx11-generic> @tensor_projection_tiled_rdna3

amdgpu.target<gfx11-generic> @i8_projection_wave32 {subgroup_size = 32}

template.decl @projection.linear_tiled_i8(%rows: index, %inputs: index, %outputs: index, %row_tile_group_size: index, %layout: encoding<layout>, %input: buffer, %input_scales: buffer, %weight: buffer, %weight_scales: buffer, %output: buffer)
'''
    return header + "".join(render_case(shape) for shape in SHAPES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("linear_i8.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its independent reference")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
