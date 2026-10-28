#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author exact projection epilogues independently of matrix contraction."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import struct

from binary_reference import add_f32, f32, multiply_f32, rounded_float


@dataclass(frozen=True)
class Case:
    # Stable scenario name.
    name: str
    # Epilogue selected by the leaf.
    operation: str
    # Complete rows in the logical tensor.
    rows: int = 32
    # Complete output columns in the logical tensor.
    columns: int = 48
    # Physical base/output layout.
    layout: str = "dense"
    # Immutable bias storage representation.
    bias_type: str = "f32"
    # Runtime scalar strength for adapter composition.
    strength: float = 1.0
    # Whether output reuses the complete base allocation.
    inplace: bool = False


CASES = (
    Case("store_dense", "store", rows=16, columns=32),
    Case("store_column", "store", layout="column"),
    Case("store_padded", "store", layout="padded"),
    Case("bias_bf16", "bias", bias_type="bf16"),
    Case("bias_f32_column", "bias", layout="column"),
    Case("bias_f32_padded", "bias", layout="padded"),
    Case("adapter_half", "adapter", strength=0.5),
    Case("adapter_negative", "adapter", layout="column", strength=-0.625, inplace=True),
    Case("adapter_midpoint", "adapter", layout="padded", strength=1 + 1 / 256),
    Case("adapter_inplace", "adapter", layout="padded", strength=1.25, inplace=True),
    Case("adapter_zero", "adapter", strength=0.0),
    Case("adapter_negative_zero", "adapter", layout="column", strength=-0.0, inplace=True),
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


def geometry(case):
    if case.layout == "column":
        return 1, case.rows, case.rows * case.columns
    stride = case.columns + (16 if case.layout == "padded" else 0)
    return stride, 1, case.rows * stride


def values():
    result = [0.0, -0.0]
    for sign in (-1, 1):
        for mantissa in (1, 3, 7, 15, 63, 127):
            midpoint = sign * (1 + mantissa / 256)
            bits = struct.unpack("<I", struct.pack("<f", midpoint))[0]
            for displacement in (-1, 0, 1):
                result.append(struct.unpack("<f", struct.pack("<I", bits + displacement))[0])
        result.extend(sign * 2.0**exponent for exponent in (-149, -134, -133, -126, -60, 60))
    generator = random.Random(0xE91106)
    result.extend(f32(generator.uniform(-8, 8)) for _ in range(128))
    return result


def evaluate(case, result, bias, base, omitted=""):
    if case.operation == "store":
        return result
    if case.operation == "bias":
        return add_f32(result, bias if omitted == "bias" else rounded_float(bias, 8))
    if case.strength == 0 and omitted != "disabled":
        return base
    delta = result if omitted == "delta" else rounded_float(result, 8)
    scaled = multiply_f32(delta, case.strength)
    if omitted != "scaled":
        scaled = rounded_float(scaled, 8)
    return add_f32(base, scaled)


def observations(case):
    choices = values()
    result = [choices[(row * 37 + column * 11) % len(choices)]
              for row in range(case.rows) for column in range(case.columns)]
    bias = [choices[(column * 7 + 2) % len(choices)] for column in range(case.columns)]
    if case.bias_type == "bf16":
        bias = [rounded_float(value, 8) for value in bias]
    row_stride, column_stride, count = geometry(case)
    base = [123.0] * count
    output = [123.0] * count
    wide = []
    differences = dict.fromkeys(("bias", "delta", "scaled", "disabled"), 0)
    for row in range(case.rows):
        for column in range(case.columns):
            physical = row * row_stride + column * column_stride
            logical = row * case.columns + column
            value = -0.0 if logical % 7 == 0 else rounded_float(choices[(logical * 13 + 1) % len(choices)], 8)
            base[physical] = value
            expected = evaluate(case, result[logical], bias[column], value)
            wide.append(expected)
            output[physical] = rounded_float(expected, 8)
            for omitted in differences:
                alternative = evaluate(case, result[logical], bias[column], value, omitted)
                differences[omitted] += encode([expected], "f32") != encode([alternative], "f32")
    if case.operation == "bias" and case.bias_type == "f32":
        assert differences["bias"]
    if case.operation == "adapter":
        if case.strength == 0:
            assert differences["disabled"]
        elif case.strength not in (0.5, 1.0):
            # Power-of-two scaling can commute with BF16 rounding. The other
            # strengths separately distinguish both intermediate boundaries.
            assert differences["delta"], case.name
            assert differences["scaled"], case.name
    return result, bias, base, wide, output, differences


def render_case(case):
    name = case.name
    result, bias, base, wide, output, _ = observations(case)
    logical_count = len(result)
    row_stride, column_stride, count = geometry(case)
    arrays = (("input", result, "f32"), ("bias", bias, case.bias_type),
              ("strength", [case.strength], "f32"), ("base", base, "bf16"),
              ("wide", wide, "f32"), ("expected", output, "bf16"))
    rodata = "\n\n".join(f'global.rodata.def @{name}_{key} = align({4 if dtype == "f32" else 2}) bytes("{encode(data, dtype)}")'
                         for key, data, dtype in arrays)
    initialize = []
    for key, data, dtype in arrays[:4]:
        initialize.extend((f"  %{key}_bytes = index.constant {len(data) * (4 if dtype == 'f32' else 2)} : offset",
                           f"  %{key}_values = global.load @{name}_{key} : buffer",
                           f"  buffer.copy %{key}_values[%origin], %{key}[%origin], %{key}_bytes"))
    call = ""
    selected = "%result"
    if case.operation == "bias":
        call = f'''  %bias_is_bf16 = scalar.constant {'true' if case.bias_type == 'bf16' else 'false'} : i1
  %combined = template.apply<@projection.bias_storage_bf16>(%columns, %output_origin, %bias_is_bf16, %result, %bias) : (index, index, i1, vector<8xf32>, buffer) -> (vector<8xf32>)
'''
        selected = "%combined"
    elif case.operation == "adapter":
        call = '''  %combined = template.apply<@projection.adapter_epilogue_bf16>(%rows, %columns, %row_origin, %output_origin, %layout, %result, %strength, %base) : (index, index, index, index, encoding<layout>, vector<8xf32>, buffer, buffer) -> (vector<8xf32>)
'''
        selected = "%combined"
    storage = []
    expectations = []
    for key, length, dtype in (("input", logical_count, "f32"), ("bias", case.columns, case.bias_type),
                               ("strength", 1, "f32"), ("base", count, "bf16"),
                               ("wide", logical_count, "f32"), ("output", count, "bf16")):
        item_bytes = 4 if dtype == "f32" else 2
        guard_type = "f32" if dtype == "f32" else "i16"
        guard_value = "123.25" if dtype == "f32" else "17142"
        owner = "base" if key == "output" and case.inplace else key
        if owner == key:
            storage.append(f"    %{key}_storage = check.generate.fill value({guard_value}) : tensor<{length + 32}x{guard_type}>")
            expectations.append(f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{length + 32}x{guard_type}>")
        storage.append(f"    %{key} = check.tensor.view %{owner}_storage offset({item_bytes * 16}) : tensor<{length + 32}x{guard_type}> -> tensor<{length}x{dtype}>")
    types = f"tensor<{logical_count}xf32>, tensor<{case.columns}x{case.bias_type}>, tensor<1xf32>, tensor<{count}xbf16>"
    return f'''
{rodata}

func.def @initialize_{name}(%input: buffer, %bias: buffer, %strength: buffer, %base: buffer) {{
  %origin = index.constant 0 : offset
{chr(10).join(initialize)}
  func.return
}}

kernel.def target(@projection_rdna3_wave32) @apply_{name}() {{
  %row_tiles = index.constant {case.rows // 16} : index
  %output_tiles = index.constant {case.columns // 16} : index
  %threads = index.constant 32 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%row_tiles, %output_tiles, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %bias: buffer, %strength: buffer, %base: buffer, %wide: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %rows = index.constant {case.rows} : index
  %columns = index.constant {case.columns} : index
  %tile = index.constant 16 : index
  %last_row = index.constant {case.rows // 16 - 1} : index
  %last_output = index.constant {case.columns // 16 - 1} : index
  %grid_row = kernel.workgroup.id<x> : index
  %grid_output = kernel.workgroup.id<y> : index
  %row_tile = index.sub %last_row, %grid_row : index
  %output_tile = index.sub %last_output, %grid_output : index
  %row_origin = index.mul %row_tile, %tile : index
  %output_origin = index.mul %output_tile, %tile : index
  %input_view = buffer.view %input[%origin] : buffer -> view<[%rows]x[%columns]xf32>
  %result = vector.fragment.load<result> %input_view[%row_origin, %output_origin] shape [%tile, %tile] : view<[%rows]x[%columns]xf32> -> vector<8xf32>
  %row_stride = index.constant {row_stride} : index
  %column_stride = index.constant {column_stride} : index
  %layout = encoding.layout.strided [%row_stride, %column_stride] : encoding<layout>
{call}  %wide_view = buffer.view %wide[%origin] : buffer -> view<[%rows]x[%columns]xf32>
  vector.fragment.store<result> {selected}, %wide_view[%row_origin, %output_origin] shape [%tile, %tile] : vector<8xf32>, view<[%rows]x[%columns]xf32>
  template.apply<@projection.store_bf16>(%rows, %columns, %row_origin, %output_origin, %layout, {selected}, %output) : (index, index, index, index, encoding<layout>, vector<8xf32>, buffer)
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %bias: buffer, %strength: buffer, %base: buffer, %wide: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %wide_bytes = index.constant {logical_count * 4} : offset
  %output_bytes = index.constant {count * 2} : offset
  %wide_values = global.load @{name}_wide : buffer
  %output_values = global.load @{name}_expected : buffer
  buffer.copy %wide_values[%origin], %wide[%origin], %wide_bytes
  buffer.copy %output_values[%origin], %output[%origin], %output_bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
{chr(10).join(storage)}
    check.generate<@initialize_{name}>(%input, %bias, %strength, %base) : ({types}) -> ()
    check.compare<@apply_{name}, @reference_{name}>(%input, %bias, %strength, %base, %wide, %output) : ({types}, tensor<{logical_count}xf32>, tensor<{count}xbf16>) -> () {{
{chr(10).join(expectations)}
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    header = '''// Generated by generate_linear_epilogue_oracle.py.
// Rational scalar expressions author every F32/BF16 boundary independently.

amdgpu.target<gfx11-generic> @tensor_projection_rdna3

amdgpu.target<gfx11-generic> @projection_rdna3_wave32 {subgroup_size = 32}

template.decl @projection.store_bf16(%rows: index, %outputs: index, %row_origin: index, %output_origin: index, %output_layout: encoding<layout>, %result: vector<8xf32>, %output: buffer)

template.decl @projection.bias_storage_bf16(%outputs: index, %output_origin: index, %weight_is_bf16: i1, %result: vector<8xf32>, %bias: buffer) -> (vector<8xf32>)

template.decl @projection.adapter_epilogue_bf16(%rows: index, %outputs: index, %row_origin: index, %output_origin: index, %base_layout: encoding<layout>, %result: vector<8xf32>, %strength: buffer, %base: buffer) -> (vector<8xf32>)
'''
    return header + "".join(render_case(case) for case in CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("linear_epilogue_test.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its scalar reference")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
