#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author twelve-layer mixing with exact rational F32/BF16 boundaries."""

import argparse
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import random

from binary_reference import add_f32, f32, multiply_f32, rounded_float
from binary_reference import encode


@dataclass(frozen=True)
class Case:
    # Stable fixture identity.
    name: str
    # Token rows in layer-major input storage.
    rows: int
    # Adjacent channels within each layer.
    width: int
    # Active low-rank factor extent; zero selects the base-only contract.
    rank: int = 0
    # Checkpoint coefficient storage representation.
    weight_type: str = "f32"
    # Runtime factor strength; either signed zero suppresses factor reads.
    strength: float = 1.0
    # Explicit arithmetic corpus, otherwise deterministic varied inputs.
    corpus: str = "varied"


CASES = (
    Case("base_bf16", 3, 17, weight_type="bf16"),
    Case("base_f32", 2, 65),
    Case("model_width", 1, 2560),
    Case("ordered", 1, 3, weight_type="bf16", corpus="ordered"),
    Case("rank1", 2, 19, 1, strength=1.25),
    Case("rank7", 2, 31, 7, "bf16", -0.625),
    Case("rank16", 2, 33, 16, strength=0.5),
    Case("rank32", 2, 65, 32, strength=1 + 1 / 256),
    Case("rank33", 3, 17, 33, "bf16", 1.25),
    Case("zero_strength", 1, 17, 32, strength=0.0, corpus="tiny"),
    Case("negative_zero_strength", 1, 17, 32, "bf16", -0.0, "tiny"),
)

BOUNDARIES = ("base_weight", "base", "down_weight", "low", "up_weight",
              "delta", "scaled", "accumulator", "order")


def fused(left, right, accumulator):
    exact = Fraction(left) * Fraction(right) + Fraction(accumulator)
    return rounded_float(exact if exact else left * right + accumulator, 24)


def dot(values, coefficients, boundary, omitted):
    total = 0.0
    pairs = list(zip(values, coefficients))
    if omitted == "order":
        pairs.reverse()
    for value, stored in pairs:
        coefficient = stored if omitted == boundary else rounded_float(stored, 8)
        total = fused(value, coefficient, total)
        if omitted == "accumulator":
            total = rounded_float(total, 8)
    return total


def evaluate(case, values, weight, down, up, omitted=""):
    base = dot(values, weight, "base_weight", omitted)
    if omitted != "base":
        base = rounded_float(base, 8)
    if not case.rank or case.strength == 0:
        return rounded_float(base, 8)
    low = [dot(values, down[element * 12:(element + 1) * 12],
               "down_weight", omitted) for element in range(case.rank)]
    if omitted != "low":
        low = [rounded_float(value, 8) for value in low]
    delta = dot(low, up, "up_weight", omitted)
    if omitted != "delta":
        delta = rounded_float(delta, 8)
    scaled = multiply_f32(delta, case.strength)
    if omitted != "scaled":
        scaled = rounded_float(scaled, 8)
    return rounded_float(add_f32(base, scaled), 8)


def observations(case):
    generator = random.Random(0x12BF16)
    inputs = [rounded_float(generator.uniform(-4, 4), 8)
              for _ in range(case.rows * 12 * case.width)]
    weight = [f32(generator.uniform(-2, 2)) for _ in range(12)]
    down = [f32(generator.uniform(-2, 2)) for _ in range(case.rank * 12)]
    up = [f32(generator.uniform(-2, 2)) for _ in range(case.rank)]
    # Include exact opposite tie directions in F32 checkpoint conversion.
    weight[:4] = [1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, -1 - 3 / 256]
    if case.corpus == "ordered":
        columns = ([2.0**24, 1.0, -2.0**24] + [0.0] * 9,
                   [1.0] + [2.0**-9] * 11,
                   [-2.0**24, -1.0, 2.0**24] + [0.0] * 9)
        inputs = [columns[channel][layer] for layer in range(12)
                  for channel in range(case.width)]
        weight = [1.0] * 12
    elif case.corpus == "tiny":
        choices = (-2.0**-133, 2.0**-133, -2.0**-131, 2.0**-131)
        inputs = [choices[channel % 4] if layer == 0 else 0.0
                  for layer in range(12) for channel in range(case.width)]
        weight = [0.25] + [0.0] * 11
    if case.weight_type == "bf16":
        weight = [rounded_float(value, 8) for value in weight]
    if not case.rank or case.strength == 0:
        # These operands have no required factor extent on the inactive path.
        down, up = [float("nan")], [float("nan")]
    strength = [case.strength if case.rank else float("nan")]
    gathered, base, output = [], [], []
    differences = dict.fromkeys(BOUNDARIES, 0)
    for row in range(case.rows):
        for channel in range(case.width):
            values = [inputs[(row * 12 + layer) * case.width + channel]
                      for layer in range(12)]
            gathered.extend(values)
            base.append(rounded_float(dot(values, weight, "base_weight", ""), 8))
            expected = evaluate(case, values, weight, down, up)
            output.append(expected)
            for omitted in BOUNDARIES:
                alternative = evaluate(case, values, weight, down, up, omitted)
                differences[omitted] += encode([expected], "bf16") != encode([alternative], "bf16")
    if case.corpus == "tiny":
        assert encode(output, "bf16").startswith("0080"), case.name
        assert encode(output, "bf16") == encode(base, "bf16")
    return (("input", inputs, "bf16"), ("weight", weight, case.weight_type),
            ("down", down, "f32"), ("up", up, "f32"),
            ("strength", strength, "f32"), ("gathered", gathered, "f32"),
            ("base", base, "bf16"), ("output", output, "bf16")), differences


def render_case(case, arrays):
    name = case.name
    input_arrays, result_arrays = arrays[:5], arrays[5:]
    rodata = "\n\n".join(f'global.rodata.def @{name}_{key} = align({4 if dtype == "f32" else 2}) bytes("{encode(data, dtype)}")'
                         for key, data, dtype in arrays)

    def copies(selected):
        return "\n".join(
            f"  %{key}_bytes = index.constant {len(data) * (4 if dtype == 'f32' else 2)} : offset\n"
            f"  %{key}_values = global.load @{name}_{key} : buffer\n"
            f"  buffer.copy %{key}_values[%origin], %{key}[%origin], %{key}_bytes"
            for key, data, dtype in selected)

    storage, expectations = [], []
    for key, data, dtype in arrays:
        length = len(data)
        item_bytes = 4 if dtype == "f32" else 2
        guard_type, guard_value = ("f32", "123.25") if dtype == "f32" else ("i16", "17142")
        storage.extend((f"    %{key}_storage = check.generate.fill value({guard_value}) : tensor<{length + 32}x{guard_type}>",
                        f"    %{key} = check.tensor.view %{key}_storage offset({item_bytes * 16}) : tensor<{length + 32}x{guard_type}> -> tensor<{length}x{dtype}>"))
        expectations.append(f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{length + 32}x{guard_type}>")
    arguments = ", ".join(f"%{key}: buffer" for key, _, _ in arrays)
    operands = ", ".join(f"%{key}" for key, _, _ in arrays)
    types = ", ".join(f"tensor<{len(data)}x{dtype}>" for _, data, dtype in arrays)
    input_arguments = ", ".join(f"%{key}: buffer" for key, _, _ in input_arrays)
    input_operands = ", ".join(f"%{key}" for key, _, _ in input_arrays)
    input_types = ", ".join(f"tensor<{len(data)}x{dtype}>" for _, data, dtype in input_arrays)
    threads = 256 if case.width == 2560 else 32
    return f'''
{rodata}

func.def @initialize_{name}({input_arguments}) {{
  %origin = index.constant 0 : offset
{copies(input_arrays)}
  func.return
}}

kernel.def target(@layer_mix_wave32) @apply_{name}() {{
  %rows = index.constant {case.rows} : index
  %groups = index.constant {(case.width + threads - 1) // threads} : index
  %threads = index.constant {threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %groups, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch({arguments}) {{
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %layers = index.constant 12 : index
  %rows = index.constant {case.rows} : index
  %width = index.constant {case.width} : index
  %rank = index.constant {case.rank} : index
  %threads = index.constant {threads} : index
  %row = kernel.workgroup.id<x> : index
  %group = kernel.workgroup.id<y> : index
  %lane = kernel.workitem.id<x> : index
  %channel = index.madd %group, %threads, %lane : index
  %valid = index.cmp ult, %channel, %width : index
  scf.if %valid {{
    %adapted = scalar.constant {str(bool(case.rank)).lower()} : i1
    %weight_is_bf16 = scalar.constant {str(case.weight_type == 'bf16').lower()} : i1
    %values = template.apply<@projection.gather_12_bf16>(%rows, %width, %row, %channel, %input) : (index, index, index, index, buffer) -> (vector<12xf32>)
    %base_value = template.apply<@projection.dot_12_storage_bf16>(%c1, %c0, %weight_is_bf16, %values, %weight) : (index, index, i1, vector<12xf32>, buffer) -> (bf16)
    %result = template.apply<@projection.layer_mix_12_bf16>(%rows, %width, %rank, %row, %channel, %adapted, %weight_is_bf16, %input, %weight, %down, %up, %strength) : (index, index, index, index, index, i1, i1, buffer, buffer, buffer, buffer, buffer) -> (bf16)
    %gathered_view = buffer.view %gathered[%origin] : buffer -> view<[%rows]x[%width]x12xf32>
    scf.for %layer = [%c0 to %layers step %c1] unroll {{
      %value = vector.extract %values[%layer] : vector<12xf32> -> f32
      view.store %value, %gathered_view[%row, %channel, %layer] : f32, view<[%rows]x[%width]x12xf32>
    }}
    %base_view = buffer.view %base[%origin] : buffer -> view<[%rows]x[%width]xbf16>
    %output_view = buffer.view %output[%origin] : buffer -> view<[%rows]x[%width]xbf16>
    view.store %base_value, %base_view[%row, %channel] : bf16, view<[%rows]x[%width]xbf16>
    view.store %result, %output_view[%row, %channel] : bf16, view<[%rows]x[%width]xbf16>
  }}
  kernel.return
}}

func.def @reference_{name}({arguments}) {{
  %origin = index.constant 0 : offset
{copies(result_arrays)}
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
{chr(10).join(storage)}
    check.generate<@initialize_{name}>({input_operands}) : ({input_types}) -> ()
    check.compare<@apply_{name}, @reference_{name}>({operands}) : ({types}) -> () {{
{chr(10).join(expectations)}
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    header = '''// Generated by generate_layer_mix_oracle.py.
// Exact rational arithmetic authors each ordered F32/BF16 boundary.

amdgpu.target<gfx11-generic> @layer_mix_wave32 {subgroup_size = 32}

template.decl @projection.gather_12_bf16(%rows: index, %width: index, %row: index, %channel: index, %input: buffer) -> (vector<12xf32>)

template.decl @projection.dot_12_storage_bf16(%outputs: index, %row: index, %weight_is_bf16: i1, %values: vector<12xf32>, %weight: buffer) -> (bf16)

template.decl @projection.layer_mix_12_bf16(%rows: index, %width: index, %rank: index, %row: index, %channel: index, %adapted: i1, %weight_is_bf16: i1, %input: buffer, %weight: buffer, %down: buffer, %up: buffer, %strength: buffer) -> (bf16)
'''
    results = [observations(case) for case in CASES]
    for boundary in BOUNDARIES:
        assert any(differences[boundary] for _, differences in results), boundary
    return header + "".join(render_case(case, arrays)
                            for case, (arrays, _) in zip(CASES, results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    path = Path(__file__).with_name("layer_mix_test.loom")
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its independent reference")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
