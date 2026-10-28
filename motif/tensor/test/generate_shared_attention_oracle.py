#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author shared attention from dense contractions and separate gate algebra."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random
import struct

from binary_reference import f32, rounded_float
from generate_linear_epilogue_oracle import encode


@dataclass(frozen=True)
class Case:
    # Stable numerical/ownership construction.
    pattern: str
    # Key rows shared by the four waves per iteration.
    tile: int
    # Complete query/key tiles in the tensor.
    tiles: int = 1
    # Independent groups of four query heads sharing one KV head.
    heads: int = 1
    # Whether context and sigmoid round separately before multiplication.
    gated: bool = False

    @property
    def name(self):
        return f"k{self.tile}_{self.pattern}"

    @property
    def tokens(self):
        return self.tile * self.tiles


CASES = tuple(
    case for tile in (16, 32) for case in (
        Case("dense", tile),
        Case("groups", tile, 3, 2, True),
        Case("model_heads", tile, 1, 12),
        Case("leading", tile, 3),
        Case("middle", tile, 3, gated=True),
        Case("trailing", tile, 3),
        Case("sparse", tile, 3, gated=True),
        Case("single", tile, 3),
        Case("zero", tile),
        Case("uniform", tile, 2, gated=True),
        Case("late", tile, 4),
        Case("negative", tile, 3),
        Case("probability", tile),
    )
)


def margin(value):
    """Distance from a positive normal value to its BF16 rounding boundaries."""
    rounded = rounded_float(value, 8)
    word = struct.unpack("<I", struct.pack("<f", rounded))[0] >> 16
    previous = struct.unpack("<f", struct.pack("<I", (word - 1) << 16))[0]
    following = struct.unpack("<f", struct.pack("<I", (word + 1) << 16))[0]
    return min(value - (previous + rounded) / 2,
               (rounded + following) / 2 - value)


def inputs(case):
    generator = random.Random(0x4EAD + case.tile)
    query_count = case.tokens * case.heads * 512
    key_count = case.tokens * case.heads * 128
    query = [generator.randrange(-127, 128) / 128 for _ in range(query_count)]
    key, value = ([generator.randrange(-127, 128) / 128 for _ in range(key_count)]
                  for _ in range(2))
    gate = [generator.randrange(-64, 65) / 16 for _ in range(query_count)]
    mask = [1] * case.tokens
    if case.pattern in ("leading", "negative"):
        mask[:2 * case.tile] = [0] * (2 * case.tile)
    elif case.pattern == "middle":
        mask[case.tile:2 * case.tile] = [0] * case.tile
    elif case.pattern == "trailing":
        mask[case.tile:] = [0] * (case.tokens - case.tile)
    elif case.pattern == "sparse":
        mask = [(0, 128, 0, 255, 1, 0, 2)[row % 7] for row in range(case.tokens)]
    elif case.pattern == "single":
        mask = [0] * (case.tokens - 1) + [255]
    if case.pattern in ("zero", "uniform"):
        query = [0.0] * query_count
        if case.pattern == "zero":
            value = [0.0] * key_count
        else:
            gate = [0.0] * query_count
    if case.pattern in ("late", "negative", "probability"):
        query, key = [0.0] * query_count, [0.0] * key_count
        for row in range(case.tokens):
            for head in range(case.heads * 4):
                query[(row * case.heads * 4 + head) * 128 + 127] = (
                    2.0**56 if case.pattern == "negative" else 4.0)
            for head in range(case.heads):
                key[(row * case.heads + head) * 128 + 127] = (
                    -2.0**56 if case.pattern == "negative" else
                    (row // case.tile) * 8 + (row % case.tile) / 8)
        if case.pattern == "negative":
            value = [1.0] * key_count
        elif case.pattern == "probability":
            # The maximum is exactly zero. Select negative dyadic keys whose
            # exp values have unambiguous BF16 rounding under the fixed native
            # score/exp budget, before executing any device implementation.
            choices = [number / 32 for number in range(-128, 0)]
            choices = [number for number in choices
                       if margin(math.exp(number * 4 / math.sqrt(128))) >
                       math.exp(number * 4 / math.sqrt(128)) * 2.0**-14]
            assert len(choices) >= case.tokens - 1
            for row in range(case.tokens):
                selected = choices[row] if row < case.tokens - 1 else 0.0
                key[row * 128 + 127] = selected
                probability = math.exp(selected * 4 / math.sqrt(128))
                sign = 1.0 if rounded_float(probability, 8) >= probability else -1.0
                # Keep the final context away from a BF16 midpoint while
                # retaining the independently observable probability error.
                if row == (2 if case.tile == 16 else 5):
                    sign = -sign
                for channel in range(128):
                    value[row * 128 + channel] = sign
    return query, key, value, mask, gate


def context(case, query, key, value, mask, row, head, *, rounded=False,
            omit_channel=False, ignore_mask=False, wrong_head=False):
    key_head = head // 4
    if wrong_head:
        key_head = (key_head + 1) % case.heads
    channels = range(127 if omit_channel else 128)
    query_start = (row * case.heads * 4 + head) * 128
    selected = [position for position, valid in enumerate(mask) if valid or ignore_mask]
    scores = [math.fsum(query[query_start + channel] *
                        key[(position * case.heads + key_head) * 128 + channel]
                        for channel in channels) / math.sqrt(128)
              for position in selected]
    maximum = max(scores)
    probabilities = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(probabilities)
    weights = ([rounded_float(probability, 8) for probability in probabilities]
               if rounded else probabilities)
    return [math.fsum(weight * value[(position * case.heads + key_head) * 128 + channel]
                      for position, weight in zip(selected, weights)) / denominator
            for channel in range(128)]


def epilogue(values, gates, gated, omitted=""):
    if not gated:
        return [rounded_float(value, 8) for value in values]
    result = []
    for value, gate in zip(values, gates):
        context_value = value if omitted == "context" else rounded_float(value, 8)
        sigmoid = 1 / (1 + math.exp(-gate))
        if omitted != "sigmoid":
            sigmoid = rounded_float(sigmoid, 8)
        result.append(rounded_float(context_value * sigmoid, 8))
    return result


def observations(case):
    query, key, value, mask, gate = inputs(case)
    round_probability = case.pattern == "probability"
    # Twenty bits of input-scaled native contraction accuracy, explicit F32
    # recurrence rounding, and 2^-16 per approximate exponential/reciprocal.
    # The general dense reference omits BF16 P: its allowance includes that
    # conversion and both sides of final BF16 rounding. Gating additionally
    # rounds context and sigmoid. These budgets precede all device results.
    maximum_value = max(map(abs, value))
    maximum_query_sum = max(math.fsum(map(abs, query[start:start + 128]))
                            for start in range(0, len(query), 128))
    same_score = case.pattern in ("zero", "uniform", "negative", "single")
    score_error = (0.0 if same_score else maximum_query_sum * max(map(abs, key)) *
                   2.0**-20 / math.sqrt(128))
    arithmetic_error = maximum_value * (
        math.expm1(2 * score_error) + 4 * 2.0**-16 +
        (8 * case.tiles + case.tokens) * 2.0**-24 + 2.0**-20)
    tolerance = arithmetic_error + maximum_value * (5 if case.gated else 3) * 2.0**-8
    exact = case.pattern in ("zero", "single", "negative")
    expected = []
    for row in range(case.tokens):
        for head in range(case.heads * 4):
            values = context(case, query, key, value, mask, row, head,
                             rounded=round_probability)
            start = (row * case.heads * 4 + head) * 128
            if round_probability:
                # This single-tile case isolates P conversion independently of
                # the online schedule. Final rounding must also be stable.
                assert all(margin(abs(element)) > arithmetic_error for element in values), (
                    case.name, values[0], arithmetic_error)
                exact = True
            expected.extend(epilogue(values, gate[start:start + 128], case.gated))
    mutations = {}
    for name, option in (("mask", "ignore_mask"), ("score_channel", "omit_channel"),
                         ("head", "wrong_head")):
        changed = context(case, query, key, value, mask, 0, 0,
                          rounded=round_probability, **{option: True})
        changed = epilogue(changed, gate[:128], case.gated)
        mutations[name] = any(abs(a - b) > (0 if exact else tolerance)
                              for a, b in zip(changed, expected[:128]))
    changed = context(case, query, key, value, mask, 0, 0)
    changed = epilogue(changed, gate[:128], case.gated)
    mutations["probability"] = round_probability and changed != expected[:128]
    # Logical V is token-first for the reference, physically channel-first.
    packed_value = [value[(row * case.heads + head) * 128 + channel]
                    for head in range(case.heads) for channel in range(128)
                    for row in range(case.tokens)]
    return (query, key, packed_value, mask, gate), expected, exact, tolerance, mutations


def render_case(case, observation):
    arrays, expected, exact, tolerance, _ = observation
    name = case.name
    names = ("query", "key", "value", "mask", "gate")
    dtypes = ("bf16", "bf16", "bf16", "i8", "bf16")
    data = []
    initialize = []
    allocations = []
    expectations = []
    for label, values, dtype in zip(names, arrays, dtypes):
        count = len(values)
        byte_width = 1 if dtype == "i8" else 2
        encoded = bytes(values).hex() if dtype == "i8" else encode(values, dtype)
        data.append(f'global.rodata.def @{name}_{label} = align(16) bytes("{encoded}")')
        initialize.extend((f"  %{label}_bytes = index.constant {count * byte_width} : offset",
                           f"  %{label}_values = global.load @{name}_{label} : buffer",
                           f"  buffer.copy %{label}_values[%origin], %{label}[%origin], %{label}_bytes"))
        allocations.extend((f"    %{label}_storage = check.generate.fill value({'123' if dtype == 'i8' else '123.0'}) : tensor<{count + 32}x{dtype}>",
                            f"    %{label} = check.tensor.view %{label}_storage offset({16 * byte_width}) : tensor<{count + 32}x{dtype}> -> tensor<{count}x{dtype}>"))
        expectations.append(f"      check.expect.bitwise actual(%{label}_storage) expected(%{label}_storage) : tensor<{count + 32}x{dtype}>")
    data.append(f'global.rodata.def @{name}_expected = align(16) bytes("{encode(expected, "bf16")}")')
    signature = ", ".join(f"%{label}: buffer" for label in (*names, "output"))
    arguments = ", ".join(f"%{label}" for label in (*names, "output"))
    types = ", ".join(f"tensor<{len(values)}x{dtype}>" for values, dtype in zip(arrays, dtypes))
    count = len(expected)
    provider = "shared_four_head_bf16" if case.tile == 16 else "shared_four_head_k32_bf16"
    output_check = (f"check.expect.bitwise actual(%output) expected(%output) : tensor<{count}xbf16>"
                    if exact else f"check.expect.close actual(%output) expected(%output) atol({tolerance:.17g}) rtol(0.0) nan(different) : tensor<{count}xbf16>")
    return "\n\n".join(data) + f'''

func.def @initialize_{name}({", ".join(f"%{label}: buffer" for label in names)}) {{
  %origin = index.constant 0 : offset
{chr(10).join(initialize)}
  func.return
}}

kernel.def target(@shared_attention_wave32) @apply_{name}() {{
  %groups = index.constant {case.tokens // 16} : index
  %heads = index.constant {case.heads} : index
  %threads = index.constant 128 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %heads, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch({signature}) {{
  %tokens = index.constant {case.tokens} : index
  %heads = index.constant {case.heads} : index
  %gated = scalar.constant {str(case.gated).lower()} : i1
  template.apply<@attention.{provider}>(%tokens, %heads, %gated, %query, %key, %value, %mask, {"%gate" if case.gated else "%output"}, %output) : (index, index, i1, buffer, buffer, buffer, buffer, buffer, buffer)
  kernel.return
}}

func.def @reference_{name}({signature}) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 2} : offset
  %values = global.load @{name}_expected : buffer
  buffer.copy %values[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
{chr(10).join(allocations)}
    %output_storage = check.generate.fill value(123.0) : tensor<{count + 32}xbf16>
    %output = check.tensor.view %output_storage offset(32) : tensor<{count + 32}xbf16> -> tensor<{count}xbf16>
    %prefix = check.tensor.view %output_storage offset(0) : tensor<{count + 32}xbf16> -> tensor<16xbf16>
    %suffix = check.tensor.view %output_storage offset({(count + 16) * 2}) : tensor<{count + 32}xbf16> -> tensor<16xbf16>
    check.generate<@initialize_{name}>({", ".join(f"%{label}" for label in names)}) : ({types}) -> ()
    check.compare<@apply_{name}, @reference_{name}>({arguments}) : ({types}, tensor<{count}xbf16>) -> () {{
      {output_check}
{chr(10).join(expectations)}
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xbf16>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xbf16>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    header = '''// Generated by generate_shared_attention_oracle.py.
// Whole dense attention and independent scalar gate algebra.

amdgpu.target<gfx11-generic> @tensor_shared_attention_rdna3

amdgpu.target<gfx11-generic> @shared_attention_wave32 {subgroup_size = 32}

template.decl @attention.shared_four_head_bf16(%tokens: index, %heads: index, %gated: i1, %query: buffer, %key: buffer, %value: buffer, %mask: buffer, %gate: buffer, %output: buffer)

template.decl @attention.shared_four_head_k32_bf16(%tokens: index, %heads: index, %gated: i1, %query: buffer, %key: buffer, %value: buffer, %mask: buffer, %gate: buffer, %output: buffer)
'''
    results = [observations(case) for case in CASES]
    for mutation in ("mask", "score_channel", "head", "probability"):
        assert any(result[-1][mutation] for result in results), mutation
    return header + "\n" + "\n".join(render_case(case, result)
                                    for case, result in zip(CASES, results))


def render_store():
    header = '''// Generated by generate_shared_attention_oracle.py.
// Context and sigmoid round independently before the final BF16 product.

amdgpu.target<gfx11-generic> @tensor_shared_attention_rdna3

amdgpu.target<gfx11-generic> @shared_attention_wave32 {subgroup_size = 32}

template.decl @attention.shared_four_head_store(%tokens: index, %query_width: index, %query_origin: index, %column: index, %values: vector<8xf32>, %gated: i1, %gate: buffer, %output: buffer)
'''
    # Exact BF16 gate outcomes have a margin exceeding the approximate-math
    # budget. Context values include both neighbors of conversion midpoints.
    choices = [number / 16 for number in range(-96, 97)
               if margin(1 / (1 + math.exp(-number / 16))) > 2.0**-14]
    generator = random.Random(0x6A7E)
    values = [f32(generator.uniform(-8, 8)) for _ in range(1536)]
    for index in range(384):
        sign = -1 if index % 2 else 1
        midpoint = sign * (1 + (2 * (index % 64) + 1) / 256)
        bits = struct.unpack("<I", struct.pack("<f", midpoint))[0] + index % 3 - 1
        values[index] = struct.unpack("<f", struct.pack("<I", bits))[0]
    values[:2] = [0.0, -0.0]
    gates = [choices[(index * 7) % len(choices)] for index in range(1536)]
    expected = epilogue(values, gates, True)
    for omitted in ("context", "sigmoid"):
        assert expected != epilogue(values, gates, True, omitted), omitted
    result = [header]
    for gated in (False, True):
        name = "gated" if gated else "ungated"
        expected = epilogue(values, gates, gated)
        result.append(f'''
global.rodata.def @{name}_input = align(16) bytes("{encode(values, 'f32')}")

global.rodata.def @{name}_gate = align(16) bytes("{encode(gates, 'bf16')}")

global.rodata.def @{name}_expected = align(16) bytes("{encode(expected, 'bf16')}")

func.def @initialize_{name}(%input: buffer, %gate: buffer) {{
  %origin = index.constant 0 : offset
  %input_bytes = index.constant 6144 : offset
  %gate_bytes = index.constant 3072 : offset
  %input_values = global.load @{name}_input : buffer
  %gate_values = global.load @{name}_gate : buffer
  buffer.copy %input_values[%origin], %input[%origin], %input_bytes
  buffer.copy %gate_values[%origin], %gate[%origin], %gate_bytes
  func.return
}}

kernel.def target(@shared_attention_wave32) @apply_{name}() {{
  %row_groups = index.constant 2 : index
  %column_groups = index.constant 3 : index
  %threads = index.constant 32 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%row_groups, %column_groups, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %gate: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %rows = index.constant 32 : index
  %columns = index.constant 48 : index
  %tile = index.constant 16 : index
  %row_group = kernel.workgroup.id<x> : index
  %column_group = kernel.workgroup.id<y> : index
  %row = index.mul %row_group, %tile : index
  %column = index.mul %column_group, %tile : index
  %input_view = buffer.view %input[%origin] : buffer -> view<32x48xf32>
  %values = vector.fragment.load<result> %input_view[%row, %column] shape [%tile, %tile] : view<32x48xf32> -> vector<8xf32>
  %gated = scalar.constant {str(gated).lower()} : i1
  template.apply<@attention.shared_four_head_store>(%rows, %columns, %row, %column, %values, %gated, {"%gate" if gated else "%output"}, %output) : (index, index, index, index, vector<8xf32>, i1, buffer, buffer)
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %gate: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant 3072 : offset
  %values = global.load @{name}_expected : buffer
  buffer.copy %values[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(123.0) : tensor<1568xf32>
    %input = check.tensor.view %input_storage offset(64) : tensor<1568xf32> -> tensor<1536xf32>
    %gate_storage = check.generate.fill value(123.0) : tensor<1568xbf16>
    %gate = check.tensor.view %gate_storage offset(32) : tensor<1568xbf16> -> tensor<1536xbf16>
    %output_storage = check.generate.fill value(123.0) : tensor<1568xbf16>
    %output = check.tensor.view %output_storage offset(32) : tensor<1568xbf16> -> tensor<1536xbf16>
    %prefix = check.tensor.view %output_storage offset(0) : tensor<1568xbf16> -> tensor<16xbf16>
    %suffix = check.tensor.view %output_storage offset(3104) : tensor<1568xbf16> -> tensor<16xbf16>
    check.generate<@initialize_{name}>(%input, %gate) : (tensor<1536xf32>, tensor<1536xbf16>) -> ()
    check.compare<@apply_{name}, @reference_{name}>(%input, %gate, %output) : (tensor<1536xf32>, tensor<1536xbf16>, tensor<1536xbf16>) -> () {{
      check.expect.bitwise actual(%output) expected(%output) : tensor<1536xbf16>
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<1568xf32>
      check.expect.bitwise actual(%gate_storage) expected(%gate_storage) : tensor<1568xbf16>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xbf16>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xbf16>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
''')
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in (("attention_shared_test.loom", render()),
                          ("attention_shared_store_test.loom", render_store())):
        path = Path(__file__).with_name(name)
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"{path} differs from its independent reference")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
