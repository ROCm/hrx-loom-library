#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author expanded attention from whole dense contractions and scalar gates."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random

from binary_reference import f32, rounded_float
from generate_linear_epilogue_oracle import encode
from generate_shared_attention_oracle import Case as SharedCase
from generate_shared_attention_oracle import epilogue, inputs as shared_inputs, margin


@dataclass(frozen=True)
class Case:
    # Stable fixture identity.
    name: str
    # Logical rows, including complete sixteen-row padding tiles.
    tokens: int
    # Number of independent query heads.
    query_heads: int = 4
    # Number of KV heads shared by an integral number of query heads.
    key_value_heads: int = 1
    # Whether sequence position limits each query's visible keys.
    causal: bool = False
    # Whether the scalar BF16 sigmoid gate follows attention.
    gated: bool = False
    # Physical value layout, separate from logical token/channel order.
    layout: str = "dense"
    # Numerical or masking construction.
    pattern: str = "varied"


CASES = (
    Case("tile16", 16, 1),
    Case("causal32", 32, 4, 2, True),
    Case("tail48", 48, 2, gated=True),
    Case("tile64", 64, 2, 2),
    Case("leading80", 80, 1, pattern="leading"),
    Case("middle192", 192, 1, pattern="middle"),
    Case("trailing128", 128, 1, causal=True, pattern="trailing"),
    Case("encoder176", 176, 4, causal=True),
    Case("encoder_heads", 16, 32, 8, True),
    Case("refiner_heads", 16, 20, 20, gated=True),
    Case("transposed80", 80, 2, layout="transposed", pattern="sparse"),
    Case("padded32", 32, 4, 2, gated=True, layout="padded"),
    Case("zero", 16, pattern="zero"),
    Case("single80", 80, 1, pattern="single"),
    Case("common128", 128, 2, causal=True, pattern="common"),
    Case("key_residual", 32, pattern="residual"),
    Case("negative160", 160, 1, pattern="negative"),
    Case("probability16", 16, pattern="probability"),
    Case("probability32", 32, pattern="probability"),
)


def inputs(case):
    if case.pattern == "probability":
        return shared_inputs(SharedCase("probability", case.tokens))
    generator = random.Random(0xE29A + case.tokens)
    query_count = case.tokens * case.query_heads * 128
    key_count = case.tokens * case.key_value_heads * 128
    query = [generator.randrange(-127, 128) / 128 for _ in range(query_count)]
    key, value = ([generator.randrange(-127, 128) / 128 for _ in range(key_count)]
                  for _ in range(2))
    gate = [generator.randrange(-64, 65) / 16 for _ in range(query_count)]
    mask = [1] * case.tokens
    if case.pattern == "leading":
        mask[:64] = [0] * 64
    elif case.pattern == "middle":
        mask[64:128] = [0] * 64
    elif case.pattern == "trailing":
        mask[64:] = [0] * (case.tokens - 64)
    elif case.pattern == "sparse":
        mask = [(128, 0, 255, 0, 1, 2, 0)[row % 7] for row in range(case.tokens)]
    elif case.pattern == "single":
        mask = [0] * (case.tokens - 1) + [255]
    if case.pattern == "zero":
        query, value = [0.0] * query_count, [0.0] * key_count
    elif case.pattern in ("common", "residual", "negative"):
        query, key = [0.0] * query_count, [0.0] * key_count
        for row in range(case.tokens):
            for head in range(case.query_heads):
                start = (row * case.query_heads + head) * 128
                query[start + 127] = 2.0**56 if case.pattern == "negative" else 4.0
                if case.pattern == "common":
                    query[start] = 1024.0
                    query[start + 127] = 1.0
            for head in range(case.key_value_heads):
                start = (row * case.key_value_heads + head) * 128
                if case.pattern == "common":
                    key[start] = 1024.0
                    key[start + 127] = (row % 17 - 8) / 32
                    raw = 2.0**20 + key[start + 127]
                    residual = raw - f32(raw)
                    # Align values with information discarded by an
                    # uncentered F32 logit, without simulating GPU arithmetic.
                    sign = 1.0 if residual > 0 else -1.0 if residual < 0 else 0.0
                    for channel in range(128):
                        value[start + channel] = sign
                elif case.pattern == "residual":
                    key[start + 127] = 256.0 if row == case.tokens - 1 else (row % 16) / 32
                else:
                    key[start + 127] = 0.0 if row == case.tokens - 1 else -2.0**56
        if case.pattern == "residual":
            mask[-1] = 0
        elif case.pattern == "negative":
            mask = [0] * case.tokens
            mask[64:96] = [128] * 32
            value = [1.0] * key_count
    return query, key, value, mask, gate


def context(case, query, key, value, mask, row, head, *, mutation=""):
    key_head = head // (case.query_heads // case.key_value_heads)
    if mutation == "head":
        key_head = (key_head + 1) % case.key_value_heads
    visible = [position for position in range(case.tokens)
               if (mask[position] or mutation == "mask") and
               (not case.causal or position <= row or mutation == "causal")]
    start = (row * case.query_heads + head) * 128
    scores = []
    for position in visible:
        key_start = (position * case.key_value_heads + key_head) * 128
        if mutation == "residual":
            anchor = 0 if case.causal else case.tokens - 1
            anchor_start = (anchor * case.key_value_heads + key_head) * 128
            operands = [rounded_float(f32(key[key_start + channel] -
                                         key[anchor_start + channel]), 8)
                        for channel in range(128)]
        else:
            operands = key[key_start:key_start + 128]
        raw = math.fsum(query[start + channel] * operand
                       for channel, operand in enumerate(operands))
        if mutation == "centering":
            raw = f32(raw)
        scores.append(raw / math.sqrt(128))
    maximum = max(scores)
    probabilities = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(probabilities)
    weights = ([rounded_float(probability, 8) for probability in probabilities]
               if mutation == "probability" else probabilities)
    return [math.fsum(weight * value[(position * case.key_value_heads + key_head) * 128 + channel]
                      for position, weight in zip(visible, weights)) / denominator
            for channel in range(128)]


def value_storage(case, values):
    width = case.key_value_heads * 128
    row_stride, column_stride = (1, case.tokens + 16) if case.layout == "transposed" else (
        width + (16 if case.layout == "padded" else 0), 1)
    count = width * column_stride if row_stride == 1 else case.tokens * row_stride
    packed = [123.0] * count
    for row in range(case.tokens):
        for channel in range(width):
            packed[row * row_stride + channel * column_stride] = values[row * width + channel]
    return packed, row_stride, column_stride


def observations(case):
    query, key, value, mask, gate = inputs(case)
    anchor = 0 if case.causal else case.tokens - 1
    width = case.key_value_heads * 128
    centered = [f32(element - key[anchor * width + index % width])
                for index, element in enumerate(key)]
    # Fixture differences are exactly representable by the two BF16 terms.
    # The oracle still contracts the original operands without centering.
    for element in centered:
        high = rounded_float(element, 8)
        low = rounded_float(f32(element - high), 8)
        assert f32(high + low) == element
    maximum_query_sum = max(math.fsum(map(abs, query[start:start + 128]))
                            for start in range(0, len(query), 128))
    same_score = case.pattern in ("zero", "single", "negative")
    score_error = (0.0 if same_score else maximum_query_sum * max(map(abs, centered)) *
                   (2.0**-20 + 2.0**-23) / math.sqrt(128))
    maximum_value = max(map(abs, value))
    arithmetic_error = maximum_value * (
        math.expm1(2 * score_error) + 4 * 2.0**-16 + 2.0**-20 +
        (case.tokens + 8 * ((case.tokens + 63) // 64)) * 2.0**-24)
    relative = (4 if case.gated else 2) * 2.0**-8 / (1 - 2.0**-8)
    exact = case.pattern in ("zero", "single", "negative", "probability")
    expected = []
    witness_values = {}
    for row in range(case.tokens):
        for head in range(case.query_heads):
            values = context(case, query, key, value, mask, row, head)
            start = (row * case.query_heads + head) * 128
            if case.pattern == "probability":
                assert all(margin(abs(element)) > arithmetic_error for element in values), (
                    case.name, values[0], arithmetic_error)
            output = epilogue(values, gate[start:start + 128], case.gated)
            expected.extend(output)
            # Mutation observations span sequence positions, not just row zero
            # where causal attention has only one visible key.
            if head == 0 and row in (0, case.tokens // 2, case.tokens - 1):
                for mutation in ("head", "mask", "causal", "centering", "residual", "probability"):
                    changed = context(case, query, key, value, mask, row, head, mutation=mutation)
                    changed = epilogue(changed, gate[start:start + 128], case.gated)
                    sensitive = any(abs(a - b) > (0 if exact else arithmetic_error + relative * abs(b))
                                    for a, b in zip(changed, output))
                    witness_values[mutation] = witness_values.get(mutation, False) or sensitive
    packed_value, row_stride, column_stride = value_storage(case, value)
    arrays = [("query", "bf16", query), ("key", "bf16", key),
              ("value", "bf16", packed_value), ("mask", "i8", mask),
              ("gate", "bf16", gate)]
    return arrays, expected, exact, arithmetic_error, relative, row_stride, column_stride, witness_values


def render_fixture(name, arrays, expected, exact, absolute, relative, launch, body):
    data, initialize, allocations, expectations = [], [], [], []
    for label, dtype, values in arrays:
        count = len(values)
        width = 1 if dtype == "i8" else 4 if dtype == "f32" else 2
        encoded = bytes(values).hex() if dtype == "i8" else encode(values, dtype)
        data.append(f'global.rodata.def @{name}_{label} = align(16) bytes("{encoded}")')
        initialize.extend((f"  %{label}_bytes = index.constant {count * width} : offset",
                           f"  %{label}_values = global.load @{name}_{label} : buffer",
                           f"  buffer.copy %{label}_values[%origin], %{label}[%origin], %{label}_bytes"))
        allocations.extend((f"    %{label}_storage = check.generate.fill value({'123' if dtype == 'i8' else '123.0'}) : tensor<{count + 32}x{dtype}>",
                            f"    %{label} = check.tensor.view %{label}_storage offset({16 * width}) : tensor<{count + 32}x{dtype}> -> tensor<{count}x{dtype}>"))
        expectations.append(f"      check.expect.bitwise actual(%{label}_storage) expected(%{label}_storage) : tensor<{count + 32}x{dtype}>")
    data.append(f'global.rodata.def @{name}_expected = align(16) bytes("{encode(expected, "bf16")}")')
    names = [label for label, _, _ in arrays]
    signature = ", ".join(f"%{label}: buffer" for label in (*names, "output"))
    types = ", ".join(f"tensor<{len(values)}x{dtype}>" for _, dtype, values in arrays)
    count = len(expected)
    comparison = (f"check.expect.bitwise actual(%output) expected(%output) : tensor<{count}xbf16>"
                  if exact else f"check.expect.close actual(%output) expected(%output) atol({absolute:.17g}) rtol({relative:.17g}) nan(different) : tensor<{count}xbf16>")
    return "\n\n".join(data) + f'''

func.def @initialize_{name}({", ".join(f"%{label}: buffer" for label in names)}) {{
  %origin = index.constant 0 : offset
{chr(10).join(initialize)}
  func.return
}}

kernel.def target(@expanded_attention_wave64) @apply_{name}() {{
{launch}
}} launch({signature}) {{
{body}
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
    check.compare<@apply_{name}, @reference_{name}>({", ".join(f"%{label}" for label in (*names, "output"))}) : ({types}, tensor<{count}xbf16>) -> () {{
      {comparison}
{chr(10).join(expectations)}
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xbf16>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xbf16>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


HEADER = '''// Generated by generate_expanded_attention_oracle.py.
// Independent whole-query softmax and scalar BF16 gate expressions.

amdgpu.target<gfx11-generic> @tensor_expanded_attention_rdna3

amdgpu.target<gfx11-generic> @expanded_attention_wave64 {subgroup_size = 64}

template.decl @attention.expanded_masked_gqa_128_bf16(%tokens: index, %query_heads: index, %key_value_heads: index, %causal: i1, %value_layout: encoding<layout>, %query: buffer, %key: buffer, %value: buffer, %mask: buffer) -> (vector<8xbf16>)

template.decl @attention.store_128_bf16(%tokens: index, %heads: index, %values: vector<8xbf16>, %output: buffer)

template.decl @attention.gate_128_bf16(%tokens: index, %heads: index, %values: vector<8xbf16>, %gate: buffer) -> (vector<8xbf16>)
'''


def launch(rows, heads):
    return f'''  %groups = index.constant {rows // 16} : index
  %heads = index.constant {heads} : index
  %threads = index.constant 256 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %heads, %unit) workgroup_size(%threads, %unit, %unit) : index'''


def render():
    results = [observations(case) for case in CASES]
    for mutation in ("head", "mask", "causal", "centering", "residual", "probability"):
        assert any(result[-1][mutation] for result in results), mutation
    sections = [HEADER]
    for case, result in zip(CASES, results):
        arrays, expected, exact, absolute, relative, row_stride, column_stride, _ = result
        body = f'''  %tokens = index.constant {case.tokens} : index
  %query_heads = index.constant {case.query_heads} : index
  %key_value_heads = index.constant {case.key_value_heads} : index
  %causal = scalar.constant {str(case.causal).lower()} : i1
  %value_layout = encoding.layout.strided [{row_stride}, {column_stride}] : encoding<layout>
  %values = template.apply<@attention.expanded_masked_gqa_128_bf16>(%tokens, %query_heads, %key_value_heads, %causal, %value_layout, %query, %key, %value, %mask) : (index, index, index, i1, encoding<layout>, buffer, buffer, buffer, buffer) -> (vector<8xbf16>)'''
        if case.gated:
            body += '\n  %gated = template.apply<@attention.gate_128_bf16>(%tokens, %query_heads, %values, %gate) : (index, index, vector<8xbf16>, buffer) -> (vector<8xbf16>)'
        body += f'\n  template.apply<@attention.store_128_bf16>(%tokens, %query_heads, {"%gated" if case.gated else "%values"}, %output) : (index, index, vector<8xbf16>, buffer)'
        sections.append(render_fixture(case.name, arrays, expected, exact, absolute, relative,
                                       launch(case.tokens, case.query_heads), body))
    return "\n".join(sections)


def render_epilogue():
    generator = random.Random(0x128E91)
    count = 32 * 3 * 128
    values = [rounded_float(generator.uniform(-8, 8), 8) for _ in range(count)]
    values[:2] = [0.0, -0.0]
    choices = [number / 16 for number in range(-96, 97)
               if margin(1 / (1 + math.exp(-number / 16))) > 2.0**-14]
    gates = [choices[(index * 7) % len(choices)] for index in range(count)]
    expected = epilogue(values, gates, True)
    assert expected != epilogue(values, gates, True, "sigmoid")
    sections = [HEADER]
    for gated in (False, True):
        body = '''  %origin = index.constant 0 : offset
  %tokens = index.constant 32 : index
  %heads = index.constant 3 : index
  %tile = index.constant 16 : index
  %rows = index.constant 4 : index
  %pair = index.constant 2 : index
  %head_width = index.constant 128 : index
  %group = kernel.workgroup.id<x> : index
  %head = kernel.workgroup.id<y> : index
  %wave = kernel.subgroup.id : index
  %lane = kernel.subgroup.lane.id : index
  %tile_origin = index.mul %group, %tile : index
  %row = index.madd %wave, %rows, %tile_origin : index
  %head_origin = index.mul %head, %head_width : index
  %column = index.madd %lane, %pair, %head_origin : index
  %input_view = buffer.view %input[%origin] : buffer -> view<32x384xbf16>'''
        for index in range(4):
            body += f'''\n  %offset{index} = index.constant {index} : index
  %row{index} = index.add %row, %offset{index} : index
  %pair{index} = vector.load %input_view[%row{index}, %column] : view<32x384xbf16> -> vector<2xbf16>'''
        body += '\n  %values = vector.concat<0> %pair0, %pair1, %pair2, %pair3 : vector<2xbf16>, vector<2xbf16>, vector<2xbf16>, vector<2xbf16> -> vector<8xbf16>'
        if gated:
            body += '\n  %gated = template.apply<@attention.gate_128_bf16>(%tokens, %heads, %values, %gate) : (index, index, vector<8xbf16>, buffer) -> (vector<8xbf16>)'
        body += f'\n  template.apply<@attention.store_128_bf16>(%tokens, %heads, {"%gated" if gated else "%values"}, %output) : (index, index, vector<8xbf16>, buffer)'
        sections.append(render_fixture("gated" if gated else "ungated",
                                       [("input", "bf16", values), ("gate", "bf16", gates)],
                                       epilogue(values, gates, gated), True, 0, 0,
                                       launch(32, 3), body))
    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in (("attention_expanded_test.loom", render()),
                          ("attention_expanded_epilogue_test.loom", render_epilogue())):
        path = Path(__file__).with_name(name)
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"{path} differs from its independent reference")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
