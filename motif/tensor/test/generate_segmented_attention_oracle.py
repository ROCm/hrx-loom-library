#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author packed attention from independent dense contractions and softmax."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import random

from binary_reference import rounded_float
from generate_linear_epilogue_oracle import encode


@dataclass(frozen=True)
class Case:
    # Stable fixture identity.
    name: str
    # Live rows in each independent segment.
    length: int
    # Number of complete contiguous segments.
    segments: int
    # Independent 128-channel heads per row.
    heads: int
    # Independent query rows assigned to one workgroup.
    waves: int = 4
    # Input distribution selecting an arithmetic or ownership boundary.
    pattern: str = "varied"

    @property
    def rows(self):
        return self.length * self.segments

    @property
    def width(self):
        return self.heads * 128


CASES = (
    Case("single_key", 1, 3, 2),
    Case("pairs", 2, 3, 2, waves=1),
    Case("odd_segments", 3, 3, 1),
    Case("twelve_layers", 12, 2, 3),
    Case("twenty_heads", 12, 2, 20),
    Case("length31", 31, 2, 2),
    Case("length32", 32, 2, 1),
    Case("length33", 33, 2, 2),
    Case("constant_values", 12, 3, 2, pattern="constant"),
    Case("zero_scores", 12, 2, 2, pattern="zero_query"),
    Case("common_key_offset", 12, 2, 2, pattern="common"),
    Case("last_channel", 3, 3, 2, pattern="last_channel"),
)


def inputs(case):
    generator = random.Random(0x128BF16)
    count = case.rows * case.width
    query, key, value = ([generator.randrange(-63, 64) / 64
                          for _ in range(count)] for _ in range(3))
    if case.pattern == "common":
        query = [element / 64 for element in query]
        key = [4096.0 + generator.randrange(-3, 4) * 32 for _ in range(count)]
    for row in range(case.rows):
        segment, local = divmod(row, case.length)
        for head in range(case.heads):
            start = row * case.width + head * 128
            for channel in range(128):
                position = start + channel
                if case.pattern in ("constant", "zero_query"):
                    query[position] = 0.0
                if case.pattern == "constant":
                    value[position] = ((segment * 3 + head) % 7 - 3) / 4
                elif case.pattern == "last_channel":
                    query[position] = 8.0 if channel == 127 else 0.0
                    key[position] = float(local - 1) if channel == 127 else 0.0
                    value[position] = float(local - 1) * (1 if channel % 2 else -1)
    assert all(rounded_float(element, 8) == element
               for array in (query, key, value) for element in array)
    return query, key, value


def dense_row(case, query, key, value, row, head, *, cross_segment=False,
              drop_channel=False):
    start = row * case.width + head * 128
    segment_begin = (row // case.length) * case.length
    positions = range(case.rows) if cross_segment else range(
        segment_begin, segment_begin + case.length)
    channels = range(127 if drop_channel else 128)
    # BF16 products are exact in binary64. One whole contraction followed by
    # one whole softmax is independent of wave reductions and online updates.
    scores = [math.fsum(query[start + channel] *
                        key[position * case.width + head * 128 + channel]
                        for channel in channels) / math.sqrt(128)
              for position in positions]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(weights)
    return [math.fsum(weight * value[position * case.width + head * 128 + channel]
                      for position, weight in zip(positions, weights)) / denominator
            for channel in range(128)]


def accuracy_budget(case, query, key, value):
    def gamma(operations):
        product = operations * 2.0**-24
        return product / (1 - product)

    maximum_difference = max(
        abs(key[row * case.width + channel] -
            key[(row // case.length) * case.length * case.width + channel])
        for row in range(case.rows) for channel in range(case.width))
    score_error = gamma(132) * 128 * max(map(abs, query)) * maximum_difference / math.sqrt(128)
    # These are independent workload accuracy requirements, not a fit to GPU
    # observations. The absolute budget permits F32 score/online error and
    # 2^-16 relative error at each approximate exponential/reciprocal stage.
    # The relative term covers rounding both observations to BF16.
    absolute = max(map(abs, value)) * (
        math.expm1(2 * score_error) + gamma(8 * case.length) +
        (2 * case.length + 2) * 2.0**-16)
    relative = 2.0**-7 / (1 - 2.0**-8)
    return absolute, relative


def observations(case):
    query, key, value = inputs(case)
    expected = [rounded_float(element, 8)
                for row in range(case.rows) for head in range(case.heads)
                for element in dense_row(case, query, key, value, row, head)]
    exact = case.length == 1 or case.pattern == "constant"
    if exact:
        assert expected == value, case.name
    absolute, relative = accuracy_budget(case, query, key, value)
    original = expected[:128]
    mutations = {}
    for name, option in (("segment", "cross_segment"), ("channel", "drop_channel")):
        alternative = dense_row(case, query, key, value, 0, 0, **{option: True})
        mutations[name] = any(abs(rounded_float(a, 8) - b) > absolute + relative * abs(b)
                              for a, b in zip(alternative, original))
    return (query, key, value, expected), exact, absolute, relative, mutations


def render_case(case, observation):
    arrays, exact, absolute, relative, _ = observation
    name = case.name
    count = case.rows * case.width
    named = list(zip(("query", "key", "value", "output"), arrays))
    rodata = "\n\n".join(
        f'global.rodata.def @{name}_{key} = align(8) bytes("{encode(values, "bf16")}")'
        for key, values in named)
    inputs_copy = "\n".join(
        f"  %{key}_values = global.load @{name}_{key} : buffer\n"
        f"  buffer.copy %{key}_values[%origin], %{key}[%origin], %bytes"
        for key, _ in named[:3])
    storage = "\n".join(
        f"    %{key}_storage = check.generate.fill value(17142) : tensor<{count + 32}xi16>\n"
        f"    %{key} = check.tensor.view %{key}_storage offset(32) : tensor<{count + 32}xi16> -> tensor<{count}xbf16>"
        for key, _ in named)
    immutable = "\n".join(
        f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{count + 32}xi16>"
        for key, _ in named[:3])
    comparison = (f"check.expect.bitwise actual(%output) expected(%output) : tensor<{count}xbf16>"
                  if exact else
                  f"check.expect.close actual(%output) expected(%output) atol({absolute:.17g}) rtol({relative:.17g}) nan(different) : tensor<{count}xbf16>")
    return f'''
{rodata}

func.def @initialize_{name}(%query: buffer, %key: buffer, %value: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 2} : offset
{inputs_copy}
  func.return
}}

kernel.def target(@segmented_wave32) @apply_{name}() {{
  %groups = index.constant {(case.rows + case.waves - 1) // case.waves} : index
  %heads = index.constant {case.heads} : index
  %threads = index.constant {case.waves * 32} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %heads, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%query: buffer, %key: buffer, %value: buffer, %output: buffer) {{
  %rows = index.constant {case.rows} : index
  %length = index.constant {case.length} : index
  %heads = index.constant {case.heads} : index
  %width = index.constant {case.width} : index
  %waves = index.constant {case.waves} : index
  %channels_per_lane = index.constant 4 : index
  %head_width = index.constant 128 : index
  %group = kernel.workgroup.id<x> : index
  %head = kernel.workgroup.id<y> : index
  %wave = kernel.subgroup.id : index
  %lane = kernel.subgroup.lane.id : index
  %row = index.madd %group, %waves, %wave : index
  %valid = index.cmp ult, %row, %rows : index
  scf.if %valid {{
    %result = template.apply<@attention.segmented_128_bf16>(%rows, %length, %heads, %row, %head, %query, %key, %value) : (index, index, index, index, index, buffer, buffer, buffer) -> (vector<4xbf16>)
    %origin = index.constant 0 : offset
    %lane_channel = index.mul %lane, %channels_per_lane : index
    %column = index.madd %head, %head_width, %lane_channel : index
    %view = buffer.view %output[%origin] : buffer -> view<[%rows]x[%width]xbf16>
    vector.store %result, %view[%row, %column] : vector<4xbf16>, view<[%rows]x[%width]xbf16>
  }}
  kernel.return
}}

func.def @reference_{name}(%query: buffer, %key: buffer, %value: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 2} : offset
  %expected = global.load @{name}_output : buffer
  buffer.copy %expected[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
{storage}
    %prefix = check.tensor.view %output_storage offset(0) : tensor<{count + 32}xi16> -> tensor<16xi16>
    %suffix = check.tensor.view %output_storage offset({(count + 16) * 2}) : tensor<{count + 32}xi16> -> tensor<16xi16>
    check.generate<@initialize_{name}>(%query, %key, %value) : (tensor<{count}xbf16>, tensor<{count}xbf16>, tensor<{count}xbf16>) -> ()
    check.compare<@apply_{name}, @reference_{name}>(%query, %key, %value, %output) : (tensor<{count}xbf16>, tensor<{count}xbf16>, tensor<{count}xbf16>, tensor<{count}xbf16>) -> () {{
      {comparison}
{immutable}
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xi16>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xi16>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    header = '''// Generated by generate_segmented_attention_oracle.py.
// Independent dense QK, whole softmax, and weighted-value references.

amdgpu.target<gfx11-generic> @segmented_wave32 {subgroup_size = 32}

template.decl @attention.segmented_128_bf16(%rows: index, %length: index, %heads: index, %row: index, %head: index, %query: buffer, %key: buffer, %value: buffer) -> (vector<4xbf16>)
'''
    results = [observations(case) for case in CASES]
    for mutation in ("segment", "channel"):
        assert any(result[-1][mutation] for result in results), mutation
    return header + "".join(render_case(case, result)
                            for case, result in zip(CASES, results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    path = Path(__file__).with_name("segmented_attention_test.loom")
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its independent reference")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
