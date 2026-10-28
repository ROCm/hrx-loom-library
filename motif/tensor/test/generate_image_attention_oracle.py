#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author image attention from whole contractions with explicit conversions."""

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
    # Stable fixture identity.
    name: str
    # Complete sixteen-position query/key tiles.
    tokens: int
    # Input construction selecting a numerical or ownership boundary.
    pattern: str = "varied"


CASES = (
    Case("tile16", 16),
    Case("tile32", 32),
    Case("tile48", 48),
    Case("tile80", 80),
    Case("tile144", 144),
    Case("zero", 16, "zero"),
    Case("uniform", 32, "uniform"),
    Case("value_rounding", 32, "value"),
    Case("query_rounding", 16, "query"),
    Case("key_rounding", 16, "key"),
    Case("probability_rounding", 16, "probability"),
    Case("late_maxima", 64, "late"),
    Case("large_negative_scores", 16, "large_negative"),
)


def inputs(case):
    generator = random.Random(0x384BF16)
    count = case.tokens * 384
    query, key, value = ([generator.randrange(-127, 128) / 128
                          for _ in range(count)] for _ in range(3))
    if case.pattern in ("zero", "uniform", "value"):
        query = [0.0] * count
    if case.pattern == "large_negative":
        query, key, value = [0.0] * count, [0.0] * count, [1.0] * count
        for row in range(case.tokens):
            query[row * 384] = 2.0**56
            key[row * 384] = -2.0**56
    elif case.pattern == "zero":
        key, value = [0.0] * count, [0.0] * count
    elif case.pattern == "value":
        value = [0.0] * count
        for channel in range(384):
            scale = (-1 if channel % 2 else 1) * (1 + channel % 4) / 4
            value[channel] = (1 + 1 / 256) * scale
            value[384 + channel] = (1 + 1 / 128) * scale
    elif case.pattern in ("query", "key", "probability", "late"):
        query, key = [0.0] * count, [0.0] * count
        for row in range(case.tokens):
            query[row * 384 + 383] = 16 + 1 / 16 if case.pattern == "query" else 16.0
            if case.pattern == "key":
                selected_key = (1 + (2 * (row % 8) + 1) / 256) * (-1 if row < 8 else 1)
            elif case.pattern == "late":
                selected_key = (row // 16) * 8 + (row % 16) / 8
            else:
                selected_key = (row - 8) / (4 if case.pattern == "query" else 8)
            key[row * 384 + 383] = selected_key
            for channel in range(384):
                scale = (-1 if channel % 2 else 1) * (1 + channel % 4) / 4
                value[row * 384 + channel] = (-1 if row < case.tokens // 2 else 1) * scale
        if case.pattern == "probability":
            scores = [16 * key[row * 384 + 383] / math.sqrt(384)
                      for row in range(case.tokens)]
            probabilities = [math.exp(score - max(scores)) for score in scores]
            for row, probability in enumerate(probabilities):
                sign = 1 if rounded_float(probability, 8) > probability else -1
                for channel in range(384):
                    value[row * 384 + channel] = sign * (1 + channel % 4) / 4
    return query, key, value


def row_result(case, query, key, value, row, *, round_probability=False,
               drop_score_channel=False, drop_value_channel=False):
    channels = range(383 if drop_score_channel else 384)
    scores = [math.fsum(query[row * 384 + channel] * key[position * 384 + channel]
                        for channel in channels) / math.sqrt(384)
              for position in range(case.tokens)]
    maximum = max(scores)
    probabilities = [math.exp(score - maximum) for score in scores]
    denominator = math.fsum(probabilities)
    weights = ([rounded_float(probability, 8) for probability in probabilities]
               if round_probability else probabilities)
    result = [math.fsum(weight * value[position * 384 + channel]
                        for position, weight in enumerate(weights)) / denominator
              for channel in range(384)]
    if drop_value_channel:
        result[383] = 0.0
    return result, probabilities


def probability_margin(value):
    rounded = rounded_float(value, 8)
    word = struct.unpack("<I", struct.pack("<f", rounded))[0] >> 16
    previous = struct.unpack("<f", struct.pack("<I", (word - 1) << 16))[0]
    following = struct.unpack("<f", struct.pack("<I", (word + 1) << 16))[0]
    return min(value - (previous + rounded) / 2,
               (rounded + following) / 2 - value)


def observations(case):
    stored = inputs(case)
    converted = [[rounded_float(value, 8) for value in array] for array in stored]
    query, key, value = converted
    exact = case.pattern == "zero"
    round_probability = case.pattern in ("query", "key", "probability")
    maximum_query_sum = max(math.fsum(abs(value) for value in query[row * 384:(row + 1) * 384])
                            for row in range(case.tokens))
    # A twenty-bit input-scaled native contraction requirement, independent
    # of device observations. Approximate exp/reciprocal stages each receive
    # a 2^-16 allowance. General multi-tile cases also allow BF16 probability
    # rounding; dedicated single-tile cases model that boundary explicitly.
    uniform_scores = case.pattern in ("zero", "uniform", "value", "large_negative")
    score_error = (0.0 if uniform_scores else
                   maximum_query_sum * max(map(abs, key)) * 2.0**-20 / math.sqrt(384))
    tolerance = max(map(abs, value)) * (
        math.expm1(2 * score_error) + 3 * 2.0**-16 + 2.0**-20 +
        (8 * (case.tokens // 16) + case.tokens) * 2.0**-24 +
        (0.0 if round_probability else 2.0**-8))
    if exact:
        tolerance = 0.0
    elif uniform_scores:
        # Identical logits make the mean independent of their common value.
        # Native matrix arithmetic and approximate normalization retain an
        # accuracy contract, including for exactly representable dyadic data.
        tolerance = max(map(abs, value)) * 2.0**-20
    rows = []
    for row in range(case.tokens):
        result, probabilities = row_result(case, *converted, row,
                                           round_probability=round_probability)
        if round_probability:
            # The authored probability values stay away from BF16 midpoints
            # even with the declared score/transcendental error. Their rounded
            # values therefore have an unambiguous independent reference.
            for probability in probabilities:
                assert probability_margin(probability) > probability * (
                    math.expm1(2 * score_error) + 2.0**-16), case.name
        rows.append([f32(element) for element in result])
    mutations = {}
    for index, name in enumerate(("query", "key", "value")):
        alternative = list(converted)
        alternative[index] = stored[index]
        result, _ = row_result(case, *alternative, 0, round_probability=round_probability)
        mutations[name] = any(abs(f32(a) - b) > tolerance for a, b in zip(result, rows[0]))
    result, _ = row_result(case, *converted, 0)
    mutations["probability"] = round_probability and any(
        abs(f32(a) - b) > tolerance for a, b in zip(result, rows[0]))
    for name, option in (("score_channel", "drop_score_channel"),
                         ("value_channel", "drop_value_channel")):
        result, _ = row_result(case, *converted, 0, round_probability=round_probability,
                               **{option: True})
        mutations[name] = any(abs(f32(a) - b) > tolerance for a, b in zip(result, rows[0]))
    mutations["output"] = any(abs(rounded_float(value, 8) - value) > tolerance
                              for value in rows[0])
    # Q/K are channel-first planes, followed by one token-first V plane.
    packed = [array[row * 384 + channel] for array in stored[:2]
              for channel in range(384) for row in range(case.tokens)] + stored[2]
    expected = [rows[row][channel] for channel in range(384) for row in range(case.tokens)]
    return packed, expected, exact, tolerance, mutations


def render_case(case, observation):
    packed, expected, exact, tolerance, _ = observation
    name = case.name
    count = case.tokens * 384
    bits_view = (f"    %output_bits = check.tensor.view %output offset(0) : tensor<{count}xf32> -> tensor<{count}xi32>\n"
                 if exact else "")
    comparison = (f"check.expect.bitwise actual(%output_bits) expected(%output_bits) : tensor<{count}xi32>"
                  if exact else
                  f"check.expect.approximate actual(%output) expected(%output) atol({tolerance:.17g}) rtol(0.0) nan(different) : tensor<{count}xf32>, tensor<{count}xf32>")
    return f'''
global.rodata.def @{name}_input = align(16) bytes("{encode(packed, 'f32')}")

global.rodata.def @{name}_expected = align(16) bytes("{encode(expected, 'f32')}")

func.def @initialize_{name}(%input: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 12} : offset
  %values = global.load @{name}_input : buffer
  buffer.copy %values[%origin], %input[%origin], %bytes
  func.return
}}

kernel.def target(@image_attention_wave32) @apply_{name}() {{
  %groups = index.constant {case.tokens // 16} : index
  %threads = index.constant 128 : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %unit, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %output: buffer) {{
  %tokens = index.constant {case.tokens} : index
  template.apply<@attention.single_head_384_matrix_nchw_f32>(%tokens, %input, %output) : (index, buffer, buffer)
  kernel.return
}}

func.def @reference_{name}(%input: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 4} : offset
  %expected = global.load @{name}_expected : buffer
  buffer.copy %expected[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(123.25) : tensor<{count * 3 + 32}xf32>
    %input = check.tensor.view %input_storage offset(64) : tensor<{count * 3 + 32}xf32> -> tensor<{count * 3}xf32>
    %output_storage = check.generate.fill value(123.25) : tensor<{count + 32}xf32>
    %output = check.tensor.view %output_storage offset(64) : tensor<{count + 32}xf32> -> tensor<{count}xf32>
{bits_view}    %prefix = check.tensor.view %output_storage offset(0) : tensor<{count + 32}xf32> -> tensor<16xf32>
    %suffix = check.tensor.view %output_storage offset({(count + 16) * 4}) : tensor<{count + 32}xf32> -> tensor<16xf32>
    check.generate<@initialize_{name}>(%input) : (tensor<{count * 3}xf32>) -> ()
    check.compare<@apply_{name}, @reference_{name}>(%input, %output) : (tensor<{count * 3}xf32>, tensor<{count}xf32>) -> () {{
      {comparison}
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{count * 3 + 32}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xf32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    header = '''// Generated by generate_image_attention_oracle.py.
// Independent dense attention with explicit operand/probability boundaries.

amdgpu.target<gfx11-generic> @tensor_image_attention_rdna3

amdgpu.target<gfx11-generic> @image_attention_wave32 {subgroup_size = 32}

template.decl @attention.single_head_384_matrix_nchw_f32(%tokens: index, %qkv: buffer, %output: buffer)
'''
    results = [observations(case) for case in CASES]
    for mutation in ("query", "key", "value", "probability", "score_channel", "value_channel", "output"):
        assert any(result[-1][mutation] for result in results), mutation
    return header + "".join(render_case(case, result)
                            for case, result in zip(CASES, results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    path = Path(__file__).with_name("image_attention_matrix_test.loom")
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its independent reference")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
