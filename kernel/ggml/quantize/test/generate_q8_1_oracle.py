#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author exact Q8_1 x4 bytes from scalar IEEE arithmetic and integer sums."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import struct


@dataclass(frozen=True)
class Case:
    # Stable correctness and benchmark scenario name.
    name: str
    # Number of contiguous activation rows.
    rows: int
    # Number of F32 elements per row, divisible by 128.
    width: int
    # Complete logical input, excluding the storage guards.
    values: tuple[float, ...]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half_bytes(value):
    try:
        return struct.pack("<e", value)
    except OverflowError:
        # IEEE nearest-even conversion overflows to the signed infinity.
        return struct.pack("<H", 0xFC00 if value < 0 else 0x7C00)


def round_away(value):
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def neighbors(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return [struct.unpack("<f", struct.pack("<I", bits + delta))[0]
            for delta in (-1, 0, 1)]


def scaled_block(scale, rotation=0):
    codes = [127, -127] + list(range(-15, 15))
    codes = codes[rotation:] + codes[:rotation]
    return [f32(scale * value) for value in codes]


def cases():
    result = [
        Case("signed_zero", 1, 128, tuple([0.0, -0.0] * 64)),
        Case("constant_signs", 1, 128,
             tuple(value for value in (1.0, -1.0, 0.5, -0.5) for _ in range(32))),
    ]
    # Reserve each block's last value for a fixed maximum. This makes its
    # scale exactly one while covering every output code from -127 to 127.
    codes = list(range(-127, 128))
    blocks = []
    for begin in range(0, len(codes), 31):
        block = codes[begin:begin + 31]
        blocks.extend(block + [0] * (31 - len(block)) + [127])
    blocks.extend([0] * (384 - len(blocks)))
    result.append(Case("signed_codes", 1, 384, tuple(map(float, blocks))))

    halfway = [127.0, -127.0]
    for value in (0.5, 1.5, 2.5, 63.5):
        halfway.extend(neighbors(value))
        halfway.extend(-item for item in neighbors(value))
    halfway.extend([0.0] * (32 - len(halfway)))
    result.append(Case("rounding_neighbors", 1, 128,
                       tuple(f32(value * scale) for scale in (1, 2**-8, 2**8, 2)
                             for value in halfway)))

    values = []
    for ordinal, peak in enumerate((1.0, 3.0, 0.7, 17.0)):
        peak = f32(peak)
        block = [f32(peak * (((index * 37 + ordinal * 11) % 251) - 125) / 127)
                 for index in range(32)]
        block[ordinal * 9] = peak if ordinal % 2 else -peak
        values.extend(block)
    result.append(Case("non_power_scales", 1, 128, tuple(values)))

    values = []
    for scale, sign in ((1.0, 1), (0.25, -1), (1.25, 1), (2**-8, -1)):
        block = [127.0, -127.0] + [sign * 0.49] * 15 + [sign * 1.51] * 15
        values.extend(f32(scale * value) for value in block)
    result.append(Case("quantized_sums", 1, 128, tuple(values)))

    for name, scales in (
        ("metadata_midpoints", (1 + 2**-11, 1 + 3 * 2**-11,
                                2**-14 - 2**-25, 2**-14 + 2**-25)),
        ("subnormal_metadata", (2**-24, 2**-25, 3 * 2**-25, 2**-20)),
        ("large_metadata", (65504.0, 65520.0, 65536.0, 32768.0)),
    ):
        result.append(Case(name, 1, 128,
                           tuple(value for ordinal, scale in enumerate(scales)
                                 for value in scaled_block(scale, ordinal * 7))))

    values = []
    for block in range(36):
        scale = 2.0 ** ((block % 9) - 4)
        values.extend(f32(scale * (value + (block % 7) / 8))
                      for value in scaled_block(1.0, block % 32))
    result.append(Case("row_and_group_order", 3, 384, tuple(values)))
    return tuple(result)


def quantize(values, mutation=""):
    """Encode complete logical blocks without production lane/word mappings."""
    result = bytearray()
    for group_begin in range(0, len(values), 128):
        group = values[group_begin:group_begin + 128]
        metadata = bytearray()
        payload = bytearray()
        for block_begin in range(0, 128, 32):
            block = group[block_begin:block_begin + 32]
            peak = max(abs(value) for value in
                       (group if mutation == "shared_scale" else block))
            scale = f32(peak / 127)
            if mutation == "early_half_scale":
                scale = struct.unpack("<e", half_bytes(scale))[0]
            inverse = f32(1 / scale) if scale else 0.0
            scaled = [f32(value * inverse) for value in block]
            if mutation == "ties_even":
                codes = [round(value) for value in scaled]
            elif mutation == "truncate":
                codes = [int(value) for value in scaled]
            else:
                codes = [round_away(value) for value in scaled]
            summed = f32(sum(codes) * scale)
            if mutation == "input_sum":
                summed = f32(sum(block))
            elif mutation == "half_scale_sum":
                summed = f32(sum(codes) * struct.unpack("<e", half_bytes(scale))[0])
            metadata.extend(half_bytes(scale) + half_bytes(summed))
            payload.extend(struct.pack("<32b", *codes))
        if mutation == "metadata_planes":
            metadata = b"".join(metadata[block + field:block + field + 2]
                                for field in (0, 2)
                                for block in range(0, 16, 4))
        if mutation == "reversed_words":
            payload = b"".join(payload[index:index + 4][::-1]
                               for index in range(0, 128, 4))
        result.extend(metadata)
        result.extend(payload)
    return bytes(result)


def verify(corpus):
    assert round_away(0.5) == 1 and round_away(-0.5) == -1
    assert round_away(neighbors(0.5)[0]) == 0
    assert round_away(-neighbors(0.5)[0]) == 0
    assert quantize([0.0] * 128) == bytes(144)
    assert quantize([1.0] * 128) == bytes.fromhex("08200050" * 4 + "7f" * 128)
    assert quantize([-1.0] * 128) == bytes.fromhex("082000d0" * 4 + "81" * 128)
    assert half_bytes(1 + 2**-11) == bytes.fromhex("003c")
    assert half_bytes(1 + 3 * 2**-11) == bytes.fromhex("023c")
    assert half_bytes(2**-25) == bytes(2)
    assert half_bytes(3 * 2**-25) == bytes.fromhex("0200")
    assert half_bytes(65520.0) == bytes.fromhex("007c")
    for case in corpus:
        assert len(case.values) == case.rows * case.width
        assert case.width % 128 == 0
        assert all(math.isfinite(value) and f32(value) == value for value in case.values)
    encoded = [quantize(case.values) for case in corpus]
    codes = encoded[2]
    observed = {value for begin in range(0, len(codes), 144)
                for value in struct.unpack("<128b", codes[begin + 16:begin + 144])}
    assert observed == set(range(-127, 128))
    for mutation in ("ties_even", "truncate", "shared_scale", "early_half_scale",
                     "input_sum", "half_scale_sum", "metadata_planes", "reversed_words"):
        assert any(quantize(case.values, mutation) != expected
                   for case, expected in zip(corpus, encoded)), mutation


def render_case(case, ordinal):
    name = case.name
    count = len(case.values)
    packed = quantize(case.values)
    input_bytes = struct.pack(f"<{count}f", *case.values)
    # Word-aligned subviews exercise ordinary caller-owned buffer origins.
    input_origin = 4 + (ordinal % 3) * 4
    output_origin = 4 + (ordinal % 3) * 4
    input_count = count + 8
    output_count = len(packed) + 32
    return f'''
global.rodata.def @{name}_input = align(4) bytes("{input_bytes.hex()}")

global.rodata.def @{name}_packed = align(4) bytes("{packed.hex()}")

func.def @initialize_{name}(%input: buffer) {{
  %source = global.load @{name}_input : buffer
  %origin = index.constant 0 : offset
  %length = index.constant {len(input_bytes)} : offset
  buffer.copy %source[%origin], %input[%origin], %length
  func.return
}}

func.def @reference_{name}(%tokens: index, %width: index, %input: buffer, %output: buffer) {{
  %source = global.load @{name}_packed : buffer
  %origin = index.constant 0 : offset
  %length = index.constant {len(packed)} : offset
  buffer.copy %source[%origin], %output[%origin], %length
  func.return
}}

check.scenario public @{name} {{
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %tokens = check.literal value({case.rows}) : index
    %width = check.literal value({case.width}) : index
    %input_storage = check.generate.fill value(123.25) : tensor<{input_count}xf32>
    %input = check.tensor.view %input_storage offset({input_origin}) : tensor<{input_count}xf32> -> tensor<{case.rows}x{case.width}xf32>
    check.generate<@initialize_{name}>(%input) : (tensor<{case.rows}x{case.width}xf32>) -> ()
    %output_storage = check.generate.fill value(-99) : tensor<{output_count}xi8>
    %output = check.tensor.view %output_storage offset({output_origin}) : tensor<{output_count}xi8> -> tensor<{len(packed)}xi8>
    check.compare<@ggml_quantize_q8_1_x4_f32, @reference_{name}>[%tokens, %width](%tokens, %width, %input, %output) : [index, index](index, index, tensor<{case.rows}x{case.width}xf32>, tensor<{len(packed)}xi8>) -> () {{
      check.expect.bitwise actual(%output_storage) expected(%output_storage) : tensor<{output_count}xi8>
      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{input_count}xf32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_quantize
'''


def render():
    corpus = cases()
    verify(corpus)
    header = '''// Generated by generate_q8_1_oracle.py.
// Scalar F32 arithmetic, integer rounding away from zero, and IEEE F16
// conversion define exact expected bytes independently of native lane layout.
// Every case compares complete guards and immutable input storage.

kernel.decl @ggml_quantize_q8_1_x4_f32(%tokens: index, %width: index) launch(%tokens_argument: index, %width_argument: index, %input: buffer, %output: buffer)
'''
    return header + "".join(render_case(case, ordinal)
                           for ordinal, case in enumerate(corpus))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("q8_1.loom"))
    args = parser.parse_args()
    content = render()
    if args.check:
        if args.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{args.output} differs from its authored oracle")
    else:
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
