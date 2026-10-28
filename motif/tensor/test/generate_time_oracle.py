#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author sinusoidal time references with binary64 transcendental evaluation."""

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from binary_reference import f32


@dataclass(frozen=True)
class Case:
    # Stable scenario name.
    name: str
    # Number of frequencies in each trigonometric half.
    half_width: int
    # Logical BF16 time values, distinct from physical padding.
    times: tuple[float, ...]
    # Physical output row count.
    capacity: int
    # Negative natural logarithm of the frequency period, rounded to F32.
    negative_log_period: float = f32(-math.log(10000))
    # Multiplier applied to time before its frequency, rounded to F32.
    time_scale: float = 1000.0
    # Threads per workgroup in the test leaf.
    threads: int = 256
    # Whether the entire output has exact zero/one answers.
    exact: bool = False


CASES = (
    Case("zero", 128, (0.0,), 1, threads=32, exact=True),
    Case("padded_model_rows", 128, (1.0, 0.5, 0.25), 16),
    Case("model_range", 128, (1.0, 0.75, 0.5, 0.125, 2**-8, 2**-16), 6),
    Case("negative", 128, (-1.0, -0.25, -2**-16), 4, threads=64),
    Case("single_frequency", 1, (0.25,), 1, time_scale=1.0, threads=32),
    Case("odd_half", 3, (-0.5, 0.125, 1.0), 4,
         negative_log_period=f32(-math.log(16)), time_scale=4.0, threads=32),
    Case("short_period", 7, (-1.0, 0.125, 1.0), 4,
         negative_log_period=f32(-math.log(2)), time_scale=1.0, threads=64),
    Case("channel_tail", 129, (1.0, 0.5, 2**-8), 3, threads=64),
    Case("zero_scale", 7, (0.5, 1.0), 3, time_scale=0.0, threads=32, exact=True),
)


def float_literal(value):
    text = format(value, ".17g")
    return text if "." in text or "e" in text else text + ".0"


def reference(case):
    expected = []
    maximum_phase = 0.0
    for row in range(case.capacity):
        time = case.times[row] if row < len(case.times) else 0.0
        scaled_time = f32(time * case.time_scale)
        # Build the complete frequency row first, then concatenate the two
        # trigonometric halves; no production modulo or lane mapping is used.
        phases = []
        for frequency in range(case.half_width):
            exponent = f32(f32(case.negative_log_period * f32(frequency)) /
                           f32(case.half_width))
            factor = f32(math.exp(exponent))
            phase = f32(scaled_time * factor)
            phases.append(phase)
            maximum_phase = max(maximum_phase, abs(phase))
        expected.extend(math.cos(phase) for phase in phases)
        expected.extend(math.sin(phase) for phase in phases)
    # The BF16 rounding term is relative; native F32 phase and transcendental
    # error has a separate absolute requirement scaled by the largest phase.
    # Sixteen F32 unit roundoffs per phase, plus the same unit absolute floor,
    # leave the reference independent of a device's approximation recipe.
    absolute = (maximum_phase + 1) * 2**-20
    return expected, absolute


def render_case(case):
    name = case.name
    count = len(case.times)
    width = case.half_width * 2
    elements = case.capacity * width
    groups = (width + case.threads - 1) // case.threads
    input_values = (123.0,) * 16 + case.times + (123.0,) * 16
    input_words = []
    for value in input_values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        assert not bits & 0xFFFF, (name, value)
        input_words.append(bits >> 16)
    input_bytes = struct.pack(f"<{len(input_words)}H", *input_words)
    expected, absolute = reference(case)
    expected_bytes = struct.pack(f"<{elements}f", *expected)
    comparison = (
        f"check.expect.bitwise actual(%output) expected(%output) : tensor<{elements}xf32>"
        if case.exact else
        f"check.expect.approximate actual(%output) expected(%output) atol({absolute:.17g}) rtol(0.00390625) nan(different) : tensor<{elements}xf32>, tensor<{elements}xf32>"
    )
    padding_view = padding_check = ""
    if case.capacity > count:
        padding_elements = (case.capacity - count) * width
        padding_view = f"    %padding = check.tensor.view %output offset({count * width * 4}) : tensor<{elements}xf32> -> tensor<{padding_elements}xf32>\n"
        padding_check = f"      check.expect.bitwise actual(%padding) expected(%padding) : tensor<{padding_elements}xf32>\n"
    return f'''
global.rodata.def @{name}_input = align(4) bytes("{input_bytes.hex()}")

global.rodata.def @{name}_expected = align(4) bytes("{expected_bytes.hex()}")

kernel.def target(@time_wave32) @features_{name}() {{
  %rows = index.constant {case.capacity} : index
  %groups = index.constant {groups} : index
  %threads = index.constant {case.threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %groups, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %output: buffer) {{
  %count = index.constant {count} : index
  %rows = index.constant {case.capacity} : index
  %half_width = index.constant {case.half_width} : index
  %width = index.constant {width} : index
  %threads = index.constant {case.threads} : index
  %row = kernel.workgroup.id<x> : index
  %group = kernel.workgroup.id<y> : index
  %lane = kernel.workitem.id<x> : index
  %channel = index.madd %group, %threads, %lane : index
  %live = index.cmp ult, %channel, %width : index
  scf.if %live {{
    %origin = index.constant 0 : offset
    %inputs = buffer.view %input[%origin] : buffer -> view<[%count]xbf16>
    %outputs = buffer.view %output[%origin] : buffer -> view<[%rows]x[%width]xf32>
    %logical = index.cmp ult, %row, %count : index
    %time = scf.if %logical -> (bf16) {{
      %stored = view.load %inputs[%row] : view<[%count]xbf16> -> bf16
      scf.yield %stored : bf16
    }} else {{
      %padding_time = scalar.constant 0.0 : bf16
      scf.yield %padding_time : bf16
    }}
    %negative_log_period = scalar.constant {float_literal(case.negative_log_period)} : f32
    %time_scale = scalar.constant {float_literal(case.time_scale)} : f32
    %feature = template.apply<@time.sinusoidal_element_bf16>(%half_width, %channel, %negative_log_period, %time_scale, %time) : (index, index, f32, f32, bf16) -> (bf16)
    %value = scalar.extf %feature : bf16 to f32
    view.store %value, %outputs[%row, %channel] : f32, view<[%rows]x[%width]xf32>
  }}
  kernel.return
}}

func.def @initialize_{name}(%input: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {len(input_bytes)} : offset
  %values = global.load @{name}_input : buffer
  buffer.copy %values[%origin], %input[%origin], %bytes
  func.return
}}

func.def @reference_{name}(%input: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {len(expected_bytes)} : offset
  %values = global.load @{name}_expected : buffer
  buffer.copy %values[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %input_storage = check.generate.fill value(123.0) : tensor<{count + 32}xbf16>
    check.generate<@initialize_{name}>(%input_storage) : (tensor<{count + 32}xbf16>) -> ()
    %input = check.tensor.view %input_storage offset(32) : tensor<{count + 32}xbf16> -> tensor<{count}xbf16>
    %output_storage = check.generate.fill value(123.25) : tensor<{elements + 32}xf32>
    %output = check.tensor.view %output_storage offset(64) : tensor<{elements + 32}xf32> -> tensor<{elements}xf32>
    %prefix = check.tensor.view %output_storage offset(0) : tensor<{elements + 32}xf32> -> tensor<16xf32>
    %suffix = check.tensor.view %output_storage offset({(elements + 16) * 4}) : tensor<{elements + 32}xf32> -> tensor<16xf32>
{padding_view}    check.compare<@features_{name}, @reference_{name}>(%input, %output) : (tensor<{count}xbf16>, tensor<{elements}xf32>) -> () {{
      {comparison}
{padding_check}      check.expect.bitwise actual(%input_storage) expected(%input_storage) : tensor<{count + 32}xbf16>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xf32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_features
'''


def render():
    header = '''// Generated by generate_time_oracle.py.
// Binary64 trigonometric answers retain precision across BF16 publication.
// Exact padding and guards are checked independently of approximate features.

amdgpu.target<gfx11-generic> @time_wave32 {subgroup_size = 32}

template.decl @time.sinusoidal_element_bf16(%half_width: index, %channel: index, %negative_log_period: f32, %time_scale: f32, %time: bf16) -> (bf16)
'''
    return header + ''.join(render_case(case) for case in CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(__file__).with_name('time_test.loom')
    content = render()
    if args.check:
        if path.read_text(encoding='utf-8') != content:
            raise SystemExit(f'{path} differs from its authored oracle')
    else:
        path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
