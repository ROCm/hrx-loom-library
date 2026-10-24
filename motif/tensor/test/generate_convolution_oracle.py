#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author exact F32 convolution checks from expanded, padded logical images."""

import argparse
import random
import struct
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from binary_reference import round_binary


@dataclass(frozen=True)
class Case:
    # Stable scenario and benchmark name.
    name: str
    # Physical input height, in pixels.
    height: int
    # Physical input width, in pixels.
    width: int
    # Number of input channels.
    inputs: int
    # Number of output channels.
    outputs: int
    # Odd spatial kernel width and height.
    extent: int = 1
    # Stored temporal planes; the final plane carries the active coefficients.
    planes: int = 1
    # Integer nearest-neighbor expansion factor in each spatial dimension.
    upsample: int = 1
    # Threads per workgroup in the caller's launch.
    threads: int = 32
    # Whether the caller adds a residual after the rounded bias addition.
    residual: bool = False
    # Whether output uses the residual's storage exactly.
    alias: bool = False


CASES = (
    Case("single", 1, 1, 1, 1),
    Case("padded_single", 1, 1, 3, 2, extent=3, planes=3, residual=True),
    Case("rectangle_tail", 3, 5, 3, 4, extent=3, planes=3),
    Case("nearest_double", 2, 3, 2, 3, extent=3, planes=2,
         upsample=2, threads=64, residual=True, alias=True),
    Case("nearest_triple", 2, 1, 2, 2, extent=5, planes=2, upsample=3),
    Case("exact_group", 4, 8, 2, 2, residual=True),
    Case("multiple_groups", 9, 17, 2, 3, threads=128, residual=True, alias=True),
    Case("wide_input", 1, 3, 384, 2, threads=64),
    Case("wide_output", 1, 2, 3, 384, threads=256),
    Case("decoder_output", 2, 2, 96, 3, extent=3, planes=3, threads=128),
    Case("fused_product", 1, 1, 2, 1),
    Case("bias_after_sum", 1, 1, 2, 1),
    Case("residual_after_bias", 1, 1, 2, 1, residual=True, alias=True),
)


def float_bytes(values):
    return b"".join(struct.pack("<f", value) for value in values)


def random_float(source):
    bits = (
        (source.getrandbits(1) << 31)
        | ((126 + source.randrange(-2, 3)) << 23)
        | source.getrandbits(23)
    )
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fixtures(case):
    source = random.Random(0xC07 + case.height * 37 + case.inputs * 13 + case.outputs)
    values = [random_float(source) for _ in range(case.inputs * case.height * case.width)]
    coefficient_count = case.outputs * case.inputs * case.extent * case.extent
    active = [random_float(source) for _ in range(coefficient_count)]
    bias = [random_float(source) for _ in range(case.outputs)]
    pixels = case.height * case.width * case.upsample**2
    residual = [random_float(source) for _ in range(case.outputs * pixels)]
    if case.name == "fused_product":
        values, active, bias = [-1.0, 1.0 + 2**-23], [1.0, 1.0 - 2**-23], [0.0]
    elif case.name == "bias_after_sum":
        values, active, bias = [float(2**24), float(-2**24)], [1.0, 1.0], [1.0]
    elif case.name == "residual_after_bias":
        values, active, bias = [float(2**24), 0.0], [1.0, 1.0], [1.0]
        residual = [float(-2**24)]

    # The oracle sees an explicitly expanded and padded image. It never uses
    # the production modulo/divide expression to recover source coordinates.
    padding = case.extent // 2
    expanded_width = case.width * case.upsample
    padded = []
    for channel in range(case.inputs):
        rows = [[0.0] * (expanded_width + 2 * padding) for _ in range(padding)]
        for y in range(case.height):
            row = [0.0] * padding
            for x in range(case.width):
                row.extend([values[(channel * case.height + y) * case.width + x]] * case.upsample)
            row.extend([0.0] * padding)
            rows.extend([row] * case.upsample)
        rows.extend([[0.0] * (expanded_width + 2 * padding) for _ in range(padding)])
        padded.append(rows)

    # Author logical coefficients separately, then embed them in the final
    # temporal plane. Earlier planes contain distinct finite values.
    weight = []
    for output in range(case.outputs):
        for channel in range(case.inputs):
            begin = (output * case.inputs + channel) * case.extent**2
            for plane in range(case.planes):
                if plane == case.planes - 1:
                    weight.extend(active[begin:begin + case.extent**2])
                else:
                    weight.extend([float(1000 + plane + channel)] * case.extent**2)

    expected = []
    for output in range(case.outputs):
        for y in range(case.height * case.upsample):
            for x in range(expanded_width):
                total = Fraction(0)
                for channel in range(case.inputs):
                    begin = (output * case.inputs + channel) * case.extent**2
                    for ky in range(case.extent):
                        for kx in range(case.extent):
                            product = Fraction(padded[channel][y + ky][x + kx])
                            product *= Fraction(active[begin + ky * case.extent + kx])
                            total = round_binary(total + product, 24)
                total = round_binary(total + Fraction(bias[output]), 24)
                if case.residual:
                    total = round_binary(total + Fraction(residual[len(expected)]), 24)
                expected.append(float(total))
    if case.name == "fused_product":
        assert expected == [-2**-46]
    elif case.name == "bias_after_sum":
        assert expected == [1.0]
    elif case.name == "residual_after_bias":
        assert expected == [0.0]
    return {"input": values, "weight": weight, "bias": bias, "residual": residual}, expected


def render_case(case):
    arrays, expected = fixtures(case)
    name = case.name
    count = len(expected)
    pixels = case.height * case.width * case.upsample**2
    groups = (pixels + case.threads - 1) // case.threads
    counts = {key: len(values) for key, values in arrays.items()}
    storage = {key: float_bytes([123.25] * 16 + values + [123.25] * 16)
               for key, values in arrays.items()}
    output_argument = "" if case.alias else ", %output: buffer"
    output = "%residual" if case.alias else "%output"
    arguments = ", ".join("%" + key for key in arrays) + ("" if case.alias else ", %output")
    types = ", ".join(f"tensor<{value}xf32>" for value in counts.values())
    if not case.alias:
        types += f", tensor<{count}xf32>"
    declarations = "\n\n".join(
        f'global.rodata.def @{name}_{key} = align(4) bytes("{data.hex()}")'
        for key, data in {**storage, "expected": float_bytes(expected)}.items()
    )
    initialize_arguments = ", ".join(f"%{key}: buffer" for key in arrays)
    initialize = "\n".join(f"""  %bytes_{key} = index.constant {len(data)} : offset
  %values_{key} = global.load @{name}_{key} : buffer
  buffer.copy %values_{key}[%origin], %{key}[%origin], %bytes_{key}"""
                           for key, data in storage.items())
    allocations = "\n".join(
        f"    %{key}_storage = check.generate.fill value(123.25) : tensor<{value + 32}xf32>"
        for key, value in counts.items()
    )
    storage_arguments = ", ".join(f"%{key}_storage" for key in arrays)
    storage_types = ", ".join(f"tensor<{value + 32}xf32>" for value in counts.values())
    views = "\n".join(
        f"    %{key} = check.tensor.view %{key}_storage offset(64) : tensor<{value + 32}xf32> -> tensor<{value}xf32>"
        for key, value in counts.items()
    )
    if not case.alias:
        views += f"""
    %output_storage = check.generate.fill value(123.25) : tensor<{count + 32}xf32>
    %output = check.tensor.view %output_storage offset(64) : tensor<{count + 32}xf32> -> tensor<{count}xf32>"""
    output_storage = "%residual_storage" if case.alias else "%output_storage"
    unchanged = "\n".join(
        f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{value + 32}xf32>"
        for key, value in counts.items() if key != "residual" or not case.alias
    )
    residual = f"""    %residual_view = buffer.view %residual[%origin] : buffer -> view<[%outputs]x[%pixels]xf32>
    %skip = view.load %residual_view[%channel, %pixel] : view<[%outputs]x[%pixels]xf32> -> f32
    %combined = scalar.addf %value, %skip : f32
""" if case.residual else ""
    result = "%combined" if case.residual else "%value"
    return f"""
{declarations}

kernel.def target(@convolution_wave32) @convolve_{name}() {{
  %groups = index.constant {groups} : index
  %channels = index.constant {case.outputs} : index
  %threads = index.constant {case.threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%groups, %channels, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%input: buffer, %weight: buffer, %bias: buffer, %residual: buffer{output_argument}) {{
  %height = index.constant {case.height} : index
  %width = index.constant {case.width} : index
  %inputs = index.constant {case.inputs} : index
  %outputs = index.constant {case.outputs} : index
  %extent = index.constant {case.extent} : index
  %planes = index.constant {case.planes} : index
  %upsample = index.constant {case.upsample} : index
  %pixels = index.constant {pixels} : index
  %threads = index.constant {case.threads} : index
  %origin = index.constant 0 : offset
  %group = kernel.workgroup.id<x> : index
  %channel = kernel.workgroup.id<y> : index
  %lane = kernel.workitem.id<x> : index
  %pixel = index.madd %group, %threads, %lane : index
  %live = index.cmp ult, %pixel, %pixels : index
  scf.if %live {{
    %value = template.apply<@image.convolution_nchw_element_f32>(%height, %width, %inputs, %outputs, %extent, %planes, %upsample, %channel, %pixel, %input, %weight, %bias) : (index, index, index, index, index, index, index, index, index, buffer, buffer, buffer) -> (f32)
{residual}    %output_view = buffer.view {output}[%origin] : buffer -> view<[%outputs]x[%pixels]xf32>
    view.store {result}, %output_view[%channel, %pixel] : f32, view<[%outputs]x[%pixels]xf32>
  }}
  kernel.return
}}

func.def @initialize_{name}({initialize_arguments}) {{
  %origin = index.constant 0 : offset
{initialize}
  func.return
}}

func.def @reference_{name}(%input: buffer, %weight: buffer, %bias: buffer, %residual: buffer{output_argument}) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {count * 4} : offset
  %values = global.load @{name}_expected : buffer
  buffer.copy %values[%origin], {output}[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
{allocations}
    check.generate<@initialize_{name}>({storage_arguments}) : ({storage_types}) -> ()
{views}
    %prefix = check.tensor.view {output_storage} offset(0) : tensor<{count + 32}xf32> -> tensor<16xf32>
    %suffix = check.tensor.view {output_storage} offset({(count + 16) * 4}) : tensor<{count + 32}xf32> -> tensor<16xf32>
    check.compare<@convolve_{name}, @reference_{name}>({arguments}) : ({types}) -> () {{
      check.expect.bitwise actual({output}) expected({output}) : tensor<{count}xf32>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xf32>
{unchanged}
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_convolve
"""


def render():
    header = """// Generated by generate_convolution_oracle.py.
// The reference explicitly expands/pads images and embeds active coefficients
// in their temporal storage separately. FMA, bias, and residual rounding are exact.

amdgpu.target<gfx11-generic> @convolution_wave32 {subgroup_size = 32}

template.decl @image.convolution_nchw_element_f32(%height: index, %width: index, %inputs: index, %outputs: index, %extent: index, %planes: index, %upsample: index, %channel: index, %pixel: index, %input: buffer, %weight: buffer, %bias: buffer) -> (f32)
"""
    return header + "".join(render_case(case) for case in CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("convolution.loom")
    content = render()
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{path} differs from its authored oracle")
    else:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
