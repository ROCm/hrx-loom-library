# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Guarded convolution storage and native-comparison fixture authoring."""

import struct
from dataclasses import dataclass


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


def float_bytes(values):
    return b"".join(struct.pack("<f", value) for value in values)


def render_fixture(case, arrays, expected, kernel, *, absolute_tolerance=None):
    name = case.name
    count = len(expected)
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
    if absolute_tolerance is None:
        comparison = f"check.expect.bitwise actual({output}) expected({output}) : tensor<{count}xf32>"
    else:
        comparison = f"check.expect.approximate actual({output}) expected({output}) atol({absolute_tolerance:.17g}) rtol(0.0) nan(different) : tensor<{count}xf32>, tensor<{count}xf32>"
    return f"""
{declarations}

{kernel}

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
      {comparison}
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xf32>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xf32>
{unchanged}
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_convolve
"""
