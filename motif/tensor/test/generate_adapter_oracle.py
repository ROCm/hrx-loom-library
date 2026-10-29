# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Generate native adapter scenarios from independent scalar references."""

import argparse
from pathlib import Path
import struct

from adapter_reference import MatrixReference, rodata


def cohort_fixture(leaf):
    reference = MatrixReference()
    selected = (((0, 1.0), (1, -0.5), (2, 0.75)), ((1, 0.25), (3, -1.0)))
    values = {
        key: value for key, value in reference.shared.items() if key in ("a", "b")
    }
    for item in range(2):
        values["input" + str(item)] = reference.shared["input"]
        values["base" + str(item)] = reference.shared["base"]
        values["selection" + str(item)] = struct.pack(
            "<i12x", len(selected[item])
        ) + b"".join(
            struct.pack("<Qf4x", bank * reference.stride, strength)
            for bank, strength in selected[item]
        )
    expected = [reference.evaluate(selection) for selection in selected]
    tolerances = [reference.allowances(selection)[0] for selection in selected]
    names = (
        "input0",
        "input1",
        "a",
        "b",
        "selection0",
        "selection1",
        "base0",
        "base1",
        "output0",
        "output1",
    )
    sizes = {key: len(value) for key, value in values.items()} | {
        "output0": len(expected[0]),
        "output1": len(expected[1]),
    }
    declarations = ", ".join("%" + key + ": buffer" for key in names)
    arguments = ", ".join("%" + key for key in names)
    tensors = ", ".join(f"tensor<{sizes[key]}xi8>" for key in names)
    parts = [leaf]
    for key, value in values.items():
        parts.append(rodata(key, value))
    for item in range(2):
        parts.append(rodata("expected" + str(item), expected[item]))
    copies = []
    for key in names[:-2]:
        copies += [
            f"  %{key}_bytes = index.constant {sizes[key]} : offset",
            f"  %{key}_source = global.load @{key} : buffer",
            f"  buffer.copy %{key}_source[%origin], %{key}[%origin], %{key}_bytes",
        ]
    parts.append(f"""func.def @initialize_cohort({declarations}) {{
      %origin = index.constant 0 : offset
    {chr(10).join(copies)}
      func.return
    }}
    func.def @reference_cohort({declarations}) {{
      %origin = index.constant 0 : offset
      %bytes = index.constant {len(expected[0])} : offset
      %expected0 = global.load @expected0 : buffer
      %expected1 = global.load @expected1 : buffer
      buffer.copy %expected0[%origin], %output0[%origin], %bytes
      buffer.copy %expected1[%origin], %output1[%origin], %bytes
      func.return
    }}
    """)
    for order in ((0, 1), (1, 0)):
        name = "cohort_" + "".join(map(str, order))
        parts.append(f"""command.program.def @{name}() launch({declarations}) {{
      %scratch_bytes = index.constant {max(map(len, selected)) * reference.rows * reference.rank * 2} : offset
      %low = buffer.alloca<global> align(16) %scratch_bytes : buffer
    """)
        for item in order:
            parts.append(f"""  kernel.launch @count_down(%input{item}, %a, %selection{item}, %low) : (buffer, buffer, buffer, buffer)
      kernel.launch @count_up(%low, %b, %selection{item}, %base{item}, %output{item}) : (buffer, buffer, buffer, buffer, buffer)
    """)
        parts.append(f"""  command.return
    }}
    check.scenario public @{name}_test {{
      check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
      check.trial[1](%trial: index, %entropy: check.entropy) {{
    """)
        for key in names:
            parts.append(
                f"    %{key}_storage = check.generate.fill value(77) : tensor<{sizes[key] + 32}xi8>\n    %{key} = check.tensor.view %{key}_storage offset(16) : tensor<{sizes[key] + 32}xi8> -> tensor<{sizes[key]}xi8>\n"
            )
        for item in range(2):
            count = sizes["output" + str(item)]
            parts.append(f"""    %values{item} = check.tensor.view %output{item}_storage offset(16) : tensor<{count + 32}xi8> -> tensor<{count // 2}xbf16>
        %prefix{item} = check.tensor.view %output{item}_storage offset(0) : tensor<{count + 32}xi8> -> tensor<16xi8>
        %suffix{item} = check.tensor.view %output{item}_storage offset({count + 16}) : tensor<{count + 32}xi8> -> tensor<16xi8>
    """)
        parts.append(
            f"    check.generate<@initialize_cohort>({arguments}) : ({tensors}) -> ()\n    check.compare<@{name}, @reference_cohort>({arguments}) : ({tensors}) -> () {{\n"
        )
        for key in names[:-2]:
            parts.append(
                f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{sizes[key] + 32}xi8>\n"
            )
        for item in range(2):
            parts.append(f"""      check.expect.approximate actual(%values{item}) expected(%values{item}) atol({tolerances[item]!r}) rtol(0.0) nan(different) : tensor<{sizes["output" + str(item)] // 2}xbf16>, tensor<{sizes["output" + str(item)] // 2}xbf16>
          check.expect.bitwise actual(%prefix{item}) expected(%prefix{item}) : tensor<16xi8>
          check.expect.bitwise actual(%suffix{item}) expected(%suffix{item}) : tensor<16xi8>
    """)
        parts.append("    }\n  }\n  check.return\n}\n")
        parts.append(f"check.benchmark<@{name}_test> @{name}_benchmark\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf", type=Path, required=True)
    args = parser.parse_args()
    leaf = args.leaf.read_text()
    text = cohort_fixture(leaf)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
