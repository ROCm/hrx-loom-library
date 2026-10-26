# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Generate native adapter scenarios from independent scalar references."""

import argparse
import json
from pathlib import Path
import struct

from adapter_reference import MatrixReference, bf16, f32, rodata


def matrix_fixture(leaf, exact, options):
    reference = MatrixReference(exact, options)
    rows, inputs, outputs, rank = (
        reference.rows,
        reference.inputs,
        reference.outputs,
        reference.rank,
    )
    stride, column_major, alias = (
        reference.stride,
        reference.column_major,
        reference.alias,
    )
    shared, selections = reference.shared, reference.selections
    oracle, allowances = reference.evaluate, reference.allowances
    parts = [
        "// Independent mathematical adapter oracle and untouched-storage checks.\n"
    ]
    for name, value in shared.items():
        parts.append(rodata(name, value))
    parts.append(
        """
    kernel.def target(@count_wave32) @copy_base() {
      %groups = index.constant 40 : index
      %threads = index.constant 128 : index
      %c1 = index.constant 1 : index
      kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%threads, %c1, %c1) : index
    } launch(%base: buffer, %output: buffer) {
      %origin = index.constant 0 : offset
      %threads = index.constant 128 : index
      %group = kernel.workgroup.id<x> : index
      %lane = kernel.workitem.id<x> : index
      %position = index.madd %group, %threads, %lane : index
      %source = buffer.view %base[%origin] : buffer -> view<5120xi16>
      %target = buffer.view %output[%origin] : buffer -> view<5120xi16>
      %value = view.load %source[%position] : view<5120xi16> -> i16
      view.store %value, %target[%position] : i16, view<5120xi16>
      kernel.return
    }

    command.program.def @apply_count(%count: index) launch(%input: buffer, %a: buffer, %b: buffer, %selections: buffer, %base: buffer, %output: buffer) where [range(%count, 0, 2147483647)] {
      %c0 = index.constant 0 : index
      %active = index.cmp ne, %count, %c0 : index
      scf.if %active {
        %bank_bytes = index.constant 5120 : offset
        %low_bytes = index.scale %count, %bank_bytes : index, offset -> offset
        %low = buffer.alloca<global> align(16) %low_bytes : buffer
        kernel.launch @count_down(%input, %a, %selections, %low) : (buffer, buffer, buffer, buffer)
        kernel.launch @count_up(%low, %b, %selections, %base, %output) : (buffer, buffer, buffer, buffer, buffer)
      } else {
        kernel.launch @copy_base(%base, %output) : (buffer, buffer)
      }
      command.return
    }
    """.replace(
            "%groups = index.constant 40 : index",
            f"%groups = index.constant {rows * outputs // 128} : index",
        )
        .replace("view<5120xi16>", f"view<{rows * outputs}xi16>")
        .replace(
            "%bank_bytes = index.constant 5120 : offset",
            f"%bank_bytes = index.constant {rows * rank * 2} : offset",
        )
    )
    for name, selected in selections:
        metadata = struct.pack("<i12x", len(selected)) + b"".join(
            struct.pack("<Qf4x", bank * stride, scale) for bank, scale in selected
        )
        expected = oracle(selected)
        tolerance, _ = allowances(selected)
        parts.extend(
            (
                rodata(name + "_selections", metadata),
                rodata(name + "_expected", expected),
            )
        )
        values = dict(shared, selections=metadata)
        names = ("input", "a", "b", "selections", "base", "output")
        declarations = ", ".join("%" + key + ": buffer" for key in names)
        arguments = ", ".join("%" + key for key in names)
        sizes = {key: len(value) for key, value in values.items()} | {
            "output": len(expected)
        }
        tensors = ", ".join(f"tensor<{sizes[key]}xi8>" for key in names)
        copies = []
        for key in names[:-1]:
            symbol = name + "_selections" if key == "selections" else key
            copies += [
                f"  %{key}_bytes = index.constant {sizes[key]} : offset",
                f"  %{key}_source = global.load @{symbol} : buffer",
                f"  buffer.copy %{key}_source[%origin], %{key}[%origin], %{key}_bytes",
            ]
        parts.append(f"""func.def @initialize_{name}({declarations}) {{
      %origin = index.constant 0 : offset
    {chr(10).join(copies)}
      func.return
    }}
    func.def @reference_{name}({declarations}) {{
      %origin = index.constant 0 : offset
      %bytes = index.constant {len(expected)} : offset
      %expected = global.load @{name}_expected : buffer
      buffer.copy %expected[%origin], %output[%origin], %bytes
      func.return
    }}
    check.scenario public @{name} configure[1](%configuration: index, %configuration_entropy: check.entropy) {{
      check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
      %count = check.literal value({len(selected)}) : index
      check.trial[1](%trial: index, %entropy: check.entropy) {{
    """)
        for key in names:
            if key == "output" and alias:
                parts.append(
                    f"    %output_storage = check.tensor.view %base_storage offset(0) : tensor<{sizes[key] + 32}xi8> -> tensor<{sizes[key] + 32}xi8>\n"
                )
            else:
                parts.append(
                    f"    %{key}_storage = check.generate.fill value(77) : tensor<{sizes[key] + 32}xi8>\n"
                )
            parts.append(
                f"    %{key} = check.tensor.view %{key}_storage offset(16) : tensor<{sizes[key] + 32}xi8> -> tensor<{sizes[key]}xi8>\n"
            )
        parts.append(
            f"    %output_values = check.tensor.view %output_storage offset(16) : tensor<{sizes['output'] + 32}xi8> -> tensor<{sizes['output'] // 2}xbf16>\n"
        )
        parts.append(
            f"    %output_prefix = check.tensor.view %output_storage offset(0) : tensor<{sizes['output'] + 32}xi8> -> tensor<16xi8>\n"
        )
        parts.append(
            f"    %output_suffix = check.tensor.view %output_storage offset({sizes['output'] + 16}) : tensor<{sizes['output'] + 32}xi8> -> tensor<16xi8>\n"
        )
        parts.append(
            f"    check.generate<@initialize_{name}>({arguments}) : ({tensors}) -> ()\n    check.compare<@apply_count, @reference_{name}>[%count]({arguments}) : [index]({tensors}) -> () {{\n"
        )
        for key in names[:-1]:
            if key == "base" and alias:
                continue
            parts.append(
                f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{sizes[key] + 32}xi8>\n"
            )
        if tolerance == 0:
            parts.append(
                f"      check.expect.bitwise actual(%output_storage) expected(%output_storage) : tensor<{sizes['output'] + 32}xi8>\n"
            )
        else:
            parts.append(
                f"      check.expect.close actual(%output_values) expected(%output_values) atol({tolerance!r}) rtol(0.0) nan(different) : tensor<{sizes['output'] // 2}xbf16>\n"
            )
            parts.append(
                "      check.expect.bitwise actual(%output_prefix) expected(%output_prefix) : tensor<16xi8>\n      check.expect.bitwise actual(%output_suffix) expected(%output_suffix) : tensor<16xi8>\n"
            )
        parts.append("    }\n  }\n  check.return\n}\n")
        parts.append(f"check.benchmark<@{name}> @{name}_benchmark\n")
    for name, selected in selections:
        if not selected:
            continue
        metadata = struct.pack("<i12x", len(selected)) + b"".join(
            struct.pack("<Qf4x", bank * stride, scale) for bank, scale in selected
        )
        expected_low = oracle(selected, return_low=True)
        _, tolerance = allowances(selected)
        parts.append(rodata(name + "_expected_low", expected_low))
        values = dict(
            input=shared["input"], a=shared["a"], selections=metadata, low=expected_low
        )
        names = tuple(values)
        declarations = ", ".join("%" + key + ": buffer" for key in names)
        arguments = ", ".join("%" + key for key in names)
        sizes = {key: len(value) for key, value in values.items()}
        tensors = ", ".join(f"tensor<{sizes[key]}xi8>" for key in names)
        copies = []
        for key in names[:-1]:
            symbol = name + "_selections" if key == "selections" else key
            copies += [
                f"  %{key}_bytes = index.constant {sizes[key]} : offset",
                f"  %{key}_source = global.load @{symbol} : buffer",
                f"  buffer.copy %{key}_source[%origin], %{key}[%origin], %{key}_bytes",
            ]
        parts.append(f"""func.def @initialize_down_{name}({declarations}) {{
      %origin = index.constant 0 : offset
    {chr(10).join(copies)}
      func.return
    }}
    func.def @reference_down_{name}({declarations}) {{
      %origin = index.constant 0 : offset
      %bytes = index.constant {len(expected_low)} : offset
      %expected = global.load @{name}_expected_low : buffer
      buffer.copy %expected[%origin], %low[%origin], %bytes
      func.return
    }}
    check.scenario public @down_{name} {{
      check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
      check.trial[1](%trial: index, %entropy: check.entropy) {{
    """)
        for key in names:
            parts.append(
                f"    %{key}_storage = check.generate.fill value(77) : tensor<{sizes[key] + 32}xi8>\n    %{key} = check.tensor.view %{key}_storage offset(16) : tensor<{sizes[key] + 32}xi8> -> tensor<{sizes[key]}xi8>\n"
            )
        parts.append(
            f"    %low_values = check.tensor.view %low_storage offset(16) : tensor<{sizes['low'] + 32}xi8> -> tensor<{sizes['low'] // 2}xbf16>\n"
        )
        parts.append(
            f"    %low_prefix = check.tensor.view %low_storage offset(0) : tensor<{sizes['low'] + 32}xi8> -> tensor<16xi8>\n"
        )
        parts.append(
            f"    %low_suffix = check.tensor.view %low_storage offset({sizes['low'] + 16}) : tensor<{sizes['low'] + 32}xi8> -> tensor<16xi8>\n"
        )
        parts.append(
            f"    check.generate<@initialize_down_{name}>({arguments}) : ({tensors}) -> ()\n    check.compare<@count_down, @reference_down_{name}>({arguments}) : ({tensors}) -> () {{\n"
        )
        for key in names[:-1]:
            parts.append(
                f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{sizes[key] + 32}xi8>\n"
            )
        if exact:
            parts.append(
                f"      check.expect.bitwise actual(%low_storage) expected(%low_storage) : tensor<{sizes['low'] + 32}xi8>\n"
            )
        else:
            parts.append(
                f"      check.expect.close actual(%low_values) expected(%low_values) atol({tolerance!r}) rtol(0.0) nan(different) : tensor<{sizes['low'] // 2}xbf16>\n"
            )
            parts.append(
                "      check.expect.bitwise actual(%low_prefix) expected(%low_prefix) : tensor<16xi8>\n      check.expect.bitwise actual(%low_suffix) expected(%low_suffix) : tensor<16xi8>\n"
            )
        parts.append("    }\n  }\n  check.return\n}\n")
        parts.append(f"check.benchmark<@down_{name}> @down_{name}_benchmark\n")
    down, up = leaf.split("kernel.def target(@count_wave32) @count_up", 1)
    down = down.replace(
        "%groups_x = index.constant 5 : index",
        f"%groups_x = index.constant {rows // 16} : index",
    )
    up = up.replace(
        "%groups_x = index.constant 1 : index",
        f"%groups_x = index.constant {(outputs + 127) // 128} : index",
    ).replace(
        "%groups_y = index.constant 2 : index",
        f"%groups_y = index.constant {(rows + 63) // 64} : index",
    )
    leaf = (
        (down + "kernel.def target(@count_wave32) @count_up" + up)
        .replace(
            "%rows = index.constant 80 : index",
            f"%rows = index.constant {rows} : index",
        )
        .replace(
            "%inputs = index.constant 64 : index",
            f"%inputs = index.constant {inputs} : index",
        )
        .replace(
            "%outputs = index.constant 64 : index",
            f"%outputs = index.constant {outputs} : index",
        )
    )
    if column_major:
        leaf = leaf.replace(
            "%output_layout = encoding.layout.dense",
            "%output_layout = encoding.layout.strided [1, %rows]",
        )
    assert rows > 0 and rows % 16 == 0 and inputs % 64 == 0 and outputs % 16 == 0
    return leaf + "\n".join(parts)


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
            parts.append(f"""      check.expect.close actual(%values{item}) expected(%values{item}) atol({tolerances[item]!r}) rtol(0.0) nan(different) : tensor<{sizes["output" + str(item)] // 2}xbf16>
          check.expect.bitwise actual(%prefix{item}) expected(%prefix{item}) : tensor<16xi8>
          check.expect.bitwise actual(%suffix{item}) expected(%suffix{item}) : tensor<16xi8>
    """)
        parts.append("    }\n  }\n  check.return\n}\n")
        parts.append(f"check.benchmark<@{name}_test> @{name}_benchmark\n")
    return "\n".join(parts)


def scalar_fixture(leaf):
    rows, width, rank, banks, stride = 2, 65, 32, 7, 784

    def packed_scalar(values, storage="bf16"):
        if storage == "f32":
            return struct.pack("<" + "f" * len(values), *values)
        words = [
            struct.unpack("<I", struct.pack("<f", bf16(value)))[0] >> 16
            for value in values
        ]
        return struct.pack("<" + "H" * len(words), *words)

    selections = (
        ("empty", ()),
        ("single", ((3, 0.5),)),
        ("pair", ((1, -0.75), (5, 1.0))),
        ("odd", ((0, 1.0), (1, -0.5), (2, 0.75))),
        ("even", ((0, 1.0), (2, -0.5), (4, 0.25), (6, 0.75))),
        ("seven", tuple((i, (1.0, -0.5, 0.25)[i % 3]) for i in range(banks))),
        ("duplicate", ((1, 0.25), (3, -1.0), (1, 0.75))),
        ("cancel", ((1, 1.0), (1, -1.0))),
        ("fractional", ((1, 0.3), (4, 0.7), (6, -0.625))),
        ("zero", ((1, 0.0), (6, -0.0), (4, 0.0))),
    )
    inputs = [
        [
            [
                [
                    (((row * 7 + channel * 5 + layer * 3) % 15) - 7) / 8
                    for layer in range(12)
                ]
                for channel in range(width)
            ]
            for row in range(rows)
        ]
    ][0]
    weight = [1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, -1 - 3 / 256] + [
        (k - 4) / 8 for k in range(8)
    ]
    down = [
        [
            [(((bank * 11 + rank * 5 + layer * 3) % 7) - 3) / 8 for layer in range(12)]
            for rank in range(rank)
        ]
        for bank in range(banks)
    ]
    up = [
        [(((bank * 5 + rank * 7) % 11) - 5) / 8 for rank in range(rank)]
        for bank in range(banks)
    ]

    def factors(banks):
        result = bytearray()
        for bank in banks:
            encoded = packed_scalar(bank)
            result.extend(encoded)
            result.extend(bytes([77]) * (stride - len(encoded)))
        return bytes(result)

    def dot(values, coefficients):
        total = 0.0
        for value, coefficient in zip(values, coefficients):
            total += value * coefficient
            assert f32(total) == total
        return total

    def oracle(selected):
        result = []
        for row in inputs:
            for values in row:
                base = bf16(dot(values, [bf16(value) for value in weight]))
                total = 0.0
                for bank, strength in selected:
                    for element in range(rank):
                        low = bf16(f32(dot(values, down[bank][element]) * f32(strength)))
                        total += low * up[bank][element]
                        assert f32(total) == total
                result.append(bf16(f32(base + total)))
        return packed_scalar(result)

    shared = {
        "input": packed_scalar(
            [
                inputs[row][channel][layer]
                for row in range(rows)
                for layer in range(12)
                for channel in range(width)
            ]
        ),
        "a": factors([[value for rank in bank for value in rank] for bank in down]),
        "b": factors(up),
    }
    parts = [leaf]
    for key, value in shared.items():
        parts.append(rodata(key, value))
    for storage in ("f32", "bf16"):
        parts.append(rodata("weight_" + storage, packed_scalar(weight, storage)))
        for selection_name, selected in selections:
            name = storage + "_" + selection_name
            metadata = struct.pack("<i12x", len(selected)) + b"".join(
                struct.pack("<Qf4x", bank * stride, strength)
                for bank, strength in selected
            )
            expected = oracle(selected)
            values = dict(
                shared, weight=packed_scalar(weight, storage), selections=metadata
            )
            names = ("input", "weight", "a", "b", "selections", "output")
            sizes = {key: len(value) for key, value in values.items()} | {
                "output": len(expected)
            }
            declarations = ", ".join("%" + key + ": buffer" for key in names)
            arguments = ", ".join("%" + key for key in names)
            tensors = ", ".join(f"tensor<{sizes[key]}xi8>" for key in names)
            parts.extend(
                (
                    rodata(name + "_selections", metadata),
                    rodata(name + "_expected", expected),
                )
            )
            copies = []
            for key in names[:-1]:
                symbol = (
                    name + "_selections"
                    if key == "selections"
                    else "weight_" + storage
                    if key == "weight"
                    else key
                )
                copies += [
                    f"  %{key}_bytes = index.constant {sizes[key]} : offset",
                    f"  %{key}_source = global.load @{symbol} : buffer",
                    f"  buffer.copy %{key}_source[%origin], %{key}[%origin], %{key}_bytes",
                ]
            parts.append(f"""func.def @initialize_{name}({declarations}) {{
      %origin = index.constant 0 : offset
    {chr(10).join(copies)}
      func.return
    }}
    func.def @reference_{name}({declarations}) {{
      %origin = index.constant 0 : offset
      %bytes = index.constant {len(expected)} : offset
      %expected = global.load @{name}_expected : buffer
      buffer.copy %expected[%origin], %output[%origin], %bytes
      func.return
    }}
    check.scenario public @{name} {{
      check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
      check.trial[1](%trial: index, %entropy: check.entropy) {{
    """)
            for key in names:
                parts.append(
                    f"    %{key}_storage = check.generate.fill value(77) : tensor<{sizes[key] + 32}xi8>\n    %{key} = check.tensor.view %{key}_storage offset(16) : tensor<{sizes[key] + 32}xi8> -> tensor<{sizes[key]}xi8>\n"
                )
            parts.append(
                f"    check.generate<@initialize_{name}>({arguments}) : ({tensors}) -> ()\n    check.compare<@count_scalar_{storage}, @reference_{name}>({arguments}) : ({tensors}) -> () {{\n"
            )
            for key in names:
                parts.append(
                    f"      check.expect.bitwise actual(%{key}_storage) expected(%{key}_storage) : tensor<{sizes[key] + 32}xi8>\n"
                )
            parts.append("    }\n  }\n  check.return\n}\n")
            parts.append(f"check.benchmark<@{name}> @{name}_benchmark\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("matrix", "cohort", "scalar"), default="matrix"
    )
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--options", type=json.loads, default={})
    args = parser.parse_args()
    if args.mode != "matrix" and (args.exact or args.options):
        parser.error("--exact and --options apply only to matrix scenarios")
    leaf = args.leaf.read_text()
    if args.mode == "matrix":
        text = matrix_fixture(leaf, args.exact, args.options)
    elif args.mode == "cohort":
        text = cohort_fixture(leaf)
    else:
        text = scalar_fixture(leaf)
    args.output.write_text(text)


if __name__ == "__main__":
    main()
