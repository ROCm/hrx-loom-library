#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Author exact embedding-row comparisons over raw BF16 storage words."""

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    # Stable scenario name.
    name: str
    # Number of immutable table rows.
    vocabulary: int
    # Number of BF16 values in each row.
    width: int
    # Ordered IDs selected by the caller, including duplicates.
    ids: tuple[int, ...]
    # Threads per workgroup in the caller's launch.
    threads: int
    # Raw-word pattern period; a non-power-of-two distinguishes high row IDs.
    word_period: int = 65536


CASES = (
    Case("single", 1, 1, (0,), 32),
    Case("odd_width", 17, 31, (16, 0, 7, 7), 32),
    Case("vocabulary_boundary", 151936, 3, (151935, 0, 65536, 65535, 128), 64, 65521),
    Case("encoder_width", 3, 2560, (2, 0, 1, 2), 256),
    Case("all_encodings", 256, 256, tuple(reversed(range(256))), 64),
)


def render_case(case):
    name = case.name
    rows = len(case.ids)
    elements = rows * case.width
    weights = case.vocabulary * case.width
    groups = (case.width + case.threads - 1) // case.threads
    assert all(0 <= token < case.vocabulary for token in case.ids)
    data = struct.pack(f"<{rows}i", *case.ids)
    return f'''
global.rodata.def @{name}_ids = align(4) bytes("{data.hex()}")

kernel.def target(@embedding_wave32) @gather_{name}() {{
  %rows = index.constant {rows} : index
  %groups = index.constant {groups} : index
  %threads = index.constant {case.threads} : index
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%rows, %groups, %unit) workgroup_size(%threads, %unit, %unit) : index
}} launch(%ids: buffer, %weight: buffer, %output: buffer) {{
  %vocabulary = index.constant {case.vocabulary} : index
  %width = index.constant {case.width} : index
  %rows = index.constant {rows} : index
  %threads = index.constant {case.threads} : index
  %row = kernel.workgroup.id<x> : index
  %group = kernel.workgroup.id<y> : index
  %lane = kernel.workitem.id<x> : index
  %channel = index.madd %group, %threads, %lane : index
  %live = index.cmp ult, %channel, %width : index
  scf.if %live {{
    %origin = index.constant 0 : offset
    %tokens = buffer.view %ids[%origin] : buffer -> view<[%rows]xi32>
    %outputs = buffer.view %output[%origin] : buffer -> view<[%rows]x[%width]xbf16>
    %token = view.load %tokens[%row] : view<[%rows]xi32> -> i32
    %value = template.apply<@embedding.lookup_bf16>(%vocabulary, %width, %token, %channel, %weight) : (index, index, i32, index, buffer) -> (bf16)
    view.store %value, %outputs[%row, %channel] : bf16, view<[%rows]x[%width]xbf16>
  }}
  kernel.return
}}

func.def @initialize_{name}(%ids: buffer) {{
  %origin = index.constant 0 : offset
  %bytes = index.constant {len(data)} : offset
  %values = global.load @{name}_ids : buffer
  buffer.copy %values[%origin], %ids[%origin], %bytes
  func.return
}}

func.def @reference_{name}(%ids: buffer, %weight: buffer, %output: buffer) {{
  %origin = index.constant 0 : offset
  %begin = index.constant 0 : index
  %unit = index.constant 1 : index
  %rows = index.constant {rows} : index
  %row_bytes = index.constant {case.width * 2} : offset
  %tokens = buffer.view %ids[%origin] : buffer -> view<[%rows]xi32>
  scf.for %row = [%begin to %rows step %unit] {{
    %token = view.load %tokens[%row] : view<[%rows]xi32> -> i32
    %selected = index.cast %token : i32 to index
    %source_origin = index.scale %selected, %row_bytes : index, offset -> offset
    %destination_origin = index.scale %row, %row_bytes : index, offset -> offset
    buffer.copy %weight[%source_origin], %output[%destination_origin], %row_bytes
  }}
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %id_storage = check.generate.fill value(-1) : tensor<{rows + 32}xi32>
    %ids = check.tensor.view %id_storage offset(64) : tensor<{rows + 32}xi32> -> tensor<{rows}xi32>
    check.generate<@initialize_{name}>(%ids) : (tensor<{rows}xi32>) -> ()
    %weight_storage = check.generate.iota offset(-32768) step(1) period({case.word_period}) : tensor<{weights + 32}xi16>
    %weight = check.tensor.view %weight_storage offset(32) : tensor<{weights + 32}xi16> -> tensor<{weights}xbf16>
    %output_storage = check.generate.fill value(-12345) : tensor<{elements + 32}xi16>
    %output = check.tensor.view %output_storage offset(32) : tensor<{elements + 32}xi16> -> tensor<{elements}xbf16>
    %prefix = check.tensor.view %output_storage offset(0) : tensor<{elements + 32}xi16> -> tensor<16xi16>
    %suffix = check.tensor.view %output_storage offset({(elements + 16) * 2}) : tensor<{elements + 32}xi16> -> tensor<16xi16>
    check.compare<@gather_{name}, @reference_{name}>(%ids, %weight, %output) : (tensor<{rows}xi32>, tensor<{weights}xbf16>, tensor<{elements}xbf16>) -> () {{
      check.expect.bitwise actual(%output) expected(%output) : tensor<{elements}xbf16>
      check.expect.bitwise actual(%id_storage) expected(%id_storage) : tensor<{rows + 32}xi32>
      check.expect.bitwise actual(%weight_storage) expected(%weight_storage) : tensor<{weights + 32}xi16>
      check.expect.bitwise actual(%prefix) expected(%prefix) : tensor<16xi16>
      check.expect.bitwise actual(%suffix) expected(%suffix) : tensor<16xi16>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_gather
'''


def render():
    header = '''// Generated by generate_embedding_cases.py.
// Integer words exercise the BF16 bit patterns without numeric conversion.
// The independent reference copies complete selected rows as byte ranges.

amdgpu.target<gfx11-generic> @embedding_wave32 {subgroup_size = 32}

template.decl @embedding.lookup_bf16(%vocabulary: index, %width: index, %token_id: i32, %channel: index, %weight: buffer) -> (bf16)
'''
    return header + ''.join(render_case(case) for case in CASES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = Path(__file__).with_name('embedding_test.loom')
    content = render()
    if args.check:
        if path.read_text(encoding='utf-8') != content:
            raise SystemExit(f'{path} differs from its authored cases')
    else:
        path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
