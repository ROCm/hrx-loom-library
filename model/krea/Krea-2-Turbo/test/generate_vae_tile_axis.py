#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Derive decoder tile ownership by assigning pixels to physical centers."""

import argparse
import struct
from pathlib import Path


EXTENTS = (2, 16, 30, 32, 34, 48, 64, 66, 80, 96, 112, 128, 130, 256, 1022, 1024)


def decoder_support(phase):
    """Back-propagate an output position through the spatial decoder graph."""
    first = last = phase
    # The output convolution and six final-resolution residual convolutions.
    first -= 7
    last += 7
    for _ in range(3):
        # Resize convolves the nearest-expanded input, so undo its three-tap
        # neighborhood before mapping each endpoint to a source pixel.
        first = (first - 1) // 2
        last = (last + 1) // 2
        first -= 6
        last += 6
    # The post-attention middle residual precedes the upsampling stages.
    return first - 2, last + 2


def partition(extent, tile):
    if extent == tile:
        return 1, [(0, 0, extent)]
    stride = tile - 30
    origins = list(range(0, extent - tile + 1, stride))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    centers = [2 * origin + tile for origin in origins]
    owners = [
        min(range(len(origins)), key=lambda ordinal: (abs(2 * pixel + 1 - centers[ordinal]), -ordinal))
        for pixel in range(extent)
    ]
    intervals = []
    for ordinal, origin in enumerate(origins):
        pixels = [pixel for pixel, owner in enumerate(owners) if owner == ordinal]
        assert pixels and pixels == list(range(pixels[0], pixels[-1] + 1))
        first, end = pixels[0], pixels[-1] + 1
        assert origin <= first < end <= origin + tile
        if first:
            assert first - origin >= 15
        if end < extent:
            assert origin + tile - end >= 15
        intervals.append((origin, first, end))
    assert intervals[0][1] == 0 and intervals[-1][2] == extent
    assert all(left[2] == right[1] for left, right in zip(intervals, intervals[1:]))
    return stride, intervals


def render_case(extent, tile):
    stride, intervals = partition(extent, tile)
    values = [stride, len(intervals)] + [value for row in intervals for value in row]
    words = len(values)
    expected = struct.pack("<" + "i" * words, *values).hex()
    name = f"axis_{extent}_{tile}"
    return f'''
global.rodata.def @{name}_expected = align(4) bytes("{expected}")

kernel.def target(@tile_axis_wave32) @{name}_apply() {{
  %unit = index.constant 1 : index
  kernel.launch.config workgroups(%unit, %unit, %unit) workgroup_size(%unit, %unit, %unit) : index
}} launch(%output: buffer) {{
  %extent = index.constant {extent} : index
  %tile = index.constant {tile} : index
  %stride, %count = template.apply<@krea2.vae_tile_axis_plan>(%extent, %tile) : (index, index) -> (index, index)
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %view = buffer.view %output[%origin] : buffer -> view<{words}xi32>
  %stride_word = index.cast %stride : index to i32
  %count_word = index.cast %count : index to i32
  view.store %stride_word, %view[0] : i32, view<{words}xi32>
  view.store %count_word, %view[1] : i32, view<{words}xi32>
  scf.for %ordinal = [%c0 to %count step %c1] {{
    %start, %begin, %end = template.apply<@krea2.vae_tile_axis_interval>(%extent, %tile, %stride, %ordinal) : (index, index, index, index) -> (index, index, index)
    %position = index.madd %ordinal, %c3, %c2 : index
    %begin_position = index.add %position, %c1 : index
    %end_position = index.add %position, %c2 : index
    %start_word = index.cast %start : index to i32
    %begin_word = index.cast %begin : index to i32
    %end_word = index.cast %end : index to i32
    view.store %start_word, %view[%position] : i32, view<{words}xi32>
    view.store %begin_word, %view[%begin_position] : i32, view<{words}xi32>
    view.store %end_word, %view[%end_position] : i32, view<{words}xi32>
  }}
  kernel.return
}}

func.def @{name}_reference(%output: buffer) {{
  %expected = global.load @{name}_expected : buffer
  %origin = index.constant 0 : offset
  %bytes = index.constant {words * 4} : offset
  buffer.copy %expected[%origin], %output[%origin], %bytes
  func.return
}}

check.scenario public @{name} {{
  check.requires<hal.amdgpu.target> {{target = "gfx11-generic"}}
  check.trial[1](%trial: index, %entropy: check.entropy) {{
    %storage = check.generate.fill value(-77) : tensor<{words + 32}xi32>
    %output = check.tensor.view %storage offset(64) : tensor<{words + 32}xi32> -> tensor<{words}xi32>
    check.compare<@{name}_apply, @{name}_reference>(%output) : (tensor<{words}xi32>) -> () {{
      check.expect.bitwise actual(%storage) expected(%storage) : tensor<{words + 32}xi32>
    }}
  }}
  check.return
}}

check.benchmark<@{name}> @{name}_benchmark
'''


def render():
    supports = [decoder_support(phase) for phase in range(8)]
    assert min(first for first, _ in supports) == -15
    assert max(last for _, last in supports) == 15
    cases = set()
    for extent in EXTENTS:
        for tile in (extent, 32, 34, 48, 64, ((extent + 30) // 2 + 1) // 2 * 2, extent - 2):
            if 2 <= tile <= extent and (tile == extent or tile > 30):
                cases.add((extent, tile))
    header = '''// Generated by generate_vae_tile_axis.py.
// Whole-pixel nearest-center ownership is independent of the midpoint formulas.

amdgpu.target<gfx11-generic> @tile_axis_wave32 {subgroup_size = 32}

template.decl @krea2.vae_tile_axis_plan(%extent: index, %tile: index) -> (index, index)

template.decl @krea2.vae_tile_axis_interval(%extent: index, %tile: index, %stride: index, %ordinal: index) -> (index, index, index)
'''
    return header + "".join(render_case(*case) for case in sorted(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("vae_tile_axis.loom")
    content = render()
    if args.check:
        if path.read_text() != content:
            raise SystemExit(f"{path} differs from its authored oracle")
    else:
        path.write_text(content)


if __name__ == "__main__":
    main()
