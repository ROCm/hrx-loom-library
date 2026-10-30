#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Records normalization output from the pinned Pocket TTS 3.3.0 reference.

The Bazel input pins the upstream text module by revision and digest. Only its
preparation functions and constants are evaluated; tokenizer-dependent sentence
chunking is outside this oracle. Normalization runs unchanged on CPython 3.12.
"""

import argparse
import ast
import re
import struct
import unicodedata
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    if unicodedata.unidata_version != "15.0.0":
        raise SystemExit("The oracle requires Unicode 15.0.0 (CPython 3.12)")
    functions = {"prepare_text_prompt", "_ensure_terminal_punctuation"}
    constants = {"_TERMINAL_PUNCTUATION", "_WEAK_PUNCTUATION", "_CLOSERS"}
    source = ast.parse(args.upstream_source.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in source.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
    ]
    namespace = {"re": re}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(args.upstream_source),
            "exec",
        ),
        namespace,
    )
    prepare_text_prompt = namespace["prepare_text_prompt"]
    examples = [
        "",
        " \t\r\n",
        "hello",
        "HELLO",
        "one two three four",
        "one two three four five",
        "a\x00b",
        "a\x1cb\x85c\u2009d\u3000e",
        "你好世界",
        "𝔞 message",
        "ǳ message",
        "hello\n\rworld",
        "hello\r\nworld",
        "hello\t\tworld",
        '"',
        " ) ] ” ’ » ",
        " ; - , : ",
        " , ; : ) ] ",
        "line\n\n\n\n\nends",
    ]
    examples.extend("a" + " " * count + "b" for count in range(1, 10))
    for point in range(0x110000):
        character = chr(point)
        if character.isspace():
            examples.append(character + "hi" + character + "there" + character)
        if len(character.upper()) > 1:
            examples.append(character + "eta")
    for ending in ".!?… ,;:-–—":
        for closer in ("", '"', "'", "”", "’", ")", "]", "»", " ” ]"):
            examples.append("hi" + ending + closer)
            examples.append("hi" + ending + " " + closer)
    inputs = bytearray()
    outputs = bytearray()
    rows = []
    for example in dict.fromkeys(examples):
        if example.strip():
            normalized, tail = prepare_text_prompt(example, False, False)
            tail += 2
        else:
            # Pure normalization exposes emptiness to request admission.
            normalized, tail = "", 5
        input_bytes = example.encode()
        output_bytes = normalized.encode()
        rows.append(
            (len(inputs), len(input_bytes), len(outputs), len(output_bytes), tail)
        )
        inputs.extend(input_bytes)
        outputs.extend(output_bytes)
    records = b"".join(struct.pack("<5I", *row) for row in rows)
    for name, data, dtype, shape in (
        ("records", records, "<i4", (len(rows), 5)),
        ("inputs", inputs, "|i1", (len(inputs),)),
        ("outputs", outputs, "|i1", (len(outputs),)),
    ):
        header = str(
            {"descr": dtype, "fortran_order": False, "shape": shape}
        ).encode("ascii")
        header += b" " * ((-len(header) - 11) % 64) + b"\n"
        (args.output_directory / f"text_normalization_{name}.npy").write_bytes(
            b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data
        )


if __name__ == "__main__":
    main()
