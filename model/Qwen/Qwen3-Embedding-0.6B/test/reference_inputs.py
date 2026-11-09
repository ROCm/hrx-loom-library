# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Expand losslessly stored publisher token fixtures without model code."""

import hashlib
import struct


def pattern_tokens(case: dict) -> list[int]:
    pattern = case["token_pattern"]
    count = case["length"] - len(pattern["suffix"])
    remaining = count - len(pattern["prefix"])
    copies = (remaining + len(pattern["repeat"]) - 1) // len(pattern["repeat"])
    tokens = (pattern["prefix"] + pattern["repeat"] * copies)[:count]
    tokens += pattern["suffix"]
    encoded = struct.pack(f"<{len(tokens)}i", *tokens)
    assert len(tokens) == case["length"]
    assert hashlib.sha256(encoded).hexdigest() == case["token_sha256"]
    return tokens
