# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Independent filtered rows and inputs check the image regression's helpers."""

import base64
import struct
import unittest
import zlib

from image_check import decode_png, pattern, prompt


def chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def png(rows, *, color=6):
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 5, 8, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


# One row per PNG filter. Rows zero through three reconstruct identical pixels;
# the last row changes every channel, including both byte-wrapping directions.
ROWS = bytes([
    0, 10, 20, 30, 255, 40, 50, 60, 255,
    1, 10, 20, 30, 255, 30, 30, 30, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0,
    3, 5, 10, 15, 128, 15, 15, 15, 0,
    4, 251, 5, 15, 0, 210, 216, 10, 0,
])


class ImageDecoderTest(unittest.TestCase):
    def test_all_filters(self):
        shape, pixels = decode_png(png(ROWS))
        self.assertEqual(shape, (2, 5))
        self.assertEqual(
            pixels,
            bytes([10, 20, 30, 40, 50, 60]) * 4 + bytes([5, 25, 45, 250, 10, 70]),
        )

    def test_crc_mismatch(self):
        broken = bytearray(png(ROWS))
        broken[-1] ^= 1
        with self.assertRaisesRegex(AssertionError, "checksum"):
            decode_png(broken)

    def test_wrong_geometry(self):
        with self.assertRaisesRegex(AssertionError, "dimensions"):
            decode_png(png(ROWS[:-1]))

    def test_unsupported_pixel_format(self):
        with self.assertRaisesRegex(AssertionError, "RGBA8"):
            decode_png(png(ROWS, color=2))

    def test_transparent_output(self):
        rows = bytearray(ROWS)
        rows[4] = 0
        with self.assertRaisesRegex(AssertionError, "non-opaque"):
            decode_png(png(rows))

    def test_unknown_filter(self):
        with self.assertRaisesRegex(AssertionError, "filter"):
            decode_png(png(bytes([5]) + ROWS[1:]))

    def test_truncated(self):
        with self.assertRaisesRegex(AssertionError, "truncated"):
            decode_png(png(ROWS)[:-1])


class InputTest(unittest.TestCase):
    def test_pattern_corners(self):
        pixels = pattern(64, 4)
        self.assertEqual(len(pixels), 64 * 4 * 3)
        # Red ramps across columns and green down rows.
        self.assertEqual(pixels[0:2], bytes([0, 0]))
        self.assertEqual(pixels[63 * 3 : 63 * 3 + 2], bytes([255, 0]))
        last = (3 * 64) * 3
        self.assertEqual(pixels[last : last + 2], bytes([0, 255]))

    def test_prompt_is_quarter_size(self):
        raw, text = prompt((768, 400))
        self.assertEqual(len(raw), 192 * 100 * 3)
        self.assertEqual(base64.b64decode(text, validate=True), raw)
        self.assertEqual(len(text), 192 * 100 * 4)

    def test_urlsafe_prompt(self):
        raw, text = prompt((256, 16), urlsafe=True)
        self.assertEqual(base64.urlsafe_b64decode(text), raw)
        self.assertFalse(set(text) & set("+/"))


if __name__ == "__main__":
    unittest.main()
