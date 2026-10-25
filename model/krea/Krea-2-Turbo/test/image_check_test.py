# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Independent filtered rows check the image regression's pixel decoder."""

import struct
import unittest
import zlib

from image_check import decode_png


def chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def png(rows, *, color=2):
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 5, 8, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


# One row per PNG filter. Rows zero through three reconstruct identical pixels;
# the last row changes every channel, including both byte-wrapping directions.
ROWS = bytes(
    [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        1,
        10,
        20,
        30,
        30,
        30,
        30,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        5,
        10,
        15,
        15,
        15,
        15,
        4,
        251,
        5,
        15,
        210,
        216,
        10,
    ]
)


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
        with self.assertRaisesRegex(AssertionError, "RGB8"):
            decode_png(png(ROWS, color=6))

    def test_unknown_filter(self):
        with self.assertRaisesRegex(AssertionError, "filter"):
            decode_png(png(bytes([5]) + ROWS[1:]))

    def test_truncated(self):
        with self.assertRaisesRegex(AssertionError, "truncated"):
            decode_png(png(ROWS)[:-1])


if __name__ == "__main__":
    unittest.main()
