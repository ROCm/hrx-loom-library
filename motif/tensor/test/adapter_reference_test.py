# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Check that the numerical envelope distinguishes incorrect bank selection."""

import struct
import unittest

from adapter_reference import MatrixReference


def unpack(values):
    return [
        struct.unpack("<f", struct.pack("<I", word << 16))[0]
        for (word,) in struct.iter_unpack("<H", values)
    ]


class ReferenceTest(unittest.TestCase):
    def test_selection_sensitivity(self):
        mutation_count = 0
        for exact in (False, True):
            reference = MatrixReference(exact)
            for name, selected in reference.selections:
                tolerance, _ = reference.allowances(selected)
                expected = unpack(reference.evaluate(selected))
                mutations = []
                for ordinal, (bank, strength) in enumerate(selected):
                    prefix, suffix = selected[:ordinal], selected[ordinal + 1 :]
                    if strength:
                        mutations.append(("drop", prefix + suffix))
                        mutations.append(
                            ("negate", prefix + ((bank, -strength),) + suffix)
                        )
                        mutations.append(
                            (
                                "wrong_bank",
                                prefix
                                + (((bank + 1) % reference.banks, strength),)
                                + suffix,
                            )
                        )
                    else:
                        mutations.append(
                            ("ignore_zero", prefix + ((bank, 1.0),) + suffix)
                        )
                for mutation, values in mutations:
                    with self.subTest(exact=exact, selection=name, mutation=mutation):
                        actual = unpack(reference.evaluate(values))
                        error = max(abs(a - e) for a, e in zip(actual, expected))
                        self.assertGreater(error, 2 * tolerance)
                        mutation_count += 1
        self.assertEqual(mutation_count, 138)


if __name__ == "__main__":
    unittest.main()
