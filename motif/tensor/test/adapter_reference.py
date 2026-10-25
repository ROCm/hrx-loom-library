# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Independent adapter contractions and propagation of native arithmetic error."""

from functools import cache
import struct


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    word = (word + 0x7FFF + ((word >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", word))[0]


def packed(values):
    words = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
    assert all(word & 65535 == 0 for word in words)
    return struct.pack("<" + "H" * len(words), *(word >> 16 for word in words))


def flatten(rows):
    return [value for row in rows for value in row]


def rodata(name, value):
    return f'global.rodata.def @{name} = align(16) bytes("{value.hex()}")\n'


class MatrixReference:
    """Synthetic original operands, mathematical results, and input-derived bounds."""

    def __init__(self, exact=False, options=None):
        options = options or {}
        # Whether contributions stay exactly representable at every boundary.
        self.exact = exact
        # Number of activation rows, including complete sixteen-row tails.
        self.rows = options.get("rows", 80)
        # Reduction width of the down projection.
        self.inputs = options.get("inputs", 64)
        # Channel count of the up projection.
        self.outputs = options.get("outputs", 64)
        # Reduction rank shared by all selected banks.
        self.rank = 32
        # Number of distinct physical factor banks in the input allocation.
        self.banks = options.get("banks", 7)
        # Physical base/output layout; logical reference indexing is unchanged.
        self.column_major = options.get("column_major", False)
        # Whether the production caller updates its exact base allocation.
        self.alias = options.get("alias", False)
        # Byte spacing between physical banks, with optional untouched padding.
        self.stride = self.rank * max(self.inputs, self.outputs) * 2 + (
            16 if options else 0
        )
        # Authored runtime selections preserve ordering and independent strengths.
        self.selections = (
            ("empty", ()),
            ("single", ((3, 0.5),)),
            ("pair", ((1, -0.75), (5, 1.0))),
            ("odd", ((0, 1.0), (1, -0.5), (2, 0.75))),
            ("shared", ((1, 0.25), (3, -1.0))),
            ("even", ((0, 1.0), (2, -0.5), (4, 0.25), (6, 0.75))),
            ("seven", tuple((i, (1.0, -0.5, 0.25)[i % 3]) for i in range(self.banks))),
            ("reordered", ((2, 0.75), (0, 1.0), (1, -0.5))),
            ("zero_strength", ((1, 0.0), (6, 0.0), (4, 0.0))),
        )
        if options.get("duplicates"):
            self.selections += (
                ("duplicate_bank", ((1, 0.25), (3, -1.0), (1, 0.75))),
                ("cancel_bank", ((1, 1.0), (1, -1.0))),
                ("fractional", ((1, 0.3), (4, 0.7), (6, -0.625))),
            )
        if "selections" in options:
            self.selections = tuple(
                row for row in self.selections if row[0] in options["selections"]
            )
        self.selections = tuple(
            (
                name,
                tuple(
                    (bank, f32(abs(scale) if exact else scale))
                    for bank, scale in selected
                ),
            )
            for name, selected in self.selections
        )
        # Row-major activations, independent of native fragment distribution.
        self.input_values = [
            [((row * 7 + k * 3) % 15 - 7) / 8 for k in range(self.inputs)]
            for row in range(self.rows)
        ]
        # Logical down factors indexed by physical bank, rank, and input column.
        self.down = [
            [
                [
                    (bank * 11 + rank * 5 + k * 3) % 7 / 8 - 3 / 8
                    for k in range(self.inputs)
                ]
                for rank in range(self.rank)
            ]
            for bank in range(self.banks)
        ]
        # Logical up factors indexed by physical bank, output channel, and rank.
        self.up = [
            [
                [
                    (bank * 5 + channel * 3 + rank * 7) % 11 / 8 - 5 / 8
                    for rank in range(self.rank)
                ]
                for channel in range(self.outputs)
            ]
            for bank in range(self.banks)
        ]
        # Base values are included only after all selected up contributions.
        self.base = [
            [((row * 3 + channel * 5) % 11 - 5) / 4 for channel in range(self.outputs)]
            for row in range(self.rows)
        ]
        if exact:
            self.input_values = [
                [float(1 + row % 2) for _ in range(self.inputs)]
                for row in range(self.rows)
            ]
            self.down = [
                [
                    [(bank + 1) / 8 if k == rank else 0.0 for k in range(self.inputs)]
                    for rank in range(self.rank)
                ]
                for bank in range(self.banks)
            ]
            self.up = [
                [
                    [
                        1 / 8 if rank == channel % self.rank else 0.0
                        for rank in range(self.rank)
                    ]
                    for channel in range(self.outputs)
                ]
                for _ in range(self.banks)
            ]
            self.base = [
                [(1 + channel % 2) / 8 for channel in range(self.outputs)]
                for _ in range(self.rows)
            ]
        # Immutable serialized inputs include guard bytes between factor banks.
        self.shared = {
            "input": packed(flatten(self.input_values)),
            "a": self.factor_storage(self.down),
            "b": self.factor_storage(self.up),
            "base": packed(self.output_order(self.base)),
        }
        # Cache only the synthetic operand periods, never a device observation.
        self.contraction = cache(self._contraction)

    def output_order(self, rows):
        if self.column_major:
            return [
                rows[row][column]
                for column in range(self.outputs)
                for row in range(self.rows)
            ]
        return flatten(rows)

    def factor_storage(self, banks):
        result = bytearray()
        for bank in banks:
            values = packed(flatten(bank))
            result.extend(values)
            result.extend(bytes([77]) * (self.stride - len(values)))
        return bytes(result)

    def _contraction(self, bank, row, rank):
        products = [
            x * y for x, y in zip(self.input_values[row], self.down[bank][rank])
        ]
        return sum(products), sum(abs(value) for value in products)

    def down_contraction(self, bank, row, rank):
        return self.contraction(
            bank, row % (2 if self.exact else 15), rank if self.exact else rank % 7
        )

    def evaluate(self, selected, return_low=False):
        low = []
        for bank, scale in selected:
            rows = []
            for row in range(self.rows):
                values = []
                for rank in range(self.rank):
                    exact = self.down_contraction(bank, row, rank)[0]
                    assert f32(exact) == exact
                    values.append(bf16(f32(exact * scale)))
                rows.append(values)
            low.append(rows)
        if return_low:
            return packed(flatten([row for bank in low for row in bank]))

        @cache
        def output(row, channel):
            total = 0.0
            for ordinal, (bank, _) in enumerate(selected):
                for rank in range(self.rank):
                    total += low[ordinal][row][rank] * self.up[bank][channel][rank]
                    assert f32(total) == total
            value = total + self.base[row][channel]
            assert f32(value) == value
            return bf16(value)

        expected = [
            [
                output(row, channel % (32 if self.exact else 11))
                for channel in range(self.outputs)
            ]
            for row in range(self.rows)
        ]
        return packed(self.output_order(expected))

    def allowances(self, selected):
        """Propagate sum(abs(products))*2^-20 through each BF16 boundary.

        Rounded endpoints include signs, runtime strengths, rank scratch,
        and the final base addition. Device results are never inputs.
        """
        low = []
        low_tolerance = 0.0
        for bank, scale in selected:
            bank_rows = []
            for row in range(self.rows):
                values = []
                for rank in range(self.rank):
                    exact, norm = self.down_contraction(bank, row, rank)
                    error = norm * 2**-20
                    endpoints = sorted(
                        (
                            bf16(f32((exact - error) * scale)),
                            bf16(f32((exact + error) * scale)),
                        )
                    )
                    nominal = bf16(exact * scale)
                    low_tolerance = max(
                        low_tolerance, *(abs(value - nominal) for value in endpoints)
                    )
                    values.append((nominal, *endpoints))
                bank_rows.append(values)
            low.append(bank_rows)
        tolerance = 0.0
        for row in range(self.rows):
            for channel in range(min(self.outputs, 32 if self.exact else 11)):
                nominal, lower, upper, norm = 0.0, 0.0, 0.0, 0.0
                for ordinal, (bank, _) in enumerate(selected):
                    for rank in range(self.rank):
                        center, lo, hi = low[ordinal][row][rank]
                        weight = self.up[bank][channel][rank]
                        contributions = sorted((lo * weight, hi * weight))
                        nominal += center * weight
                        lower += contributions[0]
                        upper += contributions[1]
                        norm += max(abs(value) for value in contributions)
                error = norm * 2**-20
                nominal = bf16(nominal + self.base[row][channel])
                endpoints = (
                    bf16(f32(lower - error) + self.base[row][channel]),
                    bf16(f32(upper + error) + self.base[row][channel]),
                )
                tolerance = max(
                    tolerance, *(abs(value - nominal) for value in endpoints)
                )
        if self.exact:
            assert tolerance == 0.0 and low_tolerance == 0.0
        return tolerance, low_tolerance
