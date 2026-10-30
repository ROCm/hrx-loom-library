# Text correctness and benchmark workloads

`unicode_test.loom` checks every changed uppercase codepoint and every whitespace
and digit member, their immediate neighbors, and UTF-8 width boundaries. Its
3,293 rows come independently from CPython 3.12's Unicode 15.0 implementation.
Each row checks the three uppercase results, both properties, and forward and
backward UTF-8 round trips. The captured oracle is independent of the shared
runtime implementation; these properties agree across Unicode 15.0 and 15.1.

`utf8_test.loom` compares literal encodings at both ends of all four UTF-8 widths,
including NUL, against the produced bytes and decoded scalars. Trim cases
cover empty input, all-whitespace input, mixed-width leading/trailing spaces,
and preserved interior whitespace. These explicit byte checks prevent a pair
of matching encoder/decoder mistakes from satisfying only a round trip.

The VM execution profile exercises the same functions used by model request
programs. Benchmarks cover Unicode properties, ASCII and four-byte encoding,
and mixed whitespace trimming. They define repeatable workloads without
asserting a performance threshold.

The [development guide](../../../docs/development.md#build-and-check) owns
normal test and benchmark commands.
