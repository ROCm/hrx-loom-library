# Text correctness and benchmark workloads

`unicode_test.loom` checks every changed uppercase codepoint and every whitespace
and digit member, their immediate neighbors, and UTF-8 width boundaries. Its
3,293 rows come independently from CPython 3.12's Unicode 15.0 implementation.
Each row checks the three uppercase results, both properties, and forward and
backward UTF-8 round trips. The generated oracle is separate from the UCD
parser that supplies the production tables.

`utf8_test.loom` compares literal encodings at both ends of all four UTF-8 widths,
including NUL, against the produced bytes and decoded scalars. Trim cases
cover empty input, all-whitespace input, mixed-width leading/trailing spaces,
and preserved interior whitespace. These explicit byte checks prevent a pair
of matching encoder/decoder mistakes from satisfying only a round trip.

The VM execution profile exercises the same functions used by model request
programs. Benchmarks select the low, middle, and high Unicode property ranges,
ASCII and four-byte encoding, and mixed whitespace trimming. They define
repeatable workloads without asserting a performance threshold.

```sh
bazel test --config=asan //motif/text/test:all
python3.12 -B motif/text/test/generate_unicode_oracle.py
```

Regeneration requires Unicode 15.0.0 and fails on a different Python UCD
version. A data-version change updates the production tables, oracle, and
documented semantic version together.
