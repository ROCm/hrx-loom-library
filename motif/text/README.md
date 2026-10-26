# Text transforms

These motifs implement text semantics shared by model preparation and request
programs. They consume Unicode scalars and valid UTF-8; request parsing or a
producing tokenizer establishes validity once, before calling them. They make
no model, target, locale, or tokenization choice.

`utf8.loom` traverses and writes scalars and trims whitespace by returning byte
bounds into its input. Forward traversal requires a position before the end;
backward traversal requires a nonzero, codepoint-aligned end. Writing requires
four bytes of available storage. Trimming an empty string returns `(0, 0)`;
trimming an all-whitespace string returns `(length, length)`.

`unicode.loom` provides full uppercase, whitespace membership, and digit
membership using [Unicode 15.0.0](https://www.unicode.org/versions/Unicode15.0.0/).
Uppercase returns up to three codepoints. Whitespace follows Python's category
Zs or bidi classes WS, B, and S; digits include Decimal and Digit numeric types.
The tables contain 1,525 uppercase rows, 10 whitespace ranges, and 83 digit
ranges in 25,144 immutable bytes. Binary search serves non-ASCII properties;
ASCII whitespace and digits use direct predicates. The data's notice is
[LICENSE.unicode](LICENSE.unicode).

Each source has a separate `loom_motif_library`; `utf8` declares its dependency
on `unicode`. Model-specific sentence splitting and normalization remain in
the model that owns those policies.

`generate_unicode.py` verifies the canonical UCD files' pinned hashes and
refreshes only the data region, preserving the handwritten interpreter:

```sh
python3 -B motif/text/generate_unicode.py /path/to/ucd/15.0.0 --check
```

The independent CPython oracle and explicit UTF-8 byte/trim checks run through
the public Loom runner:

```sh
bazel test --config=asan //motif/text/...
```

The [test documentation](test/README.md) describes coverage and regeneration.
