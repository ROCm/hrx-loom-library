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

`unicode.loom` calls the Loom runtime's `string` module through its `codepoint.*`
operations for full uppercase, whitespace membership, and digit membership.
Models carry no Unicode tables. The runtime supplies
[Unicode 15.1](https://www.unicode.org/versions/Unicode15.1.0/)
through native imports in both serving and standalone VM execution.
Uppercase returns up to three codepoints, with zero in unused expansion slots.
Whitespace follows Python's category Zs or bidi classes WS, B, and S, including
its additional ASCII controls; digits include Decimal and Digit numeric types.
ASCII whitespace and digits use direct predicates.

Each source has a separate `loom_motif_library`; `utf8` declares its dependency
on `unicode`. Model-specific sentence splitting and normalization remain in
the model that owns those policies.

The [test package](test/) owns independent CPython observations and UTF-8 byte
checks; [LICENSE.unicode](LICENSE.unicode) covers Unicode-derived oracle data.
[Component authoring](../../docs/component-authoring.md) describes the common
build and test workflow.
