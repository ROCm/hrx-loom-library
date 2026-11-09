# Embedding service regression

The source and archive tests run the built providers through the real
HTTP embedding service on an eight-column Strix Halo NPU context with explicit
fixed resident backing. The caller
supplies a `loom-fetch` installation containing the original checkpoint through
`LOOM_MODEL_ASSETS`. Each server run exercises native parameter conversion
during loading before the HTTP checks begin. The small
publisher reference, checker, card, acquisition lock, and providers are Bazel
inputs; full model weights are not. External assets are trusted to match the
lock. Changing that lock invalidates cached results without rescanning the
checkpoint during every test.

Three original synthetic sentences cover English and French tokenization.
`reference.json` records their exact publisher token IDs and the EOS convention.
Explicit 129-, 513-, and 32,768-token inputs exercise panel and context
boundaries. `publisher.f32` contains six independently computed, normalized 1,024-element
vectors in little-endian F32 order. The reference uses the pinned publisher
checkpoint through Transformers `Qwen3Model`, F32 attention, one sequence
at a time, and the final real token's hidden state. The full-context case uses
PyTorch's CPU Flash SDPA implementation to bound reference memory; its
agreement with eager attention on shorter inputs is recorded in the reference.
The other cases use eager attention. Text cases append EOS;
explicit token inputs retain their authored contents. It does not use the Loom
panel packing, matrix operations, or normalization implementation. Dependency
versions, the checkpoint revision, and the reference digest are recorded beside
the data. Long explicit inputs use a lossless prefix/repeated-pattern/suffix
representation with a digest of the expanded little-endian token IDs.

Reproduce those publisher vectors from the HF cache without acquiring weights:

```sh
uv run --offline model/Qwen/Qwen3-Embedding-0.6B/test/generate_reference.py
```

The pinned Python environment must also be cached for `uv run --offline`.
Omit uv's offline option to install its authoring dependencies; the script
still uses only cached checkpoint files unless `--download` is explicitly
supplied. Regeneration changes publisher vectors and tokenization, while leaving
acceptance limits unchanged.

The quantized native output must have cosine similarity of at least 0.995 with
each publisher vector. Unit-norm and prefix-renormalization errors are bounded
by 2e-6. These limits are fixed independently of the candidate run. Native
vector digests are recorded for comparing source and archive results, while
identical request replays must remain bit-exact.

The checker compares single requests, explicit token IDs, reordered cohorts,
concurrent requests, float and base64 encodings, 64- and 257-dimensional
prefixes, and a thirty-three-sequence cohort. A 129-token sequence crosses the panel
boundary alone and in both positions of a mixed-length cohort. Each placement
is checked against the independent publisher vector because placement changes
the attention reduction partitions. Repeating the same mixed cohort checks
exact state reuse after another request. A logical request containing a
32,768-token sequence, a 513-token sequence, and a short sequence requires two
cohorts. A late short peer must finish before that logical request, whose
output order, numerical agreement, and total token usage are checked together.
An accepted continuing request is then disconnected; cancellation must retire
and a later 513-token replay must reproduce its output. Long text and its
explicit token IDs must agree exactly. Invalid IDs, dimensions, and sequences
longer than 32,768 tokens are rejected. Successful results are written only
after the owned server exits cleanly.

Both full-model targets are explicit `manual` stress tests. Full-context
attention makes them unsuitable for the ordinary short CI suite.

The small native attention regression exercises the production recurrence
leaves over 512 equal-score key blocks. Its independent reference requires an
exact sum of 32,768, a unit rescaling factor, and the exact final reciprocal.
This catches exponential approximations that slightly discount retained state
even when the maximum is unchanged; such an error compounds across a long
context. The fixture compiles the private production functions into its test
module without adding model exports.

The broader qualification also replays 2,624 frozen retrieval inputs: a Quora
subset containing 256 queries and 2,048 documents, plus 320 SciFact inputs.
Those corpora are separate from these compact in-repository tests and evaluate
retrieval quality against the frozen publisher ranking. A pass of these compact
service regressions does not establish the broader retrieval qualification.

Replacement implementations preserve or improve aggregate nDCG@10 and
recall@10 against the qualified baseline, while retaining the 0.995 publisher
cosine requirement for every embedding. Individual query rankings can change
within that numerical envelope. Such changes remain qualification diagnostics;
they are not an exact-ranking requirement on the quantized implementation.
