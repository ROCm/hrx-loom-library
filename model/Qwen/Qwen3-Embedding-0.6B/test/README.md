# Embedding service regression

The source, archive, and mixed tests run the built providers through the real
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
`publisher.f32` contains three independently computed, normalized 1,024-element
vectors in little-endian F32 order. The reference uses the pinned publisher
checkpoint through Transformers `Qwen3Model`, eager F32 attention, one sequence
at a time, and the final EOS hidden state. It does not use the Loom panel
packing, matrix operations, or normalization implementation. Dependency versions,
the checkpoint revision, and the reference digest are recorded beside the data.

Reproduce those publisher vectors from the HF cache without acquiring weights:

```sh
uv run --offline model/Qwen/Qwen3-Embedding-0.6B/test/generate_reference.py
```

The pinned Python environment must also be cached for `uv run --offline`.
Omit uv's offline option to install its authoring dependencies; the script
still uses only cached checkpoint files unless `--download` is explicitly
supplied. Regeneration changes publisher vectors and tokenization, while leaving
native regression identities and acceptance limits under separate review.

The quantized native output must have cosine similarity of at least 0.995 with
each publisher vector. Unit-norm and prefix-renormalization errors are bounded
by 2e-6. These limits are fixed independently of the candidate run. Exact native
vector digests additionally detect any change from the qualified representation;
an intended arithmetic change therefore needs a new accuracy qualification even
when it satisfies the cosine limit.

The checker compares single requests, explicit token IDs, reordered cohorts,
concurrent requests, float and base64 encodings, 64-dimensional prefixes, and
a full sixteen-sequence cohort. It checks token accounting and rejects invalid
IDs, dimensions, sequence lengths, and rounded cohort overflow. Successful
results are written only after the owned server exits cleanly.

The broader qualification also replays a frozen retrieval subset containing
256 queries and 2,048 documents. That corpus is separate from these compact
in-repository tests. The source port reproduces every retained native vector;
the frozen independent publisher and native nDCG@10 scores are approximately
0.95074 and 0.94833. This is a bounded subset comparison, not a full benchmark
score or a performance measurement.
