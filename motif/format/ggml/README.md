# GGML tensor format motifs

These motifs describe GGML in-memory encodings independently of a file
container or launch ABI. GGUF carries persistent weight encodings; backends
also use transient activation encodings. Loaders and command programs resolve
containers into buffers and representation facts before calling this layer.

The reusable boundary separates packed codes from their scales, affine
metadata, and physical addressing. A consumer can retain a header or code
packet across multiple activation rows, stage it in shared memory, and choose
where to apply coefficients and rounding. Buffer readers and retained-carrier
readers share the format interpretation without imposing a contraction schedule
or an expanded-weight allocation.

Each source owns its byte layout, alignment, element interpretation, and
upstream provenance. Consumers declare each format provider whose public
symbols they use; an aggregate dependency does not re-export its symbols.
Word carriers in the halfword-aligned Q5_1, Q6_K, and Q8_0 formats use
`align(2)` views. A carrier's `i32` type describes its bits; it does not make
packed records four-byte aligned. Q4_K/Q5_K readers require naturally aligned
word origins, and the Q8_1 x4 representation contains aligned `i32` words.
The [format checks](test/) use independent scalar-byte interpretations and
exercise both lane-dependent and uniform packed addresses.
[Component authoring](../../../docs/component-authoring.md) covers declarations
and the shared build/test workflow.

The [GGUF specification](https://github.com/ggml-org/ggml/blob/30bf8685ed4eb0a47f2b06229543327749904150/docs/gguf.md)
describes the container boundary.
