# GGML Tensor Format Motifs

This package family implements GGML in-memory tensor encodings as reusable Loom
functions and templates. GGUF is the container and metadata format: it carries
persistent tensor encodings used by model weights. GGML backends also introduce
transient compute encodings for activations. Both are representation motifs;
neither inherits a GGML launch ABI. Host loaders and command programs own GGUF
parsing, and these motifs begin at the resulting buffer and preserved
representation facts.

Format sources own their physical byte layout, element interpretation, block
addressing, and links to the specification or implementation snapshot from
which those contracts were derived. Reusable accessors are `func.def`
operations; reusable encoders and decoders that require specialization are
`template.def` operations. Neither form chooses grid dimensions or exposes a
kernel launch ABI.

GGML-compatible kernels, native GEMM and attention kernels, and model-level
fusions all link these motifs directly. This keeps physical encodings and
transient backend packings from being trapped inside one operator catalog.
Consumers name each format library whose public symbols they use. Loom keeps
those modules distinct through linking and requires direct dependencies, so a
source-less aggregate cannot masquerade as a re-exporting archive.

Q4_K and Q5_K share one affine metadata interpretation. A retained 16-byte
header can supply coefficients for several groups, while buffer accessors
load a selected half-group from an explicit row origin and block ordinal.
Q4_K also exposes the four-code packet used by matrix staging. These forms
read only the requested physical packets and leave coefficient application
to the contraction, preserving reuse across activation rows.

Q6_K offers signed carrier unpacking and a buffered four-value F16 reader
for matrix staging. It handles the format's alternating word alignment
across 210-byte blocks. Q8_1 x4 readers expose a full 32-value block, a
16-value half-block, or one four-value word. Full and half-block readers
return the original block scale and scaled sum; consumers own any affine
minimum correction, including how paired half-blocks share that sum.

Q8_0 exposes one four-code word and its separate F32 scale from a raw
34-byte record. Its word reader preserves signed I8 codes and handles the
alternating word alignment between records. Integer dot products can use
the codes directly; matrix staging decides where to round scaled values.

## References

- [GGUF format specification](https://github.com/ggml-org/ggml/blob/30bf8685ed4eb0a47f2b06229543327749904150/docs/gguf.md)
