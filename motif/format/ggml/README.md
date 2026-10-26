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
Both formats decode four-code carrier words directly; their sixteen-code
readers compose that same interpretation over four retained words. Matrix
staging can therefore retain and reuse a high-plane word independently of
low-word loads. These forms read only the requested physical packets and
leave coefficient application to the contraction, preserving reuse across
activation rows.

Q5_1 loads its 24-byte record into retained nibble pairs, one high-bit word,
and independent F32 scale and minimum. A separate half-block decoder consumes
those carriers, including carriers already staged in shared memory. Both
16-value halves can reuse the original header across several activation
rows; consumers own the affine arithmetic and its rounding boundaries.

Q6_K offers signed I8 carrier unpacking, unscaled four-code F32 packets,
retained pairs of F32 coefficients, and a buffered four-value F16 reader.
A coefficient pair covers one 32-value group and can stay resident across
its eight code packets. Consumers own metadata staging and coefficient
application; the F16 reader owns its single rounding boundary. Buffer readers
handle the alternating word alignment across 210-byte blocks. Small record
tests check codes, coefficients, and rounded values independently against a
scalar byte-level oracle, including row padding and output guards.

Q8_1 x4 readers expose a full 32-value block, a
16-value half-block, or one four-value word. Full and half-block readers
return the original block scale and scaled sum; consumers own any affine
minimum correction, including how paired half-blocks share that sum.

Q8_0 exposes exact signed I8 codes with a separate F32 scale. Full 32-code
and half-block readers accept independent scale and code byte origins, so
raw records and prepared scale/code planes use the same representation
without repacking. A four-code word reader addresses raw 34-byte records
and handles their alternating word alignment. Consumers retain packets
across activation rows and decide where to round scaled values; cooperative
matrix staging owns its separate scale publication schedule.

IQ4_NL exposes one complete 18-byte block as 32 signed I8 codebook values
and a separate F32 scale. The first 16 values come from the low nibbles;
the remaining values come from the high nibbles. Consumers retain these
exact codes across activation formats and choose when to apply the scale.
The reader accepts a byte origin without imposing a row or tensor layout.

## References

- [GGUF format specification](https://github.com/ggml-org/ggml/blob/30bf8685ed4eb0a47f2b06229543327749904150/docs/gguf.md)
