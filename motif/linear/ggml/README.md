# GGML block contractions

Q4_K and Q5_K use the same affine contraction: a signed integer dot of the
nonnegative weight codes with Q8_1 activation codes, followed by the two scales
and the stored activation-sum correction. The format accessors remain in
[`motif/format/ggml`](../../format/ggml/); this package owns how their packets
participate in a complete projection.

Sixteen lanes own one 256-value weight block. A 32-lane subgroup visits two
blocks per iteration, and a 64-lane subgroup visits four. Two lanes split each
32-value affine group, each applying half its minimum correction. Incomplete
iterations mask absent blocks before reading their inputs. The caller supplies
256-thread groups and the exact launch grid described by each source.

Q5 supports raw channel-major blocks and groups of eight interleaved channels.
Its four-row path reads and decodes each weight packet once, retains four dot
accumulators, and publishes four rows. The single-row and four-row schedules
share the physical format and affine arithmetic. Model-specific matrix and
cohort schedules can compose the same format contracts independently.

Q6_K has three schedules with different reuse boundaries. The F32 activation
path shares four activation packets across two output channels. The Q8_1 path
assigns one output channel to each 32-lane cluster, retaining low/high packed
weight words across two dot4 packets. The 64-lane `block4` path instead advances
across four weight blocks per iteration. Each has a four-row form with separate
accumulators and common immutable weight addresses. Block-level primitives
remain available to model-specific cohort and output-reduction schedules.
All Q6 forms use the signed carrier decoder without first rounding weights to
F16 or materializing an expanded tensor.

These are reusable Loom schedules, qualified for correctness through the
[`linear` kernels](../../../kernel/ggml/linear/). The native tests establish
packed-byte semantics, addressing, and numerical results; they do not establish
a throughput comparison against another implementation.
