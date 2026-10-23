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

Q8_0 contracts raw signed-byte records with Q8_1 x4 activations. Eight lanes
own one 32-value block, so a subgroup advances four blocks on wave32 or eight
on wave64. One subgroup reduces an output channel, masking absent blocks and
tail channels before reads. The launch can dispatch fewer token rows than
the supplied buffer capacity without touching the remaining rows.

Cohort forms accept a row-bank extent fixed by model composition before JIT.
Q5 raw/channel8 schedules decode each weight packet before applying it to all
rows. Q6 Q8 schedules support 16 or 32 lanes per block, and Q6 F32 schedules
share activation packets across paired output channels. Each retains separate
row accumulators before advancing K. Their private model-shaped leaves and
independent CPU references live in
[`kernel/ggml/linear/test`](../../../kernel/ggml/linear/test/), where named row
specializations execute both subgroup widths. The motifs carry no global
configuration or concrete target records.

These are reusable Loom schedules, qualified for correctness through the
[`linear` kernels](../../../kernel/ggml/linear/). The native tests establish
packed-byte semantics, addressing, and numerical results; they do not establish
a throughput comparison against another implementation.
