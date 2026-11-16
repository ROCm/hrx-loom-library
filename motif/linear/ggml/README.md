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
These scalar and integer-dot forms use the signed carrier decoder without
first rounding weights to F16 or materializing an expanded tensor.

The Q6_K/Q8_1 x4 panel template retains a row-by-panel accumulator bank around
that same block decoder. Immutable weight packets serve the row cohort, and
activation packets serve multiple output panels. The caller specializes a
cohort of 1–32 rows, provides 256-workitem groups, and chooses their population.
The motif derives its bounded accumulator bank and useful grid from that row
count, so launch geometry and traversal share one shape contract. Groups
stride over output tiles; partial tiles mask channels before loads and stores
while keeping clustered reductions outside the divergent mask. K and N remain
arguments, and the execution context supplies subgroup width. This template
does not choose the caller's worker budget or a concrete target record.

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

The CDNA Q5 matrix schedule stages a 128-channel by 256-element weight panel
and a 64-row activation panel in 61.5 KiB of shared memory. Eight 64-lane
subgroups reuse those panels with integer matrix products, applying F32
affine coefficients after each 32-element group. Raw and channel8 inputs
share the same schedule. Concrete targets and the live row count remain
caller-owned; packed weights never become a persistent expanded tensor.

These are reusable Loom schedules, qualified for correctness through the
[`linear` kernels](../../../kernel/ggml/linear/). The native tests establish
packed-byte semantics, addressing, and numerical results; they do not establish
a throughput comparison against another implementation.

## Paired Q5_K matrices with F16 storage

`q5_k_f16_pair.loom` contracts two projections and publishes their SwiGLU
combination without intermediate global gate/up matrices. A 256-row by
128-channel output tile stages K64 activations and both weight panels in
64 KiB of workgroup memory. Encoded packets and their K256 headers are read
ahead for each K64 step. Reconstructed coefficients
round to F16, matrix products accumulate in F32, and the fused result rounds
once to F16. Raw and channel-eight records share this arithmetic.

The tile accepts explicit origins. The traversal uses a caller-bounded worker
grid and groups output panels to retain a useful weight working set. It visits
only live rows, masks partial tiles, and retires shared readers before reusing
storage. The concrete [paired kernel](../../../kernel/ggml/linear/q5_k_f16_pair.loom)
owns the count-buffer ABI, target binding, and capacity/budget calculation.
The matrix fragments require the RDNA wave32 interpretation; changing the
target flag does not supply a wave64 implementation.

## Q6_K matrices with F16 storage

`q6_k_f16_matrix.loom` reuses a 256-row by 256-channel tile with K64 read-ahead
and the same permuted F16 staging as paired Q5. Raw six-bit weights round to
F16 at publication into shared memory; the dot accumulates in F32 and either
overwrites F32 output or adds into it. Activations arrive as F16, so conversion
placement is part of the producer contract. The existing F32 and integer-dot
schedules retain their distinct arithmetic and storage contracts.

The tile owns numerical work and the traversal groups eight channel tiles to
bound the active weight set. Explicit capacity and worker-budget facts retain
single-tile specialization while larger assignments reuse worker storage.
The concrete wave32 kernel owns count-buffer sampling and target selection;
empty prefixes and both tile tails require no padded global operands.

## Q6_K register panels with F32 storage

`q6_k_f32_panel.loom` shares a 512-element activation panel across 48 output
channels and retains decoded weight codes across 16 activation rows. A
256-thread worker uses 40 KiB of shared memory for activations and registers
for codes, scales, and row accumulators. Each K panel needs a publication
barrier and a retirement barrier; weight decoding and row contractions need
no intermediate shared-memory exchange. The row reduction distributes the
sixteen results across logical 32-lane groups using subgroup shuffles.

Inputs, products, and accumulation remain F32. The packed format accessors
supply exact six-bit codes and F32 scale pairs without rounding reconstructed
weights or creating an expanded weight tensor. This is a separate arithmetic
and reuse strategy from a matrix-instruction schedule, not a new quantization.

The tile entry accepts explicit row/channel origins. The traversal entry
assigns tiles to the caller's bounded worker grid and reuses each worker's
storage. Both overwrite and residual publication mask live rows and channels;
an odd final 256-element K block also masks missing activation and weight
reads. Buffers need no padding. The public
[`panel kernels`](../../../kernel/ggml/linear/q6_k_f32_panel.loom) own the
sampled-count ABI and worker budget, while a model owns selection between
this schedule and its other projection implementations.

## Q8_0 panels with BF16 storage

`q8_0_bf16_matrix.loom` retains a tile of F32 accumulator fragments for a
caller-owned epilogue. The three numerical contracts make operand rounding
explicit:

| Contract | Matrix operands | Scale application | Staging |
| --- | --- | --- | --- |
| `ggml.q8_0_bf16.contract` | Original BF16 inputs and exactly represented I8 codes | F32 FMA after each K32 dot | K32, 40-half pitch |
| `ggml.q8_0_f16.contract` | Inputs and reconstructed coefficients rounded to F16 | Included in F16 coefficients | K32, 40-half pitch |
| `ggml.q8_0_f16.contract_k64` | Inputs and reconstructed coefficients rounded to F16 | Included in F16 coefficients | K64, 72-half pitch |

All three use F32 accumulation. Tile dimensions, the wave32 grid, read-ahead
depth, live input rows, and physical weight layout are caller facts. The motif
contains no workload threshold, model configuration, epoch-header decoder, or
global converted-weight allocation. Full panels reuse staged weights across
rows and staged inputs across columns; returned fragments can feed fused
operations before publication.

Raw interleaved records and prepared four-channel scale/code planes share the
same contraction. A byte origin and physical row stride select a submatrix of
a larger weight bank without copying it. Stream-major weight rows can supply
multiple feature panels to each wave for fused epilogues. These fragments use
the RDNA wave32 layout; a CDNA/wave64 contraction is a different implementation,
not an interchangeable target flag.

The independent [Q8 matrix suite](../../../kernel/ggml/linear/test/) owns tile,
wave, staging, stream, layout, tail, and numerical coverage. Model suites retain
only their distinct launch policy, preparation, and fused state/publication
contracts, reusing its scalar-byte oracle.
