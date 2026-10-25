# Flash-Next numerical providers

The numerical library implements the checkpoint's concrete device operations.
Target records stay here, while shared GGML decoding, packed sequence
descriptors, and sampling algorithms come from the library's motifs and
kernels. Host command sequencing lives in the adjacent dispatch package.

Packed epoch capacities determine launch geometry and specialization. The
immutable epoch header separately bounds live work. Inactive rows retain
their storage, and partial tiles mask their loads and publications. Parameter
views carry exact checkpoint shapes and encodings; kernels consume those
prepared contracts rather than discovering a format from bytes or filenames.

Kernels shared with recursive drafting take `neural_enabled` as a staged
launch argument. A false value selects zero workgroups; the numerical body
and its runtime bindings remain identical. Ordinary callers supply a constant
true value, allowing specialization to retain the direct launch geometry.
The projection template forwards this argument to every shape-selected entry.

| Family | Storage and numerical contract |
| --- | --- |
| Q8 projection | Original 32-code GGML blocks, BF16 activations, and F32 block-scaled reductions. Long reductions use paired K64 stages from capacity 256, with F16 operands and F32 accumulators. Prepared K2560 QKV weights use four-channel code/scale planes in the original allocation. |
| Hyperconnection | Four BF16 residual streams, F32 normalization scales, rank-320 Q8 projections, and target F32 or predictor Q8 injection weights. |
| GDN | BF16 convolution publication and F32 recurrence over 128 keys per head. Ordinary, windowed, fused, and replay paths share the retained-state layout and reader/writer ownership. |
| Sparse attention | Paged BF16 K/V, compressed index blocks, stable top-block selection, and per-query visibility. Cohort records retain resolved ownership and membership so matrix consumers do not repeat span lookup or selection. |
| Experts | Stable top-ten routing among 512 experts, one shared expert, and the checkpoint's mixed GGML encodings. From capacity 256, task records group 128 routes per expert and share decoded F16 panels across them. Small-row paths retain their block-scaled dot arithmetic. |
| PLE | Checkpoint hash multipliers and primes, packed IQ4_NL n-gram lookup, two prior token IDs, and nine prior BF16 convolution vectors. |
| MTP and output | Accepted IDs in a registered I32 plane sharing draft KV page ownership, accepted-history vocabulary membership, independent embedding/hidden projection rounding, private draft cache, target-authoritative acceptance, and selected output-row publication. |

The tests use scalar and closed-form references, original encoded bytes,
canaries, inactive NaNs, retained forks, and exact state comparisons. Matrix
tiles, wave arrangements, pipeline depths, and preparation layouts have
explicit test configurations. Correctness and instrumentation runs establish
these contracts; they do not establish a throughput or latency claim.

The paired Q8 stage keeps two K32 blocks resident at each publication boundary,
sharing their producer work and synchronization. Fused GDN uses that stage for
capacities from 256 through 4095 and retains its projection in workgroup memory
for convolution. The prepared weights, persistent state, and output formats do
not change. The operand boundary rounds decoded coefficients and BF16 inputs
to F16 before F32 accumulation; its narrower exponent range is part of this
matrix path's numerical contract. Independent references apply that boundary
explicitly instead of treating it as exact real arithmetic.
