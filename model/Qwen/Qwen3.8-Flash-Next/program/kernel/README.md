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

| Family | Storage and numerical contract |
| --- | --- |
| Q8 projection | Original 32-code GGML blocks, BF16 activations, and F32 block-scaled reductions. Wide matrix schedules explicitly select their F16 operand boundary. Prepared K2560 QKV weights use four-channel code/scale planes in the original allocation. |
| Hyperconnection | Four BF16 residual streams, F32 normalization scales, rank-320 Q8 projections, and target F32 or predictor Q8 injection weights. |
| GDN | BF16 convolution publication and F32 recurrence over 128 keys per head. Ordinary, windowed, fused, and replay paths share the retained-state layout and reader/writer ownership. |
| Sparse attention | Paged BF16 K/V, compressed index blocks, stable top-block selection, and per-query visibility. Cohort records retain resolved ownership and membership so matrix consumers do not repeat span lookup or selection. |
| Experts | Stable top-ten routing among 512 experts, one shared expert, and the checkpoint's mixed GGML encodings. Task records group token rows by expert; matrix and small-row paths retain their specified activation rounding. |
| PLE | Checkpoint hash multipliers and primes, packed IQ4_NL n-gram lookup, two prior token IDs, and nine prior BF16 convolution vectors. |
| MTP and output | Accepted-history vocabulary membership, independent embedding/hidden projection rounding, private draft cache, target-authoritative acceptance, and selected output-row publication. |

The tests use scalar and closed-form references, original encoded bytes,
canaries, inactive NaNs, retained forks, and exact state comparisons. Matrix
tiles, wave arrangements, pipeline depths, and preparation layouts have
explicit test configurations. Correctness and instrumentation runs establish
these contracts; they do not establish a throughput or latency claim.
