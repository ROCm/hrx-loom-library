# Quantized linear projections

The affine projection family multiplies Q8_1 x4 activation rows by Q4_K or
Q5_K weight rows and writes contiguous F32 outputs. The configuration and launch
dimensions are `(token_count, input_size, output_size)`. The launch buffers are
`(activations, weights, output)`, with no aliasing between them. Each source
states its exact shape bounds and storage requirements.

Launch dimensions follow the current workload. There are no package-global
capacity settings, so a single program can use different projections without
rebinding a library configuration. Concrete target selection belongs to the
caller; the schedules require 32- or 64-lane subgroups. The supplied target and
hardware qualification uses gfx11.

Q5's `channel8` entry points consume complete groups of eight interleaved weight
channels, and its `m4` entry points process exactly four activation rows while
sharing decoded weight packets. The reusable schedules live in
[`motif/linear/ggml`](../../../motif/linear/ggml/). Kernel dependencies contain
only those motifs; quantization is a separate launch composed by the caller.

`affine_q8_1_x4_test.loom` owns the checks. Small seeded workloads use an
independent CPU reference with scalar packed-byte decoding, I64 dot products,
and F64 scaling and accumulation. Cases cover raw and channel8 layouts, single
and four-row launches, odd block counts, distinct rows, signed and zero scales,
and output guards. Larger decode and prefill workloads have a nonzero exact
answer derived from constant codes and coefficients. Every benchmark workload
therefore includes a correctness assertion. Benchmark smoke execution is
correctness evidence; controlled comparative performance is a separate task.
