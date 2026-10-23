# Quantized linear projections

The affine projection family multiplies Q8_1 x4 activation rows by Q4_K or
Q5_K weight rows and writes contiguous F32 outputs. The configuration and launch
dimensions are `(token_count, input_size, output_size)`. The launch buffers are
`(activations, weights, output)`, with no aliasing between them. Each source
states its exact shape bounds and storage requirements.

Launch dimensions follow the current workload. There are no package-global
capacity settings, so a single program can use different projections without
rebinding a library configuration. Concrete target selection belongs to the
caller or a target-specific kernel leaf; the schedules require 32- or 64-lane
subgroups. Generic leaves have gfx11 and gfx9-4 compilation profiles; the
same numerical scenarios execute on either family, exercising the native
subgroup width. The explicitly bound gfx11 provider remains gfx11-only.

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

The Q6_K family accepts either contiguous F32 or Q8_1 x4 activation rows. Its
F32 kernels process one to four token rows and share activation loads between
pairs of output channels. Its generic Q8 kernels use eight independent 32-lane
output clusters per workgroup. The gfx11 provider binds a 64-lane target and
processes four packed weight blocks concurrently for each output channel. These
providers share export families, so a linked model can select the concrete
target implementation while keeping the same command ABI. The `m4` forms
process exactly four rows with separate accumulators.

The three Q6 kernel libraries compile their providers independently, so each
resulting device binary has unique export names. Their adjacent test sources
check all six implementations against the private CPU scalar-byte decoder in
`test/reference.loom`, with F64 accumulation. One-, three-, and five-block
workloads exercise incomplete subgroup iterations, one-channel outputs, and
odd channel tails.
Random data covers signed six-bit values, signed group scales, zero and signed
block factors, and independently generated activation rows. Larger workloads
include the 17,408-value F32 reduction, wide Q8 projections, and prefill through
2,048 rows. Their uniform nonzero construction has an exact analytical result,
so large workloads retain correctness coverage without a large CPU reference
contraction. Prefix and suffix output guards and unchanged inputs are checked
bit for bit.

Q8_0 uses the same Q8_1 activation carrier with raw 34-byte signed weight
records. Its grid configuration names the live row count, while the first
launch argument bounds the available row capacity. This permits a prefill
stage to reuse a larger allocation without computing inactive rows. The
adjacent tests cover that distinction explicitly, signed-byte extrema,
signed and zero scales, two-byte-aligned weight origins, partial channel
tiles, reductions through 32,768 values, and prefill through 2,048 rows.
Small cases use independent scalar-byte F64 accumulation; larger cases use
a nonzero analytical result. Inputs, output guards, and inactive rows remain
bit exact.
