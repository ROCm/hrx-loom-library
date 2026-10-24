# Tensor numerical motifs

`random.loom` supplies stateless Philox4x32-10 and F32 normal-pair templates.
A caller assigns counter words to request, position, layer, or tensor element
identities and supplies the seed. The motifs retain no mutable generator state
and contain no launch or target configuration. Reordering workgroups therefore
leaves a sample unchanged when its counter and key stay unchanged.

`random.philox4x32` accepts four counter words and two key words as unsigned
32-bit bit patterns. Arithmetic wraps modulo 2^32. Ten rounds produce four
output words with the [Random123 Philox4x32-10](https://github.com/DEShawResearch/random123)
mapping. Counter allocation belongs to the caller; the motif does not prevent
two callers from choosing the same counter and key.

`random.normal_pair` applies Box-Muller to two words. Each word is converted to
F32, offset by half a bin in F32, and scaled by 2^-32. The radial uniform is
clamped below one at `1 - 2^-24`; zero retains the half-bin offset. The angular
uniform is measured in turns, and a rounded value of one wraps naturally to
zero turns. F32 logarithm and trigonometry permit the target's approximate
operations. Normal samples have numerical tolerance rather than a cross-device
bitwise identity; Philox words have exact identity.

`sampling.loom` consumes finite F32 logits and applies temperature, top-k,
nucleus selection, and a categorical draw entirely on the device. Temperature
zero selects the greatest logit, breaking ties by the lowest vocabulary ID.
Positive temperature uses one Philox counter per token ID and absolute target
position, keyed by the full 64-bit request seed. Packing a request into another
row or changing speculative depth leaves that random stream unchanged.
Floating-point differences can still move a draw across a probability boundary.

The caller supplies a complete 256-thread workgroup, `1..2^24` vocabulary
entries, nonnegative top-k, and top-p in `(0, 1]`. A positive temperature is a
normal F32 value. Top-k zero includes every eligible ID; a larger top-k clamps
to the eligible vocabulary. The exclusion variant removes one ID before
support selection, and `-1` excludes nothing. At least one finite eligible
logit remains. The sampler keeps no probability array or mutable random state;
support selection partitions a strict logit/ID ordering using workgroup
reductions.

The same module exposes `sampling_key` for consumers that select from a
different candidate domain, such as a draft vocabulary. Unsigned greater
keys select larger finite logits and break ties by smaller nonnegative I32
IDs, treating both signed zeros equally. The complemented ID occupies the
low 32 bits, and zero is below every valid key. Sharing this order keeps
target sampling and subset selection consistent without sharing their scans.

`topk.loom` exposes a different result: the ordered values and IDs themselves.
Each 256-thread workgroup retains sixteen values per thread, producing up to
sixteen winners from a 4096-element tile. Its merge consumes disjoint partial
lists with at most 4096 candidates, so it does not reread the full vocabulary.
Equal values choose the lowest unique nonnegative ID. Short tiles fill missing
winners with negative infinity and `INT_MAX`; valid input logits are finite.
Inputs and outputs are separate, and the composition owns both launch geometry
and temporary partial-list storage.

`linear.loom` separates BF16 projection tiles from their publication. The
storage-selected contraction reads BF16 activations and either BF16 weights
or F32 weights rounded to BF16, with checkpoint rows indexed by output channel.
It accumulates sixteen-input matrix operations within 256-input partials and
adds the completed partials in F32. The native-BF16 provider instead uses four
independent chains within each partial. These recurrences remain distinct;
matrix operations follow the selected target's native arithmetic contract.

Composition supplies each tile's row and output-column origins, independently
of launch-grid traversal. One complete 32-thread workgroup owns a 16×16 tile.
Rows, output columns, and tile origins are multiples of sixteen; reduction
widths are multiples of sixteen for storage-selected contraction or sixty-four
for the four-chain form. Operands, their conversions, and intermediates remain
finite. The external `tensor_projection_rdna3` target binds the RDNA3 fragment
layout in a composition leaf; the motif contains no concrete target or global
configuration. Launch targets separately select wave32 execution.

`projection.store_bf16` rounds the result into a caller-supplied injective
layout, including dense, column-major, or padded storage. The bias epilogue
reads native BF16 bias or rounds F32 bias to BF16 before adding in F32.
`projection.adapter_epilogue_bf16` rounds the projected delta to BF16,
multiplies by the runtime F32 strength, rounds that product to BF16, and adds
the BF16 base in F32. Publication performs the final BF16 rounding. Either
signed zero strength returns the base directly, preserving its signed zeros.

Activations and weights are immutable, naturally aligned, and disjoint from
output. Bias and strength are also separate from output. An adapter update
may reuse its exact base allocation with the same layout because each tile
owns every value it reads and writes. Partial overlap is unsupported. These
helpers allocate no scratch, publish no global partials, and add no dispatch
or synchronization boundary.

`layer_mix.loom` reduces twelve BF16 layer taps for one token/channel. A lane
reads 24 bytes from `[row, layer, channel]` input once and retains twelve F32
values for the base and optional low-rank contractions. Adjacent lanes gather
adjacent channels; no transpose or intermediate rank tensor is materialized.
The caller owns the selected coordinates, launch, and separate result storage.

Base coefficients may be BF16 or F32 storage. F32 coefficients round to BF16
before ordered F32 fused multiply-adds, and the result rounds to BF16. For
enabled low-rank factors, each F32 down/up coefficient also rounds to BF16.
Each down result rounds before the up contraction; its complete delta and
scaled delta round separately before the final addition and BF16 result.
An inactive adapter does not read its strength or factors. Either signed-zero
strength returns the base bits without accessing factor storage. These scalar
and gather providers are target-neutral, with finite active arithmetic and
naturally aligned immutable inputs.

`attention_shared.loom` and `attention_shared_k32.loom` implement noncausal
128-channel BF16 attention with four query heads sharing each KV head. A
128-thread workgroup contains four wave32 subgroups, each owning sixteen
queries of one head. Q, gate, and output are row-major
`[tokens, 4 * key_value_heads * 128]`; K is row-major
`[tokens, key_value_heads * 128]`, and V is physically transposed to
`[key_value_heads * 128, tokens]`. Inputs are immutable, naturally aligned,
and complete; output is disjoint. The token count is a positive multiple of
sixteen for K16 and thirty-two for K32. Every workgroup thread participates
in shared-memory publication and retirement.

The query panels remain resident in registers and shared memory while all
four waves share each K/V tile. K16 uses 19,712 shared bytes; K32 uses 32,256.
Read-ahead carriers clamp the final prefetch to a valid complete tile. A byte
mask enables keys by nonzero value, with at least one enabled key overall;
leading, interior, and trailing empty tiles are supported. Negative infinity
represents the empty maximum. Its indeterminate rescale delta selects zero
through `minnum`, preserving the empty accumulator until a valid key arrives.

Unnormalized probabilities round to BF16 for the value contraction, while
normalization retains their unrounded F32 sum. Gating rounds both context and
sigmoid to BF16 before multiplying and rounding the output. With gating
disabled, the gate buffer is never read and may name output. The store
provider exposes this same epilogue for a caller-owned sixteen-by-sixteen
result fragment. The external `tensor_shared_attention_rdna3` target qualifies
the fragment layout; composition supplies its concrete backend and wave32
launch. Active inputs, contractions, and normalized values are finite, with
native matrix intermediate precision following the target contract.

`segmented_attention.loom` attends within independent packed segments of
128-channel BF16 heads. One complete 32-lane subgroup owns a query/head, with
four adjacent channels per lane. Positive row, segment, and head extents are
explicit; rows contain complete segments, coordinates are subgroup-uniform
and in bounds, and Q/K/V storage is eight-byte aligned. The inputs are
immutable and may overlap. The caller owns launch bounds, subsequent gating,
and publication into separate output storage.

The subgroup retains its query and first key while streaming one key/value
row at a time. Centering keys removes their common logit term. F32 online
maximum, normalization, and value accumulation avoid a score tensor, shared
memory, and additional dispatches. Per query/head, logical reads are 256 bytes
each for the query and anchor, then 512 bytes per segment row. Approximate
exponentials and reciprocal division remain explicit, and the final values
round to BF16. The provider requires subgroup size 32 without embedding a
concrete backend or model configuration; active arithmetic remains finite.

`linear_tiled.loom` shares BF16 activation and weight panels across eight
wave32 subgroups. A 256-thread workgroup owns a 64×64 output tile and acquires
64 reduction elements at a time into two padded 64×72 workgroup-memory views.
Four independent K16 accumulator chains combine every 256 inputs, matching
the single-wave four-chain recurrence. Publication and retirement barriers
protect the shared panels while a two-deep pipeline acquires global inputs.
Partial row tiles zero-fill missing rows without removing barrier participants.

Rows are positive multiples of sixteen, reduction widths of 256, and output
widths of 64. The leaf launches `ceil(rows/64) × (outputs/64)` workgroups and
supplies an injective result layout, separate immutable input/weight storage,
and finite arithmetic. Full output tiles write 8 KiB. Each K64 step acquires
at most 8 KiB from activations and 8 KiB from weights, with 18 KiB of padded
workgroup memory shared by the complete tile. No global partial tensor or
checkpoint preparation is required.

`projection.linear_tiled_coordinates` maps the physical launch grid into
bounded groups of row tiles, traversing columns within each group to retain
input-panel locality. A final incomplete group uses its actual row count;
groups spanning the full height preserve the original coordinates. The
caller supplies positive grid extents and group size. This coordinate helper
is target-neutral. The cooperative matrix provider uses the external
`tensor_projection_tiled_rdna3` fragment-layout binding, and composition
selects its wave32 launch target separately.

`rmsnorm.loom` computes `x * rsqrt(mean(x*x) + epsilon) * weight` for one
F32 row per workgroup. A thread retains its four-element packets across the
reduction, so normalization reads each input once. The caller specializes the
row width and thread count and supplies byte origins for each input and output
row. Output is either F32 or rounded directly to F16; an output origin can
place the result into a larger concatenated row without another copy.

Widths are positive multiples of four times the workgroup size. Input,
weights, and output do not overlap, and origins support aligned four-element
accesses. The arithmetic contract requires finite operands, a positive
normal `mean(x*x) + epsilon`, and finite F32 intermediates. The motif neither
allocates persistent state nor chooses a target, launch grid, or global
configuration.

The BF16 RMS providers expose the row inverse and two different epilogues.
`normalization.rms_inverse_bf16` reduces an immutable row in F32, divides by
its width, adds the caller's epsilon, and evaluates an approximate reciprocal
square root. Every workgroup lane participates; positive width fits U32 and
is divisible by the actual thread count. The biased mean is positive and
normal, and the sums remain finite.

`normalization.zero_centered_bf16` rounds its F32 weight to BF16, adds one
and rounds the scale to BF16, then performs normalization and scaling in F32
before the final BF16 result. `normalization.scaled_bf16` instead rounds the
normalized value to BF16 before multiplying its BF16 weight. Its scale has
no added one. These helpers preserve distinct tensor boundaries when fused
with other work.

The corresponding whole-row providers take epsilon explicitly and use one
complete workgroup per row. Input is read during the reduction and again by
the epilogue; no normalized tensor is materialized. Naturally aligned output
may equal input exactly or occupy disjoint storage. Immutable weights remain
separate. Every participant finishes the reduction before row stores begin,
and each lane then owns its original strided channels.

`rotary.loom` preserves two channel layouts and their tensor rounding rules.
`attention.normalize_rotary_bf16` normalizes with the zero-centered RMS
epilogue, then rotates adjacent channels using F32 coefficient tables and
F32 arithmetic before its final BF16 output. The normalized values remain
in registers at their BF16 boundary.

`attention.half_pair_bf16` instead joins corresponding channels from the two
halves of a row. Each cosine and sine product rounds to BF16 before the final
sum rounds again. `attention.normalize_half_rotary_bf16` combines that pair
operation with scaled BF16 RMS; `attention.half_rotary_bf16` applies it without
normalization. Both use BF16 coefficient tables.

Composition supplies positive token/head counts, an even width fitting U32,
and exactly `width/2` threads per head row. Tables index tokens and channels,
sharing each token's coefficients across its heads. Normalized variants take
epsilon explicitly. Each thread reads both pair values before either store,
allowing output to equal input exactly. Weights and tables remain immutable
and separate from output. Launch geometry and coefficient generation belong
to the caller.

`quantization.loom` prepares BF16 activation rows for ConvRot-256 signed INT8
projections. Each 256-element block receives the fourth Kronecker power of
the regular H4 matrix, normalized by `1/16`, and rounds to BF16. One maximum
over the complete row determines its scale: `max(amax / 127, 1e-30)` in F32.
The divisor rounds to BF16, and the quotient rounds to BF16 before nearest-even
integer rounding and saturation to `[-128, 127]`. The published row scale
retains F32 precision for the subsequent integer contraction.

One 128-thread workgroup with four wave32 subgroups owns each row. Positive
column counts are multiples of 256 up to 16384. Each input is loaded once;
the rotated BF16 row stays in workgroup memory across the maximum reduction.
A workgroup barrier publishes both the row and the four subgroup maxima.
Input, signed-I8 output, and F32 scales occupy disjoint, naturally aligned
storage. Composition can reuse one prepared row across multiple projections
while retaining the original BF16 input for other consumers.

`quantization.convrot_digit` supplies one regular H4 digit over a complete
32-lane subgroup with distance 1 or 4. Its eight independent F32 components
correspond to different positions in the enclosing 256-element transform.

`grouped_convolution.loom` applies a causal F32 convolution independently to
each block. A coefficient combines a static per-channel weight with a dynamic
weight shared by a contiguous group of channels. Dynamic weights vary by
block, position, branch, and tap. The selected branch produces a new tensor
or adds its update to a separate residual tensor. Taps before position zero
are padding; they never read another block's history.

The vector primitive returns four adjacent channels for a caller-owned
position. Its whole-tensor forms use 128 threads and a 512-channel tile, with
the grid supplied by the leaf. Group widths are positive multiples of four
that divide the channel count. Input and coefficient storage never overlap
the output; only the residual destination is updated in place. The source
documents the exact row-major coefficient layouts and branch bounds.

`embedding.loom` selects one channel from an immutable row-major BF16
embedding table. The caller supplies the vocabulary size, row width, token
ID, and channel. The lookup preserves the stored bits, including signed
zeros, subnormals, infinities, and NaN payloads; it performs no arithmetic
conversion.

Vocabulary and width are positive, the nonnegative I32 token ID is below
the vocabulary size, and the channel is below the width. The table is
naturally aligned and large enough for the declared geometry. Request
validation establishes the token bounds before launching device work. The
motif returns one value without owning the output, launch, or padding, so
the leaf can distribute channels and represent padded rows independently.

`pointwise.loom` supplies BF16 SiLU, tanh-polynomial GELU, and gated residual
arithmetic. Activations evaluate in F32 and return a BF16 scalar. The residual
rounds `update * gate` to BF16 before adding the old value and rounding again.
Those two tensor boundaries are part of the operation: a fused multiply-add
or a single final rounding computes a different function.

`diffusion.loom` exposes the corresponding Euler scheduler step. Its velocity
and sample are BF16, while its time delta remains F32. The F32 velocity product
rounds to BF16 before the sample addition and final BF16 rounding. Each call
returns one scalar; the composition owns tensor access, in-place updates, and
launch geometry. Neither module imposes a workgroup size or target. Operands
and arithmetic intermediates are finite; approximate activation math does not
promise bitwise agreement across devices.

`modulation.loom` combines time-dependent BF16 values with learned checkpoint
tables. `modulation.combine_bf16` rounds the F32 table value to BF16, adds the
modulation value in F32, and rounds the sum back to BF16. Its affine form
combines scale and shift separately, rounds scale-plus-one to BF16, rounds
the scaled input to BF16, and then adds the combined shift and rounds again.
Removing an intermediate rounding changes the function.

The caller supplies finite values whose arithmetic intermediates remain
finite. Each helper returns a scalar without owning buffers, normalization,
or launch geometry. A model can fuse normalization and modulation without
materializing an expanded scale/shift tensor. Checkpoint storage selection
stays in the leaf: an F32 table and a widened BF16 table use the same scalar
contract.

`image_unpack.loom` decodes one F32 pixel from a tensor of packed 2×2 BF16
patches. Patches are ordered by spatial row and column, then channel, then the
four pixels within the patch. The affine table contains one positive inverse
standard deviation per channel followed by the channel means. Decoding divides
by the inverse standard deviation and then adds the mean, rounding each F32
operation separately.

The caller supplies positive even image geometry and a valid channel and pixel
index. Input buffers are immutable and naturally aligned for their element
types; operands and intermediates are finite. The motif returns a scalar, so
the composition owns tail masking, work distribution, and output placement.
It reads each selected BF16 sample once and contains no target or launch
configuration.

`image_attention_matrix.loom` owns a 384-channel image-attention head with
F32 global storage and BF16 matrix operands. Q/K occupy channel-first planes,
followed by a token-first V plane; output is channel-first and disjoint.
Sixteen queries share one 128-thread workgroup. Four wave32 subgroups split
the score contraction into 96-channel partials, then each owns 96 output
channels. The token count is a positive multiple of sixteen, and all threads
participate in shared-memory publication and retirement.

The query panel stays resident throughout the key scan. Each key tile reads
24 KiB of K and 24 KiB of V once for the complete workgroup; padded operands
and partial scores occupy 55,552 bytes of shared memory. Unnormalized
probabilities round to BF16 for the value contraction, while their denominator
retains unrounded F32 values. Online rescaling, accumulators, and output remain
F32. The external `tensor_image_attention_rdna3` target qualifies the fragment
layout, and the leaf supplies its concrete backend binding and wave32 launch.
Operands, their conversions, contractions, and normalized values remain
finite; native matrix intermediate precision follows the selected target
contract. A negative-infinity initial maximum lets the first finite score
tile establish normalization, including when every score is very negative.

`channel_norm.loom` normalizes one pixel across a channel-first F32 tensor.
It accumulates channel squares in order with F32 fused multiply-add, takes
the native approximate square root, and clamps that norm to a caller-supplied
minimum. Each channel is divided by the bounded norm, multiplied by the square
root of the channel count, and multiplied by its learned gamma, with separate F32
rounding. Square roots carry explicit approximation permission; the operation
does not promise correctly rounded normalization. An optional SiLU epilogue
uses approximate exponential math.

The caller owns pixel assignment and bounds. Geometry is positive, channel
counts fit unsigned I32, the minimum norm is positive, and all operands and
intermediates are finite. Buffers are naturally aligned. Input and output
may be disjoint or identical; gamma is immutable and separate, and partial
input/output overlap is unsupported. The complete reduction precedes output
stores, so the in-place form preserves the same arithmetic. Input is read
once for the reduction and once for output, without an intermediate tensor.

`convolution.loom` computes one selected channel-first F32 image output. It
reads original OI(T)HW coefficients from the final temporal plane, accumulates
input channels and spatial taps in order with F32 FMA, and adds bias after
the reduction. Nearest-neighbor upsampling is an input-coordinate transform;
padding applies to the expanded image before mapping to physical input.

Geometry and plane counts are positive, the spatial extent is odd, and the
caller selects a valid output channel and pixel. Inputs, coefficients, and
bias are immutable and naturally aligned; index expressions and finite F32
intermediates must fit their types. The motif returns a scalar. The leaf owns
thread distribution, tails, output placement, and any residual addition, so
it can preserve a separate rounding or combine the value with other work.

`time.loom` evaluates one BF16 sinusoidal time feature. The caller supplies
the number of frequencies, selected channel, negative logarithm of the
frequency period, time scale, and BF16 time. Cosine features precede sine
features. The implementation constructs the frequency with approximate
exponential math, separately rounds scaled time and phase in F32, evaluates
the selected approximate trigonometric function, and rounds the result to BF16.

Half-width is positive and fits U32, the channel is below twice that width,
and all arithmetic remains finite. The motif has no buffer or launch ownership.
Krea uses 128 frequencies, period 10000, and time scale 1000; its leaf owns
the padded row count and supplies positive zero time for padding. Other
frequency grids and launch widths do not require copying the arithmetic.

`convolution_matrix.loom` cooperatively computes a 3×3 channel-first image
convolution. Its original F32 inputs and OI(T)HW weights are converted to BF16
inside a workgroup, with F32 matrix accumulation and separate F32 bias and
optional residual additions. Matrix arithmetic follows the target's native
intermediate precision and rounding; it does not promise the bits of an ordered
scalar FMA reduction. Padding, nearest expansion, and final temporal-plane
selection follow the scalar convolution's indexing contract.

Composition binds `tensor_convolution_rdna3` to an identity-only RDNA 3 target
and launches four wave32 subgroups per workgroup. One workgroup computes
32 output channels by 32 output pixels; input and output channel counts are
positive multiples of 32. The launch grid is `ceil(output_pixels / 32)` by
`output_channels / 32`. All 128 threads enter the motif, including threads
outside a partial spatial tile. Each reduction block publishes 32×32 input
and weight panels in padded BF16 workgroup storage totaling 4736 bytes. Both
workgroup barriers are collective: the second retires readers before the next
block reuses that storage.

Input, weight, and bias storage is immutable and naturally aligned. Output
may exactly alias the residual, but overlaps no other input. Geometry and
index expressions fit their types, and BF16 conversions and F32 intermediates
remain finite. Nearest expansion gathers repeated positions directly from
physical input; it does not allocate an expanded image or converted weights.

`matrix.loom` contracts a caller-owned sixteen-row F16 panel against a
sixteen-column window of another matrix. The left panel may use explicit
physical strides, including a transposed and padded probability tile; the
right input and output are row-major with independent origins and column
windows. A complete 64-lane subgroup owns one output tile. The reduction is a
compile-time multiple of sixteen, while buffer origins may remain dynamic.
Callers own staging, barriers, output placement, and non-overlapping storage.

Composition binds the external `tensor_matrix_rdna3` and
`tensor_matrix_cdna3` targets. RDNA uses replicated sixteen-element F16 operand
payloads and an eight-element F16 accumulator payload; CDNA uses four-element
F16 operands and four-element F32 accumulators. Both publish F16 directly.
The operand-count query expresses this difference explicitly instead of
inferring it from subgroup width. Native matrix arithmetic determines internal
reduction order and precision, so F16 accumulation does not promise the result
of a full F32 dot product rounded once. The
[RDNA 3 ISA, section 7.9](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf)
describes the WMMA instruction and its replicated operand contract.

The [checks](test/README.md) use published integer answers, independent
double-precision random transforms, and serial order-statistic references.
These motifs participate in native GPU model programs while their composition
owns tensor layout, launch configuration, and batch membership.

`prefix_norm.loom` scales a BF16 row by the reciprocal L2 norm of a selected
prefix, producing F32 values. The caller supplies the row width, prefix length,
positive squared-norm floor, and two nonoverlapping views. Width is a multiple
of 32 through 65,504, and the views are dense and 64-byte aligned. Every output
lane receives the same scale; exposing only the selected prefix belongs to the
caller. Three BF16 limbs carry the F32 reciprocal norm into native dot products,
preserving precision without requiring a target-specific declaration or a
model-specific row width. The typed views can refer directly to channel-owned
storage, without transferring that storage's ownership to the motif.

```sh
bazel test --config=asan --config=amdgpu //motif/tensor/...
```
