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
