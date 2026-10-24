# Tensor correctness and benchmark workloads

`random.loom` executes the three published
[Random123 known-answer vectors](https://github.com/DEShawResearch/random123/blob/d8a0c25e49992685c0b9d91b553c1d04f81b6b4b/tests/kat_vectors)
for Philox4x32-10 through the VM reference provider: zero words, all-one words,
and mixed counter/key words. Every output word is checked exactly.

`random_gpu.loom` loads those same inputs from device buffers and compares
native GPU results exactly under the `gfx11-generic` execution contract. Its
second kernel checks 64 normal pairs crossing
radial endpoints, F32 conversion boundaries, and angular octants. The expected
values come from double-precision libm after the declared F32 uniform
conversion, independently of Loom's approximate target instructions. The
accepted error is `2e-9 + 5e-7 * abs(expected)`, including the near-zero angular
results and the smallest positive radial sample.

```sh
bazel test --config=asan --config=amdgpu //motif/tensor/test:all
python3 -B motif/tensor/test/generate_normal_oracle.py
```

The benchmarks select complete counter blocks and a 64-pair normal workload.
They exercise runtime buffer inputs so native arithmetic remains in the
generated kernel. These are correctness and reproducibility checks, not a
claim about model throughput or a selected tuning configuration.

`sampling.loom` compares native categorical draws against an ordinal-rank
reference that directly counts each logit's predecessors. The reference
evaluates exponential-race scores with F64 logarithms and does not call the
production sampler's support-selection or random-transform helpers. Philox
counter generation shares the separately verified integer primitive. The
cases cross greedy and sampled requests, bounded and unbounded top-k, nucleus
filtering, four exclusion choices, and a 67-element vocabulary with ties.
A 248320-element case has an analytically known nineteen-token support, which
checks the production vocabulary extent without a quadratic full-size oracle.

`sampling_key.loom` compares every pair of runtime logit/ID records against
direct numeric and ID comparisons. Directed inputs cover finite extrema,
signed zero, normal/subnormal endpoints, and tied IDs through `INT_MAX`;
random logits cover ordinary orders. Complemented-ID recovery, the empty
candidate sentinel, output guards, and unchanged inputs are checked exactly.

`topk.loom` checks exact values and IDs against a serial strict-order oracle.
It covers the 248320-element vocabulary, multiple rows, a short nine-element
tail, tied maxima, and merging 61 disjoint partial lists. The complete
two-dispatch case expects the sixteen lowest IDs among equal winners from
separate tiles. Its benchmark declarations retain these different input and
merge shapes rather than reporting one synthetic size.

`rmsnorm.loom` compares retained-row normalization with a sequential F64
sum of squares and Newton-refined inverse square root. Widths 1024 and 5120,
workgroups of 128 and 256 threads, and F32/F16 publication exercise distinct
register-bank and rounding paths. Multiple rows, nonzero input/output byte
origins, and concatenated output strides retain sentinels outside the live
rows. Epsilon-dominated and zero inputs cover the normalization denominator.
F32 results use a `1e-6` absolute and relative bound. F16 comparisons use
`1e-7 + 0.001 * abs(expected)` because F32 reduction can move a midpoint
across the final rounding boundary. Zero outputs are checked exactly.

`grouped_convolution.loom` uses a scalar F64 oracle that combines the static
and dynamic coefficients before multiplication and explicitly pads every
tap. It checks three independent eight-position, 5120-channel blocks,
ordinary and residual output, a 516-channel tail, and a single position with
four taps. The different branch and group sizes exercise coefficient
addressing, while bitwise input checks guard the separate-output contract.
All four shapes have benchmark declarations.

`pointwise.loom` checks all four BF16 arithmetic contracts through native
runtime inputs. SiLU and tanh GELU use binary64 libm formulas at 265 points,
including both curves' negative minima, small signed inputs, and tails through
±16. Their error bound is `2e-6 + abs(expected) / 256`: the relative term is
BF16 round-to-nearest's maximum error, while the absolute term admits F32
approximation and cancellation near the negative tail. The expected values
retain the independent formula's F32 precision rather than rounding to BF16.

The 128 residual and 132 Euler cases compare exact bits against rational
arithmetic with integer ties-to-even rounding. They cover positive and negative
products, both midpoint parities, cancellation, nonzero residuals, and a time
delta that must remain F32. Directed Euler inputs also place F32 multiplication
on either side of BF16 midpoints. The corpus includes explicit differences
from a fused update and from prematurely rounding the time delta. Each family
has its own benchmark declaration, and nonmultiple launch tails are bounded
by the wrapper kernel. Regenerate the authored answers with:

```sh
python3 -B motif/tensor/test/generate_pointwise_oracle.py
```

`image_unpack.loom` starts from independently authored channel-first pixels
and packs its BF16 input by traversing complete 2×2 patches. Its expected
outputs never reconstruct coordinates from a packed index. Minimum, portrait,
landscape, exact-group, tail, and multiple-group shapes exercise one, odd, and
sixteen channel counts with 32-, 64-, 128-, and 256-thread leaf launches.

Division and mean addition use exact rational ties-to-even references for
their separate F32 roundings. Two cancellation cases distinguish that result
from dividing and adding before rounding. Every output bit, prefix and suffix
guard, input sample, and affine coefficient is compared. Both ordinary native
execution and access instrumentation use the same scenarios; each shape also
has a benchmark declaration. The authoring generators share the exact binary
rounding primitive in `binary_reference.py`.

```sh
python3 -B motif/tensor/test/generate_image_unpack_oracle.py --check
```

`channel_norm.loom` covers affine and SiLU epilogues with disjoint and exactly
aliased output. Twelve cases include one and odd channel counts, the decoder's
96/192/384-channel widths, complete workgroups, tails, multiple workgroups,
signed zeros, and values on both sides of the norm clamp. A distinct minimum
norm also checks that the caller owns that parameter.

The reference uses exact rational F32 accumulation and arithmetic. Square
roots use integer bounds and squared midpoint comparisons for ties-to-even
rounding. The native implementation permits approximate roots: the
[RDNA 3 ISA, section 16.8](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf)
specifies one-ULP F32 square-root accuracy. With `u = 2^-24`, each native root
differs from the rounded reference by at most `3u / (1-u)` relatively.
Propagating two roots and three rounded arithmetic steps gives less than
`16u = 2^-20` relative error for the normal finite corpus. General affine
checks use that bound with zero absolute tolerance; directed exact-root and
clamp-dominated cases preserve bitwise results, including signed zeros.

SiLU uses a binary64 expression with absolute tolerance `5e-7` and relative
tolerance `2e-6` for approximate root and exponential math. Tensor views expose
live values and their guards separately, so prefix/suffix guards and immutable
inputs remain bitwise checks for every epilogue. Normal and access-instrumented
executions share these scenarios and their benchmark declarations.

```sh
python3 -B motif/tensor/test/generate_channel_norm_oracle.py --check
```

`prefix_norm.loom` compares native NPU output with a serial binary64 sum of
squares and reciprocal norm. Widths 32, 64, 96, and 1024 exercise prefix lengths
on either side of 16- and 32-element boundaries, including single-element and
complete prefixes. Zero and floor-dominated rows cover the normalization
denominator. Nonzero buffer origins, exact guards, and unchanged input checks
retain storage ownership; the complete output row checks that the selected
prefix controls the scale without suppressing later lanes. The error bound is
`2e-7 + 2e-6 * abs(expected)`. Benchmark declarations retain each row width;
their execution is a workload smoke check, with no throughput claim.

`matrix_rdna.loom` and `matrix_cdna.loom` bind the same external template
contracts to their concrete target families. One- and four-subgroup launches
cover reductions of sixteen and sixty-four, row-major and transposed left
panels, padded rows, nonzero right byte and row origins, and independent
right/output column windows. Full output arenas include prefix, suffix, and
row-padding guards; inputs and the queried operand payload count are exact
observations.

The scalar F64 oracle indexes flat left storage directly, without production
fragment operations or layout objects. RDNA random inputs are integer values
from -3 to 3 divided by sixteen: each product is an integer multiple of
1/256, and every partial sum has magnitude at most 576/256. All such sums fit
exactly in F16, independently of the native dot-product order. CDNA uses the
wider -31 to 31 range, whose sums fit exactly in F32 before final F16 rounding.
Both comparisons are bitwise.

Separate constructed cases exercise native rounding and accumulator precision.
Two nonzero terms reach both midpoint parities with both signs. The live result
may differ from scalar nearest-even by one F16 step, `2^-10` at these values
near magnitude one. This bounds numerical accuracy without requiring native
matrix instructions to share scalar tie-breaking. Prefix and suffix guards,
input bytes, and operand metadata remain bitwise comparisons on both targets.

A sixty-four-term case first produces 64, then adds three sixteen-term tiles
of 1/1024. F16 accumulation retains 64, while F32 accumulation publishes
64.0625 in F16. This precision distinction remains exact. Every shape has a
benchmark workload; correctness timings do not establish a throughput claim.
