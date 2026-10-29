# Tensor correctness and benchmark workloads

The adapter corpus uses the twelve original matrix shape families and the
fused twelve-layer scalar contraction. Separate leaves supply metadata views
to the reusable providers. Python standard-library generators create synthetic
operands and independent scalar results in Bazel output; they use no model
checkpoints. The same GPU scenarios support optional Loom address and race
checks through `--config=loom-asan` and `--config=loom-tsan`. Native tool
instrumentation is independent; `--config=asan` alone does not select Loom
checks. See [development](../../../docs/development.md) for configuration.

Selections cover empty, single, odd/even, seven-bank, reordered, duplicate,
canceling, zero, and fractional strengths. Dense and column-major layouts
exercise exact base/output aliasing and disjoint publication. Mixed `[A,B,C]`
and `[B,D]` members run in both orders over shared factors and one maximum-sized
rank allocation. Complete immutable inputs and storage guards compare exactly.

Matrix acceptance propagates the input-derived F32 contraction budget
`sum(abs(products)) / 2^20` through strength multiplication, BF16 rank scratch,
the up contraction, and final base addition. A separate nonzero corpus stays
exactly representable throughout. Scalar cases compare every bit. The reference
test proves that 138 missing-bank, wrong-bank, and wrong-strength mutations
exceed the accepted bounds; device observations never determine a tolerance.
Each scenario also declares a benchmark over its initialized inputs. Benchmarks
run separately through the [benchmark workflow](../../../docs/development.md#sweep-repository-benchmarks).
Correctness runs do not establish model throughput.

```sh
bazel test --config=asan --config=amdgpu \
  //motif/tensor/test:adapter //motif/tensor/test:adapter_exact \
  //motif/tensor/test:adapter_layer_mix //motif/tensor/test:adapter_cohort \
  //motif/tensor/test:adapter_reference_test
```

`random_test.loom` executes the three published
[Random123 known-answer vectors](https://github.com/DEShawResearch/random123/blob/d8a0c25e49992685c0b9d91b553c1d04f81b6b4b/tests/kat_vectors)
for Philox4x32-10 through the VM reference provider: zero words, all-one words,
and mixed counter/key words. Every output word is checked exactly.

`random_gpu_test.loom` loads those same inputs from device buffers and compares
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

`sampling_test.loom` compares native categorical draws against an ordinal-rank
reference that directly counts each logit's predecessors. The reference
evaluates exponential-race scores with F64 logarithms and does not call the
production sampler's support-selection or random-transform helpers. Philox
counter generation shares the separately verified integer primitive. The
cases cross greedy and sampled requests, bounded and unbounded top-k, nucleus
filtering, four exclusion choices, and a 67-element vocabulary with ties.
A 248320-element case has an analytically known nineteen-token support, which
checks the production vocabulary extent without a quadratic full-size oracle.

`sampling_key_test.loom` compares every pair of runtime logit/ID records against
direct numeric and ID comparisons. Directed inputs cover finite extrema,
signed zero, normal/subnormal endpoints, and tied IDs through `INT_MAX`;
random logits cover ordinary orders. Complemented-ID recovery, the empty
candidate sentinel, output guards, and unchanged inputs are checked exactly.

`topk_test.loom` checks exact values and IDs against a serial strict-order oracle.
It covers the 248320-element vocabulary, multiple rows, a short nine-element
tail, tied maxima, and merging 61 disjoint partial lists. The complete
two-dispatch case expects the sixteen lowest IDs among equal winners from
separate tiles. Its benchmark declarations retain these different input and
merge shapes rather than reporting one synthetic size.

`linear_test.loom` checks eleven BF16 projection shapes against independently
authored whole-matrix products. Both checkpoint storage formats, short and
partial 256-input blocks, multiple blocks, and the 2560/6144-column model
reductions are covered. Reversed grid traversal checks that logical tile
origins reach contraction and publication unchanged. BF16 output in dense,
column-major, and padded layouts compares bitwise, including untouched holes.

Native F32 matrix results have an input-scaled accuracy requirement:
`atol = max(sum(abs(BF16 products))) / 2^20`, with zero relative tolerance.
The generator verifies this budget is below half the smallest nonzero product
in each case. This measures contraction accuracy without assuming native matrix
instructions reproduce ordered scalar accumulation bits. Two positive-term
constructions separately require exact F32 results that BF16 cannot represent.
Input conversion midpoints distinguish rounded weights from their F32 originals.
All immutable allocations and output guards compare bitwise.

```sh
python3 -B motif/tensor/test/generate_linear_oracle.py --check
```

`attention_expanded_test.loom` compares nineteen causal/noncausal grouped-attention
cases with whole binary64 QK contractions, a complete masked softmax, and
weighted values. Cases span sixteen-row tails around sixty-four-key panels,
176-row encoder geometry, the model's 32/8 and 20/20 head ratios, arbitrary
nonzero masks, empty leading/interior/trailing tiles, single visible keys,
and dense, transposed, and padded V storage. Full input allocations and output
guards compare bitwise under native, access, and race instrumentation.

The numerical budget uses centered operand magnitudes, twenty-bit native
contraction accuracy, F32 recurrence bounds, and an allowance of `2^-16` per
approximate exponential/reciprocal. Final BF16 rounding has an explicit
relative allowance. Single-key, zero, very-negative-score, and dedicated
probability cases compare bitwise; the latter remain away from output
rounding boundaries under the fixed arithmetic budget. Mutation checks
require sensitivity to causal visibility, masks, head grouping, centered-key
residuals, F32 rounding of an uncentered common logit, and replacing expanded
probabilities with one BF16 term. The reference never recreates the online
GPU recurrence or its shared-memory exchanges.

`attention_expanded_epilogue_test.loom` separately checks every returned row/channel
coordinate and the BF16 sigmoid/product boundary using independent scalar
expressions. Values include signed zero, and sigmoid conversions have a margin
against the approximate-math allowance. Omitting sigmoid rounding changes
expected bits. Every numerical and epilogue scenario has a benchmark.

```sh
python3 -B motif/tensor/test/generate_expanded_attention_oracle.py --check
```

`attention_shared_test.loom` checks both K16 and K32 against whole binary64
contractions and a dense masked softmax. Twenty-six scenarios cover complete
query/key tiles, multiple KV heads, the model's twelve-KV-head layout, arbitrary
nonzero mask bytes, leading/interior/trailing empty tiles, a single valid key,
late maxima, and finite logits below a conventional finite sentinel. General
accuracy includes explicitly budgeted BF16 probability conversion, native
contraction, F32 recurrence, and output/gate rounding. The budget is derived
from input magnitude, independently of observed device errors.

Single-tile probability cases instead round the unnormalized probabilities
explicitly and retain the unrounded denominator. Their probabilities and final
contexts stay away from BF16 midpoints under the fixed arithmetic budget, so
the BF16 outputs compare bitwise. Zero, single-key, and identical very-negative
score cases also compare bitwise. Mutation checks distinguish masking, KV-head
ownership, all score channels, and probability conversion.

`attention_shared_store_test.loom` checks the gate epilogue separately. Exact
references include both neighbors of BF16 context midpoints and gate values
whose sigmoid conversion remains unambiguous under approximate math. Omitting
either context or sigmoid rounding changes the expected bits. Ungated cases
pass output as the unused gate, matching the production caller. Full immutable
input storage and output guards compare bitwise with ordinary execution or
optional Loom address and race checks. Every scenario includes a benchmark
declaration.

```sh
python3 -B motif/tensor/test/generate_shared_attention_oracle.py --check
```

`segmented_attention_test.loom` uses an independent binary64 dense QK contraction,
whole-segment softmax, and weighted-value contraction. Twelve scenarios cover
segment lengths 1/2/3/12/31/32/33, multiple segments and heads, incomplete
four-wave groups, the twenty-head model layout, zero scores, signed values,
and a large common key offset. Single-key and segment-constant values compare
bitwise. Mutation checks require crossing segment boundaries and dropping a
channel contribution to change an observation beyond its declared budget.

Other outputs use an absolute accuracy budget based on the input magnitude,
132 F32 score operations, eight online operations per segment row, and a
`2^-16` relative allowance per approximate exponential/reciprocal stage. The
generator states the complete expression. The relative budget
`2^-7 / (1 - 2^-8)` accounts for rounding both observations to BF16. These
requirements are fixed independently of device results. Full immutable input
allocations and output guards compare bitwise under native, access, and race
instrumentation; every scenario also has a benchmark declaration.

```sh
python3 -B motif/tensor/test/generate_segmented_attention_oracle.py --check
```

`linear_coordinates_test.loom` is an authored scenario with eighteen grid
configurations and one GPU kernel. A command program specializes the grid and
group size, while `coordinate_reference_test.loom` independently enumerates
groups, columns, and rows in the VM. The reference does not reuse the production
division/remainder formulas. Configurations include every remainder of an
eight-row group, singleton and oversized groups, several complete groups,
model grids, and the maximum 1024×256 grid. Every coordinate pair, the input
header, and both storage guards compare bitwise. The benchmark uses the same
configuration domain. Run the authored scenario directly:

```sh
python dev.py test --target=//motif/tensor/test:linear_coordinates
```

`linear_temporal_coordinates_test.loom` uses the same authored scenario pattern
and shared VM reference for twenty-five logical grids, including both rounding
axes, every remainder of an eight-row group, the model's 4224/4608-row selections,
and the maximum 65536×16384 shape. Explicit expected tile counts keep the
reference independent of production rounding as well as division/remainder
mapping. Complete coordinate pairs, the input header, and guards compare
bitwise. Both coordinate scenarios accept the ordinary optional
`--config=loom-asan`; their production archives contain neither tests nor
reference code.

```sh
python dev.py test --target=//motif/tensor/test:linear_temporal_coordinates
```

`rmsnorm_test.loom` compares retained-row normalization with a sequential F64
sum of squares and Newton-refined inverse square root. Widths 1024 and 5120,
workgroups of 128 and 256 threads, and F32/F16 publication exercise distinct
register-bank and rounding paths. Multiple rows, nonzero input/output byte
origins, and concatenated output strides retain sentinels outside the live
rows. Epsilon-dominated and zero inputs cover the normalization denominator.
F32 results use a `1e-6` absolute and relative bound. F16 comparisons use
`1e-7 + 0.001 * abs(expected)` because F32 reduction can move a midpoint
across the final rounding boundary. Zero outputs are checked exactly.

The BF16 RMS corpus separates scalar rounding, reduction accuracy, and row
storage.

`rmsnorm_bf16_inverse_test.loom` compares seven whole-row inverse norms with a
binary64 sum and reciprocal square root. Its relative accuracy requirement
is `2^-16`. `rmsnorm_bf16_rows_test.loom` covers both epilogues across the same
shapes plus exact input/output aliasing. Widths 32, 64, 128, 2560, and 6144
exercise one or multiple values per lane; epsilon values include zero with
positive mean, `1e-6`, and `1e-5`. Zero and unit rows have bitwise outputs.

For other rows, the relative budget is
`(1 + 2^-16) * (1 + 2^-24)^2 * (1 + 2^-8)^n / (1 - 2^-8) - 1`,
with zero absolute tolerance. This accounts for the inverse requirement,
two F32 products, one BF16 boundary for zero-centered output or two for scaled
output (`n`), and the reference's BF16 publication. The scalar cases separately
verify the rounding boundaries exactly. These are workload accuracy
requirements, independent of a device's observed error.

All storage guards and immutable inputs are exact. Native, access-instrumented,
and race-instrumented execution cover reduction and whole-row ownership;
ordinary execution and optional address checks cover scalar arithmetic. Each
scenario has a benchmark declaration.

```sh
python3 -B motif/tensor/test/generate_rmsnorm_bf16_row_oracle.py --check
```

The rotary corpus distinguishes adjacent pairs with F32 tables from half-split
pairs with BF16 products. `rotary_scalar_test.loom` checks 304 exact pair outputs,
including cancellation, signed zeros, subnormals, and wide exponents. Omitting
any one of the four intermediate BF16 product roundings changes the corpus.

`rotary_rows_test.loom` checks all three row providers over six shapes each. Widths
64, 96, 128, and 256 cover one wave, a partially occupied last wave, and multiple
waves. Distinct token/head inputs and token coefficient rows detect table and
pair-index errors. Every output, storage guard, and immutable buffer compares
bitwise, including exact input/output aliasing. Native, access-instrumented,
and race-instrumented modes exercise the row cases.

The normalized cases derive a whole-row inverse from binary64 arithmetic,
then enclose the RMS corpus's `2^-16` inverse accuracy requirement with
outward-rounded F32 endpoints. The generator verifies that both endpoints
produce the same BF16 normalized tensor through the independent epilogue.
Monotonic scalar rounding therefore makes the expected tensor unique over
that interval. Exact rotary arithmetic can then detect cancellation errors
without a broad output tolerance. Scalar RMS cases independently exercise
normalization midpoints. No expectation depends on a device's observed output.

```sh
python3 -B motif/tensor/test/generate_rotary_oracle.py --check
```

The ConvRot reference constructs the dense fourth Kronecker power of an
explicit regular H4 matrix, independently of the production shuffle network.
Its dyadic input sums are exactly representable in F32. Rational rounding
then supplies the BF16 rotated values, F32 row scales, BF16 divisors and
quotients, nearest-even integers, and saturation.

`convrot_rows_test.loom` checks every one of the 256 basis vectors, uneven block
distribution across four subgroups, the real 6144/16384-column shapes, and
multiple rows. Zero, tiny, and large values exercise the scale floor and
range. Both signs and parities of integer midpoints occur, including a
negative quotient rounding to -128 and a positive one saturating to 127.
The generator verifies that omitting any intermediate BF16 boundary changes
the corpus. Signed-I8 outputs, F32 scales, all guards, and immutable input
compare bitwise under native, access, and race instrumentation.

`convrot_digits_test.loom` checks both supported digit distances directly against
dense H4 rows. It includes native and access-instrumented execution with exact
full-buffer observations. Every scenario has a benchmark declaration.

```sh
python3 -B motif/tensor/test/generate_convrot_oracle.py --check
```

`grouped_convolution_test.loom` uses a scalar F64 oracle that combines the static
and dynamic coefficients before multiplication and explicitly pads every
tap. It checks three independent eight-position, 5120-channel blocks,
ordinary and residual output, a 516-channel tail, and a single position with
four taps. The different branch and group sizes exercise coefficient
addressing, while bitwise input checks guard the separate-output contract.
All four shapes have benchmark declarations.

`pointwise_test.loom` checks all four BF16 arithmetic contracts through native
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

`image_attention_matrix_test.loom` compares complete image-attention heads with
binary64 dense contractions and a whole softmax. Thirteen scenarios span one
through nine query/key tiles, all 384 channels, zero and uniform scores,
conversion midpoints, signed values, maxima that increase in later tiles,
and finite scores below a conventional finite softmax sentinel.
General accuracy cases round Q/K/V independently and include an absolute
allowance for BF16 probability conversion, native matrix contraction, and
F32 normalization; the generator states the input-scaled budget.

Separate single-tile cases round unnormalized probabilities explicitly while
leaving the denominator unrounded. Their probabilities stay away from BF16
midpoints under the declared score/transcendental budget, making this boundary
independently observable. Uniform and value-conversion cases require twenty
significant binary bits relative to the largest input value, including means
that require more precision than BF16; zero output compares bitwise. The
generator verifies sensitivity to each Q/K/V/probability conversion, omitted
score/value channels, and premature BF16 output. Complete immutable inputs
and output guards are bitwise checks with ordinary execution or optional Loom
address and race checks. Every scenario includes a benchmark declaration.

```sh
python3 -B motif/tensor/test/generate_image_attention_oracle.py --check
```

`channel_norm_test.loom` covers affine and SiLU epilogues with disjoint and exactly
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

`convolution_test.loom` checks selected-pixel F32 convolution against independently
expanded and padded logical images. Active coefficients are authored before
their final temporal-plane placement; earlier planes contain distinct values.
Exact rational arithmetic rounds every F32 FMA, final bias, and leaf residual
addition separately. Directed cases distinguish fused products, bias before
the reduction, and reordered bias/residual addition.

Thirteen cases cover 1×1, 3×3, and 5×5 extents; one and multiple temporal
planes; nearest-neighbor scales one, two, and three; rectangular images;
complete, tail, and multiple workgroups; 384 input or output channels; and
the decoder's final 96-to-3 convolution. Output may alias the residual in
the wrapper. Every output bit, storage guard, and immutable input is checked
in both native and access-instrumented execution. Each shape also has a
benchmark declaration.

```sh
python3 -B motif/tensor/test/generate_convolution_oracle.py --check
```

`convolution_matrix_test.loom` checks the cooperative BF16/F32 3×3 implementation
against independently expanded, padded logical images and exact rational
BF16 quantization. Nine cases cover partial and complete 32-pixel tiles,
multiple channel tiles, unequal input/output channels, reductions through
384 inputs, nearest scales one/two/three, one/two/three temporal planes,
residual aliasing, and positive/negative BF16 midpoints with both parities.
All 128 threads enter the implementation even for a single live pixel.

Native matrix arithmetic may differ from exact F32 sums even when every
mathematical partial sum is representable. The accuracy requirement is twenty
significant binary bits relative to the input contribution scale, twelve bits
beyond the BF16 operands. For each scenario, the generator computes the largest
sum of absolute quantized products plus absolute bias/residual contributions;
the absolute error budget is that scale divided by `2^20`, with zero relative
tolerance and no accepted NaNs. This bounds error through cancellation without
substituting hardware-derived expected values. Every budget is less than half
the smallest ordinary nonzero product, so losing one such product fails.

Two directed cases remain bitwise: an accumulated result that F16 cannot
represent and a cancellation that distinguishes residual addition before or
after rounded bias. Every storage guard and immutable input is also exact.
Normal, access-instrumented, and race-instrumented execution share the cases;
each shape has a benchmark declaration. Scalar and matrix tests share only
guarded fixture storage and comparison authoring, with separate arithmetic
references and launch bodies.

```sh
python3 -B motif/tensor/test/generate_convolution_matrix_oracle.py --check
```

`time_test.loom` compares sinusoidal features with binary64 exponential and
trigonometric evaluation after explicit F32 frequency and phase boundaries.
The independent reference constructs whole frequency rows, then concatenates
cosine and sine halves. Expected values keep F32 precision so the check
observes BF16 publication error. Nine cases cover the model's complete
256-feature grid, negative and small times, alternative periods and scales,
one/odd/129-frequency halves, channel tails, several launch widths, and
logical rows shorter than physical storage.

The error requirement separates BF16 rounding (`abs(expected) / 256`) from
approximate F32 phase and transcendental evaluation. The latter's absolute
budget is `(largest_abs_phase + 1) / 2^20`: sixteen F32 unit roundoffs per
unit phase, plus the same absolute floor. This is an accuracy requirement,
not a target-specific expected result. Positive zero time and zero scale
are checked bitwise. Padded rows must independently contain exact cosine
ones and sine zeros; native access instrumentation checks that their
construction does not read beyond the logical input. Input bytes and output
guards remain exact, and every shape has a benchmark declaration.

```sh
python3 -B motif/tensor/test/generate_time_oracle.py --check
```

`prefix_norm_test.loom` compares native NPU output with a serial binary64 sum of
squares and reciprocal norm. Widths 32, 64, 96, and 1024 exercise prefix lengths
on either side of 16- and 32-element boundaries, including single-element and
complete prefixes. Zero and floor-dominated rows cover the normalization
denominator. Nonzero buffer origins, exact guards, and unchanged input checks
retain storage ownership; the complete output row checks that the selected
prefix controls the scale without suppressing later lanes. The error bound is
`2e-7 + 2e-6 * abs(expected)`. Benchmark declarations retain each row width;
their execution is a workload smoke check, with no throughput claim.

`matrix_rdna_test.loom` and `matrix_cdna_test.loom` bind the same external template
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

`matrix_publication_test.loom` checks native wave32 F32 fragments published to BF16
without running a matrix contraction. Complete, partial, and wholly inactive
tiles share one scalar row/column oracle. Six geometries cross both tile edges;
rounding midpoints exercise both publication paths, and prefix/suffix guards
and the immutable input compare bitwise. Native GPU execution launches nine
workgroups per case, with at most 33 rows and 37 columns. Its benchmark is
selected separately through the repository benchmark workflow.
