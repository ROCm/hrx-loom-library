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

`prefix_norm.loom` compares native NPU output with a serial binary64 sum of
squares and reciprocal norm. Widths 32, 64, 96, and 1024 exercise prefix lengths
on either side of 16- and 32-element boundaries, including single-element and
complete prefixes. Zero and floor-dominated rows cover the normalization
denominator. Nonzero buffer origins, exact guards, and unchanged input checks
retain storage ownership; the complete output row checks that the selected
prefix controls the scale without suppressing later lanes. The error bound is
`2e-7 + 2e-6 * abs(expected)`. Benchmark declarations retain each row width;
their execution is a workload smoke check, with no throughput claim.
