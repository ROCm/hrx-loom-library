# Vision component qualification

These tests and benchmarks call the shared vision templates directly, without
model providers, checkpoints, or a serving process. Test launch adapters supply
geometry and storage types; the numerical references use independent scalar
arithmetic rather than the implementation's matrix fragments or tile layouts.
New precision, target, and shape coverage belongs here once for all consumers.

| Target | Independent reference and coverage |
| --- | --- |
| `patch` | Coordinate-coded RGB8, signed sparse temporal convolutions, last-K taps, and a nonzero source origin. F16/BF16 compute crosses half/F32 weight storage at 4, 64, and 72 patches. |
| `normalization` | Analytical spatial planes and population moments cover interpolation, merge order, affine normalization, and zero variance. Minimum, square, landscape, and portrait geometries share one fixture. |
| `key_value` | Signed sparse products and scalar 2D rotary check every head, concurrent cohort banks, and transposed values at 4, 32, and 36 patches in both half formats. |
| `attention` | Scalar rotary/dots and complete-row softmax check nonuniform attention. A separate sparse projection oracle isolates output errors. Both formats cover 4, 64, and 72 patches. |
| `ffn` | Analytical statistics, signed three-tap projections, and scalar GELU check intermediate activation and residual output at 4, 64, and 68 rows, including the 4304-column reduction tail. |
| `merger` | Two signed projections and scalar GELU check 1, 64, and 70 merged rows. Both half formats run at both 2560 and 5120 feature columns. |
| `workspace` | Literal byte counts cover allocation padding, output-backed residual thresholds, both feature widths, maximum geometry, and all resident attention cohort counts. No represented payload is allocated. |

Numerical tests run native and access-instrumented AMD profiles. Minimum, full,
and partial tiles cover distinct specialization paths. Fixtures retain nonzero
output origins and allocation guards; access instrumentation checks the active
view boundaries for every shape. Independent half conversion and GELU helpers
in `reference.loom` are private test roots, never production dependencies.
Precision-specific tolerances remain next to their assertions.

```sh
bazel test --config=asan --config=amdgpu //motif/vision/qwen3_8/test:all
python dev.py benchmark --target=//motif/vision/qwen3_8/test:all \
  --config=amdgpu --device=amdgpu://0 --output-dir=.reports/benchmark/vision
```

Benchmark declarations use the same bounded component workloads. CI benchmark
smoke establishes execution, not performance. Measurements use the Release SDK
or optimized source configuration and the qualified measurement route described
in the [development guide](../../../../docs/development.md).
Image packets, cohort policy, and GPU filtering are qualified together in the
[shared text-program tests](../../../../program/text/qwen3_8/test/README.md).
