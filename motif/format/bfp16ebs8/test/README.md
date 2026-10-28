# Native BFP16EBS8 packing checks

An independent rational-arithmetic encoder supplies exact high and residual
records. Its 24 input blocks cover zero and signed zero, subnormals, both signs,
half-way parity, positive carry, the -128 endpoint, shared exponent spreads,
and normal exponents from 1 through 254 with finite reconstructed values.
`generate_oracle.py` refreshes the checked-in bytes without a native encoder,
model checkpoint, or third-party Python dependency.

Thirteen native trials select 0, 1, 2, 7, 8, 9, 15, 16, 31, 32, 33, 63, and
64 complete blocks. They exercise FIFO boundaries, repeated record ordering,
nonzero origins, and both packing paths. The complete 18 KiB allocation is
compared exactly, covering input and identity preservation, output payload,
padding, inactive capacity, and patterned guards. CPU code constructs the
oracle; an explicit NPU pipeline establishes local-memory and core-assembly
ownership for the actual encoder. The benchmark declaration uses these same
block-count regimes as execution smokes.

From the repository root:

```sh
python motif/format/bfp16ebs8/test/generate_oracle.py
loom-format --in-place motif/format/bfp16ebs8/test/aie2p_test.loom
bazel test --config=asan \
  //motif/format/bfp16ebs8/test:aie2p_execute_xdna_hardware_test
```
