# Sampling correctness and benchmark workloads

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
