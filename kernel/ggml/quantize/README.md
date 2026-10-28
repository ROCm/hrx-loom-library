# GGML Quantization Kernels

This package contains launchable quantization entry points matching GGML
representations and operation contracts. The physical layouts and reusable
encode/decode bodies live in [`motif/format/ggml/`](../../../motif/format/ggml/).

Quantization kernels own the externally visible conversion semantics, shape
contract, launch configuration, and buffer ABI. Format motifs own byte layout,
block access, and reusable encode/decode computation. Keeping that boundary
explicit lets compatibility entry points and native fused kernels share the
same representation code without sharing an execution contract.

The production source documents its upstream reference and ABI. Adjacent
[`q8_1_x4_f32_test.loom`](q8_1_x4_f32_test.loom) owns the nonzero correctness
cases and representative benchmarks through a `loom_test` dependency on the
production library. BUILD declarations keep those test inputs out of reusable
archives and model closures, with target coverage independent of execution.

The [quantization checks](test/) add exact byte-level oracles for rounding,
scale/sum metadata, and physical ordering through the production launch ABI.
Small guarded fixtures exercise native, access, and race execution without
model parameters.
