# Develop a reusable component

A shared component owns a contract that its production callers already need.
The [layer boundaries](../CONTRIBUTING.md#admission-surfaces) determine where
that contract belongs. This guide follows two examples from the library and
shows how to qualify changes without downloading a model checkpoint.

## Start at a real caller

The Qwen Q6 matrix implementation consumes the
[GGML Q6_K motif](../motif/format/ggml/q6_k.loom) to read codes and coefficients.
The motif owns the 210-byte record layout, signed code interpretation, and
coefficient precision. Its caller supplies byte origins and record/group/packet
coordinates. The [matrix kernel](../model/Qwen/Qwen3.8-27B/program/kernel/linear_q6k_f16_wmma_metadata.loom)
owns tiling, metadata reuse, synchronization, and F16 staging. Sharing record
access does not require sharing that schedule or introducing a new launch ABI.

The Qwen [storage program](../program/text/qwen3_8/storage.loom) owns another
kind of contract: choosing packet capacity from the service's shape directory
and isolated prefill capacity. Both text models call the same policy before
allocating storage. Model preparation still owns checkpoint names, stage roots,
and model-specific storage. A shared program can express this policy without
depending back on either model.

For a proposed extraction, trace the actual arguments and results: units,
layout, valid ranges, precision, aliasing, lifetime, and the facts available for
specialization. The caller supplies those facts explicitly. Public symbols
describe the shared contract; private helpers remain implementation details.
Existing package providers and vocabulary are the first place to look for an
owner, before adding a module or public abstraction.

Similar names do not establish interchangeable semantics. Layer normalization
may differ in accumulation precision, variance convention, epsilon placement,
and whether a checkpoint stores gamma or gamma minus one. A family inventory
records those differences and identifies which forms share a mechanism. A
structural extraction preserves rounding, memory traffic, and execution shape;
changes to those properties require their own numerical and performance proof.
New shared execution strategies start from a measured oracle and an explicit
data-reuse budget, not from the easiest scalar reference to extract.

## Put the contract under small independent checks

The [Q6 buffer cases](../motif/format/ggml/test/q6_buffer.loom) exercise seven
trials over two padded rows of three records. They cover signed scale extrema,
both alignments of a 210-byte record, byte/lane order, and guarded outputs.
Independent scalar addressing and coefficient arithmetic check the production
vector readers. The test publishes codes, coefficients, and rounded values
separately, so an error cannot cancel between decoding and scaling.

The [packet-capacity cases](../program/text/qwen3_8/test/packet_capacity.loom)
use literal expected results and at most three shape records. Odd/even rounding,
empty catalogs, the selected prefix, and competing capacities are distinct
cases. The VM execution profile exercises the real function without a GPU or
model parameters. Existing model storage tests separately exercise its consumer.

A compact test set follows semantic boundaries rather than a Cartesian product
of sizes. One nontrivial stride, a tail, an extreme value, and an aliasing case
can expose different bugs; hundreds of random dimensions may exercise the same
path. Random trials supplement explicit boundary cases. Numerical references
retain their independence from the implementation being checked, and numerical
tolerances follow the operation's precision contract.

Each ordinary `loom_test` declares its production dependencies and an execution
profile. For example, packet capacity needs only:

```starlark
loom_test(
    name = "packet_capacity",
    srcs = ["packet_capacity.loom"],
    execution_profiles = [VM_REFERENCE_PROFILE],
    deps = ["//program/text/qwen3_8:storage"],
)
```

The [complete BUILD file](../program/text/qwen3_8/test/BUILD.bazel) includes the
loads. A focused iteration selects the owning targets:

```sh
bazelisk test --config=asan --config=amdgpu \
  //motif/format/ggml/test:q6_buffer \
  //program/text/qwen3_8/test:packet_capacity
```

## Keep representative performance workloads available

`check.benchmark` names the workloads that matter for measurement. Small cases
can serve both correctness and benchmark smoke coverage. Production-size
workloads remain useful when they expose different occupancy, reuse, or memory
behavior, even when ordinary regression checks need much smaller inputs.

The [Qwen metadata tests](../model/Qwen/Qwen3.8-27B/test/README.md) keep all
asserting cases in their ordinary target. A separate `q6_metadata_performance`
target carries the full-width launch-only workloads and the standard `manual`
tag. Explicit benchmark selection still discovers it; ordinary CI excludes it.
This uses the same rules and profiles, without a special runner or workflow.
Removing a large workload from presubmit is justified only when the retained
checks still establish its correctness contract.

The [benchmark workflow](development.md#sweep-repository-benchmarks) explains
target selection, completion modes, artifact reuse, and reports. Measurements
use a Release SDK or source `--config=opt` on a qualified, controlled machine.
Shared-runner test durations describe test cost, not a performance regression
threshold. Planning time, JIT compilation, fixture allocation, and execution
all contribute to that cost.

## Integrate the consumer with the component

A complete change includes the shared implementation, actual callers, focused
checks, package documentation, and explicit provider dependencies. Model source
maps describe the changed program closure; asset locks change only when their
asset dependencies change. Existing source, archive, and mixed packages should
still link through those providers; their builds do not fetch weights.

For a pure extraction, compare the actual emitted products as well as numerical
outputs. Identical executable bytes provide strong evidence that a new source
boundary preserved specialization and scheduling. A changed product needs an
explanation; a passing numerical check alone cannot rule out added traffic,
spills, or synchronization. The
[compiler comparison workflow](development.md#compare-repository-compiler-evidence) covers
target-qualified products; model test captures can also expose runtime-specialized
kernels that are absent from a package's static target products.

The [repository admission sequence](../CONTRIBUTING.md#local-admission-sequence)
checks source policy and reachability in addition to the focused tests. Component
checks establish their own contracts; complete-service checks remain responsible
for the model composition and user-visible outputs.
