# Maintain library CI

The library tests the installed `hrx-loom` product. Its
[workflow](../.github/workflows/ci.yml) does not build the compiler, runtime, or
SDK. The [SDK setup action](../.github/actions/setup-sdk/action.yml) installs one
qualified Linux Release wheel, verifies its SHA-256, and selects it with
`dev.py setup --sdk`. `MODULE.bazel` supplies the matching public build rules.
The host job rejects any C/C++ compilation or linking in the library graph.

Release is the normal compiler/runtime profile. ASAN is an explicitly selected
SDK for diagnosing native memory errors; Bazel's `--config=asan` cannot change
an installed Release wheel. Independent Loom flags select JIT instrumentation
on the same test targets: `--config=loom-asan`, `--config=loom-tsan`, or
`--config=loom-ubsan`. These flags instrument generated programs even when the
SDK itself is a Release wheel. Native sanitizer configurations alone do not
enable Loom checks. Ordinary CI leaves Loom instrumentation disabled.

## Jobs and ownership

| Job suffix | Work |
| --- | --- |
| `Host` | Source policy, formatting, linking, model-card checks, and CPU/VM tests. |
| `AMD gfx942`, `AMD gfx110x`, `AMD gfx1151`, `AMD gfx120x` | Native AMD GPU correctness scenarios. |
| `AMD gfx1150 GPU + NPU4` | The GPU checks plus native XDNA tests, with both device classes exposed. |

Hardware jobs are advisory while the runner/model combinations are qualified.
Jobs select tests through the public `iree-run-requirement` tags. Host runs
tests without special resource requirements; GPU and NPU jobs run profiles
whose required resources are all available.
Source and compile checks remain in Host. Each lane tests `//...` with
`--build_tests_only` and one `--test_tag_filters` argument. The CI resource
selector reads the actual query tags, excludes every unavailable resource,
and includes hardware tests only
when the lane declares hardware. The matrix lists available resources; adding
a requirement cannot silently admit a test onto a runner missing that resource.
Wildcard selection preserves configured incompatibility skips, such as the
source-only request API check described below.

GPU profiles serialize local execution with `exclusive-if-local` and
the shared `gpu-device` resource group. Build concurrency remains independent.
The hardware job also resolves the runner's `ROCR_VISIBLE_DEVICES` or
`PARENT_GPU_DEVICES` allocation to physical PCI addresses, then selects the
addresses supported by the native HAL. An allocation may include older GPUs
that the HAL does not enumerate; the selected set must still contain at least
one allocated GPU. ROCr and native HAL ordinals can differ.
The preflight uses KFD/sysfs and the installed SDK's query-only `--dump_devices`;
it needs no ROCm runtime. Explicit native selectors must stay within the
allocation, and a multi-GPU host without an allocation fails preflight.

The job verifies the filtered device lists and forwards `HAL_VISIBLE_*_DEVICES`
into Bazel tests, including empty lists for excluded device families. Its
`gpu-selection.json` artifact records the physical identities, selectors, and
allocated PCI addresses unavailable to the native HAL. Those selections also
enter the test cache identity. Local serialization alone cannot prevent
separate runner jobs from using the same physical GPU.

Adding a model or motif with an existing resource requires no workflow edit.
A new device family needs an SDK containing its runtime, an appropriate runner, and its execution
profile in the matrix. Full learned-model checks stay manual and use externally
provisioned assets; no model weights are downloaded by these CI jobs.

Stress sources end in `_stress.loom`, and their tests carry both `manual` and
`stress`. Ordinary wildcard runs exclude them. An explicit stress run selects
their labels by the `stress` tag, with the same optional sanitizer configuration
as ordinary tests. Benchmarks remain separately available through
`dev.py benchmark`; correctness runs do not execute an additional benchmark loop.

The Qwen internal request-API check remains automatic in source mode, as
[described in the development guide](development.md#build-and-check). It is
incompatible with installed SDK mode rather than rebuilding serving internals.

## Update the SDK

The [SDK pin](../build_tools/ci/sdk.json) records the HRX commit, qualified
workflow artifact, wheel filename, and wheel digest. Updating the compiler is
one reviewed change to that pin and the HRX commit in `MODULE.bazel`. Setup
rejects mismatched revisions. Select an artifact exported after installed-wheel
qualification (`loom-sdk-wheel-*`), not the Windows input handoff
(`loom-sdk-inputs-*`). The action installs only the recorded wheel and fails if
its content differs; there is no latest-success or source-build fallback.

Workflow artifacts have finite retention. They bootstrap qualification before
an SDK release exists; a missing or expired artifact requires a new reviewed
pin. Durable library publication uses a published SDK release asset and its
recorded checksum, replacing the download step in the single setup action.
[Maintaining Loom SDK releases](https://github.com/ROCm/hrx-system/blob/main/loom/packaging/python/RELEASING.md)
owns production, platform qualification, and publication of those artifacts.

The pinned SDK includes native AMD GPU, XDNA, and Vulkan execution. Vulkan
loads the execution host's Vulkan loader and driver; the wheel does not bundle
them. A Vulkan lane requires a qualified runner with that runtime and the
matching execution resource in the matrix. Windows likewise requires selecting
a qualified Windows wheel at the pinned HRX revision. A Linux cross-build input
archive does not establish native Windows qualification. Neither lane is
represented by a source-built substitute.

## Cache and diagnostic evidence

CI uses Bazel's terse test summary: failures and aggregate counts remain visible
without a row for every cached success. Tool diagnostics and failed-test logs
remain enabled. Authoring commands omit successful subprocess echoes; use
`python dev.py --verbose test` to include their complete command lines. Failed
commands are printed regardless of verbosity, with their original output and
exit status. Formatting reports come from `loom-format` itself.

Each lane restores a persistent Bazel disk cache and saves its new action and
test results, including successful actions from failed runs. Snapshot keys use
run identity so new source changes can extend the cache; restore prefixes stay
stable across commits. Hardware lanes have separate cache namespaces. Kernel,
driver, device, and firmware identity also enters the Bazel test environment,
so changing that environment invalidates execution results. Test caching remains
enabled; run identity is not a test input.

CI uses `--nobuild_runfile_links` to defer runfiles-tree creation until execution.
Cached tests need no filesystem tree of interpreter, SDK, and data symlinks.
Bazel still tracks the complete runfiles manifest and creates the execution
environment for cache misses and `bazel run`.

Bazel hashes individual SDK inputs, linked modules, cards, acquisition locks,
and test/oracle inputs. Replacing a wheel does not invalidate a test merely
because its archive checksum changed. Native tests still JIT their modules, so
a changed compiler library correctly invalidates those tests even if it would
produce identical code. Removing that dependency requires separating the
compiler action from execution and making its complete compiled product the
test input; hiding a compiler dependency from the current JIT test is incorrect.

Job artifacts retain Bazel profiles, selected resource filters, and hardware
identity. Compare action execution with cache hits before attributing a job's
wall time to compilation. Correctness benchmark-smoke timings are not isolated
performance measurements; the [benchmark workflow](development.md#sweep-repository-benchmarks)
owns deliberate measurements.
