# Maintain library CI

The library tests the installed `hrx-loom` product. Its
[workflow](../.github/workflows/ci.yml) does not build the compiler, runtime, or
SDK. The [SDK setup action](../.github/actions/setup-sdk/action.yml) installs one
qualified Linux Release wheel, verifies its SHA-256, and selects it with
`dev.py setup --sdk`. `MODULE.bazel` supplies the matching public build rules.
The host job rejects any C/C++ compilation or linking in the library graph.

Release is the normal compiler/runtime profile. ASAN is an explicitly selected
SDK for diagnosing native memory errors; Bazel's `--config=asan` cannot change
an installed Release wheel. Loom's device access and race instrumentation are
separate execution profiles and can run with the Release SDK.

## Jobs and ownership

| Job suffix | Work |
| --- | --- |
| `Host` | Source policy, formatting, linking, model-card checks, generated oracles, and CPU/VM tests. |
| `AMD gfx942`, `AMD gfx110x`, `AMD gfx1151`, `AMD gfx120x` | Native AMD GPU correctness, instrumentation, and benchmark-smoke tests. |
| `AMD gfx1150 GPU + NPU4` | The GPU checks plus native XDNA tests, with both device classes exposed. |

Hardware jobs are advisory while the runner/model combinations are qualified.
Jobs select tests through the public `iree-run-requirement` tags. Host runs
tests without special resource requirements; GPU and NPU jobs run profiles
whose required resources are all available, including instrumented profiles.
Source and compile checks remain in Host. Each lane tests `//...` with
`--build_tests_only` and one `--test_tag_filters` argument. The CI resource
selector reads the actual query tags, excludes every unavailable resource,
and includes hardware tests only
when the lane declares hardware. The matrix lists available resources; adding
a requirement cannot silently admit a test onto a runner missing that resource.
Wildcard selection preserves configured incompatibility skips, such as the
source-only request API check described below.

Adding a model or motif with an existing resource requires no workflow edit.
A new device family needs an SDK containing its runtime, an appropriate runner, and its execution
profile in the matrix. Full learned-model checks stay manual and use externally
provisioned assets; no model weights are downloaded by these CI jobs.

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

The pinned SDK includes native AMD GPU and XDNA execution but lacks the Vulkan
HAL. A Vulkan lane requires enabling that HAL in HRX's `loom-sdk` configuration,
qualifying and publishing the resulting wheel, and then adding a Vulkan runner
here. Windows likewise requires a qualified Windows wheel at the selected HRX
revision. A Linux cross-build input archive does not establish native Windows
qualification. Neither lane is represented by a source-built substitute.

## Cache and diagnostic evidence

Each lane restores a persistent Bazel disk cache and saves its new action and
test results, including successful actions from failed runs. Snapshot keys use
run identity so new source changes can extend the cache; restore prefixes stay
stable across commits. Hardware lanes have separate cache namespaces. Kernel,
driver, device, and firmware identity also enters the Bazel test environment,
so changing that environment invalidates execution results. Test caching remains
enabled; run identity is not a test input.

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
