# Develop the library

This guide covers reusable Loom components and the tools for checking their
correctness and cost. For a complete model package, start with
[model authoring](model-authoring.md). Commands run from the repository root.

## Select a compiler

Library development uses Git, Python 3.12+, Bazelisk, and an installed
[`hrx-loom` SDK](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/).
Bazelisk selects the Bazel version in `.bazelversion`. Select the SDK installed
in the Python environment running `dev.py`:

```sh
python dev.py setup --sdk
```

`dev.py` normally finds `bazelisk`, then `bazel`, on `PATH`. Environments with
a required build launcher can select it with `BAZEL=/path/to/launcher`.
`BAZEL` names one executable, without shell arguments; an invalid selection
fails instead of falling back to another client. Exporting it also selects
the launcher for developer commands invoked by the repository's Git hooks.

`--sdk=/path/to/venv/bin/python` selects another environment. Bazel consumes
that installation's native commands, shared libraries, and Python tools as
declared inputs. Building a model with this toolchain does not rebuild HRX's
native compiler or runtime. The HRX revision in `MODULE.bazel` still supplies
the public Bazel rules and their Python build/test helpers. SDK mode disables
optional C++ toolchain discovery, so authoring Loom programs does not require
a host C++ compiler.

The wheel selects native optimization and instrumentation. An ASAN SDK uses
the `+asan` wheel; `--config=asan` instruments tools built from source and does
not change an installed Release SDK into an ASAN build.

To develop the compiler and library together, use the
[HRX source-build prerequisites](https://github.com/ROCm/hrx-system/blob/main/BUILDING.md)
and select an HRX checkout:

```sh
python dev.py setup --hrx-source=/path/to/hrx-system
```

The selection lives in the ignored `.bazelrc.hrx`; it does not change the
published module dependency. Uncommitted compiler edits become inputs to the
next library build through the same Bazel graph. `--pinned-source` builds the
tools from the revision in `MODULE.bazel`; `--sdk` returns to the installed
package. `.bazelrc.local` remains available for other local settings. The model
BUILD files and public Loom rule contracts are the same in each mode.

With Lefthook installed, `python dev.py hook` enables the repository's
pre-commit hook. It checks source ownership in the Git index, then formats
staged `.loom` files, stages their canonical form,
and stops after any rewrite so the result can be reviewed. It refuses
partially staged Loom files. It also checks generated model-card compiler lists
against Bazel analysis; [model authoring](model-authoring.md#derive-compiler-files-from-provider-dependencies)
describes the check/fix workflow and staged-input requirements.

## Repository layers

| Path | Responsibility |
| --- | --- |
| [`model/`](../model/) | Model identity, parameter interpretation, and model-specific execution. |
| [`program/`](../program/) | Service policy and executable compositions shared across models. |
| [`kernel/`](../kernel/) | Concrete launchable kernels with tests and benchmarks. |
| [`motif/`](../motif/) | Reusable representations and algorithms without a launch interface. |
| [`target/`](../target/) | Repository-specific target or execution policy. |
| [`build_tools/`](../build_tools/) | Setup, build integration, and repository policy. |

Dependencies flow toward reusable components; those components do not depend
back on models. Motifs receive configuration facts from their callers. Global
`config.*` operations and concrete target records belong in composition leaves.
For example, the [GGML Q8_1 x4 representation](../motif/format/ggml/) is separate
from the [quantization kernel](../kernel/ggml/quantize/) that launches it.
The [contribution contract](../CONTRIBUTING.md) defines the full layer and
correctness requirements.

## Build and check

The repository wrappers add formatting, source policy, explicit dependencies,
correctness cases, benchmark plans, and target profiles to Loom's public rules:

```sh
python dev.py format --check
python dev.py model-card --check
python dev.py lint
python dev.py test
python dev.py build
```

For focused iteration, select a package and its execution configuration:

```sh
python dev.py test --target=//kernel/ggml/... -- --config=asan --config=amdgpu
```

`dev.py test` first checks formatting across the repository, including
experimental sources, public Loom authoring lint, repository source policy,
and managed model-card compiler lists,
then runs the selected tests. `format --fix` rewrites
sources through the selected compiler and verifies the result; passing paths
limits formatting to those files. Direct `bazelisk test` invocations also retain
the generated per-library checks. Hardware execution requires a matching device
and runtime; a skipped hardware test supplies no execution evidence.

The Qwen request-ownership integration check links HRX's native serving test
API. Source mode includes it in the ordinary test suite; installed SDK mode
marks it incompatible because that API is not a distributed tool. The shared
Loom policy checks run in both modes, and the model service checks exercise
the selected `loom-serve` through HTTP. This keeps internal API qualification
available without rebuilding the compiler behind an installed SDK selection.

`loom_motif_library` emits linkable bytecode and checks that its sources are
non-launchable. `loom_kernel_library` adds concrete kernel products, target
profiles, and compiler reports. `loom_test` owns correctness cases and benchmarks
in a separate module that depends on production providers. The [Bazel rule guide](../build_tools/bazel/)
explains their declarations; the [public Loom build guide](https://rocm.github.io/hrx-system/loom/workflows/build-with-bazel/)
owns the underlying toolchain and linker behavior.

The [CI maintenance guide](ci.md) describes SDK pins, runner coverage, and
cache ownership.

## Check a kernel and inspect its cost

Build a quantization kernel together with its motif dependencies and test module:

```shell
bazelisk build --config=opt \
  //kernel/ggml/quantize:q8_1_x4_f32_checks_benchmark_module
LOOM_MODULE="$(realpath bazel-bin/kernel/ggml/quantize/q8_1_x4_f32.loombc)"
LOOM_LIBRARY="$(realpath bazel-bin/motif/format/ggml/q8_1_x4.loombc)"
LOOM_TEST_MODULE="$(realpath bazel-bin/kernel/ggml/quantize/q8_1_x4_f32_checks_module.loombc)"
```

The resolved paths keep these inputs stable when a later tool invocation changes
Bazel configuration. Run every public correctness case on an AMD GPU:

```shell
bazelisk run --config=asan --config=amdgpu \
  @loom_sdk//:loom-test -- \
  "$LOOM_TEST_MODULE" \
  --device=amdgpu://0 \
  --sanitizer=access
```

The native AMD driver uses libamdf. It requires a supported operating-system
device driver and permission to open the GPU; no ROCm SDK installation is
required.

Benchmark a declared workload from the same module closure:

```shell
bazelisk run --config=opt --config=amdgpu \
  @loom_sdk//:loom-benchmark -- \
  "$LOOM_TEST_MODULE" \
  --device=amdgpu://0 \
  --benchmark=@ggml_quantize_q8_1_x4_f32_decode \
  --measure=dispatch_complete \
  --batch-size=64 \
  --profile-final-batch=true \
  --output=/tmp/q8_1_x4_f32_decode.json

jq '{
  summary,
  benchmarks,
  work_items: [
    .work_items[] |
    {benchmark, state, correctness, measurement, profiled_dispatch_timing}
  ]
}' /tmp/q8_1_x4_f32_decode.json
```

`dispatch_complete` measures execution through device completion. The benchmark
result also contains the final profiled device distribution and the structured
compile report for the specialized executable.

Repository-scale generated evidence lives below `.reports/`. Git and Bazel both
exclude this local output root, and each report producer owns a child namespace
so its schemas and lifecycle remain independent. Benchmark sweep workspaces use
`.reports/benchmark/<profile>/`; compiler comparison workspaces default to
`.reports/compile/`.

### Sweep repository benchmarks

`dev.py benchmark` discovers every test module admitted to the
repository benchmark corpus, uses HRX's linked root-owned test module plus the
public benchmark runner from the selected SDK or HRX source checkout, and
maintains a local sweep workspace:

```shell
SWEEP_ROOT=.reports/benchmark/gfx1100
python dev.py benchmark \
  --config=amdgpu \
  --device=amdgpu://0 \
  --output-dir="$SWEEP_ROOT" \
  -- \
  --measure=dispatch_complete \
  --batch-size=64 \
  --profile-final-batch=true
```

Repeating the command is incremental. Bazel rebuilds only invalidated actions,
and the sweep executes only roots whose closed benchmark `.loombc`, benchmark
runner and its shared libraries, or invocation identity changed. A shared motif
edit therefore reruns the exact consumers whose linked runner input changed;
an edit that produces a byte-identical module is reported as potentially affected
without repeating the physical benchmark.
`--rerun-all` deliberately captures fresh measurements for every selected
module. A changed benchmark runner invalidates every module in the selected
scope, so compiler development can use a narrow target pattern while ordinary
`.loom` authoring retains artifact-level reuse.

Configured `cquery` edges map every module back to its exact `.loom` source
closure. `latest.json` records that graph together with changed sources,
artifact digests, affected/executed/reused state, benchmark names, and native
result paths:

```shell
jq '[
  .modules[] |
  select(.potentially_affected) |
  {label, changed_sources, artifact_changed, state}
]' "$SWEEP_ROOT/latest.json"

jq '[
  .modules[] |
  select(.executed) |
  {label, benchmarks, work_item_count, results}
]' "$SWEEP_ROOT/latest.json"
```

Each invocation also writes an immutable manifest under `runs/`. Native
`loom-benchmark` bundles live below `artifacts/` and retain their public
`results.json` or `results.jsonl` schema; unchanged modules point back to the
matching earlier bundle instead of copying or reinterpreting it. The mutable
`index.json` keeps prior module evidence available across targeted subsets.
The workspace is rooted at
`--output-dir`, not in Bazel's output tree, so build-cache cleanup does not erase
measurement history.

`--target` accepts Bazel target patterns. Recursive scopes select admitted
benchmark modules below a package, exact test labels select one module, and
repeating the option forms a union:

```shell
python dev.py benchmark \
  --config=amdgpu \
  --device=amdgpu://0 \
  --output-dir="$SWEEP_ROOT" \
  --target=//kernel/ggml/... \
  --target=//motif/format/ggml/test/... \
  -- \
  --measure=dispatch_complete
```

Arguments after `--` are passed directly to `loom-benchmark`.

These captures are comparable when they come from the same stable machine,
software configuration, measurement policy, and exclusive-use discipline.
The repository correctness matrix intentionally makes no performance claim.

### Compare repository compiler evidence

Compare every target-qualified compilation in the current working tree with an
exact Git base:

```shell
python dev.py compile-report --base=origin/main
```

The current tree may be dirty. The command does not commit, stash, check out, or
rewrite it. The base reference is resolved once to an exact commit and exported
into a persistent archived source workspace below `.reports/compile/`.
The live tree and archived base keep independent Bazel output bases while
sharing Bazel's disk action cache, so matching links and target compilations are
reused safely across both workspaces.

A local HRX checkout selected by `python dev.py setup --hrx-source=...` is
applied to the archived base as well. This isolates library-source changes while
both sides use the same uncommitted compiler. The comparison records the public
report tool's implementation identity and fails rather than crossing
incompatible tool revisions.

Bazel discovers final kernel products through Loom's public `LoomBinaryInfo`
provider and resolves each product's typed target profile. Each immutable
capture pairs the final target artifact with its summary compile report and a
public `loom-compile-report show` view. Common products are compared by digest
first; byte-identical pairs require no semantic report work. Changed pairs run
the public
`loom-compile-report diff` command and preserve its complete text and JSON views.
Added, removed, changed, unchanged, and incomparable leaves remain distinct
outcomes.

A clean comparison is intentionally compact:

```text
Loom repository compile-report comparison
  corpus: 4 common, 0 added, 0 removed
  comparison: 0 changed, 4 unchanged, 0 incomparable
  semantic diff: 0 changed, 0 unchanged, 4 byte-identical skipped
```

Target patterns bound the analysis while retaining Bazel dependency semantics:

```shell
python dev.py compile-report \
  --base=origin/main \
  --target=//kernel/ggml/quantize:q8_1_x4_f32

python dev.py compile-report \
  --base=origin/main \
  --target=//motif/format/ggml/...
```

An exact library label selects its final target products. A motif or package
scope also selects downstream products that depend on the selected sources, so
changing a shared motif reports every affected kernel
without maintaining an inventory. Repeating `--target` forms a union. Repeating
`--config` applies the same named Bazel configurations to both workspaces.

`latest.json` provides the query surface for the most recent invocation:

```shell
REPORT_ROOT=.reports/compile

jq '{
  source: .candidate_capture.source,
  counts: .comparison.counts
}' "$REPORT_ROOT/latest.json"

jq '[
  .comparison.entries[] |
  select(.state != "unchanged") |
  {label, state, artifact_changed, report_changed, semantic_state, metrics}
]' "$REPORT_ROOT/latest.json"
```

Content-addressed captures live under `captures/`; comparisons and complete
per-leaf diffs live under `comparisons/`; exact base source trees live under
`sources/`. Captures and comparisons verify their manifests and materialized
evidence before reuse. The mutable `latest.json` is only a convenient view over
those immutable records.

### Inspect compiler evidence

Cross-compile the module closure and retain a detailed report as an
optimization baseline:

```shell
LOOM_COMPILE_TOOL="@loom_sdk//:loom-compile"
LOOM_REPORT_TOOL="@loom_sdk//:loom-compile-report"

bazelisk run --config=opt "$LOOM_COMPILE_TOOL" -- \
  "$LOOM_MODULE" \
  --library="$LOOM_LIBRARY" \
  --root=@ggml_quantize_q8_1_x4_f32 \
  --format=amdgpu-hsaco \
  --target=amdgpu:gfx11-generic \
  --compile-report=details \
  --compile-report-output=/tmp/q8_1_x4_f32.baseline.json \
  --output=/tmp/q8_1_x4_f32.baseline.hsaco

bazelisk run "$LOOM_REPORT_TOOL" -- \
  show /tmp/q8_1_x4_f32.baseline.json
bazelisk run "$LOOM_REPORT_TOOL" -- \
  suggest /tmp/q8_1_x4_f32.baseline.json --format=json

# After an edit, repeat the compile with candidate output filenames.
bazelisk run "$LOOM_REPORT_TOOL" -- \
  diff /tmp/q8_1_x4_f32.baseline.json \
       /tmp/q8_1_x4_f32.candidate.json --format=json
```

Each key under `findings[].evidence` maps back into the full report; prefix it
with `.` to use it as a `jq` expression. Format and rebuild the `.loombc` before
emitting the candidate. Report diffs are intentionally strict about schema,
target, configuration, workload, and entry identity so deltas remain
attributable to the kernel or compiler change under investigation.
