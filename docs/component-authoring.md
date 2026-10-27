# Author a reusable component

This guide uses the existing Q8 quantization package to show where code goes,
how tests stay independent, and how model cards follow dependencies. Commands
run from the repository root after [selecting an SDK or source checkout](development.md#select-a-compiler).

## Place the component

| Path | Owns |
| --- | --- |
| `motif/` | Reusable representations, functions, and templates without a launch ABI. |
| `kernel/` | Launchable operations and their ABI contracts. |
| `program/` | Service policy and executable compositions shared by models. |
| `model/<publisher>/<model>/` | Model-specific parameter interpretation and composition. |

Dependencies point toward reusable components; shared code has no dependency
back on a model. Motifs receive configuration facts from their callers. The
[contribution contract](../CONTRIBUTING.md#source-contract) explains these
boundaries and the source documentation expected at each layer.

## Keep tests out of production dependencies

The [quantization package](../kernel/ggml/quantize/) has one implementation and
an adjacent test source:

```text
kernel/ggml/quantize/
  BUILD.bazel
  README.md
  q8_1_x4_f32.loom
  q8_1_x4_f32_test.loom
```

Its [BUILD file](../kernel/ggml/quantize/BUILD.bazel) separates their ownership:

```starlark
load("@hrx//loom/target/amdgpu:execution_profiles.bzl", "AMD_GPU_HARDWARE_PROFILE")
load("//build_tools/bazel:defs.bzl", "loom_kernel_library", "loom_test")

package(default_visibility = ["//visibility:public"])

loom_kernel_library(
    name = "q8_1_x4_f32",
    srcs = ["q8_1_x4_f32.loom"],
    targets = ["@hrx//loom/target/amdgpu:gfx11-generic"],
    deps = ["//motif/format/ggml:q8_1_x4"],
)

loom_test(
    name = "q8_1_x4_f32_checks",
    srcs = ["q8_1_x4_f32_test.loom"],
    execution_profiles = [AMD_GPU_HARDWARE_PROFILE],
    deps = [":q8_1_x4_f32"],
)
```

The production rule builds reusable bytecode and target artifacts. The test
rule owns cases, scenarios, benchmarks, and their execution profile. Its
`kernel.decl` names the production ABI, resolved through `deps`. Shared
independent oracles can be `testonly` filegroups in `test/` and enter only the
test rule's `srcs`.

A test edit changes its test module and result while leaving production
archives and consuming models cached. Repository source checks reject
`check.*` operations and `_test.loom` inputs in production providers.
`dev.py lint` checks that each motif and kernel reaches a benchmarkable test.

Shapes are inputs to a parameterized implementation; cases and scenarios
provide workload coverage. Related ABIs can share a source file and helpers.
The target list above chooses compiled artifact checks for this package;
execution profiles choose where its test module runs.

## Connect consumers through dependencies

A model program depends on the kernel provider, such as
`//kernel/ggml/quantize:q8_1_x4_f32`. That provider brings its motif dependencies.
The model's `loom_model` declaration names its program roots and selects source
or module representations. [Qwen's BUILD file](../model/Qwen/Qwen3.8-27B/BUILD.bazel)
is a complete example.

For provider-derived compositions, Bazel analysis supplies the transitive
compiler file lists. After changing dependencies:

```sh
python dev.py model-card --fix
python dev.py model-card --check
```

The generated paths are repository-relative. Model facts, acquisition origins,
variants, presets, and asset locks remain authored. The
[model-card guide](model-authoring.md#derive-compiler-files-from-provider-dependencies)
explains representation selection and publication paths. Other models can
adopt this declaration incrementally; explicit file mappings remain supported.

## Check the change

```sh
python dev.py format --fix kernel/ggml/quantize/q8_1_x4_f32.loom kernel/ggml/quantize/q8_1_x4_f32_test.loom
python dev.py test --target=//kernel/ggml/quantize:all -- --config=asan --config=amdgpu
```

`dev.py test` checks repository formatting, layer policy, benchmark
reachability, and generated cards before running the selected Bazel tests.
`python dev.py hook` installs the pre-commit checks for staged Loom formatting
and generated cards. Small component checks use fixtures rather than full
model weights; model package builds also leave weight acquisition to
`loom-fetch`.

The [benchmark workflow](development.md#sweep-repository-benchmarks) covers
explicit measurements, and [model authoring](model-authoring.md#exercise-the-model)
covers complete model tests. Component correctness, model behavior, and
performance measurements have separate evidence requirements.
