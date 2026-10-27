# Contributing to the Loom standard library

## Scope

This repository maintains the official Loom model set and shared library
components. Forks, remixes, and independently published models are encouraged;
using Loom does not require contributing a model here. The
[model-authoring guide](docs/model-authoring.md) covers that workflow.

Official additions are curated with maintainers. The library's organization,
review process, and capacity for maintaining more models are still being
established, so this is not an open submission queue for every model port.
Open an issue to discuss a proposed model or substantial shared component
before investing in a large pull request. Bug reports, focused fixes, and
feedback on the authoring workflow help establish that foundation.

## Review contract

This repository admits reusable compiler inputs and complete model compositions. A contribution earns its place by strengthening a composable motif, exposing a deliberate launch or program ABI, or delivering a checked model through those components. The unit of review is the complete library closure: source, documentation, correctness programs, benchmark plans, target qualification, and the dependency edges that make it reusable.

## Admission surfaces

| Surface | Ownership | Admission evidence |
| --- | --- | --- |
| `motif/` | Target-neutral data representations and algorithms expressed as `func.def`, `template.def`, encodings, and related declarations. | Canonical and lint-clean source, scoped documentation, no launch surface, and reachability from a qualified kernel or test leaf. |
| `kernel/` | Concrete `kernel.def` entry points implementing one operation contract and ABI. | At least one target profile, reachability from an executing benchmark test, and dependencies confined to production motifs. |
| `program/` | Shared service policy and executable compositions used by multiple models. | Explicit `loom_library` providers, scoped documentation, real service or command callers, and dependencies that do not point back to a model. |
| Adjacent `*_test.loom` or `test/` below each layer | Test-only programs that exercise production contracts. | Private root-owned test module, semantic benchmarks and correctness cases, and one execution profile. |
| `target/` | Genuinely repository-specific target or execution policy. | A concrete policy difference that cannot be expressed by canonical HRX profiles; upstream profiles are consumed directly otherwise. |
| `model/` descriptions | Portable model facts, asset variants, service contracts, and provider composition. | A card validated through `loom_model_documentation`, a standalone overview with exact upstream links, and a precise supported-variant and hardware contract. |
| `model/` executable ports | Model-specific programs and portable serving compositions. | Explicit `loom_library` providers, a card published through `loom_model`, and an independently checked native service output for each supported representation. |

Source-bearing `target/` packages still require their own admission rule. Model programs use the repository's `loom_library` wrapper for canonical source, lint, explicit dependencies, and relocatable bytecode. The `loom_model` wrapper selects the card's preset and exact providers and auxiliary compiler inputs through HRX's public rule. Full checkpoints retain their acquisition descriptors for `loom-fetch` and stay outside Bazel's payload graph. Model-local `test/` packages use the same `loom_test` contract as other layers; small fixtures and swatches declare their full input closure for test caching. Full-model checks consume provisioned weights, trust that they match the declared acquisition lock, and run current built programs against independent inputs and checked outputs. The card, lock, and providers participate in Bazel cache identity; full checkpoint hashing belongs to acquisition.

Model identity follows the original publisher and base model under `model/<publisher>/<model>/`. Source and bytecode inputs are program representations within a card, not separate model identities. Quantized checkpoints and services are selected through the card. Physical placement remains deployment input; request-time adapters do not become base-model parameters. Model-specific code composes shared programs, kernels, and motifs. Shared programs own service policy and multi-stage execution; kernels own individual launch contracts; motifs own reusable representations and algorithms. These layers cannot depend back on a model, and motifs cannot depend on a program or kernel.

Authored `model.json.lock` files use version 2 asset scope. Program and auxiliary files, including shared library providers, follow the source checkout or declared origin revision without per-file checksum maintenance. Generated publication locks may cover complete executable bytes; those belong with their published artifacts. The [model-authoring guide](docs/model-authoring.md) describes editing, asset updates, and publication.

## Branch-local incubation

Long-running ports and kernel searches may live below `experimental/` on
feature branches. Experimental packages use the public Loom and Bazel rules;
normal `//...` builds, tests, target-profile execution queries, formatting, and
benchmark discovery traverse them. This gives incubation work the same compiler
and hardware evidence as admitted code without turning it into a weaker
standard-library layer.

Pull requests and pushes targeting the repository release line fail while the
candidate tree tracks any `experimental/**` path. The gate reads the candidate
tree rather than branch history, so a branch may carry experiments for as long
as needed, extract reusable work into `motif/`, `kernel/`, `program/`, `model/`, or
`target/`, and remove the incubation payload before promotion. Feature branches
may compose experimental targets with sources at any repository path; those
paths are working locations until the candidate tree passes release admission.
Promotion moves the reusable sources into their durable layer, and the ordinary
complete build proves that no required experimental package was left behind.

## Source contract

Each `.loom` file stands on its own as an implementation artifact. Its leading documentation establishes the represented format or operation, public symbol contract, parameter and shape meaning, storage layout, ABI ownership, and any specification or oracle used to establish behavior. Package READMEs carry information that cannot be recovered from `ls`: scope boundaries, external specifications, ABI relationships, supported behavior, and the intended composition layer. Per-symbol inventories remain beside the source so they cannot drift from it.

Public symbols are the intended linking surface. Helpers whose only consumers are in the same module remain private. Motifs expose representation and algorithm contracts without a launch ABI; GGML-compatible kernels own GGML graph-operation ABI details, while native kernels may consume the same GGUF carriers without inheriting that ABI.

Dynamic workloads and shapes remain values whenever the implementation can support them. Configuration represents facts that materially select the compiled program—model architecture choices, formats, or target capabilities—not convenient constants such as one observed element count or workgroup size. Production motifs neither declare nor read global configuration: `config.decl`, `config.def`, and `config.get` stay in composition and test leaves. Motif functions and templates receive values and facts from composition roots so one linked motif can serve every invocation in a command program. Concrete target records also stay in leaves; motifs may declare an external target contract or query their execution context. Test-only wrappers may declare case configuration when it is part of the execution witness. Range facts on `config.decl`, `index.assume`, `buffer.assume`, and precise view types carry caller knowledge into specialization. Target-specific source belongs in a target-scoped motif only when the algorithm truly cannot be expressed from target facts supplied at the leaf.

Canonical text and authoring policy are independent gates. `loom-format` parses, verifies, and canonicalizes syntax. `loom-lint` checks human-facing source conventions such as semantic constant names. Repository policy then verifies layer placement, dependency direction, explicit BUILD declarations, scoped documentation, and qualification closure.

## Correctness and benchmark programs

The [component-authoring guide](docs/component-authoring.md) connects these
requirements to production callers, bounded tests, and complete ownership
changes using examples from the library.

A `check.case` is an executable statement of behavior, not merely sample data. Cases cover the contracts most likely to be lost during specialization: signedness, byte and lane order, extrema, tails, non-power-of-two extents, dynamic dimensions, aliasing, and numerical tolerance. Expected values come from a transparent reference or independently established oracle.

Production `.loom` sources and adjacent `_test.loom` sources have separate owners. `loom_kernel_library` owns production code; `loom_test` owns cases, scenarios, benchmarks, and independent oracles, and depends on the production provider. Test sources never enter production archives or model-card file lists. A test edit changes its test module and result without invalidating models that consume the kernel. The [component guide](docs/component-authoring.md#keep-tests-out-of-production-dependencies) shows the declarations.

Related semantic ABIs can share a file and helpers: overwrite and in-place residual operations are distinct entry points in one coherent family. Shapes, tile sizes, and subgroup widths specialize those entries through templates; scenarios provide workload coverage. Different algorithms or storage lifecycles, such as a paged block pool and a dense tensor, can own different files. Motifs use test-only wrapper kernels only when no production kernel exercises the same contract.

Every test leaf is benchmark-planned by the public Loom runner. A module with no `check.benchmark` fails planning. Benchmarks name correctness cases and select workloads that expose meaningful operating regimes rather than one flattering shape. Decode, small-batch, prefill, tails, and format-specific stress shapes are separate benchmark declarations when they exercise different specialization or hardware behavior.

Each kernel names at least one target profile proving that the linked program lowers through a real backend and produces a final artifact. Each kernel is reachable from a test leaf that names at least one execution profile proving its cases through a real runner. Test wrappers remain execution witnesses rather than standalone target products. Profiles are reusable and target-owned; common library rules contain no gfx, Vulkan, HIP, or other device-specific conditionals. Additional kernel targets strengthen portability without recompiling target-neutral motif modules independently for every leaf.

## Performance evidence

Correctness admission and performance claims are separate. Shared CI establishes that programs build and execute; its changing machines and load cannot establish a regression threshold. Performance evidence comes from a stable, identified machine with baseline and candidate captured under the same software, target, workload, measurement mode, and exclusive-use discipline.

`python dev.py benchmark` writes immutable, package-shaped artifact bundles using the public benchmark report schema. A focused claim names the benchmark declarations, device, baseline revision, candidate revision, and relevant distribution rather than quoting one best iteration. Compiler changes include the corresponding `python dev.py compile-report --base=<revision>` comparison so register, LDS, spill, occupancy, and emitted-code changes remain attributable. Targeting the edited library or motif keeps the comparison bounded while Bazel reverse dependencies include every affected compilation leaf. Roofline proxies and oracle implementations bound the search space; dispatch-count reduction alone is not evidence that a fusion improved end-to-end execution.

## Local admission sequence

```shell
python dev.py setup --sdk
python dev.py format --check
python dev.py lint
python dev.py test
python dev.py build
python dev.py compile-report --base=origin/main
```

`dev.py lint` checks filesystem policy and asks Bazel to prove that every source-bearing motif and kernel reaches a benchmarkable test leaf. `dev.py test` first checks canonical formatting for the complete repository, then runs source policy, public lint, benchmark-plan, target-compilation, and available execution tests. Hardware tests without a matching local resource skip locally and execute through the repository runner matrix.

A focused performance capture adds an explicit target and device:

```shell
python dev.py benchmark \
  --config=amdgpu \
  --device=amdgpu://0 \
  --output-dir=/tmp/hrx-loom-library-candidate \
  --target=//kernel/ggml/quantize:q8_1_x4_f32_checks \
  -- \
  --measure=dispatch_complete \
  --batch-size=64 \
  --profile-final-batch=true
```

Review centers on the boundary questions: whether the code lives at the lowest reusable layer, whether its public symbols and ABI are deliberate, whether specialization receives all available facts without freezing incidental shapes, whether correctness cases cover the dangerous contracts, whether target qualifications exercise the intended backend, and whether any performance claim is bounded by reproducible baseline and oracle evidence.
