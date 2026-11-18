# Bazel Repository Policy

[`defs.bzl`](defs.bzl) wraps Loom's public Bazel rules with the layer contracts
owned by this repository:

- `loom_motif_library` is confined to `motif/`, rejects kernel dependencies,
  emits a linkable bytecode module, marks source-bearing motifs for qualification
  closure, and adds source-policy coverage.
- `loom_kernel_library` is confined to `kernel/`, keeps GGML compatibility
  entry points as leaf launch surfaces, confines dependencies to motifs,
  requires at least one typed target profile, and checks production source
  formatting and final-product compilation. Its archive excludes test inputs.
- `loom_test` composes checks from `srcs` and directly listed `deps` into
  one linked test module. Multiple sources and dependencies-only declarations
  use the same interface. Test leaves declare execution profiles
  and depend on production providers. Scenarios own configuration and trial
  fanout. `--config=loom-asan`, `--config=loom-tsan`, and `--config=loom-ubsan`
  instrument the same test targets independently of native tool instrumentation;
  sanitizer-specific execution profiles are rejected. Stress tests carry both
  `manual` and `stress` tags.
- `loom_library` is confined to production `model/` and `program/` packages.
  It delegates source verification, lint, strict linking, and relocatable bytecode to HRX,
  and applies the model source policy without requiring host/control programs
  to masquerade as individually benchmarkable kernels.
- `loom_cxx_library` provides the same program/model ownership for C/C++
  sources, with explicit public `hdrs`, implementation headers in `srcs`, and
  local `copts`. Its imported module receives the same layer-policy check.
- `loom_cxx_test` imports C/C++ checks and delegates to the same test,
  benchmark-discovery, execution-profile, and fixture-budget policies.
- `loom_model` is confined to `model/` and selects a card preset with an
  exact mapping of providers and their auxiliary compiler inputs. Runtime
  parameter and tokenizer files retain their acquisition descriptors for
  `loom-fetch`; small fixtures may instead be supplied as declared assets.
  The rule publishes a selected card and native serving deployment.
  `encoding="preserve"` retains each declared provider's bytes and format;
  `text` and `bc` normalize providers using the public linker.
- `loom_model_documentation` re-exports the public metadata-only reference rule.
  It accepts a card, with optional authored Markdown, complete acquisition
  lock, and publication URL inputs.
  The generated page combines user guidance, upstream links, and exact card
  facts without depending on weight or provider artifacts.

Each test declaration emits a private filegroup carrying the stable
`loom-benchmark-module` tag. It forwards HRX's linked test module, which retains
cases and benchmarks owned by sources and direct dependency libraries.
Transitive dependencies supply code without selecting their checks. Separate
libraries retain their private symbols. The marker adds discovery metadata without introducing another
link or re-export boundary. Generated format, compile, and execution tests do
not carry benchmark-discovery metadata. Correctness tests run cases and
scenarios; benchmark execution is selected separately with `dev.py benchmark`.

[`source_policy.py`](source_policy.py) verifies the filesystem-level portion of
the contract. Every motif, kernel, and model source package has an adjacent `BUILD.bazel`,
explicitly declares each Loom or C/C++ source and header, loads the repository wrappers, and has
a scoped README chain up to its layer root. Motif sources contain no
`kernel.def`, `config.*`, or concrete target records; kernel packages contain
at least one `kernel.def`. External target declarations and contextual target
queries remain valid in motifs. Authored Loom checks live in `*_test.loom` or `*_stress.loom` sources.
These test sources may also contain `config.*` under `motif/` and declare
private wrapper kernels without weakening the production motif contract.
C++ checks use the corresponding `_test` or `_stress` suffix and
`loom_cxx_test`. Production declarations reject test source files.
The pre-commit source-policy check enforces operation ownership on staged files.
C++ operation policy examines the imported component through `loom-format`,
using the same operation checks without interpreting C++ spelling. This is a
test-side inspection: compilation and publication consume the original bytecode.
Header-only providers have no owned code to inspect; instantiated bodies belong
to their importing consumer.

A `model.json` has an adjacent repository `loom_model` or
`loom_model_documentation` declaration. A description can precede its executable
port. Once a package declares program providers, it also supplies `loom_model`:
its `files` mapping supplies providers, and `data` supplies importer auxiliary
files such as a weight preparation policy. Both mappings exactly cover the
selected compiler inputs in the card's relative namespace. Optional `assets`
supply small runtime files for fixtures or swatches; they participate in test
caching without entering linker action inputs.

Full checkpoints remain external acquisition descriptors in the selected card.
`loom-fetch` installs those files with the built providers, verifies the lock,
and shares cached content across installations. Native serving opens the local
installation without downloading files or consulting fetch locks. A learned
model check takes a provisioned installation explicitly, trusts that its weights
match the declared acquisition lock, and executes the current built providers.
Its card, lock, providers, and test inputs determine the Bazel cache identity.
Small fixture tests declare their full bounded input closure.

`python dev.py lint` runs the complete repository check. `python dev.py build`
and `python dev.py test` run it before invoking Bazel, while generated Bazel
source-policy tests preserve the source-layer checks for direct Bazel users.
The lint command also queries repository-owned tags and ordinary dependency
edges to reject any source-bearing motif or kernel that is not reachable from a
benchmarkable test leaf; no global target inventory or upstream
private rule kind participates.

The complete admission and review contract is documented in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## C++ programs and tests

```starlark
load("@hrx//loom/target/vm:execution_profiles.bzl", "VM_REFERENCE_PROFILE")
load("//build_tools/bazel:defs.bzl", "loom_cxx_library", "loom_cxx_test")

loom_cxx_library(
    name = "publication",
    srcs = ["publication.cxx", "publication_private.h"],
    hdrs = ["publication.h"],
    deps = [":records"],
    copts = ["std=c++23"],
)

loom_cxx_test(
    name = "publication_checks",
    srcs = ["ordinary_test.cxx", "continuation_test.cxx"],
    deps = [":publication"],
    execution_profiles = [VM_REFERENCE_PROFILE],
)
```

`hdrs`, `defines`, and package-relative `includes` form the exported source
interface. `copts` and `local_defines` apply only to the owning component.
`copts` uses Loom importer options such as `std=c++23`, rather than host compiler
flags. Runtime fixtures belong in `data`. A header-only component uses `hdrs`
without manufacturing an empty bytecode module.

Imports and test linking precede workload and execution-profile expansion.
Normal builds strip diagnostic metadata; `-c dbg` retains it. Comment-only
header edits rerun import but can preserve downstream bytecode and cached test
results. Source packages include the declared headers; normalized bytecode
packages retain runtime data without carrying unused compiler headers.

## Toolchain selection

[`sdk_repository.bzl`](sdk_repository.bzl) registers the public HRX toolchain
roles against either an installed SDK or the selected HRX source tree.
`dev.py setup --sdk` selects the Python environment containing `hrx-loom`;
`--hrx-source=PATH` and `--pinned-source` select source-built tools.
The developer guide owns [setup examples](../../docs/development.md#select-a-compiler).

The installed wheel's RECORD supplies the file inventory. Individual program,
library, and Python file contents remain Bazel inputs; the wheel archive and
RECORD are not blanket execution-test dependencies. Native tool outputs include
their adjacent shared libraries in both the action-tool closure and runfiles,
so local, sandboxed, and remote execution use the same SDK bytes. Selecting an
invalid or incomplete SDK fails during repository resolution.
