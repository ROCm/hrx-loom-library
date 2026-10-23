# Bazel Repository Policy

[`defs.bzl`](defs.bzl) wraps Loom's public Bazel rules with the layer contracts
owned by this repository:

- `loom_motif_library` is confined to `motif/`, rejects kernel dependencies,
  emits a linkable bytecode module, marks source-bearing motifs for qualification
  closure, and adds source-policy coverage.
- `loom_kernel_library` is confined to `kernel/`, keeps GGML compatibility
  entry points as leaf launch surfaces, confines dependencies to motifs,
  requires at least one typed target profile and execution profile, and adds
  the normal format, planning, final-product compilation, and execution checks.
- `loom_test` is confined to explicit `test/` packages and delegates to HRX's
  root-owned linked test module. Test leaves require one execution profile but
  do not manufacture standalone target products.
- `loom_library` is confined to production `model/` packages. It delegates
  source verification, lint, strict linking, and relocatable bytecode to HRX,
  and applies the model source policy without requiring host/control programs
  to masquerade as individually benchmarkable kernels.
- `loom_model` is confined to `model/` and selects a card preset with an
  exact mapping of providers and their auxiliary compiler inputs. Runtime
  parameter and tokenizer files retain their acquisition descriptors for
  `loom-fetch`; small fixtures may instead be supplied as declared assets.
  The rule publishes a selected card and native serving deployment.
  `encoding="preserve"` retains authored source, bytecode, or
  mixed inputs; `text` and `bc` normalize providers using the public linker.
- `loom_model_documentation` re-exports the public metadata-only reference rule.
  It accepts a card, with optional authored Markdown, complete acquisition
  lock, and publication URL inputs.
  The generated page combines user guidance, upstream links, and exact card
  facts without depending on weight or provider artifacts.

Each kernel and test declaration emits a private filegroup carrying the stable
`loom-benchmark-module` tag. It forwards HRX's linked test module, which retains
cases and benchmarks owned by the root while excluding test-only symbols from
dependencies. The marker adds discovery metadata without introducing another
link or re-export boundary. Generated format, plan, compile, and execution tests
do not carry benchmark-discovery metadata.

[`source_policy.py`](source_policy.py) verifies the filesystem-level portion of
the contract. Every motif, kernel, and model source package has an adjacent `BUILD.bazel`,
explicitly declares each `.loom` source, loads the repository wrappers, and has
a scoped README chain up to its layer root. Motif sources contain no
`kernel.def`, `config.*`, or concrete target records; kernel packages contain
at least one `kernel.def`. External target declarations and contextual target
queries remain valid in motifs. Sources below an explicit
`test/` component use `loom_test` and may declare private wrapper
kernels without weakening the production motif contract.

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
edges to reject any source-bearing motif that is not reachable from a
benchmarkable kernel or test leaf; no global target inventory or upstream
private rule kind participates.

The complete admission and review contract is documented in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).
