# Model and Command Programs

Model packages are composition roots. They assemble reusable motifs and
launchable kernels into model-specific subgraphs or complete command programs,
own the configuration that is genuinely global to that composition, and
preserve value facts across orchestration boundaries for specialization.

A model package may launch native kernels and GGML compatibility kernels while
coverage is being migrated. Reusable representation or algorithm code flows
down into `motif/`; reusable launch surfaces flow into domain packages under
`kernel/`. Code remains under `model/<publisher>/<model>/` when its contract is
inherently tied to that model architecture or command schedule. Publisher and
base-model identity follow the original model, rather than a serving framework
or one derived quantization.

This layer is also where GGUF metadata becomes buffers, dimensions, encoding
identities, and other compile-time or invocation-time facts. Device motifs
consume those projected facts and never parse the container directly.

## Map

Model namespaces are created with their first checked composition. Each
package owns a user overview with supported variants, weight and tokenizer
assumptions, service behavior, and setup instructions. Its developer README
covers kernel and motif dependencies, independent correctness evidence,
benchmark scenarios, and the build workflow.

## Generated model references

Each card produces a standalone Markdown reference combining its authored
overview with compatible choices, canonical presets, effective JIT
settings, scoped parameter shards, tokenizers, and exact source locations
beside local destination names:

```sh
bazel build //model/moonshine-ai/moonshine-streaming-tiny:reference
```

The output appears under the same package in `bazel-bin/`, with `.md` appended
to the target name. The overview uses absolute publication links so the page
can be hosted independently of the repository README. The action consumes
only the card, overview, and optional lock. The SDK also supplies
`python -m loom.model.documentation --card model.json --overview OVERVIEW.md --output model.md`
for authors working outside this repository. A publication `--card-url`
resolves card-relative provider links in the generated page.

## Build boundary

Model-owned program sources use `loom_library` from the repository's
`//build_tools/bazel:defs.bzl`. The wrapper retains the public HRX rule's explicit
dependency and relocatable bytecode behavior; lower-level libraries remain
separate linker inputs. Text sources receive formatting, lint, and model-layer
policy checks. Bytecode is supported through the same public rule.

A `loom_model` declaration selects one card preset and maps its exact
providers through `files` and importer auxiliary files through `data`. It emits
`model.json` and a separate `serve.json` in the authored relative namespace.
Source, bytecode, and mixed providers use the same native service. Physical
devices and request options are supplied independently from the portable model
definition.

The selected card retains external parameter and tokenizer acquisition
descriptors. `loom-fetch` combines built providers with verified cached assets
in a runtime installation. Full checkpoints stay outside Bazel repositories,
action inputs, outputs, and runfiles. The optional `assets` mapping is for small
declared fixtures and swatches that need ordinary Bazel test caching.

Learned checks explicitly consume a fetched installation, verify its checkpoint
identity, and run a fresh native service against independent input/output
evidence. Small numerical and state tests remain hermetic Bazel tests. Source
checks and a successful card build do not establish model accuracy.
