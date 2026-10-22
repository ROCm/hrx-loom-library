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
namespace README owns supported variants, weight and tokenizer assumptions,
service contracts, kernel and motif dependencies, independent correctness
evidence, benchmark scenarios, and deployment status.

## Build boundary

Model-owned program sources use `loom_library` from the repository's
`//build_tools/bazel:defs.bzl`. The wrapper retains the public HRX rule's explicit
dependency and relocatable bytecode behavior; lower-level libraries remain
separate linker inputs. Text sources receive formatting, lint, and model-layer
policy checks. Bytecode is supported through the same public rule.

A `loom_model` declaration selects one card configuration and maps its exact
provider and asset files. Its default output is `serve.json`; runfiles contain
the selected card and full closure in the authored relative namespace. Source,
bytecode, and mixed providers use the same native service. Physical devices and
request options are supplied independently from the portable model definition.

Declared asset repositories pin remote revisions and digests. Bazel's normal
repository cache and `--distdir` can reuse existing downloads; test execution
receives those files as runfiles instead of depending on a developer's cache.
The server owns no network acquisition. Tests of learned behavior supply
independent input/output evidence and a real native execution profile. Source
checks and a successful card build do not establish model accuracy.
