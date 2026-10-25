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

| Model | Service | Described variants | Execution status |
| --- | --- | --- | --- |
| [Moonshine Streaming Tiny](moonshine-ai/moonshine-streaming-tiny/OVERVIEW.md) | Streaming speech recognition | Original F32; source, bytecode, and mixed providers | Runnable on AMDGPU; independent transcript checks |
| [Krea 2 Turbo](krea/Krea-2-Turbo/OVERVIEW.md) | Image generation | BF16 and INT8 ConvRot; source, bytecode, and mixed providers | Runnable on AMDGPU; shared runtime adapters, exact image regressions, and bounded VAE workspace |
| [Pocket TTS](kyutai/pocket-tts/OVERVIEW.md) | Streaming speech synthesis | Three-voice default, four-voice noncommercial variant, and native custom uploads; source, bytecode, and mixed providers | Runnable on AMDGPU; original-recording preparation, independent numerics, and native PCM lifecycle checks |
| [Qwen3.8 27B](Qwen/Qwen3.8-27B/OVERVIEW.md) | Text generation with optional image input | UD-Q5_K_XL, vision projector, and DFlash2 draft weights; source, bytecode, and mixed providers | Runnable text on AMD RDNA/CDNA; vision and DFlash2 on RDNA with concurrent requests, checkpoint forks, and parameter lifecycle checks |
| [Qwen3.8 Flash-Next](Qwen/Qwen3.8-Flash-Next/OVERVIEW.md) | Text generation | Four ordered UD-Q4_K_XL shards, optional MTP weights; source, bytecode, and mixed providers | Runnable on AMD RDNA; independent recall, checkpoint forks, and parameter lifecycle checks |
| [Qwen3 Embedding 0.6B](Qwen/Qwen3-Embedding-0.6B/OVERVIEW.md) | Text embeddings | Original BF16 checkpoint with runtime BFP16EBS8 conversion; source, bytecode, and mixed providers | Runnable on the Strix Halo NPU; independent publisher checks |

Model namespaces can begin with a portable card and generated reference before
their executable programs are ported. Such packages describe the required
provider and asset closure and state their execution status in the overview;
they do not supply a runnable installation. An executable port adds the actual
providers, acquisition lock, build declarations, and independent service checks.
Each package owns a user overview with supported variants, weight and tokenizer
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
to the target name. Each reference links to its published model card and resolves
card-relative file locations against that URL. The overview also uses absolute
publication links, so the page can be hosted independently of the repository
README. The action consumes only the card, overview, and optional lock. The SDK
also supplies
`python -m loom.model.documentation --card model.json --overview OVERVIEW.md --output model.md`
for authors working outside this repository. A publication `--card-url`
resolves card-relative provider links in the generated page.

## Discover runnable packages

The repository-root `catalog.json` lists packages with supplied providers and
qualified execution. Packages enter this index when their executable ports are
admitted and all selected assets have published acquisition sources. The embedding
port loads the original Qwen BF16 checkpoint and performs its native representation
conversion during model loading. Qwen3.8-27B includes text, image-conditioned
chat, and DFlash2 drafting with the complete native service checks.
Installed SDK users can inspect a published snapshot without a clone:

```sh
loom-fetch --catalog=github:ROCm/hrx-loom-library@4317508bf5adb251a284ea8dc4e5a24a11bd1c17 --list --json
```

This snapshot lists all six packages above: recognition, speech synthesis,
text generation, image-conditioned chat, NPU embeddings, and image generation.
The same variants and presets are available through the checkout's catalog.
Checkpoint files are acquired directly from the named publishers.

A local checkout supports the same discovery and selection:

```sh
loom-fetch --catalog=./catalog.json --list --json | jq '.models | keys'
loom-fetch --catalog=./catalog.json --model=moonshine-ai/moonshine-streaming-tiny \
  --preset=streaming --output=./moonshine
loom-serve --model=./moonshine/model.json --device=amdgpu2
```

The same relative card paths resolve within a published GitHub catalog snapshot.
Release wheels can associate that snapshot with their qualified library tag.
Development wheels require an explicit catalog. Update the checked-in index
from its admitted cards with the SDK:

```sh
python -m loom.model.catalog \
  --card=model/Qwen/Qwen3-Embedding-0.6B/model.json=model/Qwen/Qwen3-Embedding-0.6B/model.json \
  --card=model/Qwen/Qwen3.8-27B/model.json=model/Qwen/Qwen3.8-27B/model.json \
  --card=model/Qwen/Qwen3.8-Flash-Next/model.json=model/Qwen/Qwen3.8-Flash-Next/model.json \
  --card=model/krea/Krea-2-Turbo/model.json=model/krea/Krea-2-Turbo/model.json \
  --card=model/kyutai/pocket-tts/model.json=model/kyutai/pocket-tts/model.json \
  --card=model/moonshine-ai/moonshine-streaming-tiny/model.json=model/moonshine-ai/moonshine-streaming-tiny/model.json \
  --output=catalog.json
```

`//build_tools/bazel:catalog_test` checks that discovery metadata stays in sync
with these cards. Neither generation nor this test reads program payloads or
model weights.

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

The selected card retains external parameter, tokenizer, and voice-recording
acquisition descriptors. `loom-fetch` combines built providers with verified cached assets
in a runtime installation. Full checkpoints stay outside Bazel repositories,
action inputs, outputs, and runfiles. The optional `assets` mapping is for small
declared fixtures and swatches that need ordinary Bazel test caching. Pocket's
original recordings remain fetch inputs; CPU Loom prepares their audio and
generates rotary tables during native model initialization and registration.

Learned checks explicitly consume a provisioned installation and run the current
built programs against independent input/output evidence. Their declared card,
lock, and providers determine the Bazel cache identity; they trust the provisioned
weights to match the lock instead of hashing full checkpoints on every test.
`loom-fetch` verifies acquisition identities. Small numerical and state tests
remain hermetic Bazel tests. Source checks and a successful card build do not
establish model accuracy.
