# Choose a model

These are the official Loom model implementations maintained in this library.
Each overview explains the model's supported hardware, checkpoint variants,
licenses, execution limits, and a complete fetch-to-request workflow.

| Model | Use |
| --- | --- |
| [Moonshine Streaming Tiny](moonshine-ai/moonshine-streaming-tiny/OVERVIEW.md) | English speech recognition. |
| [Pocket TTS](kyutai/pocket-tts/OVERVIEW.md) | English speech synthesis with stock voices or uploaded voice samples. |
| [Qwen3.8 27B](Qwen/Qwen3.8-27B/OVERVIEW.md) | Text and image-conditioned chat, with optional speculative drafting. |
| [Qwen3.8 Flash-Next](Qwen/Qwen3.8-Flash-Next/OVERVIEW.md) | Mixture-of-experts text generation with optional multi-token prediction. |
| [Qwen3 Embedding 0.6B](Qwen/Qwen3-Embedding-0.6B/OVERVIEW.md) | Text embeddings on the Strix Halo NPU. |
| [Krea 2 Turbo](krea/Krea-2-Turbo/OVERVIEW.md) | Image generation with BF16 or INT8 parameters and request-selected adapters. |

Support applies to the variants and targets described by each implementation.
The model card names the exact compatible files; other files or quantizations
in an upstream repository are not implicitly supported. Checkpoints retain
their publishers' licenses, which may differ from the program sources and
from one variant to another.

## Discover runnable packages

With the [Loom SDK](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
installed, query the published catalog without cloning it:

```sh
loom-fetch --catalog=github:ROCm/hrx-loom-library@main --list --json
```

Replace `main` with a tag or commit to select a specific library revision.
Use a revision compatible with your SDK. From a checkout, the equivalent is:

```sh
loom-fetch --catalog=./catalog.json --list --json | jq '.models | keys'
loom-fetch --catalog=./catalog.json --model=moonshine-ai/moonshine-streaming-tiny --preset=streaming --output=./moonshine
```

The resulting `moonshine/model.json` can be passed directly to `loom-serve`.
The [root quickstart](../README.md#run-a-model) shows that step. A model-card
URL or local path can also be fetched directly; membership in this catalog is
not required.

A **variant** selects compatible parameters and model options. A **preset**
selects a variant, programs, and execution defaults for a particular use.
Program sets can contain source, bytecode, or both; their names do not define
a file format. Source programs run with the installed SDK without a build.
Publishing compiled programs is an author workflow, not a prerequisite for
using these models.

## Adapt an implementation

The directories follow `model/<original-publisher>/<base-model>/`, using the
original Hugging Face identity where available. Quantizations and other
compatible variants stay under that base model, even when their assets come
from a different publisher.

Each model's `OVERVIEW.md` is for users; its `README.md` describes the program
composition, builds, and tests. [Model authoring](../docs/model-authoring.md)
explains how to fork a package, add a variant or implementation, generate its
standalone reference, and publish it independently.
