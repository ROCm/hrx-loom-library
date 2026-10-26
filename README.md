# HRX Loom Library

The official collection of model implementations and reusable components for
[Loom](https://rocm.github.io/hrx-system/loom/). Use it to serve models, adapt
an implementation to your workload, or build a model of your own. The
[catalog](model/README.md) covers text generation, image generation, speech
recognition and synthesis, and text embeddings.

Each model includes its Loom programs, a machine-readable model card, supported
checkpoint variants, serving presets, and correctness checks. Shared programs,
kernels, and motifs provide building blocks for other models and applications.
The [HRX System SDK](https://github.com/ROCm/hrx-system/tree/main/loom) supplies
the compiler, runtime, and tools; this repository supplies the implementations.
This is an inference library, not a training framework or checkpoint mirror.
Weights are fetched from their publishers under their respective licenses.
A checkpoint alone is not a Loom implementation: the model's programs must
support its architecture, format, and execution target.

## Run a model

Install the **`hrx-loom` SDK with its optional `[hf]` extra** using the
[installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/),
which covers Linux and Windows prerequisites. With a supported AMD GPU,
fetch Moonshine Streaming Tiny and start its speech-recognition service:

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/moonshine-ai/moonshine-streaming-tiny/model.json --preset=streaming --output=./moonshine
loom-serve --model=./moonshine/model.json --device=amdgpu --listen=127.0.0.1:8080
```

No repository clone or Bazel build is needed. Fetch copies the selected
programs, verifies locked assets, and reuses cached weights. The SDK
compiles the programs when serving. AMD device access uses libamdf and the
operating-system driver; it does not require a ROCm SDK installation.

Continue with the [Moonshine request example](model/moonshine-ai/moonshine-streaming-tiny/OVERVIEW.md#acquire-and-serve)
to transcribe a recording, or [choose another model](model/README.md).
The [fetch-to-serve guide](https://rocm.github.io/hrx-system/loom/serving/workflows/fetch-and-serve/)
covers discovery, gated downloads, offline use, and custom serving settings.
Each model's overview owns its hardware requirements, supported inputs, and
complete request examples.

## Make it yours

Loom models can live anywhere. This repository is the supported official set,
not a registry that every Loom model must join. Fork and remix the library, or
publish an independent repository using the same model-card format and public
Loom build rules. A model card and its programs can be hosted on GitHub,
Hugging Face, or another HTTPS host and used directly with `loom-fetch`.

- [Author or adapt a model](docs/model-authoring.md): start from a fork, build
  and test a package, iterate with a local compiler, and publish your own card.
- [Develop reusable components](docs/development.md): repository layers, kernel
  checks, benchmarks, and compiler reports.
- [Contribute to the official library](CONTRIBUTING.md): scope and review
  expectations. Model additions are coordinated with maintainers while the
  library's organization and contribution workflow are being established.
