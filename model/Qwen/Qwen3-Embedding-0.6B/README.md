# Developing Qwen3 Embedding

The [model overview](OVERVIEW.md) describes the installed workflow, original
checkpoint, prepared representation, and embedding request contract. Serving
uses the installed SDK. Preparing the NPU bank uses the pinned Python
authoring environment; neither operation needs Bazel.

## Prepare an installation

From a library checkout, convert the pinned publisher checkpoint into a new
model directory:

```sh
uv run model/Qwen/Qwen3-Embedding-0.6B/prepare_assets.py \
  --output=../qwen3-embedding

loom-serve --model=../qwen3-embedding/model.json \
  --device='xdna://0?columns=8' --pool_backing=fixed \
  --listen=127.0.0.1:8080
```

The converter acquires the exact checkpoint, architecture configuration, and
tokenizer through the Hugging Face cache. It checks the architecture, packs
projection panels layer by layer, and writes one 808,069,248-byte Safetensors
file, including its header. Its tensor payload must match the qualified bank's
SHA-256 before publication succeeds. The output also contains the tokenizer,
complete source and motif closure, a relative-path model card, and its lock.
The model card is written last, after every referenced file exists.

The original checkpoint remains in the HF cache; the converted bank is a
different representation required by this composition. `--offline` forbids
checkpoint downloads. `uv run --offline` separately requires the Python
packages to be cached. The converter refuses an existing output directory.
The pinned preparation functions preserve native panel ordering, exponent and
rounding rules, 64-byte lookahead, and the publisher's BF16 rotary values.
Safetensors alignment by itself cannot perform this conversion.

The generated directory can be served immediately or hosted as a complete
source publication. A recipient can then use `loom-fetch URL/model.json
--output=./qwen3-embedding` and the same serving command. The repository's
unprepared card refers to `prepared/weights.safetensors`, which is not hosted
here or supplied by the publisher. It is therefore excluded from the public
picker until a prepared bank is published.

## Build source, bytecode, and mixed programs

```sh
bazel build \
  //model/Qwen/Qwen3-Embedding-0.6B:source \
  //model/Qwen/Qwen3-Embedding-0.6B:archive \
  //model/Qwen/Qwen3-Embedding-0.6B:mixed \
  //model/Qwen/Qwen3-Embedding-0.6B:reference
```

The selected outputs live below
`bazel-bin/model/Qwen/Qwen3-Embedding-0.6B/`. `source` retains Loom text;
`archive` uses bytecode; `mixed` keeps preparation and control in text while
using bytecode numerical and motif providers. All three retain `policy.loom`
as an explicit auxiliary compiler input. They remain inputs to runtime linking
and JIT specialization.

Each package contains a selected `model.json`, deployment `serve.json`, and
its provider closure. Full parameters and the tokenizer retain their acquisition
descriptors. Bazel neither fetches nor reads the prepared bank or original
checkpoint. `reference.md` combines the standalone overview with card facts.

The [program documentation](program/README.md) explains the fixed 32-core
placement, shared-row ownership, and numerical boundaries. The public roots are
`embedding_prepare_v1`, `embedding_prepare_request_v1`,
`embedding_prepare_batch_v1`, and `embedding_run_v1`. Parameters load through
the `model` index scope and the tokenizer through the `text` identifier.

## Run learned regressions

After preparing the installation above, run the current built providers on the
NPU with externally provisioned assets:

```sh
bazel test --config=asan --config=amdgpu \
  --strategy=TestRunner=standalone \
  --test_env=LOOM_MODEL_ASSETS="$(realpath ../qwen3-embedding)" \
  //model/Qwen/Qwen3-Embedding-0.6B/test:embedding_source_test \
  //model/Qwen/Qwen3-Embedding-0.6B/test:embedding_archive_test \
  //model/Qwen/Qwen3-Embedding-0.6B/test:embedding_mixed_test
```

The repository's `amdgpu` build configuration enables the native AMD HAL,
including XDNA. These explicit `manual` targets select an eight-column NPU
context and fixed resident backing. They check independent publisher vectors, exact native
regressions, concurrent and reordered requests, tokenization, explicit token
IDs, output encodings, normalized prefixes, cohort bounds, and joined shutdown.
The [test documentation](test/README.md) records the numerical acceptance and
reference provenance.

Tests trust that provisioned assets match the checked-in acquisition lock.
The card, lock, built providers, server, checker, and test configuration determine
Bazel cache identity. A cache hit reads no model weights. Changing a parameter
file without updating its lock intentionally permits a cached result.
The authored lock covers source files; a publication of newly built bytecode
needs its own lock for those output bytes.
