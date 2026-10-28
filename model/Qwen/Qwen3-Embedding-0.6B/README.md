# Developing Qwen3 Embedding

The [standalone model overview](OVERVIEW.md) describes acquisition, serving,
requests, and the supported NPU. The installed SDK fetches Qwen's original
BF16 checkpoint and serves it directly. Running the model requires neither
Bazel nor a weight-preparation script.

From a library checkout, select the local providers and fetch the published
checkpoint with the checked-in lock:

```sh
loom-fetch model/Qwen/Qwen3-Embedding-0.6B/model.json \
  --preset=embedding --origin=library="$PWD" \
  --output=~/models/qwen3-embedding

loom-serve --model=~/models/qwen3-embedding/model.json \
  --device='xdna://0?columns=8' --pool_backing=fixed
```

The `model` parameter scope contains `parameter/model.safetensors` from the
original Qwen repository. The weight policy validates tensor shapes and BF16
data types, selects source row ranges, and asks the runtime to pack the
projection columns into their final device allocations. The model owns that
layout; the runtime owns asynchronous reads, bounded staging, and lifetime.
No transformed checkpoint is produced or required.

## Build source and bytecode programs

```sh
bazel build \
  //model/Qwen/Qwen3-Embedding-0.6B:source \
  //model/Qwen/Qwen3-Embedding-0.6B:archive \
  //model/Qwen/Qwen3-Embedding-0.6B:reference
```

The selected outputs live below
`bazel-bin/model/Qwen/Qwen3-Embedding-0.6B/`. `source` retains Loom text;
`archive` uses bytecode. Both retain `policy.loom` as an explicit auxiliary
compiler input. They remain inputs to runtime linking
and JIT specialization.

Each package contains a selected `model.json`, deployment `serve.json`, and
its provider closure. Full parameters and the tokenizer retain their acquisition
descriptors. Bazel neither fetches nor reads the checkpoint. `reference.md`
combines the standalone overview with card facts.

The [program documentation](program/README.md) explains the fixed 32-core
placement, shared-row ownership, and numerical boundaries. The public roots are
`embedding_prepare_v1`, `embedding_prepare_request_v1`,
`embedding_prepare_batch_v1`, and `embedding_run_v1`. Parameters load through
the `model` index scope and the tokenizer through the `text` identifier.

## Run learned regressions

After fetching the model, run the current built providers on the NPU with
that installation supplying the external assets:

```sh
bazel test --config=asan --config=amdgpu \
  --strategy=TestRunner=standalone \
  --test_env=LOOM_MODEL_ASSETS="$HOME/models/qwen3-embedding" \
  //model/Qwen/Qwen3-Embedding-0.6B/test:embedding_source_test \
  //model/Qwen/Qwen3-Embedding-0.6B/test:embedding_archive_test
```

The repository's `amdgpu` build configuration enables the native AMD HAL,
including XDNA. These explicit `manual` targets select an eight-column NPU
context and fixed resident backing. They check independent publisher vectors,
exact native regressions, concurrent and reordered requests, tokenization, explicit token
IDs, output encodings, normalized prefixes, cohort bounds, and joined shutdown.
The [test documentation](test/README.md) records the numerical acceptance and
reference provenance.

Tests trust that provisioned assets match the checked-in acquisition lock.
The card, lock, built providers, server, checker, and test configuration determine
Bazel cache identity. A cache hit reads no model weights. Changing a parameter
file without updating its lock intentionally permits a cached result.
The authored lock covers assets; editing source programs requires no checksum
update. An immutable bytecode publication can additionally seal its output
bytes with a generated `--lock-scope=all` lock outside the source tree.
