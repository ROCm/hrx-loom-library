# Developing Moonshine Streaming Tiny

The [model overview](OVERVIEW.md) describes ordinary acquisition and serving,
model behavior, and supported inputs. This page covers provider builds and
independent execution checks from a library checkout.

Running the default source model with installed `loom-fetch` and `loom-serve`
requires no Bazel build. The checked-in `model.json.lock` covers the model's
assets while programs remain editable. Serve the fetched `model.json` directly.
Follow the overview for this user workflow; the commands below are for model
development.

## Provider builds

From the repository root:

```sh
bazel build --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny:source \
  //model/moonshine-ai/moonshine-streaming-tiny:archive
```

These targets build program files and a portable selected card without acquiring
weights or a tokenizer. Runtime assets retain their fetch descriptors, including
the inline rotary data. The small `program/weights.loom` policy is declared
provider auxiliary data and remains part of the build. No full model checkpoint
is a Bazel repository, action input, output, or runfile.

| Target | Prepare/control providers | Command/kernel providers |
| --- | --- | --- |
| `source` | Loom text | Loom text |
| `archive` | Loom bytecode | Loom bytecode |

Each target emits `model.json` and a separate `serve.json` selecting the served
name and preset. The source card supplies the exact provider closure;
changing its representation does not change learned parameters or state layout.

Generate a standalone reference without acquiring runtime assets:

```sh
bazel build //model/moonshine-ai/moonshine-streaming-tiny:reference
```

The output is `bazel-bin/model/moonshine-ai/moonshine-streaming-tiny/reference.md`.

## Install built providers

Use the SDK's `loom-fetch` with the generated card. The Hugging Face provider
reuses the existing cache; `--offline` ensures this example cannot download a
missing checkpoint or tokenizer. Without those cached files, explicit ordinary
acquisition belongs to `loom-fetch`, not a Bazel build or test.

```sh
REPRESENTATION=source
PROGRAMS="$(realpath "bazel-bin/model/moonshine-ai/moonshine-streaming-tiny/$REPRESENTATION")"
INSTALL="$PWD/.reports/model/moonshine/$REPRESENTATION"
LOCK="$PWD/.reports/model/moonshine/$REPRESENTATION.lock.json"
mkdir -p "$PWD/.reports/model/moonshine"

loom-fetch "$PROGRAMS/model.json" --preset=streaming \
  --offline --update-lock --lockfile="$LOCK" --output="$INSTALL"
```

The explicit lock lives outside `bazel-bin`; acquisition never writes beside a
generated build input. Resolving the program directory immediately after its
build keeps later Bazel invocations from redirecting it through a changed
`bazel-bin` configuration symlink. `--update-lock` records the exact selected
asset identities; subsequent fetches use the same `--lockfile` without
`--update-lock` to check those assets while preserving program edits. Add
`--lock-scope=all` when generating an immutable publication that also seals
the built program bytes. The installed `model.json` is published only after all
selected files have been acquired successfully. Default materialization copies
programs and shares assets rather than making one weight copy per provider
choice.
The installed card serves directly; its model ID is the request selector.
Use `archive` to install the bytecode representation independently.

## Checks

Checkpoint-free numerical/state checks remain hermetic Bazel tests:

```sh
bazel test --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny/program/... \
  //model/moonshine-ai/moonshine-streaming-tiny/test/...
```

Learned transcription checks are ordinary cacheable tests. They are tagged
`manual` because their runner needs provisioned weights. After the preceding
fetch, run both representations against that installation:

```sh
bazel test --config=asan --config=amdgpu \
  --strategy=TestRunner=standalone \
  --test_env=LOOM_MODEL_ASSETS="$INSTALL" \
  //model/moonshine-ai/moonshine-streaming-tiny/test:transcription_source_test \
  //model/moonshine-ai/moonshine-streaming-tiny/test:transcription_archive_test
```

This example executes tests locally while leaving the build execution strategy
unchanged. A remote test runner needs the same assets provisioned at its
configured path and an available AMDGPU runtime. The test's hardware requirements
use the shared Loom execution profile.

Each test combines its current built providers and auxiliary files with shared
parameters and tokenizers from `LOOM_MODEL_ASSETS`. Installed program files are
ignored, so one installation supports both tests and compiler iteration
requires no refetch. Model files are linked into temporary storage; retained
test outputs contain only server diagnostics and `result.json`.

The original `model.json`, checked-in `model.json.lock`, built programs, server,
test code, PCM/reference, and test configuration determine the cache identity.
Weights are not hashed, downloaded, or passed through Bazel. Users and CI own
keeping provisioned assets consistent with the lock. Changing them without
updating the lock intentionally permits a cached result. Updating the lock
invalidates tests without recompiling providers. A normal cache hit performs no
asset access. `--nocache_test_results` explicitly requests a fresh execution.

On execution, the checker verifies the small input PCM's digest, launches a
fresh real server, uploads the PCM, checks exact reference text, and joins
shutdown before writing success. Missing runtime files fail visibly. These
checks establish correctness, not performance.

[Test provenance](test/README.md) describes the independent eager F32 oracle
and synthetic audio attribution. The upstream model license is in
[LICENSE.moonshine](LICENSE.moonshine).
