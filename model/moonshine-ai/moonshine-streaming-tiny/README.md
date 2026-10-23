# Developing Moonshine Streaming Tiny

The [model overview](OVERVIEW.md) describes ordinary acquisition and serving,
model behavior, and supported inputs. This page covers provider builds and
independent execution checks from a library checkout.

Running the default source model with installed `loom-fetch` and `loom-serve`
requires no Bazel build. The checked-in `model.json.lock` covers that source
selection. Serve the fetched `model.json` directly. Follow the
overview for this user workflow; the commands below are for model development.

## Provider builds

From the repository root:

```sh
bazel build --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny:source \
  //model/moonshine-ai/moonshine-streaming-tiny:archive \
  //model/moonshine-ai/moonshine-streaming-tiny:mixed
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
| `mixed` | Loom text | Loom bytecode |

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
development inputs; subsequent fetches use the same `--lockfile` without `--update-lock` to
check those identities. The installed `model.json` is published only after all
selected files have been acquired successfully. Default shared materialization
reuses cache files rather than making one weight copy per provider choice.
The installed card serves directly; its model ID is the request selector.
Use `archive` and `mixed` to install the other representations independently.

## Checks

Checkpoint-free numerical/state checks remain hermetic Bazel tests:

```sh
bazel test --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny/program/... \
  //model/moonshine-ai/moonshine-streaming-tiny/test/...
```

The learned check is an explicitly invoked executable, not a cached test with
ambient weights. After the preceding fetch, run:

```sh
bazel run --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny/test:transcription_check -- \
  --model="$INSTALL/model.json" --device=amdgpu \
  --output="$PWD/.reports/model/moonshine/check-$REPRESENTATION"
```

The output directory must be new. The checker verifies the checkpoint/tokenizer
and audio against their independent recorded digests, launches a fresh real
server, uploads the PCM, checks exact reference text, and joins shutdown before
writing a successful result. Device selection is explicit; the example selects
the AMDGPU driver enabled by its build configuration. Run it once for each
installed representation. It never fetches model files or invents a fallback
when files are absent. These checks establish correctness, not performance.

[Test provenance](test/README.md) describes the independent eager F32 oracle
and synthetic audio attribution. The upstream model license is in
[LICENSE.moonshine](LICENSE.moonshine).
