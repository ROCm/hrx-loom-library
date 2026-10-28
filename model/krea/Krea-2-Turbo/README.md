# Developing Krea 2 Turbo

The [model overview](OVERVIEW.md) gives the installed fetch-to-serve workflow,
checkpoint choices, geometry constraints, and exact upstream files. The
portable inputs are [model.json](model.json) and its
[acquisition lock](model.json.lock). An SDK user can fetch and serve the source
package without building this repository.

## Build the provider packages

From the repository root:

```sh
bazel build --config=amdgpu \
  //model/krea/Krea-2-Turbo:source \
  //model/krea/Krea-2-Turbo:archive
```

The unsuffixed targets select the BF16 `image` preset. Suffixes `_int8`,
`_bounded_workspace`, and `_lifecycle` select the other presets; for example,
`:archive_bounded_workspace` contains the INT8 model with its 768 MiB workspace
policy. `source` retains text and `archive` uses relocatable bytecode. Each
produces a selected `model.json`, a deployment `serve.json`, and the complete
program closure. The full checkpoints and tokenizer remain acquisition
descriptors; none enters Bazel's repositories, action inputs, or content cache.

The model-owned [program package](program/README.md) composes the text encoder,
transformer, and VAE. Generic algorithms remain under `motif/tensor`; model
providers own shapes, targets, and configuration. `weights.loom` is an explicit
auxiliary compiler input in every representation, alongside
`adapter_weights.loom`. They describe runtime parameter preparation. The
overview includes a complete deployment with named original adapter files and
a request selecting an adapter strength.

Generate the standalone reference without acquiring checkpoints:

```sh
bazel build //model/krea/Krea-2-Turbo:reference
```

The result combines the overview with tables generated from the card, so it
can be published independently of this README.

## Acquire and test

Provision the BF16 installation once:

```sh
loom-fetch model/krea/Krea-2-Turbo/model.json \
  --preset=image --output=../krea2
```

The authored lock covers both checkpoint variants and the tokenizer. Shared
motifs follow the model card's checkout or published snapshot through relative
source paths, with no library pins, program checksums, or origin overrides to
maintain. Asset verification is unchanged. Bazel builds the current declared
sources using the pinned SDK or the local HRX override.

Geometry, numerical, and tile tests require no learned weights:

```sh
bazel test --config=asan --config=amdgpu \
  //model/krea/Krea-2-Turbo/program/... \
  //model/krea/Krea-2-Turbo/test/...
```

Run the full image regression against the provisioned installation:

```sh
bazel test --config=asan --config=amdgpu \
  --strategy=TestRunner=standalone \
  --test_env=LOOM_MODEL_ASSETS="$(realpath ../krea2)" \
  //model/krea/Krea-2-Turbo/test:image_source_test \
  //model/krea/Krea-2-Turbo/test:image_archive_test
```

These explicit `manual` tests run the real server, compare complete decoded
images, exercise geometry changes and prompt capacities, reject invalid
requests, and join shutdown. The `_lifecycle` tests additionally exercise
concurrent seeded replay, changed seeds, client disconnection, and row reuse.
[Test documentation](test/README.md) describes the reference boundaries.

For INT8, fetch `--preset=image-int8` into a separate installation and use the
`_int8` test suffix. The `_bounded_workspace` tests use the same INT8 assets
while checking tiled output against the untiled reference. The card, lock,
current programs, server, and checker determine test cache identity. A cached
test reads no external weights. Changing provisioned bytes without updating
the lock intentionally permits reuse of that cached result.

Adapter regressions use the original softwatercolor file from the overview's
download example:

```sh
bazel test --config=asan --config=amdgpu \
  --strategy=TestRunner=standalone \
  --test_env=LOOM_MODEL_ASSETS="$(realpath ../krea2)" \
  --test_env=LOOM_KREA_SOFTWATERCOLOR="$(realpath ../krea2/adapter/softwatercolor/softwatercolor.safetensors)" \
  //model/krea/Krea-2-Turbo/test:adapter_source_test \
  //model/krea/Krea-2-Turbo/test:adapter_archive_test
```

The `_int8` suffix selects the corresponding INT8 base installation. These
tests cover ordered selections, runtime factor preparation, actual shared
cohorts, rejection, cancellation, and replay. `test/reference.json` pins the
adapter's external identity and participates in the Bazel test key.

Supplying `LOOM_KREA_RETROANIME` enables the explicit
`:adapter_source_banks_test` and `:adapter_source_pressure_test` targets, with
equivalent archive targets. They test distinct original banks and
eviction in a finite arena using the BF16 installation. The test reference
identifies the exact retroanime repository, revision, filename, and digest.

A built package can also be fetched directly. Publication locks describe the
actual output bytes and are generated outside Bazel:

```sh
mkdir -p .reports/krea2
PROGRAMS="$(realpath bazel-bin/model/krea/Krea-2-Turbo/archive)"
loom-fetch "$PROGRAMS/model.json" --offline --update-lock --lock-scope=all \
  --lockfile="$PWD/.reports/krea2/archive.lock.json" \
  --output="$PWD/.reports/krea2/archive"
loom-serve --device=amdgpu --model=./.reports/krea2/archive/model.json
```

This offline example requires the checkpoint files in the Hugging Face cache,
or equivalent named origin overrides. The source lock participates in Bazel's
test identity; it does not attest to newly generated bytecode. Subsequent
acquisitions of the built publication use its new lock without `--update-lock`.
