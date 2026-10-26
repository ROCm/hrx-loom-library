# Author or adapt a model

A Loom model is a portable card plus the programs that implement its services.
The card connects those programs to exact compatible checkpoints, tokenizers,
and named presets. The library provides examples and reusable components;
publication does not require admission to the official catalog.

## Start from a fork

Fork this repository and clone your fork. From its root, select your installed
Loom SDK and build the small Moonshine package as a concrete starting point:

```sh
python dev.py setup --sdk
bazelisk build //model/moonshine-ai/moonshine-streaming-tiny:source //model/moonshine-ai/moonshine-streaming-tiny:archive //model/moonshine-ai/moonshine-streaming-tiny:mixed
```

The [development setup](development.md#select-a-compiler) lists prerequisites
and shows how to select a local HRX checkout. That selection keeps compiler and
model edits in one build graph: there is no wheel reinstall or compiler-copy
step between edits and the next model build.

Bazel produces the selected card and its program files. It does not fetch model
weights or put them into the build cache. The [Moonshine development guide](../model/moonshine-ai/moonshine-streaming-tiny/README.md)
shows how to install each built representation and run its native service
checks. Use the corresponding guide for the model you are adapting.

## Put each fact in its owning file

Start from the package closest to your service and architecture, keeping its
reusable dependencies explicit. A model package has these responsibilities:

| File or directory | Responsibility |
| --- | --- |
| `model.json` | Model identity, program providers, supported asset variants, service versions, and presets. |
| `model.json.lock` | Exact identities of the selected acquisition dependencies. |
| `OVERVIEW.md` | Standalone user instructions, upstream links, supported behavior, and licenses. |
| `README.md` | Developer entry point for composition, builds, and independent tests. |
| `BUILD.bazel` | Program libraries, model packages, and generated documentation. |
| `program/` | Model-specific preparation, orchestration, and device implementations. |
| `test/` | Small numerical/state cases and checks against independent model outputs. |
| `LICENSE.*` | License texts for the third-party material the package uses. |

A separate `serve.json` is useful for a deployment's aliases, placement, or
execution overrides. Users can also serve a fetched card directly; serving
configuration is not part of the model's identity.

For a new base model, use `model/<original-publisher>/<base-model>/`, matching
its original Hugging Face organization and name where available. Keep derived
quantizations under the original base model as variants. Changing a checkpoint
format is not enough to make it supported: the parameter loader and programs
must agree on its tensor names, shapes, and representation.

A card's parameter keys name scopes used by the programs. Multiple checkpoint
files can populate the same scope in order. Tokenizers and voice recordings
have explicit roles; they are not inferred from arbitrary filenames. Origins
name reusable acquisition locations, and each file can specify a different
remote name and local destination. When remixing a package, update program
origins to your fork while retaining the intended upstream checkpoint origins.

Program sets describe linker inputs and libraries. Source, bytecode, and mixed
sets can implement the same service; bytecode still supports linking and
runtime specialization. JIT `config` values select program behavior, while
presets record canonical execution choices. Device placement and request-time
adapters remain deployment or request choices.

The [model-card reference](https://rocm.github.io/hrx-system/loom/serving/model-documents/#portable-model-cards)
and its [JSON schema](https://github.com/ROCm/hrx-system/blob/main/loom/py/loom/model/model-card.v1.schema.json)
own the complete format. Keep the card's schema and service versions explicit.
The [service-program guide](https://rocm.github.io/hrx-system/loom/serving/model-programs/)
defines the callbacks the programs implement.

## Reuse and check the implementation

Keep architecture-specific parameter interpretation and orchestration in the
model. Shared service policy belongs in `program/`, concrete launch contracts
in `kernel/`, and reusable representations and algorithms in `motif/`.
[Repository layers](development.md#repository-layers) explains these boundaries.

Model programs use `loom_library` from `//build_tools/bazel:defs.bzl`.
A `loom_model` target selects a card preset, maps program files through `files`,
and declares importer auxiliary inputs through `data`. External weights,
tokenizers, and recordings retain their acquisition descriptors. Small fixtures
and swatches can be declared build inputs.

Run the small program and numerical tests during iteration:

```sh
bazelisk test --config=asan --config=amdgpu //model/moonshine-ai/moonshine-streaming-tiny/program/... //model/moonshine-ai/moonshine-streaming-tiny/test/...
```

Full-model checks use a separately provisioned installation and the **current
built programs**, with independent inputs and expected outputs. Follow the
model's developer README to select those checks and provide its assets. The
card, lock, programs, and test inputs determine cache identity; full checkpoint
bytes stay outside Bazel. Tests trust the provisioned assets to match the lock,
while `loom-fetch` owns acquisition verification.

For source experiments, `loom-fetch --no-lock` permits intentionally modified
inputs. Named origin overrides can reuse local assets; cards with a `library`
origin can select this checkout with `--origin=library=/path/to/your/fork`.
`--offline` fails if a required input is unavailable locally. The
[fetch guide](https://rocm.github.io/hrx-system/loom/serving/workflows/fetch-and-serve/#reuse-local-assets-and-configure-a-deployment)
explains these controls. The [live-development guide](https://rocm.github.io/hrx-system/loom/serving/workflows/live-model-development/)
covers compatible program replacement on a running server.

## Document and publish

Write `OVERVIEW.md` so it works without the surrounding repository: introduce
the model, link its upstream sources, state supported inputs and hardware,
and show acquisition followed by a real request. Use absolute publication
links in this overview. The generated reference appends variant and preset
settings, exact asset URLs, scopes, and destination paths from the card:

```sh
bazelisk build //model/moonshine-ai/moonshine-streaming-tiny:reference
```

The output is
`bazel-bin/model/moonshine-ai/moonshine-streaming-tiny/reference.md`. The action
reads the card, overview, and optional lock, without downloading weights.
Outside Bazel, the SDK provides the same generator:

```sh
python -m loom.model.documentation --card model.json --overview OVERVIEW.md --card-url https://example.org/my-model/model.json --output reference.md
```

Refresh acquisition locks with `loom-fetch --update-lock` for the exact set you
intend to publish. Generated bytecode needs a lock for its own bytes; the source
package's lock cannot attest to a different program representation. Lock updates
and asset acquisition run outside Bazel. Keep development output and locks
outside `bazel-bin`, as the [Moonshine package guide](../model/moonshine-ai/moonshine-streaming-tiny/README.md#install-built-providers)
demonstrates.

Publish the card, its adjacent lock, and referenced program files at locations
reachable through their declared paths and origins. Retain the source and
checkpoint licenses. A GitHub repository, a Hugging Face repository, or another
HTTPS host can supply those files; users pass the raw `model.json` URL to
`loom-fetch`, then the installed card to `loom-serve`.

An independent model repository can use the SDK format and
[public HRX build rules](https://rocm.github.io/hrx-system/loom/workflows/build-with-bazel/#depend-on-hrx)
without this repository's layout or policy wrappers. Reusable library providers
can be dependencies instead of copied implementations. Neither a fork nor an
independent model needs an entry in the official `catalog.json`.

For a proposed official addition, follow the [contribution scope](../CONTRIBUTING.md#scope)
before doing a large port. The official catalog is generated from its selected model cards
with `python -m loom.model.catalog`; `//build_tools/bazel:catalog_test` checks it
against those cards without reading weights. Its checked-in BUILD declaration
is the authoritative list, not a second inventory in this guide.
