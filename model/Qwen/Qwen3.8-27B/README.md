# Qwen3.8-27B

This package composes quantized text chat, checkpoint multi-token prediction,
DFlash2 drafting, images, and timestamped video frames through Loom's text service.
[The model overview](OVERVIEW.md) describes the exact upstream files, presets,
and installed-SDK workflow. [model.json](model.json) and its
[acquisition lock](model.json.lock) are the portable installation inputs.

A user with the SDK can fetch the source preset and serve the resulting local
card without running Bazel. Authors use Bazel to check the numerical programs
and produce equivalent source or relocatable-bytecode packages:

```sh
bazel build //model/Qwen/Qwen3.8-27B:source_vision \
  //model/Qwen/Qwen3.8-27B:archive_vision
bazel test --config=asan --config=amdgpu \
  //model/Qwen/Qwen3.8-27B/test/...
```

The unsuffixed `source` and `archive` targets select `chat`.
Suffixes `_vision`, `_target_only`, `_dflash2`, and `_cdna_text` select the
corresponding card presets. Each target publishes its selected `model.json`,
`serve.json`, and exact compiler inputs below its output directory.
The full checkpoint and tokenizer remain acquisition descriptors; a build
neither downloads them nor puts them into Bazel's content cache.

The authored asset lock participates in the tests' cache identity. Programs
remain editable without checksum updates. An immutable publication can also
seal its program bytes with `loom-fetch --update-lock --lock-scope=all` outside
Bazel, writing the generated lock and installation to a directory you own.
That complete lock belongs with the published artifacts, rather than the
source tree.

## Composition and ownership

The [BUILD declaration](BUILD.bazel) names the service's primary providers and
library roots. Providers own their direct dependencies; Bazel derives the
source and bytecode file lists from that graph. After changing those
dependencies, `python dev.py model-card --fix` synchronizes the card's compiler
lists. The [authoring guide](../../../docs/model-authoring.md#derive-compiler-files-from-provider-dependencies)
explains the generated fields and repository-relative installation layout.

The [program package](program/README.md) binds named `model`, `vision`, and
`draft` parameter scopes to shape-specialized stages. The `qwen38.draft_kind`
JIT setting selects checkpoint MTP or DFlash2; draft depth and continuation
are execution settings. Shared text policy owns request packing, chat
formatting, retained request state, and the versioned service callbacks.
Model commands own the 64-layer target, predictor, and projector composition.

[Device implementations](program/kernel/README.md) bind model shapes and
concrete targets to the shared GGML carriers, matrix/tensor algorithms, and
sequence kernels. Private helpers stay within their provider. Test-only
oracles and check programs live in [test/](test/README.md), including the
observation roots that need to inspect a private implementation contract.

`weights.loom` and `dflash_weights.loom` are explicit auxiliary compiler inputs.
They describe parameter preparation independently of source/bytecode selection.
A bytecode representation may still need those text policies; representation
names do not determine file type or parameter roles.
