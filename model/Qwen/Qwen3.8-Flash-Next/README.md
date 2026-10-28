# Qwen3.8 Flash-Next

This composition runs [Qwen3.8 Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) through Loom's text service using the [Unsloth UD-Q4_K_XL checkpoint](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL), with optional images, timestamped video frames, and Q8 MTP. It includes the model programs, shared Qwen service providers, exact acquisition lock, numerical checks, and learned checkpoint/fork regressions. The device programs target AMD RDNA GPUs through libamdf.

## Fetch and serve

With the `hrx-loom` SDK and its optional `hf` extra installed, run these commands from the library checkout:

```sh
loom-fetch model/Qwen/Qwen3.8-Flash-Next/model.json \
  --preset chat \
  --output "$HOME/models/qwen3.8-flash-next"
loom-serve --model="$HOME/models/qwen3.8-flash-next/model.json" \
  --device=amdgpu://0
```

Fetching selects the source providers and the four required GGUF shards, verifies their acquisition identities, and materializes one selected card. Shared providers follow the same checkout or published snapshot as the model card. A published card URL can replace the local card path; neither flow needs an origin override or a Bazel build.

The target bank occupies approximately 103.68 GiB before request state, compiler memory, or workspace. MTP adds about 2.59 GiB. The [standalone model overview](OVERVIEW.md) explains the checkpoint layout, exact upstream links, capacities, and additional retained storage. Changing a preset does not require another copy of the same checkpoint files.

| Preset | Parameter variant | Execution |
| --- | --- | --- |
| `chat` | `ud-q4-k-xl` | Four execution rows and 16,384-position contexts. |
| `mtp` | `ud-q4-k-xl-mtp` | The same capacities with three recursive predictor proposals. |
| `mtp-continuation` | `ud-q4-k-xl-mtp` | Three proposals and two-epoch continuation. |
| `long-context` | `ud-q4-k-xl` | Eight rows and 262,144-position contexts. |
| `long-context-mtp` | `ud-q4-k-xl-mtp` | The same native-context capacities with three predictor proposals. |
| `vision` | `ud-q4-k-xl-vision` | Chat capacities with the original BF16 image projector. |
| `vision-mtp` | `ud-q4-k-xl-vision-mtp` | Images and video frames with three recursive predictor proposals. |

MTP remains part of the text service. The model card selects the predictor parameter scope and JIT configuration; deployment decides devices, process memory, and request admission. Request-time sampling remains a service option. These capacities are configuration contracts, not throughput or peak-memory estimates.

The vision presets acquire `mmproj-BF16.gguf` directly into the separate `vision` scope. They accept `image_url` and `video_frames` parts interleaved with text, up to 768 supplied images/frames within the request and context budgets. Video uses the trained two-frame temporal projection and preserves client timestamps; an odd tail aliases its final frame. The projector activates on demand and can release its weights while retained conversations keep their image-conditioned state. Text-only presets retain their original stage catalog and do not acquire or compile the projector. The [overview](OVERVIEW.md#image-conditioned-chat) includes a complete request.

## Build representations and regressions

`source` and `archive` describe provider representations. A source package retains `.loom` files; an archive package contains `.loombc` providers for runtime linking. Both retain the explicit weight-preparation source as auxiliary compiler data. The programs share the same public service contracts and parameter identities.

The model BUILD file names preparation inputs and runtime-selected library roots; their provider dependencies supply the complete source or bytecode closure. After changing those dependencies, `python dev.py model-card --fix` refreshes the checked-in card.

```sh
bazel build //model/Qwen/Qwen3.8-Flash-Next:source \
  //model/Qwen/Qwen3.8-Flash-Next:archive
bazel test --config=asan --config=amdgpu \
  //model/Qwen/Qwen3.8-Flash-Next/test/...
```

Each representation also has `_mtp`, `_mtp_continuation`, `_long_context`, and `_long_context_mtp` targets. For example, `:archive_mtp` builds the compiled provider package selected by the `mtp` preset. Bazel builds program artifacts and records the card and asset lock as regression inputs; checkpoint bytes stay outside its graph. Editing programs or shared providers requires no lock update. An immutable publication can additionally seal its built program bytes with `loom-fetch --update-lock --lock-scope=all`; that generated lock stays with the published artifacts.

Image compositions use `_vision` and `_vision_mtp` targets, such as `:archive_vision_mtp`. They add the shared image service and model-specific projector to the same text provider closure. The service checks exercise both representations, including video recall through a retained partial sparse-attention pool and replay after a server restart.

The manual learned service tests accept a provisioned installation through `LOOM_MODEL_ASSETS` or `--assets`. They compare independent original-model responses, simultaneous checkpoint forks, continuation across a page boundary, fresh-history replay, weight deactivation/reactivation, client disconnect recovery, and joined shutdown. The [test guide](test/README.md) distinguishes those checks from the small numerical fixtures. The [program guide](program/README.md) describes preparation, recurrent ownership, sparse attention, and speculative publication.
