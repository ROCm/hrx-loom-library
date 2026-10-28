## About the model

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) supports text chat,
reasoning, tool calls, and image-conditioned conversations. This composition
uses Unsloth's mixed-format UD-Q5_K_XL checkpoint. Optional multi-token
prediction uses the checkpoint's prediction block; DFlash2 instead adds a
separate draft checkpoint. Drafting remains part of the text service.

The composition contains native AMD text and image providers with source and
relocatable-bytecode representations. The source preset works with
the installed SDK without building the library. After installing `hrx-loom`
with its `hf` extra, select text chat:

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/Qwen/Qwen3.8-27B/model.json \
  --preset chat --output "$HOME/models/qwen3.8-27b"
loom-serve --model="$HOME/models/qwen3.8-27b/model.json" \
  --device=amdgpu://0
```

The `chat` preset uses checkpoint MTP without the image projector;
`target-only` selects ordinary target decoding. Select `vision` to fetch the
image projector, or `dflash2` to fetch the separate draft checkpoint. For example:

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/Qwen/Qwen3.8-27B/model.json \
  --preset dflash2 --output "$HOME/models/qwen3.8-27b-dflash2"
loom-serve --model="$HOME/models/qwen3.8-27b-dflash2/model.json" \
  --device=amdgpu://0
```

Fetch resolves the exact asset and compiler-input closure and reuses cached
weights across these installations; serving opens the resulting local card.
The implementation and numerical checks are published
in the [library model package](https://github.com/ROCm/hrx-loom-library/tree/main/model/Qwen/Qwen3.8-27B).

## Choose the asset closure

All variants use the original
[tokenizer](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json)
and the exact
[UD-Q5_K_XL file](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-UD-Q5_K_XL.gguf).
The quantized source is pinned to the checkpoint revision used by this
composition. Later files with the same upstream name have different bytes;
the card and lock retain the exact compatible revision and identity.

| Variant | Additional parameter scope | Additional file |
| --- | --- | --- |
| `ud-q5-k-xl` | None | Text-only, including the checkpoint's MTP block. |
| `ud-q5-k-xl-vision` | `vision` | [Matching F16 projector](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/mmproj-F16.gguf). |
| `ud-q5-k-xl-dflash2` | `draft` | [DFlash2 Q4_K_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/resolve/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q4_K_M.gguf). |

The text-only choices do not require the projector. Image conditioning is a
versioned `image` component of the text service, with separate preparation and
request control. It is not an image-generation service. The DFlash2 checkpoint
is additional model storage, not a request adapter.

Text, checkpoint MTP, vision, and DFlash2 have native service checks for source and
relocatable-bytecode programs. The checks exercise concurrent
requests, exact responses, checkpoint forks and continuation, parameter
unload/reload, disconnected clients, and joined shutdown. Vision also checks
independent projector residency. DFlash2 retains the eight-row preset and its
seven-proposal depth. CDNA text is checked on MI300 with the `cdna-text` preset.

The original model, selected quantization, and selected draft repository declare
Apache-2.0. Their original cards are available from
[Qwen](https://huggingface.co/Qwen/Qwen3.8-27B),
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), and
[Z Lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF).

## Chat and drafting presets

`chat` and `vision` use eight execution rows, 32,768 positions per context,
65,536 pooled positions, 512-token packed epochs, three MTP proposals, and
two-epoch continuation. Every preset reserves four explicit checkpoint slots
for conversation forks and reuse across parameter unloading. These retained
endpoints are independent of the execution rows. `target-only` disables
drafting and continuation.
`dflash2` uses seven proposals and the separate draft parameter scope. The
`qwen38.draft_kind` JIT setting distinguishes checkpoint MTP (`0`) from
DFlash2 (`1`); a filename is not used to infer the drafting algorithm.

The numerical provider set contains RDNA and CDNA text kernels. `cdna-text`
records the text-only configuration of four rows, 4,096-position contexts,
32,768 pooled positions, and 512-token epochs without drafting. The image
projector requires the RDNA implementation. Presets record model execution
capacities; device selection remains part of the serving deployment.

The canonical requests use greedy sampling and disable thinking. Requests may
select temperature, top-p, top-k, and a signed 64-bit seed. Reasoning is returned
as `reasoning_content`, independently of answer content and tool calls.
Exact retained-history reuse requires the complete matching conversation,
including reasoning and tool results. Speculative acceptance commits only
target-approved tokens and state.

## Image contract

The vision variant accepts up to four images in conversation order. Each may be
at most 4096×4096 pixels and 64 MiB compressed, with aspect ratio at most 200:1.
Projection preserves aspect ratio with 32-pixel processing alignment and a
default budget of 1,024 image tokens. A request can select 128 through 16,384
tokens per image; all image and text tokens share the text context capacity.

Image orientation, alpha compositing over white, and antialiased resizing occur
before feature extraction. Actual container support follows the host's codecs.
Projector weights may be activated independently of retained text state; live
feature rows have their own lifetime. Device placement and HTTP routing remain
serving configuration rather than model identity.
