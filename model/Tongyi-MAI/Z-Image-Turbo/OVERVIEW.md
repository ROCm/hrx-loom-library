## Generate an image

[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) generates
images from text with a Qwen3 text encoder, an eight-step distilled
single-stream diffusion transformer, and a VAE decoder. This Loom package runs
those programs on AMD GPUs using the native libamdf driver. It reads the
publishers' checkpoints directly; parameter preparation and specialization
happen during model preparation.

Start with an installed `hrx-loom` SDK and its optional `hf` extra. The
[SDK installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
describes supported systems and installation. These commands need no source
build or Bazel installation:

```sh
model_url=https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/Tongyi-MAI/Z-Image-Turbo
loom-fetch "$model_url/model.json" --list
loom-fetch "$model_url/model.json" --preset=image --output=./z-image-turbo
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 --model=./z-image-turbo/model.json
```

Fetch reads the adjacent acquisition lock, verifies the selected files, and
writes a local card with the selected preset. It shares cached checkpoint
storage by default. The server uses local files and performs no downloads.
Its listener becomes available before asynchronous model preparation finishes;
`GET /v1/status` reports preparation progress and errors, and `GET /v1/models`
lists ready models.

In another terminal, submit a request and save its PNG response. This POSIX
shell example uses `curl`, `jq`, and `base64`:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Tongyi-MAI/Z-Image-Turbo","prompt":"A small brass robot tending red flowers in a sunlit greenhouse.","seed":42,"response_format":"b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 --decode > image.png
```

The default image is 1024×1024. Supply `"size":"768x1344"` for a different
supported geometry. A seed makes request noise repeatable; another seed selects
a different image. The [image-generation protocol](https://rocm.github.io/hrx-system/loom/serving/http-profiles/#image-generation)
describes the request fields and error responses. Ctrl-C stops admission and
joins accepted work before the server exits.

## Choose a preset

| Preset | Default image | Text stage | Workspace policy |
| --- | --- | --- | --- |
| `image` | 1024×1024 | 512 tokens | Untiled VAE up to 1024 pixels per axis |
| `image-bounded-workspace` | 1024×1024 | 512 tokens | 1.5 GiB bound |
| `lifecycle` | 384×384 | 512 tokens | Untiled VAE |

All presets read the same BF16 transformer, text encoder, VAE, and tokenizer.
The distilled checkpoint runs eight denoising steps without classifier-free
guidance, so one transformer evaluation serves each step.

The decoder runs untiled while the latent fits within 128 pixels on each axis,
which covers images up to 1024 pixels per side. Larger axes, and the
bounded-workspace preset when its bound is below the untiled requirement, use
overlapping VAE tiles with thirty latent pixels of overlap and no blending.
Global VAE attention always runs on the full latent before tiling. GroupNorm
statistics are computed per tile, so a tiled decode approximates the reference
decoder, while an untiled decode does not. Preparation rejects a geometry whose
complete stage cannot satisfy the bound. The bound controls compiled command
workspace, not total model memory or checkpoint storage.

Physical backing also follows the server's slab geometry. Smaller
`--weight_slab_bytes` values reduce rounding and idle retention, with more
allocation and mapping work. `--memory_bytes` caps the shared physical budget;
idle backing can be reclaimed by the idle policy or explicit trim. The
[server memory controls](https://rocm.github.io/hrx-system/loom/serving/workflows/operate-model-server/#trim-without-changing-the-budget)
explain those controls separately from the model's workspace bound.

The tokenizer applies the encoder's Qwen3 chat framing and right-pads with
`<|endoftext|>` or right-truncates the framed prompt to the 512-token stage,
matching the reference pipeline. The transformer reads only the live caption
rows, padded to a multiple of 32 with its learned pad token. Request-specific
geometry is specialized at runtime. Neither prompt length nor image size
selects another checkpoint file.

## Exact upstream files

The `bf16` variant reads the text encoder, VAE, and tokenizer from the
[original checkpoint](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/f332072aa78be7aecdf3ee76d5c247082da564a6):

- `text_encoder/model-00001-of-00003.safetensors`
- `text_encoder/model-00002-of-00003.safetensors`
- `text_encoder/model-00003-of-00003.safetensors`
- `vae/diffusion_pytorch_model.safetensors`
- `tokenizer/tokenizer.json`

The original repository stores the transformer in F32 across three shards
(24.6 GB). The package instead reads the single BF16 transformer file (12.3 GB)
from
[Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo/tree/6fc90a3b1b653e935a0d175e260736de25b84df5):

- `split_files/diffusion_models/z_image_turbo_bf16.safetensors`

That file holds the original F32 parameters rounded to nearest-even BF16, the
precision the programs read their matrix operands in, with each block's query,
key, and value projections concatenated into one `attention.qkv.weight`, the
layout the fused projection reads. Spot checks of 30 tensors and row slices
across both refiners, the first and last blocks, the input embedder, and the
output layer match the rounded original values exactly.

The other three components stay on the original repository. Comfy-Org has no
tokenizer; its Qwen3-4B file holds the same 398 BF16 tensors as the original
encoder shards; and its VAE is an F32 copy under the original FLUX autoencoder
names, where the original repository's BF16 decoder is what the programs read.

The generated file tables identify each source path, its local destination,
and its role. The transformer loads into `parameters.dit`, the three encoder
shards into `parameters.text_encoder`, and the decoder into `parameters.vae`.
`tokenizers.text` identifies the tokenizer independently of its filename.
The transformer reads the encoder's penultimate hidden state, so 35 of the 36
encoder blocks run and the last block and final norm are loaded but not used.

The checkpoints are published under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), as declared
in the [publisher's model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
which also describes model training, intended uses, and limitations.

## Geometry and program choices

Width and height must be positive multiples of 16, at most 8192. The image
patch count, `(height / 16) * (width / 16)`, must be at least 32 and divisible
by 32. Text capacity is a positive multiple of 32. Image patches plus text
positions must fit within 65,536 positions. Unsupported image geometry fails
instead of silently resizing the output.

The source publication contains Loom text providers and an acquisition lock.
Fetch selects it automatically. Authors can also build bytecode publications;
both retain multiple linkable providers and use the same runtime specialization
contract. `--programs` chooses among published sets; `--preset` chooses
execution settings. Consumers of the source publication do not need to build
bytecode.
