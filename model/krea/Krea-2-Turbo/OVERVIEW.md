## Generate an image

[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) generates images from
text with a text encoder, an eight-step diffusion transformer, and a VAE
decoder. This Loom package runs those programs on AMD GPUs using the native
libamdf driver. It reads the publisher's checkpoints directly; parameter
packing and specialization happen during model preparation.

Start with an installed `hrx-loom` SDK and its optional `hf` extra. The
[SDK installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
describes supported systems and installation. These commands need no source
build or Bazel installation:

```sh
model_url=https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/krea/Krea-2-Turbo
loom-fetch "$model_url/model.json" --list
loom-fetch "$model_url/model.json" --preset=image --output=./krea2
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 --model=./krea2/model.json
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
  -d '{"model":"krea/Krea-2-Turbo","prompt":"A small brass robot tending red flowers in a sunlit greenhouse.","seed":42,"response_format":"b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 --decode > image.png
```

The default image is 1024×1024. Supply `"size":"768x1024"` for a different
supported geometry. A seed makes request noise repeatable; another seed selects
a different image. The [image-generation protocol](https://rocm.github.io/hrx-system/loom/serving/http-profiles/#image-generation)
describes the request fields and error responses. Ctrl-C stops admission and
joins accepted work before the server exits.

## Choose checkpoint precision and workspace

| Preset | Transformer | Default image | Text stage | Workspace policy |
| --- | --- | --- | --- | --- |
| `image` | Original BF16 | 1024×1024 | 512 tokens | Untiled VAE |
| `image-int8` | INT8 ConvRot | 1024×1024 | 512 tokens | Untiled VAE |
| `image-bounded-workspace` | INT8 ConvRot | 1024×1024 | 512 tokens | 768 MiB bound |
| `lifecycle` | Original BF16 | 384×384 | 512 tokens | Untiled VAE |

For the smaller transformer representation, fetch `--preset=image-int8` into
a separate output directory. Both variants use the same original encoder,
tokenizer, and VAE; the INT8 selection does not also download or retain the
BF16 transformer. Multiple installations can share those cached files.

The bounded-workspace preset uses exact overlapping VAE tiles when required,
without blending tile boundaries. Global VAE attention runs before tiling;
only the spatial decoder is partitioned. Preparation rejects a geometry whose
complete stage cannot satisfy the bound. The bound controls compiled command
workspace, not total model memory or checkpoint storage.

Physical backing also follows the server's slab geometry. Smaller
`--weight_slab_bytes` values reduce rounding and idle retention, with more
allocation and mapping work. `--memory_bytes` caps the shared physical budget;
idle backing can be reclaimed by the idle policy or explicit trim. The
[server memory controls](https://rocm.github.io/hrx-system/loom/serving/workflows/operate-model-server/#trim-without-changing-the-budget)
explain those controls separately from the model's workspace bound.

Short prompts use a retained 128-token text stage; longer prompts use the
512-token stage. The tokenizer preserves the encoder's framing and selects
the retained prompt prefix that fits that stage. Request-specific geometry is
specialized at runtime. Neither prompt length nor image size selects another
checkpoint file.

## Exact upstream files

The `bf16` variant reads `turbo.safetensors` from the
[original checkpoint](https://huggingface.co/krea/Krea-2-Turbo/tree/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7).
It does not require the alternate Diffusers transformer shards. The
`int8-convrot` variant reads
`diffusion_models/krea2_turbo_int8_convrot.safetensors` from the
[quantized checkpoint](https://huggingface.co/Comfy-Org/Krea-2/tree/1171f4dc7e325bf3734586942e737872506c73e1).
That representation uses signed I8 projection matrices, F32 channel scales,
and Hadamard-256 rotation. It is a published checkpoint, not an instruction to
quantize or rewrite weights before serving.

Both variants also require these files from the original model repository:

- `text_encoder/model.safetensors`
- `vae/diffusion_pytorch_model.safetensors`
- `tokenizer/tokenizer.json`

The generated file tables identify each source path, its local destination,
and its role. The transformer loads into `parameters.dit`, the encoder into
`parameters.text_encoder`, and the decoder into `parameters.vae`.
`tokenizers.text` identifies the tokenizer independently of its filename.
The variant's `krea2.dit_int8` setting selects the corresponding JIT program.

Access may require accepting the publisher's terms on the
[Hugging Face model page](https://huggingface.co/krea/Krea-2-Turbo). Fetch reports
the access URL and authentication guidance when needed; `hf auth login`
supplies credentials. The checkpoints use the
[Krea 2 Community License](https://huggingface.co/krea/Krea-2-Turbo/blob/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7/LICENSE.pdf).
The quantized repository carries the same license designation. Model training,
intended uses, and evaluation are described in the
[publisher's model card](https://huggingface.co/krea/Krea-2-Turbo).

## Select an adapter per request

Adapters belong to the serving deployment. The model card continues to describe
the base model and its checkpoint variants. Register local original adapter
files under names that clients can select; request bodies never contain file
paths. The model prepares the F32 factors into shared BF16 banks at runtime.

For example, download the publisher's
[softwatercolor adapter](https://huggingface.co/krea/Krea-2-LoRA-softwatercolor)
beside an installation:

```sh
hf download krea/Krea-2-LoRA-softwatercolor softwatercolor.safetensors \
  --revision=12ff680b14a869b0e72f225bbdb237ad244b6a94 \
  --local-dir=./krea2/adapter/softwatercolor
```

Save this deployment as `serve.json`, next to the `krea2` directory:

```json
{
  "name": "krea2",
  "model": {"path": "./krea2/model.json"},
  "definition": {
    "kind": "image",
    "width": 1024,
    "height": 1024,
    "text_tokens": 512,
    "options": {
      "adapter_capacity_bytes": 268435456,
      "adapters": {
        "watercolor": "adapter/softwatercolor/softwatercolor.safetensors"
      }
    }
  }
}
```

The card path is relative to `serve.json`. Adapter file paths are relative to
the selected local model card; absolute paths can point at an existing cache.
A registration may also contain an ordered array of shard paths, all loaded
into the same parameter index. Factors must use the model's original tensor
names, F32 storage, and rank 32.

```sh
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 --model=./serve.json
curl --fail-with-body http://127.0.0.1:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"krea2","prompt":"A deer grazing in a sunlit forest, Art Deco watercolor style","seed":42,"loras":[{"id":"watercolor","scale":0.75}],"response_format":"b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 --decode > watercolor.png
```

Omitting `loras` selects the base image program. Each selection has its own
finite strength, defaulting to `1.0`; repeated IDs retain their order and
individual strengths. Floating-point accumulation follows that order, so
replaying an image uses the same ordered selection list. Registered
zero-strength selections use the base path without capturing their banks.
Unknown IDs fail even at zero strength.

`adapter_capacity_bytes` is a shared residency budget. The selected distinct
banks must fit; adding another occurrence of an already selected adapter
does not duplicate its factors. Banks can be evicted and prepared again from
their original files. Selection count determines small rank-activation
scratch, while the GPU kernel interface stays fixed. The model imposes no
separate adapter-count cap. These settings work with either base checkpoint
variant and with source, bytecode, or mixed provider packages.

## Geometry and program choices

Width and height must be positive multiples of 16, at most 8192. The image
patch count, `(height / 16) * (width / 16)`, must be at least 16 and divisible
by 16. Text capacity is a positive multiple of 16. Image patches plus text
positions must fit within 65,536 positions. Unsupported image geometry fails
instead of silently resizing the output.

The source publication contains Loom text providers and an acquisition lock.
Fetch selects it automatically. Authors can also build bytecode and mixed
publications; all retain multiple linkable providers and use the same runtime
specialization contract. A mixed publication keeps preparation and request
control as text while using bytecode numerical providers. `--programs` chooses
among published sets; `--preset` chooses execution settings and checkpoint
precision. Consumers of the source publication do not need to build bytecode.
