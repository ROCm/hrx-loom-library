## About the model

[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) generates images from
text using a text encoder, an eight-step diffusion transformer, and a VAE
decoder. The card describes both the original BF16 transformer and the
[INT8 ConvRot derivative](https://huggingface.co/Comfy-Org/Krea-2). Both use the
original text encoder, tokenizer, and VAE. The file tables below give the exact
repository paths and their independent local names.

**Execution status:** this package contains the model definition and reference.
The native providers declared under `program/` and `library/` are not yet
included, so this card cannot currently be fetched into a runnable Loom model.
Its presets describe the execution contract to qualify when those providers
are ported. Metadata validation does not establish image correctness.

## Checkpoint choices

The `bf16` variant reads `turbo.safetensors` from the
[original checkpoint](https://huggingface.co/krea/Krea-2-Turbo/tree/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7).
It does not also require the alternate Diffusers transformer shards. The
`int8-convrot` variant instead reads
`diffusion_models/krea2_turbo_int8_convrot.safetensors` from the
[quantized checkpoint](https://huggingface.co/Comfy-Org/Krea-2/tree/1171f4dc7e325bf3734586942e737872506c73e1).
That representation uses signed I8 projection matrices, F32 channel scales,
and Hadamard-256 rotation. It is a particular prepared checkpoint, not an
instruction to quantize arbitrary weights while downloading.

Each variant loads its transformer into `parameters.dit`, its encoder into
`parameters.text_encoder`, and its decoder into `parameters.vae`.
`tokenizers.text` names the tokenizer explicitly. The `krea2.dit_int8` JIT
setting selects the corresponding transformer implementation. The INT8 path
does not require an additional resident BF16 transformer.

Access to the original model may require accepting the publisher's terms on
the [Hugging Face model page](https://huggingface.co/krea/Krea-2-Turbo) and
authenticating the Hugging Face client. The weights are governed by the
[Krea 2 Community License](https://huggingface.co/krea/Krea-2-Turbo/blob/98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7/LICENSE.pdf).
The quantized repository carries the same license designation. Request-specific
adapters are separate from this base-model card.

## Image geometry and presets

`image` and `image-int8` select 1024×1024 images and a 512-token text stage.
`lifecycle` uses 384×384 images for a smaller execution check. Width and height
must be positive multiples of 16, at most 8192. The image patch count is
`(height / 16) * (width / 16)` and must be at least 16 and divisible by 16.
Text capacity is a positive multiple of 16. Image patches plus text tokens
must fit within 65,536 positions. Unsupported geometry fails rather than
silently resizing the image.

The ordinary presets use the untiled VAE decoder. `image-bounded-workspace`
selects the INT8 ConvRot variant at 1024×1024 with a 512-token text stage and
adds a 768 MiB workspace bound. Its decoder uses exact overlapping tiles when
needed, preserving the untiled pixels without blending tile boundaries.
Preparation still rejects a shape whose complete stage cannot satisfy the
bound. This is a workspace policy, not an estimate of total model memory.

The model's training, intended uses, and evaluation are described in the
[publisher's model card](https://huggingface.co/krea/Krea-2-Turbo).
