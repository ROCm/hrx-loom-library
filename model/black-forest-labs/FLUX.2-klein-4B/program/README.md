# FLUX.2 klein image programs

The model's image ABI has four CPU callbacks: `image_prepare_v1`,
`image_prepare_request_v1`, `image_prepare_batch_v1`, and `image_generate_v1`.
`prepare.loom` binds the typed `text_encoder`, `dit`, and `vae` sources to
specialized stages. `control.loom` owns the Qwen3 chat framing, request
payloads, stage selection, and execution. `geometry.loom` owns the shared
cold/request admission rules and VAE workspace policy. The server receives
opaque prepared bytes and stage identities instead of reconstructing
model-specific shapes.

`request.loom` generates seeded Philox noise, the prompt rows, the encoder key
mask, rotary tables, timesteps, and Euler step sizes on the device. Its
reference semantics are the Diffusers `Flux2KleinPipeline`: the schedule
shifts `linspace(1, 1/4, 4)` by the empirical mu for the image token count.
`text_encoder.loom` runs the first 27 Qwen3 blocks and retains the outputs of
blocks 8, 17, and 26 as one `[text_tokens, 3, 2560]` tap tensor; no final norm
applies. `generate.loom`, `sample.loom`, and `dit_denoise.loom` compose the
four denoising steps, each running five double-stream and twenty
single-stream blocks over the joint `[text, image]` sequence. `vae.loom` and
its decoder providers turn the final latent into packed RGBA8 pixels; the
native image service encodes those bytes as PNG.

Generic tensor algorithms live under `motif/tensor`. These model providers
bind their geometry, configuration, and concrete AMD targets. All matrix
operands are the checkpoint's original BF16 tensors with F32 accumulation;
softmax, normalization statistics, and the Euler update stay F32, and each
stored activation is rounded to BF16 once. `weights.loom` is retained as an
auxiliary preparation policy even when the other providers are bytecode. No
model checkpoint conversion script is part of the installation.

`dit_projections.loom` owns the shared `tensor_projection_rdna3` target that
the encoder projections also use, and `encoder_attention.loom` owns the shared
expanded-attention target that the transformer attention also uses. Each
target is defined once per linked program.

## Exact spatial decoding

`vae_decoder.loom` runs the input path, the first mid-block residual, and the
single-head spatial attention on the full latent grid. The remaining mid
residual, the four up stages, and the RGBA output head form a body that runs
either once on the full grid or once per overlapping decoder tile.
`vae_tiles.loom` owns the partition and its extraction and stitching commands.

Decoder tiles are at most 128 latent pixels per side, so the 8x output stage
stays within the 1024-pixel convolution and normalization domain. Preparation
sets the tile equal to the latent grid when the latent fits. Axis planning
retains thirty latent pixels of overlap, anchors the last tile at the far
boundary, and assigns each output pixel to the nearest tile center; the
resulting half-open rectangles partition the image without blending or
duplicate writes.

`vae_norm.loom` computes GroupNorm(32) statistics in two levels: each
4096-element chunk stores its mean and centered sum of squares, and one
workgroup per group merges them with the parallel-variance identity. In a
tiled decode those statistics cover one tile rather than the full image, so
tiled output approximates the reference decoder. Untiled output does not.

The final convolution writes row-major RGBA8 directly. RGB values are clamped
and mapped from `[-1, 1]` to `[0, 255]` with nearest-even rounding; alpha is
255. Tiled decoding copies complete packed pixels, and each cohort member owns
four bytes per output pixel.
