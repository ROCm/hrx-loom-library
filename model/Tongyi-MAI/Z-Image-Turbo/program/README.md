# Z-Image-Turbo image programs

The model's image ABI has four CPU callbacks: `image_prepare_v1`,
`image_prepare_request_v1`, `image_prepare_batch_v1`, and `image_generate_v1`.
`prepare.loom` binds the typed `text_encoder`, `dit`, and `vae` sources to
specialized stages. `control.loom` owns the Qwen3 chat framing, request
payloads, stage selection, and execution. `geometry.loom` owns the shared
cold/request admission rules and VAE workspace policy. The server receives
opaque prepared bytes and stage identities instead of reconstructing
model-specific shapes.

`request.loom` generates seeded Philox noise, the prompt rows, the encoder and
transformer key masks, rotary tables, timesteps, and Euler step sizes on the
device. Its reference semantics are the Diffusers `ZImagePipeline`: the
`FlowMatchEulerDiscreteScheduler` shifts `linspace(1, 1/8, 8)` by 3.
`text_encoder.loom` runs the first 35 Qwen3 blocks and keeps the penultimate
hidden state as one `[text_tokens, 2560]` tensor; no final norm applies.
`generate.loom`, `sample.loom`, and `dit_denoise.loom` compose the eight
denoising steps. The two unmodulated context-refiner blocks run once per
request over the caption rows. Each step then runs the two noise-refiner
blocks over the image rows and the thirty main blocks over the unified
`[image, caption]` sequence. `dit_time.loom` forms every step's adaLN
modulation once per request. `vae.loom` and its decoder providers turn the
final latent into packed RGBA8 pixels; the native image service encodes those
bytes as PNG.

Generic tensor algorithms live under `motif/tensor`. These model providers
bind their geometry, configuration, and concrete AMD targets. All matrix
operands are BF16 checkpoint tensors with F32 accumulation; softmax,
normalization statistics, the tanh gates, and the Euler update stay F32, and
each stored activation is rounded to BF16 once. The latent stays F32 between
steps. `weights.loom` is retained as an auxiliary preparation policy even when
the other providers are bytecode. No model checkpoint conversion script is
part of the installation.

`dit_projections.loom` owns the shared `tensor_projection_rdna3`,
`tensor_projection_tiled_rdna3`, and `tensor_projection_wide_rdna3` targets
that the linear motifs bind, and `encoder_attention.loom` owns the shared
expanded-attention target. Each target is defined once per linked program.

The caption frame is `text_tokens` rows, while a prompt usually fills far
fewer. Request preparation marks live caption rows and their learned padding
in the key mask. The main blocks' caption projections read that mask: a
64-row tile whose first row is unattended stores zeros instead of contracting.
The mask is a prefix, so every row in such a tile is unattended, and zero rows
stay finite under the masked attention. Attended tiles run the shared tiled
projection unchanged. The main-block attention counts the visible caption
blocks from the same mask, ends its key loop there, and stores zeros for a
query block made only of unattended rows. Skipped keys would have contributed
zero probability, so the image rows do not change.

## Exact spatial decoding

`vae.loom` unpatchifies the transformer's packed F32 latent, undoes the latent
normalization, and runs the 16-channel input convolution in one pass.
`vae_decoder.loom` runs the remaining input path, the first mid-block residual,
and the single-head spatial attention on the full latent grid. The remaining
mid residual, the four up stages, and the RGBA output head form a body that
runs either once on the full grid or once per overlapping decoder tile.
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
