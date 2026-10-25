# Krea image programs

The model's image ABI has four CPU callbacks: `image_prepare_v1`,
`image_prepare_request_v1`, `image_prepare_batch_v1`, and `image_generate_v1`.
`prepare.loom` binds the typed `text_encoder`, `dit`, and `vae` sources to
specialized stages. `control.loom` owns tokenizer framing, request payloads,
stage selection, and execution. `geometry.loom` owns the shared cold/request
admission rules and VAE workspace policy. The server receives opaque prepared
bytes and stage identities instead of reconstructing model-specific shapes.

`request.loom` generates seeded Philox noise and diffusion constants on the
device. `text_encoder.loom` composes the encoder, and the text-fusion providers
mix its retained layer outputs. `generate.loom`, `sample.loom`, `denoise.loom`,
and the transformer/block providers compose the eight denoising steps.
`vae.loom` and its decoder providers turn the final latent into F32 RGB planes;
the native image service encodes those planes as PNG.

Generic tensor algorithms live under `motif/tensor`. These model providers
bind their geometry, configuration, and concrete AMD targets. The signed-I8
path uses the published ConvRot checkpoint's scales and rotation; the BF16
path reads the original transformer. Both share the original encoder and VAE.
`weights.loom` is retained as an auxiliary preparation policy even when the
other providers are bytecode. No model checkpoint conversion script is part
of the installation.

## Shared request adapters

`adapter_selection.loom` validates deployment options and registers original
files, including ordered shards. Its cold registry retains names and native
source ordinals. Request preparation resolves each occurrence once, keeps its
independent F32 strength, and removes signed zeros before selecting a base or
adapted stage. The registry and parameter files stay outside the model card.

Cold state retains the original encoder, transformer, and VAE source ordinals.
An adapted request declares these same three parameter bindings and adds the
shared arena at binding 3. Native preparation reuses the retained base domains,
captures the distinct adapter banks, and calls `krea2.bind_adapter_banks` once
with their resolved offsets. That exact source-selected callback name is
independent of the versioned image service roles.

`adapter_weights.loom` validates each original F32 factor and declares runtime
packing through `adapter_pack.loom`. `command.parameter.base` retains each
factor's compiler-owned position within the bank. The model-owned
`adapter.loom` leaf exposes the source selection as typed offset and strength
views to generic tensor motifs; those motifs know nothing about request
records, checkpoint paths, or serving APIs.

Compatible cohorts contain up to two images. Each member retains its actual
selection count, while the maximum count sizes shared rank scratch. Execution
reuses that scratch between members and writes disjoint RGB output ranges.
All matrix families accumulate selected contributions before one base update;
the twelve-layer text fusion remains a fused scalar operation. Native bank
leases survive device completion and cancellation, so another request cannot
recycle a still-referenced range.

## Exact spatial decoding

`vae_tiles.loom` owns the decoder's spatial partition and its two copy
commands. Global VAE attention runs before this partition. A tile receives
384 channel-first F32 feature planes; its local decoder produces three F32
image planes at eight times the latent dimensions. Extraction preserves the
full-image channel and row pitches, and stitching copies only the tile's
owned output rectangle. Both commands use separate input and output storage.

The full and tile extents are even values from 2 through 1024. A tile fits
within the full extent and exceeds 30 on every tiled axis. Preparation owns
these relations. Axis planning retains thirty latent pixels of overlap,
anchors the last tile at the far boundary, and assigns each output pixel to
the nearest tile center. Midpoint ties belong to the later tile. The resulting
half-open rectangles partition the image without blending or duplicate writes.

The post-attention decoder has a maximum spatial dependency radius of fifteen
latent pixels. It has eight convolutions at latent resolution, six at each of
the next three resolutions, three nearest-neighbor resize/convolutions, and
one final output convolution. Channel normalization, activation, and clamping
do not enlarge that spatial dependency. The retained overlap therefore keeps
interior output ownership away from artificial tile boundaries; real image
boundaries retain the decoder's original padding.

The kernels read `krea2.latent_height`, `krea2.latent_width`,
`krea2.vae_tile_height`, and `krea2.vae_tile_width` from the composition's
configuration. Origins and crop rectangles come from the axis planner. The
public commands are `krea2.vae_extract_tile` and `krea2.vae_stitch_tile`.
They allocate no intermediate storage; the decoder owns and reuses one feature
tile and one output tile. Sequential tiles share that storage while writing
disjoint portions of the full image.
