# Z-Image-Turbo model checks

Small authored workloads check request geometry, workspace admission, and
spatial tile ownership without learned parameters. Independent integer and
storage references exercise the same planners and copy commands used by
preparation and execution. Test overviews explain their oracles; scenarios own
shapes, boundary cases, and numerical acceptance.

Reusable numerical checks belong to [tensor motifs](../../../../motif/tensor/test/).
This package owns model composition and the learned regression below. The
[development guide](../../../../docs/development.md#build-and-check) covers
ordinary execution, optional instrumentation, and benchmarks.

## Full model regression

`image_check.py` consumes a built card plus a `loom-fetch` installation
provided through `LOOM_MODEL_ASSETS`. Programs always come from the build;
checkpoint files stay in the external installation and are shared in place.
The test trusts those files to match the authored lock, which is a declared
Bazel input. No checkpoint is downloaded, prepared offline, or archived by the
test action.

The checker launches and joins a native server, requests real images, decodes
the RGBA8 PNG response, verifies opaque alpha, and compares whole RGB digests
from `reference.json`. The fixed requests cover 384×384 and 1024×1024 with the
short prompt, and 1024×1024 with a long prompt. The long prompt frames to more
than 512 tokens inside the Qwen3 chat template, so it checks right-truncation
to the 512-token text stage; the short prompt checks end-of-text padding.

The untiled decoder produces the `rgb_sha256` digests. Under the 1.5 GiB
`image-bounded-workspace` bound, a 1024×1024 decode runs as overlapping tiles
with per-tile GroupNorm statistics, so its pixels legitimately differ from the
untiled image. Those geometries are pinned separately in
`bounded_workspace_rgb_sha256`, which the checker applies whenever the selected
preset declares `workspace_limit_bytes`. The bound leaves 384×384 untiled, and
that digest is shared. These references establish port equivalence and
determinism for the pinned checkpoint; they are not a perceptual-quality
evaluation or a performance measurement.

The decoder is test-only and uses the Python standard library. Independent
filtered rows test its five PNG predictors, checksum handling, geometry,
and rejection of unsupported formats. The installed model itself runs native
Loom programs and needs no Python model implementation or preparation script.

Lifecycle packages also check concurrent deterministic requests, changed seeds,
graceful and reset-peer disconnection, terminal outcome accounting, and a
subsequent clean replay. A disconnect can race successful completion; either
terminal outcome must retire the request exactly once. Protocol rejections and
the model's patch-count rule (64×64 is 16-aligned but yields 16 patches) are
checked separately, followed by another successful request.

The full tests are explicit `manual` targets. Their card, lock, providers,
server, checker, and configuration participate in cache identity. Editing
external weights without updating the lock intentionally does not invalidate
a cached result. The parent README gives provisioned execution commands.

## Decoder output and tiles

`vae_output_test.loom` exercises the production final convolution with sparse
filters selecting three distinct input channels. Known normalized samples cover
saturation, nearest-even ties, and values around quantization boundaries.
Literal byte oracles check packed channel order, opaque alpha, and guards
without sharing the device conversion arithmetic. The general convolution
corpus owns tap and reduction coverage.

`vae_tile_axis_test.loom` checks the axis planner on the VM for even latent
extents up to 1024 against an independent reference that walks pixel centers
in order and never computes the production midpoint formula. A separate case
pins the decoder's fifteen-latent support radius. `vae_tiles_test.loom` runs the
extraction and stitching commands on the device for seven partitions within
the decoder's 128-latent tile cap, including the 80×80 tiling that the bounded
preset selects at 1024×1024, and checks packed-pixel crop ownership byte for
byte.

`geometry_test.loom` checks image admission, the VAE workspace tile policy,
and cohort keys at their request boundaries. Admission cases cover the
32-row caption alignment and the 32-patch minimum and alignment that the
single-stream transformer adds to the 16-pixel extent rule.
