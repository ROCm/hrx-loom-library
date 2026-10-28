# Krea correctness checks

The ordinary test suite requires no learned checkpoint. `geometry_test.loom`
exercises the production request geometry and workspace policy through the VM,
including negative and oversized inputs, alignment, the exact combined-token
limit, and the tile-policy threshold. These are the functions used by both
cold preparation and request-specific stage declarations.

`vae_tile_axis_test.loom` is generated from an independent coordinate reference.
It checks the overlap, last-tile anchoring, midpoint ownership, and complete
axis partition. `vae_tiles_test.loom` checks actual extraction and stitching with
full-image pitches, independent input/output storage, and untouched sentinel
regions. Its seven configurations cover untiled, minimum-overlap, landscape,
portrait, 1024-image, one-axis, and long-axis partitions. Native, access-sanitized,
and race-sanitized executions use the same expected buffers.

The reusable tensor tests own their numerical contracts. In particular,
`motif/tensor/test/linear_i8_test.loom` uses an independent integer-dot and F32/BF16
reference for signed extremes, scales, rounding boundaries, row/column tails,
K64/K128 panels, and the model's 6144/16384 reduction lengths. This establishes
arithmetic correctness separately from the final image regression.

## Full model regression

`image_check.py` consumes a built card plus a `loom-fetch` installation
provided through `LOOM_MODEL_ASSETS`. Programs always come from the build;
checkpoint files stay in the external installation and are shared in place.
The test trusts those files to match the authored lock, which is a declared
Bazel input. No checkpoint is downloaded, prepared offline, or archived by the
test action.

The checker launches and joins a native server, requests real images, decodes
the PNG response, and compares whole RGB digests from `reference.json`.
The fixed requests cover 384×384 and 1024×1024, with both 128- and 512-token text
stages for both checkpoint variants. The same INT8 reference checks ordinary and bounded-workspace
presets, so tiling must preserve every pixel. These references establish port
and representation equivalence for the pinned checkpoints; they are not a
perceptual-quality evaluation or a performance measurement.

The decoder is test-only and uses the Python standard library. Independent
filtered rows test its five PNG predictors, checksum handling, geometry,
and rejection of unsupported formats. The installed model itself runs native
Loom programs and needs no Python model implementation or preparation script.

Lifecycle packages also check concurrent deterministic requests, changed seeds,
graceful and reset-peer disconnection, terminal outcome accounting, and a
subsequent clean replay. A disconnect can race successful completion; either terminal outcome
must retire the request exactly once. Protocol and model-level rejections are
checked separately, followed by another successful request.

The full tests are explicit `manual` targets. Their card, lock, providers,
server, checker, and configuration participate in cache identity. Editing
external weights without updating the lock intentionally does not invalidate
a cached result. The parent README gives provisioned execution commands.

## Original adapter regression

`adapter_check.py` loads the original softwatercolor checkpoint named in
`reference.json` through the real runtime preparation policy. Its four manual
targets cover source and bytecode providers with BF16 and INT8 base
checkpoints. Supply `LOOM_KREA_SOFTWATERCOLOR` as the absolute checkpoint path,
alongside `LOOM_MODEL_ASSETS` for the selected base installation.

The checker compares complete decoded images for one, two, three, four, and
seven selections; independent positive and negative strengths; a second name
for the same original file; and reverse-order replay. Signed-zero selections
must reproduce the base image. Unknown IDs fail even at zero strength, and
non-finite strengths fail at the HTTP input boundary.

Public drain/undrain operations hold concurrent requests until all clients
have arrived. Native execution counters establish that the requests actually
share cohorts, while per-image digests check differing selection counts and
reused scratch. An abortive client disconnect is followed by a deterministic
adapted replay and joined shutdown. The separate tensor family corpus owns
independent numerical error bounds; these image references record complete
model regression behavior for the pinned original files. Each ordered list
has its own reference: changing reduction order can change BF16 rounding and
the resulting image. The arithmetic bounds do not establish a final-image
error tolerance.

`adapter_binding_test.loom` separately executes the real source-selected callback
through the VM. Complete byte expectations check two cohort members with
different counts, repeated and nonmonotonic offsets beyond four GiB, unchanged
strengths and packet bytes, and an unmodified native offset table.

The adapter identity is test data rather than a model-card acquisition
dependency. Changing that identity in `reference.json` invalidates the test
cache. Its payload, like the base checkpoints, never enters Bazel.

The `_banks` and `_pressure` targets additionally use the original retroanime
file through `LOOM_KREA_RETROANIME`. They exercise two distinct banks with the
BF16 base in both provider representations. The bank check covers mixed
strengths, source order, repeated selections, cancellation, and replay. The
pressure check rejects an oversized union before the first successful capture,
alternates the two originals in space for one bank, rejects the union again,
and verifies that subsequent single-bank and base requests remain exact.
