# Qwen3.8 vision algorithms

These motifs own the shared 1152-wide vision encoder's arithmetic and storage
contracts. The dense and MoE models use different matrix precision and final
feature widths; checkpoint names, geometry selection, launches, and retained
output ownership belong to their composition leaves.

`resize.loom` consumes the shared image-service packet, resolving RGB/RGBA
orientation and straight alpha while filtering. An 8×32×3 resident F32 panel
reuses horizontal samples across sixteen output rows without an image-sized
intermediate. The leaf supplies a wave32 launch and selects one cohort image;
the host's exact coefficient plans and nearest-even byte output are preserved.

`workspace.loom` defines the allocation lengths and aligned private peak used
by both command composition and admission forecasting. A final feature width
of at least 4608 can hold one temporary residual bank; narrower outputs need
two private banks. The activation bank serves RGB, context/K/V, FFN, and merger
at disjoint dispatch lifetimes. K/V origins depend on the resident cohort
count, so sequential and concurrent head schedules share the same layout rule.
Image count affects retained output, while sequential images reuse workspace.

`attention.loom` keeps query projection and online softmax in one resident
program, streaming K/V panels without a global score matrix. Each workgroup
owns sixteen queries and four heads, with F32 maxima, sums, and accumulation.
Padded resident rows distribute LDS accesses across banks; query storage
changes representation only after the preceding reads complete. The caller
selects a logical head cohort, its K/V byte origin, and the first row in
stacked residual/statistics/context storage independently. Each segment keeps
its own rotary coordinates and softmax, while linear operators can consume
all completed context rows together. This permits sequential bank reuse,
concurrent heads, and independent frame pairs. A separate shared output projection
reuses each matrix panel across 64 rows and adds the F32 residual. Original
F16 and BF16 checkpoints share both algorithms through explicit precision.

`key_value.loom` projects four 72-channel heads at a time and applies the
trained two-dimensional NEOX rotary coordinates to keys. It publishes row-major
keys and transposed values, keeping 32-row weight reuse, cyclic resident weight
rows, and padded input/transpose storage. The caller supplies the logical head
cohort and output byte origin independently: dense composition reuses one bank
after each attention dispatch, while concurrent cohorts own disjoint banks.
Original F16/BF16 weights use the same arithmetic without global conversion.
The independent K/V checks exercise the dense launch and concurrent BF16
cohorts, including the four-channel head tails and a partial row tile.

`merger.loom` turns each ordered group of four patch rows into one language
feature. The up projection consumes the F32 4608-wide row and applies tanh
GELU; the down projection emits the caller's feature width in F32. Dense and
Flash-Next select 5120 and 2560 columns respectively. Original F16/BF16 weights
and the rounded intermediate use the precision provider, while output storage
and permission to reuse an earlier residual stay with the command owner.
F16 up keeps conversion live ranges within one panel; BF16 up and both down
projections overlap two panels. Precision specializes this schedule without
introducing runtime branches or changing external reuse.

`panels.loom` owns matching XOR input stores/reads and cyclic weight reads for
the FFN and merger. Its raw half-bit layouts preserve reuse without global
packing or padding. Callers own storage and synchronization; the provider owns
the correspondence between logical matrix coordinates and physical LDS rows.

`patch.loom` projects packed RGB8 directly into the trained 2×2 patch order.
It retains two independent temporal weight contractions in F32. Two plane
offsets select real successive frames; equal offsets repeat a still image or
an odd final frame and share one resident input panel. Checkpoint storage and
operand precision are independent specialization facts: original dense F16 is
read unchanged, while original Flash-Next F32 weights round to BF16 inside the
resident tile.
No image-sized patch matrix or converted checkpoint is materialized.

`precision.loom` provides rounding, result publication, and native F16/BF16
matrix interpretation for the shared projections. It carries original half
bits between arithmetic boundaries and allocates no storage or synchronization.
The fragment contract is explicitly RDNA wave32 with F32 accumulation.

`normalization.loom` owns F32 row moments, aligned-corners interpolation of
the learned 48×48 position table, and final affine normalization. Position
addition updates the residual and computes its moments in one pass. The
original 2×2 patch order is preserved without an expanded position matrix.
The dense launch provider supplies image dimensions and one 256-thread group
per row. Rectangular checks use an analytical spatial field with alternating
channel signs, independently known moments, and both image orientations.

The feed-forward implementation accepts original F16 or BF16 matrix storage
and accumulates in F32. Precision is a specialization input, so one schedule
serves both representations without checkpoint conversion or runtime type
branches. The source documents exact layouts, rounding boundaries, and launch
requirements. The [dense model leaf](../../../model/Qwen/Qwen3.8-27B/program/kernel/vision_ffn.loom)
provides the F16 wave32 launch. The [shared independent checks](test/ffn.loom)
exercise both original F16 and BF16 matrix interpretations.

Shared normalization supplies eight statistics bytes per patch. Up projection
normalizes while staging and publishes only the 4304-wide half-precision GELU
activation; down projection updates the F32 residual in place. Input and weight
tiles reuse external reads across multiple rows and output channels. Their
matched LDS permutations and read-ahead schedule belong to the algorithm,
not model policy.
