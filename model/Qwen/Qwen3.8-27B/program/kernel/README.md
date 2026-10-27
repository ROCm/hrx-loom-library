# Qwen numerical leaves

These implementations own concrete model launches, checkpoint layouts, and
target-specific schedules. Shared format providers decode GGML carriers;
shared tensor and sequence providers own reusable algorithms. Model leaves
supply geometry and configuration explicitly. Every public launch contract
has its declaration and storage assumptions beside the source.

## Text and prediction

Canonical and channel-eight Q5_K weights feed decode, cohort, and matrix
projections. Q4_K, Q6_K, and Q8_0 entries preserve their own packed layouts and
rounding boundaries. RDNA WMMA and CDNA MFMA leaves select concrete matrix
geometry while sharing decoding and publication contracts. Split-reduction
forms publish partials for a separate ordered merge.

Attention separates Q/K preparation, cache publication, history partitions,
and softmax merge. Span descriptors distinguish compact activation rows from
logical positions and physical cache pages. Gated DeltaNet retains causal
convolution and recurrence within each span, with capture and accepted-prefix
replay for speculative work. Neither padded rows nor rejected proposals may
advance committed history.

The output head normalizes selected rows, projects the vocabulary, and samples
with request-owned policy and random counters. MTP input pairing retains
accepted carry separately from proposal-private state. DFlash2 leaves add
candidate selection, grouped convolution, and transition correction while
borrowing the target embedding and vocabulary sources.

## Vision projector

Image leaves cover patch embedding, learned positions, LayerNorm statistics,
K/V preparation, tiled attention, feed-forward projection, and feature merging.
They retain the trained patch/merge order and explicit scratch ownership.
The command owner can reuse intermediates across a cohort while preserving
each image's output feature range.

`vision_patch_embedding.loom` selects original F16 weights and the launch for
shared direct RGB8 projection. The same motif supports BF16 operands and F32
checkpoint weights, converting only the resident panel. Its two temporal
contractions remain independent until the F32 epilogue adds them and the bias.

`vision_merger.loom` supplies the shared merger with original F16 matrices and
the dense model's 5120-wide output contract. It shares the FFN's resident panel
layouts and overlaps down-projection panel loads with matrix work. The command
preserves the postnormalized input before allowing the final feature range to
replace the earlier residual bank.

`vision_ffn.loom` specializes the [shared vision FFN](../../../../../motif/vision/qwen3_8/README.md)
with image geometry and original F16 matrix storage. The motif owns normalized
input staging, LDS permutations, read-ahead, and F32 matrix accumulation. The
model leaf owns the wave32 launch and existing up/down buffer ABI.
The motif's BF16 specialization uses the same schedule without converting
checkpoints.

`vision_normalization.loom` supplies the shared F32 position/statistics/apply
algorithms with geometry and launches. Learned-position addition computes the
first layer's moments while updating the residual; subsequent layers use the
ordinary statistics entry. These algorithms retain the same eight-byte row
record across the dense and MoE matrix representations.

### Pixel filtering

`vision_resize.loom` consumes the packet layout owned by
[`vision_pixels.loom`](../vision_pixels.loom). Its specialization arguments are
the two workgroup counts; launch arguments select a directory entry from zero
through three, the packed input, and a distinct RGB8 output buffer. The caller
launches `ceil(width / 32)` by `ceil(height / 16)` groups.

The gfx11 wave32 implementation resolves orientation while reading decoded
RGB/RGBA pixels. Straight alpha composites over white before filtering. An
8×32×3 F32 workgroup panel retains the horizontal results while threads
accumulate sixteen destination rows. Final values round to nearest-even and
clamp to the byte range. This schedule allocates no image-sized intermediate.

The model-local [resize scenarios](../../test/vision_resize.loom) compare every
output byte against analytical pixels across portrait and landscape geometry,
all display orientations, identity filtering, and strong scaling. Native and
access-instrumented executions use the same production leaf and fixtures.
