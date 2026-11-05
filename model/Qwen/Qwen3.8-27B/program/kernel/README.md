# Qwen numerical leaves

These leaves bind checkpoint layouts, model geometry, and launch ABIs to shared
format, tensor, sequence, and vision algorithms. They own explicit storage and
target contracts; model commands own complete layer ordering and intermediate
lifetimes. A projection's packed representation and rounding boundaries remain
part of its contract when selecting a native matrix implementation.

The 32-token Q5 projection shares a 32x64 output tile across GPU targets.
Subgroup width selects the distribution of its 16x16 result fragments: eight
32-lane subgroups retain one fragment each, while four 64-lane subgroups retain
two. Both arrangements use 256 workitems and the same encoded-weight,
activation, and metadata panels, preserving staging reuse and read-ahead.
Canonical and channel-eight weight orders share this body and their command
ABI; the matrix motif supplies the target's operand carrier.

The narrow alpha projection retains seven aligned K partitions and its ordered
reducer. Four subgroups share each decoded panel; their width determines the
staging stride, matrix carrier, and publication packets. Each partition owns
one 32x48 F32 plane, so changing subgroup width leaves the command's partial
storage and subsequent reduction unchanged.

The wider Q6_K and Q8_0 projections share one signed-weight contraction.
Four, eight, or sixteen subgroups retain the existing C64/T64, C64/T128, or
C128/T128 panels. Subgroup width determines accumulator carriers, staging
strides, and publication packets while the decoded panels and F32 recurrence
remain unchanged. Whole and partitioned Q6 contractions use the same range
body; command selection retains the tuned split preference. The matrix motif
owns native operand layouts, so these entries inherit the execution target.

Attention separates preparation, cache publication, history partitions, and
merge. Gated DeltaNet retains causal state within each span. Capture and
accepted-prefix replay keep rejected proposals from advancing committed history.
The output head and predictors preserve request-owned policy and accepted carry.

Vision leaves specialize the [shared encoder](../../../../../motif/vision/qwen3_8/)
with original F16 weights and the dense model's feature width. The command owner
can reuse K/V and temporary banks after their consumers retire while preserving
each image's output range. Shared preprocessing and numerical checks live with
those providers; model tests retain composition and learned service behavior.

The [program overview](../README.md) explains model-wide ownership.
[Component authoring](../../../../../docs/component-authoring.md) covers
production dependencies and test separation.
