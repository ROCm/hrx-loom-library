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

Narrow RDNA Q6 projections use the shared GGML F32 register panel. A wave64
worker retains decoded weights and sixteen activation rows across a 512-value
K panel, with 40 KiB of reusable workgroup storage. The queue's execution-unit
count bounds the worker grid; each worker traverses any remaining logical
tiles. Query/key projections use this path through 32 live rows. Residual
projections use it through 16 rows and retain the matrix schedule above that
crossover. Other targets retain their matrix providers. The enclosing command
samples the live count once and forwards it through the layers, allowing all
indirect launch counts to share one device producer.

Four/eight-row DFlash projections bind the shared Q4_K and Q6_K F32 panel
templates to 512-workitem wave64 kernels. Every raw weight packet serves the
complete row bank, and K1024 activation panels serve 64 Q4 or 48 Q6 channels.
The same Q6 bank handles eight-row target down projections. Four-row target
down projections retain their separate channel distribution. The launch
adapter caps queued work at eight groups per execution unit; channel traversal,
tail masking, and arithmetic remain in the shared motifs. Native command
selection preserves the other targets' matrix and cohort providers.

Wide RDNA gate/up prefill calls the shared paired Q5_K/F16 kernel directly.
Its 256-token/128-channel tile retains K64 activations and both projections in
64 KiB of workgroup memory, then publishes SwiGLU without intermediate global
gate/up matrices. The model supplies prepared channel-eight weights and its
live-prefix contract. Numerical and tile-boundary checks belong to the shared
kernel; the model checks in-place preparation followed by consumption.

Attention separates preparation, cache publication, history partitions, and
merge. Four subgroups share each 16-query by 64-cache-row panel. Their actual
width determines the staging stride, fragment carriers, score packets, and
output channels per lane; the entries inherit the execution target. The
softmax tile retains each lane's score packets across its maximum and sum
reductions and publishes probabilities in the shared transposed layout.
Before traversal, each prefill subgroup stages a query row in contiguous
packets sized from its width.
Prefill and split decode retain their native F32 accumulators across history
tiles, using the existing F32 product panel to broadcast row scales and then
publish the final result. The matrix motif returns an updated accumulator;
the caller owns its lifetime, storage, and synchronization.
Contiguous, paged, and split attention use that same ownership contract.
Q/K/V and probabilities remain F16, while native products and online
numerators stay F32 through normalization or partial-state publication. The
paired-query split path retains its native accumulator fragments across
history tiles; its merge consumes the same F32 partial-state layout.

Gated DeltaNet retains each 128-key state column across its complete token
span. Four adjacent lanes share a column and call the shared tensor delta-rule
motif; the model owns normalization, decay, reader selection, and publication.
Four workgroups per execution unit traverse the logical span/column tiles,
keeping the grid bounded as concurrent requests increase. Span and tile
coordinates advance separately, so large span capacities cannot overflow a
flattened work count. Capture and accepted-prefix replay keep rejected
proposals from advancing committed history.
The output head and predictors preserve request-owned policy and accepted carry.

The output-head contraction lives in the shared Q6_K/Q8_1 x4 panel kernel
family. Its model-local leaf adds device-controlled live rows and vocabulary
extent. The rare/control table uses the same packed-block arithmetic while
sharing each selected weight row across an owner's proposal positions.
Greedy and sampled partial selection receive explicit logical
row/tile coordinates, allowing their groups to stride under the composition's
budget while preserving tie-breaking, excluded tokens, and output ownership.

Vision leaves specialize the [shared encoder](../../../../../motif/vision/qwen3_8/)
with original F16 weights and the dense model's feature width. The command owner
can reuse K/V and temporary banks after their consumers retire while preserving
each image's output range. Shared preprocessing and numerical checks live with
those providers; model tests retain composition and learned service behavior.

The [program overview](../README.md) explains model-wide ownership.
[Component authoring](../../../../../docs/component-authoring.md) covers
production dependencies and test separation.
