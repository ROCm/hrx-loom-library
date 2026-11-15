# Quantized linear projections

These kernels expose launch ABIs for projections over GGML packed weights and
activation carriers. Reusable decoding belongs to the [format motifs](../../../motif/format/ggml/);
contraction and panel reuse belong to the [linear motifs](../../../motif/linear/ggml/).
The kernel leaf supplies launch geometry and publication through an explicit
buffer contract. Quantization is a separate operation composed by the caller.

Workload dimensions are arguments or specialization facts rather than shared
package-global capacity settings. Raw and prepared storage, live row count and
allocated capacity, and scalar or matrix arithmetic remain distinct contracts.
Each source documents its layouts, rounding, aliasing, and target requirements.
A concrete fragment implementation cannot be made portable merely by changing
its target flag.

The paired Q5_K/F16 entry consumes two weight banks and publishes SwiGLU
directly. Its configuration supplies capacity, K/N, channel interleaving, and
a positive worker budget. Execution receives the live row count and operands;
zero live rows preserve output even when the capacity grid is nonempty.
Workers traverse 256-row/128-channel tiles, retaining the same 64 KiB shared
allocation across tiles. The shared motif owns operand staging and arithmetic;
the kernel owns the wave32 ABI and launch policy. Raw and channel-eight weights
need no persistent expanded representation.

`q6_k_f16_matrix.loom` exposes overwrite and in-place residual projections
for F16 input rows, raw Q6_K weights, and F32 output. Its 256-row/256-channel
matrix schedule uses the same capacity/budget ABI as paired Q5, with 64 KiB
of worker-local staging and a bounded grid. The independent scalar oracle
rounds reconstructed weights to F16 before its F64 dot; scenarios cover both
publication modes, live/capacity differences, tails, and worker reuse.

`q6_k_f32_panel.loom` exposes overwrite and residual projections for up to
32 live F32 rows. Configuration receives the sampled live count, K/N, and a
positive worker budget; execution receives the device count buffer, K/N, and
the three operand buffers. The count remains equal to the sample until
completion. Zero rows launch no work. Workers traverse 16-row/48-channel tiles
in a flattened grid capped by the budget, so increasing output dimensions
does not require an unbounded dispatch. The caller selects its budget from
device capacity and useful resident work; the API's 10,000-group limit is an
admission bound, not a tuning recommendation.

The panel's ordinary scenario covers unpadded rows and channels, a maximum-K
reduction, and tile traversal with one or two workers. Its separately selected
`q6_k_f32_panel_stress` target provides a configurable full-projection benchmark
and an exact analytical check, without adding large allocations or reference
contractions to the ordinary test run.

The Q6_K/Q8_1 x4 panel entries bound their grid with a positive workgroup budget
supplied by the caller. Groups stride through output tiles instead of growing
the grid with the vocabulary. M1/M2/M3/M4 retain four channel panels, M8 retains
two, and M32 retains one; these entries fix the accumulator bank while K, N,
and the budget remain values. The same contraction uses eight logical 32-lane
clusters in a 256-workitem group on either 32- or 64-lane subgroups. Composition
owns the deployment budget and row partition; the kernel clamps the budget to
useful tiles and masks channel tails before accessing storage.

Small independent scalar-byte references check packed arithmetic and complete
storage. Larger stress workloads use analytical results where a full reference
contraction would dominate the test. The [test package](test/) owns shared
oracles; [component authoring](../../../docs/component-authoring.md) explains
production/test separation and the standard workflow.
