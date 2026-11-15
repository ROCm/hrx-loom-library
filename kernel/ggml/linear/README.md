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

Small independent scalar-byte references check packed arithmetic and complete
storage. Larger stress workloads use analytical results where a full reference
contraction would dominate the test. The [test package](test/) owns shared
oracles; [component authoring](../../../docs/component-authoring.md) explains
production/test separation and the standard workflow.
