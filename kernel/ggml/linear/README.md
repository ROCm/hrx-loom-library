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

Small independent scalar-byte references check packed arithmetic and complete
storage. Larger stress workloads use analytical results where a full reference
contraction would dominate the test. The [test package](test/) owns shared
oracles; [component authoring](../../../docs/component-authoring.md) explains
production/test separation and the standard workflow.
