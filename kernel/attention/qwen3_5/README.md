# Qwen3.5 attention kernels

`projection.loom` owns the single-token full-attention input projection. One
semantic launch consumes the shared planar Q8_1 activation row, Q4_K
query/gate and key weights, and either Q4_K or Q6_K value weights. It publishes
separate query, gate, key, and value arrays without an intermediate combined
projection buffer.

The public configuration is `(input_size, query_heads, key_value_heads)`.
Every head has 256 channels, and the kernel derives its private row grid from
those semantic values in the launch-config region. Query and gate rows are
adjacent within each query head in the packed weight, while key and value rows
follow their key/value head order. Physical workgroup counts are not part of
the caller ABI.

Each workitem retains one complete 32-value activation group. Q4_K contributes
one affine integer dot and minimum correction. Q6_K contributes two complete
16-value integer dots, retaining the raw signed half-group scales and shared
block factor until the portable Vulkan-reference F32 scaling boundary. Two
32-thread cohorts reduce a row with a fixed XOR tree and publish through eight
bytes of workgroup storage.

The adjacent test independently decodes the packed records, distinguishes the
Q6_K floating-point boundary from integer-scaled alternatives, checks direct
query/gate routing, and verifies every input buffer remains immutable. See
[component authoring](../../../docs/component-authoring.md) for the shared
workflow.
