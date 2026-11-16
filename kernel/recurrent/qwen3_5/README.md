# Qwen3.5 recurrent kernels

`gdn_decode.loom` owns the single-token Gated DeltaNet convolution transition.
One 32-thread workgroup owns each query head, key head, or 32-channel value-head
quarter. The same launch publishes shifted convolution history and the query
and key inverse norms, replacing three separately launched operations.

The public Qwen3.5 family fixes 16 query heads, 16 key heads, 128 channels per
head, and four F32 convolution taps. Its value-head count is 16, 32, 48, or 64;
the kernel configuration specializes that count and derives the complete
channel width, workgroup grid, recurrent-state extent, history origin, and
per-layer stride. Adding a supported model size is therefore a configuration
value rather than another kernel.

The launch consumes the family's trusted one-token work record and forkable
row-origin table. State consists of the value-head recurrence matrices followed
by three time-major F32 history rows for each layer. Fresh rows read zeros;
continued rows read their selected source slot; aliased and forked publication
write the same shifted history transition. Multi-token spans retain their
separate span-aware path in the calling program.

The adjacent test uses an independent sequential oracle and covers every
value-head geometry, fresh and continued rows, aliased and forked publication,
inactive work, immutable operands, and complete state storage. See
[component authoring](../../../docs/component-authoring.md) for the shared
workflow.
