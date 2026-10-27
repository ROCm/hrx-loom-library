# Resident token panels

This provider runs all 28 Qwen3-Embedding-0.6B layers in one invocation on the
eight-column, four-row Strix Halo NPU. It processes four 128-token panels as
two independent 256-token attention windows. Sequence IDs mask attention, and
each result comes from its sequence's final live token. Padding has sequence
ID -1 and contributes no result.

The dense path streams each column's projection records once for the complete
cohort. Six shared worker bodies cover 27 cores; five ingress, peer
normalization, and terminal roles have distinct ownership. Placement supplies
bank addresses, immediate lock identities, and DMA-control fields. The same
`run_layer` routine executes every layer. Finite DMA queues are refilled at
four-layer epoch boundaries while the invocation retains its model bindings
and working storage.

## Kernel boundary

`qwen3_embedding.panels.cohort` borrows thirteen buffers in this order:

| Binding | Bytes | Ownership and representation |
| --- | ---: | --- |
| Projection columns 0–7 | 62,146,560 each | Read-only BFP16EBS8 records, packed during model loading. |
| Constants | 311,296 | Learned BF16 scales and model lookup/rotary tables. |
| Embedding rows | 310,618,112 | Original publisher BF16 embedding tensor. |
| Metadata | 32,768 | 512 rows of sixteen I32 fields, including token, sequence, position, and requested dimension count. |
| Output | 2,097,152 | 512 rows of 1,024 F32 values; the service selects each sequence's final live row. |
| State | 11,698,176 | Invocation-owned projection and attention intermediates. |

Preparation installs full 64-bit binding addresses. The native configuration
starts the streams, runs the layer and epoch schedule, waits for final output,
and retires all cores before returning. Worker locks and completion waits own
local-storage reuse; a single invocation does not imply that every intermediate
stays in local memory. Dense weights account for one 497,172,480-byte pass.

Dense and attention products use compensated BFP operands and F32 accumulation.
Normalization, rotary transforms, softmax, and the SiLU lookup retain their BF16
rounding boundaries. Prefix normalization returns between one and 1,024
dimensions. No host-side model arithmetic or checkpoint conversion is part of
inference.

## Source ownership

`array.loom` owns concrete targets, placement, and the kernel entry. All 32
physical entries pass the cohort's panel count and attention-window size to
eleven mathematical worker roles. The roles derive their dense traversal,
attention-wave bounds, and window-local key ranges from those facts. Fixed
matrix fragments and four-layer DMA epochs retain their own dimensions;
changing a cohort count cannot change an 8x8 matrix fragment.

The current `cohort_shape` supplies four panels and 256-token windows to match
the finite DMA plan and binding extents. A different geometry requires matching
worker traversal, queue record counts, descriptors, and storage extents in the
array configuration. Worker arguments alone do not select another executable
geometry.

Worker files own complete mathematical roles and their lock contracts. `matrix.loom`,
`attention.loom`, `packing.loom`, and `activation.loom` own tile arithmetic.
Resource and transport files carry immediate instruction forms and bound
external transfers. Configuration files own stream routes, descriptors,
request preparation, layer execution, and epoch transitions.

All source files are direct inputs to the single `resident` library. Their
helpers and data tables are private; explicit declarations permit each file to parse
independently. The initial epoch-bound and output-lane data symbols are shared
across worker source modules. The array supplies the concrete target through
the ordinary `target.decl` contract.

The source representation supplies these files as primary linker inputs so
they retain one private ownership scope. The archive representation supplies
the merged `resident.loombc` in that same scope. Dependencies remain ordinary
providers; neither representation needs an extraction script or precompiled
device image.
