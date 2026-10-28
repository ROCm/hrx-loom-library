# Qwen3.8 sequence layout

The dense Qwen3.8 and Flash-Next programs share these sequence carriers while
retaining their own numerical kernels, preparation, and scheduling. Functions
consume validated, caller-owned storage and preserve the original public
`qwen38_*` linking vocabulary. Accessors and continuation allocate nothing;
continuation reuses scratch inside the caller's packet storage. None chooses
a target or configuration.

`input.loom` reads and updates contiguous token IDs or the ID field of an
eight-word media record. The composition supplies the layout choice. Media
records retain temporal, height, and width coordinates in words 2–4; the
32 rotary pairs interleave those axes with counts 11, 11, and 10, and both
halves use the same coordinate. The caller selects which lanes undergo
rotation. Position lookup, token lookup, and mutation use the caller's positive
record capacity; token mutation preserves every other record field.
Feature lookup returns the byte offset in word 1, or zero for vocabulary
lookup. Offsets remain relative to the original input allocation when accepted
rows are selected through a compact origin map.

`span.loom` interprets the three-word epoch header and six-word descriptors:
length, causal position, packed-token begin, resident identity, source position,
and selected-output index. Source positions and output indices have independent
I32 fields; `-1` denotes no selection. Causal positions and source coordinates
remain distinct. Packed token positions and resident identities are independent
nonnegative I32 values. Their actual extents belong to the caller rather than
to a fixed binding-helper capacity.

`packet.loom` defines storage geometry from a caller-supplied token capacity.
The packet holds two epoch banks, cohort-local sampling policies and identities,
continuation indices, and compaction scratch. Device feedback uses the same
capacity. Resident count does not determine packet geometry: a compact cohort
can reference any resident in the model's state directory.
Both model preparation programs use the packet extent function for their
allocation records, so allocation and field addressing share one geometry.
Byte extents remain 64-bit even when token capacity fits a 32-bit index.

Each retained row contains five I64 words: immutable anchor origin, private
writer origin, K/V origin, fork position, and page-map origin. The recurrent
binding reports whether a span begins at the fork; its caller chooses the corresponding
state and owns the aliasing contract. K/V layer origins are 256-byte aligned.
Preparation publishes each optional page table's byte origin in its resident
record. Tables map 64-token logical pages to physical pages; neither resident
count nor epoch width participates in lookup. Physical capacity bounds the
mapped pages. A zero physical capacity selects dense addressing without reading
a map.

Span consumers see immutable epoch metadata and retained origins until their
queued work retires. Continuation then publishes the next live epoch.
Page mappings must be published before their consumers; writers cannot
overlap another live row's anchor or writer. These are producer contracts,
not repeated runtime validation inside each accessor.

`continuation.loom` constructs the next live epoch from a queued plan and the
first epoch's accepted results. It preserves causal and source positions as
separate facts, compacts ordinary or media input, rewrites selected outputs,
and retains first-result identities. Mixed-media source frontiers are saved
before overlapping input compaction. The caller owns disjoint epoch, token,
and result allocations and their completion lifetime. Both the device launch
and host-side service integration checks use this same public algorithm.
