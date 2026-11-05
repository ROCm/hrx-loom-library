# Qwen3.8 sequence carriers

The dense and Flash-Next models share sequence records and continuation logic
while retaining separate numerical programs and scheduling. This package
interprets caller-owned token/media records, epoch descriptors, packet geometry,
and retained-state origins. It allocates no storage and chooses no target or
configuration.

Causal position, source position, packed-token ordinal, resident identity, and
selected-output ordinal are distinct coordinates. Packet capacity is independent
of resident count. Accessors preserve those distinctions across ordinary and
media inputs, page mappings, forks, and compacted continuations; field layouts
and bounds live beside their declarations.

Paged lookup uses 32-bit page IDs and signed native token indices. Its physical
extent is a whole number of 64-token pages, up to 2,147,483,584 tokens; this is
an address representation bound, not a model default or resident allocation.
The serving allocator supplies the actual extent from backing capacity, while
the trained per-session context remains a separate model constraint. Byte
addresses widen before scaling, so planes and individual accesses can cross
4 GiB without widening the page map or token coordinates.

Epoch metadata and retained origins stay immutable until queued consumers
retire. The producer publishes page mappings before use and gives writers
storage that cannot overwrite a live anchor. Continuation then advances the
caller-owned packet, preserving original feature offsets and first-result
identity even when source and destination records overlap.

The [checks](test/) compare complete records with independent expectations.
[Component authoring](../../../docs/component-authoring.md) explains the shared
dependency and verification workflow.
