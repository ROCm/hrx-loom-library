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

Epoch metadata and retained origins stay immutable until queued consumers
retire. The producer publishes page mappings before use and gives writers
storage that cannot overwrite a live anchor. Continuation then advances the
caller-owned packet, preserving original feature offsets and first-result
identity even when source and destination records overlap.

The [checks](test/) compare complete records with independent expectations.
[Component authoring](../../../docs/component-authoring.md) explains the shared
dependency and verification workflow.
