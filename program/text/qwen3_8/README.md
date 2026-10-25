# Qwen3.8 text-service policy

The dense Qwen3.8 and Flash-Next compositions share chat framing, sampling
options, and queued text execution. The providers expose Loom's text-service
v1 exports; each composing model supplies `text_model_name_v1`, its tokenizer,
preparation, and numerical stages. The program chooses no physical device and
contains no checkpoint paths.

`chat.loom` consumes the native service's validated request records and its
JSON, tokenizer, input-validation, and tool host imports. It owns the ChatML
framing, optional reasoning block, XML tool syntax, image markers, and sampling
packet. Generated text becomes reusable retained history only when canonical
history preserves the exact emitted bytes. Partial reasoning and tool markers
remain withheld during streaming and are resolved at completion. Returned
buffers have VM ownership; request-local scalar state remains opaque to the
service caller.

`epoch.loom` consumes consecutive first and continuation cohorts. Each record
carries length, causal and source positions, resident row, known-input range,
flags, output credit, and continuation owner.
Encoding builds the source's packed descriptors and selected-row mapping.
It returns fourteen explicit upload ranges; unused capacity remains untouched
and is not uploaded. The retained control allocation contains an 8-word stage
header, one 24-byte sampling record per resident identity, and fourteen 40-byte
transfer records. The composition's preparation provider owns this allocation.
Encoding gathers the current cohort's sampling records; output identities name
their first-cohort policy ordinal, not their resident state ID. Reordered
continuations preserve that policy ordinal through their first-result tags.

Runner imports enqueue target, proposal, verification, catch-up, and feedback
work. The returned completion edge covers work using mutable state; feedback
has its own transfer edge, which the owner joins before publishing progress.
Publication consumes completed immutable feedback and reports consumed inputs,
known inputs, output IDs, verification counts, and the next source position.
Continuation flags are indexed by first-cohort ordinal, independently of the
resident identity. Every plan record names its first-cohort owner explicitly,
including records in the first cohort itself.
State ownership and feedback publication use the supplied buffer extents and
live cohort size. Device packet encoding still uses the composition's fixed
sampling and descriptor layout described above.
No transform waits for a device or recovers ownership from buffer contents.

`media.loom` fills eight-word mixed-input records and tracks compressed image
positions. Mixed packets have at most 512 records per plane and 1024 across
two planes; this bound is independent of the larger ordinary Flash packet.
`vision_geometry.loom` evaluates the processor's smart-resize policy on the
integer grid of 32-pixel merged patches, including ties-to-even alignment and
minimum/maximum image area. Image availability remains a model-component
choice; linking chat policy does not supply a vision encoder.

The [tests](test/) use the real native chat caller, small tokenizers, and VM
checks for source-defined packet and streaming contracts. Learned service
checks in each model qualify the numerical stages and complete device lifetime.
