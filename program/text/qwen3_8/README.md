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

The model has one leading instruction envelope. Chat requests may spell that
role `system` or `developer`; both render the same Qwen system tokens. A second
instruction message or one after user/assistant history is rejected, since the
checkpoint template has no corresponding role or priority boundary.

`epoch.loom` consumes consecutive first and continuation cohorts. Each record
carries length, causal and source positions, resident row, known-input range,
flags, output credit, and continuation owner.
Encoding builds the source's packed descriptors and selected-row mapping.
It returns fourteen explicit upload ranges; unused capacity remains untouched
and is not uploaded. The retained control allocation contains a 10-word stage
header, one 24-byte sampling record per resident identity, a 64-byte layout
recipe, and fourteen 40-byte transfer records. The composition's preparation
provider owns this allocation.
Encoding gathers the current cohort's sampling records; output identities name
their first-cohort policy ordinal, not their resident state ID. Reordered
continuations preserve that policy ordinal through their first-result tags.

`storage.loom` owns the shared control recipe, descriptor writers, and capacity
policy. Packet capacity covers isolated prefill and the largest prepared token
bucket, rounded upward to an even value. Resident growth preserves that capacity,
stage handles, and existing sampling policies; new residents start with empty
policies. The packet-capacity checks use literal small catalogs, and each
model's storage checks exercise its real allocation evaluator through growth.
Both paths have VM benchmarks without model parameters or device execution.

The allocation writer publishes eleven records of byte size, 256-byte alignment,
and initial zero extent. The row-view writer publishes offset/length pairs in
control, attention, input, and progress order. Both update caller-owned tables;
model preparation computes the extents and the native materializer owns the
actual allocations. Exact record checks include guards, untouched slots,
replacement writes, and quantities above 32-bit byte addressing.

Runner imports enqueue target, proposal, verification, catch-up, and feedback
work. Proposal variants cover power-of-two cohort capacities, and the live span
count selects the smallest sufficient variant. Ordinary and mixed-input
catalogs retain explicit proposal and catch-up origins. Every variant uses the
same parameter domain and fixed packet layout; growing resident storage
preserves the stage handles. The returned completion edge covers mutable work;
feedback has its own transfer edge, which the owner joins before publishing progress.
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
