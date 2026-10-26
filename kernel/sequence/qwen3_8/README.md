# Qwen3.8 speculative sequence control

These launch contracts connect target verification, a separate multi-token
predictor, and queued continuation. The dense and Flash-Next models share the
control layout while owning their numerical programs and retained-state
implementation. Each dispatch uses one device thread. This preserves the
established ordering of a small control packet; it is not a throughput claim.

Every launch receives `mixed_inputs` explicitly: zero selects contiguous I32
token IDs and one selects eight-word media records. The model command supplies
its configured layout. The kernels own no global defaults and declare no
concrete target. Their BUILD targets qualify gfx11 and gfx9-4 artifacts, with
execution through the native AMD profile.

`accept.loom` consumes target predictions and supplied or generated candidates.
The first mismatch, EOS token, or exhausted output credit ends the accepted
prefix. The final prediction remains a pending input. Accepted control retains
the original packed offsets; replay control compacts speculative rows while
retaining their captured-transition offsets. Known inputs are never committed
twice. Rows outside the live output prefix retain their previous bytes.

`mtp_proposal.loom` extracts only generated-candidate rows into a separate draft
batch, preserving target order and resident-row identity. Each proposal round
writes both the next draft ID and its reserved verifier slot. Draft causal and
source positions advance together; target frontiers remain unchanged until
acceptance. Draft storage is cleared through its actual allocation capacity,
including a partial vector tail.

`continuation.loom` launches the shared continuation motif. The allocation's
ordinary plane capacity remains independent of an epoch's selected shape.
Mixed-media planes carry their own capacities, and compaction snapshots source
frontiers before overwriting records. Actual acceptance, remaining credit,
EOS, and context capacity determine which speculative rows survive. Result
identity tags retain the original first-epoch slot after compaction.

The adjacent test files retain independent minimum-frontier and explicit
publication oracles. They cover reordered resident rows, sparse and full draft
batches, every proposal round, large ordinary planes, non-vector tails,
retirement boundaries, and overlapping mixed-media records. Whole-buffer
comparisons detect writes outside the owned output. The shared text-program
tests additionally exercise continuation after real image-packet encoding.

Acceptance uses sixteen frontier trials: widths one through eight in each
input layout. Multi-span fixtures mix speculative, known, and hidden spans,
with separate EOS and credit-limited rows; one trial covers a single span.
EOS reaches its selected middle or final position without an earlier injected
mismatch; output credit includes zero.
Two empty epochs and five wide cohorts independently cover empty publication
and span counts beyond ordinary packing boundaries. These explicit combinations
retain complete output and immutable-input checks without multiplying every
width, capacity, span count, and stopping position together.

Proposal preparation has twelve explicit combinations covering empty, sparse,
and full drafts, minimum storage, each four-wide clear tail, high mask bits,
and mask wrap at 32 and 64 spans. Publication covers every round in each input
layout, followed by three- and seven-round chains. The thirty total samples
check complete descriptor, origin, input, and target buffers against the
independent fixture; a full 65-row draft exercises the last allocated slot.
