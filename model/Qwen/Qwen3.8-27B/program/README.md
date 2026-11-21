# Qwen target, predictor, and image programs

The model-specific providers bind one dense checkpoint to the shared Qwen
text service. A parameter source is resolved once through its named scope;
all specialized stages borrow that same source. The target has 64 layers,
arranged as sixteen groups of three Gated DeltaNet layers and one full-
attention layer. Activations use the model's 5120-channel residual ABI.

## Packed execution and retained state

The preparation program describes stage roots, capacities, allocation records,
and retained reader/writer layouts. Epochs carry compact token rows plus
immutable span and resident-row records. Attention uses pooled KV pages;
Gated DeltaNet carries convolution and recurrent state. Dense projections can
share a token matrix while every span retains its own causal position and
state ownership. Private writer endpoints preserve forked history.

When the submitted rows fit in one 32-row projection panel, the target can
read each weight tile once for those rows. Wider prompts use the matrix
schedule. The single-owner decode path keeps its dot-product kernel. These
choices use the same packet layout for every live token and owner count.
Only selected output rows need a vocabulary projection. A solitary row uses
the one-row contraction and empty windows submit zero workgroups.

MTP proposes through the checkpoint's NextN block. Target verification chooses
the accepted prefix before recurrent replay; accepted target hidden rows feed
predictor catch-up. DFlash2 uses a separate parameter scope, five target-layer
feature taps, and proposal-private K/V. Only accepted features reach its
retained context. Both strategies remain parts of the text service.

DFlash2 supports a `draft_depth` ceiling from one through seven, with seven as
the default. A single live request uses that ceiling; two through four use at
most three proposals, and five through eight use one. The packet token budget
divided by the verification width bounds how many owners fit, and the span
budget bounds how many distinct owners can be submitted. Preparation uses
both bounds to declare the reachable shorter proposal/verifier pairs.
The shared text encoder selects a
pair and compacts both queued banks while preserving the native reservation and
request identities. The selected width includes one anchor and specializes
convolution, bidirectional attention, candidate selection, and private storage
together. Shorter blocks reduce proposal and verification work without changing
the target's acceptance rule.

RDNA proposals containing four or eight neural positions use shared raw
Q4_K/Q6_K register panels. Each decoded weight serves the complete row bank
before retirement, reducing inactive matrix lanes at those sizes. Model
composition selects the projection schedule for the active proposal width.
Epochs can select a different width as peers arrive or finish. Already queued
work retains its selected stage handles.

## Link and parameter boundaries

`config` owns model defaults and concrete matrix target bindings. The
[numerical providers](kernel/README.md) group attention, recurrent state,
projections, predictors, and vision by their storage and algorithm contracts.
Each names its shared carriers, algorithms, and required configuration.
`commands` composes those providers into complete layers, epochs, and
predictor execution. Host preparation and service callbacks are separate
providers. Lower layers do not call back into a model command.

The `weights.loom` import policy prepares target and MTP FFN gate/up tensors
into channel-eight storage. The transform reads each disjoint group before
replacing its bytes, preserving the packed footprint. `dflash_weights.loom`
selects draft preparation. These policies are auxiliary compiler inputs in
both representations; checkpoints stay outside the build payload graph.

Small Q5 projections pass physical row geometry from the prepared stage into
the shared GGML family: at most sixteen initialized rows select its 16-row
entry, and larger small stages select 32. The existing launch-count observation
controls enablement separately. Alpha and residual consumers keep their padded
storage contracts; this changes neither allocations nor the count producer.

Wide prefill composes the shared paired-Q5 gate/up kernel and Q6 residual
projection. Their matrix tiles reuse decoded weights across 256 token rows;
the Q5 pair publishes the fused SiLU product directly as F16, which the Q6
down projection consumes before adding its F32 result to the residual. The
Gated DeltaNet output projection also consumes producer-rounded F16 values.
These shared kernels own format decoding, panel tails, and bounded workers;
the model owns their shapes and the choice of narrower schedules for small
cohorts. Both raw Q6 and prepared channel-eight Q5 retain their checkpoint
representations, with no additional weight copy at execution time.

## Output-head execution scope

Activation packing, the Q6_K vocabulary contraction, and partial winner
selection use bounded workgroup populations. Groups visit multiple logical
tiles as the workload grows. Projection cohorts retain weights across rows
through the shared panel template; subgroup width comes from the execution
target. Greedy selection still ignores padded vocabulary entries, and sampled
selection applies each request's token permissions before choosing winners.

The head launches up to four workgroups per execution unit in the selected
queue's deployment scope, clamped to useful tiles. The model requires
`target.execution.unit.count`; its launch counts cannot compile without that
fact. The serving runtime supplies the selected queue's count during target
specialization. Source and relocatable-bytecode packages both use that path.
Native AOT compilation requires a deployment profile carrying the count; an
architecture name alone does not specify device capacity. Four groups per unit
is the head's launch policy, not a claim about exact kernel residency.

The proposal vocabulary projection shares each weight pass across up to 32 rows,
including an exact row specialization for the final short cohort. It preserves
complete inactive cohorts and computes every initialized row of a partially
live cohort; selection consumes only the live prefix. The shared motif owns
the accumulator bank and matching launch geometry, while the model owns the
control record and row partition. Ordinary target projection uses the same
numerical body through its fixed-row entries.

DFlash candidates normally score a 98,304-token prefix, all 33 control tokens,
and up to 991 rare IDs learned from accepted target inputs. Each request retains
its membership bitmap, first-use table, and recent miss positions in a
22,784-byte carry record. The shared sequence motif owns that representation;
the service's existing checkpoint, fork, growth, and retirement paths preserve
its lifetime. Fresh publication initializes the header and bitmap without
clearing unused table entries.

An unknown anchor or two misses in the preceding 32 committed positions selects
the full vocabulary. One full-head request selects a shared full pass for the
cohort. Otherwise, a small table projection reuses each raw Q6 row across that
request's draft positions. Selection compares canonical token IDs, including
ties, and excludes physical head padding. This reduces proposal work; target
weights, target sampling, and speculative verification remain unchanged.

## Image input

The image-input providers connect the text image service to Qwen's trained
pixel preprocessing and feature layout. They consume the codec-validated native
image packet and the output dimensions chosen by the shared
[Qwen chat policy](../../../../program/text/qwen3_8/README.md). The checkpoint's
image preprocessing specification is published in its
[processor configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/preprocessor_config.json).

The shared [pixel provider](../../../../program/text/qwen3_8/vision_pixels.loom)
prepares antialiased bicubic coefficient tables, preserving
the decoded RGB/RGBA planes and display orientations. A cohort has one native
range directory followed by prepared image or frame-pair packets. Their offsets
address each plane's axis plans and aligned decoded pixels. F32 weights use
the specified F64 sampling-center evaluation before explicit rounding. The
[device leaf](kernel/README.md) performs the pixel filtering.

`vision_control.loom` binds the model's feature width and projector capacity to
the shared image service and provides `text_image_prepare_request_v1`,
`text_image_prepare_batch_v1`, and `text_image_run_v1`. The request callback tags
compatible groups by selected dimensions and temporal plane count. Batch
preparation consumes the native packer's canonical directory and describes
disjoint output feature ranges. Linear and feed-forward stages share weights
across stacked rows; attention remains local to each image or frame pair. Each
32×32 output pixel block becomes one 5120-element F32 row.
The shared stage provider declares the geometry-specific projector through the
prepare host; the runner owns device execution and storage. There is no second
cohort-sized pixel copy.

Cold preparation registers a descriptor-only memory forecast for bounded
serving. It accounts for the projector bank, the upload ceiling, and the largest
cohort's shared private workspace. Command composition and forecasting use the
same shared vision workspace query. The 5120-column final output temporarily
holds a residual bank, then receives features after post-normalization has
preserved the merger input. Retained output estimates include this request's
compact feature bytes plus one maximal shared output root. Native admission
permits at most one mixed-owner cohort per request; subsequent cohorts are
owner-private. Expanded feature storage counts only this request's rows.
The forecast needs neither decoded pixels nor activated weights; it lets the
text consumer check projection and continuation against the configured memory
budget before either phase runs.

The [tests](../test/README.md) check coefficient values, cohort feature plans,
all eight display orientations, and RGB/RGBA device filtering independently of
learned weights. The complete text and image-projector composition is described
by the [model card](../model.json).
