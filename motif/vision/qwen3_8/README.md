# Qwen3.8 vision algorithms

These motifs own the shared vision encoder's arithmetic and storage contracts.
Dense and Flash-Next compositions supply checkpoint precision, geometry,
launches, and final feature width. Shared algorithms consume original F16 or
BF16 matrices, converting only resident panels rather than materializing a
converted checkpoint.

Linear and feed-forward work shares panels across rows and output channels;
attention retains each image or temporal pair's own rotary coordinates and
softmax. Caller-supplied head cohorts and byte origins support sequential
workspace reuse or concurrent disjoint banks without changing the arithmetic.
The source declarations own fragment layouts, rounding, and synchronization.

Command composition and service admission use the same workspace query.
Private intermediates follow dispatch lifetimes, while feature output belongs
to its eventual consumer. A sufficiently wide final feature allocation can
hold an earlier residual bank until the merger publishes the result. Model
preparation owns that lifetime; the workspace query owns its size and alignment.

The [shared text program](../../../program/text/qwen3_8/) owns media policy,
geometry, CPU coefficient planning, and the service packet. These motifs own
device filtering and learned encoder computation. Their [independent checks](test/)
need neither checkpoints nor a server and cover both model precisions once.
[Component authoring](../../../docs/component-authoring.md) explains the common
build and test workflow.
