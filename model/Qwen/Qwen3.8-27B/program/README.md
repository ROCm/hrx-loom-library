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

MTP proposes through the checkpoint's NextN block. Target verification chooses
the accepted prefix before recurrent replay; accepted target hidden rows feed
predictor catch-up. DFlash2 uses a separate parameter scope, five target-layer
feature taps, and proposal-private K/V. Only accepted features reach its
retained context. Both strategies remain parts of the text service.

## Link and parameter boundaries

`config` owns model defaults and concrete matrix target bindings. `kernel:numerics`
composes shared carriers and algorithms; `commands` composes complete layers,
epochs, and predictor execution. Host preparation and service callbacks are
separate providers. Lower layers do not call back into a model command.

The `weights.loom` import policy prepares target and MTP FFN gate/up tensors
into channel-eight storage. The transform reads each disjoint group before
replacing its bytes, preserving the packed footprint. `dflash_weights.loom`
selects draft preparation. These policies are auxiliary compiler inputs in
both representations; checkpoints stay outside the build payload graph.

## Image input

The image-input providers connect the text image service to Qwen's trained
pixel preprocessing and feature layout. They consume the codec-validated native
image packet and the output dimensions chosen by the shared
[Qwen chat policy](../../../../program/text/qwen3_8/README.md). The checkpoint's
image preprocessing specification is published in its
[processor configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/preprocessor_config.json).

`vision_pixels.loom` prepares antialiased bicubic coefficient tables, preserving
the decoded RGB/RGBA plane and display orientation. A packet begins with a
64-byte reserved directory followed by a 64-byte resize header. Header-relative
offsets address the two axis plans and aligned decoded pixels. F32 weights use
the specified F64 sampling-center evaluation before explicit rounding. The
[device leaf](kernel/README.md) performs the pixel filtering.

`vision_control.loom` provides `text_image_prepare_request_v1`,
`text_image_prepare_batch_v1`, and `text_image_run_v1`. The request callback tags
compatible images by selected dimensions. Batch preparation fills the native
packer's directory in place for one to four images and describes their output
feature ranges. Each 32×32 output pixel block becomes one 5120-element F32 row.
`vision_stages.loom` declares the geometry-specific projector stage through the
prepare host; the runner owns device execution and storage. There is no second
cohort-sized pixel copy.

Cold preparation registers a descriptor-only memory forecast for bounded
serving. It accounts for the projector bank, the upload ceiling, and the largest
image's shared private workspace. Retained output estimates include a full
four-image cohort for each image in the request because peer requests can share
the allocation. Expanded feature storage counts only this request's rows.
The forecast needs neither decoded pixels nor activated weights; it lets the
text consumer check projection and continuation against the configured memory
budget before either phase runs.

The [tests](../test/README.md) check coefficient values, cohort feature plans,
all eight display orientations, and RGB/RGBA device filtering independently of
learned weights. The complete text and image-projector composition is described
by the [model card](../model.json).
