# Qwen3.8-27B composition

The card distinguishes text and image-conditioned provider sets and asset
variants. Both retain the ordered numerical provider closure; the vision set
adds the image preparation and control inputs. The `model`, `vision`, and
`draft` parameter scopes replace filename-driven asset roles. Auxiliary
`weights.loom` and `dflash_weights.loom` policies remain explicit compiler inputs.

The declared native providers are absent. Build the standalone description with:

```sh
bazel build //model/Qwen/Qwen3.8-27B:reference
```

The native port implements `text_prepare_v1` and `text_control_v1` plus
`text_image_prepare_v1` and `text_image_control_v1` for the vision component.
Its preparation program reads `qwen38.draft_kind`: zero selects the checkpoint
prediction path when drafting is enabled, and one selects the DFlash2 scope.
Draft depth and continuation remain execution settings. No separate drafting
service or arbitrary file-path option is introduced.

Qualification must preserve target-only, MTP, and DFlash2 token/state semantics;
RDNA and CDNA text selection; image ordering and projector lifetime; exact
retained-history behavior; cancellation; and joined shutdown. Image correctness
and text outputs need independent reference evidence. Provider source cleanup,
actual native implementations, and acquisition locks precede executable
publication. The card does not import experimental code or its deployment files.
