# Krea 2 Turbo composition

This package specifies the native image-service port. `model.json` owns the
two checkpoint variants, three parameter scopes, tokenizer identity, complete
ordered provider set, image geometry constraints, and canonical presets.
`OVERVIEW.md` supplies the standalone user-facing part of the generated reference.

The card's `program/` and `library/` paths describe the installation namespace;
the corresponding native providers are not present. The provider set separates
model preparation and request control from encoder, transformer, and VAE
composition, with reusable tensor operations under `library/motif/tensor/`.
The preparation policy `weights.loom` is an explicit auxiliary compiler input.
The model-specific `krea2.dit_int8` setting stays at the composition root.

Build the description without acquiring checkpoints:

```sh
bazel build //model/krea/Krea-2-Turbo:reference
```

The executable port must implement `image_prepare_v1` and `image_control_v1`,
load the declared parameter scopes, and retain request-specific geometry
specialization. Its source providers use `loom_library`; its selected package
uses `loom_model`. Qualification includes independent image comparisons for
both checkpoint variants, exact tiled/untiled VAE agreement, request geometry
rejection, and joined service shutdown. The acquisition lock is generated from
that actual provider and asset closure. No lock or executable placeholder is
substituted for the absent implementation.
