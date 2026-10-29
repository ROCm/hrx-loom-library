# Vision component checks

These checks call the shared vision motifs without model providers, checkpoints,
or a serving process. Sparse products, analytical spatial fields, and scalar
arithmetic supply independent answers instead of reusing matrix fragments or
tile layouts. Geometry, precision, tails, and guard coverage belong to the
owning scenarios and assertions.

Frame fixtures represent decoded pixels, so component tests need no video
container or decoder. Workspace checks use descriptor-only byte extents rather
than allocating the represented payload. Private half-conversion and reference
helpers are test roots and never enter production dependencies.

The [shared text-program checks](../../../../program/text/qwen3_8/test/) cover
packet preparation, cohort policy, and CPU-to-GPU filtering together. Complete
learned service behavior remains with each model. The
[development guide](../../../../docs/development.md#build-and-check) owns test,
instrumentation, and benchmark commands.
