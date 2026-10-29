# Tensor numerical motifs

These motifs provide numerical building blocks for model programs: tensor
contractions, attention, normalization, sampling, and elementwise transforms.
They operate on caller-owned values and storage. Composition owns model shapes,
parameter identities, launch geometry, concrete targets, and storage lifetimes.

The reuse boundary is an arithmetic or storage contract. A scalar transform can
be fused into a larger program; a cooperative contraction can retain panels and
accumulators before a separate publication step. Different rounding boundaries,
layouts, or aliasing rules remain explicit even when two operations implement
the same mathematical formula. Each source documents those contracts beside
its public declarations.

Target-neutral algorithms coexist with native fragment implementations. A
fragment provider names its required subgroup and external target contract;
the composition leaf supplies the concrete binding. Storage and synchronization
remain explicit, so linking a motif introduces no hidden allocation, dispatch,
or model-wide configuration.

The [test package](test/) owns independent oracles and bounded workloads.
[Component authoring](../../docs/component-authoring.md) covers dependencies,
source/test separation, and the shared verification workflow.
