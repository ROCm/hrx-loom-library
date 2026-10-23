# Motifs

Motifs are reusable Loom functions and templates. They capture representation,
algorithm, scheduling, or target knowledge without choosing an externally
launchable kernel ABI. Kernel, model, and command-program packages link motifs
and specialize them with the value facts available at their composition roots.

## Map

| Path | Contents |
| --- | --- |
| [`format/`](format/) | Physical tensor representations and their reusable operations. |
| [`linear/`](linear/) | Packed-weight contractions and reusable row schedules. |
| [`text/`](text/) | Unicode properties, full uppercase, UTF-8 traversal, and trimming. |
| [`tensor/`](tensor/) | Sampling, ordered selection, row normalization, and grouped causal convolution. |
| `quantize/` | Representation-independent quantization algorithms once concrete formats establish a shared contract. |
| `attention/` | Reusable attention algorithms and schedules. |
| `target/` | Target-specific realizations of otherwise shared motifs. |

Only namespaces with an implemented owner are created. Their scoped READMEs
become the durable home for specifications, source provenance, design
constraints, and implementation status.

## Dependency contract

A motif package contains `func.def` and `template.def` building blocks and is
declared with `loom_motif_library`. Each library owns one direct module and
names every motif whose public symbols it uses as a direct dependency. Motifs
may depend on other motifs and on external Loom libraries. They never depend on
`kernel/` or declare `kernel.def`; launch configuration and public execution ABI
belong to kernel or composition roots.

Configuration values enter motifs through function or template arguments.
The `config.decl`, `config.def`, and `config.get` operations stay in composition
and test leaves. Concrete target records such as `amdgpu.target` also belong
to those leaves; motifs may declare an external target contract or query their
execution context without embedding a target choice in every linked use.

Format-specific and target-specific motifs remain motifs. Reuse is the layer
boundary; universality is not. Shared abstractions are extracted only after
multiple concrete implementations establish the same invariant.
