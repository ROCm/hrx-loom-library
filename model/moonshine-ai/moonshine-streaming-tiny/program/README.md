# Moonshine recognition programs

The host programs export recognition ABI 1 roles. Preparation registers
`reset`, `encode`, and `decode` and binds the card's named parameter scopes.
Control maintains request state, packs compatible work, schedules a stage,
decodes output token IDs, and emits typed transcript revisions. No filesystem
path or physical device identity is embedded in the model's parameter contract.

The three command programs compose the trained architecture with explicit
tensor layouts and resident-state offsets. Kernels consume packed activations
but locate histories using the cohort's resident IDs. Parameters stay F32,
row-major, and unchanged from the checkpoint. Shared numerical kernels retain
the same arithmetic across layers; layer-specific parameter keys and offsets
belong to the command programs.

Each file has an independently declared relocatable bytecode target. The
portable card can consume these products, the text sources, or text host
programs with bytecode device providers. The weight policy is auxiliary input
to preparation and returns zero staging bytes, preserving the original tensors.

The numerical port derives from the IREE Moonshine implementation under
Apache-2.0 WITH LLVM-exception. Attribution and the complete license are retained
in the parent package. Source text carries operation/ABI documentation rather
than repeated copyright headers.
