# Qwen models

Model identities follow the original [Qwen](https://huggingface.co/Qwen)
training repositories. Derived quantizations, draft checkpoints, projectors,
and prepared hardware layouts belong to variants of those base models.

The model cards describe their asset and service contracts alongside their
native providers. [Qwen3.8-27B](Qwen3.8-27B/OVERVIEW.md) supplies text, drafting,
and image-conditioned compositions; its overview distinguishes qualified
presets from the separate DFlash2 service gate.
[Flash-Next](Qwen3.8-Flash-Next/OVERVIEW.md) supplies sharded text and MTP
compositions. [Qwen3 Embedding](Qwen3-Embedding-0.6B/OVERVIEW.md) supplies an
NPU composition with a pinned local checkpoint converter. Each package has
source, bytecode, and mixed representations, plus numerical and service checks.
