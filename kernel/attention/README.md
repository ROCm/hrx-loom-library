# Attention kernels

Attention kernel packages own launchable operations that project, score,
normalize, or combine attention state. Their public configuration describes
semantic model dimensions; each `kernel.def` derives target-private launch
geometry from those values. Reusable packed formats and numerical algorithms
remain in `motif/`.

## Map

| Path | Contents |
| --- | --- |
| [`qwen3_5/`](qwen3_5/) | Qwen3.5 full-attention projection kernels. |

Callers compose these kernels into complete attention programs and own the
lifetime of activation, cache, and output buffers. Kernel packages own one
operation ABI and its correctness and benchmark witnesses, not a model's layer
schedule or service policy.
