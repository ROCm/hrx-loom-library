## About the model

[Qwen3.8 Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) combines
36 recurrent layers and 12 sparse-attention layers. Its mixture-of-experts
blocks select ten of 512 experts plus a shared expert. This card describes the
48-layer text composition using the original packed Unsloth UD-Q4_K_XL files,
with an optional Q8_0 multi-token predictor.

**Execution status:** the native library providers are not included. The card
and standalone reference describe the required inputs and qualification
presets; they do not yet form a runnable Loom installation. The intended
numerical implementation targets AMDGPU RDNA.

## Ordered checkpoint shards

The `ud-q4-k-xl` variant loads four files from the pinned
[Unsloth checkpoint directory](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL)
into one `parameters.model` scope. Shard 00001 carries metadata; shards 00002
through 00004 carry tensor payloads. Their array order is intentional. The
file table below preserves each exact origin path while naming the installed
files `parameter/model-00001.gguf` through `parameter/model-00004.gguf`.

The tokenizer comes from the
[original training repository](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json).
Only these four weight shards and the tokenizer are required for target-only
text execution. The presence of other files in the upstream repository does
not make them inputs to this model selection. No image projector is declared.

The base model uses the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE).
The quantized repository retains that model license.

## Optional multi-token prediction

`ud-q4-k-xl-mtp` adds
[`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/mtp-Qwen3.8-Flash-Next-Q8_0.gguf)
in the `draft` parameter scope. It supports one through seven recursive
proposals. The predictor proposes greedily; the target verifies under the
request's sampling policy and commits only the accepted prefix. The matching
embedding and vocabulary tensors can share target allocations. The source layout requires
2,775,621,632 additional parameter bytes for the remaining predictor tensors.

`chat` uses target-only execution. `mtp` enables three proposals, and
`mtp-continuation` also enables two-epoch continuation. These are execution
choices within the same text service. More speculative work is not itself a
throughput guarantee; qualification compares equal prompts, sampling, output
budgets, and retained state.

The predictor's vocabulary includes a 98,304-token prefix, all 33 checkpoint
control tokens, and up to 991 rare tokens learned from accepted request
history. An unknown anchor or two new misses within the last 32 accepted
tokens selects the full vocabulary. Proposals do not teach this history;
only accepted tokens do. The choice stays fixed through a recursive proposal
chain, and checkpoint forks retain the history with the request. The target
continues to score the full vocabulary when verifying proposals.

## Capacity and memory

The canonical chat configuration uses four rows, 16,384-position contexts,
32,768 pooled tokens, 16,384-token packed epochs, four checkpoints, and a
256-token default generation limit. `long-context` records eight rows,
131,072-position contexts, 262,144 pooled tokens, and eight checkpoints.
Context capacity must be divisible by 64, an epoch holds at most 32,768 tokens,
and the composition requires pooled cache storage.

The source parameter layout occupies approximately 103.68 GiB, including the
26.82 GiB packed n-gram table; adding MTP brings it to approximately 106.26 GiB.
The card rounds those estimates upward to 104 and 107 GiB. It does not infer a
hardware fit or reserve memory from those figures.

Mutable storage is additional: 25,344 bytes per pooled KV token and
115,651,840 bytes per recurrent endpoint. MTP adds 44,032 bytes of carry per
retained owner, 2,112 bytes per pooled token, and a 768-byte working tail per
execution row. The retained carry includes 22,784 bytes for draft-vocabulary
membership and history. Allocation granularity, workspace, compiler,
tokenizer, and driver overhead are also additional, so no complete transient
peak is claimed.
Physical placement remains a serving decision; the shard array describes
parameter loading rather than a multi-server topology.
