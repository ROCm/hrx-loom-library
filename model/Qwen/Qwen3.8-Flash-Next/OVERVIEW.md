## About the model

[Qwen3.8 Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) combines
36 recurrent layers and 12 sparse-attention layers. Its mixture-of-experts
blocks select ten of 512 experts plus a shared expert. This card describes the
48-layer composition using the original packed Unsloth UD-Q4_K_XL files,
with an optional BF16 image projector and Q8_0 multi-token predictor.

The composition includes native AMD RDNA providers for Loom's text service,
with source and bytecode program representations. Its source preset
can be fetched and served using the installed SDK, without building this
repository. For example, after installing the `hrx-loom` SDK with its `hf` extra:

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/Qwen/Qwen3.8-Flash-Next/model.json \
  --preset chat --output "$HOME/models/qwen3.8-flash-next"
loom-serve --model="$HOME/models/qwen3.8-flash-next/model.json" \
  --device=amdgpu://0
```

The source card selects the programs and orders the checkpoint shards; the
acquisition lock records checkpoint and tokenizer identities. Fetching resolves
the selected preset, copies editable programs, and reuses cached checkpoint
files; serving loads the resulting local card.
The source programs, numerical checks, and model service tests are available
in the [library model package](https://github.com/ROCm/hrx-loom-library/tree/main/model/Qwen/Qwen3.8-Flash-Next).

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
not make them inputs to the text-only selection.

The base model uses the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE).
The quantized repository retains that model license.

## Image-conditioned chat

The `vision` and `vision-mtp` presets add the matching original
[`mmproj-BF16.gguf`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/766911a6b7369840a91dbcd95f9f997acaab6cd6/mmproj-BF16.gguf)
in a separate `vision` parameter scope. The same source, bytecode, and mixed
representations support target-only and speculative image input. No offline
weight conversion is required.

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/Qwen/Qwen3.8-Flash-Next/model.json \
  --preset vision-mtp --output "$HOME/models/qwen3.8-flash-next-vision"
loom-serve --model="$HOME/models/qwen3.8-flash-next-vision/model.json" \
  --device=amdgpu://0
```

Send an image as a data URL through the chat endpoint:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

image = base64.b64encode(Path("photo.png").read_bytes()).decode("ascii")
body = {
    "model": "Qwen/Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
        {"type": "text", "text": "Describe this image."},
    ]}],
    "enable_thinking": False,
    "max_tokens": 128,
}
request = Request("http://127.0.0.1:8080/v1/chat/completions",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
print(json.load(urlopen(request)))
```

Up to four images may appear in a conversation. Each compressed image is
bounded to 64 MiB and 4096 pixels per dimension. Processing aligns dimensions
to 32 pixels and defaults to at most 1024 image tokens; `max_image_tokens`
can select 128 through 16384. Image tokens, chat text, and generated output
share the selected text context capacity. These image presets start with
the same 16,384-position context as `chat`.

The projector uses 27 transformer blocks and produces 2560-wide F32 language
features from original BF16 matrix and F32 affine parameters. Its parameter
bank occupies 907,530,240 bytes. It activates on the first image request and
can release that storage independently of retained image-conditioned
conversations. Encoder workspace, uploaded pixels, and retained features
are separate from the parameter bank. The text-only presets do not acquire
the projector or prepare image-specific stages.

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

Greedy requests can also reuse a complete proposal tail from accepted token
history. The GPU searches repeated suffixes, copies only committed IDs, and
compacts the remaining requests into the neural predictor cohort. Target
verification applies to both sources. The ID history shares draft KV page
ownership, so retained forks and restored requests preserve it without a
separate host-side history cache. Sampled requests use neural proposals.

Proposal programs specialize for power-of-two cohort sizes. Each invocation
selects from the live span count, so a one-request decode does not execute the
largest prepared grid. All variants share parameters and the fixed packet
layout; resident growth preserves their stage handles. Two-epoch continuation
queues its second proposal and verification without waiting for CPU acceptance
feedback.

## Capacity and memory

The canonical chat configuration starts with four resident rows,
16,384-position contexts, 32,768 pooled tokens, 16,384-token packed epochs,
four checkpoints, and a 256-token default generation limit. `long-context` starts with eight rows,
262,144-position contexts, 524,288 pooled tokens, and eight checkpoints.
`long-context-mtp` uses those capacities with three predictor proposals.
The context limit includes both input and output tokens.
Context capacity must be divisible by 64, an epoch holds at most 32,768 tokens,
and the composition requires pooled cache storage. Resident rows can grow
without redeclaring stages or changing the prepared packet capacity; cache
space and execution shapes bound admitted work independently.

The source parameter layout occupies approximately 103.68 GiB, including the
26.82 GiB packed n-gram table; adding MTP brings it to approximately 106.26 GiB.
The card rounds those estimates upward to 104 and 107 GiB. It does not infer a
hardware fit or reserve memory from those figures.

Mutable storage is additional: 25,344 bytes per pooled KV token and
115,652,096 bytes per recurrent endpoint. MTP adds 44,032 bytes of carry per
retained owner, 2,116 bytes per pooled token, and a 1,024-byte working tail per
execution row. The retained carry includes 22,784 bytes for draft-vocabulary
membership and history. Allocation granularity, workspace, compiler,
tokenizer, and driver overhead are also additional, so no complete transient
peak is claimed.
Physical placement remains a serving decision; the shard array describes
parameter loading rather than a multi-server topology.
