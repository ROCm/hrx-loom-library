## About the model

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) produces
text embeddings for retrieval and related tasks. This card describes a
Strix Halo NPU composition that executes all 28 transformer layers across
32 AIE2P cores arranged in eight columns and four rows. It returns an
L2-normalized prefix of between one and 1,024 dimensions.

**Execution status:** source, bytecode, and mixed providers implement embedding
service ABI 1. This port uses an eight-column Strix Halo NPU context on Linux
with fixed resident backing. The prepared bank is reproducible from the pinned
checkpoint, but it is not hosted here. The repository card is excluded from
the public picker until a complete prepared publication is available.

## Original and prepared assets

The original
[BF16 weights](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/model.safetensors),
[architecture configuration](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json), and
[tokenizer](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/tokenizer.json)
come from the pinned publisher snapshot. The model uses
[Apache-2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/LICENSE).

Native execution consumes a prepared Safetensors file containing one I8 tensor
named `resident`, with 808,069,120 bytes and a 64-byte-aligned payload. I8 is
the bank's byte-storage type: its projection panels contain native BFP16EBS8
records, while scales, embeddings, rotary values, and the SiLU table use BF16.
Each packed projection record includes 64 bytes of required FIFO lookahead.
The exact regions, dimensions, and offsets appear in the card below.

The card installs this archive as `parameter/resident.safetensors` from its
card-relative `prepared/weights.safetensors` source. That prepared archive is
not supplied by the original HF repository. The library includes its pinned
converter and publishes a standalone card and lock alongside the generated
archive. `loom-fetch` does not perform this BFP conversion as part of its
Safetensors alignment support. The original checkpoint is a preparation input,
not a second runtime parameter bank.

## Prepare and serve

Install the Loom SDK using the
[installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/).
The first PyPI release is pending; development wheels provide the same native
tools. With [uv](https://docs.astral.sh/uv/) installed, clone the
[library](https://github.com/ROCm/hrx-loom-library), then run from its root:

```sh
uv run model/Qwen/Qwen3-Embedding-0.6B/prepare_assets.py \
  --output=../qwen3-embedding

loom-serve --model=../qwen3-embedding/model.json \
  --device='xdna://0?columns=8' --pool_backing=fixed \
  --listen=127.0.0.1:8080
```

Preparation writes an approximately 808 MB bank plus the tokenizer and source
closure, while retaining the original checkpoint in the HF cache. It verifies
the exact prepared payload before publishing the card. Add `--offline` to use
only checkpoint files already cached; `uv run --offline` separately controls
Python package acquisition. Serving the result requires neither Bazel nor the
preparation Python dependencies. The fixed-pool option is explicit deployment
configuration, separate from the model's program representation.

Request a 256-dimensional embedding:

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"The library is open on Saturday.","dimensions":256,"encoding_format":"float"}'
```

The response contains one normalized vector of 256 numbers and reports eight
prompt tokens, including the end token. `encoding_format: "base64"` returns
little-endian F32 bytes instead of a JSON array. The prepared directory is a
portable source publication: a publisher can host it, and a recipient can fetch
its `model.json` URL with `loom-fetch` before serving with the same device and
pool options.

## Input and cohort contract

One execution cohort contains 128 token slots and at most sixteen independent
sequences. Each sequence consumes its token count rounded upward to eight
slots. Four 32-token inputs or sixteen inputs of at most eight tokens therefore
share one complete weight pass. A sequence longer than 128 tokens, or a request
whose rounded sequences exceed the cohort, is rejected.

Text tokenization includes the publisher's end token; explicit token IDs are
used as supplied. For retrieval queries, the caller provides the intended task
instruction, for example:

```text
Instruct: Given a question, retrieve passages that answer it.
Query: When is the library open?
```

Documents use their ordinary text. Instruction tokens count against the same
128-token limit. The `embedding` preset selects a 256-dimensional normalized
prefix and a 32-request pending queue. Queue size does not enlarge a cohort.

## Storage and qualification

The immutable prepared bank accounts for the 808,069,120-byte memory estimate.
The composition also declares 2,924,544 bytes of mutable state, 8,192 bytes of
control data, and a 524,288-byte output region. These counts do not include
every allocator, compiler, or runtime cost and do not establish a complete
transient-memory peak. A resident invocation still transfers intermediate data
through global memory; it does not imply that every intermediate stays in NPU
local memory.

Native service checks compare independent publisher embeddings and exact
regression outputs, sequence isolation and ordering, concurrent requests,
normalized prefixes, token usage, cohort rejection, and joined shutdown on the
actual NPU. The source model's general capabilities
and intended uses are documented in the
[publisher's model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
