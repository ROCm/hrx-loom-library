## About the model

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) produces
text embeddings for retrieval and related tasks. This card describes a
Strix Halo NPU composition that executes all 28 transformer layers across
32 AIE2P cores arranged in eight columns and four rows. It returns an
L2-normalized prefix of between one and 1,024 dimensions.

**Execution status:** this package contains the model description. Its native
providers and prepared resident-weight archive are not included, so the card
does not yet form a runnable Loom installation. The NPU layout is specific to
this composition; it is not a generic repacking of arbitrary embedding models.

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
not supplied by the original HF repository. A model publisher must provide the
reproducible native conversion and its output before publishing a complete
package. `loom-fetch` does not perform this BFP conversion as part of its
Safetensors alignment support. The original checkpoint is a preparation input,
not a second runtime parameter bank.

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

Qualification requires independent embedding comparisons, sequence isolation
and ordering, normalized prefixes, token usage, cohort rejection, cancellation,
and service shutdown on the actual NPU. The source model's general capabilities
and intended uses are documented in the
[publisher's model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
