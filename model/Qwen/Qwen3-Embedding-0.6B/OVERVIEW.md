## About the model

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) produces
text embeddings for retrieval and related tasks. This composition runs all 28
transformer layers on a Strix Halo NPU using eight columns and four core rows.
It returns an L2-normalized prefix of between one and 1,024 dimensions through
Loom's embedding service ABI 1.

## Fetch and serve

Install the Loom SDK using the
[installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/).
With the SDK installed, fetch the model card and start the server:

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/Qwen/Qwen3-Embedding-0.6B/model.json \
  --preset=embedding --output=~/models/qwen3-embedding

loom-serve --model=~/models/qwen3-embedding/model.json \
  --device='xdna://0?columns=8' --pool_backing=fixed \
  --listen=127.0.0.1:8080
```

This path requires a supported Strix Halo NPU on Linux. The eight-column context
and fixed resident backing are deployment settings. The default source programs
are linked and compiled by the installed SDK; a repository clone or Bazel build
is not required. Hugging Face acquisition uses the optional `hrx-loom[hf]`
dependency and reuses its cache. `--offline` restricts acquisition to cached or
locally overridden origins.

The only learned checkpoint is Qwen's original
[model.safetensors](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/model.safetensors),
installed as `parameter/model.safetensors`. The
[tokenizer](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/tokenizer.json)
comes from the same pinned snapshot. The publisher's
[architecture configuration](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/config.json)
describes the original model. Its weights use
[Apache-2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/LICENSE).

During model loading, Loom reads the original BF16 tensors and converts the
projection panels into the NPU's BFP16EBS8 representation in device memory.
Embedding rows and normalization weights retain their BF16 values. Conversion
runs once for the resident parameters, and requests reuse the resulting data.
The original checkpoint stays unchanged; serving creates no converted weight
file and requires no preparation command or Python conversion package.

Request a 256-dimensional embedding:

```sh
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"The library is open on Saturday.","dimensions":256,"encoding_format":"float"}'
```

The response contains one normalized vector of 256 numbers and reports eight
prompt tokens, including the end token. `encoding_format: "base64"` returns
little-endian F32 bytes instead of a JSON array. A publisher can host the card
and its program files at another URL while retaining the original checkpoint
origin. Recipients fetch that `model.json` URL and serve the resulting local
installation with the same device and pool options.

## Input and cohort contract

Each sequence accepts between one and 32,768 tokens. A logical request may
contain more sequences than fit in one execution cohort: the model processes
successive cohorts and returns their embeddings in original input order.
There is no model-specific limit on the number of rows in a logical request;
the service's request-body and available-memory limits still apply.

A cohort contains up to 256 panels of 128 token slots. Each sequence occupies
its token count rounded upward to eight slots, and stays whole within its
cohort. For example, a 32,768-token sequence fills one cohort; a second sequence
continues in the next cohort. Short sequences share panels. The service retains
completed embeddings while reusing the cohort's working storage.

Native execution yields after bounded amounts of work. Another request can
make progress between those epochs without recompiling the model or replacing
its resident parameter and storage allocations. Concurrent cohorts receive
disjoint ranges in a shared 512-panel arena; those ranges become reusable only
after captured device work retires.

Text tokenization includes the publisher's end token; explicit token IDs are
used as supplied. For retrieval queries, the caller provides the intended task
instruction, for example:

```text
Instruct: Given a question, retrieve passages that answer it.
Query: When is the library open?
```

Documents use their ordinary text. Instruction tokens count against the same
32,768-token limit. The `embedding` preset selects a 256-dimensional normalized
prefix and a 32-request pending queue. Queue size counts service requests,
independently of the number of input rows in each request.

## Storage and qualification

The immutable device parameters account for the 808,069,120-byte memory estimate.
Loading also uses 420 MiB of temporary conversion staging, retired after
activation. The fixed arena declares 2,042,626,048 bytes of numerical state,
4,194,304 bytes of token metadata, 268,435,456 bytes of vector storage, and
128 bytes of control records. Cohort result capacity is 16,777,216 bytes;
completed logical results also remain owned until the response is released.
These counts exclude allocator, compiler, and other runtime costs and do not
establish a complete transient-memory peak. Intermediate activations pass
through global memory between native phases.

Native service checks compare independent publisher embeddings and exact
regression outputs, sequence isolation and ordering, concurrent requests,
normalized prefixes, token usage, full-context and multi-cohort inputs,
cancellation, storage reuse, and joined shutdown on the
actual NPU. The source model's general capabilities
and intended uses are documented in the
[publisher's model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
