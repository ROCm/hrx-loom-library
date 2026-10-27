# Flash-Next text programs

The model composes the shared Qwen text service with its own parameter layout,
48-layer execution, recurrent state, sparse attention, and multi-token
predictor. The shared service owns chat formatting, request packing, retained
request ownership, and the versioned text service exports. These providers
own the checkpoint-specific stage roots, tensor bindings, and device work.

## Preparation and parameter ownership

`prepare.loom` resolves the card's `model` and optional `draft` parameter
scopes once. Every specialized stage refers to those same source handles;
changing a batch shape does not load another checkpoint. Four ordered GGUF
files form the target scope. The first contains metadata, and the remaining
three contain tensor payloads. The predictor scope contains one Q8_0 file.
Parameter identities allow its identical embedding and vocabulary tensors to
share the target's storage.

Preparation declares the stage catalog and immutable capacities. Packed
epochs carry the live token count, spans, row ownership, page IDs, and
recurrent frontiers. `weights.loom` is an explicit auxiliary compiler input:
it selects the in-place preparation command for the 36 GDN QKV tensors.
Each disjoint four-channel group is read into workgroup storage before its
codes and scales replace the original bytes. Other weights retain their
GGUF representation. Reactivation loads original checkpoint bytes before
preparing them again.

## Execution and retained state

The target preserves four BF16 residual streams of width 2,560. Three GDN
layers alternate with one sparse-attention layer. GDN carries F32 recurrence
and three BF16 convolution rows per layer. Sparse attention uses pooled
physical KV pages, compressed index blocks, and a private partial-block tail.
PLE adds hashed n-gram lookup and its own token and convolution history.
Read anchors and writable endpoints are distinct arguments, allowing forks
to share immutable history while publishing into private state.

MTP has separate proposal, target-verification, accepted-prefix replay, and
catch-up stages. Recursive proposals consume private working history.
Target sampling decides acceptance; compact captured inputs replay only the
accepted prefix into non-page state. Catch-up uses accepted target hidden
rows to advance predictor state. Draft vocabulary membership is learned from
accepted tokens and is retained with the request.

The input-content kernels distinguish ordinary IDs from media records using
composition facts. Target embedding reads original Q8 vocabulary rows or F32
features and publishes BF16 residuals. Accepted predictor embedding binds the
original feature packet through the compact span-origin map; features are
never repacked for catch-up. Ordinary catch-up retains its compact-ID path.
PLE hashes, token-history publication, and speculative ID capture read IDs
through the shared input contract. Replay and draft vocabulary continue to
consume compact I32 histories. These input-content contracts do not by
themselves advertise an image service on the model card.

Rotary coordinates are independent of causal cache positions. The four-word
token map carries causal position, resident row, packed span beginning, and
source position. Image records supply temporal, height, and width coordinates;
ordinary text uses the source position. Compact queries and accepted predictor
rows resolve original records through their span-origin map, whose extent is
independent of the original input allocation.

A compressed index key rotates at the first member of its four-token group.
When a group crosses an epoch boundary, its original coordinate survives with
the partial raw tail in one 16-byte record shared by all twelve indexer layers.
The last layer publishes that record after every layer has read the old value.
Forks read the immutable anchor, and later epochs read their private writer.
Accepted-prefix replay publishes from descriptors and original inputs without
retaining a token map. Predictor restore and commit transfer the same record
within the existing carry stride. Fresh or aligned endpoints never read it.
No per-context coordinate history, packet copy, or extra dispatch is required.

Greedy proposal rounds first search accepted token history. A trigram bitset
locates the newest sixteen occurrences, and backward comparison selects the
longest suffix, breaking ties toward the newest occurrence. A match of at least
twelve tokens supplies the complete remaining proposal tail only when those
IDs are already committed. The target still verifies every proposed token.
Sampled requests retain neural drafting.

The accepted-ID plane shares the draft KV page map and registered retention
geometry, so forks, page reuse, and restoration move it with the cache. Pending
proposal IDs join the query without advancing the accepted frontier. Copied
rows leave the neural cohort; compaction preserves both original target input
and selected-output coordinates, independently of resident row identity. A
GPU-authored count controls every remaining neural launch without a host
readback. Hidden state gathers into a separate transient buffer to keep
cross-workgroup reads independent of compaction writes.

Output-only stages at capacities of 128 tokens and above compact the final
attention layer after every input row has been appended to its cache. Only
selected query cohorts continue through attention and experts; results
scatter back to their original packed rows.
The normalized path retains its separate normalization and injection below
2,048-token capacity, while larger shapes retain the fused arithmetic.
Speculative verification and predictor catch-up consume the full hidden-state
root and keep every row. Packet and scratch extents follow the configured
span and token capacities rather than a fixed client count.

The vocabulary projection shares each decoded Q8 row across four selected
queries for small output capacities and eight for larger ones. Query groups
occupy the contiguous grid axis, allowing neighboring workgroups to reuse the
same vocabulary rows. Each query retains its original block reduction order
and BF16 logit publication; inactive queries neither read inputs nor publish
outputs.

## Panel reuse within an epoch

Large GDN projection cohorts use 128-token by 256-channel workgroups. Eight
waves share a K64 input panel; each lane retains two prepared K32 weight
blocks in registers and exchanges them within its wave to form matrix
operands. This doubles input reuse across output channels. Narrow cohorts
keep their smaller schedules so that the wider register footprint does not
penalize short requests.

The paired projection publishes one 128-channel head at a time through the
same 33 KiB shared-memory region. A collective barrier separates each
producer phase from convolution and halo consumers, and another prevents
the next head from overwriting live values. Both schedules call one
publication body. BF16 rounding, retained history ownership, and speculative
capture remain at the same boundaries.

Sparse scoring keeps each 64-key panel resident while two four-query phases
consume it. Query scratch changes only after all waves finish the previous
phase; each query retains its own reduction order and cache-page ownership.
Sparse attention exchanges eight-word lane records as two four-word planes
so 128-bit LDS transfers use neighboring bank groups. Both BF16 probability
components and the original peer-reduction order are retained.

Q8 matrix panels express signed byte-to-half conversion directly with
`vector.sitofp`. Every I8 value is exactly representable in F16; block scaling
and F32 accumulation follow the existing operand-rounding contract. These
schedules change operand reuse and exchange layout without changing the
checkpoint quantization or adding a reduced-accuracy execution mode.

## Link boundaries

The dependency graph follows execution ownership:

```text
shared format, sequence, and tensor providers
                      |
                kernel:numerics
                      |
                dispatch:dispatch
                      |
                  :commands

shared text service + :prepare + :identity + :config
                      |
                   model card
```

The numerical library contains device implementations and their specialization
templates. Dispatch binds checkpoint tensors, allocates intermediates, and
orders device launches. Model commands compose complete layers and epochs.
Neither lower library calls back into a model command. The `config` provider
owns fixed checkpoint facts and default specialization values.

`source` and `archive` select the same public roots. Archive inputs
are independently linked relocatable bytecode, still specialized at runtime.
Both representations retain the weight policy as an auxiliary input. Full
checkpoints never enter the
Bazel payload graph.
