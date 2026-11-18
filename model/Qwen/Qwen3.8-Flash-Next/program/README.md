# Flash-Next text and vision programs

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
it configures one in-place preparation command for the QKV, gate, and output
matrices in all 36 GDN layers. Each disjoint four-channel group is read into
workgroup storage before publishing contiguous scale and code planes. The
original F16 scales and I8 codes keep their exact bytes and total extent;
preparation adds no persistent copy or numerical conversion.

The GDN projection commands carry the prepared-layout fact through both dot
and matrix selection. Other dense callers explicitly retain their original
GGUF layout. This makes storage a property of the bound tensor rather than a
guess from its shape. Reactivation loads original checkpoint bytes before
preparing them again.

## Execution and retained state

The target preserves four BF16 residual streams of width 2,560. Three GDN
layers alternate with one sparse-attention layer. GDN carries F32 recurrence
and three BF16 convolution rows per layer. Sparse attention uses pooled
physical KV pages, compressed index blocks, and a private partial-block tail.
PLE adds hashed n-gram lookup and its own token and convolution history.
Read anchors and writable endpoints are distinct arguments, allowing forks
to share immutable history while publishing into private state.

`ple_prepare.loom` declares the request and independent-prefix catalogs plus
one cached-host request root. `ple_control.loom` implements the shared epoch
template with a finite HAL file batch: request dispatch, producer fence,
independent layer-zero work, and a completion join before the suffix. The file
resolver supplies the retained checkpoint handle and exact tensor byte window;
ordinary queue capture retains the file and buffers through completion. The
source stores only stable stage/parameter/storage ordinals and one cache
initialization flag; row growth preserves them. No host request-list readback,
per-row host submission, or model-specific runtime API is involved.

The finite PLE stages separate request generation, layer zero, and the remaining
target layers. The requester reads the old token history and writes a live byte
range list without advancing that history. Layer zero has no PLE dependency and
can execute while file reads fill the packed row cache. The suffix starts after
read completion, publishes filled cache entries, and uses the same decoding,
projection, and convolution as the resident path. Layer bounds and input mode
specialize the existing unfused, normalized, and fused bodies; they do not
introduce runtime layer dispatch.

Speculative verification retains both prefix and suffix weight domains, captures
all transition inputs, and replays only accepted history. Its layer-zero replay
uses the prefix domain. Resident verification supplies one domain for both and
retains the complete-body root. No request or cache-planning stage publishes
speculative state on behalf of the verifier.

Each sparse-attention layer pairs K and V within four-token, single-head
records. A record contains a 2 KiB K tile followed by a 2 KiB V tile. K keeps
its `[channel / 16, member, channel % 16]` order; V keeps its
`[channel, member]` order. Appenders write those encodings directly, and
attention reads contiguous operand packets without a conversion pass.
Sixteen panels across both heads fill one 128 KiB, 64-token ownership page.
Target and predictor storage publish this byte geometry through the ordinary
retention descriptors; the shared runtime does not interpret the tile format.
The independently scanned compressed index remains a separate plane.
Native-code variants must preserve this retained-state representation;
changing its geometry requires a fresh model instance.

MTP has separate proposal, target-verification, accepted-prefix replay, and
catch-up stages. Recursive proposals consume private working history.
Target sampling decides acceptance; compact captured inputs replay only the
accepted prefix into non-page state. Catch-up uses accepted target hidden
rows to advance predictor state. Draft vocabulary membership is learned from
accepted tokens and is retained with the request.

Text preparation registers shorter paired proposal/verifier stages for rounded
cohorts of four and eight requests. The shared packet owner selects both stage
identities before enqueue; queued work keeps its captured choice while later
cohorts can choose another. A three- or four-owner cohort uses at most two
proposals, and a five- through eight-owner cohort uses one. One or two owners
retain the configured ceiling. Alternative stages borrow the same parameters
and retained storage; preparation adds no copies or model-specific host policy.

Within each recursive chain, a predicted probability below 0.85 stops that
owner's subsequent neural steps. The current prediction remains a target
verification candidate. Stable compaction retains the original target span in
shared proposal origins, so removing stopped owners cannot change request or
media identity. The threshold and cohort policy do not modify target sampling
or accepted-prefix verification. Prefill, media, supplied candidates, and
cohorts beyond the prepared alternatives retain their ordinary stage pairing.

The input-content kernels distinguish ordinary IDs from media records using
composition facts. Target embedding reads original Q8 vocabulary rows or F32
features and publishes BF16 residuals. Accepted predictor embedding binds the
original feature packet through the compact span-origin map; features are
never repacked for catch-up. Ordinary catch-up retains its compact-ID path.
PLE hashes, token-history publication, and speculative ID capture read IDs
through the shared input contract. Replay and draft vocabulary continue to
consume compact I32 histories.

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

Neural sampling also computes the winning token's probability over the active
draft vocabulary. A probability below 0.85 publishes that candidate but stops
further neural proposals for its owner. The final scheduled publication packs
the surviving target prefixes, including when all owners stopped earlier.
Target verification still uses the full vocabulary and unchanged sampling and
acceptance rules; the confidence threshold changes only how much draft work
is attempted.

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
queries for small output capacities and eight for larger ones. A device-sized
worker pool traverses live query/vocabulary panels while retaining that reuse.
Head mixing also shares weights across query panels. Each query retains its
original block reduction order and BF16 publication; inactive queries neither
read inputs nor publish outputs. Large residual stages reuse workers across
live rows with fixed stream/feature ownership, while small shapes specialize
to direct ownership without a row loop.

The recurrent update shares the register-only gated-delta motif with Qwen3.8-27B.
Four adjacent lanes retain one 128-key state column, with contiguous fragments
of 32 keys per lane. A bounded worker pool traverses the live span/column pairs
without multiplying the unbounded span coordinate by the column count. The
model adapter owns BF16 publication, normalization, the near-one decay
polynomial, and immutable-reader/private-writer selection; the retained F32
state layout and accepted-prefix replay contract are unchanged.

## Original image and video projector

Image-enabled variants set `flash_next.image_inputs` and add the image service
exports. Preparation declares ordinary and mixed-input target, proposal, warm,
verification, and continuation stages using one catalog builder and the same
immutable parameter handles. The shared controller chooses the corresponding
catalog from each packet's actual input representation. Text-only variants
retain only the ordinary catalog.

The projector reads original BF16 matrices and F32 affine parameters from the
separate `vision` scope. `vision_projector.loom` owns the complete resize,
27-layer encoder, and 2560-wide merger command. Its static dispatch leaves
specialize shared vision motifs with checkpoint dimensions and BF16 precision.
The dense Qwen model uses those same algorithms with its own F16 policy.

Each image or temporal pair keeps all heads' keys and values resident through
its own attention segment. Compatible groups stack rows across linear and
feed-forward projections to share weights. Two F32 residual banks alternate
without copying complete layers. Both temporal planes reuse the activation
bank during preprocessing; an image or odd video tail aliases one plane.

One pure workspace query serves command allocation and service forecasting.
Additive patch-row and upload-byte admission bounds each cohort; allocations
follow live extents, and output ranges are disjoint. A retained request owns
its compact features plus at most one shared output root. The service owns
upload completion, feature retention, and release; projector weight residency
does not own any retained conversation's lifetime.

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

Sparse scoring keeps each 64-key panel resident across a 128-query window,
consumed in four-query phases. Query scratch changes only after all waves
finish the previous phase; each query retains its own reduction order and
cache-page ownership. Workers retire both shared panels before advancing to
another key panel or request. The physical worker grid scales with device
capacity, while each worker traverses its share of the live spans and panels.
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
