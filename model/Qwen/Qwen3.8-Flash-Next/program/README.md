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
accepted tokens and is retained with the request. Output-only target stages
may compact final-layer work; MTP's full-hidden consumers keep their own root.

The vocabulary projection shares each decoded Q8 row across four selected
queries for small output capacities and eight for larger ones. Query groups
occupy the contiguous grid axis, allowing neighboring workgroups to reuse the
same vocabulary rows. Each query retains its original block reduction order
and BF16 logit publication; inactive queries neither read inputs nor publish
outputs.

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

`source`, `archive`, and `mixed` select the same public roots. Archive inputs
are independently linked relocatable bytecode, still specialized at runtime.
Mixed inputs keep preparation, identity, and shared chat control as text and
use bytecode for numerical and command providers. All representations retain
the weight policy as an auxiliary input. Full checkpoints never enter the
Bazel payload graph.
