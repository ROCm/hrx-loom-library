# Qwen3.8 Flash-Next composition

The card preserves the four-file parameter scope and optional MTP checkpoint,
shared Qwen request/control providers, Flash-Next numerical composition,
capacity constraints, and ordinary/long-context/speculative presets. Ordered
providers under `library/model/Qwen/Qwen3.8-27B/` are installation inputs; they
do not require a particular relative checkout layout.

The native providers are absent. The description builds without any weights:

```sh
bazel build //model/Qwen/Qwen3.8-Flash-Next:reference
```

The native text port loads all four base shards into `parameters.model` in
order and uses `parameters.draft` only for the MTP variant. Weight preparation
retains packed representations, shared target/predictor tensors, and identical
parameter identities across specialized batch shapes. The source preparation
policy remains an explicit auxiliary input.

Executable admission requires provider cleanup, the versioned text service
exports, and independently checked target-only/MTP outputs and retained state.
Tests cover shard ordering, packed epochs, context bounds, accepted-prefix
replay, EOS, cancellation, and joined shutdown. The publication lock covers
actual providers and checkpoint identities; Bazel receives the card and lock
as regression inputs while full weights stay in provisioned storage.
