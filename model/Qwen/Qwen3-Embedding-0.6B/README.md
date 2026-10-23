# Qwen3 Embedding NPU composition

The card preserves the complete NPU provider set, fixed cohort geometry,
resident parameter layout, tokenizer identity, and original preparation inputs.
`policy.loom` is an explicit auxiliary compiler input. `OVERVIEW.md` describes
the original BF16 checkpoint and the distinct prepared byte bank independently
of this README.

The native providers and packed archive are absent. Generate the description:

```sh
bazel build //model/Qwen/Qwen3-Embedding-0.6B:reference
```

The executable port implements `embedding_prepare_v1` and
`embedding_control_v1`, loading the bank through `parameters.model` and the
tokenizer through `tokenizers.text`. Its eight-column placement, packet and
FIFO contracts, native packing, and 64-byte lookahead remain coordinated across
the parameter producer and the NPU programs. Alignment alone cannot convert
the original checkpoint to this representation.

Admission requires reproducible prepared assets, a lock on the actual output
archive, and independently checked embeddings on the intended NPU. Tests also
exercise exact cohort admission, independent sequence histories, selected
prefix normalization, explicit-token input, cancellation, and joined shutdown.
Full checkpoints remain outside Bazel; the card and acquisition lock provide
the identity inputs for provisioned learned-model regressions.
