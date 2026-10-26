# Sequence carriers

These motifs interpret sequence work records and retained-state coordinates
shared by model compositions. They expose the producer's layout directly;
allocation, publication, request scheduling, and completion ownership stay
with the composition. They contain no launch, target, or configuration choice.

[`qwen3_8/`](qwen3_8/) carries the packed epochs, token/media records, and
retained row/page coordinates shared by the dense Qwen3.8 and Flash-Next
programs. Its layout is a concrete model-family contract, not a universal
serving protocol.
