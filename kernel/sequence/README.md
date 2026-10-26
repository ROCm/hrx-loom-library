# Sequence kernels

These kernels publish sequence-control data between model stages. They consume
validated model-family layouts and make their launch ABI explicit; request
admission, allocation, scheduling, and completion ownership stay with the
calling program. Shared storage representations live in `motif/sequence/`.

[`qwen3_8/`](qwen3_8/) implements greedy acceptance, multi-token proposal
publication, and queued continuation for the dense Qwen3.8 and Flash-Next
compositions.
