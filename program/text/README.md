# Text-service programs

Text programs own source-defined policy consumed through Loom's text service:
chat formatting, request options, streaming output interpretation, and queued
execution. Public exports follow the versioned `text_*_vN` ABI. Transport and
request scheduling remain in the service runtime.

[`qwen3_8/`](qwen3_8/) implements the shared dense/Flash Qwen policy, including
ordinary and media epochs, speculative acceptance, and retained history.
