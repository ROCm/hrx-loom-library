# Text-service programs

Text programs own source-defined policy consumed through Loom's text service:
chat formatting, request options, streaming output interpretation, and queued
execution. Public exports follow the versioned `text_*_vN` ABI. Transport and
request scheduling remain in the service runtime.

[`publication.h`](publication.h) is the C++ source interface to completed-epoch
publication. Its nested records group borrowed inputs, completed results, and
caller-owned outputs while flattening to the native fourteen-argument v1 ABI.
The `:publication` provider exports the header without a bytecode module.
Model-specific feedback layouts remain with their producers; this interface
carries them as opaque borrowed buffers.

[`qwen3_8/`](qwen3_8/) implements the shared dense/Flash Qwen policy, including
ordinary and media epochs, speculative acceptance, and retained history.
