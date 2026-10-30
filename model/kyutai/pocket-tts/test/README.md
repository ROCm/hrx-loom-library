# Pocket model checks

Synthetic tensors exercise causal histories, voice conditioning, streaming
state, sampling identity, and model-specific numerical composition without
checkpoints or reference recordings. Reordered cohorts compare with isolated
rows while complete storage observations retain inactive residents and guards.
Shared text and tensor motifs own their reusable arithmetic and Unicode checks.

Each test file records independent reference provenance and numerical
acceptance; scenarios own geometry and directed cases. The
[development guide](../../../../docs/development.md#build-and-check) covers
ordinary tests, optional instrumentation, and separate benchmark execution.

## Full learned speech

The explicit `speech_*_test` targets exercise both voice variants in source
and archive form. Each test starts the real native server with built
providers and a separately provisioned installation. Sequential requests
establish each voice's PCM; concurrent requests must reproduce those bytes
while rows and active cohorts change. Changing the seed must change the output.
A paragraph longer than one text chunk replays exactly, and a disconnected
response must retire with an incremented cancellation counter before its row
can be reused. Invalid voices, empty input, unsupported speed/instructions,
and excessive input fail with HTTP diagnostics.

The checker verifies service and lifecycle invariants. The numerical
expressions and upstream text corpus provide independent operation-level
references. Perceptual model quality requires a separate evaluation corpus.

Full checks retain short PCM responses, the server log, and `result.json` only
after successful shutdown. They trust provisioned parameters to match the
authored acquisition lock; the [developer workflow](../README.md) explains
their cache identity and invocation. No learned-model checkpoint is a test
input or retained output, and valid asynchronous work has no per-request
wall-clock deadline.
