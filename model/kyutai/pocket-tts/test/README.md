# Pocket numerical and state checks

These workloads use synthetic tensors and the public Loom case/benchmark
runner. They need no checkpoint, prepared voice bank, or network access.
Each test leaf declares its exact geometry and direct provider dependencies.

Projection, convolution, and normalization checks use independent numerical
expressions. Streaming-state checks cover causal histories, overlap tails,
latent recurrence, and the waveform startup fade. Batched checks compare
reordered cohorts with isolated rows while observing complete output and state
banks, including nonzero origins, padding, inactive residents, and guards.
Conditioning checks use independent rectangular prefix copies to verify
retained voices, one writer per voice, and reused request rows.

Sampling checks cover the full request seed, distinct generation counters,
replay, and row-membership changes. Their expected normal samples follow
Random123 Philox4x32-10 and F32 Box–Muller arithmetic; shared motif tests also
check canonical Philox vectors and an independent double-precision normal
oracle. The maximum context layout check exercises byte extents beyond signed
32-bit storage without allocating that maximum-size model.

The text normalization corpus records the upstream Pocket English policy for
357 inputs, including Unicode uppercase expansion, whitespace, punctuation,
closers, and tail selection. Its generated data is separate from the Loom
implementation. Numerical tests run through the native AMD GPU profile; pure
text checks use the VM reference profile.

```sh
bazel test --config=asan --config=amdgpu \
  //model/kyutai/pocket-tts/program/... \
  //model/kyutai/pocket-tts/test/...
```

Each case has a benchmark declaration for the same bounded workload. Normal
test execution also runs one benchmark iteration. These checks establish
correctness and executable workload coverage; performance measurements use
the optimized profile and a qualified measurement host.

## Full learned speech

The explicit `speech_*_test` targets exercise both voice banks in source,
archive, and mixed form. Each test starts the real native server with built
providers and a separately provisioned installation. Sequential requests
establish each voice's PCM; concurrent requests must reproduce those bytes
while rows and active cohorts change. Changing the seed must change the output.
A paragraph longer than one text chunk replays exactly, and a disconnected
response must retire with an incremented cancellation counter before its row
can be reused. Invalid voices, empty input, unsupported speed/instructions,
and excessive input fail with HTTP diagnostics.

The checker verifies service and lifecycle invariants. The numerical
expressions and upstream text corpus above provide independent operation-level
references. Perceptual model quality requires a separate evaluation corpus.

Full checks retain short PCM responses, the server log, and `result.json` only
after successful shutdown. They trust provisioned parameters to match the
authored acquisition lock; the [developer workflow](../README.md) explains
their cache identity and invocation. No learned-model checkpoint is a test
input or retained output, and valid asynchronous work has no per-request
wall-clock deadline.

Regenerate the normalization literals from the pinned upstream implementation:

```sh
uv run model/kyutai/pocket-tts/test/generate_text_oracle.py
python dev.py format model/kyutai/pocket-tts/test/text_normalization.loom
```

The generator requires Pocket TTS 3.3.0 and Unicode 15.0.0, imports only prompt
preparation, and does not load model weights.
