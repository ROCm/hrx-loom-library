# Moonshine checks and independent reference

`numerics.loom` uses closed-form F64 expressions for shared-weight projection
and frame normalization; it does not reuse the device reduction loop.
`state.loom` checks two resident histories through lookahead, reordered batch
membership, and EOF. Their benchmark declarations name these same bounded
workloads without claiming measured performance.

`transcription_check.py` exercises current built model programs through the real
HTTP server. Bazel supplies the card, acquisition lock, providers, checker,
server, test helpers, and small PCM/reference files. The caller supplies runtime
assets through `LOOM_MODEL_ASSETS`; full checkpoints never enter action inputs,
runfiles, or retained test outputs. The three `transcription_*_test` targets use
normal Bazel caching. They are explicit `manual` targets because their runner
needs provisioned assets and an AMDGPU device.
The [developer workflow](../README.md) shows build, fetch, and invocation.

Source and archive cases use the same exact reference. The checker
verifies the input PCM digest, uploads the audio with deliberately odd HTTP
chunk boundaries, and checks the exact transcript. It joins normal server
retirement before publishing success. Bazel test outputs retain server
diagnostics and `result.json` with the model path, device, observed audio digest,
and checked text. A failed request or shutdown leaves diagnostics without a
successful result. The outer
runner or operator owns interruption and hang detection; valid native work has
no arbitrary per-request timer.

The acquisition lock is a trusted cache identity for provisioned weights and
tokenizers. Tests do not verify those bytes. Updating the lock, model card,
program outputs, runtime, or other declared test inputs invalidates the result;
external asset edits without a lock update intentionally do not. The recorded
oracle hashes below describe the independent reference's provenance. They are
not rescanned on every regression run.

## Audio and oracle provenance

`short.pcm` is 61,440 mono PCM16LE samples at 16 kHz (122,880 bytes).
Its SHA256 and exact expected transcript/token IDs are in
[reference.json](reference.json). The text is:

> Good day, streaming speech should remain clear when the listener interrupts.

The audio was synthesized independently with Pocket TTS 3.3.0, seed 20261014,
using Kyutai's original English weights and the
[Casual voice reference by Alba MacKenna](https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/alba-mackenna/casual.wav).
It is synthetic speech, not an authentic recording of the speaker.
The voice reference is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
as recorded in the
[pinned voice catalog](https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/README.md).
Transformations are synthesis, 24-to-16 kHz polyphase resampling, and
ties-to-even rounding to PCM16. The reference waveform itself is not included.

The transcript comes from a separate eager F32
`MoonshineStreamingForConditionalGeneration` decoder in Transformers 5.19.0,
using the exact PCM bytes after conversion to F32 and division by 32768.
An independent incremental encoder fed 320-frame chunks; sequential greedy
decoding ran every 6,400 frames and at EOF. BOS/EOS are retained in recorded
token IDs and skipped in rendered text. The final exact sequence was also
checked against a separate draft-verification path. The ASR result begins
`Good day,`; the synthesis prompt begins `Good day.`. The latter is not used
as the expected transcript.

The compact record preserves model revisions, asset hashes, dependency
versions, the original oracle report/script hashes, and preprocessing hashes.
No performance result is inferred from that CPU oracle or this endpoint test.
