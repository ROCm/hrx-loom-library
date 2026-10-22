# Moonshine checks and independent reference

`numerics.loom` uses closed-form F64 expressions for shared-weight projection
and frame normalization; it does not reuse the device reduction loop.
`state.loom` checks two resident histories through lookahead, reordered batch
membership, and EOF. Their benchmark declarations name these same bounded
workloads without claiming measured performance.

`transcription_test.py` exercises the native portable-card path through the
real HTTP server. Its server, selected model closure, audio, and reference are
declared Bazel inputs. Source, archive, and mixed cases use the same exact
reference. Full-weight targets are explicitly selected; they are not part of
ordinary wildcard test execution. Server diagnostics are retained in Bazel's
undeclared test outputs, and the outer test runner owns hang detection.

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
