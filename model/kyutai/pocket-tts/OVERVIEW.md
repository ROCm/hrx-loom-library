## About the model

[Pocket TTS](https://huggingface.co/kyutai/pocket-tts) is Kyutai's English
text-to-speech model. This composition uses the original BF16 checkpoint,
the English tokenizer, a prepared reference-voice waveform bank, and rotary
tables. The intended speech service returns mono audio at 24 kHz in 1,920-sample
blocks, or 80 ms per block. FlowLM generates acoustic latents and Mimi decodes
them to audio.

**Execution status:** this package describes the model and its native provider
closure. The declared programs and the two prepared archives are not included.
It is not yet a runnable Loom model. `loom-fetch` can acquire published files
and supported archive transforms; it does not run Pocket's voice preparation
or synthesize these rotary tables.

## Required assets

The model weights are
[`languages/english/model.safetensors`](https://huggingface.co/kyutai/pocket-tts/resolve/7f364c05d1be3faf2b234cdbe4cbb68a026d4218/languages/english/model.safetensors).
The matching tokenizer comes from
[`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/00eac05ed3d16bdc3f6b5d598874019c34a89214/languages/english/tokenizer.json).
Both repositories declare CC-BY-4.0. This tokenizer source does not change the
base model's identity or replace its voice-conditioning weights.

`parameters.voices` expects a prepared Safetensors archive containing the F32
tensor `waveforms` with shape `[4, 154, 1920]`. Its four rows correspond to:

| Voice | Valid blocks | Reference recording |
| --- | ---: | --- |
| `alba` | 126 | [Alba Mackenna](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/alba-mackenna/casual.wav) |
| `marius` | 126 | [Marius](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/voice-donations/Selfie.wav) |
| `jean` | 123 | [Jean](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/ears/p010/freeform_speech_01_enhanced.wav) |
| `anna` | 154 | [Anna](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/vctk/p228_023_enhanced.wav) |

The waveform preparation follows
[Pocket TTS revision ec569b2](https://github.com/kyutai-labs/pocket-tts/tree/ec569b2dd3fec4a601b391aace185791e92b3785):
downmix and resample to mono 24 kHz, apply its end-on-pause trimming, and
zero-pad to complete 1,920-sample blocks. Rows are then padded to the longest
reference. Each reference retains its own recording provenance; the weights'
license alone does not describe the rights in every voice recording.

`parameters.rotary` expects F32 `rotary.cosine` and `rotary.sine` tensors, each
with shape `[4096, 32]`. Their frequencies and angles preserve the F32 rounding
points of the reference implementation. The card-relative `prepared/` sources
name these expected output archives, not additional files in the upstream
checkpoint repository. A publisher must supply them before publishing a
complete fetchable package.

## Canonical streaming configuration

The `streaming` preset registers the four voices above, a 512-position context,
4,096 rotary positions, 50-token text chunks, and four request and batch rows.
Each registered voice has a stable ID; a request selects that ID. Arbitrary
per-request audio uploads are not part of this contract.

The service validates voice lengths and capacity relationships together.
Generation reserves `ceil(chunk_tokens * 25 / 6) + 25` frames. Context must cover
the voice, text chunk, generation frames, and initial position; rotary tables
must cover both context positions and the expanded voice/generation positions.
Changing the voice bank therefore requires matching metadata and tables, not
just renaming another audio file.

The card's declared numerical ranges include alternative batch and request
capacities. Only the canonical preset is an execution qualification target;
these ranges do not claim that every combination has been measured.
Training and intended use are described in the
[upstream model card](https://huggingface.co/kyutai/pocket-tts).
