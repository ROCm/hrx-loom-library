# Pocket TTS

[Pocket TTS](https://huggingface.co/kyutai/pocket-tts) turns English text into
streaming speech. This Loom package implements its learned voice encoder,
FlowLM acoustic generator, and Mimi waveform decoder. The native service
returns mono PCM16 audio at 24 kHz in 1,920-sample model blocks (80 ms).
It uses Kyutai's original BF16 checkpoint; inference runs in Loom on an AMD GPU
through libamdf, with no Python model runtime or ROCm SDK.

## Fetch and speak

With the matching `hrx-loom[hf]` SDK installed, acquire the default voices:

```sh
loom-fetch \
  https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/kyutai/pocket-tts/model.json \
  --preset=streaming --output=./pocket-tts

loom-serve --model=./pocket-tts/model.json --device=amdgpu2 \
  --workers=4 --listen=127.0.0.1:8080
```

A library checkout can use `model/kyutai/pocket-tts/model.json` as the first
argument instead. Both workflows use the published source programs directly;
users need no Bazel build or asset-preparation environment. The default package
has about 224 MB of parameters, reference waveforms, rotary data, and tokenizer,
plus the small program files. Fetch reuses cached files and verifies the lock.
If Hugging Face requires approval, fetch identifies the model page and rechecks
access after approval. The
[SDK installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
provides Linux and Windows prerequisites and optional HF setup.

In another terminal, save a spoken sentence:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kyutai/pocket-tts","input":"Good day.","voice":"alba","seed":42,"response_format":"pcm"}' \
  --output=speech.pcm
```

The output is raw signed little-endian PCM16, without a WAV header. A player
needs its sample rate and channel count; for example:

```sh
ffplay -nodisp -autoexit -f s16le -sample_rate 24000 -ch_layout mono speech.pcm
```

An HTTP client can consume audio as it arrives. HTTP reads can split or combine
model blocks, and closing the response cancels that request. A slow or cancelled
request retains its own state rather than advancing another request's history.

## Voice variants and provenance

The `streaming` preset selects `english-three-voices`: Alba, Marius, and Anna.
`streaming-noncommercial` selects `english-four-voices`, which adds Jean from
the noncommercial EARS collection. Acquire that preset into its own directory
and use the same serving command with its card.

All reference recordings come from the
[pinned Kyutai voice collection](https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/README.md):

| Voice ID | Prepared blocks | Original reference | Recording license |
| --- | ---: | --- | --- |
| `alba` | 126 | [Alba MacKenna, Casual](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/alba-mackenna/casual.wav) | CC BY 4.0 |
| `marius` | 126 | [Unmute voice donation, Selfie](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/voice-donations/Selfie.wav) | CC0 |
| `anna` | 154 | [VCTK speaker p228](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/vctk/p228_023_enhanced.wav) | CC BY 4.0 |
| `jean` | 123 | [EARS speaker p010](https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/ears/p010/freeform_speech_01_enhanced.wav) | CC BY-NC 4.0 |

The published banks downmix and resample these references to mono 24 kHz,
apply the pinned upstream end-on-pause policy, and zero-pad complete blocks.
Rows are padded to 154 blocks; the preset retains each voice's valid extent.
Their archive metadata preserves recording paths and source digests.

The original learned weights are
[`languages/english/model.safetensors`](https://huggingface.co/kyutai/pocket-tts/resolve/7f364c05d1be3faf2b234cdbe4cbb68a026d4218/languages/english/model.safetensors).
The English tokenizer comes from
[`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/00eac05ed3d16bdc3f6b5d598874019c34a89214/languages/english/tokenizer.json).
The model card records the checkpoint's CC-BY-4.0 license separately from the
voice-recording licenses. The architecture and preparation policy follow
[Pocket TTS revision ec569b2](https://github.com/kyutai-labs/pocket-tts/tree/ec569b2dd3fec4a601b391aace185791e92b3785).

## Request behavior

A request names a registered voice. Its first use encodes the reference
waveform and retains an immutable conditioning prefix; later requests share
that prefix while keeping independent decoder histories and sampling counters.
Reference audio is registered when preparing the model, rather than uploaded
with each speech request.

A fixed seed replays on the same target, including when concurrent request
membership changes. Sampling includes the complete 64-bit seed, generation
position, and text-chunk index. Equal seeds do not promise identical waveforms
across different implementations or device arithmetic.

Text preparation follows the English model's Unicode normalization,
punctuation, decimal-aware sentence grouping, and tokenizer-boundary rules.
The canonical preset admits at most 4,096 input tokens and 50 tokens per text
chunk. Every chunk is validated before device execution; an indivisible
oversized sentence fails explicitly. Each chunk restores the voice prefix and
starts fresh codec history while keeping the published audio stream continuous.
The service accepts speed `1` and no instruction text. Unknown voice IDs,
unsupported options, and excessive input return request errors.

The preset provides four resident request rows and four-row acoustic batches,
a 512-position FlowLM context, and 4,096 rotary positions. Geometry is validated
together: context covers the voice, text, and bounded generation allowance;
rotary data covers expanded codec positions. A different voice bank needs
matching metadata and capacities. The standalone generated reference below
lists exact parameter files, provider choices, option bounds, and presets.
