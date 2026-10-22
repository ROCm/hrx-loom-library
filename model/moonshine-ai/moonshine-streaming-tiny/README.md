# Moonshine Streaming Tiny

This package expresses Moonshine Streaming Tiny's learned waveform frontend,
incremental encoder, autoregressive decoder, and recognition control in Loom.
The portable [model card](model.json) binds original F32 safetensors and a named
Hugging Face tokenizer to the versioned recognition service. Device placement
is a deployment choice, not a field in this model definition.

The implementation targets AMDGPU. Its workgroup-shaped kernels use four
resident histories and up to four independently advancing packed rows. CPU and
NPU execution require a target port of those workitem-based numerical programs;
a device-independent HTTP fixture does not establish learned execution there.

## Build and run

Generate the complete variant, service, and file reference without fetching
weights or compiling providers:

```sh
bazel build //model/moonshine-ai/moonshine-streaming-tiny:reference
```

The output is `bazel-bin/model/moonshine-ai/moonshine-streaming-tiny/reference.md`.
It includes source, archive, and mixed choices, named parameter scopes, exact
checkpoint/tokenizer links, and local destination names. The authored card is
the source of this reference; the execution checks below own qualification.

From the library repository root, explicitly select a learned deployment:

```sh
bazel build --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny:source

bazel run --config=amdgpu @hrx//loom/src/loom/tools/loom-serve -- \
  --device=amdgpu --workers=4 --listen=127.0.0.1:8080 \
  --model=bazel-bin/model/moonshine-ai/moonshine-streaming-tiny/source/serve.json
```

The root module declares the checkpoint and tokenizer repositories at immutable
Hugging Face revisions and SHA256 digests. Building a selected learned target
may acquire approximately 178 MB of assets. Ordinary source checks do not fetch
the checkpoint. The card's 136-byte rotary asset is materialized from its
declared inline content; no Python inference runtime or separate weight-prep
script runs in the server.

The generated `model.json` is the portable model definition.
The separate `serve.json` selects its configuration and served name.
`source`, `archive`, and `mixed` build the same service with different provider
representations:

| Target | Prepare/control providers | Command/kernel providers |
| --- | --- | --- |
| `source` | Loom text | Loom text |
| `archive` | Loom bytecode | Loom bytecode |
| `mixed` | Loom text | Loom bytecode |

The small weight-preservation policy is declared auxiliary provider data.
Every deployment shares the same external weight/tokenizer inputs; a compiled
provider changes neither parameter contents nor state layout.

Send 16 kHz mono signed little-endian PCM16 to the ordinary endpoint:

```sh
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=moonshine-streaming-tiny \
  -F 'file=@recording.pcm;headers="Content-Type: audio/pcm;rate=16000;channels=1"'
```

The response is `{"text":"..."}`. Encoded WAV/MP3/FLAC inputs require the server's
optional audio codecs. Raw PCM exercises the core path with no codec dependency.
The HTTP upload endpoint collects the complete input before feeding the
continuing recognition service; it is not a live microphone transport.

## Execution and state contract

| Property | Bound |
| --- | --- |
| Service input | Mono F32 at 16 kHz |
| Encoder quantum | 320 samples / 20 ms |
| Resident requests / compatible cohort | 4 / 4 |
| Input length | 655,360 samples / 40.96 seconds |
| Encoder positions | 2,048 |
| Decoder hypothesis | 512 tokens including EOS |
| Provisional hypothesis cadence | 6,400 samples / 400 ms |
| Device history per resident request | 40,320,208 bytes |
| Decoding | English, empty prompt, temperature zero |

Unknown model options and unsupported request options fail explicitly.
These bounds belong to this package, not the generic recognition service.
Original F32 parameters bind under `model`; rotary frequencies bind under
`rotary`; output decoding uses tokenizer `text`.

The frontend normalizes 80-sample frames and applies learned compression,
projection, and two causal stride-two convolutions. Six encoder layers retain
32-position rings; four lookahead layers contribute 240 ms of right context.
EOF flushes those layers while preserving the real source sample clock.
Published encoder memory and cross-attention projections remain immutable to
decoder steps. Each hypothesis starts a greedy prefix over the available memory.

Packed row ordinals identify submitted work, while resident IDs identify
histories. The accepted cohort mapping remains valid through completion,
including cancellation. Typed transcript packets carry one stable utterance ID,
increasing revisions, consumed sample ranges, and a final flag. The checkpoint
does not supply diarization or word alignment, so the package emits neither.

## Checks and provenance

Checkpoint-free numerical checks cover frame normalization, asymmetric
shared-weight projection, and lookahead/EOF with reordered resident identities:

```sh
bazel test --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny/program/... \
  //model/moonshine-ai/moonshine-streaming-tiny/test/...
```

Full learned checks are manual targets so broad test queries do not download
weights or run full transcription. Select all provider representations with:

```sh
bazel test --config=amdgpu \
  //model/moonshine-ai/moonshine-streaming-tiny/test:source_transcription_test \
  //model/moonshine-ai/moonshine-streaming-tiny/test:archive_transcription_test \
  //model/moonshine-ai/moonshine-streaming-tiny/test:mixed_transcription_test
```

Each launches the real server, uploads the same declared independent PCM input,
checks exact reference text, and joins process retirement. No model mock,
ambient checkpoint path, transcript generated by Loom, or timing threshold is
involved. These checks establish correctness, not competitive performance.
The [test provenance](test/README.md) describes the independent eager F32 oracle
and audio attribution. The upstream model license is in
[LICENSE.moonshine](LICENSE.moonshine).
