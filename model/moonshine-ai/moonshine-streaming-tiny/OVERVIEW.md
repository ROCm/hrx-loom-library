## About the model

[Moonshine Streaming Tiny](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny)
is an English speech recognition model from Moonshine AI. This Loom implementation
runs its learned audio frontend, incremental encoder, and autoregressive decoder
on AMDGPU. It uses the original F32 checkpoint and tokenizer from the
[pinned upstream snapshot](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny/tree/f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a).
The file tables below identify the exact downloads and their local names.

The `original` variant supports source, bytecode, and mixed program packages.
These choices change the compiler inputs while sharing the same parameters and
recognition behavior. A mixed package keeps preparation and control in Loom text
and uses bytecode for the numerical programs. CPU and NPU execution require a
port of the numerical kernels.

## Acquire and serve

With `loom-fetch` and `loom-serve` installed, a repository checkout can run the
source model directly. The server JIT compiles the authored Loom programs;
Bazel is not required for this workflow:

```sh
git clone https://github.com/ROCm/hrx-loom-library.git
MODEL=hrx-loom-library/model/moonshine-ai/moonshine-streaming-tiny
loom-fetch "$MODEL/model.json" --configuration=streaming --output=./moonshine
cp "$MODEL/serve.json" ./moonshine/serve.json
```

The checked-in lock covers the default `source` program set. Bytecode and mixed
packages are produced by model publishers; using a published package also
requires no user build. The card argument can be a publication URL instead of
a local path. `loom-fetch` downloads only
the selected program and asset files and reuses available cached content.
Hugging Face acquisition uses the optional `hf` SDK extra; install it in the
same Python environment as `loom-fetch` when requested. The resulting
`moonshine/model.json` describes local files and needs no network access to run.

The copied `moonshine/serve.json` is a separate host configuration. For a card
acquired by URL, create that file with the same contents:

```json
{
  "name": "moonshine-streaming-tiny",
  "model": {
    "path": "model.json",
    "configuration": "streaming"
  },
  "definition": {
    "kind": "recognition"
  }
}
```

Start a server with an available AMDGPU device:

```sh
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 \
  --model=./moonshine/serve.json
```

Send 16 kHz mono signed little-endian PCM16 audio:

```sh
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=moonshine-streaming-tiny \
  -F 'file=@recording.pcm;headers="Content-Type: audio/pcm;rate=16000;channels=1"'
```

The response contains the transcript as `{"text":"..."}`. WAV, MP3, and FLAC
uploads require the corresponding optional server codecs. This HTTP endpoint
collects the upload before feeding the incremental recognition service; live
microphone transport is a separate hosting concern.

## Supported workload

The implementation performs greedy English transcription with an empty prompt.
It accepts up to 655,360 input samples (40.96 seconds at 16 kHz), retains four
independent request histories, and bounds each decoder hypothesis to 512 tokens
including EOS. The encoder advances in 320-sample quanta, and hypotheses may
revise as more audio becomes available. Finalization preserves the real input
sample clock. Speaker diarization and word alignment are not provided.

The memory estimate below covers resident prepared weights. Request histories,
workspace, idle allocator caches, tokenizer, compiler, and driver overhead are
additional; an omitted transient estimate means its complete peak is unknown.
Device placement and request admission belong to the host application.

Source, bytecode, and mixed packages have been checked against an independently
generated transcript through the native recognition service. This correctness
check does not establish an accuracy benchmark or a throughput guarantee.
For the model's training data, evaluation, and intended use, see the
[upstream model card](https://huggingface.co/moonshine-ai/moonshine-streaming-tiny).

## Licenses

The upstream weights and tokenizer use the MIT license. Model provenance and
exact checkpoint links are included below so this page remains usable
independently of a repository checkout.
