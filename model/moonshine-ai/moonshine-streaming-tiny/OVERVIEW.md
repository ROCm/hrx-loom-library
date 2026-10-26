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

Install the `hrx-loom` SDK with its optional Hugging Face provider in a Python
3.12+ virtual environment. The
[installation guide](https://rocm.github.io/hrx-system/loom/getting-started/acquiring-loom/)
covers Linux and Windows prerequisites and SDK installation.

With the SDK installed, fetch this model directly from its publication URL:

```sh
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny/model.json --list
loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny/model.json --preset=streaming --output=./moonshine
```

These URLs pin one library revision. Fetch checks its adjacent acquisition
lock, installs only the selected program and asset files, and shares available
cached weights. The resulting `moonshine/model.json` describes local files and
needs no network access to run. A repository checkout can use its local
`model/moonshine-ai/moonshine-streaming-tiny/model.json` with the same commands.
Neither path requires a Bazel build.

The checked-in lock covers the default `source` program set. The server JIT
compiles those authored Loom programs with the SDK's compiler. Bytecode and
mixed packages are produced separately by model publishers and use the same
fetch-and-serve workflow. The `streaming` preset selects source here; its name
does not imply a file format or a request transport.

The fetched card retains the selected preset, so it can be passed directly to
`loom-serve`. Requests use the card's model ID,
`moonshine-ai/moonshine-streaming-tiny`. A separate serving document is useful
for a custom routing alias, device placement, or execution overrides, but is
not needed for this setup.

The SDK uses libamdf to access AMD GPUs through the operating system device
driver. No ROCm SDK installation is required. Start a server with an available
AMD GPU:

```sh
loom-serve --device=amdgpu --workers=4 --listen=127.0.0.1:8080 \
  --model=./moonshine/model.json
```

Send 16 kHz mono signed little-endian PCM16 audio. For a first check, download
the small reference recording:

```sh
curl --fail --location https://raw.githubusercontent.com/ROCm/hrx-loom-library/4317508bf5adb251a284ea8dc4e5a24a11bd1c17/model/moonshine-ai/moonshine-streaming-tiny/test/short.pcm --output=recording.pcm
curl --fail-with-body -H 'Expect:' \
  http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=moonshine-ai/moonshine-streaming-tiny \
  -F 'file=@recording.pcm;headers="Content-Type: audio/pcm;rate=16000;channels=1"'
```

The reference recording produces
`{"text":"Good day, streaming speech should remain clear when the listener interrupts."}`.
Replace `recording.pcm` with your own PCM recording for another transcription.
For Windows PowerShell, the
[walkthrough](https://rocm.github.io/hrx-system/loom/serving/workflows/fetch-and-serve/#transcribe-a-recording)
provides the equivalent `curl.exe` command with literal multipart-header quoting.
WAV, MP3, and FLAC
uploads require the corresponding optional server codecs. This HTTP endpoint
collects the upload before feeding the incremental recognition service; live
microphone transport is a separate hosting concern.

## Supported workload

The implementation performs greedy English transcription with an empty prompt.
It accepts up to 655,360 input samples (40.96 seconds at 16 kHz), retains four
independent request histories, and bounds each decoder hypothesis to 512 tokens
including EOS. The `streaming` preset limits the pending request queue to four.
The encoder advances in 320-sample quanta, and hypotheses may
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
