# Pocket reference assets

These F32 Safetensors archives are published runtime inputs. `loom-fetch`
verifies their identities through the model lock. Model-package builds copy
the selected bank and rotary table from this checkout so generated cards keep
working after relocation. Provider-only builds and tests that use synthetic
tensors do not read them. The full learned checkpoint stays in its upstream
Hugging Face repository and never enters Bazel's payload graph.

| Archive | Tensor shape | Preset |
| --- | --- | --- |
| `english-three-voices.safetensors` | `waveforms: [3, 154, 1920]` | `streaming`: Alba, Marius, Anna |
| `english-four-voices.safetensors` | `waveforms: [4, 154, 1920]` | `streaming-noncommercial`: Alba, Marius, Jean, Anna |
| `rotary.safetensors` | `rotary.cosine`, `rotary.sine`: each `[4096, 32]` | Both |

Voice rows contain mono samples at 24 kHz, padded to 1,920-sample blocks and
then to the longest reference. Valid lengths belong to each preset's `voices`
options. The archive metadata records the pinned recording paths and source
digests. The preparation tool preserves the upstream downmix, resampling, and
end-on-pause policy; rotary frequencies and angles have explicit F32 rounding.
The archive header has deterministic key ordering and eight-byte alignment.

The [pinned Kyutai voice collection](https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/README.md)
attributes Alba MacKenna's casual recording and VCTK speaker p228 (Anna) under
CC BY 4.0, Marius's donated recording under CC0, and EARS speaker p010 (Jean)
under CC BY-NC 4.0. The four-voice archive includes that noncommercial material.
Exact recording links and their licenses appear in the model card and overview.

From the repository root, reproduce the publications in new directories:

```sh
uv run model/kyutai/pocket-tts/prepare_assets.py --output=/tmp/pocket-three
uv run model/kyutai/pocket-tts/prepare_assets.py --output=/tmp/pocket-four \
  --voice=alba --voice=marius --voice=jean --voice=anna
```

Each output contains `prepared/voices.safetensors`, `prepared/rotary.safetensors`,
and a source model card matching that bank. `--offline` uses only cached
recordings; `uv run --offline` separately forbids fetching Python dependencies.
Copy the three-voice and four-voice banks to their named publication paths and
the shared rotary table to `prepared/rotary.safetensors`, then refresh the
source publication lock. The tool also accepts `--voice=ID=/path/to/audio` for
custom publications; those retain the original checkpoint acquisition contract.
