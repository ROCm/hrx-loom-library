# Pocket TTS composition

`model.json` preserves the speech provider composition, all preparation-option
bounds, four reference voices, and typed model/voice/rotary parameter scopes.
The tokenizer has its own identity. `OVERVIEW.md` describes the exact upstream
inputs and prepared archive layouts independently of this README.

The native providers and prepared voice/rotary archives are absent. The
description builds without checkpoint acquisition:

```sh
bazel build //model/kyutai/pocket-tts:reference
```

The native port implements `speech_prepare_v1` and `speech_control_v1`, loads
the three parameter scopes, and retains independent request state across the
voice encoder, FlowLM, and Mimi stages. Voice metadata travels as declared
speech options; filenames do not identify semantic roles. `weights.loom` is
an explicit auxiliary preparation input.

Executable admission requires the provider libraries, reproducible prepared
archives, and a lock for their actual published bytes. Checks must cover
independent voices and request rows, chunk boundaries, cancellation, capacity
rejection, joined shutdown, and audio compared with an independent reference.
The card describes those inputs without executing an arbitrary preparation
script during fetch or moving full weights through Bazel.
