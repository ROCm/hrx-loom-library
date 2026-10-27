# Pocket speech programs

The host providers implement speech ABI 2. Preparation opens the `model`
parameter scope once and shares that source across the registered stages. The
`text` tokenizer is an explicit runtime identity.
The weight policy preserves original BF16 and F32 tensors without repacking.

Control owns each request's text packets, phase, chunk, generation position,
and EOS decision. It groups compatible work into a packed cohort while each
request keeps a stable resident row. Initialization, BOS, voice encoding,
conditioning publication, restoration, text prefill, and generation are
separate retired phases. The native service reserves one slot per upload and
publishes its ID after the conditioning copy retires. A synthesis request
captures that completed prefix without sharing mutable decoder state.

Numerical commands compose FlowLM and Mimi with explicit checkpoint keys and
tensor layouts. Activations and outputs use F32 channel/time order. FlowLM and
voice encoding retain F64 reductions; Mimi waveform decoding selects F32
reductions. A generation output row contains one F32 EOS score followed by
1,920 mono samples. The host converts accepted samples to the requested audio
encoding and publishes them after completion.

Batch membership separates transient packed rows from persistent request rows.
Every numerical wrapper receives offsets and strides; zero parameter strides
share immutable weights. Request seeds remain 64-bit and sampling counters
include generation and text-chunk positions, so cohort reordering cannot
change a request's random stream. Unicode and sampling algorithms come from
the shared text and tensor motifs; model geometry remains in these leaves.

The BUILD file names every provider and its direct dependencies. Source and
bytecode packages retain the same service contract and support
runtime linking and specialization. Concrete device targets come from serving
preparation; numerical test leaves supply their own target and geometry.

`audio.loom` owns one-time CPU channel mixing, polyphase Kaiser resampling,
end-on-pause trimming, fade, and block padding. `voice.loom` validates resource
geometry and creates retained upload state from native decoded F32 input.
Only one codec block at a time enters the numerical work packet. `rotary.loom`
generates the immutable conditioning prefix at model initialization; subsequent
regions hold bounded reusable voice prefixes. Neither path loads a prepared
parameter archive or invokes a Python runtime.
