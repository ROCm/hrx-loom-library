# Native embedding programs

These providers implement embedding service ABI 1 over one fixed Strix Halo
NPU composition. Preparation binds the card's `model` parameter scope and
`text` tokenizer. The checkpoint is the original publisher BF16 Safetensors
file. The weight policy validates its tensor metadata and captures the row
ranges needed by each of eight projection columns. `packing.loom` uses eight
cores to gather N8/K8 panels and encode them through the shared native BFP
motif. The runtime schedules reads through four reusable 105 MiB staging lanes
and releases those lanes after activation. Converted bytes go directly into
resident device allocations, never an intermediate weight file.

The command passes ten independently placed parameter views to the array:
eight projection columns, normalization/constants, and original embedding rows.
It allocates mutable cohort state separately. The compiler owns placement;
worker code uses offsets only within its explicit parameter view.
`math.loom` computes SiLU/rotary lookup tables once during parameter preparation.
The policy retains those bytes alongside checkpoint ranges, and `constants.loom`
packs them with the learned normalization vectors. `targets.loom` binds the
same NPU target for initialization and inference.

The final layer's gate and up projections use frozen BF16 channel factors to
reduce activation quantization error. During preparation, column `k` of both
weight matrices is multiplied by `D[k]`, and the preceding post-attention
normalization scale by the BF16 reciprocal of `D[k]`. Each product rounds to
BF16 before the weights are packed. The factors are per-channel maximum
absolute activations from five calibration inputs; publisher and retrieval
checks qualify the rounded transformation. The two tables belong to this
checkpoint and add no inference stage or activation transfer.

`input.loom` accepts text or explicit token IDs, validates the publisher's
32,768-token context and vocabulary, and retains the original sequence
directory. Output dimensions range from one to 1,024. `cohort.loom` selects
the next whole prefix fitting 32,768 eight-aligned slots and packs its token
metadata once. Sequence identities, original positions, and final-token offsets
travel with the cohort, so attention cannot cross a sequence boundary and
each returned vector comes from its own final token.

`control.loom` offers bounded work to the finite service, retains its numerical
range across epochs, and gathers completed cohort rows into independent output
storage before releasing that range. The service accumulates ordered output
prefixes until the complete logical request is ready. Cancellation and model
removal retain captured buffers through actual device retirement. This path
uses the ordinary finite scheduler and does no embedding arithmetic on the host.

`array.cxx` places four token-row groups across eight columns and four core
rows. A request record carries the numerical range, panel count, continuation,
and work allowance. The NPU advances through seven dense groups and one
attention phase per layer, then final normalization. `pipeline.serial` joins
each phase's cores and DMA before the next phase uses shared storage. Bounded
invocations return a continuation record; later invocations resume the same
model using stable executable bindings and retained state. Input shape changes
neither the native image nor its root allocations.

Each column retains its current projection group in memory-tile SRAM and
replays it over the cohort's 128-token panels. A complete dense weight pass is
497,172,480 bytes; a yield inside a group requires reloading that group when
execution resumes. Attention streams K/V windows over the selected query
panels, retaining its online normalization and partial context across bounded
work units. Sequence masks keep independent requests separate. Mutable
activations still pass through global memory between model phases.

`array_resources.h` owns core-local storage and channel layouts;
`array_phases.h` composes dense workers through the explicit provider
interfaces in `array_workers.h`; `attention_schedule.h` owns the K/V window
distribution and paired attention workers. The root owns storage spanning
phases. Each child starts fresh ingress FIFO cursors over retained SRAM.
Arrival and departure messages retire each shared-row phase before its storage
is reused. Interior, ingress, and terminal roles retain distinct barrier
ownership. The shared probability/value worker receives its peer-row
normalization responsibility as a constant from the placed strand.

Dense workers receive and pack activations before acquiring their corresponding
weight record. Those storage ranges are disjoint, allowing weight delivery to
overlap activation preparation. Continuing loop edges release the consumed
record; the final edge retains it through preparation, protecting the shared
weight storage until that phase finishes.

Ingress workers start the next raw activation transfer while computing the
current chunk's final projection. The channel releases the previous raw view
after fanout, while packed activations remain in a separate live range. The
next chunk starts after its transfer completes. The final chunk keeps its
weight reservation through the phase's preparation boundary.

Dense projections use N8/K8 BFP16EBS8 panels and F32 accumulators. Attention
uses compensated BF16 operands. BF16 normalization, rotary transforms,
sequence-masked softmax, and the SiLU lookup table keep their defined rounding
boundaries. `tile_math.cxx` expresses dense projections as complete packed-panel
loads, fragment attachments, and matrix products in C++. Each weight block feeds
four independent eight-token accumulators; the compiler owns native FIFO
state, instruction selection, and packet scheduling. The other native
attention and lookup leaves retain their contract-bound AIE2P implementations.
`attention.loom` owns the compensated attention recurrence and
shared callable row operations. Generic
native packing and requested-prefix L2 normalization come from the format and
tensor motifs. Model-specific panel shapes and the offsets within each logical
parameter remain in these providers.

The declaration graph in `BUILD.bazel` mirrors these ownership boundaries.
Each provider builds to relocatable bytecode independently; runtime linking
selects the command and service roots. The weight policy is an auxiliary
compiler input because preparation loads it by name. Full checkpoint files
are outside the Bazel build graph.

The numerical qualification compares native outputs with 2,624 frozen retrieval
inputs from Quora and SciFact and independent F32 publisher references.
Maintained service checks cover full-context and multi-cohort inputs, late peers,
cancellation, and reuse through both source and archive packages.
The shared packing and normalization motifs also have small native tests with
independent byte and binary64 oracles. These are correctness results; the
fixed layout and weight reuse do not by themselves establish competitive
performance or a complete transient-memory bound.
