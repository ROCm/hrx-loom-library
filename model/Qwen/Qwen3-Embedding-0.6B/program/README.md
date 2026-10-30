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

Request preparation accepts text or explicit token IDs and chooses an output
dimension between one and 1,024. Batch preparation assigns each sequence a
multiple of eight slots in a 128-slot cohort. Sequence identities and final
token positions travel with the packed request, so attention cannot cross a
request boundary and each returned vector comes from its own final token.
The service executes one stage and publishes results only after completion.

The pipeline places four token-row groups across eight columns and four core
rows. Each column streams its packed projection records once per cohort and
fans them out to the local workers. The array owns all memory banks, channel
capacities, endpoints, and fanout; worker providers receive those resources
explicitly. Arrival and departure messages retire each shared-row phase before
its storage is reused. Interior, ingress, and terminal roles retain distinct
barrier ownership. The shared probability/value worker receives its peer-row
normalization responsibility as a constant from the placed strand.

Projection arithmetic uses N8/K8 BFP16EBS8 panels, compensated BF16 operands,
and F32 accumulators. BF16 normalization, rotary transforms, sequence-masked
softmax, and the SiLU lookup table keep their defined rounding boundaries.
`tile_math.cxx` expresses these structured view and vector operations in C++
and retains the two contract-bound AIE2P assembly leaves inline. Generic native
packing and requested-prefix L2 normalization come from the format and tensor
motifs. Model-specific panel shapes and the offsets within each logical
parameter remain in these providers.

The declaration graph in `BUILD.bazel` mirrors these ownership boundaries.
Each provider builds to relocatable bytecode independently; runtime linking
selects the command and service roots. The weight policy is an auxiliary
compiler input because preparation loads it by name. Full checkpoint files
are outside the Bazel build graph.

The numerical qualification compares the native service with frozen outputs
from a 2,304-input retrieval corpus and an independent F32 publisher reference.
The shared packing and normalization motifs also have small native tests with
independent byte and binary64 oracles. These are correctness results; the
fixed layout and weight reuse do not by themselves establish competitive
performance or a complete transient-memory bound.
