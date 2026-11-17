# Qwen3.8 speculative sequence control

These launch contracts connect target verification, a separate multi-token
predictor, and queued continuation. Dense and Flash-Next models share the
control layout while owning their numerical programs and retained state.
The composition explicitly selects ordinary token IDs or mixed-media records;
the kernels introduce no model-wide configuration or concrete target.

Acceptance determines a committed prefix before replay advances non-page state.
Proposal work owns private candidate history, and continuation compacts only
rows that retain output credit and context capacity. Resident identity, packed
position, source position, and first-result identity remain separate throughout.
The shared [sequence motifs](../../../motif/sequence/qwen3_8/) own record layouts
and reusable continuation arithmetic; declarations here own launch ABIs.

Proposal preparation captures the original target span, input beginning, and
selected-output beginning before draft owners compact. The ordinary advance
consumes token IDs. The confidence-aware advance consumes token/probability
pairs and a caller-selected probability floor: it publishes the current
candidate, then retires low-confidence owners from subsequent drafting. The
last scheduled advance packs shortened target prefixes even if every draft
owner has retired. Neither entry point changes target acceptance or history.

Adjacent tests compare complete storage with independent frontier and
publication oracles, including overlapping compaction. Each file explains its
reference; scenarios own workload combinations. The
[component guide](../../../docs/component-authoring.md) covers the shared build,
test, and benchmark workflow.
