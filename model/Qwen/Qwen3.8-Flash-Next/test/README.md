# Flash-Next numerical checks

Small synthetic inputs exercise the production providers without downloading
the full checkpoint. The original numerical configurations cover individual
operations, packed epochs, retained state, and speculative publication. Every
public case or scenario has a corresponding benchmark declaration so the same
workload can be planned, executed, and measured through the ordinary Loom tools.

| Checks | Independent expectation |
| --- | --- |
| Embedding and mixed inputs | Exact signed Q8 values, distinct image features, an immutable source packet larger than the compact accepted batch, rejected holes, ordinary recursive proposals, and complete inactive-output comparisons. |
| Q8 projection and preparation | The shared GGML scalar-byte oracle checks model dispatch and in-place preparation; protocol adapters supply capacity and epoch bounds. |
| GDN and fused convolution | Scalar channel traversal and per-value recurrence; a closed-form Q8 projection fixture varies inputs and codes across K and tests tile boundaries, shared anchors, advancing writers, and speculative capture in narrow and paired-head schedules. |
| Hyperconnection and residual fusion | Independent normalization and projection, explicit BF16 publication, and complete inactive-buffer comparisons. |
| Sparse attention | Independent cache addressing, scalar score and attention references, stable selection, long contexts, causal tails, retained forks, and query-cohort visibility. |
| Copied-history proposals | Closed-form repeated-token fixtures cross page and subgroup boundaries, choose the newest equal-length match, reject short or incomplete continuations, and compare complete compacted buffers. Empty, copied, neural, mixed, and sampled cohorts retain distinct resident, input, output, and policy identities. |
| Experts and mixed quantization | Scalar decoding of each admitted block format, independent matrix references, route grouping, padded tasks, and routed/shared output composition. |
| MTP, output, and vocabulary | Independent projection references and explicit expected state for accepted history, compact vocabularies, fallback selection, proposal rejection, and retained carry. |

`retained`, `mtp_state`, and `speculation` run the same independent ordinary-ID
oracles against ordinary and media input carriers. Media metadata differs from
IDs, so a mistaken contiguous-I32 read cannot pass. Accepted ID capture remains
a compact plane, and predictor vocabulary learns only the accepted IDs.

Q8 panel shapes, operand arithmetic, storage layouts, and their benchmarks live
in [`kernel/ggml/linear/test`](../../../../kernel/ggml/linear/test/), independently
of this model. This package retains fused epilogues and model policy coverage.

`proposal_launch` exercises the real Q8 projection template at its wave and
two matrix entry shapes. GPU-published conditions alternate between active
and skipped rounds, including an observed false value followed by an overwrite.
Complete output comparisons distinguish preserved sentinels from the exact
unit-input contraction. The ordinary constant-true call uses the same source.

Run the package through the repository's normal build environment:

```sh
bazel test --config=asan --config=amdgpu \
  //model/Qwen/Qwen3.8-Flash-Next/test/...
```

The hardware execution profile exercises the native AMD HAL. The build
configuration selects instrumentation on the same test targets: `asan` enables
access checks, and `tsan` enables race checks. Omitting those configurations runs
without instrumentation. Correctness tests retain complete output comparisons
and their declared numerical tolerances. Benchmarks run separately through
`dev.py benchmark`.

Learned service qualification is a separate boundary: it uses the full
provisioned checkpoint and a built model representation. Numerical fixtures
cannot establish chat formatting, retained-request ownership, parameter
activation, or joined server shutdown.

## Installed service checks

The `serve_source_test` and `serve_archive_test` targets
run the installed service contract with each built representation. The same
checks have `_mtp`, `_mtp_continuation`, `_long_context`, and `_long_context_mtp` forms, for example
`serve_archive_mtp_test`. These manual targets require a provisioned asset
installation containing the card-relative parameter and tokenizer paths:

```sh
bazel test --config=asan --config=amdgpu \
  --test_env=LOOM_MODEL_ASSETS=/path/to/provisioned/flash-next \
  //model/Qwen/Qwen3.8-Flash-Next/test:serve_archive_mtp_test
```

The execution host owns that installation. The test shares its existing files
into a temporary directory alongside current built providers and trusts the
acquisition lock's parameter identity; it neither downloads nor hashes the
full checkpoint. Changing the card, lock, checker, or providers invalidates the
test's Bazel cache entry.

The expected replies recall three station codes supplied in the prompt.
Independent captures from the original model establish exact text, finish
reasons, and token accounting for every published preset. Concurrent forks
share a retained checkpoint beyond one page, continue independently, and match
fresh-history replays. The test also checks checkpoint reuse after unloading
and reloading weights without recompiling, recovery after a reset streaming
connection, and joined server shutdown. A preset check establishes these
service and allocation contracts at its configured capacities; short requests
do not establish quality or performance at the maximum context length.

The separate `serve_source_long_context_native_test` and
`serve_archive_long_context_mtp_native_test` targets exercise native context
with target-only and speculative execution. Each prefills over 262,000 tokens,
retrieves three facts placed at different depths through retained branches,
and generates to exactly 262,144 total positions with both one- and eight-token
output budgets. A fresh session also checks automatic prefix reuse. The checker
then joins the server, starts it again, and replays the complete history with
zero cached positions. This separates cold replay from retained correctness
without changing production cache policy. These checks use the same externally
provisioned assets and current built providers as the shorter service checks;
they require two complete long prefills and do not measure broad long-context
task quality or isolated throughput.

## Image service checks

`serve_source_vision_test` and `serve_archive_vision_test` use original BF16
projector parameters in addition to the text assets. Each has a
`_vision_mtp_test` counterpart for Q8 MTP. The provisioned installation also supplies `parameter/mmproj.gguf` from the card's
asset lock. The same staging and joined-server harness serves all these checks.

`vision_check.py` constructs small, lossless solid-color PNGs with independent
answer expectations. It checks two encoder geometries, four-image ordering,
ordinary text in the image-enabled service, and two retained forks whose
checkpoint ends inside a four-token sparse-attention pool. The forks recall the
image while the projector is deactivated and its committed bytes are zero.
A fresh image reactivates the projector; primary-weight eviction/reactivation
preserves the pinned endpoint without recompilation. Finally, the harness joins
the server, starts a fresh process, and replays the complete image-bearing
history with zero cached positions.

Answers are checked for the task result rather than identical capitalization
or whitespace across batch shapes and MTP. The pin seed has a canonical chat
suffix, and a noncanonical leading-space response retains the service's pin
rejection. These checks establish image/service ownership and simple visual
grounding. The shared [encoder](../../../../motif/vision/qwen3_8/test/README.md) and
[image-protocol](../../../../program/text/qwen3_8/test/README.md) packages own
numerical, preprocessing, and admission checks. Model-local retained-position
tests cover sparse-attention and speculative-state boundaries; color answers
alone do not establish broad visual-language task quality.
