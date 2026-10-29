# Flash-Next model checks

Synthetic inputs check model dispatch, retained state, speculative publication,
and fused epilogues without downloading the checkpoint. Shared
[Q8 projection](../../../../kernel/ggml/linear/test/),
[vision](../../../../motif/vision/qwen3_8/test/), and
[media-policy](../../../../program/text/qwen3_8/test/) packages own their reusable
contracts. Model scenarios preserve the distinction between resident identity,
packed rows, original media offsets, and accepted history.

The owning test source explains its oracle and numerical bounds. Whole-storage
comparisons preserve inactive outputs and immutable inputs. Full-vocabulary
stress workloads allocate more than 600 MiB of synthetic weights and remain
explicit manual selections. The
[development guide](../../../../docs/development.md#build-and-check) owns normal
execution, independent Loom instrumentation, and benchmark commands.

Learned service checks below establish formatting, activation, retained-request
ownership, and shutdown through the complete model; numerical fixtures alone
cannot establish those contracts.

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

## Image and video service checks

`serve_source_vision_test` and `serve_archive_vision_test` use original BF16
projector parameters in addition to the text assets. Each has a
`_vision_mtp_test` counterpart for Q8 MTP. The provisioned installation also supplies `parameter/mmproj.gguf` from the card's
asset lock. The same staging and joined-server harness serves all these checks.

`vision_check.py` constructs small, lossless solid-color PNGs with independent
answer expectations. Image controls cover two encoder geometries, four-image
ordering, and ordinary text. The retained history combines a labeled still
image with five irregularly timestamped video frames. Distinct red and blue
pairs plus an odd green tail establish temporal ordering. Two forks check the
final frame and a timestamp while the projector is deactivated and its
committed bytes are zero. The checkpoint ends inside a four-token
sparse-attention pool and retains no host snapshot.

A fresh image reactivates the projector; primary-weight eviction/reactivation
preserves the pinned endpoint without recompilation. Finally, the harness joins
the server, starts a fresh process, and replays the complete mixed-media history
with zero cached positions. Source, archive, target-only, and MTP configurations
use this same manual checker, not separate numerical suites.

Answers are checked for the task result rather than identical capitalization
or whitespace across batch shapes and MTP. The pin seed has a canonical chat
suffix, and a noncanonical leading-space response retains the service's pin
rejection. These checks establish media/service ownership and simple visual
grounding. The shared [encoder](../../../../motif/vision/qwen3_8/test/README.md) and
[media-protocol](../../../../program/text/qwen3_8/test/README.md) packages own
numerical, preprocessing, and admission checks. Model-local retained-position
tests cover sparse-attention and speculative-state boundaries; color answers
alone do not establish broad visual-language task quality.
