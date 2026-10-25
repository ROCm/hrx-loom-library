# Flash-Next numerical checks

Small synthetic inputs exercise the production providers without downloading
the full checkpoint. The original numerical configurations cover individual
operations, packed epochs, retained state, and speculative publication. Every
public case or scenario has a corresponding benchmark declaration so the same
workload can be planned, executed, and measured through the ordinary Loom tools.

| Checks | Independent expectation |
| --- | --- |
| Q8 projection and preparation | Scalar F64 decoding from original GGML bytes, including the selected operand-rounding boundary; batch-independent reductions and in-place preparation preserve output and untouched storage. |
| GDN and fused convolution | Scalar channel traversal and per-value recurrence; a closed-form Q8 projection fixture tests tile boundaries, shared anchors, advancing writers, and speculative capture. |
| Hyperconnection and residual fusion | Independent normalization and projection, explicit BF16 publication, and complete inactive-buffer comparisons. |
| Sparse attention | Independent cache addressing, scalar score and attention references, stable selection, long contexts, causal tails, retained forks, and query-cohort visibility. |
| Experts and mixed quantization | Scalar decoding of each admitted block format, independent matrix references, route grouping, padded tasks, and routed/shared output composition. |
| MTP, output, and vocabulary | Independent projection references and explicit expected state for accepted history, compact vocabularies, fallback selection, proposal rejection, and retained carry. |

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

The hardware execution profiles exercise the native AMD HAL. Access checking
and GPU race instrumentation are additional profiles on the configurations
that declare them. Test-runner benchmark rows are one-iteration execution
checks, not performance evidence. The tests retain complete output comparisons
and their declared numerical tolerances.

Learned service qualification is a separate boundary: it uses the full
provisioned checkpoint and a built model representation. Numerical fixtures
cannot establish chat formatting, retained-request ownership, parameter
activation, or joined server shutdown.

## Installed service checks

The `serve_source_test`, `serve_archive_test`, and `serve_mixed_test` targets
run the installed service contract with each built representation. The same
checks have `_mtp`, `_mtp_continuation`, and `_long_context` forms, for example
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
