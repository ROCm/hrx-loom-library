# Qwen model checks

Small synthetic workloads check model-specific composition, retained state,
speculative publication, and preparation without checkpoint acquisition.
Shared [projection](../../../../kernel/ggml/linear/test/),
[vision](../../../../motif/vision/qwen3_8/test/), and
[service-policy](../../../../program/text/qwen3_8/test/) packages own their
reusable numerical and protocol contracts. Model tests add only the boundaries
those packages cannot establish independently.

Test overviews explain reference independence and numerical acceptance;
scenarios and BUILD declarations own workload and target coverage. Large
vocabulary and full-width performance workloads remain explicit stress tests,
with device allocations approaching a GiB. The
[development guide](../../../../docs/development.md#build-and-check) owns their
selection, instrumentation, and benchmark workflow.

## Learned service checks

The manual `serve_source_test` and `serve_archive_test`
targets exercise the complete model using provisioned assets. The `_vision`,
`_target_only`, `_dflash2`, and `_cdna_text` suffixes select the corresponding
canonical preset, such as `serve_archive_vision_test`:

```sh
bazel test --config=asan --config=amdgpu \
  --test_env=LOOM_MODEL_ASSETS=/path/to/provisioned/qwen3.8-27b \
  //model/Qwen/Qwen3.8-27B/test:serve_archive_vision_test
```

The execution host supplies a fetched installation with the selected card's
parameter and tokenizer paths. The checker shares those files into a temporary
model directory with the current built programs. It trusts the acquisition
lock's declared parameter identity without downloading or hashing checkpoints.
The card, lock, checker, frozen responses, and built providers determine Bazel's
cache identity.

The text corpus recalls station codes supplied in the prompt. Exact predecessor
responses establish text, finish reasons, and token accounting. Concurrent forks
share a checkpoint beyond a cache page, continue independently, and agree with
fresh-history replays. The checker also reuses a checkpoint after weight unload
and reload, checks recovery from a disconnected streaming peer, and requires
joined server shutdown. Four canonical checkpoint slots keep explicit pins
independent of execution rows.

Vision checks identify independently generated red, blue, green, and yellow
images, including four rectangular images in order. They unload projector
weights, continue the complete retained image conversation without reactivating
the projector, and then send a fresh image that must reload it. Repeating only
the original prompt is a new computation; retained reuse includes the preceding
answer. These bounded requests establish service and ownership behavior, not
quality or throughput at the preset's maximum context length.
