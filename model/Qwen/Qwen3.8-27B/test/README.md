# Qwen numerical and input checks

Small fixtures exercise the production model providers without acquiring
checkpoints. Each public case or scenario has a benchmark declaration. The
same workload can therefore be checked, planned, and measured with ordinary
Loom tools; benchmark smoke establishes execution, not performance.

| Checks | Contract and reference |
| --- | --- |
| Quantized projections | Independent byte/scalar decoding, exact prepared-layout equivalence, F16 operand rounding, output guards, cohort capacities, signed data, tails, and ordered split reductions. |
| Attention | Independent cache/page fixtures, rotary coordinates, dense and pooled spans, causal history, partition merges, and alternate matrix targets. |
| Gated DeltaNet | Serial four-tap convolution and recurrence, retained forks, span ownership, speculative capture, and accepted-state publication. |
| Selection and inputs | Token/feature gather, lower-ID ties, per-owner sampling streams and permissions, selected-row padding, and MTP carry. Tied-logit sampling uses an analytic maximum-uniform reference rather than the production sampler. |
| DFlash2 | Context attention, proposal packing, candidate selection, and grouped convolution with independently specified buffers and expected state. |
| Vision projector | Minimum, square, and rectangular geometries across patch embedding, attention, feed-forward, normalization, and merger stages. |

`vision_key_value` checks all sixteen heads against two signed scalar taps per
channel, including the last reduction column. A 6×6 patch grid crosses the
32-row tile boundary and exercises both rotary axes, the four-channel head
tails, and transposed value publication. The actual dense F16 launch executes
four cohort commands; BF16 uses one concurrent all-head launch. Guards and
immutable inputs check ownership independently of the numerical comparisons.

`vision_merger` checks the complete up/GELU/down sequence at both the dense
F16/5120 and Flash-Next BF16/2560 feature widths. Seventy rows cross a full tile
and a six-row tail. A scalar two-tap reference varies signs and source columns,
includes the final reduction column, and uses F64 GELU algebra. Assertions
observe rounded activations and final features separately, with guarded output
and immutable input/weight checks. The F16 case uses the actual dense leaves.

`vision_patch_*` checks the cross-product of F16/BF16 compute and original-half/
F32 checkpoint storage. Coordinate-coded RGB8 pixels, signed channel-varying
weights, two temporal matrices, and a last-column tap give an independent
sparse convolution reference. Seventy-two patches cover a full tile and a
tail; a nonzero source origin and output guards exercise the buffer ABI.
F16/half calls the actual dense leaf. Other representations call the same
shared motif with test-owned specialization facts. No complete checkpoint or
offline tensor conversion is needed.

`vision_ffn` additionally checks the shared F16/BF16 algorithm with 68 patch
rows, covering a full 64-row tile, a four-row tail, and the 4304-column down
projection's reduction tail. Each channel has three signed, independently
addressed weight taps; analytically known row statistics and scalar products
provide the reference. The checks observe intermediate activations and final
residuals separately, with bitwise guards and unchanged weight buffers. F16
executes the actual model leaf; a BF16 leaf selects the shared alternate
precision. Both use native and access-instrumented execution profiles.

Rectangular normalization uses a linear spatial field with alternating channel
signs. Its interpolated values and population moments have a closed form, so
the reference does not repeat the GPU interpolation or reduction. Both image
orientations cover fractional positions, boundaries, and 2×2 merge-block order;
residuals, statistics, and normalized output are checked separately.

The full-width Q5 staging comparison uses weights on a `2^-10` lattice and
inputs in `{-1, 0, 1}`. Its `2^-12` absolute budget is one quarter of the
smallest nonzero mathematical result difference; native matrix instructions
may differ in low bits across reduction orders. A separate packed-byte
construction checks both schedules against the independent result `2.5`.
Zero-workload output guards remain bitwise checks.

The ordinary `q6_metadata` target checks code/scale staging, F32/F16 inputs,
fused residuals, token tails, and inactive outputs. Full-width FFN and attention
performance workloads live in the manual `q6_metadata_performance` target. They
remain part of benchmark discovery and can be measured explicitly:

```sh
python dev.py benchmark \
  --target=//model/Qwen/Qwen3.8-27B/test:q6_metadata_performance \
  --config=amdgpu --device=amdgpu://0 \
  --output-dir=.reports/benchmark/q6-metadata
```

The selected SDK or source configuration supplies the tools. Performance runs
use the Release SDK or canonical optimized source configuration and an idle,
qualified measurement device, as described in the
[benchmark workflow](../../../../docs/development.md#sweep-repository-benchmarks).

Test-only reference kernels remain root-owned. Checks of private convolution
or coefficient helpers include the unchanged implementation source as a test
root, preserving the production export boundary. Shared Philox, GGML carrier,
text-policy, geometry, acceptance, proposal, and continuation contracts remain
with their shared library owners.

Native AMD and access-instrumented profiles exercise device behavior. The
portable normalization, feature capture, and preparation checks also retain
Vulkan profiles. Target requirements select the appropriate RDNA/CDNA cases;
a run on one architecture does not qualify another architecture's entries.

## Image input

These tests exercise the image service's input boundary without model weights.
They use the production providers and independently authored expected values.

| Program | Contract |
| --- | --- |
| `vision_pixels` | Exact normalized cubic coefficients, clipped support, negative lobes, identity/reduced/magnified/degenerate axes, and independently rounded edges at a nonintegral 97/31 scale. |
| `vision_control` | One-to-four-image cohort stage arguments, in-place directory construction, image ordering, physical feature ranges, and logical row counts. Only the native stage declaration sink is substituted. |
| `vision_resize` | Complete RGB8 output from the production GPU filter. Thirty-two trials cover landscape/portrait orientation, RGB/RGBA input, identity filtering, and extreme scale changes. Analytical coordinate colors expose orientation mistakes; constant colors isolate filtering and alpha composition. |

Each workload has a `check.benchmark` declaration. The coefficient and cohort
checks run on the VM reference profile. Device filtering runs on native AMD GPU
and access-instrumented profiles. Benchmark smoke verifies executable workload
plans; performance comparisons require the normal exclusive measurement route.

Coefficient tests include their implementation as a test root to inspect its
private axis-plan helper. Cohort tests include the unchanged control source
with a test stage-declaration provider that observes its arguments. Production
libraries retain private helpers and link the actual stage provider.

```sh
bazel test //model/Qwen/Qwen3.8-27B/test:all
```

The shared chat geometry contract remains in
[`program/text/qwen3_8/test`](../../../../program/text/qwen3_8/test/README.md).
These component checks qualify preprocessing and feature-plan construction;
learned text and projector outputs require the complete model composition.

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
