# Qwen3.8 vision algorithms

These motifs own the shared 1152-wide vision encoder's arithmetic and storage
contracts. The dense and MoE models use different matrix precision and final
feature widths; checkpoint names, geometry selection, launches, and retained
output ownership belong to their composition leaves.

The feed-forward implementation accepts original F16 or BF16 matrix storage
and accumulates in F32. Precision is a specialization input, so one schedule
serves both representations without checkpoint conversion or runtime type
branches. The source documents exact layouts, rounding boundaries, and launch
requirements. The [dense model leaf](../../../model/Qwen/Qwen3.8-27B/program/kernel/vision_ffn.loom)
provides the F16 wave32 launch. Its [independent checks](../../../model/Qwen/Qwen3.8-27B/test/vision_ffn.loom)
exercise that actual leaf and a test-only BF16 interpretation of the same motif.

Statistics occupy eight bytes per patch. Up projection normalizes while
staging and publishes only the 4304-wide half-precision GELU activation; down
projection updates the F32 residual in place. Input and weight tiles reuse
external reads across multiple rows and output channels. Their matched LDS
permutations and read-ahead schedule belong to the algorithm, not model policy.
