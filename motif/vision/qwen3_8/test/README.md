# Vision storage checks

The workspace cases exercise the same allocation-free queries used by command
composition and admission forecasting. Literal byte counts cover independent
allocation padding, the exact output-backed residual threshold, both model
widths, and the largest image geometry. Attention offsets cover all supported
resident cohort counts; the four-cohort endpoint must still fit the FFN bank.
These VM checks allocate no model weights or image-sized buffers.

The `ffn` target independently executes the shared F16 and BF16 feed-forward
algorithms without a model provider or checkpoint. Signed sparse products,
analytical normalization statistics, and scalar GELU check activations and
residuals at a full 64-row tile plus a four-row tail, including the 4304-column
reduction tail. Both formats have native and access-instrumented execution and
benchmark declarations. Test-only launch adapters supply the public motif's
geometry; the independent arithmetic in `reference.loom` is never used by the
production implementation.
