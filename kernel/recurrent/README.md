# Recurrent kernels

These kernels update retained model state while publishing the values consumed
by later model stages. Their launch ABIs make state layout, fork ownership, and
numerical contracts explicit; request admission and storage allocation remain
with the calling program.

[`qwen3_5/`](qwen3_5/) implements the single-token Gated DeltaNet convolution
transition shared by the Qwen3.5 family.
