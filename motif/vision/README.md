# Vision algorithms

Vision motifs own reusable image-encoder arithmetic and storage contracts.
Their callers supply geometry, precision, and buffers; model composition owns
checkpoint names, launches, and output lifetimes.

The [Qwen3.8 family](qwen3_8/README.md) shares its 1152-wide feed-forward
schedule between original F16 and BF16 matrix representations, with F32
accumulation and residuals.
