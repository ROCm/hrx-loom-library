# GGML Format Motif Tests

These private programs wrap target-neutral GGML carrier functions in deliberate
launch and buffer boundaries. Production modules contain only the motifs in
the parent package; the wrappers, cases, and microbenchmarks remain test-only
and become roots in one linked test module. Production motifs participate only
as dependency libraries, so their own test-only symbols cannot leak into this
test's execution surface.

Direct-carrier cases isolate bit-field interpretation and selector ordering.
Record-level cases additionally cover the byte addressing, complete blocks,
and logical tails required by representative consumers. Physical AMDGPU runs
exercise target lowering and memory access, while the same authored cases can
serve as deterministic reference programs on targets that implement their
vector operations.

`affine_packets.loom` checks the four-code Q4_K and Q5_K carrier readers
against independent scalar byte decoding. All eight group selectors consume
random, ordered, zero, and all-one packets. Full output storage comparisons
include guard bytes and source immutability. The sixteen-code readers compose
these same word-sized operations and retain their record-level coverage.

`affine_buffers.loom` checks the Q4_K and Q5_K buffer accessors against a
scalar byte-by-byte reference with F64 coefficient evaluation. Six complete
blocks span two padded rows with nonzero input and output origins. All
groups, half-groups, and four-code Q4_K packets are checked exactly. Random
and zero/all-one carriers exercise metadata bit planes and code extrema;
generated F16 headers include zero and signed coefficients. Sentinel regions
and unchanged-input comparisons cover the storage boundary.

`q6_buffer.loom` independently reconstructs logical Q6_K elements from
individual bytes and applies signed scales in F64 before F16 publication.
Alternating word alignment, negative and zero factors, every group/packet,
and output sentinels are checked bitwise. `q8_buffer.loom` checks all three
reader widths over multiple physical groups and padded rows, including
signed code extrema, zero and signed metadata, and untouched sum slots for
the narrow word interface.
