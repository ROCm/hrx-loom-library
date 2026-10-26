# Sequence carrier checks

Native GPU workloads call the same public functions used by model kernels,
with independent VM references for complete storage comparisons.
Input cases compare ordinary and media token mutation with direct record
loads/stores, including first/last records of a 32K packet, both sides of the
512-record boundary, untouched fields, all rotary axes, and both lane halves.
Random input storage and complete comparisons retain guard regions and expose
accidental writes.

Descriptor cases check field endpoints and absent selection against arithmetic
encoding. Retained-state cases use explicit expected coordinates for two
nonadjacent rows, an inactive span, fork and post-fork epochs, nonzero layer
origins, and two distinct physical-page permutations. Dense addressing and
page-table strides on both sides of a 64-token boundary are separate results.
Input metadata remains unchanged throughout the state checks.
