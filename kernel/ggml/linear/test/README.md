# Packed projection checks

`reference.loom` owns the independent scalar-byte projection oracles and input
constructors shared by the adjacent kernel suites and cohort checks. Its
functions remain private within each merged test root. The parent package has
source visibility because it owns the same projection contracts; production
motifs and model programs have no reference dependency.

The cohort wrappers reproduce model-owned launch geometry and bind the row-bank
extent before compilation. Named Bazel variants select one, three, four, and
eight rows through the public configuration mechanism. Both 32- and 64-lane
targets execute Q5 raw/channel8, Q6 Q8 with 16/32 lanes per block, and Q6 F32
forms. Seeded scalar-oracle checks cover incomplete block iterations, channel
tails, distinct row data, signed scales, and exact output guards. Nonzero
analytical workloads retain correctness checks at model reduction widths.

These tests qualify numerical behavior and specialization. Their one-iteration
benchmark smoke results do not establish comparative performance.
