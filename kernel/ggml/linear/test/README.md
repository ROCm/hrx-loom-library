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

The CDNA matrix wrapper retains the model's control-buffer row count and
512-thread launch. Raw and channel8 cases cross channel and token tiles,
while partial cohorts provide only live input rows and protect the unused
output capacity bitwise. The independent scalar-byte oracle performs exact
I64 dots and the specified serial F32 affine arithmetic, permitting exact
comparison even when signed contributions cancel. Nonzero analytical
workloads cover K through 32,768 and cohorts through 512. Its requirements
select gfx9-4 hardware; a run that skips those cases is not qualification.

These tests qualify numerical behavior and specialization. Their one-iteration
benchmark smoke results do not establish comparative performance.

`q8_matrix.loom` qualifies the BF16-input Q8 panels directly, without any model
provider or checkpoint. One parameterized suite covers the production tile and
wave geometries, both read-ahead depths, F32 block scaling, and K32/K64 F16
reconstruction. Four-stream variants exercise per-wave stream panels. Six
trials cover empty, single-row, aligned, and ragged cohorts, a strided second
submatrix, and independently packed four-channel scale/code planes. Three K32
blocks force the K64 path to handle a missing final block. Inactive inputs
contain NaNs, and complete output and immutable-input comparisons protect tails.

`q8_reference.loom` decodes original bytes and accumulates in F64, with explicit
operand-rounding selection. It is a public test-only source provider so fused
model tests can reuse the oracle while retaining their own protocol adapters.
The reference never consumes the prepared storage checked by the matrix path.
Native GPU execution checks these bounded cases. Each trial launches four
workgroups over three K32 blocks, with at most 130 live rows and 144 output
channels across all streams. The benchmark smoke uses one iteration per case
without warmup; these timings do not establish throughput.
