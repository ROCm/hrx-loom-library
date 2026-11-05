# Sequence carrier checks

These checks call the public carriers used by model kernels. Independent VM
references interpret ordinary/media records and retained-state coordinates;
complete storage comparisons keep unrelated fields, padding, and input metadata
observable.

The important distinction is between identity and position: reordering a cohort
must not change resident ownership, causal/source coordinates, or physical page
selection. Each test's overview describes its oracle; scenarios own boundary
coverage. The [development guide](../../../../docs/development.md#build-and-check)
covers execution and instrumentation.

The page-address check runs the production lookup on the GPU and compares
every token coordinate and wide byte offset with independent I64 arithmetic.
It crosses the 16-bit page boundary, 128M token positions, and the last complete
native-index page without allocating a cache payload of those sizes. Model
attention checks separately exercise cache reads and writes through this lookup.
