# Qwen text-program checks

The native request test links the admitted chat and geometry providers with a
separate test identity and invokes HRX's production text caller through a
versioned model card. The bytecode is a declared build output beside the card
in the test's runfiles. A small BPE
vocabulary makes every emitted token independently inspectable. Cases cover
ordinary and thinking history, tool calls and results, request sampling,
stream fragments, ordered images, resize budgets, and rejection before partial
output. No checkpoint or network access is needed.

VM cases check exact output-span boundaries, integer-grid resize coordinates,
complete ordinary token planes, live upload descriptors, ordinary versus
verified continuation progress, and image frontiers across every epoch split.
Mixed-image checks also run the real continuation algorithm and compare its
entire compacted active plane, including splits with overlapping source and
destination records.
Ownership-event checks compare every resident row, including untouched KV
origins. The epoch fixture retains sentinel padding so
writes beyond the submitted prefix are visible. The benchmark declarations
exercise the same cases through the normal runner; they are correctness smokes
rather than performance baselines.

The storage cases run each model's actual allocation evaluator and resize
callback. Growth preserves resident payload addresses and request policies.
Seven packet capacities check exact allocation sizes, alignment, and initial
zeroing extents, including both sides of 2 GiB and the maximum supported token
capacity. Only the small descriptor tables are created; represented model
storage is never materialized. These cases and their VM benchmarks need no
weights or device.

`storage_records.loom` checks the shared record writers directly against literal
I64 tables. Guarded, nonzero-origin subviews cover first/interior/last records,
all four view roles, partial and full zeroing extents, empty attention views,
64-bit byte quantities, reordered stores, and replacement of an existing record.
Separate VM case benchmarks exercise the same writers and exact comparisons.
The largest allocation is 296 bytes; no represented payload is materialized.

`vision_pixels.loom` checks the actual shared axis planner against independent
bicubic coefficients, clipped support, negative lobes, and identity, reduced,
magnified, and degenerate axes. Nonintegral sampling covers independently rounded
opposite edges. The source root exposes its private helper to this test without
making it part of the public linking contract. Model-local image tests exercise
the same packet producer through real cohort callbacks and GPU pixel filtering.
