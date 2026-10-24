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
Ownership-event checks compare every resident row, including untouched KV
origins. The epoch fixture retains sentinel padding so
writes beyond the submitted prefix are visible. The benchmark declarations
exercise the same cases through the normal runner; they are correctness smokes
rather than performance baselines.
