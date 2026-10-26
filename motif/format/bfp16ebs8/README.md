# BFP16EBS8 operands

BFP16EBS8 stores a shared exponent followed by eight signed eight-bit mantissas.
A native 64-value operand consists of eight such records, occupying 72 bytes.
For an exponent byte `E` and signed mantissa `m`, the represented value is
`m * 2^(E - 133)`. Exponents and mantissas follow the logical input order; file
containers, matrix panel ordering, and FIFO lookahead belong to callers.

The [AIE2P implementation](aie2p.loom) converts complete blocks from BF16 with
round-to-nearest-even mantissas. A positive rounding carry increases the shared
exponent; a negative mantissa of -128 remains representable. Subnormal floating
operands and intermediates flush to zero. The contract requires finite operands
and finite F32 intermediate results, including reconstruction of the high part.

The two public functions produce a leading stream and a residual stream. The
residual is the original value minus the decoded leading value, encoded in the
same format. Its decoder uses an explicitly supplied row-major 8x8 BF16 identity
matrix. The model owns that matrix and its scratch layout.

The composition binds `bfp16ebs8.core`, supplies 64-byte-aligned byte origins in
one local storage allocation, and owns placement and synchronization. Input,
output, and identity regions are disjoint. Input requires 128 bytes per block;
output requires `max(128, align_up(block_count * 72, 64))` accessible bytes for
FIFO initialization and stores. Only the `block_count * 72` payload bytes
change. Zero blocks preserve the complete allocation while retaining the same
output and identity accessibility requirements. These functions allocate no
storage and transfer no data between memory domains.

Native [checks](test/) compare exact bytes with an independent rational encoder
and preserve patterned guards throughout the local allocation. The model's
larger panel layout composes these records; it does not become format metadata.
