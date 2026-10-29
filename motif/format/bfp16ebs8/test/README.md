# Native BFP16EBS8 packing checks

These tests exercise native FIFO packing and residual encoding. An explicit
NPU pipeline owns core execution and local memory; an independent scalar VM
encoder computes the expected records from the same BF16 inputs. Whole-storage
comparisons include caller-owned inputs, identity, output capacity, and guards.

The [development guide](../../../../docs/development.md#build-and-check) covers
the shared test workflow.
