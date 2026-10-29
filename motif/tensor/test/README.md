# Tensor correctness and benchmark workloads

These tests exercise the parent motifs through small, caller-shaped launch
adapters. Model checkpoints are unnecessary. Inputs and independent references
are authored in Loom; expected arithmetic uses scalar expressions, exact integer
rounding, or published known answers instead of the device implementation's
fragment layouts and reduction schedule.

Numerical acceptance follows the operation's rounding and approximation
contract. The owning test or reference helper explains its error budget and
oracle independence beside the assertions. Complete storage comparisons keep
input preservation, aliasing, inactive regions, and guards exact even when live
numerical outputs have a tolerance. Scenarios own shapes and directed cases.

Test helpers are private roots of the linked test module and stay out of
production archives. Larger workloads live in explicitly selected stress
sources; benchmark declarations describe measurable work independently of
correctness execution. The [development guide](../../../docs/development.md#build-and-check)
owns test, instrumentation, stress, and benchmark commands.
