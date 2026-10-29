# Q8 activation quantization checks

These checks preserve the exact packed Q8_1 x4 format through the production
[kernel](../q8_1_x4_f32.loom). The scalar IR reference uses logical block order
and integer rounding independently of the GPU's lane layout and conversion
instructions. Complete storage comparisons include input preservation and
guards around caller-owned payloads.

The [component authoring guide](../../../../docs/component-authoring.md)
describes the shared test and benchmark workflow.
