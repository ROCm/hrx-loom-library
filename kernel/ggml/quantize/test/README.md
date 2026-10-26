# Q8 activation quantization checks

These cases call the production Q8_1 x4 kernel directly. A scalar Python
reference authors exact packed bytes from binary32 division, reciprocal,
scaling, integer rounding away from zero, and binary16 metadata conversion.
Integer summation keeps the reference independent of the GPU reduction tree;
the metadata sum uses quantized codes and the original F32 scale.

Ten small scenarios cover signed zeros, distinct block scales, all output codes
from -127 through 127, midpoint neighbors, non-power-of-two scales, quantized
sums, F16 midpoints/subnormals/overflow, and multiple rows and physical groups.
Complete input and output storage comparisons include word-aligned nonzero
origins and guard regions. The largest input is 4608 bytes. Each scenario has a benchmark
and runs through native, access-sanitized, and race-sanitized execution profiles.

The oracle checks known packed answers and confirms that the corpus detects
incorrect ties, truncation, a shared group scale, early half-scale rounding,
input-sum metadata, half-rounded sum coefficients, and byte/metadata ordering
mutations. Its ordinary Python test also checks that the generated Loom source
matches the oracle. Refreshing it requires only the standard library:

```sh
python3 -B kernel/ggml/quantize/test/generate_q8_1_oracle.py
bazelisk test --config=asan --config=amdgpu //kernel/ggml/quantize/test:all
```

The [kernel](../q8_1_x4_f32.loom) names the pinned upstream quantization
reference and owns the shape/launch contract. These fixtures qualify exact
format behavior; their smoke timings do not establish model throughput.
