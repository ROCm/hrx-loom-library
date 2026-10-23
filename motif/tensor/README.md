# Tensor sampling primitives

`random.loom` supplies stateless Philox4x32-10 and F32 normal-pair templates.
A caller assigns counter words to request, position, layer, or tensor element
identities and supplies the seed. The motifs retain no mutable generator state
and contain no launch or target configuration. Reordering workgroups therefore
leaves a sample unchanged when its counter and key stay unchanged.

`random.philox4x32` accepts four counter words and two key words as unsigned
32-bit bit patterns. Arithmetic wraps modulo 2^32. Ten rounds produce four
output words with the [Random123 Philox4x32-10](https://github.com/DEShawResearch/random123)
mapping. Counter allocation belongs to the caller; the motif does not prevent
two callers from choosing the same counter and key.

`random.normal_pair` applies Box-Muller to two words. Each word is converted to
F32, offset by half a bin in F32, and scaled by 2^-32. The radial uniform is
clamped below one at `1 - 2^-24`; zero retains the half-bin offset. The angular
uniform is measured in turns, and a rounded value of one wraps naturally to
zero turns. F32 logarithm and trigonometry permit the target's approximate
operations. Normal samples have numerical tolerance rather than a cross-device
bitwise identity; Philox words have exact identity.

The [checks](test/README.md) use published integer answers and independent
double-precision normal values. The same motif participates in native GPU
sampling and model programs without fixing their tensor layout or batch size.

```sh
bazel test --config=asan --config=amdgpu //motif/tensor/...
```
