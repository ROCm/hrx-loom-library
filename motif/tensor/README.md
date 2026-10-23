# Tensor selection and sampling

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

`sampling.loom` consumes finite F32 logits and applies temperature, top-k,
nucleus selection, and a categorical draw entirely on the device. Temperature
zero selects the greatest logit, breaking ties by the lowest vocabulary ID.
Positive temperature uses one Philox counter per token ID and absolute target
position, keyed by the full 64-bit request seed. Packing a request into another
row or changing speculative depth leaves that random stream unchanged.
Floating-point differences can still move a draw across a probability boundary.

The caller supplies a complete 256-thread workgroup, `1..2^24` vocabulary
entries, nonnegative top-k, and top-p in `(0, 1]`. A positive temperature is a
normal F32 value. Top-k zero includes every eligible ID; a larger top-k clamps
to the eligible vocabulary. The exclusion variant removes one ID before
support selection, and `-1` excludes nothing. At least one finite eligible
logit remains. The sampler keeps no probability array or mutable random state;
support selection partitions a strict logit/ID ordering using workgroup
reductions.

`topk.loom` exposes a different result: the ordered values and IDs themselves.
Each 256-thread workgroup retains sixteen values per thread, producing up to
sixteen winners from a 4096-element tile. Its merge consumes disjoint partial
lists with at most 4096 candidates, so it does not reread the full vocabulary.
Equal values choose the lowest unique nonnegative ID. Short tiles fill missing
winners with negative infinity and `INT_MAX`; valid input logits are finite.
Inputs and outputs are separate, and the composition owns both launch geometry
and temporary partial-list storage.

The [checks](test/README.md) use published integer answers, independent
double-precision random transforms, and serial order-statistic references.
These motifs participate in native GPU model programs while their composition
owns tensor layout, launch configuration, and batch membership.

```sh
bazel test --config=asan --config=amdgpu //motif/tensor/...
```
