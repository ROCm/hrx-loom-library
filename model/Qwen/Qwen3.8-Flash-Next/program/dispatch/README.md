# Device dispatch composition

These commands bind the checkpoint's named tensors and sequence launches over
prepared epoch capacities. The enclosing model supplies parameter handles,
immutable span descriptors, retained-state reader/writer origins, and output
buffers. Each command owns its transient intermediates through their final
consumer. It does not allocate another persistent parameter bank.

GDN dispatch chooses the small recurrence or cooperative four-token window
from the prepared capacity. Fused projection/convolution publishes boundary
halos before a separate owner advances history. Both forms consume the same
rounded BF16 convolution values. Capture is below the layer composition so
GDN and PLE can record speculative inputs without depending on the parent
verification program.

Hyperconnection dispatch binds the target's F32 injection weights or the
predictor's Q8_0 injection weights. Expert dispatch selects the checkpoint's
Q4_K, Q5_K, Q5_1, and Q8_0 paths, constructs expert-owned task records, and
combines routed and shared outputs. Projection dispatch supplies exact
dimensions to the Q8 launch policy. Weight preparation is a separate command
whose only mutation is the selected parameter allocation.

Commands in the neural proposal path forward the caller's `neural_enabled`
value through their nested commands and projection templates. It controls
launch geometry and does not become a parameter of the device algorithm.
Callers outside that path supply a constant true value. The proposal owner
can therefore reuse the same source with a device-sampled condition.

`dispatch.loombc` depends on configuration and numerical providers. Model-wide
layer ordering, accepted-prefix replay, and service policy belong to its
callers. The explicit declarations in each source file are its linking
contract; implementations remain in the provider that owns the operation.
