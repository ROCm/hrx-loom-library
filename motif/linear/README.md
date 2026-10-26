# Linear contractions

These motifs contract activation rows with stored weight rows. They own the
loop geometry, accumulation, and reuse of weight packets, while representation
motifs own packed-byte interpretation. Launchable kernels supply workload facts,
grid geometry, and the target execution context.

The GGML family consumes block-quantized weights directly. Its vector schedules
preserve separate decode and four-row paths; the latter shares decoded packets
across all four activation rows. No expanded weight tensor is materialized.
