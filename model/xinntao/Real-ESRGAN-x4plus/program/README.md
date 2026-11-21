# Real-ESRGAN x4plus image programs

The model implements the image ABI's four CPU callbacks:
`image_prepare_v1`, `image_prepare_request_v1`, `image_prepare_batch_v1`, and
`image_generate_v1`. `prepare.loom` retains the checkpoint source and declares
the default stage. `control.loom` admits each request, passes the prompt
through as the request packet, groups requests by geometry, and executes the
selected stage. `geometry.loom` owns the shared cold/request admission rule and
the cohort tag. The server receives opaque prepared bytes and stage identities
instead of reconstructing model-specific shapes.

The image service is text to image, so the input image travels in the prompt
as base64 of its raw RGB bytes at a quarter of the requested output height and
width. Four base64 characters encode exactly one pixel, so the prompt length
fixes the input size and the device decodes each pixel independently.
`control.loom` checks the prompt length and alphabet on the CPU before any
device work; standard and URL-safe alphabets are both accepted. The seed is
unused. The card's tokenizer exists only because the image service names one;
no program reads token ids.

`request.loom` decodes the packet into the first sixteen-channel block of an
F32 activation and packs the F32 OIHW checkpoint convolutions into the BF16
layout the convolution kernel reads. `generate.loom` runs RRDBNet: the first
convolution, 23 residual-in-residual dense blocks of three five-layer dense
blocks each, the trunk convolution and long skip, two nearest-neighbour ×2
upsampling convolutions, the high-resolution convolution, and the output
convolution, which writes packed RGBA8 with opaque alpha. The native image
service encodes those bytes as PNG.

`convolution.loom` owns the 3×3 convolution kernel every layer uses and its
two matrix bodies, for layers with 32 and with 64 output channels per tile. Activations are F32, blocked by
sixteen channels so that one pixel's block is 64 contiguous bytes. The matrix
bodies round activations to BF16 operands and accumulate in F32 with RDNA3
WMMA. Dense-block concatenation is a channel
offset into a shared activation buffer rather than a copy. The same kernel
fuses the optional nearest-neighbour upsample of its input, leaky ReLU, the
dense-block and RRDB residual scaling, and the trunk skip.

Activations and packed weights live in the reflected invocation workspace,
sized from the stage geometry; the F32 checkpoint stays in shared parameter
storage. `weights.loom` is retained as an auxiliary preparation policy even
when the other providers are bytecode. The checkpoint needs no per-key
conversion, because a parameter transform cannot change dtype; the BF16
packing happens on the device instead. No checkpoint conversion script is
part of the installation.
