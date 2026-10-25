"""Original adapter shapes, layout ownership, and mixed-selection coverage."""

load("@hrx//loom/target/amdgpu:execution_profiles.bzl", "AMD_GPU_HARDWARE_PROFILE", "AMD_GPU_TSAN_PROFILE", "amd_gpu_execution_profile")
load("//build_tools/bazel:defs.bzl", "loom_test")

def adapter_tests(matrix_leaf, scalar_leaf):
    # Twelve matrix families accompany the fused twelve-layer scalar provider.
    shapes = [
        (64, 6144),
        (256, 6144),
        (2560, 2560),
        (2560, 6144),
        (2560, 6912),
        (6144, 64),
        (6144, 1536),
        (6144, 6144),
        (6144, 16384),
        (6144, 36864),
        (6912, 2560),
        (16384, 6144),
    ]
    cases = [
        ("adapter", "matrix", {}, False),
        ("adapter_exact", "matrix", {}, True),
        ("adapter_layer_mix", "scalar", {}, False),
        ("adapter_cohort", "cohort", {}, False),
    ] + [
        ("adapter_shape_%d_%d" % (inputs, outputs), "matrix", {
            "rows": 16,
            "inputs": inputs,
            "outputs": outputs,
            "banks": 3,
            "selections": ["odd"],
        }, False)
        for inputs, outputs in shapes
    ]
    for column_major in [False, True]:
        for alias in [False, True]:
            cases.append(("adapter_layout_%s_%s" % ("column" if column_major else "dense", "alias" if alias else "separate"), "matrix", {
                "rows": 80,
                "inputs": 128,
                "outputs": 192,
                "column_major": column_major,
                "alias": alias,
                "duplicates": True,
                "selections": ["empty", "odd", "even", "seven", "duplicate_bank", "cancel_bank", "zero_strength", "fractional"],
            }, False))
    for name, mode, options, exact in cases:
        leaf = scalar_leaf if mode == "scalar" else matrix_leaf
        native.genrule(
            name = name + "_fixture",
            testonly = True,
            srcs = [leaf],
            outs = [name + ".loom"],
            cmd = "$(location :generate_adapter_oracle) --output=$@ --leaf=$(location %s) --mode=%s --options='%s' %s && $(location @hrx//loom/src/loom/tools/loom-format) --in-place $@" % (leaf, mode, json.encode(options), "--exact" if exact else ""),
            tools = [":generate_adapter_oracle", "@hrx//loom/src/loom/tools/loom-format"],
        )
        loom_test(
            name = name,
            srcs = [name + ".loom"],
            deps = ["//motif/tensor:layer_mix" if mode == "scalar" else "//motif/tensor:adapter"],
            execution_profiles = [
                AMD_GPU_HARDWARE_PROFILE,
                amd_gpu_execution_profile(name = "amd_gpu_access", runner_args = ["--sanitizer=access"], tags = ["notsan"]),
                AMD_GPU_TSAN_PROFILE,
            ],
        )
