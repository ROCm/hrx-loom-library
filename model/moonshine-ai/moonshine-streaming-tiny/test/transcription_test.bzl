# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Real-server Moonshine checks with their declared AMDGPU runtime closure."""

load("@hrx//build_tools/bazel:python.bzl", "iree_py_test")
load("@hrx//build_tools/bazel:requirements.bzl", "apply_test_requirements")
load("@hrx//loom/target/amdgpu:execution_profiles.bzl", "AMDGPU_HARDWARE_PROFILE")

def transcription_test(name, deployment):
    """Checks one provider representation against the independent PCM oracle."""
    iree_py_test(**apply_test_requirements(
        {
            "name": name,
            "size": "large",
            "srcs": ["transcription_test.py"],
            "main": "transcription_test.py",
            "args": [
                "--server=$(location @hrx//loom/src/loom/tools/loom-serve)",
                "--deployment=$(location %s)" % deployment,
                "--audio=$(location short.pcm)",
                "--reference=$(location reference.json)",
                "--device=amdgpu",
            ],
            "data": [
                deployment,
                "@hrx//loom/src/loom/tools/loom-serve",
                "reference.json",
                "short.pcm",
            ],
            "deps": ["@hrx//loom/src/loom/serve/testing:http"],
            "tags": ["manual"],
        },
        build_requirements = AMDGPU_HARDWARE_PROFILE.build_requirements,
        resource_group = AMDGPU_HARDWARE_PROFILE.resource_group,
        run_requirements = AMDGPU_HARDWARE_PROFILE.run_requirements,
    ))
