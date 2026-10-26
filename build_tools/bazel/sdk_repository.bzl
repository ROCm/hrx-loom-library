# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Selects installed SDK tools or the pinned/local HRX source toolchain."""

_NATIVE_TOOLS = [
    "loom-benchmark",
    "loom-compile",
    "loom-format",
    "loom-link",
    "loom-serve",
    "loom-test",
]

_TOOLCHAINS = """
loom_tools_toolchains(
    name = "tools",
    benchmark_tool = ":loom-benchmark",
    compile_tool = ":loom-compile",
    format_tool = ":loom-format",
    link_tool = ":loom-link",
    lint_tool = ":loom-lint",
    test_tool = ":loom-test",
)
"""

def _sdk_repository_impl(ctx):
    python = ctx.getenv("LOOM_SDK_PYTHON", "")
    ctx.file(
        "config.bzl",
        "SOURCE_INTEGRATION_COMPATIBLE_WITH = %r\n" % (
            ["@platforms//:incompatible"] if python else [],
        ),
    )
    build = """load("@hrx//loom/build_tools/bazel:loom_toolchain.bzl", "loom_tools_toolchains")
package(default_visibility = ["//visibility:public"])
exports_files(["config.bzl"])
"""
    if python:
        result = ctx.execute([python, ctx.path(ctx.attr._probe)], quiet = True)
        if result.return_code:
            fail(result.stderr)
        sdk = json.decode(result.stdout)

        # Re-evaluate the inventory after pip replaces the installed wheel.
        # RECORD itself is not a tool input: individual file contents own cache
        # identity, independently of the rest of the wheel and its install path.
        ctx.read(sdk["record"])
        for destination, source in sdk["files"].items():
            ctx.symlink(source, destination)
        ctx.symlink(ctx.attr._tools, "sdk_tools.bzl")
        build = "load(\":sdk_tools.bzl\", \"installed_sdk_tools\")\n" + build
        build += "installed_sdk_tools(native_tools = %r, executable_suffix = %r)\n" % (
            _NATIVE_TOOLS,
            sdk["executable_suffix"],
        )
    else:
        aliases = {
            tool: "@hrx//loom/src/loom/tools/" + tool
            for tool in _NATIVE_TOOLS
        }
        aliases.update({
            "loom-lint": "@hrx//loom/py/loom/tools:loom-lint",
            "loom-compile-report": "@hrx//loom/py/loom/tools:loom-compile-report",
        })
        for name, actual in aliases.items():
            build += "alias(name = %r, actual = %r)\n" % (name, actual)
    ctx.file("BUILD.bazel", build + _TOOLCHAINS)

loom_sdk_repository = repository_rule(
    implementation = _sdk_repository_impl,
    attrs = {
        "_probe": attr.label(default = "//build_tools/bazel:sdk_probe.py"),
        "_tools": attr.label(default = "//build_tools/bazel:sdk_tools.bzl"),
    },
)
