# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Bazel executables for the native and Python files in one installed SDK."""

load("@hrx//build_tools/bazel:python.bzl", "iree_py_binary", "iree_py_library")

def _native_libraries_impl(ctx):
    outputs = []
    for source in ctx.files.srcs:
        output = ctx.actions.declare_file("bin/" + source.basename)
        ctx.actions.symlink(output = output, target_file = source)
        outputs.append(output)
    return [DefaultInfo(files = depset(outputs))]

_native_libraries = rule(
    implementation = _native_libraries_impl,
    attrs = {"srcs": attr.label_list(allow_files = True)},
)

def _native_tool_impl(ctx):
    output = ctx.actions.declare_file("bin/" + ctx.file.src.basename)
    ctx.actions.symlink(
        output = output,
        target_file = ctx.file.src,
        is_executable = True,
    )

    # Action tools need adjacent loader inputs in FilesToRunProvider as well as
    # runfiles. A runfiles tree alone does not put libraries beside the executable
    # passed directly to actions.run, particularly in a remote sandbox.
    return [DefaultInfo(
        executable = output,
        files = depset([output] + ctx.files.libraries),
        runfiles = ctx.runfiles(files = [output] + ctx.files.libraries),
    )]

_native_tool = rule(
    implementation = _native_tool_impl,
    executable = True,
    attrs = {
        "libraries": attr.label(allow_files = True),
        "src": attr.label(allow_single_file = True, mandatory = True),
    },
)

def installed_sdk_tools(native_tools, executable_suffix):
    """Declares SDK files as inputs, retaining the wheel's adjacent libraries."""
    _native_libraries(
        name = "native_libraries",
        srcs = native.glob(
            ["native/*"],
            exclude = ["native/" + tool + executable_suffix for tool in native_tools],
        ),
    )
    for tool in native_tools:
        _native_tool(
            name = tool,
            libraries = ":native_libraries",
            src = "native/" + tool + executable_suffix,
        )
    iree_py_library(
        name = "python",
        srcs = native.glob(["python/**/*.py"]),
        data = native.glob(["python/**/*"], exclude = ["python/**/*.py"]),
        imports = ["python"],
    )
    for name, source in {
        "loom-lint": "source_lint.py",
        "loom-compile-report": "compile_report.py",
    }.items():
        path = "python/loom/tools/" + source
        iree_py_binary(
            name = name,
            srcs = [path],
            main = path,
            deps = [":python"] if name == "loom-compile-report" else [],
        )
