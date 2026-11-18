"""Repository policy wrappers around the public Loom build rules."""

load("@hrx//build_tools/bazel:python.bzl", "iree_py_test")
load(
    "@hrx//loom/build_tools/bazel:defs.bzl",
    _loom_cxx_library = "loom_cxx_library",
    _loom_cxx_test = "loom_cxx_test",
    _loom_kernel_library = "loom_kernel_library",
    _loom_library = "loom_library",
    _loom_model = "loom_model",
    _loom_model_documentation = "loom_model_documentation",
    _loom_module = "loom_module",
    _loom_test = "loom_test",
)

_BENCHMARK_MODULE_TAG = "loom-benchmark-module"
_BENCHMARK_MODULE_SUFFIX = "_benchmark_module"
_BENCHMARK_ADMISSION_TAG = "loom-benchmark-admission"
_BENCHMARK_ADMISSION_SUFFIX = "_benchmark_admission"

loom_model_documentation = _loom_model_documentation

# Final linked products compose providers admitted by their owning source rule.
loom_module = _loom_module

def _is_package_or_subpackage(package, root):
    return package == root or package.startswith(root + "/")

def _label_package(label, current_package):
    label_text = str(label)
    if label_text.startswith("@"):
        return None
    if label_text.startswith("//"):
        return label_text[2:].split(":", 1)[0]
    return current_package

def _validate_motif_dependencies(package, deps):
    for dep in deps:
        dep_package = _label_package(dep, package)
        if dep_package != None and any([
            _is_package_or_subpackage(dep_package, layer)
            for layer in ["kernel", "program", "model"]
        ]):
            fail(
                (
                    "//%s is a motif package and cannot depend on launch/composition package %s; " +
                    "move the reusable representation or algorithm into motif/"
                ) % (package, dep),
            )

def _validate_kernel_dependencies(package, deps):
    for dep in deps:
        dep_package = _label_package(dep, package)
        if dep_package == None:
            continue
        if _is_package_or_subpackage(dep_package, "experimental"):
            continue
        if _is_package_or_subpackage(dep_package, "kernel"):
            fail(
                (
                    "//%s is a kernel package and cannot depend on kernel package %s; " +
                    "move reusable code into motif/ and compose launch surfaces above kernel/"
                ) % (package, dep),
            )
        if not _is_package_or_subpackage(dep_package, "motif"):
            fail(
                (
                    "//%s is a kernel package and can depend only on motif/ " +
                    "libraries, got %s"
                ) % (package, dep),
            )

def _validate_program_dependencies(package, deps):
    for dep in deps:
        dep_package = _label_package(dep, package)
        if dep_package != None and _is_package_or_subpackage(dep_package, "model"):
            fail(
                "//%s is a shared program and cannot depend on model package %s; " % (package, dep) +
                "supply model identity and preparation from the model composition",
            )

def _declare_source_policy_test(name, layer, srcs, tags, data = []):
    text_srcs = [src for src in srcs if str(src).endswith(".loom")]
    if not text_srcs:
        fail("%s requires at least one .loom text source" % name)
    iree_py_test(
        name = name + "_source_policy_test",
        srcs = ["//build_tools/bazel:source_policy.py"],
        main = "source_policy.py",
        args = ["--layer=" + layer] + [
            "$(location %s)" % src
            for src in text_srcs
        ] + (["--data"] + ["$(locations %s)" % item for item in data] if data else []),
        data = text_srcs + data,
        tags = tags + ["hostonly"],
        visibility = ["//visibility:private"],
    )

def _module_policy_arguments_impl(ctx):
    arguments = ctx.actions.declare_file(ctx.label.name + ".args")
    # Header-only providers have no module. Resolve the configured output here,
    # where generated sources and filegroups have already supplied their files.
    module = ctx.files.module[0].short_path if ctx.files.module else ""
    ctx.actions.write(arguments, "--module=" + module + "\n")
    return [DefaultInfo(files = depset([arguments]))]

_module_policy_arguments = rule(
    implementation = _module_policy_arguments_impl,
    attrs = {"module": attr.label(mandatory = True, allow_files = True)},
)

def _declare_module_policy_test(name, layer, module, tags, data = []):
    arguments = name + "_source_policy_arguments"
    _module_policy_arguments(
        name = arguments,
        testonly = True,
        module = module,
        visibility = ["//visibility:private"],
    )
    iree_py_test(
        name = name + "_source_policy_test",
        srcs = ["//build_tools/bazel:source_policy.py"],
        main = "source_policy.py",
        args = [
            "--layer=" + layer,
            "@$(location :%s)" % arguments,
            "--format-tool=$(location @loom_sdk//:loom-format)",
        ] + (["--data"] + ["$(locations %s)" % item for item in data] if data else []),
        data = [":" + arguments, module, "@loom_sdk//:loom-format"] + data,
        tags = tags + ["hostonly"],
        visibility = ["//visibility:private"],
    )

def _declare_benchmark_module(name, module, tags):
    native.filegroup(
        name = name + _BENCHMARK_MODULE_SUFFIX,
        srcs = [module],
        tags = tags + [_BENCHMARK_MODULE_TAG],
        testonly = True,
        visibility = ["//visibility:private"],
    )

def _declare_benchmark_admission(name, tags):
    native.filegroup(
        name = name + _BENCHMARK_ADMISSION_SUFFIX,
        srcs = [":" + name],
        tags = tags + [_BENCHMARK_ADMISSION_TAG],
        visibility = ["//visibility:private"],
    )

def _validate_execution_profiles(name, execution_profiles):
    if not execution_profiles:
        fail("%s requires at least one execution profile" % name)
    for profile in execution_profiles:
        if any([argument.startswith("--sanitizer=") for argument in profile.runner_args]):
            fail("%s: select Loom sanitizers with --config=loom-asan, --config=loom-tsan, or --config=loom-ubsan" % name)

def _validate_kernel_qualification(name, targets):
    if not targets:
        fail("%s requires at least one target profile" % name)

def _program_layer(deps):
    package = native.package_name()
    shared = _is_package_or_subpackage(package, "program")
    model = _is_package_or_subpackage(package, "model")
    if not (shared or model) or "test" in package.split("/"):
        fail("program libraries belong in program/ or model/ production packages, got //%s" % package)
    if shared:
        _validate_program_dependencies(package, deps)
    return "program" if shared else "model"

def loom_library(
        name,
        srcs = [],
        deps = [],
        data = [],
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Declares shared or model-owned programs with explicit linker dependencies."""
    layer = _program_layer(deps)
    _loom_library(
        name = name,
        srcs = srcs,
        deps = deps,
        data = data,
        tags = tags,
        visibility = visibility,
        target_compatible_with = target_compatible_with,
    )
    text_sources = [source for source in srcs if str(source).endswith(".loom")]
    if text_sources:
        _declare_source_policy_test(name, layer, text_sources, tags)

def loom_cxx_library(
        name,
        srcs = [],
        hdrs = [],
        deps = [],
        copts = [],
        defines = [],
        local_defines = [],
        includes = [],
        data = [],
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Declares C++ program providers and their exported header interfaces."""
    layer = _program_layer(deps)
    _loom_cxx_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        deps = deps,
        copts = copts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        data = data,
        tags = tags,
        visibility = visibility,
        target_compatible_with = target_compatible_with,
    )
    _declare_module_policy_test(name, layer, ":" + name, tags)

def loom_model(
        name,
        card,
        preset,
        model_name,
        files = {},
        lockfile = None,
        assets = {},
        data = {},
        inputs = [],
        libraries = [],
        form = "sources",
        programs = "",
        partition = "",
        encoding = "bc",
        tags = [],
        visibility = None,
        target_compatible_with = []):
    """Selects a model card and publishes its declared native serving closure."""
    package = native.package_name()
    if not _is_package_or_subpackage(package, "model"):
        fail("loom_model must be declared below model/, got //%s" % package)
    _loom_model(
        name = name,
        card = card,
        lockfile = lockfile,
        preset = preset,
        model_name = model_name,
        files = files,
        assets = assets,
        data = data,
        inputs = inputs,
        libraries = libraries,
        form = form,
        programs = programs,
        partition = partition,
        encoding = encoding,
        tags = tags,
        visibility = visibility,
        target_compatible_with = target_compatible_with,
    )

def loom_motif_library(
        name,
        srcs = [],
        deps = [],
        tags = [],
        visibility = None):
    """Declares a reusable motif package with no launchable kernel surface."""
    package = native.package_name()
    if not _is_package_or_subpackage(package, "motif"):
        fail("loom_motif_library must be declared below motif/, got //%s" % package)
    _validate_motif_dependencies(package, deps)
    _loom_library(
        name = name,
        srcs = srcs,
        deps = deps,
        tags = tags,
        visibility = visibility,
    )
    if srcs:
        _declare_benchmark_admission(name, tags)
        _declare_source_policy_test(name, "motif", srcs, tags)

def loom_kernel_library(
        name,
        srcs,
        deps = [],
        targets = [],
        tags = [],
        visibility = None):
    """Declares a launchable kernel package with enforced layer dependencies."""
    package = native.package_name()
    if not _is_package_or_subpackage(package, "kernel"):
        fail("loom_kernel_library must be declared below kernel/, got //%s" % package)
    _validate_kernel_dependencies(package, deps)
    _validate_kernel_qualification(name, targets)
    _loom_kernel_library(
        name = name,
        srcs = srcs,
        deps = deps,
        targets = targets,
        benchmark_smoke = False,
        tags = tags,
        visibility = visibility,
    )
    _declare_benchmark_admission(name, tags)
    _declare_source_policy_test(name, "kernel", srcs, tags)

def _declare_test(
        name,
        source_rule,
        srcs,
        deps = [],
        data = [],
        args = [],
        config = {},
        workloads = None,
        execution_profiles = [],
        size = "small",
        tags = [],
        visibility = None,
        **source_options):
    package = native.package_name()
    test_suffixes = ["_test." + extension for extension in ["loom", "c", "cc", "cpp", "cxx"]]
    stress_suffixes = ["_stress." + extension for extension in ["loom", "c", "cc", "cpp", "cxx"]]
    if srcs and "test" not in package.split("/") and not any([
        any([str(source).endswith(suffix) for suffix in test_suffixes + stress_suffixes])
        for source in srcs
    ]):
        fail("//%s: adjacent test sources require a _test or _stress source owner" % package)
    if any([any([str(source).endswith(suffix) for suffix in stress_suffixes]) for source in srcs]):
        if "manual" not in tags or "stress" not in tags:
            fail("%s: stress tests require tags = [\"manual\", \"stress\"]" % name)
    if _is_package_or_subpackage(package, "motif"):
        _validate_motif_dependencies(package, deps)
    if _is_package_or_subpackage(package, "program"):
        _validate_program_dependencies(package, deps)
    _validate_execution_profiles(name, execution_profiles)
    source_rule(
        name = name,
        srcs = srcs,
        data = data,
        args = args,
        config = config,
        workloads = workloads,
        deps = deps,
        execution_profiles = execution_profiles,
        benchmark_smoke = False,
        size = size,
        tags = tags,
        visibility = visibility,
        **source_options
    )
    _declare_benchmark_module(
        name,
        ":" + name + "_module",
        tags,
    )
    if source_rule == _loom_cxx_test or not srcs:
        _declare_module_policy_test(name, "test", ":" + name + "_module", tags, data = data)
    else:
        _declare_source_policy_test(name, "test", srcs, tags, data = data)

def loom_test(name, srcs = [], deps = [], **kwargs):
    """Qualifies checks owned by authored Loom sources and direct libraries."""
    _declare_test(name, _loom_test, srcs, deps = deps, **kwargs)

def loom_cxx_test(name, srcs = [], deps = [], copts = [], defines = [], local_defines = [], includes = [], **kwargs):
    """Qualifies C++ checks with the same execution and benchmark policies."""
    _declare_test(
        name,
        _loom_cxx_test,
        srcs,
        deps = deps,
        copts = copts,
        defines = defines,
        local_defines = local_defines,
        includes = includes,
        **kwargs
    )
