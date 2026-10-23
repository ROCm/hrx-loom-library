# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from build_tools.bazel import source_policy


class SourcePolicyTest(unittest.TestCase):
    def test_motif_accepts_functions_and_templates(self):
        violations = source_policy.check_sources(
            "motif",
            [
                ("format.loom", "func.def @load() {\n  func.return\n}\n"),
                (
                    "quantize.loom",
                    "template.decl @quantize.block()\n"
                    "template.def<@quantize.block> device @body() {\n"
                    "  template.return\n"
                    "}\n",
                ),
            ],
        )
        self.assertEqual(violations, [])

    def test_motif_rejects_kernel_definition(self):
        violations = source_policy.check_sources(
            "motif",
            [("bad.loom", "kernel.def @bad() {\n} launch() {\n  kernel.return\n}\n")],
        )
        self.assertEqual(
            violations,
            ["bad.loom: motif sources cannot declare kernel.def"],
        )

    def test_motif_rejects_configuration_operations(self):
        for operation, source in (
            ("config.decl", "config.decl @workgroup_size : index\n"),
            ("config.def", "config.def @workgroup_size = 64 : index\n"),
            (
                "config.get",
                "func.def @workgroup_size() -> (index) {\n"
                "  %size = config.get @workgroup_size : index\n"
                "  func.return %size : index\n}\n",
            ),
        ):
            with self.subTest(operation=operation):
                self.assertEqual(
                    source_policy.check_sources("motif", [("bad.loom", source)]),
                    [
                        f"bad.loom: motif sources cannot use {operation}; "
                        "pass configuration values from a composition leaf"
                    ],
                )
                for layer in ("model", "test"):
                    self.assertEqual(
                        source_policy.check_sources(layer, [("leaf.loom", source)]),
                        [],
                    )

    def test_motif_rejects_concrete_target_records(self):
        for operation, selector in (
            ("aie2p.target", "array"),
            ("amdgpu.target", "gfx1100"),
            ("spirv.target", "vulkan1_3"),
            ("test.target", "low_core"),
            ("vm.target", "core"),
            ("wasm.target", "simd128"),
            ("x86.target", "avx512"),
            ("target.generic", "reference"),
        ):
            with self.subTest(operation=operation):
                source = f"{operation}<{selector}> @selected\n"
                self.assertEqual(
                    source_policy.check_sources("motif", [("bad.loom", source)]),
                    [
                        f"bad.loom: motif sources cannot declare {operation}; "
                        "define concrete targets at a composition leaf"
                    ],
                )
                self.assertEqual(
                    source_policy.check_sources("test", [("leaf.loom", source)]), []
                )

    def test_motif_accepts_external_targets_and_contextual_queries(self):
        source = (
            "target.decl @external_target\n"
            "func.def @subgroup_size() -> (index) {\n"
            "  %size = target.subgroup.size : index\n"
            "  func.return %size : index\n}\n"
        )
        self.assertEqual(
            source_policy.check_sources("motif", [("motif.loom", source)]), []
        )

    def test_operation_names_in_literals_comments_and_symbols_are_not_code(self):
        source = (
            "// kernel.def @commented_out\n"
            "test.decl @example(%values: test.array<i32, "
            'metadata = {label = "config.get // kernel.def\\""}>)\n'
            "func.decl @config.decl()\n"
            "func.decl @amdgpu.target()\n"
            "func.def @call() { func.call @config.decl() : () func.return }\n"
        )
        self.assertEqual(
            source_policy.check_sources("motif", [("motif.loom", source)]), []
        )
        self.assertEqual(
            source_policy.check_sources("kernel", [("empty.loom", source)]),
            ["kernel package has no kernel.def declaration: empty.loom"],
        )

    def test_inline_operations_retain_their_layer_contract(self):
        source = "target.decl @target kernel.def @entry() {} launch() { kernel.return }"
        self.assertEqual(
            source_policy.check_sources("motif", [("bad.loom", source)]),
            ["bad.loom: motif sources cannot declare kernel.def"],
        )
        self.assertEqual(
            source_policy.check_sources("kernel", [("leaf.loom", source)]), []
        )

    def test_test_package_accepts_configuration_declaration(self):
        violations = source_policy.check_sources(
            "test",
            [("wrapper.loom", "config.decl @case_size : %value: index\n")],
        )
        self.assertEqual(violations, [])

    def test_test_package_accepts_private_kernel_definition(self):
        violations = source_policy.check_sources(
            "test",
            [
                (
                    "wrapper.loom",
                    "kernel.def @wrapper() {\n} launch() {\n  kernel.return\n}\n",
                )
            ],
        )
        self.assertEqual(violations, [])

    def test_kernel_requires_definition_across_package_sources(self):
        violations = source_policy.check_sources(
            "kernel",
            [
                ("helpers.loom", "func.def @helper() {\n  func.return\n}\n"),
                ("checks.loom", "check.case public @case {\n  check.return\n}\n"),
            ],
        )
        self.assertEqual(
            violations,
            ["kernel package has no kernel.def declaration: helpers.loom, checks.loom"],
        )

    def test_comments_do_not_satisfy_kernel_requirement(self):
        violations = source_policy.check_sources(
            "kernel",
            [("comment.loom", "// kernel.def @not_code()\nfunc.return\n")],
        )
        self.assertEqual(
            violations,
            ["kernel package has no kernel.def declaration: comment.loom"],
        )

    def test_repository_accepts_policy_wrapped_packages(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            motif_root = repository_root / "motif" / "format" / "weird"
            motif_root.mkdir(parents=True)
            for readme_directory in (
                repository_root / "motif",
                repository_root / "motif" / "format",
                motif_root,
            ):
                (readme_directory / "README.md").write_text("# Scope\n")
            (motif_root / "q3.loom").write_text(
                "func.def @load() {\n  func.return\n}\n"
            )
            (motif_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_motif_library")\n'
                'loom_motif_library(name = "q3", srcs = ["q3.loom"])\n'
            )
            self.assertEqual(source_policy.check_repository(repository_root), [])

    def test_repository_accepts_explicit_motif_test_package(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            test_root = repository_root / "motif" / "format" / "weird" / "test"
            test_root.mkdir(parents=True)
            for readme_directory in (
                repository_root / "motif",
                repository_root / "motif" / "format",
                repository_root / "motif" / "format" / "weird",
                test_root,
            ):
                (readme_directory / "README.md").write_text("# Scope\n")
            (test_root / "q3.loom").write_text(
                "kernel.def @q3() {\n} launch() {\n  kernel.return\n}\n"
            )
            (test_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_test")\n'
                'loom_test(name = "q3", srcs = ["q3.loom"])\n'
            )
            self.assertEqual(source_policy.check_repository(repository_root), [])

    def test_repository_leaves_experimental_sources_to_public_rules(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            experimental_root = repository_root / "experimental" / "model"
            experimental_root.mkdir(parents=True)
            (experimental_root / "program.loom").write_text(
                "config.decl @layer_count : %value: index\n"
                "kernel.def @experimental() {\n"
                "} launch() {\n"
                "  kernel.return\n"
                "}\n"
            )
            (experimental_root / "BUILD.bazel").write_text(
                'load("@hrx//loom/build_tools/bazel:defs.bzl", '
                '"loom_library")\n'
                'loom_library(name = "program", srcs = ["program.loom"])\n'
            )

            self.assertEqual(source_policy.check_repository(repository_root), [])

    def test_repository_test_package_requires_test_library_rule(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            test_root = repository_root / "motif" / "format" / "weird" / "test"
            test_root.mkdir(parents=True)
            for readme_directory in (
                repository_root / "motif",
                repository_root / "motif" / "format",
                repository_root / "motif" / "format" / "weird",
                test_root,
            ):
                (readme_directory / "README.md").write_text("# Scope\n")
            (test_root / "q3.loom").write_text(
                "kernel.def @q3() {\n} launch() {\n  kernel.return\n}\n"
            )
            (test_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_motif_library")\n'
                'loom_motif_library(name = "q3", srcs = ["q3.loom"])\n'
            )
            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "motif/format/weird/test/BUILD.bazel: test packages must "
                    "declare loom_test",
                ],
            )

    def test_repository_rejects_policy_bypass_and_implicit_source(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            kernel_root = repository_root / "kernel" / "ggml" / "quantize"
            kernel_root.mkdir(parents=True)
            for readme_directory in (
                repository_root / "kernel",
                repository_root / "kernel" / "ggml",
                kernel_root,
            ):
                (readme_directory / "README.md").write_text("# Scope\n")
            (kernel_root / "q3.loom").write_text(
                "kernel.def @q3() {\n} launch() {\n  kernel.return\n}\n"
            )
            (kernel_root / "BUILD.bazel").write_text(
                'load("@hrx//loom/build_tools/bazel:defs.bzl", '
                '"loom_kernel_library")\n'
                'loom_kernel_library(name = "q3", srcs = glob(["*.loom"]))\n'
            )
            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "kernel/ggml/quantize/BUILD.bazel: load repository rules "
                    "from //build_tools/bazel:defs.bzl",
                    "kernel/ggml/quantize/BUILD.bazel: library packages "
                    "cannot bypass repository policy wrappers",
                    "kernel/ggml/quantize/BUILD.bazel: q3.loom is not explicitly "
                    "declared",
                ],
            )

    def test_repository_requires_scoped_readme_chain(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            motif_root = repository_root / "motif" / "format" / "weird"
            motif_root.mkdir(parents=True)
            (motif_root / "q3.loom").write_text(
                "func.def @load() {\n  func.return\n}\n"
            )
            (motif_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_motif_library")\n'
                'loom_motif_library(name = "q3", srcs = ["q3.loom"])\n'
            )
            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "motif/format/weird: missing scoped README.md",
                    "motif/format: missing scoped README.md",
                    "motif: missing scoped README.md",
                ],
            )

    def test_model_accepts_host_commands_kernels_and_configuration(self):
        self.assertEqual(
            source_policy.check_sources(
                "model",
                [
                    (
                        "program.loom",
                        "config.decl @layers : %value: index\n"
                        "func.def public @prepare() { func.return }\n"
                        "command.program.def public @decode() launch() { command.return }\n"
                        "kernel.def @project() {} launch() { kernel.return }\n",
                    )
                ],
            ),
            [],
        )

    def test_repository_accepts_model_card_programs_and_test_leaf(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            model_root = repository_root / "model" / "publisher" / "recognizer"
            program_root = model_root / "program"
            test_root = model_root / "test"
            program_root.mkdir(parents=True)
            test_root.mkdir()
            for directory in (
                repository_root / "model",
                repository_root / "model" / "publisher",
                model_root,
                program_root,
                test_root,
            ):
                (directory / "README.md").write_text("# Scope\n")
            (model_root / "model.json").write_text("{}\n")
            (model_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_model")\n'
                'loom_model(name = "source", card = "model.json")\n'
            )
            (program_root / "control.loom").write_text(
                "func.def public @prepare() { func.return }\n"
            )
            (program_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_library")\n'
                'loom_library(name = "control", srcs = ["control.loom"])\n'
            )
            (test_root / "cases.loom").write_text(
                "check.case public @case { check.return }\n"
            )
            (test_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_test")\n'
                'loom_test(name = "cases", srcs = ["cases.loom"])\n'
            )
            self.assertEqual(source_policy.check_repository(repository_root), [])

            (program_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_model")\n'
                'exports_files(["control.loom"])\n'
            )
            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "model/publisher/recognizer/program/BUILD.bazel: "
                    "model packages must declare loom_library"
                ],
            )

    def test_repository_model_card_requires_publication_rule(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            model_root = repository_root / "model" / "publisher" / "recognizer"
            model_root.mkdir(parents=True)
            (model_root / "model.json").write_text("{}\n")
            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "model/publisher/recognizer: model card requires adjacent BUILD.bazel"
                ],
            )
            (model_root / "BUILD.bazel").write_text(
                'load("@hrx//loom/build_tools/bazel:defs.bzl", "loom_model")\n'
                'exports_files(["model.json"])\n'
            )
            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "model/publisher/recognizer/BUILD.bazel: model cards use "
                    "repository rules from //build_tools/bazel:defs.bzl",
                    "model/publisher/recognizer/BUILD.bazel: model card must be "
                    "explicitly declared by loom_model or loom_model_documentation",
                ],
            )

    def test_repository_accepts_description_before_executable_port(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            model_root = repository_root / "model" / "publisher" / "image"
            model_root.mkdir(parents=True)
            (model_root / "model.json").write_text("{}\n")
            (model_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_model_documentation")\n'
                'loom_model_documentation(name = "reference", card = "model.json")\n'
            )
            self.assertEqual(source_policy.check_repository(repository_root), [])

            program_root = model_root / "program"
            program_root.mkdir()
            for directory in (
                repository_root / "model",
                model_root.parent,
                model_root,
                program_root,
            ):
                (directory / "README.md").write_text("# Scope\n")
            (program_root / "control.loom").write_text(
                "func.def public @image_control_v1() { func.return }\n"
            )
            (program_root / "BUILD.bazel").write_text(
                'load("//build_tools/bazel:defs.bzl", "loom_library")\n'
                'loom_library(name = "control", srcs = ["control.loom"])\n'
            )
            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "model/publisher/image/BUILD.bazel: model programs require "
                    "loom_model; a generated reference does not qualify executable providers"
                ],
            )

    def test_repository_rejects_target_source_without_admission_rule(self):
        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            for layer in ("target",):
                source_root = repository_root / layer / "example"
                source_root.mkdir(parents=True)
                (repository_root / layer / "README.md").write_text("# Scope\n")
                (source_root / "README.md").write_text("# Scope\n")
                (source_root / "program.loom").write_text(
                    "func.def @program() {\n  func.return\n}\n"
                )

            self.assertEqual(
                source_policy.check_repository(repository_root),
                [
                    "target/example/program.loom: source-bearing target packages "
                    "require a repository admission rule",
                ],
            )


if __name__ == "__main__":
    unittest.main()
