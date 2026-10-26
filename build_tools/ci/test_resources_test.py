# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from build_tools.ci import test_resources


class TestResourcesTest(unittest.TestCase):
    def setUp(self):
        self.query = ET.Element("query")
        self.tags = {
            "host": [],
            "gpu": ["iree-run-requirement=libamdf.resource.amd_gpu"],
            "instrumented_gpu": [
                "iree-run-requirement=libamdf.resource.amd_gpu",
                "loom-execution-profile=amd_gpu_tsan",
            ],
            "npu": ["iree-run-requirement=libamdf.resource.xdna"],
            "gpu_npu": [
                "iree-run-requirement=libamdf.resource.amd_gpu",
                "iree-run-requirement=libamdf.resource.xdna",
            ],
            "peers": [
                "iree-run-requirement=libamdf.resource.amd_gpu",
                "iree-run-requirement=libamdf.resource.amd_gpu_peers",
            ],
            "new_resource": [
                "iree-run-requirement=libamdf.resource.amd_gpu",
                "iree-run-requirement=another.resource",
            ],
            "manual_host": ["manual"],
            "manual_gpu": ["manual", "iree-run-requirement=libamdf.resource.amd_gpu"],
        }
        for name, tags in self.tags.items():
            rule = ET.SubElement(self.query, "rule", name="//test:" + name)
            entries = ET.SubElement(rule, "list", name="tags")
            for tag in tags:
                ET.SubElement(entries, "string", value=tag)

    def selected_tests(self, *available):
        argument = test_resources.test_tag_filters(self.query, set(available))
        self.assertTrue(argument.startswith("--test_tag_filters="))
        filters = argument.split("=", 1)[1].split(",")
        excluded = {tag[1:] for tag in filters if tag.startswith("-")}
        included = {tag for tag in filters if not tag.startswith("-")}
        return {
            name
            for name, tags in self.tags.items()
            if not excluded.intersection(tags)
            and (not included or included.intersection(tags))
        }

    def test_host_selects_only_resource_free_tests(self):
        self.assertEqual(self.selected_tests(), {"host"})

    def test_gpu_includes_instrumentation_but_excludes_missing_resources(self):
        self.assertEqual(
            self.selected_tests("libamdf.resource.amd_gpu"),
            {"gpu", "instrumented_gpu"},
        )

    def test_gpu_npu_includes_combined_and_independent_tests(self):
        self.assertEqual(
            self.selected_tests("libamdf.resource.amd_gpu", "libamdf.resource.xdna"),
            {"gpu", "instrumented_gpu", "npu", "gpu_npu"},
        )

    def test_npu_alone_cannot_run_combined_tests(self):
        self.assertEqual(self.selected_tests("libamdf.resource.xdna"), {"npu"})

    def test_unknown_available_resource_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "absent from the test graph"):
            self.selected_tests("libamdf.resource.typo")

    def test_no_matching_tests_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "No tests match"):
            test_resources.test_tag_filters(ET.Element("query"), set())

    def test_wrong_input_document_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Expected the XML output"):
            test_resources.test_tag_filters(ET.Element("not_a_query"), set())


if __name__ == "__main__":
    unittest.main()
