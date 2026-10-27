# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from loom.model import ModelCard
from loom.model.lock import ModelLock

from build_tools.py.hrx_loom_library.commands import model_card
from build_tools.py.hrx_loom_library.context import RepositoryContext, UserError


class ModelCardTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "model/example/model.json"
        self.path.parent.mkdir(parents=True)
        self.document = {
            "schema_version": 1,
            "model": {"id": "example/model", "name": "Model-card fixture"},
            "origins": {"code": {"kind": "url", "url": "https://example.org/"}},
            "programs": {
                "source": {
                    "origin": "@code/",
                    "directory": "model/example/program",
                    "config": {"enabled": True},
                    "inputs": [
                        {
                            "path": "program/start.cxx",
                            "origin": "@code/custom/start.cxx",
                        }
                    ],
                    "libraries": ["motif/old.loom"],
                },
                "unmanaged": {"inputs": ["handwritten.loom"], "libraries": []},
            },
            "services": {"chat": {"kind": "text", "version": 1}},
            "variants": {
                "small": {
                    "services": {
                        "chat": {
                            "programs": ["source", "unmanaged"],
                            "parameters": {"model": {"files": ["weights.safetensors"]}},
                        },
                    },
                },
            },
            "presets": {
                "chat": {"variant": "small", "service": "chat", "programs": "source"},
            },
        }
        self.original = json.dumps(self.document, indent=2) + "\n"
        self.path.write_text(self.original)
        self.lock = self.path.with_suffix(".json.lock")
        self.lock_contents = (
            json.dumps(
                {
                    "schema_version": 2,
                    "scope": "assets",
                    "origins": {},
                    "files": [
                        {
                            "origin": "weights.safetensors",
                            "size": 0,
                            "sha256": hashlib.sha256(b"").hexdigest(),
                        }
                    ],
                }
            )
            + "\n"
        )
        self.lock.write_text(self.lock_contents)
        ModelLock.parse(self.lock_contents).cover(
            ModelCard(self.document).select("chat")
        )
        self.declarations = {
            "inputs": [
                {
                    "path": "program/start.cxx",
                    "format": "cxx",
                    "options": ["cxx:std=c++20"],
                    "data": ["program/include.h"],
                }
            ],
            "libraries": ["motif/new.loom", "template/chat.loom"],
        }
        self.record = {
            "card": "model/example/model.json",
            "target": "//model/example:source",
            "programs": {"source": self.declarations},
        }
        self.context = mock.Mock(spec=RepositoryContext)
        self.context.repository_root = self.root
        self.context.bazel.return_value = json.dumps(self.record) + "\n"

    def run_sync(self, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            model_card.synchronize(self.context, **kwargs)
        return output.getvalue()

    def test_check_reports_exact_diff_without_writing(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            self.assertRaisesRegex(UserError, "model-card --fix"),
        ):
            model_card.synchronize(self.context)
        self.assertIn("--- a/model/example/model.json", output.getvalue())
        self.assertIn('-        "motif/old.loom"', output.getvalue())
        self.assertIn('+        "motif/new.loom",', output.getvalue())
        self.assertEqual(self.path.read_text(), self.original)
        arguments = self.context.bazel.call_args.args[0]
        self.assertEqual(arguments[0], "cquery")
        self.context.bazel.assert_called_once()

    def test_fix_preserves_authored_facts_and_is_idempotent(self):
        self.assertIn("1 updated", self.run_sync(fix=True))
        updated = json.loads(self.path.read_text())
        expected = copy.deepcopy(self.document)
        expected["programs"]["source"].update(copy.deepcopy(self.declarations))
        expected["programs"]["source"]["inputs"][0]["origin"] = "@code/custom/start.cxx"
        self.assertEqual(updated, expected)
        self.assertEqual(self.lock.read_text(), self.lock_contents)
        ModelLock.parse(self.lock_contents).cover(ModelCard(updated).select("chat"))
        modified_time = self.path.stat().st_mtime_ns
        self.assertIn("0 updated", self.run_sync(fix=True))
        self.assertEqual(self.path.stat().st_mtime_ns, modified_time)
        self.assertIn("0 stale", self.run_sync())

    def test_conflicting_preset_compositions_do_not_write(self):
        conflict = copy.deepcopy(self.record)
        conflict["target"] = "//model/example:alternate"
        conflict["programs"]["source"]["libraries"] = []
        self.context.bazel.return_value += json.dumps(conflict) + "\n"
        with self.assertRaisesRegex(UserError, "differs between"):
            self.run_sync(fix=True)
        self.assertEqual(self.path.read_text(), self.original)

    def test_identical_preset_compositions_are_coalesced(self):
        self.context.bazel.return_value *= 2
        self.assertIn("1 checked, 1 program sets, 1 updated", self.run_sync(fix=True))

    def test_undeclared_program_does_not_get_created(self):
        self.record["programs"]["unknown"] = self.declarations
        self.context.bazel.return_value = json.dumps(self.record)
        with self.assertRaisesRegex(UserError, "undeclared program set"):
            self.run_sync(fix=True)
        self.assertEqual(self.path.read_text(), self.original)

    def test_card_paths_cannot_escape_repository(self):
        self.record["card"] = "../model.json"
        self.context.bazel.return_value = json.dumps(self.record)
        with self.assertRaisesRegex(UserError, "repository-relative"):
            self.run_sync(fix=True)

    def test_duplicate_json_keys_are_not_silently_rewritten(self):
        self.path.write_text('{"programs": {}, "programs": {}}')
        with self.assertRaisesRegex(UserError, "Duplicate JSON key"):
            self.run_sync(fix=True)

    def test_staged_check_refuses_unstaged_dependency_declarations(self):
        self.context.run.side_effect = [
            "M\0model/example/model.json\0",
            "M\0kernel/example/BUILD.bazel\0",
            "",
        ]
        with self.assertRaisesRegex(UserError, "kernel/example/BUILD.bazel"):
            model_card._staged_inputs_ready(self.context)
        self.context.bazel.assert_not_called()

    def test_staged_glob_member_addition_requires_analysis(self):
        self.context.run.side_effect = ["A\0program/new.cxx\0", "", ""]
        self.assertTrue(model_card._staged_inputs_ready(self.context))

    def test_untracked_glob_member_prevents_index_worktree_mismatch(self):
        self.context.run.side_effect = [
            "M\0model/example/model.json\0",
            "",
            "program/new.cxx\0",
        ]
        with self.assertRaisesRegex(UserError, "program/new.cxx"):
            model_card._staged_inputs_ready(self.context)

    def test_content_only_source_edit_does_not_reanalyze_file_list(self):
        self.context.run.return_value = "M\0program/prepare.loom\0"
        self.assertFalse(model_card._staged_inputs_ready(self.context))
        self.context.run.assert_called_once()

    def test_non_object_card_is_reported_without_rewrite(self):
        self.path.write_text("[]\n")
        with self.assertRaisesRegex(UserError, "Expected a JSON object"):
            self.run_sync(fix=True)
        self.assertEqual(self.path.read_text(), "[]\n")


if __name__ == "__main__":
    unittest.main()
