# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qwen framing and history through the production native text caller."""

import base64
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(sys.argv.pop(1)).resolve()
MODEL_CARD = Path(sys.argv.pop(1)).absolute()
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


class RequestCheckTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        vocabulary = {chr(index): index for index in range(128)}
        special_tokens = (
            ("<bos>", 128),
            ("<eos>", 129),
            ("<|image_pad|>", 131),
            ("<|video_pad|>", 132),
        )
        vocabulary.update({"λ": 130, **dict(special_tokens)})
        tokenizer = {
            "model": {"type": "BPE", "vocab": vocabulary, "merges": []},
            "added_tokens": [
                {
                    "id": index,
                    "content": spelling,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for spelling, index in special_tokens
            ],
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "<eos>", "type_id": 0}},
                ],
                "pair": [],
                "special_tokens": {
                    spelling: {"id": spelling, "ids": [index], "tokens": [spelling]}
                    for spelling, index in (("<bos>", 128), ("<eos>", 129))
                },
            },
        }
        self.tokenizer = self.directory / "tokenizer.json"
        self.tokenizer.write_text(json.dumps(tokenizer), encoding="utf-8", newline="\n")

    def execute(
        self,
        request,
        response=None,
        capacity=4096,
        chunk_size=0,
        defaults=None,
        truncated=False,
    ):
        request_path = self.directory / "request.json"
        request_path.write_text(
            request if isinstance(request, str) else json.dumps(request),
            encoding="utf-8",
            newline="\n",
        )
        defaults_json = (
            defaults if isinstance(defaults, str) else json.dumps(defaults or {})
        )
        command = [
            str(CHECKER),
            f"--model={MODEL_CARD}",
            f"--tokenizer={self.tokenizer}",
            f"--request={request_path}",
            f"--capacity={capacity}",
            f"--response_chunk_size={chunk_size}",
            f"--response_truncated={'true' if truncated else 'false'}",
            f"--options={defaults_json}",
        ]
        if response is not None:
            response_path = self.directory / "response.txt"
            response_path.write_text(response, encoding="utf-8", newline="\n")
            command.append(f"--response={response_path}")
        return subprocess.run(
            command, capture_output=True, encoding="utf-8", check=False
        )

    def success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        expected_tokens = []
        for text in re.split(r"(<\|(?:image|video)_pad\|>)", value["prompt"]):
            if text in ("<|image_pad|>", "<|video_pad|>"):
                expected_tokens.append(131 if text == "<|image_pad|>" else 132)
            else:
                expected_tokens.extend(
                    130 if character == "λ" else ord(character) for character in text
                )
        self.assertEqual(
            value["tokens"],
            expected_tokens,
        )
        return value

    def image(self, color, width=256, height=256):
        pam = (
            f"P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 3\nMAXVAL 255\n"
            "TUPLTYPE RGB\nENDHDR\n"
        ).encode("ascii") + bytes(color) * (width * height)
        url = "data:image/x-portable-arbitrarymap;base64," + base64.b64encode(
            pam
        ).decode("ascii")
        return {"type": "image_url", "image_url": {"url": url}}

    def request(self, model="qwen3.8-policy-test", **options):
        return {
            "model": model,
            "stream": True,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": "Hello λ."}],
            **options,
        }

    def test_production_prompt_and_plain_completion(self):
        request = self.request()
        value = self.success(self.execute(request))
        prompt = "<|im_start|>user\nHello λ.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.assertEqual(value["model"], "qwen3.8-policy-test")
        self.assertEqual(value["prompt"], prompt)
        self.assertNotIn("checkpoint", value)
        value = self.success(self.execute(request, "Hi."))
        self.assertEqual(value["checkpoint"], prompt + "Hi.<|im_end|>\n")
        self.assertEqual(value["output_cursor"], 3)
        self.assertEqual(value["tool_calls"], [])
        self.assertTrue(value["reusable"])

    def test_normalized_output_requires_replay(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        for response in (" Hi.", "Hi.\n", "Ready.\n\n" + call + "\n"):
            for chunk_size in (0, 1, 7):
                with self.subTest(response=response, chunk_size=chunk_size):
                    value = self.success(
                        self.execute(
                            self.request(tools=TOOLS),
                            response,
                            chunk_size=chunk_size,
                        )
                    )
                    self.assertFalse(value["reusable"])
                    self.assertTrue(value["checkpoint"].endswith("<|im_end|>\n"))

    def test_leading_instruction_roles_share_the_model_envelope(self):
        for thinking in (False, True):
            for tools in ([], TOOLS):
                with self.subTest(thinking=thinking, tools=bool(tools)):
                    request = self.request(
                        enable_thinking=thinking,
                        tools=tools,
                        messages=[
                            {
                                "role": "system",
                                "content": "  Follow the project rules.  ",
                            },
                            {"role": "user", "content": "Hello λ."},
                        ],
                    )
                    expected = self.success(self.execute(request))
                    request["messages"][0]["role"] = "developer"
                    actual = self.success(self.execute(request))
                    self.assertEqual(actual["prompt"], expected["prompt"])
                    self.assertEqual(actual["tokens"], expected["tokens"])
                    self.assertEqual(actual["prompt"].count("<|im_start|>system\n"), 1)
                    self.assertIn(
                        "Follow the project rules.<|im_end|>\n", actual["prompt"]
                    )

    def test_instruction_envelope_cannot_be_reintroduced(self):
        for role in ("system", "developer"):
            for preceding_role in ("system", "developer", "user"):
                with self.subTest(role=role, preceding_role=preceding_role):
                    result = self.execute(
                        self.request(
                            messages=[
                                {"role": preceding_role, "content": "First."},
                                {"role": role, "content": "Second."},
                                {"role": "user", "content": "Hello λ."},
                            ]
                        )
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(
                        "unsupported role, role fields, or message order", result.stderr
                    )

    def test_developer_tool_round_trip_with_omitted_thinking(self):
        # Generic clients use developer for a reasoning-capable model even
        # when the current turn leaves reasoning disabled by omission.
        request = {
            "model": "qwen3.8-policy-test",
            "stream": True,
            "stream_options": {"include_usage": True},
            "store": False,
            "max_completion_tokens": 16384,
            "tools": TOOLS,
            "messages": [
                {"role": "developer", "content": "Follow the project rules."},
                {"role": "user", "content": "Read a.txt."},
            ],
        }
        response = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        first = self.success(self.execute(request, response, chunk_size=7))
        self.assertTrue(first["reusable"])
        self.assertEqual(len(first["tool_calls"]), 1)
        call = {
            key: value
            for key, value in first["tool_calls"][0].items()
            if key != "index"
        }
        request["messages"] += [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "Found it."},
        ]
        following = self.success(self.execute(request))
        self.assertTrue(following["prompt"].startswith(first["checkpoint"]))
        self.assertTrue(following["prompt"].endswith("<think>\n\n</think>\n\n"))

    def test_thinking_defaults_off_and_remains_request_selectable(self):
        for tools in ([], TOOLS):
            with self.subTest(tools=bool(tools)):
                request = self.request(tools=tools)
                disabled = self.success(self.execute(request))
                self.assertTrue(disabled["prompt"].endswith("<think>\n\n</think>\n\n"))
                self.assertNotIn(
                    "Reasoning effort is set to xhigh.", disabled["prompt"]
                )

                del request["enable_thinking"]
                default = self.success(self.execute(request))
                self.assertEqual(default["prompt"], disabled["prompt"])
                self.assertEqual(default["tokens"], disabled["tokens"])

                request["enable_thinking"] = True
                enabled = self.success(self.execute(request))
                self.assertTrue(enabled["prompt"].endswith("<think>\n"))
                self.assertIn("Reasoning effort is set to xhigh.", enabled["prompt"])

    def test_images_reject_non_data_urls(self):
        result = self.execute(
            self.request(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "image.png"}}
                        ],
                    }
                ]
            ),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("image_url must be a base64", result.stderr)

    def test_reasoning_efforts_select_exact_checkpoint_prompts(self):
        instructions = {
            "none": "",
            "low": (
                "Reasoning effort is set to low. Keep your thinking brief and focused, "
                "moving directly to the conclusion without unnecessary elaboration."
            ),
            "medium": "",
            "xhigh": (
                "Reasoning effort is set to xhigh. Please think carefully through the task, "
                "validate key assumptions, consider plausible alternatives, and prioritize "
                "correctness, consistency, and clarity in the final answer."
            ),
        }
        for effort, instruction in instructions.items():
            for system in ("", "Follow the project rules."):
                with self.subTest(effort=effort, system=system):
                    request = self.request(reasoning_effort=effort)
                    del request["enable_thinking"]
                    if system:
                        request["messages"].insert(
                            0, {"role": "developer", "content": system}
                        )
                    value = self.success(self.execute(request))
                    system_text = "\n\n".join(
                        text for text in (instruction, system) if text
                    )
                    prefix = (
                        "<|im_start|>system\n" + system_text + "<|im_end|>\n"
                        if system_text
                        else ""
                    )
                    generation = "<|im_start|>assistant\n<think>\n"
                    if effort == "none":
                        generation += "\n</think>\n\n"
                    expected = (
                        prefix + "<|im_start|>user\nHello λ.<|im_end|>\n" + generation
                    )
                    self.assertEqual(value["prompt"], expected)
                    self.assertEqual(
                        bytes(value["request_options"]),
                        struct.pack("<ffiiq", 0, 1, 0, 0, 1),
                    )
                    request["enable_thinking"] = effort != "none"
                    self.assertEqual(
                        self.success(self.execute(request))["prompt"], expected
                    )

    def test_reasoning_aliases_preserve_tools_and_instruction_order(self):
        for alias, canonical in (
            ("minimal", "low"),
            ("high", "xhigh"),
            ("max", "xhigh"),
        ):
            with self.subTest(alias=alias):
                request = self.request(
                    enable_thinking=True, tools=TOOLS, reasoning_effort=canonical
                )
                expected = self.success(self.execute(request))
                request["reasoning_effort"] = alias
                self.assertEqual(self.success(self.execute(request)), expected)
        request = self.request(enable_thinking=True, tools=TOOLS)
        enabled = self.success(self.execute(request))
        request["reasoning_effort"] = "xhigh"
        self.assertEqual(self.success(self.execute(request)), enabled)
        request["reasoning_effort"] = "medium"
        medium = self.success(self.execute(request))
        instruction = enabled["prompt"].split("\n", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(
            medium["prompt"], enabled["prompt"].replace(instruction + "\n\n", "", 1)
        )

    def test_reasoning_options_reject_conflicts_and_invalid_values(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            with self.subTest(effort=effort):
                request = self.request(
                    reasoning_effort=effort, enable_thinking=effort == "none"
                )
                result = self.execute(request)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("agree with enable_thinking", result.stderr)
        for effort in (None, False, 1, [], {}, "", "LOW", "unknown"):
            with self.subTest(effort=effort):
                result = self.execute(self.request(reasoning_effort=effort))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        request = json.dumps(self.request(enable_thinking=True, reasoning_effort="low"))
        expected = self.success(self.execute(request))
        escaped = request.replace('"low"', '"l\\u006fw"')
        self.assertEqual(self.success(self.execute(escaped)), expected)
        duplicate = request.replace('"low"', '"low", "reasoning_effort": "low"')
        result = self.execute(duplicate)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_template_controls_merge_semantics_independent_of_member_order(self):
        selections = (
            {"enable_thinking": False, "reasoning_effort": "none"},
            {"enable_thinking": True, "reasoning_effort": "low"},
            {"enable_thinking": True, "reasoning_effort": "medium"},
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
            {"preserve_thinking": False},
            {"preserve_thinking": True},
        )
        for controls in selections:
            with self.subTest(controls=controls):
                request = self.request()
                del request["enable_thinking"]
                expected = self.success(self.execute({**request, **controls}))
                for combined in (
                    {**request, "chat_template_kwargs": controls},
                    {**request, **controls, "chat_template_kwargs": controls},
                    {"chat_template_kwargs": controls, **request, **controls},
                ):
                    self.assertEqual(self.success(self.execute(combined)), expected)
        request = self.request(
            enable_thinking=True,
            reasoning_effort="high",
            chat_template_kwargs={"reasoning_effort": "xhigh"},
        )
        expected = self.success(self.execute(request))
        del request["chat_template_kwargs"]
        self.assertEqual(self.success(self.execute(request)), expected)

    def test_template_controls_reject_conflicts_types_duplicates_and_unknown_keys(self):
        for name, top, nested in (
            ("enable_thinking", True, False),
            ("preserve_thinking", True, False),
            ("reasoning_effort", "low", "medium"),
        ):
            request = self.request(
                **{name: top, "chat_template_kwargs": {name: nested}}
            )
            for ordered in (request, dict(reversed(list(request.items())))):
                result = self.execute(ordered)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("must agree", result.stderr)
        invalid = (
            None,
            [],
            True,
            {"unknown": True},
            {"enable_thinking": "false"},
            {"reasoning_effort": None},
            {"preserve_thinking": 1},
        )
        for controls in invalid:
            with self.subTest(controls=controls):
                result = self.execute(self.request(chat_template_kwargs=controls))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        request = json.dumps(
            self.request(chat_template_kwargs={"preserve_thinking": True})
        )
        request = request.replace(
            '"preserve_thinking": true',
            '"preserve_thinking": true, "preserve_thinking": true',
        )
        result = self.execute(request)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def history(self):
        return [
            {"role": "user", "content": "Old question."},
            {
                "role": "assistant",
                "content": "Old answer.",
                "reasoning_content": "  Old thought λ.  ",
            },
            {"role": "user", "content": "Current question."},
            {
                "role": "assistant",
                "content": "Checking.",
                "reasoning_content": "Current thought λ.",
                "tool_calls": [
                    {
                        "id": "read_current",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "read_current", "content": "Found it."},
        ]

    def test_history_trimming_keeps_current_tool_cycle_with_thinking_on_or_off(self):
        for thinking in (False, True):
            request = self.request(
                enable_thinking=thinking, tools=TOOLS, messages=self.history()
            )
            preserved = self.success(self.execute(request))
            request["preserve_thinking"] = True
            self.assertEqual(self.success(self.execute(request)), preserved)
            request["preserve_thinking"] = False
            trimmed = self.success(self.execute(request))
            expected = preserved["prompt"].replace(
                "<|im_start|>assistant\n<think>\nOld thought λ.\n</think>\n\nOld answer.",
                "<|im_start|>assistant\nOld answer.",
                1,
            )
            self.assertNotEqual(expected, preserved["prompt"])
            self.assertEqual(trimmed["prompt"], expected)
            self.assertEqual(trimmed["request_options"], preserved["request_options"])
            self.assertIn("<think>\nCurrent thought λ.\n</think>", trimmed["prompt"])

    def test_user_form_tool_results_do_not_start_a_new_reasoning_cycle(self):
        wrapped = " \n<tool_response>\nFound λ.\n</tool_response>\t"
        for content in (
            wrapped,
            [{"type": "text", "text": text} for text in (wrapped[:8], wrapped[8:])],
            [
                {"type": "text", "text": "<tool_response>"},
                self.image((1, 2, 3)),
                {"type": "text", "text": "</tool_response>"},
            ],
        ):
            request = self.request(tools=TOOLS, messages=self.history())
            request["messages"][-1] = {"role": "user", "content": content}
            preserved = self.success(self.execute(request))
            request["preserve_thinking"] = False
            trimmed = self.success(self.execute(request))
            self.assertEqual(
                trimmed["prompt"],
                preserved["prompt"].replace(
                    "<think>\nOld thought λ.\n</think>\n\n", "", 1
                ),
            )
        for preserve in (False, True):
            result = self.execute(
                self.request(
                    preserve_thinking=preserve,
                    messages=[{"role": "user", "content": wrapped}],
                )
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")

    def test_real_user_parts_end_previous_reasoning_cycle_without_losing_images(self):
        for content in (
            "",
            None,
            [{"type": "text", "text": "New λ."}],
            [self.image((1, 2, 3), width=640, height=320)],
        ):
            request = self.request(
                tools=TOOLS,
                messages=[*self.history(), {"role": "user", "content": content}],
                model_options={"max_image_tokens": 128},
            )
            preserved = self.success(self.execute(request))
            request["preserve_thinking"] = False
            trimmed = self.success(self.execute(request))
            expected = preserved["prompt"].replace(
                "<think>\nOld thought λ.\n</think>\n\n", "", 1
            )
            expected = expected.replace(
                "<think>\nCurrent thought λ.\n</think>\n\n", "", 1
            )
            self.assertEqual(trimmed["prompt"], expected)

    def test_trimmed_history_retains_generated_tool_checkpoint_until_next_query(self):
        request = self.request(
            enable_thinking=True,
            preserve_thinking=False,
            tools=TOOLS,
            messages=self.history()[:3],
        )
        response = (
            "Current thought λ.\n</think>\n\nChecking.\n\n"
            "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        first = self.success(self.execute(request, response, chunk_size=7))
        self.assertTrue(first["reusable"])
        calls = [
            {key: value for key, value in call.items() if key != "index"}
            for call in first["tool_calls"]
        ]
        request["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": first["content"],
                    "reasoning_content": first["reasoning_content"],
                    "tool_calls": calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": "Found it.",
                },
            ]
        )
        following = self.success(self.execute(request))
        self.assertTrue(following["prompt"].startswith(first["checkpoint"]))
        request["messages"].append({"role": "user", "content": "Next question."})
        next_query = self.success(self.execute(request))
        self.assertNotIn("Current thought λ.", next_query["prompt"])
        self.assertIn("Found it.", next_query["prompt"])

    def test_text_parts_preserve_order_and_have_no_media_spans(self):
        request = self.request()
        expected = self.success(self.execute(request))
        request["messages"][0]["content"] = [
            {"type": "text", "text": text} for text in ("Hello ", "λ", ".")
        ]
        actual = self.success(self.execute(request))
        self.assertEqual(actual["prompt"], expected["prompt"])
        self.assertEqual(actual["tokens"], expected["tokens"])
        self.assertEqual(actual["media_spans"], [])

    def test_ordered_multiple_images_preserve_input_spans(self):
        parts = [
            {"type": "text", "text": "  Compare first: "},
            self.image((255, 0, 0)),
            {"type": "text", "text": " with second: "},
            self.image((0, 0, 255)),
            {"type": "text", "text": " and explain.  "},
        ]
        request = self.request(messages=[{"role": "user", "content": parts}])
        value = self.success(self.execute(request))
        marker = "<|vision_start|>" + "<|image_pad|>" * 64 + "<|vision_end|>"
        expected = "Compare first: " + marker + " with second: " + marker
        self.assertIn(expected + " and explain.<|im_end|>", value["prompt"])
        spans = value["media_spans"]
        self.assertEqual([span[0] for span in spans], [0, 1])
        for image, offset, count in spans:
            self.assertEqual(count, 64)
            self.assertEqual(value["tokens"][offset : offset + count], [131] * 64)
            self.assertNotEqual(value["tokens"][offset - 1], 131)
            self.assertNotEqual(value["tokens"][offset + count], 131)
        self.assertGreater(spans[1][1], spans[0][1] + 64)

    def test_images_across_turns_preserve_projection_ordinals(self):
        image = self.image((0, 255, 0))
        request = self.request(
            messages=[
                {"role": "user", "content": [image, image]},
                {"role": "assistant", "content": "Two green images."},
                {"role": "user", "content": [image, image]},
            ]
        )
        value = self.success(self.execute(request))
        self.assertEqual([span[0] for span in value["media_spans"]], [0, 1, 2, 3])
        request["messages"][-1]["content"].append(image)
        value = self.success(self.execute(request))
        self.assertEqual([span[0] for span in value["media_spans"]], [0, 1, 2, 3, 4])

    def video(self, count, width=64, height=32):
        # Small real JPEG inputs exercise the public video/jpeg transport;
        # numerical pixel and geometry coverage belongs to the shared motifs.
        image = Path(__file__).with_name(f"video_{width}x{height}.jpg").read_bytes()
        frame = base64.b64encode(image).decode("ascii")
        return {
            "type": "video_url",
            "video_url": {"url": "data:video/jpeg;base64," + ",".join([frame] * count)},
        }

    def video_options(self, timestamps):
        return {
            "video": {
                "fps": 100,
                "frames_indices": [round(timestamp * 100) for timestamp in timestamps],
                "num_frames": -1,
            }
        }

    def test_video_pairs_timestamps_and_odd_tail_among_images(self):
        parts = [
            {"type": "text", "text": "  Compare "},
            self.image((255, 0, 0), 32, 32),
            self.video(5),
            {"type": "text", "text": " then "},
            self.image((0, 0, 255), 32, 32),
            {"type": "text", "text": " please.  "},
        ]
        value = self.success(self.execute(self.request(
            messages=[{"role": "user", "content": parts}],
            media_io_kwargs=self.video_options([0.0, 0.75, 2.1, 3.9, 8.2]),
        )))
        pair = "<|vision_start|>" + "<|video_pad|>" * 2 + "<|vision_end|>"
        video = "".join(
            f"<{timestamp} seconds>" + pair for timestamp in ("0.4", "3.0", "8.2")
        )
        self.assertIn(video + " then ", value["prompt"])
        self.assertIn("please.<|im_end|>", value["prompt"])
        self.assertEqual([span[0] for span in value["media_spans"]], list(range(5)))
        self.assertEqual([span[2] for span in value["media_spans"]], [64, 2, 2, 2, 64])
        for projection, offset, count in value["media_spans"]:
            token = 132 if 1 <= projection <= 3 else 131
            self.assertEqual(value["tokens"][offset:offset + count], [token] * count)

    def test_video_rounding_and_group_count_through_native_caller(self):
        # One tiny clip covers both tie directions and more than four groups.
        times = [0.25, 0.25, 0.75, 0.75, 4.0, 5.0, 7.0, 8.0, 9.25]
        value = self.success(self.execute(self.request(messages=[{
            "role": "user", "content": [self.video(len(times), 1, 1)]
        }], media_io_kwargs=self.video_options(times))))
        pair = "<|vision_start|><|video_pad|><|vision_end|>"
        expected = ("0.2", "0.8", "4.5", "7.5", "9.2")
        self.assertIn(
            "".join(f"<{time} seconds>" + pair for time in expected),
            value["prompt"],
        )
        self.assertEqual([span[2] for span in value["media_spans"]], [1] * 5)

    def test_video_requires_a_pair_and_uniform_display_geometry(self):
        short = self.video(1)
        mismatched = self.video(1)
        mismatched["video_url"]["url"] += "," + self.video(1, 32, 32)[
            "video_url"
        ]["url"].split(",", 1)[1]
        for video in (short, mismatched):
            with self.subTest(video=video):
                result = self.execute(self.request(
                    messages=[{"role": "user", "content": [video]}]
                ))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "at least two frames with identical display dimensions", result.stderr
                )

    def test_general_image_shapes_and_request_budget(self):
        for width, height, budget, expected_rows in (
            (320, 224, 1024, 70),
            (224, 320, 1024, 70),
            (333, 257, 1024, 80),
            (1024, 768, 128, 117),
            (1024, 768, 1024, 768),
            (1, 1, 1024, 64),
        ):
            with self.subTest(width=width, height=height, budget=budget):
                request = self.request(
                    model_options={"max_image_tokens": budget},
                    messages=[
                        {"role": "user", "content": "Look carefully."},
                        {"role": "assistant", "content": "Ready."},
                        {
                            "role": "user",
                            "content": [self.image((3, 17, 29), width, height)],
                        },
                    ],
                )
                value = self.success(self.execute(request))
                self.assertEqual(len(value["media_spans"]), 1)
                image, offset, count = value["media_spans"][0]
                self.assertEqual((image, count), (0, expected_rows))
                self.assertEqual(
                    value["tokens"][offset : offset + count], [131] * count
                )

    def test_image_budget_validation(self):
        for options in (
            {"max_image_tokens": 0},
            {"max_image_tokens": 127},
            {"max_image_tokens": 16385},
            {"max_image_tokens": -1},
            {"max_image_tokens": 1024.5},
            {"max_image_tokens": True},
            {"max_image_tokens": "1024"},
            {"unknown": 1024},
            [],
        ):
            with self.subTest(options=options):
                result = self.execute(self.request(model_options=options))
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_production_xml_completion(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        value = self.success(
            self.execute(self.request(tools=TOOLS), "Ready.\n\n" + call)
        )
        self.assertEqual(value["output_cursor"], len("Ready.\n\n"))
        self.assertEqual(
            value["checkpoint"], value["prompt"] + "Ready.\n\n" + call + "<|im_end|>\n"
        )
        self.assertEqual(value["tool_calls"][0]["id"], "call_1_0")
        function = value["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "read")
        self.assertEqual(json.loads(function["arguments"]), {"path": "a.txt"})
        self.assertTrue(value["reusable"])

    def test_token_limit_omits_only_the_unfinished_tool_and_disables_reuse(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        prefix = "Check λ.\n</think>\n\nReady.\n\n"
        for stream, chunk_size in ((False, 0), (True, 1), (True, 7)):
            for complete_prefix in ("", call + "\n"):
                for tail in (
                    "<tool_call>",
                    "<tool_call><function=read><parameter=path>unfinished",
                    call[:-1],
                ):
                    with self.subTest(stream=stream, prefix=complete_prefix, tail=tail):
                        request = self.request(
                            stream=stream, enable_thinking=True, tools=TOOLS
                        )
                        value = self.success(
                            self.execute(
                                request,
                                prefix + complete_prefix + tail,
                                chunk_size=chunk_size,
                                truncated=True,
                            )
                        )
                        self.assertEqual(value["reasoning_content"], "Check λ.\n")
                        self.assertEqual(value["content"], "Ready.\n\n")
                        self.assertEqual(
                            len(value["tool_calls"]), bool(complete_prefix)
                        )
                        if complete_prefix:
                            function = value["tool_calls"][0]["function"]
                            self.assertEqual(function["name"], "read")
                            self.assertEqual(
                                json.loads(function["arguments"]), {"path": "a.txt"}
                            )
                        self.assertFalse(value["reusable"])
                        self.assertEqual(
                            value["checkpoint"],
                            value["prompt"]
                            + "Check λ.\n</think>\n\nReady."
                            + ("\n\n" + call if complete_prefix else "")
                            + "<|im_end|>\n",
                        )
                        following = self.request(
                            enable_thinking=True,
                            tools=TOOLS,
                            messages=[
                                *request["messages"],
                                {
                                    "role": "assistant",
                                    "content": value["content"],
                                    "reasoning_content": value["reasoning_content"],
                                    "tool_calls": [
                                        {
                                            key: entry
                                            for key, entry in tool.items()
                                            if key != "index"
                                        }
                                        for tool in value["tool_calls"]
                                    ],
                                },
                                {"role": "user", "content": "Continue."},
                            ],
                        )
                        replay = self.success(self.execute(following))
                        self.assertTrue(
                            replay["prompt"].startswith(value["checkpoint"])
                        )

    def test_token_limit_does_not_turn_malformed_tools_into_success(self):
        for truncated in (False, True):
            result = self.execute(
                self.request(tools=TOOLS),
                "Ready.\n\n<tool_call><funX",
                truncated=truncated,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("generated tool call expected", result.stderr)
            self.assertEqual(result.stdout, "")
        result = self.execute(self.request(tools=TOOLS), "<tool_call><function=read")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unterminated generated tool name", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_tool_catalog_matches_native_admission(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{index}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for index in range(33)
        ]
        value = self.success(self.execute(self.request(tools=tools), capacity=8192))
        catalog = value["prompt"].split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
        self.assertEqual([json.loads(line) for line in catalog.splitlines()], tools)

    def test_thinking_stream_fragments_and_retained_tool_history(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        request = self.request(enable_thinking=True, tools=TOOLS)
        response = "Check λ first.\n</think>\n\nReady.\n\n" + call
        reference = self.success(self.execute(request, response))
        self.assertEqual(reference["reasoning_content"], "Check λ first.\n")
        self.assertEqual(reference["content"], "Ready.\n\n")
        self.assertTrue(reference["reusable"])
        self.assertIn("Reasoning effort is set to xhigh.", reference["prompt"])
        self.assertTrue(
            reference["prompt"].endswith("<|im_start|>assistant\n<think>\n")
        )
        for chunk_size in (1, 2, 3, 7, 8, 13, 64):
            with self.subTest(chunk_size=chunk_size):
                value = self.success(
                    self.execute(
                        request,
                        response,
                        chunk_size=chunk_size,
                    )
                )
                self.assertEqual(value, reference)
        following = self.request(
            enable_thinking=True,
            tools=TOOLS,
            messages=[
                *request["messages"],
                {
                    "role": "assistant",
                    "content": reference["content"],
                    "reasoning_content": reference["reasoning_content"],
                    "tool_calls": [
                        {key: value for key, value in call.items() if key != "index"}
                        for call in reference["tool_calls"]
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1_0", "content": "Found it."},
            ],
        )
        value = self.success(self.execute(following))
        self.assertTrue(value["prompt"].startswith(reference["checkpoint"]))

    def test_truncated_thinking_cannot_reuse_closed_history(self):
        for response in (
            "still thinking",
            "unfinished </thin",
            "<tool_call> is an example",
        ):
            for chunk_size in (0, 1, 7):
                with self.subTest(response=response, chunk_size=chunk_size):
                    value = self.success(
                        self.execute(
                            self.request(enable_thinking=True),
                            response,
                            chunk_size=chunk_size,
                        )
                    )
                    self.assertEqual(value["reasoning_content"], response)
                    self.assertEqual(value["content"], "")
                    self.assertEqual(value["tool_calls"], [])
                    self.assertFalse(value["reusable"])

    def test_retained_history_requires_unchanged_generated_text(self):
        call = "<tool_call>\n<function=read>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"
        cases = [
            (False, "Ready.", True),
            (False, "Ready.\n\n", False),
            (False, "\tReady.", False),
            (False, " \t", False),
            (True, "Check.\n</think>\n\nReady.", True),
            (True, "Check.  \n</think> \nReady.", False),
            (True, "\n</think>\n\nReady.", True),
            (True, "</think>\n\nReady.", False),
            (False, call, True),
            (False, call.replace("\n<function", "\n  <function"), False),
        ]
        for thinking, response, reusable in cases:
            with self.subTest(thinking=thinking, response=response):
                value = self.success(
                    self.execute(
                        self.request(enable_thinking=thinking, tools=TOOLS),
                        response,
                        chunk_size=7,
                    )
                )
                self.assertEqual(value["reusable"], reusable)
                if reusable:
                    self.assertEqual(
                        value["checkpoint"],
                        value["prompt"] + response + "<|im_end|>\n",
                    )

    def test_empty_reasoning_and_incomplete_answer_marker(self):
        for prefix in ("", "\n"):
            for chunk_size in (0, 1, 7):
                value = self.success(
                    self.execute(
                        self.request(enable_thinking=True),
                        prefix + "</think>\n\nUse <tool_ca",
                        chunk_size=chunk_size,
                    )
                )
                self.assertEqual(value["reasoning_content"], prefix)
                self.assertEqual(value["content"], "Use <tool_ca")
                self.assertEqual(value["reusable"], bool(prefix))

    def test_source_sampling_defaults_and_request_overrides(self):
        defaults = {
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
                "seed": 9007199254740993,
            }
        }
        value = self.success(self.execute(self.request(), defaults=defaults))
        temperature, top_p, top_k, explicit_seed, seed = struct.unpack(
            "<ffiiq", bytes(value["request_options"])
        )
        self.assertAlmostEqual(temperature, 0.7)
        self.assertAlmostEqual(top_p, 0.95)
        self.assertEqual((top_k, explicit_seed, seed), (20, 1, 9007199254740993))
        value = self.success(
            self.execute(
                self.request(
                    temperature=0, top_p=1, top_k=0, seed=-9223372036854775808
                ),
                defaults=defaults,
            )
        )
        self.assertEqual(
            struct.unpack("<ffiiq", bytes(value["request_options"])),
            (0, 1, 0, 1, -9223372036854775808),
        )
        value = self.success(self.execute(self.request()))
        self.assertEqual(
            struct.unpack("<ffiiq", bytes(value["request_options"])), (0, 1, 0, 0, 1)
        )

    def test_invalid_sampling_fails_before_admission(self):
        for field, values in {
            "temperature": [-1, 1e100, 1e-45, None, "0.7"],
            "top_p": [0, -1, 1.01, 1e-45, None],
            "top_k": [-1, 1.5, 2147483648],
            "seed": [1.5, 9223372036854775808, -9223372036854775809],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    result = self.execute(self.request(**{field: value}))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
        for defaults in (
            {"sampling": {"top_p": 0}},
            {"sampling": {"typo": 1}},
            {"samplng": {}},
            {"sampling": []},
            {"sampling": {"tool_choice": "none"}},
        ):
            with self.subTest(defaults=defaults):
                result = self.execute(self.request(), defaults=defaults)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def test_tool_permission_is_request_local_and_preserves_history(self):
        for thinking in (False, True):
            request = self.request(enable_thinking=thinking, tools=TOOLS)
            ordinary = self.success(self.execute(request))
            for choice, expected_flag in (("auto", 0), ("none", 2)):
                request["tool_choice"] = choice
                value = self.success(self.execute(request))
                self.assertEqual(value["prompt"], ordinary["prompt"])
                self.assertEqual(value["tokens"], ordinary["tokens"])
                catalog = (
                    value["prompt"].split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
                )
                self.assertEqual(
                    [json.loads(line) for line in catalog.splitlines()], TOOLS
                )
                self.assertEqual(
                    struct.unpack("<ffiiq", bytes(value["request_options"])),
                    (0, 1, 0, expected_flag, 1),
                )
            # Seed and turn permission compose regardless of JSON field order,
            # and a request-local choice does not modify deployment defaults.
            for order in (
                {"seed": 17, "tool_choice": "none"},
                {"tool_choice": "none", "seed": 17},
            ):
                value = self.success(self.execute(self.request(tools=TOOLS, **order)))
                self.assertEqual(
                    struct.unpack("<ffiiq", bytes(value["request_options"])),
                    (0, 1, 0, 3, 17),
                )
            request["tool_choice"] = "none"
            value = self.success(
                self.execute(request, defaults={"sampling": {"seed": 19}})
            )
            self.assertEqual(
                struct.unpack("<ffiiq", bytes(value["request_options"])),
                (0, 1, 0, 3, 19),
            )

    def test_sampling_spelling_and_extreme_exponents(self):
        request = json.dumps(
            self.request(temperature=0.7, top_p=0.95, seed=9007199254740993)
        )
        ordinary = self.success(self.execute(request))
        defaults = '{"sampl\\u0069ng":{"temperat\\u0075re":0.7,"top_p":0.95,"s\\u0065ed":9007199254740993}}'
        self.assertEqual(
            self.success(self.execute(self.request(), defaults=defaults)),
            ordinary,
        )
        for literal in ("1e-999", "1e999", "-1e-999", "5e-324"):
            with self.subTest(literal=literal):
                result = self.execute(
                    request.replace('"temperature": 0.7', f'"temperature": {literal}'),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        duplicate = '{"sampling":{"temperature":0.7,"temperat\\u0075re":0.5}}'
        result = self.execute(self.request(), defaults=duplicate)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_capacity_and_model_identity_rejections_are_atomic(self):
        for request, capacity, diagnostic in (
            (self.request(), 1, "input exceeds the available context size"),
            (self.request(model="another-model"), 4096, "select model"),
        ):
            with self.subTest(capacity=capacity, request=request):
                result = self.execute(request, capacity=capacity)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn(diagnostic, result.stderr)


if __name__ == "__main__":
    unittest.main()
