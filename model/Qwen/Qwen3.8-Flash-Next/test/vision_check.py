# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Image/video answers, retained partial pools, and parameter lifetimes.

Solid-color PNG/JPEG inputs provide independent task expectations without
checkpoint-sized fixtures. Numerical encoder and speculative-state tests
establish the lower boundaries; these requests exercise their complete
public-service composition.
"""

import base64
import json
import re
import struct
import uuid
import zlib
from pathlib import Path

from loom_serve_testing.chat import request_body
from loom_serve_testing.http_harness import control, wait_status


COLOR_QUESTION = "Hi! What color fills this image? Reply with only the color name."
VIDEO_SEQUENCE_QUESTION = (
    "Name the video colors in temporal order. Reply with only the color names."
)
VIDEO_RECALL_QUESTION = (
    "What color fills the final video frame? Reply with only the color name."
)
RED = (320, 256, (255, 0, 0))


def image_url(width, height, color):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    pixels = (b"\0" + bytes(color) * width) * height
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def history(images, question):
    content = [
        {"type": "image_url", "image_url": {"url": image_url(*image)}}
        for image in images
    ]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def video_history():
    # Distinct temporal planes, irregular pair times (0.4/3.0/8.2 s), an odd
    # final frame, and a separate still image exercise the full ordered input.
    frames = [
        base64.b64encode(
            Path(__file__).with_name(f"video_{name}.jpg").read_bytes()
        ).decode("ascii")
        for name in ("red", "dark_red", "blue", "dark_blue", "green")
    ]
    content = [
        {"type": "text", "text": "A still image: "},
        {
            "type": "image_url",
            "image_url": {"url": image_url(320, 256, (255, 255, 0))},
        },
        {"type": "text", "text": "A video clip: "},
        {
            "type": "video_url",
            "video_url": {"url": "data:video/jpeg;base64," + ",".join(frames)},
        },
        {"type": "text", "text": VIDEO_SEQUENCE_QUESTION},
    ]
    return [{"role": "user", "content": content}]


def append(messages, answer, question):
    return messages + [
        {"role": "assistant", "content": answer["text"]},
        {"role": "user", "content": question},
    ]


def cached(answer):
    return answer["usage"]["prompt_tokens_details"]["cached_tokens"]


def chat(address, model, session, messages, checkpoint=None):
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "enable_thinking": False,
        "max_tokens": 24,
    }
    # Every video in this lifecycle is the same five-frame color sequence.
    if any(
        part.get("type") == "video_url"
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
    ):
        body["media_io_kwargs"] = {
            "video": {
                "fps": 100,
                "frames_indices": [0, 75, 210, 390, 820],
                "num_frames": -1,
                "do_sample_frames": False,
            }
        }
    headers = {"X-Loom-Session": session}
    if checkpoint is not None:
        headers["X-Loom-Checkpoint"] = checkpoint
    result = request_body(address, body, headers)
    print(json.dumps({"case": session, **result}), flush=True)
    assert result["finish_reason"] == "stop", result
    assert not result["reasoning_content"] and not result["tool_calls"], result
    return result


def color(answer, expected):
    assert answer["text"].strip().lower().rstrip(".") == expected, answer


def run(address, model):
    """Check mixed media ownership through speculative or target-only execution."""
    prefix = "vision-check-" + uuid.uuid4().hex
    headers = {"Content-Type": "application/json", "X-Loom-Model": model}
    record = {}

    def settled():
        while True:
            status = control(address, "GET", "/v1/status", 200)
            state = next(item["loom"] for item in status["data"] if item["id"] == model)
            if "error" in state:
                raise RuntimeError(state)
            image = state.get("parameters", {}).get("image", {})
            if "error" in image:
                raise RuntimeError(image)
            if state["parameter_operation"] == "idle":
                return state

    def operation(name, expected=200, component=None):
        body = {"model": model}
        if component is not None:
            body["component"] = component
        return control(
            address,
            "POST",
            "/v1/models/" + name,
            expected,
            headers=headers,
            body=json.dumps(body),
        )

    def request(label, messages, checkpoint=None):
        answer = chat(address, model, prefix + "-" + label, messages, checkpoint)
        record[label] = answer
        return answer

    ordinary = request(
        "text-control",
        [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
    )
    assert ordinary["text"].strip() == "4", ordinary
    red = request("red", history([RED], COLOR_QUESTION))
    color(red, "red")
    wait_status(address, model=model, phase="idle", queued_requests=0, active_requests=0)
    if red["text"] != red["text"].strip():
        # Chat framing trims leading whitespace, so this answer cannot be
        # pinned as a byte-identical reusable suffix. Preserve that guard.
        control(
            address,
            "POST",
            "/v1/checkpoints/" + prefix + "-noncanonical",
            409,
            session=prefix + "-red",
            headers=headers,
        )

    retained_history = video_history()
    retained = request("retained-root", retained_history)
    assert re.findall(r"[a-z]+", retained["text"].lower()) == [
        "red",
        "blue",
        "green",
    ], retained
    wait_status(address, model=model, phase="idle", queued_requests=0, active_requests=0)
    checkpoint = prefix + "-video"
    checkpoint_path = "/v1/checkpoints/" + checkpoint
    receipt = control(
        address,
        "POST",
        checkpoint_path,
        201,
        session=prefix + "-retained-root",
        headers=headers,
    )
    record["checkpoint"] = receipt
    try:
        assert receipt["host_snapshot_bytes"] == 0, receipt
        # A later fork completes this QSA group with coordinates carried from
        # the retained video endpoint, rather than current-packet coordinates.
        assert receipt["position"] % 4 != 0, receipt
        blue = request("blue", history([(1024, 1024, (0, 0, 255))], COLOR_QUESTION))
        color(blue, "blue")
        multiple = request(
            "multiple",
            history(
                [
                    RED,
                    (320, 256, (0, 0, 255)),
                    (256, 320, (0, 255, 0)),
                    (256, 256, (255, 255, 0)),
                ],
                "Name the colors of the four images, in order. Reply with only the four color names.",
            ),
        )
        assert re.findall(r"\b(red|blue|green|yellow)\b", multiple["text"].lower()) == [
            "red",
            "blue",
            "green",
            "yellow",
        ], multiple

        operation("drain")
        try:
            operation("deactivate", 202, component="image")
            cold = settled()
            record["cold_image"] = cold
            assert not cold["parameters"]["image"]["active"], cold
            assert cold["parameters"]["image"]["committed_bytes"] == 0, cold
            assert cold["parameters"]["primary"]["active"], cold
        finally:
            operation("undrain")

        branches = []
        for ordinal, (prompt, expected) in enumerate(
            [
                (VIDEO_RECALL_QUESTION, "green"),
                (
                    "What color fills the video at 3 seconds? "
                    "Reply with only the color name.",
                    "blue",
                ),
            ]
        ):
            messages = append(retained_history, retained, prompt)
            answer = request("fork-" + str(ordinal), messages, checkpoint)
            color(answer, expected)
            assert cached(answer) == receipt["position"], (answer, receipt)
            branches.append((messages, expected))
        assert not settled()["parameters"]["image"]["active"], (
            "fork reactivated projector"
        )

        for ordinal, (messages, expected) in enumerate(branches):
            color(request("replay-" + str(ordinal), messages), expected)
        purple = request(
            "fresh-purple", history([(512, 320, (128, 0, 128))], COLOR_QUESTION)
        )
        color(purple, "purple")
        before = settled()
        assert before["parameters"]["image"]["active"], before

        operation("drain")
        try:
            operation("deactivate", 202)
            cold = settled()
            record["cold_primary"] = cold
            assert not cold["parameters"]["primary"]["active"], cold
            assert cold["parameters"]["primary"]["committed_bytes"] == 0, cold
            operation("activate", 202)
            warm = settled()
            record["reactivated_primary"] = warm
            assert warm["parameters"]["primary"]["active"], warm
            assert warm["incarnation"] == before["incarnation"], (before, warm)
            assert warm["compilation"] == before["compilation"], (before, warm)
        finally:
            operation("undrain")
        restored = request("restored", branches[0][0], checkpoint)
        color(restored, branches[0][1])
        assert cached(restored) == receipt["position"], (restored, receipt)
    finally:
        control(address, "DELETE", checkpoint_path, 200, headers=headers)
    return record


def replay(address, model, record):
    """Replay mixed media after the prior server and device have joined."""
    messages = append(
        video_history(), record["retained-root"], VIDEO_RECALL_QUESTION
    )
    answer = chat(address, model, "vision-cold-" + uuid.uuid4().hex, messages)
    color(answer, "green")
    assert cached(answer) == 0, answer
    return answer
