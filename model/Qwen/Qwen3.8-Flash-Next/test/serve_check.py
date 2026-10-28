# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check built Flash-Next providers against provisioned checkpoint assets.

The card, acquisition lock, providers, and frozen native responses are Bazel
inputs. Full parameters remain outside the build graph and are trusted to
match that lock. The checker owns its server and joins it before returning.
"""

import argparse
import http.client
import json
import os
import socket
import struct
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import context_check
import vision_check
from loom_serve_testing.chat import request
from loom_serve_testing.http_harness import control, running_server, wait_status
from loom_serve_testing.model_staging import stage_model


def messages(text):
    return [{"role": "user", "content": text}]


def append(history, reply, text):
    return history + [
        {"role": "assistant", "content": reply["text"]},
        {"role": "user", "content": text},
    ]


def cached(result):
    return result["usage"]["prompt_tokens_details"]["cached_tokens"]


def same_output(actual, expected):
    for field in ("text", "reasoning_content", "tool_calls", "finish_reason"):
        assert actual[field] == expected[field], (field, actual, expected)
    for field in ("prompt_tokens", "completion_tokens"):
        assert actual["usage"][field] == expected["usage"][field], (
            field,
            actual,
            expected,
        )


def run(address, model, reference):
    prefix = "flash-check-" + uuid.uuid4().hex
    headers = {"Content-Type": "application/json", "X-Loom-Model": model}
    selector = json.dumps({"model": model})
    consumed = {label: 0 for label in reference["responses"]}

    def chat(label, history, maximum=16, checkpoint=None):
        result = request(
            address,
            prefix + "-" + label,
            history,
            maximum,
            model=model,
            checkpoint=checkpoint,
        )
        print(
            json.dumps(
                {
                    "case": label,
                    "text": result["text"],
                    "usage": result["usage"],
                    "finish_reason": result["finish_reason"],
                }
            ),
            flush=True,
        )
        ordinal = consumed[label]
        expected = reference["responses"][label][ordinal]
        consumed[label] += 1
        for field in ("text", "finish_reason"):
            assert result[field] == expected[field], (label, field, result, expected)
        for field, value in expected["usage"].items():
            assert result["usage"][field] == value, (label, field, result, expected)
        assert not result["reasoning_content"] and not result["tool_calls"], result
        return result

    def operation(name, expected=200):
        return control(
            address,
            "POST",
            "/v1/models/" + name,
            expected,
            headers=headers,
            body=selector,
        )

    def settled():
        while True:
            status = control(address, "GET", "/v1/status", 200)
            state = next(item["loom"] for item in status["data"] if item["id"] == model)
            if "error" in state:
                raise RuntimeError(state)
            if state["parameter_operation"] == "idle":
                return state

    wait_status(address, model=model, phase="idle", active_requests=0)
    cases = [
        ("arithmetic", "What is 2+2? Reply with only the number.", "4"),
        ("instruction", "Reply with exactly the word COBALT.", "COBALT"),
    ]
    with ThreadPoolExecutor(max_workers=2) as clients:
        pending = [
            clients.submit(chat, label, messages(text), 8) for label, text, _ in cases
        ]
        replies = [future.result() for future in pending]
    for (_, _, expected), reply in zip(cases, replies):
        assert reply["text"].strip() == expected, reply

    history = messages(
        "Remember these three station codes for later questions. "
        "The north station code is COBALT. The south station code is AMBER. "
        "The east station code is IVORY. "
        + "The archive stores timetables, route maps, and maintenance records. " * 12
        + "Confirm with exactly READY. Do not repeat any code."
    )
    first = chat("root", history, 8)
    assert first["usage"]["prompt_tokens"] > 64, first
    checkpoint_path = "/v1/checkpoints/" + prefix
    receipt = control(
        address,
        "POST",
        checkpoint_path,
        201,
        session=prefix + "-root",
        headers=headers,
    )
    try:
        assert receipt["position"] > 64, receipt
        assert receipt["host_snapshot_bytes"] == 0, receipt
        branches = [
            append(history, first, text)
            for text in (
                "What is the north station code? Reply with only the code.",
                "What is the south station code? Reply with only the code.",
            )
        ]
        with ThreadPoolExecutor(max_workers=2) as clients:
            pending = [
                clients.submit(chat, f"fork-{i}", branch, 16, prefix)
                for i, branch in enumerate(branches)
            ]
            forks = [future.result() for future in pending]
        for fork in forks:
            assert cached(fork) == receipt["position"], (fork, receipt)

        # Continue before fresh replays can evict idle execution rows.
        continuation = append(
            branches[0],
            forks[0],
            "What is the east station code? Reply with only the code.",
        )
        continued = chat("fork-0", continuation)
        assert cached(continued) > receipt["position"], continued
        for i, branch in enumerate(branches):
            replay = chat(f"replay-{i}", branch)
            same_output(replay, forks[i])
            assert cached(replay) == 0, replay
        replay = chat("replay-continuation", continuation)
        same_output(replay, continued)
        assert cached(replay) == 0, replay

        # Weight ownership is independent of retained endpoints and compiled
        # commands. Release and reload the original shards, then reuse the pin.
        wait_status(address, model=model, phase="idle", active_requests=0)
        before = settled()
        operation("drain")
        try:
            operation("deactivate", 202)
            cold = settled()
            assert not cold["parameters"]["primary"]["active"], cold
            assert cold["parameters"]["primary"]["committed_bytes"] == 0, cold
            operation("activate", 202)
            warm = settled()
            assert warm["parameters"]["primary"]["active"], warm
            assert warm["incarnation"] == before["incarnation"], (before, warm)
            assert warm["compilation"] == before["compilation"], (before, warm)
            assert (
                warm["parameters"]["primary"]["committed_bytes"]
                == (before["parameters"]["primary"]["committed_bytes"])
            ), (before, warm)
        finally:
            operation("undrain")
        restored = chat("restored", branches[0], checkpoint=prefix)
        same_output(restored, forks[0])
        assert cached(restored) == receipt["position"], restored
        print(
            json.dumps(
                {
                    "qualified": "text, concurrent forks, page boundary, continuation, replay, and weight reactivation",
                    "parameter_bytes": warm["parameters"]["primary"]["committed_bytes"],
                    "checkpoint_position": receipt["position"],
                }
            ),
            flush=True,
        )
    finally:
        control(address, "DELETE", checkpoint_path, 200, headers=headers)
    assert consumed == {
        label: len(rows) for label, rows in reference["responses"].items()
    }, consumed

    # Reset an admitted streaming peer. Completion may win the race; either
    # outcome must retire ownership and allow a fresh request to finish.
    connection = http.client.HTTPConnection(address.hostname, address.port)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(
                {
                    "model": model,
                    "messages": messages("Describe a forest in winter in detail."),
                    "stream": True,
                    "temperature": 0,
                    "enable_thinking": False,
                    "max_tokens": 256,
                }
            ),
            {"Content-Type": "application/json", "X-Loom-Session": prefix + "-reset"},
        )
        peer = connection.sock
        response = connection.getresponse()
        assert response.status == 200, response.status
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        response.close()
    finally:
        connection.close()
    wait_status(
        address, model=model, phase="idle", queued_requests=0, active_requests=0
    )
    recovery = request(
        address,
        prefix + "-recovery",
        messages("What is 2+2? Reply with only the number."),
        8,
        model=model,
    )
    assert recovery["text"].strip() == "4", recovery
    return {
        "checked_responses": sum(consumed.values()),
        "checkpoint_position": receipt["position"],
        "parameter_bytes": warm["parameters"]["primary"]["committed_bytes"],
        "peer_reset_recovered": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="amdgpu://0")
    parser.add_argument("--memory-bytes", type=int, default=163208757248)
    parser.add_argument(
        "--assets", type=Path, default=os.environ.get("LOOM_MODEL_ASSETS")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path(__file__).with_name("reference.json")
    )
    parser.add_argument("--output", type=Path)
    scenarios = parser.add_mutually_exclusive_group()
    scenarios.add_argument(
        "--native-context",
        action="store_true",
        help="check the full native context, retained forks, and cold replay",
    )
    scenarios.add_argument(
        "--vision",
        action="store_true",
        help="check image/video answers, retained forks, component eviction, and cold replay",
    )
    args = parser.parse_args()
    if args.assets is None:
        parser.error(
            "set LOOM_MODEL_ASSETS or --assets to the provisioned installation"
        )
    if args.output is not None:
        output = args.output.absolute()
        output.mkdir(parents=True, exist_ok=False)
    elif os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    else:
        parser.error("--output is required outside a Bazel test")
    card = args.model.absolute()
    model = json.loads(card.read_text())["model"]["id"]
    reference = None
    if not args.native_context and not args.vision:
        reference = json.loads(args.reference.read_text())
        assert reference["model"] == model, (reference["model"], model)
    result = None
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as temporary:
        staged = stage_model(card, args.assets.absolute(), Path(temporary) / "model")
        # A process restart makes the second history genuinely cold without
        # changing the deployment's production prefix-cache policy.
        scenario = (
            context_check if args.native_context else vision_check if args.vision else None
        )
        for phase in (("service", "cold-replay") if scenario else ("service",)):
            log = output / (phase + ".log")
            command = [
                str(args.server.absolute()),
                "--model=" + str(staged),
                "--device=" + args.device,
                "--pool_backing=elastic",
                "--memory_bytes=" + str(args.memory_bytes),
                "--workers=2",
                "--listen=127.0.0.1:0",
            ]
            with running_server(command, log) as (process, bound):
                if bound is None:
                    raise RuntimeError(
                        "server failed before readiness: " + log.read_text()
                    )
                address = urlsplit("http://" + bound)
                while True:
                    details = next(
                        row["loom"]
                        for row in control(address, "GET", "/v1/status", 200)["data"]
                        if row["id"] == model
                    )
                    if "error" in details:
                        raise RuntimeError(details)
                    if details["registration"] == "ready":
                        break
                if phase == "cold-replay":
                    if args.native_context:
                        result["cold_replay"] = context_check.replay(
                            address, model, result, cold=True
                        )
                    else:
                        result["cold_replay"] = vision_check.replay(
                            address, model, result
                        )
                elif scenario:
                    result = scenario.run(address, model)
                else:
                    result = run(address, model, reference)
                wait_status(
                    address,
                    model=model,
                    phase="idle",
                    queued_requests=0,
                    active_requests=0,
                )
            assert process.returncode == 0, process.returncode
    result.update(
        model=model, device=args.device, joined=True, returncode=process.returncode
    )
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
