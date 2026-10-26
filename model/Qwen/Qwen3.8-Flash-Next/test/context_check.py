# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native-context retrieval, retained branches, and final-token accounting."""

import hashlib
import json
import uuid

from loom_serve_testing.chat import request
from loom_serve_testing.http_harness import control


CONTEXT_CAPACITY = 262144
PARAGRAPH = (
    "The archive contains route maps and maintenance schedules. "
    "A librarian records each revision, and engineers check the date "
    "before using a drawing.\n"
)
FACTS = (("north", "COBALT"), ("south", "AMBER"), ("east", "IVORY"))


def document(count):
    parts = ["Read this archive and remember the station codes. Reply only READY.\n"]
    markers = {
        count // 20: FACTS[0],
        count // 2: FACTS[1],
        count * 19 // 20: FACTS[2],
    }
    for ordinal in range(count):
        if ordinal in markers:
            station, code = markers[ordinal]
            parts.append(f"The {station} station code is {code}.\n")
        parts.append(PARAGRAPH)
    parts.append("End of archive. Reply only READY. Do not repeat the codes.")
    return "".join(parts)


def count_history(address, model, history):
    # Count the actual chat template. The raw tokenize utility has a separate
    # text-size bound and does not account for assistant framing.
    return control(
        address,
        "POST",
        "/v1/chat/completions/input_tokens",
        200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(
            {
                "model": model,
                "messages": history,
                "enable_thinking": False,
                "max_tokens": 8,
            }
        ),
    )["input_tokens"]


def cached(result):
    return result["usage"]["prompt_tokens_details"]["cached_tokens"]


def check_answer(result, expected):
    assert result["text"].strip() == expected, result
    assert result["finish_reason"] == "stop", result
    assert not result["reasoning_content"] and not result["tool_calls"], result
    assert result["usage"]["total_tokens"] <= CONTEXT_CAPACITY, result


def run(address, model):
    """Exercise one native-context history with the deployment's cache policy."""
    paragraph_tokens = control(
        address,
        "POST",
        "/tokenize",
        200,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"model": model, "content": PARAGRAPH, "add_special": False}),
    )["tokens"]
    target = CONTEXT_CAPACITY - 96
    count = (target - 128) // len(paragraph_tokens)
    for _ in range(3):
        prompt = document(count)
        history = [{"role": "user", "content": prompt}]
        token_count = count_history(address, model, history)
        if target - len(paragraph_tokens) <= token_count <= target:
            break
        count += (target - token_count) // len(paragraph_tokens)
    else:
        raise AssertionError(("Could not construct native-context input", token_count))

    prefix = "native-context-" + uuid.uuid4().hex
    record = {
        "context_capacity": CONTEXT_CAPACITY,
        "paragraph_count": count,
        "prompt_tokens": token_count,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    print(json.dumps({"case": "native-context-input", **record}), flush=True)
    seed = request(address, prefix, history, 8, model=model)
    print(json.dumps({"case": "native-context-seed", **seed}), flush=True)
    check_answer(seed, "READY")
    assert seed["usage"]["prompt_tokens"] == token_count, seed
    assert cached(seed) == 0, seed
    record["seed"] = seed
    checkpoint_path = "/v1/checkpoints/" + prefix
    headers = {"X-Loom-Model": model}
    checkpoint = control(
        address, "POST", checkpoint_path, 201, session=prefix, headers=headers
    )
    record["checkpoint"] = checkpoint
    try:
        assert checkpoint["host_snapshot_bytes"] == 0, checkpoint
        record["forks"] = []
        for station, expected in FACTS:
            branch = history + [
                {"role": "assistant", "content": seed["text"]},
                {
                    "role": "user",
                    "content": f"What is the {station} station code? Reply with only the code.",
                },
            ]
            answer = request(
                address,
                prefix + "-" + station,
                branch,
                8,
                model=model,
                checkpoint=prefix,
            )
            print(
                json.dumps({"case": "native-context-" + station, **answer}), flush=True
            )
            check_answer(answer, expected)
            assert cached(answer) == checkpoint["position"], (answer, checkpoint)
            record["forks"].append(answer)

        record["boundary"] = []
        for maximum in (8, 1):
            padding = 0
            target = CONTEXT_CAPACITY - maximum
            for _ in range(3):
                question = (
                    "Ignore this padding:"
                    + " x" * padding
                    + ". Write the integers from 1 through 1000, separated by commas. "
                    "Start immediately with 1 and continue without explanation."
                )
                branch = history + [
                    {"role": "assistant", "content": seed["text"]},
                    {"role": "user", "content": question},
                ]
                token_count = count_history(address, model, branch)
                if token_count == target:
                    break
                padding += target - token_count
                assert padding >= 0, (token_count, target)
            else:
                raise AssertionError(
                    ("Could not construct final-token input", token_count)
                )
            answer = request(
                address,
                prefix + "-boundary-" + str(maximum),
                branch,
                maximum,
                model=model,
                checkpoint=prefix,
            )
            print(json.dumps({"case": "native-context-boundary", **answer}), flush=True)
            assert answer["finish_reason"] == "length", answer
            assert answer["usage"]["prompt_tokens"] == target, answer
            assert answer["usage"]["completion_tokens"] == maximum, answer
            assert answer["usage"]["total_tokens"] == CONTEXT_CAPACITY, answer
            assert cached(answer) == checkpoint["position"], (answer, checkpoint)
            record["boundary"].append(answer)

        # A fresh session can reuse automatic intermediate prefixes. Its
        # answer still has to agree with the explicit retained branch.
        answer = replay(address, model, record, cold=False)
        assert 0 < cached(answer) < checkpoint["position"], (answer, checkpoint)
        record["prefix_replay"] = answer
    finally:
        control(address, "DELETE", checkpoint_path, 200, headers=headers)
    return record


def replay(address, model, record, *, cold):
    """Replay the complete history, optionally after a joined server restart."""
    prompt = document(record["paragraph_count"])
    assert hashlib.sha256(prompt.encode()).hexdigest() == record["prompt_sha256"]
    history = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": record["seed"]["text"]},
        {
            "role": "user",
            "content": "What is the north station code? Reply with only the code.",
        },
    ]
    answer = request(
        address, "context-replay-" + uuid.uuid4().hex, history, 8, model=model
    )
    print(
        json.dumps({"case": "cold-replay" if cold else "prefix-replay", **answer}),
        flush=True,
    )
    check_answer(answer, FACTS[0][1])
    expected = record["forks"][0]
    for field in ("prompt_tokens", "completion_tokens"):
        assert answer["usage"][field] == expected["usage"][field], (answer, expected)
    if cold:
        assert cached(answer) == 0, answer
    return answer
