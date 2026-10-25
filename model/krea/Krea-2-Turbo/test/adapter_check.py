#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Exercise original adapter files through native image capture and retirement.

The supplied checkpoint must match the identity in reference.json. As with
base-model assets, its payload stays outside Bazel; the declared reference
records its test identity. Preparation happens inside the real server.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from image_check import check_disconnect, image
from loom_serve_testing.http_harness import (
    control,
    running_server,
    wait_counters,
    wait_status,
)
from loom_serve_testing.model_staging import stage_model


def selection(strengths):
    return [{"id": "watercolor", "scale": strength} for strength in strengths]


def check(address, model, variant, reference, output):
    request = {
        "model": model,
        "prompt": reference["short_prompt"],
        "seed": 0,
        "response_format": "b64_json",
    }
    cases = {
        "base": [],
        "single": selection([1.0]),
        "zero": selection([-0.0, 0.0]),
        "half": selection([0.5]),
        "negative": selection([-0.5]),
        "two": selection([0.5, 0.5]),
        "three": selection([0.5, -0.25, 0.75]),
        "four": selection([0.25] * 4),
        "seven": selection([0.25, -0.125, 0.125, 0.25, 0.5, -0.25, 0.25]),
        "alias": [{"id": "watercolor-alias"}],
    }
    records = []
    observed = {}
    base = reference["rgb_sha256"][variant]["384x384"]
    adapted = reference["adapter_rgb_sha256"][variant]

    def generate(name, choices, expected):
        png, shape, pixels = image(address, {**request, "loras": choices})
        if shape != (384, 384):
            raise AssertionError((name, shape))
        digest = hashlib.sha256(pixels).hexdigest()
        if digest != expected:
            raise AssertionError((name, digest, expected))
        (output / f"{name}.png").write_bytes(png)
        result = {"name": name, "shape": shape, "rgb_sha256": digest}
        print(json.dumps(result), flush=True)
        return result

    for name, choices in cases.items():
        expected = base if name in ("base", "zero") else adapted[name]
        result = generate(name, choices, expected)
        observed[name] = result["rgb_sha256"]
        records.append(result)
    for name, choices in reversed(list(cases.items())):
        records.append(generate(f"replay-{name}", choices, observed[name]))

    rejections = []
    for name, choices in (
        ("unknown", [{"id": "absent"}]),
        ("unknown-zero", [{"id": "absent", "scale": 0.0}]),
        ("nonfinite", [{"id": "watercolor", "scale": 1e100}]),
    ):
        rejected = image(address, {**request, "loras": choices}, status=400)
        rejections.append({"name": name, "response": rejected})

    before = wait_counters(address, lambda _: True, model=model)
    selector = json.dumps({"model": model})
    headers = {"Content-Type": "application/json"}
    control(address, "POST", "/v1/models/drain", 200, body=selector, headers=headers)
    wait_status(address, admission="drained", phase="idle", active_requests=0)
    # Public admission holds synchronize all clients before releasing their
    # requests. Different selection lengths share one geometry/precision tag.
    group = ["three", "two", "seven", "single", "four", "half", "two", "three"]
    with ThreadPoolExecutor(max_workers=len(group)) as clients:
        futures = [
            clients.submit(
                generate, f"cohort-{index}-{name}", cases[name], observed[name]
            )
            for index, name in enumerate(group)
        ]
        try:
            wait_status(address, held_requests=len(group))
        finally:
            control(
                address,
                "POST",
                "/v1/models/undrain",
                200,
                body=selector,
                headers=headers,
            )
        records.extend(future.result() for future in futures)
    wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    after = wait_counters(
        address,
        lambda values: values["completed"] == before["completed"] + len(group),
        model=model,
    )
    # With the ordinary unlimited command quantum each invocation is one
    # device part. Fewer invocations than images proves actual cohort use.
    invocations = after["batches"] - before["batches"]
    if not (len(group) + 1) // 2 <= invocations < len(group):
        raise AssertionError(("cohort coverage", before, after))

    disconnected = check_disconnect(
        address, {**request, "loras": cases["seven"]}, model, "reset"
    )
    records.append(generate("after-cancel", cases["seven"], observed["seven"]))
    return {
        "variant": variant,
        "images": records,
        "rejections": rejections,
        "cohort": {"requests": len(group), "native_invocations": invocations},
        "disconnected_request": disconnected,
    }


def check_banks(address, model, variant, reference, output, *, pressure):
    request = {
        "model": model,
        "prompt": reference["short_prompt"],
        "seed": 0,
        "response_format": "b64_json",
    }
    choices = {
        "base": [],
        "watercolor": [{"id": "watercolor"}],
        "anime": [{"id": "anime"}],
        "blend": [{"id": "watercolor", "scale": 0.3}, {"id": "anime", "scale": 0.7}],
        "reverse": [{"id": "anime", "scale": 0.7}, {"id": "watercolor", "scale": 0.3}],
        "mixed-three": [
            {"id": "watercolor", "scale": 0.25},
            {"id": "anime", "scale": 0.75},
            {"id": "watercolor", "scale": -0.5},
        ],
    }
    expected = reference["bank_rgb_sha256"][variant]
    result = {"variant": variant, "images": [], "rejections": []}

    def generate(name, selected):
        png, shape, pixels = image(address, {**request, "loras": choices[selected]})
        digest = hashlib.sha256(pixels).hexdigest()
        if shape != (384, 384) or digest != expected[selected]:
            raise AssertionError((name, shape, digest, expected[selected]))
        (output / f"{name}.png").write_bytes(png)
        record = {"name": name, "shape": shape, "rgb_sha256": digest}
        result["images"].append(record)
        print(json.dumps(record), flush=True)

    if pressure:
        # Reject a two-bank union before the first successful capture, then
        # alternate originals in an arena that fits exactly one of them.
        for phase in ("first-capture", "after-eviction"):
            rejected = image(
                address, {**request, "loras": choices["blend"]}, status=503
            )
            result["rejections"].append({"name": phase, "response": rejected})
            for index, selected in enumerate(
                ("watercolor", "anime", "watercolor", "anime", "base")
            ):
                generate(f"{phase}-{index}-{selected}", selected)
    else:
        for selected in choices:
            generate(selected, selected)
        for selected in (
            "anime",
            "blend",
            "watercolor",
            "mixed-three",
            "reverse",
            "base",
        ):
            generate(f"replay-{selected}", selected)
        result["disconnected_request"] = check_disconnect(
            address, {**request, "loras": choices["blend"]}, model, "reset"
        )
        generate("after-cancel", "blend")
    wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, default=os.environ.get("LOOM_MODEL_ASSETS")
    )
    parser.add_argument(
        "--adapter", type=Path, default=os.environ.get("LOOM_KREA_SOFTWATERCOLOR")
    )
    parser.add_argument(
        "--retroanime", type=Path, default=os.environ.get("LOOM_KREA_RETROANIME")
    )
    parser.add_argument(
        "--case", choices=("selections", "banks", "pressure"), default="selections"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.assets is None or args.adapter is None:
        parser.error(
            "set LOOM_MODEL_ASSETS and LOOM_KREA_SOFTWATERCOLOR, or supply their flags"
        )
    if args.case != "selections" and args.retroanime is None:
        parser.error("bank checks require LOOM_KREA_RETROANIME or --retroanime")
    if args.output:
        output = args.output.absolute()
        output.mkdir(parents=True, exist_ok=False)
    elif os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    else:
        parser.error("--output is required outside a Bazel test")
    card = args.model.absolute()
    document = json.loads(card.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    (preset,) = document["presets"].values()
    model = document["model"]["id"]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as temporary:
        root = Path(temporary)
        staged = stage_model(card, args.assets.absolute(), root / "model")
        adapter = str(args.adapter.resolve(strict=True))
        adapters = {"watercolor": adapter}
        if args.case == "selections":
            adapters["watercolor-alias"] = [adapter]
        else:
            adapters["anime"] = str(args.retroanime.resolve(strict=True))
        capacity = 536870912 if args.case == "banks" else 268435456
        deployment = {
            "name": model,
            "model": {"path": str(staged)},
            "definition": {
                "kind": "image",
                "width": 384,
                "height": 384,
                "text_tokens": 512,
                "options": {
                    "adapter_capacity_bytes": capacity,
                    "adapters": adapters,
                },
            },
        }
        serve = root / "serve.json"
        serve.write_text(json.dumps(deployment, indent=2) + "\n", encoding="utf-8")
        with running_server(
            [
                str(args.server.absolute()),
                f"--model={serve}",
                f"--device={args.device}",
                "--workers=4",
                "--listen=127.0.0.1:0",
            ],
            output / "server.log",
        ) as (process, bound):
            if bound is None:
                raise RuntimeError((output / "server.log").read_text())
            address = urlsplit(f"http://{bound}")
            while True:
                details = control(address, "GET", "/v1/status", 200)["data"][0]["loom"]
                if "error" in details:
                    raise RuntimeError(details)
                if details["registration"] == "ready":
                    break
            if args.case == "selections":
                result = check(address, model, preset["variant"], reference, output)
            else:
                result = check_banks(
                    address,
                    model,
                    preset["variant"],
                    reference,
                    output,
                    pressure=args.case == "pressure",
                )
    if process.returncode != 0:
        raise AssertionError(f"server shutdown failed: {process.returncode}")
    result.update(joined=True, returncode=process.returncode, device=args.device)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
