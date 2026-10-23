#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check built embedding providers with separately provisioned parameters.

Bazel tracks the card, lock, providers, and small publisher reference. Full
checkpoint bytes remain outside the test inputs and are trusted to match the
acquisition lock. Native exact hashes detect changes; independent F32 vectors
bound the quantized result separately from that regression identity.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import struct
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import control, running_server, wait_status
from loom_serve_testing.model_staging import stage_model


def check(endpoint: str, model: str, reference: dict, publisher: bytes) -> dict:
    acceptance = reference["acceptance"]

    def request(path, body, status=200):
        message = urllib.request.Request(
            endpoint.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            response = urllib.request.urlopen(message)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            payload = response.read()
            if response.status != status:
                raise AssertionError((path, response.status, status, payload))
            return json.loads(payload)

    def embed(inputs, dimensions=1024, encoding="base64"):
        value = request(
            "/v1/embeddings",
            {
                "model": model,
                "input": inputs,
                "dimensions": dimensions,
                "encoding_format": encoding,
            },
        )
        vectors = []
        for index, row in enumerate(value["data"]):
            assert row["index"] == index
            if encoding == "base64":
                raw = base64.b64decode(row["embedding"], validate=True)
                vector = struct.unpack(f"<{dimensions}f", raw)
            else:
                vector = row["embedding"]
            assert len(vector) == dimensions and all(map(math.isfinite, vector))
            assert (
                abs(math.sqrt(sum(x * x for x in vector)) - 1.0)
                < acceptance["unit_norm_absolute_error"]
            )
            vectors.append(tuple(vector))
        return vectors, value["usage"]

    texts = [case["text"] for case in reference["cases"]]
    tokens = [
        request("/tokenize", {"model": model, "content": text, "add_special": True})[
            "tokens"
        ]
        for text in texts
    ]
    assert tokens == [case["tokens"] for case in reference["cases"]]
    batched, usage = embed(texts)
    assert len(batched) == len(texts)
    count = sum(map(len, tokens))
    assert usage["prompt_tokens"] == usage["total_tokens"] == count
    cosines = []
    observed_hashes = []
    for index, vector in enumerate(batched):
        encoded = struct.pack("<1024f", *vector)
        digest = hashlib.sha256(encoded).hexdigest()
        assert digest == reference["cases"][index]["native_sha256"], (index, digest)
        expected = struct.unpack_from("<1024f", publisher, index * 4096)
        cosine = sum(a * b for a, b in zip(vector, expected)) / math.sqrt(
            sum(a * a for a in vector) * sum(b * b for b in expected)
        )
        assert cosine >= acceptance["minimum_cosine"], (index, cosine)
        cosines.append(cosine)
        observed_hashes.append(digest)
    for text, ids, expected in zip(texts, tokens, batched):
        single, _ = embed(text)
        explicit, _ = embed(ids)
        assert single == explicit == [expected]

    def concurrent_replay(index):
        inputs = texts[index] if index % 2 == 0 else tokens[index]
        values, _ = embed(inputs)
        assert values == [batched[index]], index

    with ThreadPoolExecutor(max_workers=4) as clients:
        list(clients.map(concurrent_replay, [0, 1, 2, 2, 1, 0, 1, 0]))
    permuted, _ = embed(list(reversed(texts)))
    assert permuted == list(reversed(batched))
    encoded, _ = embed(texts[0], encoding="float")
    for actual, expected in zip(encoded[0], batched[0]):
        assert abs(actual - expected) <= 1e-8
    shortened, _ = embed(texts[0], dimensions=64)
    prefix = batched[0][:64]
    norm = math.sqrt(sum(value * value for value in prefix))
    assert (
        max(
            abs(actual - expected / norm)
            for actual, expected in zip(shortened[0], prefix)
        )
        < acceptance["prefix_absolute_error"]
    )
    # A complete cohort exercises all sixteen independently masked rows.
    sixteen, usage = embed([[1]] * 16)
    assert len(sixteen) == 16 and all(row == sixteen[0] for row in sixteen)
    assert usage["prompt_tokens"] == usage["total_tokens"] == 16
    for body in (
        {"input": [1] * 129},
        {"input": [[1]] * 17},
        {"input": [[1] * 65, [2] * 65]},
        {"input": [151669]},
        {"input": [-1]},
        {"input": [1], "dimensions": 1025},
    ):
        request("/v1/embeddings", {"model": model, **body}, status=400)
    return {
        "native_sha256": observed_hashes,
        "publisher_cosines": cosines,
        "cohort_replay": True,
        "concurrent_replay": True,
        "prefix_dimensions": 64,
        "invalid_requests_rejected": 6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--assets", type=Path, default=os.environ.get("LOOM_MODEL_ASSETS")
    )
    parser.add_argument(
        "--reference", type=Path, default=Path(__file__).with_name("reference.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.assets is None:
        parser.error(
            "set LOOM_MODEL_ASSETS or --assets to the provisioned model installation"
        )
    if args.output is not None:
        output = args.output.absolute()
        output.mkdir(parents=True, exist_ok=False)
    elif os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    else:
        parser.error("--output is required outside a Bazel test")
    reference = json.loads(args.reference.read_text())
    publisher = args.reference.with_name(reference["reference"]["file"]).read_bytes()
    assert len(publisher) == len(reference["cases"]) * 4096
    assert hashlib.sha256(publisher).hexdigest() == reference["reference"]["sha256"]
    card = args.model.absolute()
    model_id = json.loads(card.read_text())["model"]["id"]
    log = output / "embedding.log"
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as temporary:
        staged = stage_model(card, args.assets.absolute(), Path(temporary) / "model")
        with running_server(
            [
                str(args.server.absolute()),
                f"--model={staged}",
                f"--device={args.device}",
                "--workers=4",
                # Match the documented resident deployment on the NPU.
                "--pool_backing=fixed",
                "--listen=127.0.0.1:0",
            ],
            log,
        ) as (process, bound):
            if bound is None:
                raise RuntimeError(f"server failed before readiness: {log.read_text()}")
            address = urlsplit(f"http://{bound}")
            while True:
                details = control(address, "GET", "/v1/status", 200)["data"][0]["loom"]
                if "error" in details:
                    raise RuntimeError(f"embedding registration failed: {details}")
                if details["registration"] == "ready":
                    break
            result = check(address.geturl(), model_id, reference, publisher)
            wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    assert process.returncode == 0, process.returncode
    result.update(
        model=model_id, device=args.device, joined=True, returncode=process.returncode
    )
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
