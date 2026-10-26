#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Checks built Moonshine programs against an independent PCM oracle.

The caller provisions runtime assets outside Bazel. The authored card and lock,
built programs, server, and oracle inputs determine the test's cache identity.
Each execution owns a fresh server and checks its complete output and shutdown.
"""

import argparse
import hashlib
import http.client
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import running_server
from loom_serve_testing.model_staging import stage_model


def transcribe(address, model, audio, content_type):
    boundary = "moonshine-independent-audio"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
            f"{model}\r\n--{boundary}\r\nContent-Disposition: form-data; "
            f'name="file"; filename="short.pcm"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        + audio
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection(address.hostname, address.port)
    try:
        # Odd HTTP chunk boundaries exercise byte reassembly independently of
        # the model's 320-frame quanta; no client timer controls recognition.
        connection.request(
            "POST",
            "/v1/audio/transcriptions",
            (body[start : start + 1031] for start in range(0, len(body), 1031)),
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
            encode_chunked=True,
        )
        response = connection.getresponse()
        payload = response.read()
        if response.status != 200:
            raise AssertionError(f"HTTP {response.status}: {payload!r}")
        return json.loads(payload)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--assets", type=Path, default=os.environ.get("LOOM_MODEL_ASSETS")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.assets is None:
        parser.error(
            "set LOOM_MODEL_ASSETS or --assets to an installation populated by "
            "loom-fetch; model weights are provisioned outside Bazel"
        )
    if args.output is not None:
        output = args.output.absolute()
        output.mkdir(parents=True, exist_ok=False)
    elif os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    else:
        parser.error("--output is required outside a Bazel test")

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    audio = args.audio.read_bytes()
    digest = hashlib.sha256(audio).hexdigest()
    if digest != reference["audio"]["sha256"]:
        raise AssertionError(f"oracle input digest differs: {digest}")
    if len(audio) != reference["audio"]["byte_length"]:
        raise AssertionError(f"oracle input byte length differs: {len(audio)}")
    model = args.model.absolute()
    document = json.loads(model.read_text(encoding="utf-8"))
    model_name = document["model"]["id"]
    log = output / "moonshine.log"
    # Only logs/results are retained as test outputs; shared assets stay in the
    # temporary tree so output collection cannot upload checkpoint bytes.
    with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as temporary:
        staged = stage_model(model, args.assets.absolute(), Path(temporary) / "model")
        with running_server(
            [
                str(args.server.absolute()),
                f"--model={staged}",
                f"--device={args.device}",
                "--workers=4",
                "--listen=127.0.0.1:0",
            ],
            log,
        ) as (_, bound):
            if bound is None:
                raise RuntimeError(
                    f"server failed before readiness:\n{log.read_text()}"
                )
            actual = transcribe(
                urlsplit(f"http://{bound}"),
                model_name,
                audio,
                reference["audio"]["content_type"],
            )
            expected = {"text": reference["transcript"]["text"]}
            if actual != expected:
                raise AssertionError(f"expected {expected!r}, received {actual!r}")
    # A successful transcript is published only after server retirement joins.
    result = {
        "model": str(model),
        "device": args.device,
        "audio_sha256": digest,
        **actual,
    }
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
