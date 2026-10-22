#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Runs a declared Moonshine deployment against an independent PCM oracle."""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import running_server


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
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    audio = args.audio.read_bytes()
    digest = hashlib.sha256(audio).hexdigest()
    if digest != reference["audio"]["sha256"]:
        raise AssertionError(f"oracle input digest differs: {digest}")
    if len(audio) != reference["audio"]["byte_length"]:
        raise AssertionError(f"oracle input byte length differs: {len(audio)}")
    output_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    with tempfile.TemporaryDirectory() as temporary:
        logs = Path(output_root or temporary)
        log = logs / "moonshine.log"
        with running_server(
            [
                str(args.server.absolute()),
                f"--model={args.deployment.absolute()}",
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
                "moonshine-streaming-tiny",
                audio,
                reference["audio"]["content_type"],
            )
            expected = {"text": reference["transcript"]["text"]}
            if actual != expected:
                raise AssertionError(f"expected {expected!r}, received {actual!r}")
            print(
                json.dumps({"provider": args.deployment.parent.name, **actual}),
                flush=True,
            )


if __name__ == "__main__":
    main()
