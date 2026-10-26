#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Exercise built Pocket providers with externally provisioned parameters.

The regression checks whole PCM responses, seeded replay under changing request
membership, voice-prefix reuse, request rejection, cancellation, and retirement.
The card, lock, providers, and this checker determine Bazel's cache identity;
checkpoint payloads remain in the caller's provisioned installation.
"""

import argparse
import hashlib
import http.client
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import running_server, wait_counters, wait_status
from loom_serve_testing.model_staging import stage_model


def speech(address, request, *, status=200, first_block=False):
    connection = http.client.HTTPConnection(address.hostname, address.port)
    try:
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(request),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        try:
            if response.status != status:
                raise AssertionError(f"HTTP {response.status}: {response.read()!r}")
            if status != 200:
                return json.loads(response.read())
            if response.getheader("Content-Type") != "audio/pcm;rate=24000;channels=1":
                raise AssertionError(f"unexpected audio format: {response.headers}")
            pcm = response.read(3840) if first_block else response.read()
            if not pcm or len(pcm) % 3840 or not any(pcm):
                raise AssertionError(
                    f"expected nonempty PCM16 model blocks: {len(pcm)}"
                )
            return pcm
        finally:
            response.close()
    finally:
        # Closing after the first model block cancels an unfinished paragraph.
        connection.close()


def check(address, model, voices, output):
    request = {
        "model": model,
        "input": "Good day.",
        "seed": 42,
        "response_format": "pcm",
    }
    # Startup has already registered originals through the native upload path.
    reference = {
        voice: speech(address, {**request, "voice": voice}) for voice in voices
    }
    for voice, pcm in reference.items():
        (output / f"{voice}.pcm").write_bytes(pcm)

    # More requests than resident rows exercise row reuse, and repeated voices
    # must keep their own seed/cursors while sharing completed conditioning.
    order = voices + list(reversed(voices)) + [voices[0]]

    def replay(voice):
        pcm = speech(address, {**request, "voice": voice})
        if pcm != reference[voice]:
            raise AssertionError(f"cohort membership changed {voice!r} PCM")

    with ThreadPoolExecutor(max_workers=4) as clients:
        list(clients.map(replay, order))

    changed = speech(address, {**request, "voice": voices[0], "seed": 43})
    if changed == reference[voices[0]]:
        raise AssertionError("a distinct request seed reused the same PCM")

    # Each sentence fits a chunk; together they require multiple chunk restores.
    paragraph = {**request, "voice": voices[0], "input": "Good day. " * 20}
    complete = speech(address, paragraph)
    if speech(address, paragraph) != complete:
        raise AssertionError("text-chunk replay changed PCM")
    if len(complete) <= max(map(len, reference.values())):
        raise AssertionError(
            "paragraph output did not extend beyond one short utterance"
        )
    speech(address, paragraph, first_block=True)
    wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    counters = wait_counters(address, lambda _: True, model=model)
    if counters["cancelled"] != 1:
        raise AssertionError(f"disconnect did not retire as cancellation: {counters}")
    replay(voices[0])

    result = speech(address, {**request, "voice": "unregistered-voice"}, status=404)
    if "error" not in result:
        raise AssertionError(f"missing voice has no diagnostic: {result}")
    for override in (
        {"input": ""},
        {"speed": 1.25},
        {"instructions": "whisper"},
        {"input": "word " * 5000},
    ):
        result = speech(
            address, {**request, "voice": voices[0], **override}, status=400
        )
        if "error" not in result:
            raise AssertionError(f"rejected request has no diagnostic: {result}")
    replay(voices[-1])
    wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    return {
        "voices": {
            voice: {"bytes": len(pcm), "sha256": hashlib.sha256(pcm).hexdigest()}
            for voice, pcm in reference.items()
        },
        "paragraph": {
            "bytes": len(complete),
            "sha256": hashlib.sha256(complete).hexdigest(),
        },
        "cohort_replay": True,
        "chunk_replay": True,
        "cancellation_retired": True,
        "invalid_requests_rejected": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--assets", type=Path, default=os.environ.get("LOOM_MODEL_ASSETS")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.assets is None:
        parser.error("set LOOM_MODEL_ASSETS or --assets to a loom-fetch installation")
    if args.output is not None:
        output = args.output.absolute()
        output.mkdir(parents=True, exist_ok=False)
    elif os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    else:
        parser.error("--output is required outside a Bazel test")
    model = args.model.absolute()
    document = json.loads(model.read_text(encoding="utf-8"))
    (preset,) = document["presets"].values()
    service = document["variants"][preset["variant"]]["services"][preset["service"]]
    voices = list(service["voices"])
    model_id = document["model"]["id"]
    log = output / "pocket.log"
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
            result = check(urlsplit(f"http://{bound}"), model_id, voices, output)
    result = {"model": model_id, "device": args.device, **result}
    with (output / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
