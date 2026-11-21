#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Check native Real-ESRGAN upscales using built providers and provisioned checkpoints.

The card, acquisition lock, and providers determine the Bazel test identity.
Checkpoint bytes stay in an external loom-fetch installation. This checker
owns its server, synthesizes its inputs, and compares decoded pixels,
independently of PNG compression.
"""

import argparse
import base64
import hashlib
import http.client
import json
import os
import socket
import struct
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from loom_serve_testing.http_harness import (
    control,
    running_server,
    wait_counters,
    wait_status,
)
from loom_serve_testing.model_staging import stage_model


def decode_png(png):
    """Decode the service's noninterlaced opaque eight-bit RGBA response with stdlib."""
    if png[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("image response is not PNG")
    offset = 8
    header = None
    compressed = bytearray()
    ended = False
    while offset < len(png):
        if len(png) - offset < 12:
            raise AssertionError("truncated PNG chunk")
        length, kind = struct.unpack_from(">I4s", png, offset)
        payload = png[offset + 8 : offset + 8 + length]
        end = offset + 12 + length
        if end > len(png):
            raise AssertionError("truncated PNG payload")
        (checksum,) = struct.unpack_from(">I", png, end - 4)
        if zlib.crc32(kind + payload) != checksum:
            raise AssertionError(f"invalid PNG {kind!r} checksum")
        if kind == b"IHDR":
            if header is not None or offset != 8 or length != 13:
                raise AssertionError("invalid PNG header")
            header = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or end != len(png):
                raise AssertionError("invalid PNG end")
            ended = True
        elif not kind[0] & 32:
            raise AssertionError(f"unsupported critical PNG chunk {kind!r}")
        offset = end
    if not ended or header is None or header[2:] != (8, 6, 0, 0, 0):
        raise AssertionError(f"expected noninterlaced RGBA8 PNG: {header}")
    width, height = header[:2]
    rows = zlib.decompress(compressed)
    stride = width * 4
    if not width or not height or len(rows) != (stride + 1) * height:
        raise AssertionError("PNG pixels do not match its dimensions")
    pixels = bytearray(stride * height)
    for row in range(height):
        begin = row * (stride + 1)
        method = rows[begin]
        if method > 4:
            raise AssertionError(f"invalid PNG row filter {method}")
        for column, value in enumerate(rows[begin + 1 : begin + 1 + stride]):
            position = row * stride + column
            left = pixels[position - 4] if column >= 4 else 0
            above = pixels[position - stride] if row else 0
            corner = pixels[position - stride - 4] if row and column >= 4 else 0
            if method == 1:
                value += left
            elif method == 2:
                value += above
            elif method == 3:
                value += (left + above) // 2
            elif method == 4:
                estimate = left + above - corner
                candidates = (left, above, corner)
                value += min(candidates, key=lambda item: abs(estimate - item))
            pixels[position] = value & 255
    if any(alpha != 255 for alpha in pixels[3::4]):
        raise AssertionError("Real-ESRGAN produced non-opaque alpha")
    # Preserve the learned RGB digest independently of its transport format.
    rgb = bytearray(width * height * 3)
    for channel in range(3):
        rgb[channel::3] = pixels[channel::4]
    return (width, height), bytes(rgb)


def pattern(width, height):
    """Return a deterministic raw RGB input with smooth ramps, edges, and texture.

    Red and green ramp across the axes, and blue is a hashed texture, so the
    upscaler sees flat regions, gradients, and high-frequency detail without
    a stored fixture.
    """
    pixels = bytearray(width * height * 3)
    for row in range(height):
        for column in range(width):
            position = (row * width + column) * 3
            pixels[position] = column * 255 // (width - 1)
            pixels[position + 1] = row * 255 // (height - 1)
            texture = (column * 37 + row * 101 + (column * row) % 53) & 255
            pixels[position + 2] = 255 if (column // 8 + row // 8) % 2 else texture
    return bytes(pixels)


def prompt(size, *, urlsafe=False):
    """Encode the quarter-size input for a WIDTHxHEIGHT output request."""
    width, height = size
    raw = pattern(width // 4, height // 4)
    encode = base64.urlsafe_b64encode if urlsafe else base64.b64encode
    return raw, encode(raw).decode("ascii")


def image(address, request, *, status=200):
    response = control(
        address,
        "POST",
        "/v1/images/generations",
        status,
        body=json.dumps(request),
        headers={"Content-Type": "application/json"},
    )
    if status != 200:
        if not response.get("error", {}).get("message"):
            raise AssertionError(f"rejected request has no diagnostic: {response}")
        return response
    (result,) = response["data"]
    png = base64.b64decode(result["b64_json"], validate=True)
    shape, pixels = decode_png(png)
    if min(pixels) == max(pixels):
        raise AssertionError("model produced a constant image")
    return png, shape, pixels


def check_disconnect(address, request, model, mode):
    """Join one disconnect outcome before checking a subsequent request."""
    wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    before = wait_counters(address, lambda _: True, model=model)
    connection = http.client.HTTPConnection(address.hostname, address.port)
    try:
        connection.request(
            "POST",
            "/v1/images/generations",
            json.dumps(request),
            {"Content-Type": "application/json"},
        )
        if mode == "reset":
            # FIN alone permits a response to the completed HTTP input.
            # An abortive close makes an unavailable peer observable
            # before the next model part or response write.
            linger = struct.pack("HH" if os.name == "nt" else "ii", 1, 0)
            connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        # A monotonic counter proves receipt even when the active phase
        # is shorter than the observer's polling interval.
        wait_counters(
            address,
            lambda values: values["requests"] > before["requests"],
            model=model,
        )
    finally:
        connection.close()
    # Disconnect cancellation is best effort; completion can race the
    # close. Both paths must produce exactly one accepted outcome.
    outcomes = ("cancelled", "completed", "failed", "rejected")
    before_terminal = sum(before[key] for key in outcomes)
    after = wait_counters(
        address,
        lambda values: sum(values[key] for key in outcomes) > before_terminal,
        model=model,
    )
    delta = {key: after[key] - before[key] for key in outcomes}
    if (
        delta["failed"]
        or delta["rejected"]
        or delta["cancelled"] + delta["completed"] != 1
    ):
        raise AssertionError(f"{mode} disconnect did not retire exactly once: {delta}")
    wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    return {key: delta[key] for key in ("cancelled", "completed")}


def check(address, document, reference, output, *, lifecycle=False):
    model = document["model"]["id"]
    (preset,) = document["presets"].values()
    variant = preset["variant"]
    execution = preset["execution"]
    default = (execution["width"], execution["height"])
    hashes = reference["rgb_sha256"][variant]
    records = []

    def request(size, **fields):
        raw, text = prompt(size)
        return raw, {
            "model": model,
            "prompt": text,
            "size": f"{size[0]}x{size[1]}",
            "response_format": "b64_json",
            **fields,
        }

    def compare(name, fields, expected_shape, expected_hash=None):
        png, shape, pixels = image(address, fields)
        if shape != expected_shape:
            raise AssertionError((name, shape, expected_shape))
        digest = hashlib.sha256(pixels).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise AssertionError((name, digest, expected_hash))
        (output / f"{name}.png").write_bytes(png)
        records.append({"name": name, "shape": shape, "rgb_sha256": digest})
        print(json.dumps(records[-1]), flush=True)
        return pixels

    default_key = f"{default[0]}x{default[1]}"
    raw, base = request(default)
    original = compare("default", base, default, hashes[default_key])

    # A non-square geometry needs a distinct prepared stage and input shape.
    alternate = (768, 400)
    _, fields = request(alternate)
    compare("geometry", fields, alternate, hashes["768x400"])

    # The URL-safe alphabet decodes to the same input, and the upscaler has
    # no noise, so a seed must not change any pixel.
    urlsafe = base64.urlsafe_b64encode(raw).decode("ascii")
    if urlsafe == base["prompt"]:
        raise AssertionError("input pattern does not exercise the URL-safe alphabet")
    for name, fields in (
        ("urlsafe", {**base, "prompt": urlsafe}),
        ("seeded", {**base, "seed": 7}),
    ):
        _, shape, pixels = image(address, fields)
        if shape != default or pixels != original:
            raise AssertionError(f"{name} request changed the upscaled pixels")

    # Protocol validation and the model's own admission and input checks.
    invalid = (
        {"size": "1000x1000"},
        {"size": "1040x1024"},
        {"size": "8192x1024"},
        {"size": "512x512" if default != (512, 512) else "256x256"},
        {"prompt": base64.b64encode(raw[:-3]).decode("ascii")},
        {"prompt": "!" * len(base["prompt"])},
        {"n": 2},
        {"options": {"tile": 512}},
    )
    for fields in invalid:
        image(address, {**base, **fields}, status=400)
    _, shape, pixels = image(address, base)
    if shape != default or pixels != original:
        raise AssertionError("rejected requests changed a subsequent upscale")

    disconnected = {}
    if lifecycle:

        def replay(_):
            _, shape, pixels = image(address, base)
            if shape != default or pixels != original:
                raise AssertionError("concurrent request changed upscaled pixels")

        with ThreadPoolExecutor(max_workers=3) as clients:
            list(clients.map(replay, range(3)))

        for mode in ("graceful", "reset"):
            disconnected[mode] = check_disconnect(address, base, model, mode)
            replay(0)

    wait_status(address, phase="idle", queued_requests=0, active_requests=0)
    return {
        "variant": variant,
        "images": records,
        "lifecycle": lifecycle,
        "disconnected_requests": disconnected,
        "invalid_requests_rejected": len(invalid),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--assets", type=Path, default=os.environ.get("LOOM_MODEL_ASSETS")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lifecycle", action="store_true")
    args = parser.parse_args()
    if args.assets is None:
        parser.error("set LOOM_MODEL_ASSETS or --assets to a loom-fetch installation")
    if args.output:
        output = args.output.absolute()
        output.mkdir(parents=True, exist_ok=False)
    elif os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    else:
        parser.error("--output is required outside a Bazel test")
    model = args.model.absolute()
    document = json.loads(model.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
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
            result = check(
                address, document, reference, output, lifecycle=args.lifecycle
            )
    if process.returncode != 0:
        raise AssertionError(f"server shutdown failed: {process.returncode}")
    result.update(joined=True, returncode=process.returncode, device=args.device)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
