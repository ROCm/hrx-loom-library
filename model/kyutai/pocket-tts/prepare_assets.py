#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "pocket-tts @ git+https://github.com/kyutai-labs/pocket-tts@ec569b2dd3fec4a601b391aace185791e92b3785",
#   "torch==2.9.1",
#   "numpy==2.5.4",
#   "scipy==1.18.1",
#   "soundfile==0.14.0",
#   "safetensors==0.8.0",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

"""Prepare a Pocket model card for a chosen set of reference voices.

The new publication contains source providers, F32 waveforms, rotary tables,
and a matching preset. Checkpoint and tokenizer acquisition remains with
loom-fetch; this authoring tool never loads a Python inference model. Ordinary
users fetch the already-published stock voice variants without this script.
"""

import argparse
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio, end_on_pause

CHECKPOINT_REVISION = "7f364c05d1be3faf2b234cdbe4cbb68a026d4218"
VOICES_REVISION = "323332d33f997de8394f24a193e1a76df720e01a"
REFERENCE_VOICES = {
    "alba": "alba-mackenna/casual.wav",
    "marius": "voice-donations/Selfie.wav",
    "jean": "ears/p010/freeform_speech_01_enhanced.wav",
    "anna": "vctk/p228_023_enhanced.wav",
}
SAMPLE_RATE = 24000
BLOCK_FRAMES = 1920


def voice_specifications(values: list[str]) -> list[tuple[str, Path | None]]:
    """Admit stable voice IDs before downloads or waveform preparation."""
    if not 1 <= len(values) <= 256:
        raise ValueError("Register between 1 and 256 voices")
    result = []
    names = set()
    for value in values:
        name, separator, filename = value.partition("=")
        if not 1 <= len(name.encode("utf-8")) <= 64 or name in names:
            raise ValueError(f"Voice IDs must be unique and 1–64 UTF-8 bytes: {name!r}")
        if separator:
            path = Path(filename).expanduser().resolve(strict=True)
            if not filename or not path.is_file():
                raise ValueError(
                    f"Voice reference must be a regular audio file: {value!r}"
                )
        elif name in REFERENCE_VOICES:
            path = None
        else:
            raise ValueError(f"Unknown reference voice {name!r}; use ID=/path/to/audio")
        names.add(name)
        result.append((name, path))
    return result


def prepare_voice(path: Path) -> np.ndarray:
    """Preserve the pinned reference's downmix, resampling, and pause policy."""
    audio, rate = audio_read(path)
    if audio.numel() == 0 or not torch.isfinite(audio).all():
        raise ValueError(f"Voice reference must contain finite audio: {path}")
    audio = end_on_pause(convert_audio(audio, rate, SAMPLE_RATE, 1), SAMPLE_RATE)
    if audio.numel() == 0 or not torch.isfinite(audio).all():
        raise ValueError(f"Voice conversion produced invalid audio: {path}")
    audio = torch.nn.functional.pad(audio, (0, -audio.numel() % BLOCK_FRAMES))
    return audio.numpy().reshape(-1, BLOCK_FRAMES)


def rotary_tables(capacity: int) -> dict[str, np.ndarray]:
    """Materialize Pocket's F32 frequencies and angles with libm evaluation.

    Explicit F32 round points match the numerical source contract; transcendental
    evaluation is independent of a device's approximate instruction selection.
    """

    def round_f32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", value))[0]

    factor = round_f32(-math.log(10000.0) / 32)
    frequencies = [round_f32(math.exp(round_f32(i * factor))) for i in range(32)]
    return {
        "rotary." + name: np.fromiter(
            (
                operation(round_f32(position * frequency))
                for position in range(capacity)
                for frequency in frequencies
            ),
            dtype=np.float32,
            count=capacity * 32,
        ).reshape(capacity, 32)
        for name, operation in (("cosine", math.cos), ("sine", math.sin))
    }


def write_archive(path: Path, tensors: dict[str, np.ndarray], metadata=None) -> None:
    """Write these F32 assets with deterministic Safetensors header ordering."""
    header = {"__metadata__": metadata} if metadata is not None else {}
    arrays = []
    offset = 0
    for name, tensor in sorted(tensors.items()):
        tensor = np.ascontiguousarray(tensor, dtype="<f4")
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
        arrays.append(tensor)
    encoded = json.dumps(
        header, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for tensor in arrays:
            output.write(memoryview(tensor).cast("B"))


def materialize_sources(package: Path, output: Path, program: dict) -> None:
    """Relocate the checked-in source closure into the new publication."""

    def copy_file(value):
        record = {"path": value} if isinstance(value, str) else value
        origin = record.pop("origin", record["path"])
        if origin.startswith("@library/"):
            source = package.parents[2] / origin.removeprefix("@library/")
        elif ":" not in origin and not origin.startswith("@"):
            source = package / origin
        else:
            raise ValueError(f"The source template requires a checkout file: {origin}")
        destination = output / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        for auxiliary in record.get("data", []):
            copy_file(auxiliary)

    for value in program["inputs"] + program["libraries"]:
        copy_file(value)


def prepare(output: Path, voices: list[str], name: str, offline: bool) -> Path:
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"Choose a new deployment directory: {output}")
    specifications = voice_specifications(voices)
    package = Path(__file__).resolve().parent
    card = json.loads((package / "model.json").read_text())
    preset = card["presets"]["streaming"]
    options = preset["options"]
    chunk_tokens = options["chunk_tokens"]
    generation_frames = (chunk_tokens * 25 + 5) // 6 + 25
    # Source context capacity is at most 65536. Reserve its complete text and
    # generation allowance before admitting any reference waveform.
    maximum_voice_frames = 65536 - 1 - chunk_tokens - generation_frames
    waveforms = []
    records = []
    descriptions = []
    origins = []
    source_voices = {
        voice["id"]: voice for voice in card["services"]["speech"]["interface"]["voices"]
    }
    torch.set_num_threads(2)
    for voice, path in specifications:
        if path is None:
            description = dict(source_voices[voice])
            source = REFERENCE_VOICES[voice]
            path = Path(
                hf_hub_download(
                    "kyutai/tts-voices",
                    source,
                    revision=VOICES_REVISION,
                    local_files_only=offline,
                )
            )
            origin = f"hf://kyutai/tts-voices/{source}@{VOICES_REVISION}"
        else:
            origin = str(path)
            description = {"id": voice, "reference": origin}
        waveform = prepare_voice(path)
        frames = waveform.shape[0]
        if frames > maximum_voice_frames:
            raise ValueError(
                f"Voice {voice!r} exceeds the supported context: {frames} blocks"
            )
        records.append({"id": voice, "frames": frames})
        descriptions.append({**description, "frames": frames})
        with path.open("rb") as source_file:
            source_digest = hashlib.file_digest(source_file, "sha256").hexdigest()
        origins.append(
            {
                "id": voice,
                "source": origin,
                "source_sha256": source_digest,
            }
        )
        waveforms.append(waveform)
        print(f"{voice}: {frames} blocks ({frames * BLOCK_FRAMES / SAMPLE_RATE:.2f} s)")
    maximum_frames = max(row["frames"] for row in records)
    context = max(
        options["context_capacity"],
        1 + maximum_frames + chunk_tokens + generation_frames,
    )
    rotary = max(
        options["rotary_capacity"], context, 16 * max(maximum_frames, generation_frames)
    )
    bank = np.zeros((len(waveforms), maximum_frames, BLOCK_FRAMES), dtype=np.float32)
    for row, waveform in enumerate(waveforms):
        bank[row, : waveform.shape[0]] = waveform
    options.update(voices=records, context_capacity=context, rotary_capacity=rotary)
    preset["variant"] = "custom-voices"
    preset["request"]["voice"] = records[0]["id"]
    card["model"]["name"] = name
    card["services"]["speech"]["interface"]["voices"] = descriptions
    layouts = card["services"]["speech"]["interface"]["parameter_layouts"]
    layouts["voices"]["shape"] = list(bank.shape)
    layouts["rotary"]["shape"] = [rotary, 32]
    variant = card["variants"]["english-three-voices"]
    variant["description"] = "Prepared reference voices: " + ", ".join(
        row["id"] for row in records
    )
    speech = variant["services"]["speech"]
    speech["programs"] = ["source"]
    speech["parameters"]["voices"]["files"][0]["origin"] = "prepared/voices.safetensors"
    card["variants"] = {"custom-voices": variant}
    card["presets"] = {"streaming": preset}
    card["programs"] = {"source": card["programs"]["source"]}
    card["origins"] = {key: card["origins"][key] for key in ("checkpoint", "tokenizer")}
    # Publish the card last. No running installation is changed, and full
    # checkpoint files never pass through this preparation step.
    output.mkdir(parents=True)
    materialize_sources(package, output, card["programs"]["source"])
    prepared = output / "prepared"
    prepared.mkdir()
    write_archive(prepared / "rotary.safetensors", rotary_tables(rotary))
    write_archive(
        prepared / "voices.safetensors",
        {"waveforms": bank},
        metadata={
            "sample_rate": str(SAMPLE_RATE),
            "voices": json.dumps(origins, ensure_ascii=False),
            "checkpoint_revision": CHECKPOINT_REVISION,
        },
    )
    model = output / "model.json"
    model.write_text(
        json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New source publication directory"
    )
    parser.add_argument(
        "--name", default="Pocket TTS — custom voices", help="Model display name"
    )
    parser.add_argument(
        "--voice",
        action="append",
        help="Reference name or ID=/path/to/audio; repeat for a set",
    )
    parser.add_argument(
        "--offline", action="store_true", help="Use only already-cached upstream files"
    )
    arguments = parser.parse_args()
    model = prepare(
        arguments.output,
        arguments.voice or ["alba", "marius", "anna"],
        arguments.name,
        arguments.offline,
    )
    print(f"Model card: {model}")
    print(
        "Next: loom-fetch CARD --preset=streaming --update-lock --output=INSTALLATION"
    )


if __name__ == "__main__":
    main()
