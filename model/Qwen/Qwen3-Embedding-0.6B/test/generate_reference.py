#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.5.4", "torch==2.12.0+cpu", "transformers==4.51.3",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

"""Reproduce the independent publisher vectors from the pinned cached checkpoint.

This authoring tool updates only the publisher reference and tokenization.
Native regression digests and acceptance limits retain their separate evidence.
The default is offline; --download explicitly permits checkpoint acquisition.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    path = Path(__file__).with_name("reference.json")
    reference = json.loads(path.read_text())
    checkpoint = reference["checkpoint"]
    options = {
        "revision": checkpoint["revision"],
        "local_files_only": not arguments.download,
    }
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["repository"], **options)
    model = AutoModel.from_pretrained(
        checkpoint["repository"],
        torch_dtype=torch.float32,
        attn_implementation="eager",
        **options,
    ).eval()
    vectors = []
    with torch.inference_mode():
        for case in reference["cases"]:
            ids = tokenizer(case["text"], add_special_tokens=False)["input_ids"]
            ids.append(model.config.eos_token_id)
            tokens = torch.tensor([ids], dtype=torch.int64)
            output = model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
            ).last_hidden_state[:, -1]
            normalized = torch.nn.functional.normalize(output, p=2, dim=-1)[0]
            vectors.append(normalized.numpy().astype("<f4"))
            case["tokens"] = ids
    payload = b"".join(vector.tobytes() for vector in vectors)
    provenance = reference["reference"]
    provenance.update(
        torch=torch.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
        shape=[len(vectors), len(vectors[0])],
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    path.with_name(provenance["file"]).write_bytes(payload)
    path.write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
