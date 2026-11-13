#!/usr/bin/env python3
# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

"""Maps the runner's AMD GPU allocation to the installed SDK's device views."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


GPU_SCHEMES = ("amdgpu", "rdna", "cdna")
GPU_VISIBILITY = {
    scheme: f"HAL_VISIBLE_{scheme.upper()}_DEVICES" for scheme in GPU_SCHEMES
}


@dataclass(frozen=True)
class Device:
    # Exact endpoint URI reported by the installed SDK.
    uri: str
    # Physical PCI address used to correlate independent ordinal namespaces.
    pci_address: str


def query_physical_devices(tool: str) -> dict[str, list[str | None]]:
    """Indexes structured physical identities independently of HAL visibility."""
    result = subprocess.run(
        [tool, "--engine=gpu", "--format=json"],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = json.loads(result.stdout)
    if inventory["schema_version"] != 1:
        raise ValueError("Unsupported amdf-info inventory schema")
    identities = {}
    for endpoint in inventory["endpoints"]:
        address = endpoint.get("pci_address")
        pci_address = (
            f"{address['domain']:04x}:{address['bus']:02x}:"
            f"{address['device']:02x}.{address['function']:x}"
            if address and "error" not in address
            else None
        )
        paths = [f"id/{endpoint['id']}"]
        unique_id = int(endpoint.get("gpu", {}).get("unique_id", "0"), 16)
        if unique_id:
            paths.append(f"GPU-{unique_id:016x}")
        schemes = ["amdgpu"]
        if endpoint["family"] in ("rdna", "cdna"):
            schemes.append(endpoint["family"])
        for scheme in schemes:
            for path in paths:
                identities.setdefault(f"{scheme}://{path}", []).append(pci_address)
    return identities


def parse_device_dump(
    output: str, scheme: str, identities: dict[str, list[str | None]]
) -> list[Device]:
    """Resolves the SDK's ordered HAL view without interpreting display names."""
    if f"# Enumerated devices for driver '{scheme}'" not in output:
        raise ValueError(f"Missing {scheme} enumeration in the SDK device dump")
    device_path = rf"(?:id/[0-9a-f]{{32}}|GPU-[0-9a-f]{{16}})"
    pattern = rf"^# --device=({scheme}://{device_path})$"
    uris = re.findall(pattern, output, re.MULTILINE)
    if len(uris) != output.count("# --device="):
        raise ValueError(
            f"Cannot identify every {scheme} endpoint in the SDK device dump"
        )
    devices = []
    for uri in uris:
        addresses = identities.get(uri, [])
        if len(addresses) != 1 or addresses[0] is None:
            raise ValueError(f"No unique physical PCI identity for {uri}")
        devices.append(Device(uri, addresses[0]))
    if len({device.pci_address for device in devices}) != len(devices):
        raise ValueError(f"Ambiguous {scheme} endpoints share a PCI address")
    return devices


def query_devices(
    tool: str,
    scheme: str,
    environment: dict[str, str],
    identities: dict[str, list[str | None]],
) -> list[Device]:
    result = subprocess.run(
        [tool, f"--dump_devices={scheme}"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_device_dump(result.stdout, scheme, identities)


def select_ordinals(value: str, devices: list[Device], variable: str) -> list[Device]:
    if not value.strip():
        return []
    tokens = [token.strip() for token in value.split(",")]
    if any(not re.fullmatch(r"[0-9]+", token) for token in tokens):
        raise ValueError(f"{variable} must contain decimal device ordinals")
    ordinals = [int(token) for token in tokens]
    if len(set(ordinals)) != len(ordinals) or any(
        index >= len(devices) for index in ordinals
    ):
        raise ValueError(f"{variable} contains duplicate or unavailable ordinals")
    return [devices[index] for index in ordinals]


def rocr_allocation(value: str, topology: Path, drm: Path, dri: Path) -> list[str]:
    """Resolves ROCr ranks/UUIDs through KFD nodes and physical render nodes.

    ROCr BuildTopology ranks compute-bearing KFD nodes in ascending node order.
    Inaccessible nodes can alter those ranks, so this mapper requires the complete
    KFD GPU inventory to be accessible instead of guessing a renumbering.
    """
    devices = []
    for node in sorted(topology.iterdir(), key=lambda path: int(path.name)):
        properties = dict(
            line.split() for line in (node / "properties").read_text().splitlines()
        )
        if int(properties["simd_count"]) == 0:
            continue
        render = f"renderD{int(properties['drm_render_minor'])}"
        descriptor = os.open(dri / render, os.O_RDWR | os.O_CLOEXEC)
        os.close(descriptor)
        pci_address = (drm / render / "device").resolve(strict=True).name
        devices.append((pci_address, f"GPU-{int(properties['unique_id']):016x}"))

    selected = []
    for token in value.split(","):
        token = token.strip()
        if re.fullmatch(r"[0-9]+", token) and int(token) < len(devices):
            matches = [devices[int(token)][0]]
        elif re.fullmatch(r"GPU-[0-9a-fA-F]{16}", token):
            matches = [pci for pci, uuid in devices if uuid.lower() == token.lower()]
        else:
            matches = []
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError(
                f"ROCR_VISIBLE_DEVICES token {token!r} has no unique available GPU"
            )
        selected.extend(matches)
    return selected


def configure_visibility(
    tool: str,
    amdf_info: str,
    environment: dict[str, str],
    topology: Path = Path("/sys/class/kfd/kfd/topology/nodes"),
    drm: Path = Path("/sys/class/drm"),
    dri: Path = Path("/dev/dri"),
) -> dict:
    unmasked = {
        name: value
        for name, value in environment.items()
        if name not in GPU_VISIBILITY.values()
    }
    identities = query_physical_devices(amdf_info)
    views = {
        scheme: query_devices(tool, scheme, unmasked, identities)
        for scheme in GPU_SCHEMES
    }
    all_devices = views["amdgpu"]
    if not all_devices:
        raise ValueError("The native HAL reports no available AMD GPUs")
    native_by_address = {device.pci_address: device for device in all_devices}
    allocation = None
    if "ROCR_VISIBLE_DEVICES" in environment:
        allocation = rocr_allocation(
            environment["ROCR_VISIBLE_DEVICES"], topology, drm, dri
        )
    if "PARENT_GPU_DEVICES" in environment:
        render_nodes = environment["PARENT_GPU_DEVICES"].split(",")
        parent_allocation = []
        for name in render_nodes:
            path = Path(name.strip())
            if path.parent != dri or not re.fullmatch(r"renderD[0-9]+", path.name):
                raise ValueError(
                    f"PARENT_GPU_DEVICES contains an invalid render node: {name!r}"
                )
            parent_allocation.append(
                (drm / path.name / "device").resolve(strict=True).name
            )
        if len(set(parent_allocation)) != len(parent_allocation):
            raise ValueError("PARENT_GPU_DEVICES contains duplicate physical devices")
        if allocation is not None and set(parent_allocation) != set(allocation):
            raise ValueError("PARENT_GPU_DEVICES conflicts with ROCR_VISIBLE_DEVICES")
        allocation = parent_allocation

    variable = GPU_VISIBILITY["amdgpu"]
    if variable in environment:
        selected = select_ordinals(environment[variable], all_devices, variable)
        if allocation is not None and any(
            device.pci_address not in allocation for device in selected
        ):
            raise ValueError(f"{variable} selects a GPU outside the runner allocation")
    elif allocation is not None:
        selected = [
            native_by_address[address]
            for address in allocation
            if address in native_by_address
        ]
        if not selected:
            raise ValueError(
                "The runner allocation has no GPU available to the native HAL"
            )
    elif len(all_devices) == 1:
        selected = all_devices
    else:
        raise ValueError(
            "Multiple GPUs require an explicit runner allocation or HAL_VISIBLE_AMDGPU_DEVICES"
        )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError(
            "GPU qualification requires a nonempty, distinct device allocation"
        )

    selected_addresses = {device.pci_address for device in selected}
    updates = {}
    expected = {}
    for scheme, devices in views.items():
        variable = GPU_VISIBILITY[scheme]
        if variable in environment:
            expected[scheme] = select_ordinals(environment[variable], devices, variable)
            if any(
                device.pci_address not in selected_addresses
                for device in expected[scheme]
            ):
                raise ValueError(
                    f"{variable} selects a GPU outside the native allocation"
                )
        else:
            by_address = {device.pci_address: device for device in devices}
            expected[scheme] = [
                by_address[device.pci_address]
                for device in selected
                if device.pci_address in by_address
            ]
        updates[variable] = ",".join(
            str(devices.index(device)) for device in expected[scheme]
        )

    masked = environment | updates
    for scheme in GPU_SCHEMES:
        if query_devices(tool, scheme, masked, identities) != expected[scheme]:
            raise ValueError(
                f"The SDK's filtered {scheme} devices differ from the runner allocation"
            )
    return {
        "devices": [
            {"uri": device.uri, "pci_address": device.pci_address}
            for device in selected
        ],
        "unavailable_devices": [
            address for address in allocation or () if address not in native_by_address
        ],
        "environment": updates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loom-serve", required=True, help="Installed SDK loom-serve executable."
    )
    parser.add_argument(
        "--amdf-info", required=True, help="The same SDK's amdf-info executable."
    )
    parser.add_argument(
        "--github-env",
        type=Path,
        required=True,
        help="GitHub Actions environment file.",
    )
    args = parser.parse_args()
    try:
        result = configure_visibility(args.loom_serve, args.amdf_info, dict(os.environ))
        with args.github_env.open("a") as output:
            for name, value in result["environment"].items():
                output.write(f"{name}={value}\n")
    except subprocess.CalledProcessError as error:
        parser.error(f"SDK device enumeration failed: {error.stderr.strip()}")
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
