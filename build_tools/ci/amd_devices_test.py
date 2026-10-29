# Copyright 2026 AMD ROCm™ Software
# SPDX-License-Identifier: MIT

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.ci import amd_devices
from build_tools.ci.amd_devices import Device


def device_dump(scheme: str, devices: list[Device]) -> str:
    return f"# Enumerated devices for driver '{scheme}'\n" + "".join(
        f"# --device={device.uri}\n"
        f"#   AMD GPU 1002:74a1:00 ({device.pci_address})\n"
        "# ===----------------------------------------------------------------------===\n"
        f"AMD endpoint: AMD GPU 1002:74a1:00 ({device.pci_address})\n"
        for device in devices
    )


class AmdDevicesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.topology = self.root / "nodes"
        self.drm = self.root / "drm"
        self.dri = self.root / "dri"
        for directory in (self.topology, self.drm, self.dri):
            directory.mkdir()
        self.addresses = ["0000:03:00.0", "0000:41:00.0", "0000:81:00.0"]
        # CPU nodes do not consume a ROCr GPU ordinal. Numeric ordering puts
        # nodes 10 and 11 after node 2, unlike directory/string ordering.
        for index in range(12):
            node = self.topology / str(index)
            node.mkdir()
            (node / "properties").write_text("simd_count 0\n")
        for index, node in enumerate((2, 10, 11)):
            render = f"renderD{128 + index}"
            (self.topology / str(node) / "properties").write_text(
                f"simd_count 64\ndrm_render_minor {128 + index}\nunique_id {101 + index}\n"
            )
            pci = self.root / self.addresses[index]
            pci.mkdir()
            (self.drm / render).mkdir()
            (self.drm / render / "device").symlink_to(pci, target_is_directory=True)
            # Opening the render node is an access check, not GPU execution.
            (self.dri / render).symlink_to("/dev/null")
        # The SDK's native order deliberately differs from ROCr/KFD order.
        orders = {"amdgpu": (2, 0, 1), "rdna": (1,), "cdna": (2, 0)}
        self.views = {
            scheme: [
                Device(f"{scheme}://id/{index + 1:032x}", self.addresses[index])
                for index in order
            ]
            for scheme, order in orders.items()
        }

    def sdk_query(self, command, *, env, **kwargs):
        self.assertEqual(command[0], "/sdk/loom-serve")
        self.assertTrue(kwargs["check"])
        scheme = command[1].removeprefix("--dump_devices=")
        devices = self.views[scheme]
        value = env.get(amd_devices.GPU_VISIBILITY[scheme])
        if value is not None:
            devices = [devices[int(index)] for index in value.split(",") if index]
        return mock.Mock(stdout=device_dump(scheme, devices))

    def configure(self, **environment):
        # Only the external SDK process is substituted; sysfs parsing, physical
        # correlation, selector generation, and verification are the real path.
        with mock.patch.object(
            amd_devices.subprocess, "run", side_effect=self.sdk_query
        ):
            return amd_devices.configure_visibility(
                "/sdk/loom-serve", environment, self.topology, self.drm, self.dri
            )

    def test_rocr_ordinal_maps_to_physical_device_in_each_native_view(self):
        result = self.configure(ROCR_VISIBLE_DEVICES="1", HIP_VISIBLE_DEVICES="0")
        self.assertEqual(
            result["devices"],
            [
                {
                    "uri": "amdgpu://id/00000000000000000000000000000002",
                    "pci_address": self.addresses[1],
                }
            ],
        )
        self.assertEqual(
            result["environment"],
            {
                "HAL_VISIBLE_AMDGPU_DEVICES": "2",
                "HAL_VISIBLE_RDNA_DEVICES": "0",
                "HAL_VISIBLE_CDNA_DEVICES": "",
            },
        )

    def test_uuid_and_multiple_device_order(self):
        result = self.configure(ROCR_VISIBLE_DEVICES="GPU-0000000000000067,0")
        self.assertEqual(
            [device["pci_address"] for device in result["devices"]],
            [self.addresses[2], self.addresses[0]],
        )
        self.assertEqual(result["environment"]["HAL_VISIBLE_AMDGPU_DEVICES"], "0,1")
        self.assertEqual(result["environment"]["HAL_VISIBLE_CDNA_DEVICES"], "0,1")
        self.assertEqual(result["environment"]["HAL_VISIBLE_RDNA_DEVICES"], "")

    def test_render_node_allocation(self):
        result = self.configure(PARENT_GPU_DEVICES=str(self.dri / "renderD129"))
        self.assertEqual(result["devices"][0]["pci_address"], self.addresses[1])
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.configure(
                ROCR_VISIBLE_DEVICES="0",
                PARENT_GPU_DEVICES=str(self.dri / "renderD129"),
            )

    def test_explicit_native_selection_can_narrow_allocation(self):
        result = self.configure(
            ROCR_VISIBLE_DEVICES="0,1", HAL_VISIBLE_AMDGPU_DEVICES="1"
        )
        self.assertEqual(result["devices"][0]["pci_address"], self.addresses[0])
        with self.assertRaisesRegex(ValueError, "outside the runner allocation"):
            self.configure(ROCR_VISIBLE_DEVICES="1", HAL_VISIBLE_AMDGPU_DEVICES="0")
        with self.assertRaisesRegex(ValueError, "outside the native allocation"):
            self.configure(ROCR_VISIBLE_DEVICES="1", HAL_VISIBLE_CDNA_DEVICES="0")

    def test_explicit_empty_family_is_preserved(self):
        result = self.configure(ROCR_VISIBLE_DEVICES="1", HAL_VISIBLE_RDNA_DEVICES="")
        self.assertEqual(result["environment"]["HAL_VISIBLE_RDNA_DEVICES"], "")

    def test_unallocated_multi_gpu_host_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit runner allocation"):
            self.configure()
        self.views["amdgpu"] = [
            Device("amdgpu://id/00000000000000000000000000000002", self.addresses[1])
        ]
        self.views["rdna"] = [
            Device("rdna://id/00000000000000000000000000000002", self.addresses[1])
        ]
        self.views["cdna"] = []
        self.assertEqual(
            self.configure()["environment"]["HAL_VISIBLE_AMDGPU_DEVICES"], "0"
        )

    def test_no_native_gpu_is_rejected(self):
        self.views = {scheme: [] for scheme in amd_devices.GPU_SCHEMES}
        with self.assertRaisesRegex(ValueError, "no available AMD GPUs"):
            self.configure()

    def test_invalid_or_empty_allocations_fail(self):
        for value in (
            "",
            "3",
            "-1",
            "0,0",
            "0,GPU-0000000000000065",
            "GPU-unknown",
            "0,",
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "unique available GPU"),
            ):
                self.configure(ROCR_VISIBLE_DEVICES=value)
        for value in ("", "3", "0,0", "-1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.configure(HAL_VISIBLE_AMDGPU_DEVICES=value)

    def test_missing_render_node_fails_instead_of_guessing_rocr_rank(self):
        (self.dri / "renderD128").unlink()
        with self.assertRaises(FileNotFoundError):
            self.configure(ROCR_VISIBLE_DEVICES="1")

    def test_duplicate_render_allocation_is_rejected_before_native_filtering(self):
        self.views["amdgpu"] = self.views["amdgpu"][:2]
        for name in ("renderD128", "renderD129"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "duplicate"),
            ):
                self.configure(
                    PARENT_GPU_DEVICES=",".join(
                        str(self.dri / device) for device in (name, name, "renderD130")
                    )
                )

    def test_allocated_gpu_missing_from_sdk_is_rejected(self):
        self.views["amdgpu"] = self.views["amdgpu"][:2]
        with self.assertRaisesRegex(ValueError, "no GPU available to the native HAL"):
            self.configure(ROCR_VISIBLE_DEVICES="1")

    def test_mixed_allocation_selects_native_devices_in_allocation_order(self):
        self.views["amdgpu"] = self.views["amdgpu"][:2]
        self.views["rdna"] = []
        result = self.configure(ROCR_VISIBLE_DEVICES="1,0,2")
        self.assertEqual(
            [device["pci_address"] for device in result["devices"]],
            [self.addresses[0], self.addresses[2]],
        )
        self.assertEqual(result["unavailable_devices"], [self.addresses[1]])
        self.assertEqual(
            result["environment"],
            {
                "HAL_VISIBLE_AMDGPU_DEVICES": "1,0",
                "HAL_VISIBLE_RDNA_DEVICES": "",
                "HAL_VISIBLE_CDNA_DEVICES": "1,0",
            },
        )

    def test_mixed_render_allocation_preserves_explicit_native_selection(self):
        self.views["amdgpu"] = self.views["amdgpu"][:2]
        self.views["rdna"] = []
        for selection in ({}, {"HAL_VISIBLE_AMDGPU_DEVICES": "1"}):
            with self.subTest(selection=selection):
                result = self.configure(
                    PARENT_GPU_DEVICES=",".join(
                        str(self.dri / name) for name in ("renderD129", "renderD128")
                    ),
                    **selection,
                )
                self.assertEqual(
                    result["devices"][0]["pci_address"], self.addresses[0]
                )
                self.assertEqual(result["unavailable_devices"], [self.addresses[1]])
                self.assertEqual(
                    result["environment"]["HAL_VISIBLE_CDNA_DEVICES"], "1"
                )

    def test_sdk_ignoring_visibility_is_rejected(self):
        def ignores_mask(tool, scheme, environment):
            return self.views[scheme]

        with mock.patch.object(amd_devices, "query_devices", side_effect=ignores_mask):
            with self.assertRaisesRegex(ValueError, "filtered amdgpu devices differ"):
                amd_devices.configure_visibility(
                    "/sdk/loom-serve", {"HAL_VISIBLE_AMDGPU_DEVICES": "1"}
                )

    def test_device_dump_rejects_missing_or_ambiguous_physical_identity(self):
        for output in (
            "",
            "# Enumerated devices for driver 'amdgpu'\n# --device=amdgpu://id/abc\n",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                amd_devices.parse_device_dump(output, "amdgpu")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            amd_devices.parse_device_dump(
                device_dump("amdgpu", [self.views["amdgpu"][0]] * 2), "amdgpu"
            )


if __name__ == "__main__":
    unittest.main()
