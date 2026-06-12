"""Tests for changemaster.core.hardware."""

from __future__ import annotations

from pathlib import Path

from changemaster.core.hardware import (
    GPUInfo,
    HardwareInfo,
    HardwareTier,
    classify_tier,
    detect_hardware,
)


class TestClassifyTier:
    def test_low_few_cores(self) -> None:
        assert classify_tier(2, 16384, False) is HardwareTier.LOW

    def test_low_little_ram(self) -> None:
        assert classify_tier(8, 2048, False) is HardwareTier.LOW

    def test_medium(self) -> None:
        assert classify_tier(4, 8192, False) is HardwareTier.MEDIUM
        assert classify_tier(8, 4096, False) is HardwareTier.MEDIUM

    def test_high(self) -> None:
        assert classify_tier(8, 16384, False) is HardwareTier.HIGH
        assert classify_tier(16, 32768, False) is HardwareTier.HIGH

    def test_workstation_requires_gpu(self) -> None:
        assert classify_tier(16, 32768, True) is HardwareTier.WORKSTATION
        assert classify_tier(32, 65536, True) is HardwareTier.WORKSTATION


class TestDetectHardware:
    def test_returns_valid_snapshot(self, tmp_path: Path) -> None:
        info = detect_hardware(workdir=tmp_path)
        assert info.cpu_count >= 1
        assert info.total_ram_mb > 0
        assert info.available_ram_mb > 0
        assert info.free_disk_mb >= 0
        assert info.os_name
        assert info.python_version
        assert isinstance(info.tier, HardwareTier)

    def test_to_dict_serialisable(self) -> None:
        import json

        info = detect_hardware()
        payload = json.dumps(info.to_dict())
        assert info.tier.value in payload

    def test_missing_workdir_gives_zero_disk(self) -> None:
        info = detect_hardware(workdir=Path("/nonexistent/definitely/missing"))
        assert info.free_disk_mb == 0


class TestRecommendations:
    def _info(self, tier: HardwareTier, cpu: int = 8, avail: int = 8192) -> HardwareInfo:
        return HardwareInfo(
            cpu_count=cpu,
            total_ram_mb=16384,
            available_ram_mb=avail,
            free_disk_mb=1000,
            os_name="Linux",
            os_version="x",
            python_version="3.11",
            gpus=(GPUInfo("Test GPU", 8192, "cuda"),),
            tier=tier,
        )

    def test_workers_by_tier(self) -> None:
        assert self._info(HardwareTier.LOW).recommended_workers == 1
        assert self._info(HardwareTier.MEDIUM, cpu=8).recommended_workers == 4
        assert self._info(HardwareTier.HIGH, cpu=8).recommended_workers == 7
        assert self._info(HardwareTier.WORKSTATION, cpu=16).recommended_workers == 15

    def test_tile_size_by_tier(self) -> None:
        assert self._info(HardwareTier.LOW).recommended_tile_size == 512
        assert self._info(HardwareTier.MEDIUM).recommended_tile_size == 1024
        assert self._info(HardwareTier.HIGH).recommended_tile_size == 2048
        assert self._info(HardwareTier.WORKSTATION).recommended_tile_size == 4096

    def test_max_in_memory_floor(self) -> None:
        info = self._info(HardwareTier.LOW, avail=64)
        assert info.max_in_memory_mb == 64
        assert self._info(HardwareTier.HIGH, avail=8192).max_in_memory_mb == 2048
