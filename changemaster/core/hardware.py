"""Hardware detection and capability tiering.

Probes CPU, RAM, disk and (optionally) GPU resources so the application can
adapt processing strategies (tile sizes, worker counts, in-memory limits) to
the host machine. All probing is defensive: failures degrade to conservative
defaults instead of crashing.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class HardwareTier(str, Enum):
    """Coarse machine capability classification used to pick presets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    WORKSTATION = "workstation"


@dataclass(frozen=True)
class GPUInfo:
    """Information about a detected GPU device."""

    name: str
    memory_mb: int
    backend: str  # e.g. "cuda", "opencl", "none"


@dataclass(frozen=True)
class HardwareInfo:
    """Snapshot of host machine resources.

    Attributes
    ----------
    cpu_count:
        Number of logical CPU cores (always >= 1).
    total_ram_mb:
        Total physical RAM in megabytes.
    available_ram_mb:
        Currently available RAM in megabytes.
    free_disk_mb:
        Free disk space (in MB) on the drive hosting the working directory.
    os_name / os_version / python_version:
        Platform identification strings.
    gpus:
        Detected GPU devices (empty when none / undetectable).
    tier:
        Computed :class:`HardwareTier`.
    """

    cpu_count: int
    total_ram_mb: int
    available_ram_mb: int
    free_disk_mb: int
    os_name: str
    os_version: str
    python_version: str
    gpus: tuple[GPUInfo, ...] = field(default_factory=tuple)
    tier: HardwareTier = HardwareTier.LOW

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        data = asdict(self)
        data["tier"] = self.tier.value
        data["gpus"] = [asdict(g) for g in self.gpus]
        return data

    @property
    def recommended_workers(self) -> int:
        """Recommended worker process count for parallel jobs."""
        if self.tier is HardwareTier.LOW:
            return 1
        if self.tier is HardwareTier.MEDIUM:
            return max(1, self.cpu_count // 2)
        return max(2, self.cpu_count - 1)

    @property
    def recommended_tile_size(self) -> int:
        """Recommended tile edge length (pixels) for tiled processing."""
        if self.tier is HardwareTier.LOW:
            return 512
        if self.tier is HardwareTier.MEDIUM:
            return 1024
        if self.tier is HardwareTier.HIGH:
            return 2048
        return 4096

    @property
    def max_in_memory_mb(self) -> int:
        """Maximum image size (MB) allowed to be loaded fully in memory."""
        return max(64, self.available_ram_mb // 4)


def _detect_ram_mb() -> tuple[int, int]:
    """Detect (total, available) RAM in MB, with safe fallbacks."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total // (1024 * 1024)), int(vm.available // (1024 * 1024))
    except Exception:  # noqa: BLE001 - any psutil failure degrades gracefully
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        total = int(page_size * phys_pages // (1024 * 1024))
        avail = int(page_size * avail_pages // (1024 * 1024))
        return total, avail
    except (ValueError, OSError, AttributeError):
        return 2048, 1024  # conservative defaults


def _detect_free_disk_mb(path: Path) -> int:
    """Return free disk space in MB at ``path`` (0 when undetectable)."""
    try:
        usage = shutil.disk_usage(path)
        return int(usage.free // (1024 * 1024))
    except OSError:
        return 0


def _detect_gpus() -> tuple[GPUInfo, ...]:
    """Best-effort GPU detection. Never raises; returns empty on failure."""
    gpus: list[GPUInfo] = []
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1].isdigit():
                    gpus.append(GPUInfo(name=parts[0], memory_mb=int(parts[1]), backend="cuda"))
    except (OSError, subprocess.SubprocessError):
        pass
    return tuple(gpus)


def classify_tier(cpu_count: int, total_ram_mb: int, has_gpu: bool) -> HardwareTier:
    """Classify a machine into a :class:`HardwareTier`.

    Rules (RAM in MB):
      * < 4 cores or < 4096 MB         -> LOW
      * < 8 cores or < 8192 MB         -> MEDIUM
      * >= 16 cores and >= 32768 MB and GPU -> WORKSTATION
      * otherwise                       -> HIGH
    """
    if cpu_count < 4 or total_ram_mb < 4096:
        return HardwareTier.LOW
    if cpu_count < 8 or total_ram_mb < 8192:
        return HardwareTier.MEDIUM
    if cpu_count >= 16 and total_ram_mb >= 32768 and has_gpu:
        return HardwareTier.WORKSTATION
    return HardwareTier.HIGH


def detect_hardware(workdir: Path | None = None) -> HardwareInfo:
    """Probe the host machine and return a :class:`HardwareInfo` snapshot.

    Parameters
    ----------
    workdir:
        Directory whose drive is used for free-disk measurement. Defaults to
        the current working directory.
    """
    cpu_count = os.cpu_count() or 1
    total_ram_mb, available_ram_mb = _detect_ram_mb()
    free_disk_mb = _detect_free_disk_mb(workdir if workdir is not None else Path.cwd())
    gpus = _detect_gpus()
    tier = classify_tier(cpu_count, total_ram_mb, bool(gpus))
    return HardwareInfo(
        cpu_count=cpu_count,
        total_ram_mb=total_ram_mb,
        available_ram_mb=available_ram_mb,
        free_disk_mb=free_disk_mb,
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=sys.version.split()[0],
        gpus=gpus,
        tier=tier,
    )
