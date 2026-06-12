#!/usr/bin/env python3
"""titan_info — hardware report and supported-format table (CLI).

Usage::

    python scripts/titan_info.py [--json]

Prints the detected hardware tier, resource summary, and a table of all
registered image formats with their availability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from changemaster import __app_name__, __version__, reader_registry
from changemaster.core.hardware import detect_hardware


def build_report() -> dict[str, object]:
    """Collect hardware info and format availability into one dictionary."""
    hardware = detect_hardware()
    return {
        "app": __app_name__,
        "version": __version__,
        "hardware": hardware.to_dict(),
        "recommended": {
            "workers": hardware.recommended_workers,
            "tile_size": hardware.recommended_tile_size,
            "max_in_memory_mb": hardware.max_in_memory_mb,
        },
        "formats": reader_registry.format_report(),
    }


def format_text(report: dict[str, object]) -> str:
    """Render the report as a human-readable text block."""
    hw = report["hardware"]
    rec = report["recommended"]
    assert isinstance(hw, dict) and isinstance(rec, dict)
    lines: list[str] = [
        f"{report['app']} v{report['version']}",
        "=" * 50,
        "Hardware | العتاد",
        f"  OS:        {hw['os_name']} {hw['os_version']}",
        f"  Python:    {hw['python_version']}",
        f"  CPU cores: {hw['cpu_count']}",
        f"  RAM:       {hw['total_ram_mb']} MB total / {hw['available_ram_mb']} MB free",
        f"  Disk free: {hw['free_disk_mb']} MB",
        f"  GPUs:      {', '.join(g['name'] for g in hw['gpus']) or 'none detected'}",
        f"  Tier:      {hw['tier']}",
        "",
        "Recommended settings | الإعدادات الموصى بها",
        f"  Workers:        {rec['workers']}",
        f"  Tile size:      {rec['tile_size']} px",
        f"  Max in-memory:  {rec['max_in_memory_mb']} MB",
        "",
        "Formats | الصيغ",
        f"  {'Format':<32} {'Available':<10} {'Requires':<10} Extensions",
    ]
    formats = report["formats"]
    assert isinstance(formats, list)
    for fmt in formats:
        lines.append(
            f"  {fmt['format']:<32} "
            f"{'yes' if fmt['available'] else 'NO':<10} "
            f"{fmt['requires'] or '-':<10} "
            f"{', '.join(fmt['extensions'])}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="titan_info",
        description="Report hardware capabilities and supported image formats.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
