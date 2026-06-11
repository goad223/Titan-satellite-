#!/usr/bin/env python3
"""titan_inspect — inspect any supported satellite image file (CLI).

Usage::

    python scripts/titan_inspect.py <path> [--json]

Opens the file with the best available reader and prints its normalized
metadata plus the auto-detected sensor profile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from changemaster import open_image, sensor_registry
from changemaster.core.exceptions import ChangeMasterError


def inspect(path: Path) -> dict[str, object]:
    """Open ``path`` and return a metadata + sensor report dictionary."""
    with open_image(path) as reader:
        meta = reader.metadata
        sensor = (
            sensor_registry.get(meta.sensor_id)
            if meta.sensor_id is not None
            else sensor_registry.detect(path)
        )
        return {
            "metadata": meta.to_dict(),
            "estimated_size_mb": round(meta.estimated_size_mb(), 2),
            "sensor": {
                "sensor_id": sensor.sensor_id,
                "display_name": sensor.display_name,
                "display_name_ar": sensor.display_name_ar,
                "type": sensor.sensor_type,
                "resolution_m": sensor.default_resolution_m,
            },
        }


def format_text(report: dict[str, object]) -> str:
    """Render the inspection report as human-readable text."""
    meta = report["metadata"]
    sensor = report["sensor"]
    assert isinstance(meta, dict) and isinstance(sensor, dict)
    georef = meta.get("georef") or {}
    lines = [
        "Image inspection | فحص الصورة",
        "=" * 50,
        f"  Path:       {meta['path']}",
        f"  Driver:     {meta['driver']}",
        f"  Size:       {meta['width']} x {meta['height']} px, {meta['band_count']} band(s)",
        f"  Data type:  {meta['dtype']}",
        f"  NoData:     {meta['nodata']}",
        f"  CRS:        {georef.get('crs')}",
        f"  Transform:  {georef.get('transform')}",
        f"  Acquired:   {meta.get('acquisition_datetime')}",
        f"  Bands:      {', '.join(meta['band_names'])}",
        f"  Est. size:  {report['estimated_size_mb']} MB (uncompressed)",
        "",
        "Sensor | المستشعر",
        f"  {sensor['display_name']} | {sensor['display_name_ar']}",
        f"  id={sensor['sensor_id']}, type={sensor['type']}, "
        f"resolution={sensor['resolution_m']} m",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (1 on failure)."""
    parser = argparse.ArgumentParser(
        prog="titan_inspect",
        description="Inspect a satellite image and print normalized metadata.",
    )
    parser.add_argument("path", type=Path, help="image file or product directory")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    try:
        report = inspect(args.path)
    except ChangeMasterError as exc:
        print(f"Error | خطأ: {exc.bilingual()}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
