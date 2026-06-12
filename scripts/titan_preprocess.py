#!/usr/bin/env python3
"""titan_preprocess — run the Phase-2 preprocessing pipeline (CLI).

Usage::

    python scripts/titan_preprocess.py <reference> <moving> --workdir out/
        [--mode optical|sar] [--resume] [--speckle refined_lee|frost|gamma_map]
        [--sun-elevation DEG --sun-azimuth DEG]
        [--band-roles blue=1,green=2,red=3,nir=4] [--looks N] [--json]

Runs quality gating, harmonization, co-registration, masking and
radiometric normalization (optical) or calibration + speckle filtering
(SAR), with checkpoints and a JSON report in the working directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from changemaster.core.exceptions import ChangeMasterError
from changemaster.preprocessing import PreprocessingPipeline


def parse_band_roles(text: str | None) -> dict[str, int] | None:
    """Parse ``"blue=1,green=2"`` style role mappings."""
    if not text:
        return None
    roles: dict[str, int] = {}
    for chunk in text.split(","):
        if "=" not in chunk:
            raise ValueError(f"Invalid band role '{chunk}'; expected role=index.")
        role, idx = chunk.split("=", 1)
        roles[role.strip().lower()] = int(idx)
    return roles


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="titan_preprocess",
        description="ChangeMaster Phase-2 preprocessing pipeline | أنبوب المعالجة المسبقة",
    )
    parser.add_argument("reference", type=Path, help="Reference image path")
    parser.add_argument("moving", type=Path, help="Moving (secondary) image path")
    parser.add_argument(
        "--workdir", type=Path, default=Path("preprocess_out"), help="Checkpoint/report directory"
    )
    parser.add_argument("--mode", choices=("optical", "sar"), default="optical")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument(
        "--speckle",
        choices=("refined_lee", "frost", "gamma_map"),
        default="refined_lee",
        help="SAR speckle filter",
    )
    parser.add_argument("--sun-elevation", type=float, default=None, help="Sun elevation (deg)")
    parser.add_argument("--sun-azimuth", type=float, default=None, help="Sun azimuth (deg)")
    parser.add_argument(
        "--band-roles",
        type=str,
        default=None,
        help="Spectral roles, e.g. blue=1,green=2,red=3,nir=4,swir=5",
    )
    parser.add_argument("--looks", type=float, default=1.0, help="SAR equivalent looks")
    parser.add_argument(
        "--reflectance-scale",
        type=float,
        default=1.0,
        help="DN-to-reflectance divisor for spectral masking (e.g. 10000)",
    )
    parser.add_argument(
        "--target-rmse", type=float, default=1.0, help="Registration RMSE target (px)"
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        roles = parse_band_roles(args.band_roles)
        pipeline = PreprocessingPipeline(
            workdir=args.workdir,
            mode=args.mode,
            speckle_method=args.speckle,
            target_rmse_px=args.target_rmse,
        )
        report = pipeline.run(
            args.reference,
            args.moving,
            resume=args.resume,
            sun_elevation_deg=args.sun_elevation,
            sun_azimuth_deg=args.sun_azimuth,
            band_roles=roles,
            looks=args.looks,
            reflectance_scale=args.reflectance_scale,
        )
    except ChangeMasterError as exc:
        print(f"Error | خطأ: {exc.bilingual()}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error | خطأ: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
