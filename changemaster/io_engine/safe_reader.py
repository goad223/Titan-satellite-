"""Reader for Sentinel SAFE products (directory with manifest.safe + JP2 bands).

Parses the ``manifest.safe`` XML to discover band image files (JPEG2000 for
Sentinel-2, TIFF measurements for Sentinel-1) and reads them via rasterio.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from changemaster.core.exceptions import ImageReadError, MetadataError
from changemaster.io_engine.base_reader import BaseImageReader, reader_registry
from changemaster.io_engine.metadata import ImageMetadata
from changemaster.io_engine.raster_reader import RasterReader, _georef_from_dataset, _import_rasterio

if TYPE_CHECKING:
    import numpy as np

_MANIFEST_NAME = "manifest.safe"
_BAND_RE = re.compile(r"_(B[0-9][0-9A]|TCI|AOT|WVP|SCL)(?:_\d+m)?$", re.IGNORECASE)
_DATetime_RE = re.compile(r"(\d{8}T\d{6})")


def parse_manifest(manifest_path: Path) -> dict[str, Any]:
    """Parse a ``manifest.safe`` XML file.

    Returns a dictionary with keys:

    * ``data_files``: list of product-relative file paths declared in the
      manifest's ``dataObjectSection``.
    * ``platform``: spacecraft name when present (e.g. ``"Sentinel-2A"``).
    * ``start_time``: acquisition start :class:`datetime` when present.
    """
    try:
        tree = ET.parse(manifest_path)
    except (ET.ParseError, OSError) as exc:
        raise MetadataError(
            f"Failed to parse SAFE manifest {manifest_path}: {exc}",
            f"فشل في تحليل ملف المانيفست {manifest_path}: {exc}",
        ) from exc
    root = tree.getroot()

    data_files: list[str] = []
    for file_loc in root.iter():
        if file_loc.tag.split("}")[-1] == "fileLocation":
            href = file_loc.get("href")
            if href:
                data_files.append(href.lstrip("./"))

    platform: str | None = None
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "familyName" and elem.text and "sentinel" in elem.text.lower():
            platform = elem.text.strip()
        elif tag == "number" and platform and elem.text:
            platform = f"{platform}-{elem.text.strip()}"
            break

    start_time: datetime | None = None
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "startTime" and elem.text:
            try:
                start_time = datetime.fromisoformat(elem.text.strip().rstrip("Z"))
            except ValueError:
                start_time = None
            break

    return {"data_files": data_files, "platform": platform, "start_time": start_time}


@reader_registry.register
class SafeReader(BaseImageReader):
    """Reader for Sentinel SAFE product directories.

    Parameters
    ----------
    path:
        Path to the ``.SAFE`` directory (or any directory containing
        ``manifest.safe``).
    band:
        Optional band identifier (e.g. ``"B04"``). Defaults to the first
        discovered band.
    """

    format_name: ClassVar[str] = "Sentinel SAFE"
    extensions: ClassVar[tuple[str, ...]] = (".safe",)
    required_package: ClassVar[str | None] = "rasterio"

    def __init__(self, path: Path | str, band: str | None = None) -> None:
        super().__init__(path)
        self.band: str | None = band.upper() if band else None
        self._band_files: dict[str, Path] = {}
        self._reader: RasterReader | None = None

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Match ``.SAFE`` directories or directories with a manifest.safe."""
        if not path.is_dir():
            return False
        return path.suffix.lower() == ".safe" or (path / _MANIFEST_NAME).exists()

    @staticmethod
    def discover_bands(product_dir: Path) -> dict[str, Path]:
        """Map band IDs (e.g. ``"B04"``) to image file paths in a SAFE product."""
        manifest = product_dir / _MANIFEST_NAME
        candidates: list[Path] = []
        if manifest.exists():
            info = parse_manifest(manifest)
            for rel in info["data_files"]:
                p = product_dir / Path(rel)
                if p.suffix.lower() in (".jp2", ".tif", ".tiff") and p.exists():
                    candidates.append(p)
        if not candidates:
            candidates = sorted(product_dir.rglob("*.jp2")) + sorted(product_dir.rglob("*.tiff"))

        bands: dict[str, Path] = {}
        for p in candidates:
            match = _BAND_RE.search(p.stem)
            if match:
                band_id = match.group(1).upper()
                bands.setdefault(band_id, p)
            else:
                bands.setdefault(p.stem.upper(), p)
        return bands

    def open(self) -> None:
        """Discover band files, open the selected band and build metadata."""
        _import_rasterio()  # fail early with a bilingual message
        if not self.path.is_dir():
            raise ImageReadError(
                f"SAFE product path is not a directory: {self.path}",
                f"مسار منتج SAFE ليس مجلداً: {self.path}",
            )
        self._band_files = self.discover_bands(self.path)
        if not self._band_files:
            raise ImageReadError(
                f"No band image files found in SAFE product {self.path}.",
                f"لم يتم العثور على ملفات نطاقات في منتج SAFE ‏{self.path}.",
            )
        band = self.band if self.band is not None else sorted(self._band_files)[0]
        if band not in self._band_files:
            raise ImageReadError(
                f"Band '{band}' not found in SAFE product {self.path}. "
                f"Available: {sorted(self._band_files)}",
                f"النطاق '{band}' غير موجود في منتج SAFE ‏{self.path}. "
                f"المتاح: {sorted(self._band_files)}",
            )
        self.band = band
        self._reader = RasterReader(self._band_files[band])
        self._reader.open()
        inner = self._reader.metadata

        manifest = self.path / _MANIFEST_NAME
        platform: str | None = None
        start_time: datetime | None = None
        if manifest.exists():
            info = parse_manifest(manifest)
            platform = info["platform"]
            start_time = info["start_time"]
        if start_time is None:
            dt_match = _DATetime_RE.search(self.path.name)
            if dt_match:
                try:
                    start_time = datetime.strptime(dt_match.group(1), "%Y%m%dT%H%M%S")
                except ValueError:
                    start_time = None

        self._metadata = ImageMetadata(
            path=self.path,
            driver="SAFE",
            width=inner.width,
            height=inner.height,
            band_count=inner.band_count,
            dtype=inner.dtype,
            nodata=inner.nodata,
            georef=inner.georef,
            acquisition_datetime=start_time,
            sensor_id="sentinel1" if "S1" in self.path.name.upper()[:3] else "sentinel2",
            band_names=[band],
            extra={
                "platform": platform,
                "available_bands": sorted(self._band_files),
                "band_file": str(self._band_files[band]),
            },
        )

    def close(self) -> None:
        """Close the underlying band reader (idempotent)."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def read(
        self,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Read the selected band's pixels; see base class."""
        if self._reader is None:
            self.open()
        assert self._reader is not None
        return self._reader.read(bands=bands, window=window)
