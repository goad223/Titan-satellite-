"""Unified image metadata model shared by all readers.

:class:`ImageMetadata` is the single normalized description of any raster
source, independent of the underlying format library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from changemaster.core.exceptions import MetadataError


@dataclass(frozen=True)
class GeoReference:
    """Georeferencing information for a raster.

    Attributes
    ----------
    crs:
        Coordinate reference system as a WKT/EPSG/PROJ string
        (e.g. ``"EPSG:32636"``), or ``None`` when not georeferenced.
    transform:
        Affine transform coefficients ``(a, b, c, d, e, f)`` mapping pixel
        coordinates to CRS coordinates: ``x = a*col + b*row + c``,
        ``y = d*col + e*row + f``. ``None`` when unavailable.
    """

    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None

    @property
    def is_georeferenced(self) -> bool:
        """True when both a CRS and an affine transform are present."""
        return self.crs is not None and self.transform is not None

    def pixel_to_coords(self, row: int, col: int) -> tuple[float, float]:
        """Convert a pixel (row, col) to CRS (x, y) coordinates.

        Raises :class:`MetadataError` when no transform is available.
        """
        if self.transform is None:
            raise MetadataError(
                "Cannot convert pixel to coordinates: no affine transform.",
                "تعذر تحويل البكسل إلى إحداثيات: لا يوجد تحويل أفيني.",
            )
        a, b, c, d, e, f = self.transform
        x = a * col + b * row + c
        y = d * col + e * row + f
        return x, y


@dataclass
class ImageMetadata:
    """Normalized raster metadata shared by every reader implementation.

    Attributes
    ----------
    path:
        Source file or directory path.
    driver:
        Short name of the reader/format (e.g. ``"GTiff"``, ``"PNG"``).
    width / height:
        Raster dimensions in pixels.
    band_count:
        Number of bands/channels.
    dtype:
        NumPy dtype name of pixel values (e.g. ``"uint16"``).
    nodata:
        No-data value if defined.
    georef:
        :class:`GeoReference` (always present, may be empty).
    acquisition_datetime:
        Image acquisition time when known.
    sensor_id:
        Detected sensor identifier (e.g. ``"sentinel2"``), if known.
    band_names:
        Human-readable band names; defaults to ``["Band 1", ...]``.
    extra:
        Format-specific key/value metadata (tags, MTL fields, ...).
    """

    path: Path
    driver: str
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: float | None = None
    georef: GeoReference = field(default_factory=GeoReference)
    acquisition_datetime: datetime | None = None
    sensor_id: str | None = None
    band_names: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise MetadataError(
                f"Invalid raster dimensions {self.width}x{self.height} for {self.path}.",
                f"أبعاد غير صالحة {self.width}x{self.height} للملف {self.path}.",
            )
        if self.band_count <= 0:
            raise MetadataError(
                f"Invalid band count {self.band_count} for {self.path}.",
                f"عدد نطاقات غير صالح {self.band_count} للملف {self.path}.",
            )
        if not self.band_names:
            self.band_names = [f"Band {i + 1}" for i in range(self.band_count)]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Raster shape as ``(bands, height, width)``."""
        return (self.band_count, self.height, self.width)

    @property
    def pixel_count(self) -> int:
        """Total pixels per band (``width * height``)."""
        return self.width * self.height

    def estimated_size_mb(self) -> float:
        """Estimated uncompressed in-memory size of the full raster in MB."""
        import numpy as np

        itemsize = np.dtype(self.dtype).itemsize
        return self.pixel_count * self.band_count * itemsize / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        data = asdict(self)
        data["path"] = str(self.path)
        if self.acquisition_datetime is not None:
            data["acquisition_datetime"] = self.acquisition_datetime.isoformat()
        return data
