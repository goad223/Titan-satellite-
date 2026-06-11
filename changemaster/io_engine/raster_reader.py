"""Reader for GeoTIFF/BigTIFF/JPEG2000/ENVI rasters via rasterio (lazy import).

When rasterio is not installed the reader stays registered but reports
itself unavailable; attempting to open raises
:class:`~changemaster.core.exceptions.DependencyMissingError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from changemaster.core.exceptions import DependencyMissingError, ImageReadError
from changemaster.io_engine.base_reader import BaseImageReader, reader_registry
from changemaster.io_engine.metadata import GeoReference, ImageMetadata

if TYPE_CHECKING:
    import numpy as np


def _import_rasterio() -> Any:
    """Import and return rasterio, raising a bilingual error when missing."""
    try:
        import rasterio
    except ImportError as exc:
        raise DependencyMissingError(
            "rasterio",
            "GeoTIFF/JPEG2000/ENVI reading",
            "قراءة GeoTIFF/JPEG2000/ENVI",
        ) from exc
    return rasterio


def _georef_from_dataset(dataset: Any) -> GeoReference:
    """Build a :class:`GeoReference` from an open rasterio dataset."""
    crs = dataset.crs.to_string() if dataset.crs else None
    transform = None
    if dataset.transform is not None and not dataset.transform.is_identity:
        t = dataset.transform
        transform = (t.a, t.b, t.c, t.d, t.e, t.f)
    return GeoReference(crs=crs, transform=transform)


@reader_registry.register
class RasterReader(BaseImageReader):
    """rasterio-backed reader for geospatial raster formats."""

    format_name: ClassVar[str] = "GeoTIFF/BigTIFF/JPEG2000/ENVI"
    extensions: ClassVar[tuple[str, ...]] = (".tif", ".tiff", ".jp2", ".img", ".dat", ".hdr", ".vrt")
    required_package: ClassVar[str | None] = "rasterio"

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self._dataset: Any = None

    def open(self) -> None:
        """Open the dataset with rasterio and build normalized metadata."""
        rasterio = _import_rasterio()
        try:
            self._dataset = rasterio.open(self.path)
        except Exception as exc:  # rasterio raises various error types
            raise ImageReadError(
                f"Failed to open raster {self.path}: {exc}",
                f"فشل في فتح الملف النقطي {self.path}: {exc}",
            ) from exc
        ds = self._dataset
        descriptions = [d for d in (ds.descriptions or []) if d]
        self._metadata = ImageMetadata(
            path=self.path,
            driver=ds.driver,
            width=ds.width,
            height=ds.height,
            band_count=ds.count,
            dtype=str(ds.dtypes[0]),
            nodata=ds.nodata,
            georef=_georef_from_dataset(ds),
            band_names=descriptions if len(descriptions) == ds.count else [],
            extra=dict(ds.tags()),
        )

    def close(self) -> None:
        """Close the underlying rasterio dataset (idempotent)."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def read(
        self,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Read pixels as ``(bands, height, width)``; see base class."""
        if self._dataset is None:
            self.open()
        assert self._dataset is not None
        rasterio = _import_rasterio()
        from rasterio.windows import Window

        indexes = bands if bands is not None else list(range(1, self._dataset.count + 1))
        for b in indexes:
            if b < 1 or b > self._dataset.count:
                raise ImageReadError(
                    f"Band index {b} out of range 1..{self._dataset.count} for {self.path}.",
                    f"رقم النطاق {b} خارج المدى 1..{self._dataset.count} للملف {self.path}.",
                )
        rio_window = None
        if window is not None:
            row_off, col_off, height, width = window
            rio_window = Window(col_off, row_off, width, height)
        try:
            return self._dataset.read(indexes, window=rio_window)
        except Exception as exc:
            raise ImageReadError(
                f"Failed to read pixels from {self.path}: {exc}",
                f"فشل في قراءة البكسلات من {self.path}: {exc}",
            ) from exc
