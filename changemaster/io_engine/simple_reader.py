"""Reader for common image formats (PNG/JPEG/BMP) via Pillow.

This reader always works without heavy geospatial dependencies and serves
as the guaranteed baseline I/O path of the application.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from changemaster.core.exceptions import ImageReadError
from changemaster.io_engine.base_reader import BaseImageReader, reader_registry
from changemaster.io_engine.metadata import GeoReference, ImageMetadata

if TYPE_CHECKING:
    import numpy as np


@reader_registry.register
class SimpleImageReader(BaseImageReader):
    """Pillow-backed reader for PNG, JPEG and BMP files."""

    format_name: ClassVar[str] = "PNG/JPEG/BMP"
    extensions: ClassVar[tuple[str, ...]] = (".png", ".jpg", ".jpeg", ".bmp")
    required_package: ClassVar[str | None] = "PIL"

    def __init__(self, path: Path | str) -> None:
        super().__init__(path)
        self._array: "np.ndarray | None" = None

    def open(self) -> None:
        """Load the image fully into memory and build metadata."""
        import numpy as np
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(self.path) as img:
                array = np.asarray(img)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ImageReadError(
                f"Failed to read image {self.path}: {exc}",
                f"فشل في قراءة الصورة {self.path}: {exc}",
            ) from exc

        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        else:
            array = np.transpose(array, (2, 0, 1))
        self._array = array
        bands, height, width = array.shape
        names = {1: ["Gray"], 3: ["Red", "Green", "Blue"], 4: ["Red", "Green", "Blue", "Alpha"]}
        self._metadata = ImageMetadata(
            path=self.path,
            driver=self.path.suffix.lstrip(".").upper() or "PIL",
            width=width,
            height=height,
            band_count=bands,
            dtype=str(array.dtype),
            georef=GeoReference(),
            band_names=names.get(bands, []),
        )

    def close(self) -> None:
        """Release the in-memory pixel array."""
        self._array = None

    def read(
        self,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Return pixels as ``(bands, height, width)``; see base class."""
        if self._array is None:
            self.open()
        assert self._array is not None
        data = self._array
        if bands is not None:
            self._validate_bands(bands, data.shape[0])
            data = data[[b - 1 for b in bands], :, :]
        if window is not None:
            row_off, col_off, height, width = window
            data = data[:, row_off : row_off + height, col_off : col_off + width]
        return data.copy()

    def _validate_bands(self, bands: list[int], band_count: int) -> None:
        """Raise :class:`ImageReadError` for out-of-range band indices."""
        for b in bands:
            if b < 1 or b > band_count:
                raise ImageReadError(
                    f"Band index {b} out of range 1..{band_count} for {self.path}.",
                    f"رقم النطاق {b} خارج المدى 1..{band_count} للملف {self.path}.",
                )
