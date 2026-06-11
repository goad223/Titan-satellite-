"""Tests for the rasterio-backed RasterReader (skipped if rasterio missing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")

from changemaster.core.exceptions import ImageReadError
from changemaster.io_engine.base_reader import open_image
from changemaster.io_engine.raster_reader import RasterReader


class TestRasterReader:
    def test_metadata(self, geotiff_file: Path, gray_array: np.ndarray) -> None:
        with open_image(geotiff_file) as reader:
            meta = reader.metadata
            assert isinstance(reader, RasterReader)
            assert meta.driver == "GTiff"
            assert (meta.band_count, meta.height, meta.width) == gray_array.shape
            assert meta.dtype == "uint16"
            assert meta.nodata == 0
            assert meta.georef.is_georeferenced
            assert "32636" in (meta.georef.crs or "")
            assert meta.georef.transform is not None
            assert meta.georef.pixel_to_coords(0, 0) == (500000.0, 4100000.0)

    def test_full_read(self, geotiff_file: Path, gray_array: np.ndarray) -> None:
        with open_image(geotiff_file) as reader:
            np.testing.assert_array_equal(reader.read(), gray_array)

    def test_window_read(self, geotiff_file: Path, gray_array: np.ndarray) -> None:
        with open_image(geotiff_file) as reader:
            data = reader.read(window=(5, 3, 10, 12))
            np.testing.assert_array_equal(data, gray_array[:, 5:15, 3:15])

    def test_band_out_of_range_raises(self, geotiff_file: Path) -> None:
        with open_image(geotiff_file) as reader:
            with pytest.raises(ImageReadError):
                reader.read(bands=[5])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ImageReadError):
            RasterReader(tmp_path / "nope.tif").open()

    def test_close_idempotent(self, geotiff_file: Path) -> None:
        reader = RasterReader(geotiff_file)
        reader.open()
        reader.close()
        reader.close()

    def test_lazy_open_on_read(self, geotiff_file: Path, gray_array: np.ndarray) -> None:
        reader = RasterReader(geotiff_file)
        try:
            np.testing.assert_array_equal(reader.read(), gray_array)
        finally:
            reader.close()
