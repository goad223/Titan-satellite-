"""Tests for GeoTIFF writing and PNG export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import ImageWriteError
from changemaster.io_engine.metadata import GeoReference
from changemaster.io_engine.writer import export_png, write_geotiff


class TestWriteGeoTIFF:
    @pytest.fixture(autouse=True)
    def _require(self) -> None:
        pytest.importorskip("rasterio")

    def test_roundtrip_preserves_data_and_georef(self, tmp_path: Path, gray_array: np.ndarray) -> None:
        from changemaster.io_engine.raster_reader import RasterReader

        georef = GeoReference(
            crs="EPSG:32636", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0)
        )
        out = write_geotiff(tmp_path / "out.tif", gray_array, georef=georef, nodata=0)
        with RasterReader(out) as reader:
            meta = reader.metadata
            assert meta.nodata == 0
            assert "32636" in (meta.georef.crs or "")
            assert meta.georef.transform == georef.transform
            np.testing.assert_array_equal(reader.read(), gray_array)

    def test_writes_2d_array_as_single_band(self, tmp_path: Path) -> None:
        from changemaster.io_engine.raster_reader import RasterReader

        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        out = write_geotiff(tmp_path / "f.tif", data)
        with RasterReader(out) as reader:
            assert reader.metadata.band_count == 1
            np.testing.assert_array_equal(reader.read()[0], data)

    def test_creates_parent_dirs(self, tmp_path: Path, gray_array: np.ndarray) -> None:
        out = write_geotiff(tmp_path / "a" / "b" / "out.tif", gray_array)
        assert out.exists()

    def test_invalid_array_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ImageWriteError):
            write_geotiff(tmp_path / "x.tif", np.zeros((2, 2, 2, 2)))


class TestExportPNG:
    def test_grayscale_export(self, tmp_path: Path) -> None:
        from PIL import Image

        data = np.linspace(0, 1000, 100, dtype=np.float64).reshape(10, 10)
        out = export_png(tmp_path / "g.png", data)
        with Image.open(out) as img:
            assert img.mode == "L"
            assert img.size == (10, 10)

    def test_rgb_export(self, tmp_path: Path, rgb_array: np.ndarray) -> None:
        from PIL import Image

        out = export_png(tmp_path / "rgb.png", rgb_array)
        with Image.open(out) as img:
            assert img.mode == "RGB"
            assert img.size == (48, 32)

    def test_no_stretch_clips(self, tmp_path: Path) -> None:
        from PIL import Image

        data = np.array([[0, 128], [255, 999]], dtype=np.int32)
        out = export_png(tmp_path / "c.png", data, percentile_stretch=None)
        with Image.open(out) as img:
            arr = np.asarray(img)
        assert arr.max() == 255 and arr.min() == 0
        assert arr[0, 1] == 128

    def test_constant_band_handled(self, tmp_path: Path) -> None:
        data = np.full((8, 8), 42, dtype=np.uint8)
        out = export_png(tmp_path / "k.png", data)
        assert out.exists()

    def test_two_band_falls_back_to_first(self, tmp_path: Path) -> None:
        from PIL import Image

        data = np.zeros((2, 6, 7), dtype=np.uint8)
        out = export_png(tmp_path / "two.png", data)
        with Image.open(out) as img:
            assert img.mode == "L"

    def test_invalid_array_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ImageWriteError):
            export_png(tmp_path / "x.png", np.zeros((0, 0)))
