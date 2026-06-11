"""Tests for the simple Pillow reader and the reader registry/factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import FormatNotSupportedError, ImageReadError
from changemaster.io_engine.base_reader import open_image, reader_registry
from changemaster.io_engine.simple_reader import SimpleImageReader


class TestRegistry:
    def test_builtin_readers_registered(self) -> None:
        names = {r.format_name for r in reader_registry.readers}
        assert {"PNG/JPEG/BMP", "GeoTIFF/BigTIFF/JPEG2000/ENVI", "HDF5", "NetCDF",
                "Sentinel SAFE", "Landsat"} <= names

    def test_format_report_structure(self) -> None:
        report = reader_registry.format_report()
        assert all({"format", "extensions", "available", "requires"} <= set(r) for r in report)

    def test_find_reader_png(self) -> None:
        assert reader_registry.find_reader(Path("a.png")) is SimpleImageReader

    def test_find_reader_unknown_raises(self) -> None:
        with pytest.raises(FormatNotSupportedError):
            reader_registry.find_reader(Path("file.unknown_ext"))


class TestSimpleReader:
    def test_read_png_full(self, png_file: Path, rgb_array: np.ndarray) -> None:
        with open_image(png_file) as reader:
            meta = reader.metadata
            assert (meta.band_count, meta.height, meta.width) == rgb_array.shape
            assert meta.dtype == "uint8"
            assert meta.band_names == ["Red", "Green", "Blue"]
            assert not meta.georef.is_georeferenced
            np.testing.assert_array_equal(reader.read(), rgb_array)

    def test_read_jpeg(self, jpeg_file: Path) -> None:
        with open_image(jpeg_file) as reader:
            data = reader.read()
            assert data.shape[0] == 3
            assert data.dtype == np.uint8

    def test_band_selection(self, png_file: Path, rgb_array: np.ndarray) -> None:
        with open_image(png_file) as reader:
            np.testing.assert_array_equal(reader.read(bands=[2]), rgb_array[1:2])

    def test_window_read(self, png_file: Path, rgb_array: np.ndarray) -> None:
        with open_image(png_file) as reader:
            data = reader.read(window=(4, 6, 10, 12))
            np.testing.assert_array_equal(data, rgb_array[:, 4:14, 6:18])

    def test_invalid_band_raises(self, png_file: Path) -> None:
        with open_image(png_file) as reader:
            with pytest.raises(ImageReadError):
                reader.read(bands=[9])

    def test_grayscale_png(self, tmp_path: Path) -> None:
        from PIL import Image

        rng = np.random.default_rng(0)
        gray = rng.integers(0, 256, size=(10, 11), dtype=np.uint8)
        path = tmp_path / "gray.png"
        Image.fromarray(gray, mode="L").save(path)
        with open_image(path) as reader:
            assert reader.metadata.band_count == 1
            np.testing.assert_array_equal(reader.read()[0], gray)

    def test_corrupt_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"this is not a png")
        with pytest.raises(ImageReadError):
            SimpleImageReader(bad).open()

    def test_close_idempotent(self, png_file: Path) -> None:
        reader = SimpleImageReader(png_file)
        reader.open()
        reader.close()
        reader.close()
        # read() re-opens lazily after close
        assert reader.read().shape[0] == 3
