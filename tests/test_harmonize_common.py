"""Tests for preprocessing._common helpers and rasterio reprojection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.hardware import HardwareInfo, HardwareTier
from changemaster.io_engine.metadata import GeoReference, ImageMetadata
from changemaster.preprocessing._common import (
    adaptive_tile_size,
    adaptive_window_count,
    normalize_to_uint8,
    require_cv2,
    require_scipy,
    to_float32,
)
from changemaster.preprocessing.harmonize import harmonize_arrays, reproject_to_reference


def _hw(tier: HardwareTier) -> HardwareInfo:
    return HardwareInfo(
        cpu_count=4,
        total_ram_mb=8192,
        available_ram_mb=4096,
        free_disk_mb=10000,
        os_name="Linux",
        os_version="x",
        python_version="3.12",
        tier=tier,
    )


class TestCommon:
    def test_require_cv2_returns_module(self) -> None:
        pytest.importorskip("cv2")
        assert hasattr(require_cv2(), "warpAffine")

    def test_require_scipy_returns_module(self) -> None:
        pytest.importorskip("scipy")
        assert require_scipy().__name__ == "scipy"

    def test_adaptive_tile_size_by_tier(self) -> None:
        assert adaptive_tile_size(_hw(HardwareTier.LOW)) == 512
        assert adaptive_tile_size(_hw(HardwareTier.HIGH)) == 2048

    def test_adaptive_window_count_by_tier(self) -> None:
        assert adaptive_window_count(_hw(HardwareTier.LOW)) == 9
        assert adaptive_window_count(_hw(HardwareTier.MEDIUM)) == 16
        assert adaptive_window_count(_hw(HardwareTier.WORKSTATION)) == 25

    def test_adaptive_defaults_detect_hardware(self) -> None:
        assert adaptive_tile_size(None) >= 512
        assert adaptive_window_count(None) >= 9

    def test_to_float32(self) -> None:
        out = to_float32(np.array([1, 2, 3], dtype=np.uint16))
        assert out.dtype == np.float32

    def test_normalize_to_uint8_stretch(self) -> None:
        band = np.linspace(0, 1000, 100).reshape(10, 10)
        out = normalize_to_uint8(band)
        assert out.dtype == np.uint8
        assert out.max() == 255

    def test_normalize_to_uint8_flat_and_nan(self) -> None:
        assert normalize_to_uint8(np.full((4, 4), 7.0)).max() == 0
        assert normalize_to_uint8(np.full((4, 4), np.nan)).max() == 0


class TestReproject:
    def test_reproject_to_reference_grid(self, geotiff_file: Path) -> None:
        rasterio = pytest.importorskip("rasterio")
        from changemaster.io_engine.base_reader import open_image

        with open_image(geotiff_file) as reader:
            meta = reader.metadata
            original = reader.read()
        out = reproject_to_reference(str(geotiff_file), meta)
        assert out.shape == (1, meta.height, meta.width)
        assert np.allclose(out[0], original[0], atol=1.0)
        _ = rasterio

    def test_reproject_requires_georef(self, tmp_path: Path) -> None:
        pytest.importorskip("rasterio")
        from changemaster.core.exceptions import PreprocessingError

        meta = ImageMetadata(
            path=tmp_path / "x.png",
            driver="PNG",
            width=10,
            height=10,
            band_count=1,
            dtype="uint8",
        )
        with pytest.raises(PreprocessingError):
            reproject_to_reference("whatever.tif", meta)

    def test_reproject_unknown_resampling(self, geotiff_file: Path) -> None:
        pytest.importorskip("rasterio")
        from changemaster.core.exceptions import PreprocessingError
        from changemaster.io_engine.base_reader import open_image

        with open_image(geotiff_file) as reader:
            meta = reader.metadata
        with pytest.raises(PreprocessingError):
            reproject_to_reference(str(geotiff_file), meta, resampling="lanczos")


class TestHarmonizeResampling:
    def test_pixel_size_ratio_resampling(self) -> None:
        pytest.importorskip("cv2")
        # Moving image at 20 m pixels, reference at 10 m -> ratio 2.
        ref_meta = ImageMetadata(
            path=Path("ref.tif"),
            driver="GTiff",
            width=40,
            height=40,
            band_count=1,
            dtype="float64",
            georef=GeoReference(crs="EPSG:32636", transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)),
            band_names=["Gray"],
        )
        mov_meta = ImageMetadata(
            path=Path("mov.tif"),
            driver="GTiff",
            width=20,
            height=20,
            band_count=1,
            dtype="float64",
            georef=GeoReference(crs="EPSG:32636", transform=(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)),
            band_names=["Gray"],
        )
        ref = np.zeros((1, 40, 40))
        mov = np.zeros((1, 20, 20))
        pair = harmonize_arrays(ref, mov, ref_meta, mov_meta)
        assert pair.reference.shape == pair.moving.shape
        assert any("resampled" in w or "معاينة" in w for w in pair.warnings)
