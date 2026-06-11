"""Tests for the Sentinel SAFE and Landsat readers (need rasterio)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("rasterio")

from changemaster.core.exceptions import ImageReadError, MetadataError
from changemaster.io_engine.base_reader import open_image
from changemaster.io_engine.landsat_reader import LandsatReader, parse_mtl
from changemaster.io_engine.safe_reader import SafeReader, parse_manifest


class TestSafeReader:
    def test_can_read_safe_directory(self, safe_product: Path) -> None:
        assert SafeReader.can_read(safe_product)
        assert not SafeReader.can_read(safe_product / "manifest.safe")

    def test_parse_manifest(self, safe_product: Path) -> None:
        info = parse_manifest(safe_product / "manifest.safe")
        assert len(info["data_files"]) == 2
        assert info["platform"] == "SENTINEL-2A"
        assert info["start_time"] == datetime(2024, 1, 15, 8, 33, 1, 24000)

    def test_parse_manifest_bad_xml(self, tmp_path: Path) -> None:
        bad = tmp_path / "manifest.safe"
        bad.write_text("<not-closed", encoding="utf-8")
        with pytest.raises(MetadataError):
            parse_manifest(bad)

    def test_discover_bands(self, safe_product: Path) -> None:
        bands = SafeReader.discover_bands(safe_product)
        assert set(bands) == {"B02", "B03"}

    def test_open_default_band(self, safe_product: Path) -> None:
        with SafeReader(safe_product) as reader:
            meta = reader.metadata
            assert meta.driver == "SAFE"
            assert reader.band == "B02"
            assert meta.sensor_id == "sentinel2"
            assert meta.acquisition_datetime is not None
            assert meta.extra["platform"] == "SENTINEL-2A"
            assert meta.georef.is_georeferenced
            assert reader.read().shape == (1, 12, 14)

    def test_open_named_band_and_window(self, safe_product: Path) -> None:
        with SafeReader(safe_product, band="b03") as reader:
            assert reader.band == "B03"
            assert reader.read(window=(0, 0, 5, 6)).shape == (1, 5, 6)

    def test_unknown_band_raises(self, safe_product: Path) -> None:
        with pytest.raises(ImageReadError):
            SafeReader(safe_product, band="B99").open()

    def test_open_via_factory(self, safe_product: Path) -> None:
        with open_image(safe_product) as reader:
            assert isinstance(reader, SafeReader)

    def test_non_directory_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "fake.SAFE"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ImageReadError):
            SafeReader(f).open()


class TestParseMTL:
    def test_parses_values_and_strips_quotes(self) -> None:
        mtl = parse_mtl('GROUP = X\n  SPACECRAFT_ID = "LANDSAT_8"\n  CLOUD_COVER = 3.12\nEND_GROUP = X\nEND\n')
        assert mtl["SPACECRAFT_ID"] == "LANDSAT_8"
        assert mtl["CLOUD_COVER"] == "3.12"
        assert "GROUP" not in mtl


class TestLandsatReader:
    def test_can_read(self, landsat_scene: Path, landsat_tar: Path) -> None:
        assert LandsatReader.can_read(landsat_scene)
        assert LandsatReader.can_read(landsat_tar)
        assert not LandsatReader.can_read(landsat_scene / "nonexistent.txt")

    def test_open_directory_default_band(self, landsat_scene: Path) -> None:
        with LandsatReader(landsat_scene) as reader:
            meta = reader.metadata
            assert meta.driver == "Landsat"
            assert reader.band == "B4"
            assert meta.sensor_id == "landsat8"
            assert meta.acquisition_datetime == datetime(2024, 2, 20, 8, 15, 32)
            assert meta.extra["mtl"]["CLOUD_COVER"] == "3.12"
            assert sorted(meta.extra["available_bands"]) == ["B4", "B5"]
            assert reader.read().shape == (1, 12, 14)

    def test_open_named_band(self, landsat_scene: Path) -> None:
        with LandsatReader(landsat_scene, band="b5") as reader:
            assert reader.band == "B5"

    def test_unknown_band_raises(self, landsat_scene: Path) -> None:
        with pytest.raises(ImageReadError):
            LandsatReader(landsat_scene, band="B99").open()

    def test_open_tar_archive(self, landsat_tar: Path) -> None:
        with LandsatReader(landsat_tar) as reader:
            assert reader.metadata.sensor_id == "landsat8"
            assert reader.read().shape == (1, 12, 14)

    def test_open_tar_via_factory(self, landsat_tar: Path) -> None:
        with open_image(landsat_tar) as reader:
            assert isinstance(reader, LandsatReader)

    def test_missing_mtl_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "scene"
        empty.mkdir()
        with pytest.raises(MetadataError):
            LandsatReader(empty).open()

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ImageReadError):
            LandsatReader(tmp_path / "missing.tar").open()
