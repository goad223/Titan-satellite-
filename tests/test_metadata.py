"""Tests for changemaster.io_engine.metadata."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from changemaster.core.exceptions import MetadataError
from changemaster.io_engine.metadata import GeoReference, ImageMetadata


class TestGeoReference:
    def test_empty_not_georeferenced(self) -> None:
        assert not GeoReference().is_georeferenced

    def test_full_is_georeferenced(self) -> None:
        ref = GeoReference(crs="EPSG:32636", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0))
        assert ref.is_georeferenced

    def test_pixel_to_coords(self) -> None:
        ref = GeoReference(crs="EPSG:32636", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0))
        assert ref.pixel_to_coords(0, 0) == (500000.0, 4100000.0)
        assert ref.pixel_to_coords(2, 3) == (500030.0, 4099980.0)

    def test_pixel_to_coords_without_transform_raises(self) -> None:
        with pytest.raises(MetadataError):
            GeoReference().pixel_to_coords(0, 0)


class TestImageMetadata:
    def _meta(self, **kwargs: object) -> ImageMetadata:
        defaults: dict = {
            "path": Path("x.tif"),
            "driver": "GTiff",
            "width": 100,
            "height": 50,
            "band_count": 3,
            "dtype": "uint16",
        }
        defaults.update(kwargs)
        return ImageMetadata(**defaults)

    def test_shape_and_pixel_count(self) -> None:
        meta = self._meta()
        assert meta.shape == (3, 50, 100)
        assert meta.pixel_count == 5000

    def test_default_band_names(self) -> None:
        assert self._meta().band_names == ["Band 1", "Band 2", "Band 3"]

    def test_custom_band_names_kept(self) -> None:
        meta = self._meta(band_names=["R", "G", "B"])
        assert meta.band_names == ["R", "G", "B"]

    def test_estimated_size_mb(self) -> None:
        meta = self._meta(width=1024, height=1024, band_count=1, dtype="uint8")
        assert meta.estimated_size_mb() == pytest.approx(1.0)

    @pytest.mark.parametrize("kwargs", [{"width": 0}, {"height": -2}, {"band_count": 0}])
    def test_invalid_dimensions_raise(self, kwargs: dict) -> None:
        with pytest.raises(MetadataError):
            self._meta(**kwargs)

    def test_to_dict_serialises_path_and_datetime(self) -> None:
        import json

        meta = self._meta(acquisition_datetime=datetime(2024, 1, 15, 8, 33, 1))
        data = meta.to_dict()
        assert data["path"] == "x.tif"
        assert data["acquisition_datetime"] == "2024-01-15T08:33:01"
        json.dumps(data)
