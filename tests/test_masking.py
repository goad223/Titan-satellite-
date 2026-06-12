"""Tests for masking: clouds, shadows, snow/water, nodata and the combiner."""

from __future__ import annotations

import numpy as np
import pytest

from changemaster.core.exceptions import MaskingError
from changemaster.preprocessing.masking.cloud import (
    decode_qa_pixel,
    decode_scl,
    detect_clouds,
    detect_clouds_spectral,
)
from changemaster.preprocessing.masking.combiner import (
    CODE_CLOUD,
    CODE_NODATA,
    CODE_SHADOW,
    CODE_SNOW,
    CODE_VALID,
    CODE_WATER,
    combine_masks,
)
from changemaster.preprocessing.masking.nodata import (
    detect_edges_nodata,
    detect_nodata,
    detect_saturation,
)
from changemaster.preprocessing.masking.shadow import detect_shadows, project_cloud_shadow
from changemaster.preprocessing.masking.snow_water import (
    detect_snow,
    detect_water,
    mndwi,
    ndsi,
    ndwi,
)


class TestSCL:
    def test_classes_decoded(self) -> None:
        scl = np.array([[0, 3, 6], [8, 9, 10], [11, 4, 5]])
        result = decode_scl(scl)
        assert result.nodata[0, 0]
        assert result.shadow[0, 1]
        assert result.water[0, 2]
        assert result.cloud[1].all()
        assert result.snow[2, 0]
        assert not result.cloud[2, 1]
        assert result.source == "scl"


class TestQAPixel:
    def test_bits_decoded(self) -> None:
        qa = np.zeros((2, 3), dtype=np.uint16)
        qa[0, 0] = 1 << 3  # cloud
        qa[0, 1] = 1 << 4  # shadow
        qa[0, 2] = 1 << 5  # snow
        qa[1, 0] = 1 << 7  # water
        qa[1, 1] = 1 << 0  # fill
        result = decode_qa_pixel(qa)
        assert result.cloud[0, 0]
        assert result.shadow[0, 1]
        assert result.snow[0, 2]
        assert result.water[1, 0]
        assert result.nodata[1, 1]


class TestSpectralClouds:
    def test_bright_block_detected(self) -> None:
        blue = np.full((40, 40), 0.05)
        blue[5:15, 5:15] = 0.6
        mask = detect_clouds_spectral(blue, red=blue, nir=blue)
        assert mask[10, 10]
        assert not mask[30, 30]

    def test_invalid_scale_raises(self) -> None:
        with pytest.raises(MaskingError):
            detect_clouds_spectral(np.zeros((5, 5)), reflectance_scale=0)

    def test_cirrus_or_logic(self) -> None:
        blue = np.zeros((10, 10))
        cirrus = np.zeros((10, 10))
        cirrus[0, 0] = 0.05
        mask = detect_clouds_spectral(blue, cirrus=cirrus)
        assert mask[0, 0]

    def test_fusion_with_scl(self) -> None:
        blue = np.zeros((3, 3))
        blue[0, 0] = 0.9
        scl = np.zeros((3, 3))
        scl[2, 2] = 9
        result = detect_clouds({"blue": blue, "red": blue, "nir": blue}, scl=scl)
        assert result.source == "scl+spectral"
        assert result.cloud[0, 0] and result.cloud[2, 2]

    def test_no_inputs_raises(self) -> None:
        with pytest.raises(MaskingError):
            detect_clouds({})


class TestShadow:
    def test_projection_direction(self) -> None:
        cloud = np.zeros((50, 50), dtype=bool)
        cloud[10, 25] = True
        # Sun from the north (azimuth 0): shadow falls south (larger row).
        shadow = project_cloud_shadow(cloud, 45.0, 0.0, 100.0, 1000.0)
        rows = np.argwhere(shadow)
        assert rows.size > 0
        assert rows[0][0] > 10

    def test_invalid_elevation_raises(self) -> None:
        with pytest.raises(MaskingError):
            project_cloud_shadow(np.zeros((5, 5), dtype=bool), 0.0, 0.0, 10.0, 100.0)

    def test_detect_shadows_links_dark_pixels(self) -> None:
        cloud = np.zeros((60, 60), dtype=bool)
        cloud[10:15, 28:33] = True
        nir = np.full((60, 60), 0.5)
        # A dark strip south of the cloud at the projected location.
        nir[20:40, 28:33] = 0.02
        result = detect_shadows(cloud, nir, 45.0, 0.0, 100.0)
        assert result.shadow.any()
        assert (result.shadow & cloud).sum() == 0

    def test_no_clouds_returns_empty(self) -> None:
        result = detect_shadows(
            np.zeros((10, 10), dtype=bool), np.ones((10, 10)), 45.0, 180.0, 10.0
        )
        assert not result.shadow.any()


class TestSnowWater:
    def test_indices_range(self) -> None:
        g = np.array([[0.6]])
        s = np.array([[0.1]])
        assert ndsi(g, s)[0, 0] == pytest.approx(0.7142857142857143)
        assert mndwi(g, s)[0, 0] == ndsi(g, s)[0, 0]
        assert ndwi(g, np.array([[0.2]]))[0, 0] == pytest.approx(0.5)

    def test_detect_snow(self) -> None:
        green = np.full((10, 10), 0.6)
        swir = np.full((10, 10), 0.1)
        nir = np.full((10, 10), 0.5)
        assert detect_snow(green, swir, nir).all()
        # Water has low NIR -> excluded.
        assert not detect_snow(green, swir, np.full((10, 10), 0.02)).any()

    def test_detect_water_requires_band(self) -> None:
        with pytest.raises(MaskingError):
            detect_water(np.zeros((5, 5)))

    def test_detect_water_union(self) -> None:
        green = np.full((5, 5), 0.3)
        nir = np.full((5, 5), 0.05)
        water = detect_water(green, nir=nir)
        assert water.all()


class TestNodata:
    def test_explicit_value(self) -> None:
        img = np.ones((2, 5, 5))
        img[0, 0, 0] = -999
        mask = detect_nodata(img, nodata_value=-999)
        assert mask[0, 0]
        assert mask.sum() == 1

    def test_nan_detected(self) -> None:
        img = np.ones((5, 5))
        img[1, 1] = np.nan
        assert detect_nodata(img)[1, 1]

    def test_require_all_bands(self) -> None:
        img = np.ones((2, 3, 3))
        img[0, 0, 0] = -1
        assert not detect_nodata(img, nodata_value=-1, require_all_bands=True)[0, 0]

    def test_saturation_uint8(self) -> None:
        img = np.ones((4, 4)) * 100
        img[2, 2] = 255
        assert detect_saturation(img, "uint8")[2, 2]
        assert detect_saturation(img, "float32").sum() == 0

    def test_edges_nodata_collar_only(self) -> None:
        img = np.ones((6, 6))
        img[0, :] = 0  # border collar
        img[3, 3] = 0  # interior hole
        mask = detect_edges_nodata(img, nodata_value=0)
        assert mask[0].all()
        assert not mask[3, 3]


class TestCombiner:
    def test_priority_order(self) -> None:
        shape = (4, 4)
        every = np.ones(shape, dtype=bool)
        vm = combine_masks(
            shape, cloud=every, shadow=every, snow=every, water=every, nodata=every
        )
        assert (vm.mask == CODE_NODATA).all()
        vm2 = combine_masks(shape, cloud=every, shadow=every)
        assert (vm2.mask == CODE_CLOUD).all()

    def test_codes_assigned(self) -> None:
        shape = (2, 5)
        cloud = np.zeros(shape, dtype=bool)
        cloud[0, 0] = True
        shadow = np.zeros(shape, dtype=bool)
        shadow[0, 1] = True
        snow = np.zeros(shape, dtype=bool)
        snow[0, 2] = True
        water = np.zeros(shape, dtype=bool)
        water[0, 3] = True
        nodata = np.zeros(shape, dtype=bool)
        nodata[0, 4] = True
        vm = combine_masks(shape, cloud, shadow, snow, water, nodata)
        assert vm.mask[0, 0] == CODE_CLOUD
        assert vm.mask[0, 1] == CODE_SHADOW
        assert vm.mask[0, 2] == CODE_SNOW
        assert vm.mask[0, 3] == CODE_WATER
        assert vm.mask[0, 4] == CODE_NODATA
        assert vm.mask[1, 0] == CODE_VALID
        assert vm.mask.dtype == np.uint8

    def test_low_reliability_warning(self) -> None:
        shape = (10, 10)
        cloud = np.ones(shape, dtype=bool)
        cloud[:, :3] = False
        vm = combine_masks(shape, cloud=cloud)
        assert vm.low_reliability
        assert vm.warnings

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(MaskingError):
            combine_masks((4, 4), cloud=np.zeros((5, 5), dtype=bool))

    def test_to_dict_excludes_mask(self) -> None:
        vm = combine_masks((3, 3))
        data = vm.to_dict()
        assert "mask" not in data
        assert data["invalid_fraction"] == 0.0
