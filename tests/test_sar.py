"""Tests for SAR calibration, speckle filtering and conversions."""

from __future__ import annotations

import numpy as np
import pytest

from changemaster.core.exceptions import SARCalibrationError
from changemaster.preprocessing.sar.calibration import (
    CalibrationVector,
    build_calibration_lut,
    calibrate_sigma0,
    parse_calibration_xml,
)
from changemaster.preprocessing.sar.convert import from_db, percentile_clip, to_db
from changemaster.preprocessing.sar.speckle import (
    SPECKLE_FILTERS,
    apply_speckle_filter,
    frost,
    gamma_map,
    refined_lee,
)

CAL_XML = """<?xml version="1.0"?>
<calibration>
  <calibrationVectorList count="2">
    <calibrationVector>
      <line>0</line>
      <pixel count="3">0 50 99</pixel>
      <sigmaNought count="3">500.0 510.0 520.0</sigmaNought>
    </calibrationVector>
    <calibrationVector>
      <line>99</line>
      <pixel count="3">0 50 99</pixel>
      <sigmaNought count="3">505.0 515.0 525.0</sigmaNought>
    </calibrationVector>
  </calibrationVectorList>
</calibration>
"""


class TestCalibration:
    def _vectors(self) -> list[CalibrationVector]:
        return [
            CalibrationVector(0, (0, 50, 99), (500.0, 510.0, 520.0)),
            CalibrationVector(99, (0, 50, 99), (505.0, 515.0, 525.0)),
        ]

    def test_lut_interpolation(self) -> None:
        lut = build_calibration_lut(self._vectors(), 100, 100)
        assert lut.shape == (100, 100)
        assert lut[0, 0] == pytest.approx(500.0)
        assert lut[99, 99] == pytest.approx(525.0)
        assert 500.0 < lut[50, 50] < 525.0

    def test_too_few_vectors_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            build_calibration_lut(self._vectors()[:1], 10, 10)

    def test_negative_gain_raises(self) -> None:
        bad = [
            CalibrationVector(0, (0, 9), (-1.0, 1.0)),
            CalibrationVector(9, (0, 9), (1.0, 1.0)),
        ]
        with pytest.raises(SARCalibrationError):
            build_calibration_lut(bad, 10, 10)

    def test_sigma0_formula(self) -> None:
        lut = np.full((4, 4), 10.0)
        dn = np.full((4, 4), 100.0)
        sigma0 = calibrate_sigma0(dn, lut, nodata_value=None)
        assert np.allclose(sigma0, 100.0)  # (100^2) / (10^2)

    def test_nodata_becomes_nan(self) -> None:
        lut = np.full((2, 2), 10.0)
        dn = np.array([[0.0, 50.0], [50.0, 0.0]])
        sigma0 = calibrate_sigma0(dn, lut, nodata_value=0.0)
        assert np.isnan(sigma0[0, 0]) and np.isnan(sigma0[1, 1])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            calibrate_sigma0(np.zeros((3, 3)), np.ones((4, 4)))

    def test_parse_xml(self) -> None:
        vectors = parse_calibration_xml(CAL_XML)
        assert len(vectors) == 2
        assert vectors[0].line == 0
        assert vectors[1].sigma_nought == (505.0, 515.0, 525.0)

    def test_parse_xml_no_vectors_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            parse_calibration_xml("<calibration/>")

    def test_parse_xml_invalid_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            parse_calibration_xml("not xml at all <")


class TestSpeckleFilters:
    @pytest.mark.parametrize("name", sorted(SPECKLE_FILTERS))
    def test_reduces_variance_in_homogeneous_area(
        self, name: str, speckled_image: np.ndarray
    ) -> None:
        filtered = apply_speckle_filter(speckled_image, name)
        region = (slice(8, 24), slice(4, 24))  # inside the left homogeneous half
        assert np.nanvar(filtered[region]) < np.var(speckled_image[region])

    @pytest.mark.parametrize("name", sorted(SPECKLE_FILTERS))
    def test_preserves_mean_level(self, name: str, speckled_image: np.ndarray) -> None:
        filtered = apply_speckle_filter(speckled_image, name)
        left = float(np.nanmean(filtered[:, :28]))
        right = float(np.nanmean(filtered[:, 36:]))
        assert right > left  # the edge between 0.1 and 0.5 levels survives

    def test_refined_lee_nan_passthrough(self, speckled_image: np.ndarray) -> None:
        img = speckled_image.copy()
        img[0, 0] = np.nan
        out = refined_lee(img)
        assert np.isnan(out[0, 0])

    def test_even_window_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            refined_lee(np.ones((10, 10)), window_size=4)
        with pytest.raises(SARCalibrationError):
            frost(np.ones((10, 10)), window_size=6)
        with pytest.raises(SARCalibrationError):
            gamma_map(np.ones((10, 10)), window_size=2)

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(SARCalibrationError):
            refined_lee(np.ones((10, 10)), looks=0)
        with pytest.raises(SARCalibrationError):
            frost(np.ones((10, 10)), damping=-1)
        with pytest.raises(SARCalibrationError):
            gamma_map(np.ones((10, 10)), looks=-2)

    def test_1d_input_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            refined_lee(np.ones(10))

    def test_unknown_filter_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            apply_speckle_filter(np.ones((10, 10)), "median")


class TestConvert:
    def test_db_roundtrip(self) -> None:
        linear = np.array([[0.01, 0.1], [1.0, 10.0]])
        db = to_db(linear)
        assert np.allclose(db, [[-20, -10], [0, 10]])
        assert np.allclose(from_db(db), linear)

    def test_db_floor_keeps_finite(self) -> None:
        assert np.isfinite(to_db(np.array([[0.0]])))[0, 0]

    def test_db_invalid_floor_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            to_db(np.ones((2, 2)), floor=0.0)

    def test_percentile_clip_limits(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.normal(0, 1, (100, 100))
        clipped = percentile_clip(img, 5.0, 95.0)
        lo, hi = np.percentile(img, [5.0, 95.0])
        assert clipped.min() >= lo - 1e-9
        assert clipped.max() <= hi + 1e-9

    def test_percentile_clip_invalid_range_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            percentile_clip(np.ones((3, 3)), 90.0, 10.0)

    def test_percentile_clip_all_nan_raises(self) -> None:
        with pytest.raises(SARCalibrationError):
            percentile_clip(np.full((3, 3), np.nan))
