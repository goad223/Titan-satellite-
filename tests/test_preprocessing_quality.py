"""Tests for the preprocessing quality gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import QualityGateError
from changemaster.preprocessing.quality import (
    RECOMMEND_PROCEED,
    RECOMMEND_REJECT,
    QualityReport,
    _laplacian_variance,
    _saturation_value,
    assess_quality,
)


class TestLaplacianVariance:
    def test_flat_image_has_zero_variance(self) -> None:
        assert _laplacian_variance(np.full((20, 20), 7.0)) == 0.0

    def test_textured_image_has_positive_variance(self) -> None:
        rng = np.random.default_rng(0)
        assert _laplacian_variance(rng.random((20, 20)) * 255) > 0.0

    def test_tiny_image_returns_zero(self) -> None:
        assert _laplacian_variance(np.ones((2, 2))) == 0.0


class TestSaturationValue:
    def test_uint8(self) -> None:
        assert _saturation_value("uint8") == 255.0

    def test_uint16(self) -> None:
        assert _saturation_value("uint16") == 65535.0

    def test_float_has_none(self) -> None:
        assert _saturation_value("float32") is None


class TestAssessQuality:
    def test_good_png_proceeds(self, png_file: Path) -> None:
        report = assess_quality(png_file)
        assert isinstance(report, QualityReport)
        assert report.readable
        assert report.recommendation == RECOMMEND_PROCEED
        assert report.score > 60
        assert report.sampled_window_count >= 1

    def test_report_serializes(self, png_file: Path) -> None:
        report = assess_quality(png_file)
        data = report.to_dict()
        assert data["path"].endswith(".png")
        assert 0 <= data["score"] <= 100

    def test_missing_file_raises_bilingual(self, tmp_path: Path) -> None:
        with pytest.raises(QualityGateError) as excinfo:
            assess_quality(tmp_path / "missing.png")
        assert excinfo.value.message_ar != excinfo.value.message_en

    def test_saturated_image_penalized(self, tmp_path: Path) -> None:
        from PIL import Image

        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        path = tmp_path / "saturated.png"
        Image.fromarray(arr).save(path)
        report = assess_quality(path)
        assert report.saturation_fraction > 0.9
        assert report.score < 100

    def test_blurred_flat_image_warned(self, tmp_path: Path) -> None:
        from PIL import Image

        arr = np.full((64, 64), 128, dtype=np.uint8)
        path = tmp_path / "flat.png"
        Image.fromarray(arr).save(path)
        report = assess_quality(path)
        assert any("blur" in w.lower() or "ضباب" in w for w in report.warnings)

    def test_reject_recommendation_for_corrupt(self, tmp_path: Path, png_file: Path) -> None:
        report = assess_quality(png_file, reject_threshold=101.0)
        assert report.recommendation == RECOMMEND_REJECT
