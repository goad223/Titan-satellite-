"""Tests for geometric transform models and automatic model selection."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from changemaster.core.exceptions import CoregistrationError
from changemaster.preprocessing.coregistration.transforms import (
    AffineTransform,
    HomographyTransform,
    ThinPlateSplineTransform,
    select_transform_model,
)


def _grid_points(n: int = 6, span: float = 100.0) -> np.ndarray:
    xs, ys = np.meshgrid(np.linspace(0, span, n), np.linspace(0, span, n))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


class TestAffineTransform:
    def test_recovers_translation(self) -> None:
        src = _grid_points()
        dst = src + np.array([5.0, -3.0])
        t = AffineTransform()
        t.fit(src, dst)
        assert np.allclose(t.apply_to_points(src), dst, atol=1e-9)

    def test_recovers_rotation_scale(self) -> None:
        src = _grid_points()
        theta = np.deg2rad(10)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        dst = src @ rot.T * 1.2 + np.array([2.0, 1.0])
        t = AffineTransform()
        t.fit(src, dst)
        assert float(np.max(t.residuals(src, dst))) < 1e-9

    def test_too_few_points_raises(self) -> None:
        t = AffineTransform()
        with pytest.raises(CoregistrationError):
            t.fit(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_unfitted_apply_raises(self) -> None:
        with pytest.raises(CoregistrationError):
            AffineTransform().apply_to_points(np.zeros((1, 2)))

    def test_warp_image_translates(self) -> None:
        t = AffineTransform(matrix=np.array([[1.0, 0.0, -10.0], [0.0, 1.0, 0.0]]))
        image = np.zeros((50, 50), dtype=np.float32)
        image[20:30, 20:30] = 1.0
        warped = t.warp_image(image, (50, 50))
        assert warped[25, 15] > 0.5  # block moved 10 px left
        assert warped[25, 25] < 0.5


class TestHomographyTransform:
    def test_recovers_projective(self) -> None:
        src = _grid_points()
        h = np.array([[1.05, 0.02, 3.0], [0.01, 0.98, -2.0], [1e-4, -5e-5, 1.0]])
        homog = np.hstack([src, np.ones((src.shape[0], 1))]) @ h.T
        dst = homog[:, :2] / homog[:, 2:3]
        t = HomographyTransform()
        t.fit(src, dst)
        assert float(np.max(t.residuals(src, dst))) < 1e-4

    def test_warp_image_runs(self) -> None:
        t = HomographyTransform(matrix=np.eye(3))
        out = t.warp_image(np.ones((20, 30)), (20, 30))
        assert out.shape == (20, 30)

    def test_unfitted_raises(self) -> None:
        with pytest.raises(CoregistrationError):
            HomographyTransform().apply_to_points(np.zeros((1, 2)))


class TestThinPlateSpline:
    def test_interpolates_control_points_exactly(self) -> None:
        rng = np.random.default_rng(0)
        src = _grid_points(5)
        dst = src + rng.normal(0, 2.0, src.shape)
        t = ThinPlateSplineTransform(regularization=0.0)
        t.fit(src, dst)
        assert np.allclose(t.apply_to_points(src), dst, atol=1e-6)

    def test_warp_image_shape(self) -> None:
        src = _grid_points(4, span=40.0)
        dst = src + np.array([1.0, 0.5])
        t = ThinPlateSplineTransform()
        t.fit(src, dst)
        out = t.warp_image(np.ones((50, 50), dtype=np.float32), (50, 50))
        assert out.shape == (50, 50)

    def test_unfitted_raises(self) -> None:
        with pytest.raises(CoregistrationError):
            ThinPlateSplineTransform().apply_to_points(np.zeros((1, 2)))


class TestSelectTransformModel:
    def test_prefers_affine_for_affine_data(self) -> None:
        src = _grid_points()
        dst = src + np.array([3.0, 4.0])
        model = select_transform_model(src, dst)
        assert model.model_name == "affine"

    def test_escalates_for_projective_data(self) -> None:
        src = _grid_points(8)
        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2e-3, 1e-3, 1.0]])
        homog = np.hstack([src, np.ones((src.shape[0], 1))]) @ h.T
        dst = homog[:, :2] / homog[:, 2:3]
        model = select_transform_model(src, dst)
        assert model.model_name in ("homography", "tps")
        assert float(np.sqrt(np.mean(model.residuals(src, dst) ** 2))) < 1.0
