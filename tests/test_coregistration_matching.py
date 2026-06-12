"""Tests for feature/area-based matching, pyramid registration and evaluation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from changemaster.core.exceptions import CoregistrationError
from changemaster.io_engine.metadata import GeoReference, ImageMetadata
from changemaster.preprocessing.coregistration.area_based import (
    ecc_refine,
    grid_phase_correlation,
    phase_correlation_shift,
)
from changemaster.preprocessing.coregistration.coarse import coarse_align_from_metadata
from changemaster.preprocessing.coregistration.evaluator import (
    displacement_map,
    evaluate_registration,
    split_matches,
)
from changemaster.preprocessing.coregistration.feature_based import (
    extract_and_match_features,
    pick_matching_band,
)
from changemaster.preprocessing.coregistration.pyramid import (
    build_pyramid,
    pyramid_levels_for,
    register_pyramid,
)


class TestFeatureMatching:
    def test_matches_shifted_pair(self, textured_pair: tuple[np.ndarray, np.ndarray]) -> None:
        ref, mov = textured_pair
        result = extract_and_match_features(ref, mov)
        assert result.inlier_count >= 10
        assert set(result.detector_counts) == {"sift", "orb", "akaze"}
        shifts = result.dst_points - result.src_points
        # Points move opposite to the warp: dx ~ -5.3, dy ~ +3.7.
        assert abs(np.median(shifts[:, 0]) + 5.3) < 1.0
        assert abs(np.median(shifts[:, 1]) - 3.7) < 1.0

    def test_featureless_image_raises(self) -> None:
        flat = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(CoregistrationError):
            extract_and_match_features(flat, flat)

    def test_pick_matching_band_prefers_swir(self) -> None:
        names = ["Blue", "Green", "Red", "NIR", "SWIR 1"]
        assert pick_matching_band(names, 5) == 5

    def test_pick_matching_band_prefers_nir_over_red(self) -> None:
        assert pick_matching_band(["Red", "NIR"], 2) == 2

    def test_pick_matching_band_fallback(self) -> None:
        assert pick_matching_band(["Band 1", "Band 2"], 2) == 1


class TestAreaBased:
    def test_phase_correlation_recovers_shift(
        self, textured_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = textured_pair
        (dx, dy), response = phase_correlation_shift(ref, mov)
        assert response > 0.1
        assert abs(dx + 5.3) < 0.5
        assert abs(dy - 3.7) < 0.5

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(CoregistrationError):
            phase_correlation_shift(np.zeros((10, 10)), np.zeros((12, 12)))

    def test_grid_phase_correlation(self, textured_pair: tuple[np.ndarray, np.ndarray]) -> None:
        ref, mov = textured_pair
        result = grid_phase_correlation(ref, mov)
        assert len(result.window_shifts) >= 1
        assert abs(result.global_shift_xy[0] + 5.3) < 1.0

    def test_grid_flat_image_raises(self) -> None:
        flat = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(CoregistrationError):
            grid_phase_correlation(flat, flat)

    def test_ecc_refine_returns_matrix(
        self, textured_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = textured_pair
        initial = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, 3.5]], dtype=np.float32)
        matrix = ecc_refine(ref, mov, initial_matrix=initial)
        assert matrix is not None
        assert matrix.shape == (2, 3)
        assert abs(matrix[0, 2] + 5.3) < 0.5
        assert abs(matrix[1, 2] - 3.7) < 0.5

    def test_ecc_refine_rejects_divergence(
        self, textured_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = textured_pair
        bad = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 200.0]], dtype=np.float32)
        assert ecc_refine(ref, mov, initial_matrix=bad) is None


class TestCoarseAlignment:
    def _meta(self, transform: tuple, crs: str = "EPSG:32636") -> ImageMetadata:
        return ImageMetadata(
            path=__import__("pathlib").Path("x.tif"),
            driver="GTiff",
            width=100,
            height=100,
            band_count=1,
            dtype="uint16",
            georef=GeoReference(crs=crs, transform=transform),
        )

    def test_offset_computed(self) -> None:
        ref = self._meta((10.0, 0.0, 500000.0, 0.0, -10.0, 4100000.0))
        mov = self._meta((10.0, 0.0, 500100.0, 0.0, -10.0, 4099950.0))
        result = coarse_align_from_metadata(ref, mov)
        assert result.offset_xy == pytest.approx((10.0, 5.0))
        assert result.pixel_size_ratio == pytest.approx(1.0)

    def test_crs_mismatch_raises(self) -> None:
        ref = self._meta((10.0, 0.0, 0.0, 0.0, -10.0, 0.0), "EPSG:32636")
        mov = self._meta((10.0, 0.0, 0.0, 0.0, -10.0, 0.0), "EPSG:32637")
        with pytest.raises(CoregistrationError):
            coarse_align_from_metadata(ref, mov)

    def test_missing_georef_raises(self) -> None:
        ref = self._meta((10.0, 0.0, 0.0, 0.0, -10.0, 0.0))
        mov = ImageMetadata(
            path=__import__("pathlib").Path("y.png"),
            driver="PNG",
            width=10,
            height=10,
            band_count=1,
            dtype="uint8",
        )
        with pytest.raises(CoregistrationError):
            coarse_align_from_metadata(ref, mov)


class TestPyramid:
    def test_build_pyramid_levels(self) -> None:
        rng = np.random.default_rng(0)
        pyr = build_pyramid(rng.random((512, 512)), 3)
        assert len(pyr) == 3
        assert pyr[1].shape[0] == 256

    def test_levels_for_shape(self) -> None:
        assert pyramid_levels_for((256, 256)) >= 1
        assert pyramid_levels_for((4096, 4096)) >= 3

    def test_register_pyramid_subpixel(
        self, textured_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = textured_pair
        result = register_pyramid(ref, mov)
        _, evaluation = evaluate_registration(
            result.matches.src_points, result.matches.dst_points
        )
        assert evaluation.rmse_px < 1.0
        assert evaluation.meets_target

    def test_register_with_initial_offset(
        self, textured_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = textured_pair
        result = register_pyramid(ref, mov, initial_offset_xy=(5.0, -4.0))
        aligned = result.transform.warp_image(mov, ref.shape)
        inner = (slice(30, -30), slice(30, -30))
        corr = np.corrcoef(ref[inner].ravel(), aligned[inner].ravel())[0, 1]
        assert corr > 0.95


class TestEvaluator:
    def test_split_is_deterministic_and_disjoint(self) -> None:
        rng = np.random.default_rng(0)
        src = rng.random((40, 2)) * 100
        dst = src + 1.0
        se1, de1, sv1, dv1 = split_matches(src, dst)
        se2, _, sv2, _ = split_matches(src, dst)
        assert np.array_equal(se1, se2)
        assert np.array_equal(sv1, sv2)
        assert se1.shape[0] + sv1.shape[0] == 40
        assert de1.shape == se1.shape and dv1.shape == sv1.shape

    def test_too_few_matches_raises(self) -> None:
        with pytest.raises(CoregistrationError):
            split_matches(np.zeros((4, 2)), np.zeros((4, 2)))

    def test_rmse_on_validation_only(self) -> None:
        rng = np.random.default_rng(1)
        src = rng.random((60, 2)) * 200
        dst = src + np.array([2.0, -1.0]) + rng.normal(0, 0.1, (60, 2))
        transform, evaluation = evaluate_registration(src, dst)
        assert transform.model_name == "affine"
        assert evaluation.rmse_px < 0.5
        assert evaluation.validation_count + evaluation.estimation_count == 60

    def test_warning_when_target_exceeded(self) -> None:
        rng = np.random.default_rng(2)
        src = rng.random((30, 2)) * 100
        dst = src + rng.normal(0, 5.0, (30, 2))  # very noisy
        _, evaluation = evaluate_registration(src, dst, target_rmse_px=0.1)
        assert not evaluation.meets_target
        assert evaluation.warnings

    def test_displacement_map_values(self) -> None:
        from changemaster.preprocessing.coregistration.transforms import AffineTransform

        t = AffineTransform(matrix=np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 4.0]]))
        disp = displacement_map(t, (128, 128), grid_step=64)
        assert disp.shape == (2, 2)
        assert np.allclose(disp, 5.0)
