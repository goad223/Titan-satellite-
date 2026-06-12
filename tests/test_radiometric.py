"""Tests for radiometric normalization: histogram matching, IR-MAD, PIF, selector."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("scipy")

from changemaster.core.exceptions import RadiometricError
from changemaster.preprocessing.radiometric.histogram_matching import (
    apply_lut,
    build_matching_lut,
    match_histograms,
    match_histograms_tiled,
)
from changemaster.preprocessing.radiometric.irmad import (
    IncrementalStats,
    apply_irmad_normalization,
    compute_irmad,
)
from changemaster.preprocessing.radiometric.pif_normalization import (
    apply_pif_normalization,
    fit_pif_linear,
    select_pifs_statistical,
)
from changemaster.preprocessing.radiometric.selector import (
    METHOD_HISTOGRAM,
    METHOD_IRMAD,
    METHOD_PIF,
    choose_method,
    normalize_pair,
)


class TestHistogramMatching:
    def test_matched_mean_approaches_reference(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        matched = match_histograms(ref, mov)
        assert abs(matched.mean() - ref.mean()) < 1.0

    def test_2d_input_supported(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(50, 5, (40, 40))
        mov = rng.normal(80, 10, (40, 40))
        matched = match_histograms(ref, mov)
        assert matched.shape == (40, 40)
        assert abs(matched.mean() - ref.mean()) < 2.0

    def test_band_count_mismatch_raises(self) -> None:
        with pytest.raises(RadiometricError):
            match_histograms(np.zeros((2, 10, 10)), np.zeros((3, 10, 10)))

    def test_nodata_preserved(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.normal(50, 5, (1, 30, 30))
        mov = rng.normal(70, 5, (1, 30, 30))
        mov[0, 0, 0] = -999.0
        matched = match_histograms(ref, mov, nodata=-999.0)
        assert matched[0, 0, 0] == -999.0

    def test_tiled_lut_equivalent(self) -> None:
        rng = np.random.default_rng(2)
        ref = rng.normal(100, 10, (64, 64))
        mov = rng.normal(150, 20, (64, 64))
        lo = float(min(ref.min(), mov.min()))
        hi = float(max(ref.max(), mov.max()))
        centers, mapped = match_histograms_tiled(
            [ref[:32], ref[32:]], [mov[:32], mov[32:]], (lo, hi)
        )
        out = apply_lut(mov, centers, mapped)
        assert abs(out.mean() - ref.mean()) < 2.0

    def test_empty_histogram_raises(self) -> None:
        with pytest.raises(RadiometricError):
            build_matching_lut(np.zeros(10), np.ones(10), (0.0, 1.0))


class TestIncrementalStats:
    def test_matches_numpy_cov(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (500, 4))
        stats = IncrementalStats(dim=4)
        stats.update(data[:250])
        stats.update(data[250:])
        assert np.allclose(stats.mean, data.mean(axis=0))
        assert np.allclose(stats.covariance, np.cov(data.T, bias=True), atol=1e-9)

    def test_empty_raises(self) -> None:
        with pytest.raises(RadiometricError):
            _ = IncrementalStats(dim=2).mean

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(RadiometricError):
            IncrementalStats(dim=3).update(np.zeros((5, 2)))


class TestIRMAD:
    def test_recovers_linear_distortion(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        result = compute_irmad(ref, mov)
        assert result.converged
        assert result.iterations >= 2
        # mov = 1.3 ref + 15 -> normalization gain ~ 1/1.3.
        assert np.allclose(result.gains, 1.0 / 1.3, atol=0.05)
        normalized = apply_irmad_normalization(mov, result.gains, result.offsets)
        unchanged = np.ones(ref.shape[1:], dtype=bool)
        unchanged[20:40, 30:50] = False
        diff = np.abs(normalized - ref)[:, unchanged]
        assert float(diff.mean()) < 5.0

    def test_change_region_low_probability(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        result = compute_irmad(ref, mov)
        changed = result.no_change_probability[20:40, 30:50]
        stable = result.no_change_probability[60:, 60:]
        assert changed.mean() < stable.mean()

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(RadiometricError):
            compute_irmad(np.zeros((2, 10, 10)), np.zeros((2, 12, 12)))

    def test_too_few_pixels_raises(self) -> None:
        with pytest.raises(RadiometricError):
            compute_irmad(np.zeros((3, 3, 3)), np.zeros((3, 3, 3)))

    def test_respects_valid_mask(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        mask = np.ones(ref.shape[1:], dtype=bool)
        mask[:10] = False
        result = compute_irmad(ref, mov, valid_mask=mask)
        assert not result.pif_mask[:10].any()


class TestPIF:
    def test_select_pifs_excludes_change(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        # Normalize the gross gain first so difference reflects change only.
        pifs = select_pifs_statistical(ref, (mov - 15.0) / 1.3)
        change_fraction = pifs[20:40, 30:50].mean()
        overall_fraction = pifs.mean()
        assert change_fraction < overall_fraction

    def test_fit_recovers_gain(self, multiband_pair: tuple[np.ndarray, np.ndarray]) -> None:
        ref, mov = multiband_pair
        norm = fit_pif_linear(ref, mov)
        assert norm.pif_count >= 10
        assert np.allclose(norm.gains, 1.0 / 1.3, atol=0.1)
        out = apply_pif_normalization(mov, norm)
        assert out.shape == mov.shape

    def test_no_valid_pixels_raises(self) -> None:
        ref = np.full((2, 10, 10), np.nan)
        with pytest.raises(RadiometricError):
            select_pifs_statistical(ref, ref)


class TestSelector:
    def test_choose_histogram_for_same_sensor_short_gap(self) -> None:
        method, _ = choose_method(4, 100000, same_sensor=True, acquisition_gap_days=10)
        assert method == METHOD_HISTOGRAM

    def test_choose_irmad_for_multiband(self) -> None:
        method, _ = choose_method(4, 100000, same_sensor=False, acquisition_gap_days=400)
        assert method == METHOD_IRMAD

    def test_choose_pif_for_single_band(self) -> None:
        method, _ = choose_method(1, 5000, same_sensor=False, acquisition_gap_days=None)
        assert method == METHOD_PIF

    def test_choose_histogram_for_tiny_budget(self) -> None:
        method, _ = choose_method(3, 100, same_sensor=False, acquisition_gap_days=None)
        assert method == METHOD_HISTOGRAM

    def test_normalize_pair_auto(self, multiband_pair: tuple[np.ndarray, np.ndarray]) -> None:
        ref, mov = multiband_pair
        selection = normalize_pair(ref, mov)
        assert selection.method == METHOD_IRMAD
        assert selection.normalized.shape == ref.shape

    def test_normalize_pair_forced_method(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        selection = normalize_pair(ref, mov, method=METHOD_HISTOGRAM)
        assert selection.method == METHOD_HISTOGRAM

    def test_normalize_pair_unknown_method_raises(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        with pytest.raises(RadiometricError):
            normalize_pair(ref, mov, method="bogus")

    def test_normalize_pair_same_sensor_short_gap(
        self, multiband_pair: tuple[np.ndarray, np.ndarray]
    ) -> None:
        ref, mov = multiband_pair
        selection = normalize_pair(
            ref,
            mov,
            reference_sensor="sentinel2",
            moving_sensor="sentinel2",
            reference_datetime=datetime(2024, 1, 1),
            moving_datetime=datetime(2024, 1, 15),
        )
        assert selection.method == METHOD_HISTOGRAM

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(RadiometricError):
            normalize_pair(np.zeros((2, 10, 10)), np.zeros((2, 11, 11)))
