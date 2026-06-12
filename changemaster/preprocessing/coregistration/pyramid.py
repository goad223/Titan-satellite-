"""Coarse-to-fine multi-resolution (pyramid) registration strategy.

Feature matching starts on the most downsampled level, where large offsets
become small, and the estimate is propagated and refined level by level up
to full resolution, finishing with grid phase-correlation + ECC refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import CoregistrationError
from changemaster.core.hardware import HardwareInfo, HardwareTier, detect_hardware
from changemaster.preprocessing._common import require_cv2
from changemaster.preprocessing.coregistration.area_based import grid_phase_correlation
from changemaster.preprocessing.coregistration.feature_based import (
    MatchResult,
    extract_and_match_features,
)
from changemaster.preprocessing.coregistration.transforms import (
    AffineTransform,
    GeometricTransform,
    select_transform_model,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PyramidRegistrationResult:
    """Result of pyramid (coarse-to-fine) registration.

    Attributes
    ----------
    transform:
        Final fitted :class:`GeometricTransform` (moving -> reference).
    matches:
        Full-resolution inlier matches used for the final model.
    levels_used:
        Number of pyramid levels processed.
    refined_with_ecc:
        ``True`` when the final ECC refinement succeeded.
    warnings:
        Bilingual warning strings accumulated during registration.
    """

    transform: GeometricTransform
    matches: MatchResult
    levels_used: int
    refined_with_ecc: bool = False
    warnings: list[str] = field(default_factory=list)


def build_pyramid(band: "np.ndarray", levels: int) -> list["np.ndarray"]:
    """Build a Gaussian pyramid ``[full_res, half, quarter, ...]``."""
    cv2 = require_cv2()
    import numpy as np

    current = np.asarray(band, dtype=np.float32)
    pyramid = [current]
    for _ in range(levels - 1):
        if min(current.shape) < 64:
            break
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def pyramid_levels_for(shape: tuple[int, int], hardware: HardwareInfo | None = None) -> int:
    """Choose the pyramid depth from image size and hardware tier.

    Weaker hardware gets *more* levels (smaller working windows at the top
    of the pyramid) — never reduced final accuracy.
    """
    hw = hardware if hardware is not None else detect_hardware()
    size = min(shape)
    levels = 1
    while size > 512:
        size //= 2
        levels += 1
    if hw.tier is HardwareTier.LOW:
        levels += 1
    return max(1, min(levels, 6))


def register_pyramid(
    reference_band: "np.ndarray",
    moving_band: "np.ndarray",
    hardware: HardwareInfo | None = None,
    initial_offset_xy: tuple[float, float] | None = None,
    target_rmse_px: float = 1.0,
) -> PyramidRegistrationResult:
    """Register ``moving_band`` to ``reference_band`` coarse-to-fine.

    Parameters
    ----------
    reference_band / moving_band:
        2-D arrays of the band chosen for matching (prefer SWIR/NIR).
    hardware:
        Hardware snapshot for adaptive windows/levels.
    initial_offset_xy:
        Optional coarse ``(dx, dy)`` from geographic metadata; applied
        before feature matching to maximize usable overlap.
    target_rmse_px:
        Internal model-fit RMSE target; exceeding it triggers an alternate
        strategy (area-based) and a warning.

    Returns
    -------
    PyramidRegistrationResult
        Final transform plus diagnostics.
    """
    import numpy as np

    ref = np.asarray(reference_band, dtype=np.float32)
    mov = np.asarray(moving_band, dtype=np.float32)
    warnings: list[str] = []

    pre_shift = AffineTransform(
        matrix=np.array(
            [
                [1.0, 0.0, -(initial_offset_xy[0] if initial_offset_xy else 0.0)],
                [0.0, 1.0, -(initial_offset_xy[1] if initial_offset_xy else 0.0)],
            ]
        )
    )
    mov_work = (
        pre_shift.warp_image(mov, ref.shape) if initial_offset_xy is not None else mov
    )

    levels = pyramid_levels_for(ref.shape, hardware)
    ref_pyr = build_pyramid(ref, levels)
    mov_pyr = build_pyramid(mov_work, levels)
    levels = min(len(ref_pyr), len(mov_pyr))

    matches: MatchResult | None = None
    # Coarse-to-fine: match at the coarsest level first; if it succeeds at a
    # finer level, prefer the finer (more accurate) matches.
    for level in range(levels - 1, -1, -1):
        scale = 2.0**level
        try:
            level_matches = extract_and_match_features(ref_pyr[level], mov_pyr[level])
        except CoregistrationError:
            continue
        matches = MatchResult(
            src_points=level_matches.src_points * scale,
            dst_points=level_matches.dst_points * scale,
            detector_counts=level_matches.detector_counts,
            inlier_count=level_matches.inlier_count,
        )
        if level == 0:
            break

    if matches is None:
        # Alternate strategy: pure area-based registration.
        warnings.append(
            "Feature matching failed at all pyramid levels; falling back to "
            "area-based registration. | فشلت مطابقة الميزات في كل مستويات "
            "الهرم؛ يتم التحول إلى التسجيل القائم على المساحة."
        )
        area = grid_phase_correlation(ref, mov_work, hardware=hardware)
        matrix = (
            area.ecc_matrix
            if area.ecc_matrix is not None
            else np.array(
                [
                    [1.0, 0.0, -area.global_shift_xy[0]],
                    [0.0, 1.0, -area.global_shift_xy[1]],
                ]
            )
        )
        transform = AffineTransform(matrix=matrix)
        result_matches = MatchResult(
            src_points=np.empty((0, 2)), dst_points=np.empty((0, 2))
        )
        final = _compose_with_preshift(transform, initial_offset_xy)
        return PyramidRegistrationResult(
            transform=final,
            matches=result_matches,
            levels_used=levels,
            refined_with_ecc=area.ecc_matrix is not None,
            warnings=warnings,
        )

    transform = select_transform_model(matches.src_points, matches.dst_points)
    fit_rmse = float(
        np.sqrt(np.mean(transform.residuals(matches.src_points, matches.dst_points) ** 2))
    )
    refined_with_ecc = False
    if isinstance(transform, AffineTransform) and transform.matrix is not None:
        # Final sub-pixel polish with grid phase-correlation + ECC.
        warped = transform.warp_image(mov_work, ref.shape)
        try:
            area = grid_phase_correlation(ref, warped, hardware=hardware)
            if area.ecc_matrix is not None:
                refined = _compose_affine(area.ecc_matrix, transform.matrix)
                transform = AffineTransform(matrix=refined)
                refined_with_ecc = True
        except CoregistrationError:
            pass
    if fit_rmse > target_rmse_px:
        warnings.append(
            f"Model-fit RMSE {fit_rmse:.2f}px exceeds the {target_rmse_px:.1f}px target. | "
            f"خطأ RMSE للنموذج {fit_rmse:.2f} بكسل يتجاوز الهدف {target_rmse_px:.1f} بكسل."
        )

    final = _compose_with_preshift(transform, initial_offset_xy)
    return PyramidRegistrationResult(
        transform=final,
        matches=matches,
        levels_used=levels,
        refined_with_ecc=refined_with_ecc,
        warnings=warnings,
    )


def _compose_affine(outer: "np.ndarray", inner: "np.ndarray") -> "np.ndarray":
    """Compose two ``(2, 3)`` affine matrices: result = outer o inner."""
    import numpy as np

    o = np.vstack([outer, [0.0, 0.0, 1.0]])
    i = np.vstack([inner, [0.0, 0.0, 1.0]])
    return (o @ i)[:2, :]


def _compose_with_preshift(
    transform: GeometricTransform, initial_offset_xy: tuple[float, float] | None
) -> GeometricTransform:
    """Fold the metadata pre-shift into an affine result when applicable."""
    import numpy as np

    if initial_offset_xy is None:
        return transform
    shift = np.array(
        [[1.0, 0.0, -initial_offset_xy[0]], [0.0, 1.0, -initial_offset_xy[1]]]
    )
    if isinstance(transform, AffineTransform) and transform.matrix is not None:
        return AffineTransform(matrix=_compose_affine(transform.matrix, shift))
    return transform
