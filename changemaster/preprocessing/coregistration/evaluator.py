"""Registration accuracy evaluation on independent check points.

Matches are split into an estimation set and an *independent* validation
set; RMSE is computed only on the validation set so it honestly reflects
generalization. A displacement map over a grid summarizes residual motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import CoregistrationError
from changemaster.preprocessing.coregistration.transforms import (
    GeometricTransform,
    select_transform_model,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass
class RegistrationEvaluation:
    """Accuracy report of a registration solution.

    Attributes
    ----------
    rmse_px:
        RMSE (pixels) on the independent validation points only.
    max_error_px:
        Maximum validation residual.
    validation_count / estimation_count:
        Sizes of the two point sets.
    meets_target:
        ``True`` when ``rmse_px <= target_rmse_px``.
    target_rmse_px:
        Acceptance threshold used.
    warnings:
        Bilingual warnings (e.g. when the target is exceeded).
    """

    rmse_px: float
    max_error_px: float
    validation_count: int
    estimation_count: int
    meets_target: bool
    target_rmse_px: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)


def split_matches(
    src_points: "np.ndarray",
    dst_points: "np.ndarray",
    validation_fraction: float = 0.25,
    seed: int = 12345,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    """Split matches into estimation and validation sets (deterministic).

    Returns ``(src_est, dst_est, src_val, dst_val)``.
    """
    import numpy as np

    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    n = src.shape[0]
    if n < 8:
        raise CoregistrationError(
            f"Need at least 8 matches to hold out a validation set, got {n}.",
            f"يلزم 8 تطابقات على الأقل لتخصيص مجموعة تحقق، وجد {n}.",
            suggestion_en="Lower the ratio-test threshold to harvest more matches.",
            suggestion_ar="خفّض عتبة اختبار النسبة للحصول على مزيد من التطابقات.",
        )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(2, int(round(n * validation_fraction)))
    val_idx = indices[:n_val]
    est_idx = indices[n_val:]
    return src[est_idx], dst[est_idx], src[val_idx], dst[val_idx]


def evaluate_registration(
    src_points: "np.ndarray",
    dst_points: "np.ndarray",
    target_rmse_px: float = 1.0,
    validation_fraction: float = 0.25,
    seed: int = 12345,
) -> tuple[GeometricTransform, RegistrationEvaluation]:
    """Fit on the estimation set, score on the independent validation set.

    Parameters
    ----------
    src_points / dst_points:
        Full inlier match arrays (moving -> reference).
    target_rmse_px:
        Pass/fail RMSE threshold (default 1 px; the project ambition is 0.5).
    validation_fraction:
        Fraction held out for validation.
    seed:
        RNG seed for the deterministic split.

    Returns
    -------
    tuple
        ``(fitted_transform, evaluation)``.
    """
    import numpy as np

    src_est, dst_est, src_val, dst_val = split_matches(
        src_points, dst_points, validation_fraction, seed
    )
    transform = select_transform_model(src_est, dst_est)
    residuals = transform.residuals(src_val, dst_val)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    max_err = float(np.max(residuals))
    warnings: list[str] = []
    meets = rmse <= target_rmse_px
    if not meets:
        warnings.append(
            f"Validation RMSE {rmse:.2f}px exceeds the {target_rmse_px:.1f}px target; "
            "registration accuracy is degraded — treat downstream change maps with caution. | "
            f"خطأ RMSE على مجموعة التحقق {rmse:.2f} بكسل يتجاوز الهدف "
            f"{target_rmse_px:.1f} بكسل؛ دقة التسجيل منخفضة — تعامل بحذر مع خرائط التغير اللاحقة."
        )
    evaluation = RegistrationEvaluation(
        rmse_px=rmse,
        max_error_px=max_err,
        validation_count=int(src_val.shape[0]),
        estimation_count=int(src_est.shape[0]),
        meets_target=meets,
        target_rmse_px=target_rmse_px,
        warnings=warnings,
    )
    return transform, evaluation


def displacement_map(
    transform: GeometricTransform,
    shape: tuple[int, int],
    grid_step: int = 64,
) -> "np.ndarray":
    """Residual displacement magnitude sampled on a regular grid.

    Returns a ``(rows, cols)`` float array where each cell holds the
    Euclidean displacement (pixels) the transform applies at that location.
    """
    import numpy as np

    h, w = shape
    ys = np.arange(grid_step // 2, h, grid_step, dtype=np.float64)
    xs = np.arange(grid_step // 2, w, grid_step, dtype=np.float64)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    pts = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
    mapped = transform.apply_to_points(pts)
    disp = np.linalg.norm(mapped - pts, axis=1)
    return disp.reshape(len(ys), len(xs))
