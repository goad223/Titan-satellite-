"""Geometric transform models: Affine, Homography and Thin-Plate-Spline.

Each model can be fitted to point correspondences, applied to points, and
used to warp full images. :func:`select_transform_model` automatically
chooses the simplest model that explains the matches.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import CoregistrationError
from changemaster.preprocessing._common import require_cv2

if TYPE_CHECKING:
    import numpy as np


class GeometricTransform(abc.ABC):
    """Abstract geometric transform mapping moving-image points to reference."""

    #: Short model name (e.g. ``"affine"``).
    model_name: str = ""

    @abc.abstractmethod
    def fit(self, src_points: "np.ndarray", dst_points: "np.ndarray") -> None:
        """Estimate parameters from ``(N, 2)`` source/destination points."""

    @abc.abstractmethod
    def apply_to_points(self, points: "np.ndarray") -> "np.ndarray":
        """Transform ``(N, 2)`` points, returning ``(N, 2)`` mapped points."""

    @abc.abstractmethod
    def warp_image(
        self, image: "np.ndarray", output_shape: tuple[int, int]
    ) -> "np.ndarray":
        """Warp a 2-D or ``(bands, H, W)`` image into the reference frame."""

    def residuals(self, src_points: "np.ndarray", dst_points: "np.ndarray") -> "np.ndarray":
        """Per-point Euclidean residuals after applying the transform."""
        import numpy as np

        mapped = self.apply_to_points(src_points)
        return np.linalg.norm(mapped - np.asarray(dst_points, dtype=np.float64), axis=1)


def _check_points(src: "np.ndarray", dst: "np.ndarray", minimum: int, model: str) -> None:
    """Validate matched point arrays for fitting."""
    if src.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2:
        raise CoregistrationError(
            f"Point arrays must both be (N, 2); got {src.shape} and {dst.shape}.",
            f"يجب أن تكون مصفوفتا النقاط بشكل (N, 2)؛ وجد {src.shape} و{dst.shape}.",
        )
    if src.shape[0] < minimum:
        raise CoregistrationError(
            f"Model '{model}' needs at least {minimum} matches, got {src.shape[0]}.",
            f"النموذج '{model}' يحتاج {minimum} تطابقات على الأقل، وجد {src.shape[0]}.",
            suggestion_en="Relax the matching ratio test or use a denser feature detector.",
            suggestion_ar="خفّف اختبار النسبة أو استخدم كاشف ميزات أكثر كثافة.",
        )


def _warp_bands_cv2(
    image: "np.ndarray",
    warp_fn: Any,
) -> "np.ndarray":
    """Apply a per-band cv2 warp function to a 2-D or 3-D image."""
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 2:
        return warp_fn(arr.astype(np.float32))
    out = [warp_fn(arr[b].astype(np.float32)) for b in range(arr.shape[0])]
    return np.stack(out, axis=0)


class AffineTransform(GeometricTransform):
    """2-D affine transform (6 parameters): rotation, scale, shear, shift."""

    model_name = "affine"

    def __init__(self, matrix: "np.ndarray | None" = None) -> None:
        self.matrix: "np.ndarray | None" = matrix  # shape (2, 3)

    def fit(self, src_points: "np.ndarray", dst_points: "np.ndarray") -> None:
        """Least-squares affine fit from ``(N, 2)`` correspondences."""
        import numpy as np

        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)
        _check_points(src, dst, 3, self.model_name)
        ones = np.ones((src.shape[0], 1))
        a = np.hstack([src, ones])
        coeffs, *_ = np.linalg.lstsq(a, dst, rcond=None)
        self.matrix = coeffs.T  # (2, 3)

    def apply_to_points(self, points: "np.ndarray") -> "np.ndarray":
        import numpy as np

        if self.matrix is None:
            raise CoregistrationError("Affine transform not fitted.", "التحويل الأفيني غير مُقدّر.")
        pts = np.asarray(points, dtype=np.float64)
        return pts @ self.matrix[:, :2].T + self.matrix[:, 2]

    def warp_image(self, image: "np.ndarray", output_shape: tuple[int, int]) -> "np.ndarray":
        cv2 = require_cv2()
        import numpy as np

        if self.matrix is None:
            raise CoregistrationError("Affine transform not fitted.", "التحويل الأفيني غير مُقدّر.")
        h, w = output_shape
        mat = np.asarray(self.matrix, dtype=np.float64)

        def _warp(band: "np.ndarray") -> "np.ndarray":
            return cv2.warpAffine(band, mat, (w, h), flags=cv2.INTER_LINEAR)

        return _warp_bands_cv2(image, _warp)


class HomographyTransform(GeometricTransform):
    """Projective transform (8 parameters) for perspective differences."""

    model_name = "homography"

    def __init__(self, matrix: "np.ndarray | None" = None) -> None:
        self.matrix: "np.ndarray | None" = matrix  # shape (3, 3)

    def fit(self, src_points: "np.ndarray", dst_points: "np.ndarray") -> None:
        """DLT homography fit via OpenCV (no RANSAC; matches must be clean)."""
        cv2 = require_cv2()
        import numpy as np

        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)
        _check_points(src, dst, 4, self.model_name)
        matrix, _ = cv2.findHomography(
            src.astype(np.float32), dst.astype(np.float32), method=0
        )
        if matrix is None:
            raise CoregistrationError(
                "Homography estimation failed (degenerate points).",
                "فشل تقدير التحويل الإسقاطي (نقاط منحلة).",
                suggestion_en="Provide more spatially-distributed matches.",
                suggestion_ar="وفر تطابقات موزعة مكانياً بشكل أفضل.",
            )
        self.matrix = matrix

    def apply_to_points(self, points: "np.ndarray") -> "np.ndarray":
        import numpy as np

        if self.matrix is None:
            raise CoregistrationError("Homography not fitted.", "التحويل الإسقاطي غير مُقدّر.")
        pts = np.asarray(points, dtype=np.float64)
        homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
        mapped = homog @ self.matrix.T
        return mapped[:, :2] / mapped[:, 2:3]

    def warp_image(self, image: "np.ndarray", output_shape: tuple[int, int]) -> "np.ndarray":
        cv2 = require_cv2()
        import numpy as np

        if self.matrix is None:
            raise CoregistrationError("Homography not fitted.", "التحويل الإسقاطي غير مُقدّر.")
        h, w = output_shape
        mat = np.asarray(self.matrix, dtype=np.float64)

        def _warp(band: "np.ndarray") -> "np.ndarray":
            return cv2.warpPerspective(band, mat, (w, h), flags=cv2.INTER_LINEAR)

        return _warp_bands_cv2(image, _warp)


class ThinPlateSplineTransform(GeometricTransform):
    """Thin-plate-spline transform for smooth local deformations.

    Pure NumPy implementation using the standard TPS radial basis
    ``U(r) = r^2 * log(r^2)`` with optional regularisation.
    """

    model_name = "tps"

    def __init__(self, regularization: float = 1e-3) -> None:
        self.regularization: float = regularization
        self._control: "np.ndarray | None" = None
        self._weights_x: "np.ndarray | None" = None
        self._weights_y: "np.ndarray | None" = None

    @staticmethod
    def _kernel(r2: "np.ndarray") -> "np.ndarray":
        """TPS radial basis ``U(r) = r^2 log(r^2)`` with ``U(0) = 0``."""
        import numpy as np

        out = np.zeros_like(r2)
        positive = r2 > 0
        out[positive] = r2[positive] * np.log(r2[positive])
        return out

    def fit(self, src_points: "np.ndarray", dst_points: "np.ndarray") -> None:
        """Solve the TPS linear system mapping source to destination points."""
        import numpy as np

        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)
        _check_points(src, dst, 4, self.model_name)
        n = src.shape[0]
        d2 = np.sum((src[:, None, :] - src[None, :, :]) ** 2, axis=2)
        k = self._kernel(d2) + self.regularization * np.eye(n)
        p = np.hstack([np.ones((n, 1)), src])
        a = np.zeros((n + 3, n + 3))
        a[:n, :n] = k
        a[:n, n:] = p
        a[n:, :n] = p.T
        bx = np.concatenate([dst[:, 0], np.zeros(3)])
        by = np.concatenate([dst[:, 1], np.zeros(3)])
        try:
            self._weights_x = np.linalg.solve(a, bx)
            self._weights_y = np.linalg.solve(a, by)
        except np.linalg.LinAlgError as exc:
            raise CoregistrationError(
                f"TPS system is singular: {exc}",
                f"نظام TPS منفرد: {exc}",
                suggestion_en="Increase regularization or remove duplicate control points.",
                suggestion_ar="زد معامل التنظيم أو احذف نقاط التحكم المكررة.",
            ) from exc
        self._control = src

    def apply_to_points(self, points: "np.ndarray") -> "np.ndarray":
        import numpy as np

        if self._control is None or self._weights_x is None or self._weights_y is None:
            raise CoregistrationError("TPS transform not fitted.", "تحويل TPS غير مُقدّر.")
        pts = np.asarray(points, dtype=np.float64)
        d2 = np.sum((pts[:, None, :] - self._control[None, :, :]) ** 2, axis=2)
        u = self._kernel(d2)
        basis = np.hstack([u, np.ones((pts.shape[0], 1)), pts])
        x = basis @ self._weights_x
        y = basis @ self._weights_y
        return np.stack([x, y], axis=1)

    def warp_image(self, image: "np.ndarray", output_shape: tuple[int, int]) -> "np.ndarray":
        cv2 = require_cv2()
        import numpy as np

        if self._control is None:
            raise CoregistrationError("TPS transform not fitted.", "تحويل TPS غير مُقدّر.")
        h, w = output_shape
        # Inverse mapping: we fitted forward (moving -> reference); build the
        # inverse mapping by swapping the roles of the control points.
        inverse = ThinPlateSplineTransform(self.regularization)
        forward_mapped = self.apply_to_points(self._control)
        inverse.fit(forward_mapped, self._control)
        # Sample the inverse on a coarse grid, then upsample the maps (fast
        # and accurate because TPS is smooth between control points).
        grid_step = max(8, min(h, w) // 32)
        gy = np.arange(0, h, grid_step, dtype=np.float64)
        gx = np.arange(0, w, grid_step, dtype=np.float64)
        if gy[-1] != h - 1:
            gy = np.append(gy, h - 1)
        if gx[-1] != w - 1:
            gx = np.append(gx, w - 1)
        mesh_x, mesh_y = np.meshgrid(gx, gy)
        pts = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
        mapped = inverse.apply_to_points(pts)
        map_x_coarse = mapped[:, 0].reshape(mesh_x.shape).astype(np.float32)
        map_y_coarse = mapped[:, 1].reshape(mesh_y.shape).astype(np.float32)
        map_x = cv2.resize(map_x_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y_coarse, (w, h), interpolation=cv2.INTER_LINEAR)

        def _warp(band: "np.ndarray") -> "np.ndarray":
            return cv2.remap(band, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return _warp_bands_cv2(image, _warp)


def select_transform_model(
    src_points: "np.ndarray",
    dst_points: "np.ndarray",
    affine_rmse_threshold: float = 0.75,
    homography_rmse_threshold: float = 0.75,
    tps_min_points: int = 30,
) -> GeometricTransform:
    """Automatically pick the simplest transform that explains the matches.

    Order of preference: Affine -> Homography (perspective) -> TPS (local
    deformations, only when enough control points are available).

    Parameters
    ----------
    src_points / dst_points:
        ``(N, 2)`` inlier correspondences (moving -> reference), as
        ``(x, y)`` pixel coordinates.
    affine_rmse_threshold / homography_rmse_threshold:
        Pixel RMSE below which the simpler model is accepted.
    tps_min_points:
        Minimum matches required before TPS is considered.

    Returns
    -------
    GeometricTransform
        The fitted transform with the best (simplest sufficient) model.
    """
    import numpy as np

    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)

    affine = AffineTransform()
    affine.fit(src, dst)
    affine_rmse = float(np.sqrt(np.mean(affine.residuals(src, dst) ** 2)))
    if affine_rmse <= affine_rmse_threshold or src.shape[0] < 4:
        return affine

    homography = HomographyTransform()
    homography.fit(src, dst)
    homography_rmse = float(np.sqrt(np.mean(homography.residuals(src, dst) ** 2)))
    if homography_rmse <= homography_rmse_threshold or src.shape[0] < tps_min_points:
        return homography if homography_rmse < affine_rmse else affine

    tps = ThinPlateSplineTransform()
    tps.fit(src, dst)
    tps_rmse = float(np.sqrt(np.mean(tps.residuals(src, dst) ** 2)))
    best = min(
        [(affine_rmse, affine), (homography_rmse, homography), (tps_rmse, tps)],
        key=lambda pair: pair[0],
    )
    return best[1]
