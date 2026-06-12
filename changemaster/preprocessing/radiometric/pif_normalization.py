"""Linear radiometric normalization via Pseudo-Invariant Features (PIFs).

PIFs are temporally stable pixels (roads, rooftops, bare rock). This module
selects them statistically (when no IR-MAD probability map is available)
and fits a robust per-band linear mapping moving -> reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import RadiometricError

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PIFNormalization:
    """Fitted PIF-based linear normalization.

    Attributes
    ----------
    gains / offsets:
        Per-band coefficients of ``normalized = gain * moving + offset``.
    pif_mask:
        Boolean ``(H, W)`` mask of the PIF pixels used.
    pif_count:
        Number of PIF pixels.
    r_squared:
        Per-band coefficient of determination on the PIFs.
    """

    gains: "np.ndarray"
    offsets: "np.ndarray"
    pif_mask: "np.ndarray"
    pif_count: int
    r_squared: "np.ndarray"
    warnings: list[str] = field(default_factory=list)


def select_pifs_statistical(
    reference: "np.ndarray",
    moving: "np.ndarray",
    valid_mask: "np.ndarray | None" = None,
    difference_percentile: float = 20.0,
    min_brightness_percentile: float = 30.0,
) -> "np.ndarray":
    """Select PIF candidates from low normalized-difference, bright pixels.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` co-registered arrays.
    valid_mask:
        Boolean ``(H, W)`` usable-pixel mask.
    difference_percentile:
        Keep pixels whose mean absolute normalized difference is below this
        percentile (most stable pixels).
    min_brightness_percentile:
        Discard very dark pixels (water/shadow) below this brightness
        percentile.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` PIF mask.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.ndim != 3:
        raise RadiometricError(
            f"PIF selection needs identical (bands, H, W) arrays; got {ref.shape} vs {mov.shape}.",
            f"يتطلب اختيار PIF مصفوفتين متطابقتين؛ وجد {ref.shape} و{mov.shape}.",
        )
    height, width = ref.shape[1:]
    valid = (
        np.ones((height, width), dtype=bool) if valid_mask is None else valid_mask.copy()
    )
    valid &= np.all(np.isfinite(ref), axis=0) & np.all(np.isfinite(mov), axis=0)
    if not np.any(valid):
        raise RadiometricError(
            "No valid pixels available for PIF selection.",
            "لا توجد بكسلات صالحة لاختيار PIF.",
            suggestion_en="Check the validity mask coverage.",
            suggestion_ar="تحقق من تغطية قناع الصلاحية.",
        )

    denom = np.abs(ref) + np.abs(mov)
    denom[denom == 0] = 1.0
    norm_diff = np.mean(np.abs(ref - mov) / denom, axis=0)
    brightness = np.mean(ref, axis=0)

    diff_thresh = np.percentile(norm_diff[valid], difference_percentile)
    bright_thresh = np.percentile(brightness[valid], min_brightness_percentile)
    return valid & (norm_diff <= diff_thresh) & (brightness >= bright_thresh)


def fit_pif_linear(
    reference: "np.ndarray",
    moving: "np.ndarray",
    pif_mask: "np.ndarray | None" = None,
    valid_mask: "np.ndarray | None" = None,
    trim_iterations: int = 2,
    trim_sigma: float = 2.5,
) -> PIFNormalization:
    """Fit robust per-band linear normalization on PIF pixels.

    Uses iterative sigma-trimming: after each least-squares fit, points with
    residuals beyond ``trim_sigma`` standard deviations are dropped.

    Raises
    ------
    RadiometricError
        When fewer than 10 PIF pixels are available.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if pif_mask is None:
        pif_mask = select_pifs_statistical(ref, mov, valid_mask)
    count = int(pif_mask.sum())
    if count < 10:
        raise RadiometricError(
            f"Only {count} PIF pixels; at least 10 are required for a stable fit.",
            f"وُجد {count} بكسل PIF فقط؛ المطلوب 10 على الأقل لتقدير مستقر.",
            suggestion_en="Relax the PIF selection percentiles.",
            suggestion_ar="خفّف نسب اختيار PIF المئوية.",
        )

    bands = ref.shape[0]
    gains = np.ones(bands)
    offsets = np.zeros(bands)
    r2 = np.zeros(bands)
    warnings: list[str] = []
    flat = pif_mask.ravel()
    for k in range(bands):
        x = mov[k].ravel()[flat]
        y = ref[k].ravel()[flat]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        for _ in range(max(1, trim_iterations)):
            if x.size < 2 or float(np.var(x)) <= 0:
                break
            gain, offset = np.polyfit(x, y, 1)
            residuals = y - (gain * x + offset)
            sigma = float(np.std(residuals))
            if sigma <= 0:
                break
            keep = np.abs(residuals) <= trim_sigma * sigma
            if keep.all():
                break
            x, y = x[keep], y[keep]
        if x.size >= 2 and float(np.var(x)) > 0:
            gain, offset = np.polyfit(x, y, 1)
            gains[k], offsets[k] = gain, offset
            pred = gain * x + offset
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2[k] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            warnings.append(
                f"Band {k + 1}: PIF fit degenerate, identity kept. | "
                f"النطاق {k + 1}: تقدير PIF منحل، تم الإبقاء على التحويل المحايد."
            )
    return PIFNormalization(
        gains=gains,
        offsets=offsets,
        pif_mask=pif_mask,
        pif_count=count,
        r_squared=r2,
        warnings=warnings,
    )


def apply_pif_normalization(
    moving: "np.ndarray", normalization: PIFNormalization
) -> "np.ndarray":
    """Apply a fitted :class:`PIFNormalization` to the moving image."""
    import numpy as np

    mov = np.asarray(moving, dtype=np.float64)
    return (
        mov * normalization.gains[:, None, None]
        + normalization.offsets[:, None, None]
    )
