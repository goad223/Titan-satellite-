"""Speckle filtering: Refined Lee (default), Frost and Gamma-MAP.

All filters operate on linear-power SAR images with standard, tunable
parameters. The Refined Lee filter uses 7x7 edge-direction-oriented
windows; Frost uses an exponentially damped kernel; Gamma-MAP solves the
maximum-a-posteriori estimate under Gamma scene/speckle statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from changemaster.core.exceptions import SARCalibrationError

if TYPE_CHECKING:
    import numpy as np


def _check_2d(image: "np.ndarray", name: str) -> "np.ndarray":
    """Validate and return the image as a float64 2-D array."""
    import numpy as np

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise SARCalibrationError(
            f"{name} expects a 2-D image, got {arr.ndim}-D.",
            f"يتوقع {name} صورة ثنائية الأبعاد، وجد {arr.ndim} أبعاد.",
        )
    return arr


def _sliding_windows(padded: "np.ndarray", size: int) -> "np.ndarray":
    """Return a ``(H, W, size, size)`` sliding-window view of a padded image."""
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(padded, (size, size))


# 7x7 directional masks for the Refined Lee filter. Each mask keeps the
# half-window on one side of an edge through the centre (4 edge directions
# x 2 sides = 8 oriented sub-windows).
def _directional_masks(size: int = 7) -> "np.ndarray":
    """Build the 8 oriented half-window masks ``(8, size, size)``."""
    import numpy as np

    c = size // 2
    rows, cols = np.mgrid[0:size, 0:size]
    masks = np.zeros((8, size, size), dtype=bool)
    masks[0] = cols >= c  # edge vertical, keep right
    masks[1] = cols <= c  # keep left
    masks[2] = rows >= c  # edge horizontal, keep bottom
    masks[3] = rows <= c  # keep top
    masks[4] = (cols - c) >= (rows - c)  # diagonal /
    masks[5] = (cols - c) <= (rows - c)
    masks[6] = (cols - c) >= -(rows - c)  # diagonal \
    masks[7] = (cols - c) <= -(rows - c)
    return masks


def refined_lee(
    image: "np.ndarray",
    window_size: int = 7,
    looks: float = 1.0,
) -> "np.ndarray":
    """Refined Lee speckle filter with directional 7x7 windows.

    For each pixel the local gradient picks the edge direction; statistics
    are computed only on the oriented half-window lying on the pixel's side
    of the edge, which preserves edges while smoothing homogeneous areas.

    Parameters
    ----------
    image:
        2-D linear-power SAR image.
    window_size:
        Odd window edge (default 7 as in the original formulation).
    looks:
        Equivalent number of looks of the input (controls the speckle
        coefficient of variation ``Cu = 1/sqrt(looks)``).

    Returns
    -------
    np.ndarray
        Filtered float64 image.
    """
    import numpy as np

    arr = _check_2d(image, "refined_lee")
    if window_size % 2 == 0 or window_size < 3:
        raise SARCalibrationError(
            f"Window size must be odd and >= 3, got {window_size}.",
            f"يجب أن يكون حجم النافذة فردياً و>= 3، وجد {window_size}.",
        )
    if looks <= 0:
        raise SARCalibrationError(
            f"Number of looks must be positive, got {looks}.",
            f"يجب أن يكون عدد looks موجباً، وجد {looks}.",
        )
    pad = window_size // 2
    filled = np.where(np.isfinite(arr), arr, np.nanmean(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0)
    padded = np.pad(filled, pad, mode="reflect")
    windows = _sliding_windows(padded, window_size)  # (H, W, k, k)
    masks = _directional_masks(window_size)  # (8, k, k)

    # Edge direction from gradients of the local means of quadrant halves.
    gy = np.gradient(filled, axis=0)
    gx = np.gradient(filled, axis=1)
    angle = np.arctan2(gy, gx)  # -pi..pi
    # Quantize to 4 edge orientations and pick the half-window away from
    # the gradient (the side the pixel statistically belongs to).
    sector = ((angle + np.pi) / (np.pi / 4.0)).astype(int) % 8

    cu2 = 1.0 / looks  # speckle variance coefficient (Cu^2)
    out = np.empty_like(filled)
    h, w = filled.shape
    # Compute per-direction means/vars (vectorised over the whole image),
    # then select per-pixel by the sector index.
    means = np.empty((8, h, w))
    variances = np.empty((8, h, w))
    for d in range(8):
        m = masks[d]
        count = float(m.sum())
        sel = windows[:, :, m]  # (H, W, n)
        mu = sel.mean(axis=2)
        var = sel.var(axis=2)
        means[d] = mu
        variances[d] = var
        del sel
        _ = count
    idx = sector
    mu = np.take_along_axis(means, idx[None], axis=0)[0]
    var = np.take_along_axis(variances, idx[None], axis=0)[0]

    # Lee MMSE weight: k = max(0, (Cx^2 - Cu^2) / (Cx^2 (1 + Cu^2))).
    mu_safe = np.where(mu == 0, 1e-12, mu)
    cx2 = var / (mu_safe**2)
    k = np.clip((cx2 - cu2) / (cx2 * (1.0 + cu2) + 1e-12), 0.0, 1.0)
    out = mu + k * (filled - mu)
    out[~np.isfinite(arr)] = np.nan
    return out


def frost(
    image: "np.ndarray",
    window_size: int = 5,
    damping: float = 2.0,
) -> "np.ndarray":
    """Frost speckle filter with an exponentially damped kernel.

    Each pixel is a weighted mean of its window where weights decay with
    distance, scaled by the local coefficient of variation:
    ``w = exp(-damping * (var/mean^2) * distance)``.

    Parameters
    ----------
    image:
        2-D linear-power SAR image.
    window_size:
        Odd window edge length.
    damping:
        Damping factor; larger preserves edges more strongly.
    """
    import numpy as np

    arr = _check_2d(image, "frost")
    if window_size % 2 == 0 or window_size < 3:
        raise SARCalibrationError(
            f"Window size must be odd and >= 3, got {window_size}.",
            f"يجب أن يكون حجم النافذة فردياً و>= 3، وجد {window_size}.",
        )
    if damping <= 0:
        raise SARCalibrationError(
            f"Damping factor must be positive, got {damping}.",
            f"يجب أن يكون معامل التخميد موجباً، وجد {damping}.",
        )
    pad = window_size // 2
    fill = np.nanmean(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0
    filled = np.where(np.isfinite(arr), arr, fill)
    padded = np.pad(filled, pad, mode="reflect")
    windows = _sliding_windows(padded, window_size)  # (H, W, k, k)

    rows, cols = np.mgrid[0:window_size, 0:window_size]
    distance = np.sqrt((rows - pad) ** 2 + (cols - pad) ** 2)  # (k, k)

    mu = windows.mean(axis=(2, 3))
    var = windows.var(axis=(2, 3))
    mu_safe = np.where(mu == 0, 1e-12, mu)
    b = damping * var / (mu_safe**2)  # (H, W)
    weights = np.exp(-b[:, :, None, None] * distance[None, None])
    out = (windows * weights).sum(axis=(2, 3)) / weights.sum(axis=(2, 3))
    out[~np.isfinite(arr)] = np.nan
    return out


def gamma_map(
    image: "np.ndarray",
    window_size: int = 5,
    looks: float = 1.0,
) -> "np.ndarray":
    """Gamma-MAP speckle filter (maximum a posteriori, Gamma-Gamma model).

    Classification per pixel:
      * homogeneous (``Cx <= Cu``): replace with the local mean;
      * heterogeneous (``Cu < Cx < Cmax``): MAP solution
        ``x = (alpha-L-1)mu + sqrt(D)) / (2 alpha)`` with
        ``alpha = (1+Cu^2)/(Cx^2-Cu^2)``;
      * point target (``Cx >= Cmax``, with ``Cmax = sqrt(2) Cu``): keep the
        original value.

    Parameters
    ----------
    image:
        2-D linear-power SAR image.
    window_size:
        Odd window edge length.
    looks:
        Equivalent number of looks.
    """
    import numpy as np

    arr = _check_2d(image, "gamma_map")
    if window_size % 2 == 0 or window_size < 3:
        raise SARCalibrationError(
            f"Window size must be odd and >= 3, got {window_size}.",
            f"يجب أن يكون حجم النافذة فردياً و>= 3، وجد {window_size}.",
        )
    if looks <= 0:
        raise SARCalibrationError(
            f"Number of looks must be positive, got {looks}.",
            f"يجب أن يكون عدد looks موجباً، وجد {looks}.",
        )
    pad = window_size // 2
    fill = np.nanmean(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0
    filled = np.where(np.isfinite(arr), arr, fill)
    padded = np.pad(filled, pad, mode="reflect")
    windows = _sliding_windows(padded, window_size)

    mu = windows.mean(axis=(2, 3))
    var = windows.var(axis=(2, 3))
    mu_safe = np.where(mu == 0, 1e-12, mu)

    cu = 1.0 / np.sqrt(looks)
    cmax = np.sqrt(2.0) * cu
    cx = np.sqrt(var) / mu_safe

    out = mu.copy()
    hetero = (cx > cu) & (cx < cmax)
    point = cx >= cmax
    # MAP solution on heterogeneous pixels.
    cx2 = cx**2
    denom = cx2 - cu**2
    denom = np.where(denom <= 0, 1e-12, denom)
    alpha = (1.0 + cu**2) / denom
    a_term = alpha - looks - 1.0
    d = (mu_safe**2) * (a_term**2) + 4.0 * alpha * looks * mu_safe * filled
    d = np.maximum(d, 0.0)
    x_map = (a_term * mu_safe + np.sqrt(d)) / (2.0 * alpha)
    out[hetero] = x_map[hetero]
    out[point] = filled[point]
    out[~np.isfinite(arr)] = np.nan
    return out


#: Registry of speckle filters: name -> callable(image, window_size, **kw).
SPECKLE_FILTERS = {
    "refined_lee": refined_lee,
    "frost": frost,
    "gamma_map": gamma_map,
}


def apply_speckle_filter(
    image: "np.ndarray",
    method: str = "refined_lee",
    **kwargs: float,
) -> "np.ndarray":
    """Apply a named speckle filter (default: Refined Lee).

    Raises
    ------
    SARCalibrationError
        For unknown filter names.
    """
    if method not in SPECKLE_FILTERS:
        raise SARCalibrationError(
            f"Unknown speckle filter '{method}'.",
            f"فلتر speckle غير معروف '{method}'.",
            suggestion_en=f"Use one of: {', '.join(sorted(SPECKLE_FILTERS))}.",
            suggestion_ar=f"استخدم إحدى: {', '.join(sorted(SPECKLE_FILTERS))}.",
        )
    return SPECKLE_FILTERS[method](image, **kwargs)  # type: ignore[operator]
