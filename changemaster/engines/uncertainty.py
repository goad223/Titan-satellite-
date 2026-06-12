"""Per-pixel uncertainty estimation for fused change maps.

The uncertainty map combines three documented evidence sources:

1. **Engine disagreement** — the standard deviation of the per-engine
   probability maps at each pixel.
2. **Threshold proximity** — how close the fused probability is to the
   decision threshold.
3. **Mask-boundary proximity** — pixels within a small distance of masked
   (cloud/shadow/nodata) areas inherit extra uncertainty because their
   preprocessing context is less reliable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from changemaster.core.exceptions import EngineError

if TYPE_CHECKING:
    import numpy as np


def compute_uncertainty(
    probability_maps: "list[np.ndarray]",
    fused_probability: "np.ndarray",
    threshold: float,
    valid_mask: "np.ndarray",
    boundary_radius_px: int = 2,
    weights: tuple[float, float, float] = (0.5, 0.35, 0.15),
) -> "np.ndarray":
    """Compute the per-pixel uncertainty map in ``[0, 1]``.

    Formula (documented contract)::

        disagreement(p) = std(P_1..P_n)(p) / 0.5          # 0.5 = max std on [0,1]
        proximity(p)    = 1 - |P_fused(p) - t| / max(t, 1 - t)
        boundary(p)     = 1 if p within `boundary_radius_px` of a masked
                          pixel (but itself valid) else 0
        U(p) = w1 * disagreement + w2 * proximity + w3 * boundary

    with default weights ``(w1, w2, w3) = (0.5, 0.35, 0.15)``. Masked pixels
    receive the maximum uncertainty of 1.0 (they were never evaluated).

    Parameters
    ----------
    probability_maps:
        Per-engine probability maps, each ``(H, W)`` in ``[0, 1]``.
    fused_probability:
        Fused probability map ``(H, W)``.
    threshold:
        Decision threshold applied to the fused map.
    valid_mask:
        Boolean ``(H, W)`` mask of evaluated pixels.
    boundary_radius_px:
        Pixel radius of the mask-boundary influence zone.
    weights:
        ``(w1, w2, w3)`` weights of the three terms; they are normalized to
        sum to 1.

    Returns
    -------
    numpy.ndarray
        Float32 ``(H, W)`` uncertainty map in ``[0, 1]``.
    """
    import numpy as np

    if not probability_maps:
        raise EngineError(
            "At least one probability map is required for uncertainty.",
            "مطلوب خريطة احتمالات واحدة على الأقل لحساب عدم اليقين.",
        )
    stack = np.stack([np.asarray(p, dtype=np.float32) for p in probability_maps])
    if stack.shape[1:] != valid_mask.shape or fused_probability.shape != valid_mask.shape:
        raise EngineError(
            "Probability maps and validity mask shapes do not match.",
            "أشكال خرائط الاحتمالات وقناع الصلاحية غير متطابقة.",
        )
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    disagreement = np.clip(stack.std(axis=0) / 0.5, 0.0, 1.0)
    denom = max(threshold, 1.0 - threshold)
    proximity = np.clip(
        1.0 - np.abs(np.nan_to_num(fused_probability, nan=threshold) - threshold) / denom,
        0.0,
        1.0,
    )
    boundary = _mask_boundary_zone(valid_mask, boundary_radius_px)

    uncertainty = (
        w[0] * disagreement + w[1] * proximity + w[2] * boundary.astype(np.float32)
    ).astype(np.float32)
    uncertainty[~valid_mask] = 1.0
    return np.clip(uncertainty, 0.0, 1.0)


def _mask_boundary_zone(valid_mask: "np.ndarray", radius_px: int) -> "np.ndarray":
    """Boolean zone of valid pixels within ``radius_px`` of any masked pixel."""
    import numpy as np

    if radius_px <= 0 or valid_mask.all():
        return np.zeros(valid_mask.shape, dtype=bool)
    from changemaster.preprocessing._common import require_scipy

    require_scipy()
    from scipy import ndimage

    invalid = ~valid_mask
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(invalid, structure=structure, iterations=radius_px)
    return dilated & valid_mask
