"""Cloud-shadow detection: geometric projection from the sun + NIR test.

Each cloud is projected along the anti-solar azimuth over a plausible
cloud-height range; candidate shadow pixels must also be dark in NIR.
Every confirmed shadow region is therefore *linked to its cloud* by the
projection geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import MaskingError

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ShadowDetectionResult:
    """Cloud-shadow detection output.

    Attributes
    ----------
    shadow:
        Boolean ``(H, W)`` confirmed shadow mask.
    candidate:
        Geometric candidate region before the NIR confirmation.
    heights_tested_m:
        Cloud heights (metres) tested in the projection sweep.
    """

    shadow: "np.ndarray"
    candidate: "np.ndarray"
    heights_tested_m: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def project_cloud_shadow(
    cloud_mask: "np.ndarray",
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    pixel_size_m: float,
    cloud_height_m: float,
) -> "np.ndarray":
    """Project a cloud mask to its shadow position for one cloud height.

    The shadow of a cloud at height ``h`` is displaced from the cloud by
    ``d = h / tan(elevation)`` along the anti-solar azimuth.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` projected shadow candidate mask.
    """
    import numpy as np

    if not 0.0 < sun_elevation_deg < 90.0:
        raise MaskingError(
            f"Sun elevation must be in (0, 90) degrees, got {sun_elevation_deg}.",
            f"يجب أن يكون ارتفاع الشمس بين 0 و90 درجة، وجد {sun_elevation_deg}.",
            suggestion_en="Read SUN_ELEVATION / MEAN_SUN_ANGLE from the product metadata.",
            suggestion_ar="اقرأ زاوية ارتفاع الشمس من ميتاداتا المنتج.",
        )
    if pixel_size_m <= 0:
        raise MaskingError(
            f"Pixel size must be positive, got {pixel_size_m}.",
            f"يجب أن يكون حجم البكسل موجباً، وجد {pixel_size_m}.",
        )
    mask = np.asarray(cloud_mask, dtype=bool)
    distance_m = cloud_height_m / math.tan(math.radians(sun_elevation_deg))
    distance_px = distance_m / pixel_size_m
    # Shadow falls on the opposite side of the sun.
    azimuth_rad = math.radians(sun_azimuth_deg)
    shift_x = -distance_px * math.sin(azimuth_rad)
    shift_y = distance_px * math.cos(azimuth_rad)
    # Note: image rows grow downward (south) for north-up rasters.
    dy = int(round(shift_y))
    dx = int(round(shift_x))
    shifted = np.zeros_like(mask)
    h, w = mask.shape
    src_r0, src_r1 = max(0, -dy), min(h, h - dy)
    src_c0, src_c1 = max(0, -dx), min(w, w - dx)
    dst_r0, dst_r1 = max(0, dy), min(h, h + dy)
    dst_c0, dst_c1 = max(0, dx), min(w, w + dx)
    if src_r1 > src_r0 and src_c1 > src_c0:
        shifted[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return shifted


def detect_shadows(
    cloud_mask: "np.ndarray",
    nir: "np.ndarray",
    sun_elevation_deg: float,
    sun_azimuth_deg: float,
    pixel_size_m: float,
    cloud_height_range_m: tuple[float, float] = (300.0, 3000.0),
    height_steps: int = 6,
    nir_percentile: float = 25.0,
    valid_mask: "np.ndarray | None" = None,
) -> ShadowDetectionResult:
    """Detect cloud shadows by sun-geometry projection plus NIR darkness.

    Parameters
    ----------
    cloud_mask:
        Boolean ``(H, W)`` cloud mask (each shadow is linked to these
        clouds through the projection).
    nir:
        NIR band used to confirm darkness of candidate pixels.
    sun_elevation_deg / sun_azimuth_deg:
        Solar geometry from the product metadata.
    pixel_size_m:
        Ground pixel size in metres.
    cloud_height_range_m:
        Sweep range of plausible cloud heights.
    height_steps:
        Number of heights tested across the range.
    nir_percentile:
        Pixels darker than this NIR percentile (over valid pixels) are
        accepted as shadow.
    valid_mask:
        Boolean ``(H, W)`` of usable pixels for the percentile statistics.

    Returns
    -------
    ShadowDetectionResult
        Confirmed shadow mask and diagnostics.
    """
    import numpy as np

    clouds = np.asarray(cloud_mask, dtype=bool)
    nir_arr = np.asarray(nir, dtype=np.float64)
    if clouds.shape != nir_arr.shape:
        raise MaskingError(
            f"Cloud mask and NIR band shapes differ: {clouds.shape} vs {nir_arr.shape}.",
            f"شكل قناع الغيوم لا يطابق نطاق NIR‏: {clouds.shape} مقابل {nir_arr.shape}.",
        )
    warnings: list[str] = []
    if not clouds.any():
        empty = np.zeros_like(clouds)
        return ShadowDetectionResult(shadow=empty, candidate=empty.copy())

    lo, hi = cloud_height_range_m
    heights = [lo + i * (hi - lo) / max(1, height_steps - 1) for i in range(height_steps)]
    candidate = np.zeros_like(clouds)
    for height in heights:
        candidate |= project_cloud_shadow(
            clouds, sun_elevation_deg, sun_azimuth_deg, pixel_size_m, height
        )
    candidate &= ~clouds  # a pixel cannot be cloud and its own shadow

    valid = np.ones_like(clouds) if valid_mask is None else (valid_mask & ~clouds)
    finite = np.isfinite(nir_arr)
    sample = nir_arr[valid & finite]
    if sample.size == 0:
        warnings.append(
            "No valid NIR pixels for shadow confirmation; geometric candidates "
            "returned unconfirmed. | لا توجد بكسلات NIR صالحة لتأكيد الظلال؛ "
            "أعيدت المرشحات الهندسية دون تأكيد."
        )
        return ShadowDetectionResult(
            shadow=candidate, candidate=candidate, heights_tested_m=heights, warnings=warnings
        )
    dark_threshold = float(np.percentile(sample, nir_percentile))
    dark = finite & (nir_arr <= dark_threshold)
    shadow = candidate & dark
    return ShadowDetectionResult(
        shadow=shadow, candidate=candidate, heights_tested_m=heights, warnings=warnings
    )
