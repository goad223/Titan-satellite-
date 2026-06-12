"""Post-detection morphological cleaning of binary change maps.

Cleans a raw binary change map with morphological opening (removes
salt-noise detections) and closing (bridges small gaps), removes connected
components smaller than a minimum area expressed in **square metres**
(converted to pixels via the pixel size from the raster metadata), and
fills small holes inside change regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import EngineError

if TYPE_CHECKING:
    import numpy as np


@dataclass
class MorphologyResult:
    """Outcome of binary-map cleaning.

    Attributes
    ----------
    cleaned:
        Boolean ``(H, W)`` cleaned change map.
    removed_regions:
        Number of regions removed by the minimum-area filter.
    filled_holes:
        Number of holes filled inside change regions.
    min_area_px:
        The minimum-area threshold actually applied, in pixels.
    warnings:
        Bilingual warnings.
    """

    cleaned: "np.ndarray"
    removed_regions: int = 0
    filled_holes: int = 0
    min_area_px: int = 1
    warnings: list[str] = field(default_factory=list)


def clean_binary_map(
    binary: "np.ndarray",
    pixel_size_m: float | None = None,
    min_area_m2: float = 400.0,
    max_hole_area_m2: float = 100.0,
    opening_radius_px: int = 1,
    closing_radius_px: int = 1,
) -> MorphologyResult:
    """Clean a binary change map morphologically.

    Steps (in order): opening → closing → minimum-area removal → small-hole
    filling. Area thresholds are given in square metres and converted to
    pixel counts with ``pixel_size_m``; when the pixel size is unknown the
    thresholds are interpreted directly as pixel counts and a warning is
    recorded.

    Parameters
    ----------
    binary:
        Boolean ``(H, W)`` raw change map.
    pixel_size_m:
        Ground pixel size in metres (from the raster geotransform).
    min_area_m2:
        Connected change regions smaller than this are removed.
    max_hole_area_m2:
        Holes inside change regions up to this size are filled.
    opening_radius_px / closing_radius_px:
        Radii (in pixels) of the structuring elements; 0 disables the step.

    Returns
    -------
    MorphologyResult
        The cleaned map plus statistics.
    """
    from changemaster.preprocessing._common import require_scipy

    require_scipy()
    import numpy as np
    from scipy import ndimage

    mask = np.asarray(binary, dtype=bool)
    if mask.ndim != 2:
        raise EngineError(
            f"Binary map must be 2-D, got {mask.ndim}-D.",
            f"يجب أن تكون الخريطة الثنائية ثنائية الأبعاد، وجد {mask.ndim} أبعاد.",
        )
    warnings: list[str] = []
    if pixel_size_m is not None and pixel_size_m > 0:
        pixel_area = pixel_size_m**2
        min_area_px = max(1, math.ceil(min_area_m2 / pixel_area))
        max_hole_px = max(0, math.floor(max_hole_area_m2 / pixel_area))
    else:
        min_area_px = max(1, int(round(min_area_m2)))
        max_hole_px = max(0, int(round(max_hole_area_m2)))
        warnings.append(
            "Pixel size unknown; area thresholds interpreted as pixel "
            "counts. | حجم البكسل غير معروف؛ تُفسَّر حدود المساحة كعدد بكسلات."
        )

    if opening_radius_px > 0 and mask.any():
        structure = _disk(opening_radius_px)
        mask = ndimage.binary_opening(mask, structure=structure)
    if closing_radius_px > 0 and mask.any():
        structure = _disk(closing_radius_px)
        mask = ndimage.binary_closing(mask, structure=structure)

    removed = 0
    if min_area_px > 1 and mask.any():
        eight = np.ones((3, 3), dtype=bool)
        labels, n = ndimage.label(mask, structure=eight)
        if n:
            sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
            small = np.flatnonzero(sizes < min_area_px) + 1
            removed = int(small.size)
            if removed:
                mask[np.isin(labels, small)] = False

    filled = 0
    if max_hole_px > 0 and mask.any():
        holes = ndimage.binary_fill_holes(mask) & ~mask
        if holes.any():
            four = ndimage.generate_binary_structure(2, 1)
            hole_labels, n_holes = ndimage.label(holes, structure=four)
            if n_holes:
                sizes = ndimage.sum_labels(
                    holes, hole_labels, index=np.arange(1, n_holes + 1)
                )
                fillable = np.flatnonzero(sizes <= max_hole_px) + 1
                filled = int(fillable.size)
                if filled:
                    mask[np.isin(hole_labels, fillable)] = True

    return MorphologyResult(
        cleaned=mask,
        removed_regions=removed,
        filled_holes=filled,
        min_area_px=min_area_px,
        warnings=warnings,
    )


def _disk(radius: int) -> "np.ndarray":
    """Boolean disk-shaped structuring element of the given pixel radius."""
    import numpy as np

    if radius < 1:
        raise EngineError(
            f"Structuring-element radius must be >= 1, got {radius}.",
            f"يجب أن يكون نصف قطر عنصر البنية 1 على الأقل، وجد {radius}.",
        )
    extent = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(extent, extent, indexing="ij")
    return (yy**2 + xx**2) <= radius**2
