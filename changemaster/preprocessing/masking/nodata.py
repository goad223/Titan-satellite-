"""Nodata-edge and saturation masking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from changemaster.core.exceptions import MaskingError

if TYPE_CHECKING:
    import numpy as np


def detect_nodata(
    image: "np.ndarray",
    nodata_value: float | None = None,
    require_all_bands: bool = False,
) -> "np.ndarray":
    """Detect nodata pixels (explicit value plus non-finite values).

    Parameters
    ----------
    image:
        2-D or ``(bands, H, W)`` array.
    nodata_value:
        Explicit nodata value from metadata (``None`` checks only NaN/Inf).
    require_all_bands:
        ``True`` marks a pixel nodata only when *all* bands are nodata;
        ``False`` (default) when *any* band is nodata.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` nodata mask.
    """
    import numpy as np

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise MaskingError(
            f"Expected 2-D or 3-D image, got {arr.ndim}-D.",
            f"المتوقع صورة ثنائية أو ثلاثية الأبعاد، وجد {arr.ndim} أبعاد.",
        )
    per_band = ~np.isfinite(arr)
    if nodata_value is not None:
        per_band |= arr == nodata_value
    return np.all(per_band, axis=0) if require_all_bands else np.any(per_band, axis=0)


def detect_saturation(
    image: "np.ndarray",
    dtype_name: str | None = None,
    saturation_value: float | None = None,
) -> "np.ndarray":
    """Detect saturated pixels (at the dtype maximum or an explicit value).

    Parameters
    ----------
    image:
        2-D or ``(bands, H, W)`` array.
    dtype_name:
        Original on-disk dtype (e.g. ``"uint16"``); its max is used as the
        saturation value for integer types.
    saturation_value:
        Explicit saturation level overriding the dtype-derived one.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` mask, ``True`` where any band is saturated.
    """
    import numpy as np

    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    level = saturation_value
    if level is None and dtype_name is not None:
        dtype = np.dtype(dtype_name)
        if np.issubdtype(dtype, np.integer):
            level = float(np.iinfo(dtype).max)
    if level is None:
        return np.zeros(arr.shape[1:], dtype=bool)
    return np.any(arr >= level, axis=0)


def detect_edges_nodata(
    image: "np.ndarray", nodata_value: float | None = None
) -> "np.ndarray":
    """Detect collar/edge nodata: nodata regions touching the image border.

    Useful for warped scenes whose rotated footprints leave nodata collars.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` mask of border-connected nodata pixels.
    """
    import numpy as np

    base = detect_nodata(image, nodata_value)
    if not base.any():
        return base
    h, w = base.shape
    # Flood-fill from border nodata pixels (iterative BFS, no recursion).
    visited = np.zeros_like(base)
    stack: list[tuple[int, int]] = []
    for c in range(w):
        if base[0, c]:
            stack.append((0, c))
        if base[h - 1, c]:
            stack.append((h - 1, c))
    for r in range(h):
        if base[r, 0]:
            stack.append((r, 0))
        if base[r, w - 1]:
            stack.append((r, w - 1))
    while stack:
        r, c = stack.pop()
        if visited[r, c] or not base[r, c]:
            continue
        visited[r, c] = True
        if r > 0:
            stack.append((r - 1, c))
        if r < h - 1:
            stack.append((r + 1, c))
        if c > 0:
            stack.append((r, c - 1))
        if c < w - 1:
            stack.append((r, c + 1))
    return visited
