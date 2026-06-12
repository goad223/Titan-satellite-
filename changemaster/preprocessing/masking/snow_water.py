"""Snow and water masking via NDSI, NDWI and MNDWI spectral indices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from changemaster.core.exceptions import MaskingError

if TYPE_CHECKING:
    import numpy as np


def _normalized_difference(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Safe normalized difference ``(a - b) / (a + b)`` (NaN where undefined)."""
    import numpy as np

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise MaskingError(
            f"Band shapes differ: {x.shape} vs {y.shape}.",
            f"شكلا النطاقين مختلفان: {x.shape} مقابل {y.shape}.",
        )
    denom = x + y
    out = np.full(x.shape, np.nan)
    ok = denom != 0
    out[ok] = (x[ok] - y[ok]) / denom[ok]
    return out


def ndsi(green: "np.ndarray", swir: "np.ndarray") -> "np.ndarray":
    """Normalized Difference Snow Index ``(green - swir) / (green + swir)``."""
    return _normalized_difference(green, swir)


def ndwi(green: "np.ndarray", nir: "np.ndarray") -> "np.ndarray":
    """Normalized Difference Water Index ``(green - nir) / (green + nir)``."""
    return _normalized_difference(green, nir)


def mndwi(green: "np.ndarray", swir: "np.ndarray") -> "np.ndarray":
    """Modified NDWI ``(green - swir) / (green + swir)`` (better in urban areas)."""
    return _normalized_difference(green, swir)


def detect_snow(
    green: "np.ndarray",
    swir: "np.ndarray",
    nir: "np.ndarray | None" = None,
    ndsi_threshold: float = 0.40,
    green_min: float = 0.10,
    reflectance_scale: float = 1.0,
) -> "np.ndarray":
    """Snow mask: ``NDSI > threshold`` plus a brightness check.

    Parameters
    ----------
    green / swir / nir:
        Reflectance bands (NIR refines the snow/water separation).
    ndsi_threshold:
        Standard threshold 0.4.
    green_min:
        Minimum green reflectance — rules out dark NDSI-positive water.
    reflectance_scale:
        DN-to-reflectance divisor.
    """
    import numpy as np

    g = np.asarray(green, dtype=np.float64) / reflectance_scale
    s = np.asarray(swir, dtype=np.float64) / reflectance_scale
    index = ndsi(g, s)
    mask = (index > ndsi_threshold) & (g > green_min)
    if nir is not None:
        n = np.asarray(nir, dtype=np.float64) / reflectance_scale
        mask &= n > 0.11  # snow stays bright in NIR; water does not
    return mask & np.isfinite(index)


def detect_water(
    green: "np.ndarray",
    nir: "np.ndarray | None" = None,
    swir: "np.ndarray | None" = None,
    ndwi_threshold: float = 0.0,
    mndwi_threshold: float = 0.0,
    reflectance_scale: float = 1.0,
) -> "np.ndarray":
    """Water mask via NDWI (green/NIR) and/or MNDWI (green/SWIR).

    When both NIR and SWIR are available the result is the union of the two
    indices, which captures both open and turbid/urban-shaded water.

    Raises
    ------
    MaskingError
        When neither NIR nor SWIR is provided.
    """
    import numpy as np

    if nir is None and swir is None:
        raise MaskingError(
            "Water detection needs a NIR or SWIR band alongside green.",
            "يتطلب كشف المياه نطاق NIR أو SWIR إضافة إلى الأخضر.",
            suggestion_en="Map the sensor's NIR (e.g. B08) or SWIR (e.g. B11) band.",
            suggestion_ar="حدد نطاق NIR (مثل B08) أو SWIR (مثل B11) للمستشعر.",
        )
    g = np.asarray(green, dtype=np.float64) / reflectance_scale
    mask = np.zeros(g.shape, dtype=bool)
    if nir is not None:
        n = np.asarray(nir, dtype=np.float64) / reflectance_scale
        index = ndwi(g, n)
        mask |= np.isfinite(index) & (index > ndwi_threshold)
    if swir is not None:
        s = np.asarray(swir, dtype=np.float64) / reflectance_scale
        index = mndwi(g, s)
        mask |= np.isfinite(index) & (index > mndwi_threshold)
    return mask
