"""SAR unit conversion: linear power to dB plus dynamic percentile clipping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from changemaster.core.exceptions import SARCalibrationError

if TYPE_CHECKING:
    import numpy as np


def to_db(image: "np.ndarray", floor: float = 1e-10) -> "np.ndarray":
    """Convert a linear-power SAR image to decibels ``10 log10(x)``.

    Parameters
    ----------
    image:
        Linear power values (sigma0).
    floor:
        Values below this positive floor are clamped before the log so the
        result stays finite.

    Returns
    -------
    np.ndarray
        dB image (NaN preserved).
    """
    import numpy as np

    if floor <= 0:
        raise SARCalibrationError(
            f"Floor must be positive, got {floor}.",
            f"يجب أن يكون الحد الأدنى موجباً، وجد {floor}.",
        )
    arr = np.asarray(image, dtype=np.float64)
    out = 10.0 * np.log10(np.maximum(arr, floor))
    out[~np.isfinite(arr)] = np.nan
    return out


def from_db(image_db: "np.ndarray") -> "np.ndarray":
    """Convert a dB image back to linear power ``10^(x/10)``."""
    import numpy as np

    arr = np.asarray(image_db, dtype=np.float64)
    out = np.power(10.0, arr / 10.0)
    out[~np.isfinite(arr)] = np.nan
    return out


def percentile_clip(
    image: "np.ndarray",
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> "np.ndarray":
    """Clip an image to its dynamic percentile range (NaN-aware).

    Parameters
    ----------
    image:
        Input array (dB or linear).
    low_percentile / high_percentile:
        Percentiles defining the kept dynamic range.

    Raises
    ------
    SARCalibrationError
        For invalid percentile ordering or when no finite pixels exist.
    """
    import numpy as np

    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise SARCalibrationError(
            f"Invalid percentile range [{low_percentile}, {high_percentile}].",
            f"مدى نسب مئوية غير صالح [{low_percentile}, {high_percentile}].",
        )
    arr = np.asarray(image, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise SARCalibrationError(
            "Cannot clip: image has no finite pixels.",
            "تعذر القص: لا تحتوي الصورة على بكسلات منتهية.",
            suggestion_en="Check calibration output and nodata handling.",
            suggestion_ar="تحقق من ناتج المعايرة ومعالجة nodata.",
        )
    lo, hi = np.percentile(finite, [low_percentile, high_percentile])
    out = np.clip(arr, lo, hi)
    out[~np.isfinite(arr)] = np.nan
    return out
