"""Shared internal helpers for the preprocessing package.

Lazy imports of heavy dependencies (OpenCV, SciPy) and hardware-adaptive
window sizing. On weaker machines we use *smaller* processing windows —
never lower accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import DependencyMissingError
from changemaster.core.hardware import HardwareInfo, HardwareTier, detect_hardware

if TYPE_CHECKING:
    import numpy as np


def require_cv2() -> Any:
    """Import and return :mod:`cv2`, raising a bilingual error when missing."""
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise DependencyMissingError(
            "opencv-python-headless",
            "image registration and filtering",
            "تسجيل الصور والترشيح",
        ) from exc


def require_scipy() -> Any:
    """Import and return :mod:`scipy`, raising a bilingual error when missing."""
    try:
        import scipy

        return scipy
    except ImportError as exc:
        raise DependencyMissingError(
            "scipy", "statistical preprocessing algorithms", "خوارزميات المعالجة الإحصائية"
        ) from exc


def adaptive_tile_size(hardware: HardwareInfo | None = None) -> int:
    """Return the tile edge length adapted to the machine's hardware tier.

    Weaker machines get smaller tiles (bounded memory) while keeping the
    exact same algorithms and accuracy.
    """
    hw = hardware if hardware is not None else detect_hardware()
    return hw.recommended_tile_size


def adaptive_window_count(hardware: HardwareInfo | None = None) -> int:
    """Number of sampling/refinement windows adapted to the hardware tier."""
    hw = hardware if hardware is not None else detect_hardware()
    if hw.tier is HardwareTier.LOW:
        return 9
    if hw.tier is HardwareTier.MEDIUM:
        return 16
    return 25


def to_float32(array: "np.ndarray") -> "np.ndarray":
    """Return ``array`` as a float32 copy (no-op view when already float32)."""
    import numpy as np

    return np.asarray(array, dtype=np.float32)


def normalize_to_uint8(band: "np.ndarray") -> "np.ndarray":
    """Percentile-stretch a 2-D band to uint8 for feature detectors."""
    import numpy as np

    data = np.asarray(band, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [2.0, 98.0])
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled = (np.clip(data, low, high) - low) / (high - low) * 255.0
    scaled[~np.isfinite(data)] = 0.0
    return scaled.astype(np.uint8)
