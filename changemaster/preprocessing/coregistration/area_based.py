"""Area-based registration: phase correlation plus ECC refinement.

Phase correlation provides robust sub-pixel translation estimates; ECC
(Enhanced Correlation Coefficient) refines a full affine warp. Both run on
a grid of windows distributed over the overlap so local shifts are sampled
everywhere — windows shrink on weaker hardware, accuracy stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import CoregistrationError
from changemaster.core.hardware import HardwareInfo
from changemaster.preprocessing._common import (
    adaptive_window_count,
    normalize_to_uint8,
    require_cv2,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass
class WindowShift:
    """Sub-pixel shift measured in one correlation window.

    Attributes
    ----------
    center_xy:
        Window centre ``(x, y)`` in pixel coordinates.
    shift_xy:
        Measured ``(dx, dy)`` shift (moving relative to reference).
    response:
        Phase-correlation peak response (0-1 confidence proxy).
    """

    center_xy: tuple[float, float]
    shift_xy: tuple[float, float]
    response: float


@dataclass
class AreaRegistrationResult:
    """Result of grid-based area registration.

    Attributes
    ----------
    global_shift_xy:
        Response-weighted mean ``(dx, dy)`` over all windows.
    window_shifts:
        Per-window measurements (usable as a displacement field sample).
    ecc_matrix:
        Refined ``(2, 3)`` affine matrix when ECC succeeded, else ``None``.
    """

    global_shift_xy: tuple[float, float]
    window_shifts: list[WindowShift] = field(default_factory=list)
    ecc_matrix: "np.ndarray | None" = None


def phase_correlation_shift(
    reference: "np.ndarray", moving: "np.ndarray"
) -> tuple[tuple[float, float], float]:
    """Sub-pixel ``(dx, dy)`` shift between two same-shape 2-D windows.

    Returns the shift to apply to ``moving`` to align it with ``reference``
    along with the correlation peak response.
    """
    cv2 = require_cv2()
    import numpy as np

    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    if ref.shape != mov.shape or ref.ndim != 2:
        raise CoregistrationError(
            f"Phase correlation needs same-shape 2-D windows; got {ref.shape} vs {mov.shape}.",
            f"الارتباط الطوري يتطلب نافذتين ثنائيتي الأبعاد بنفس الشكل؛ وجد {ref.shape} و{mov.shape}.",
        )
    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(mov, ref, window)
    return (float(dx), float(dy)), float(response)


def ecc_refine(
    reference: "np.ndarray",
    moving: "np.ndarray",
    initial_matrix: "np.ndarray | None" = None,
    iterations: int = 200,
    eps: float = 1e-6,
) -> "np.ndarray | None":
    """Refine an affine warp with the ECC algorithm.

    Returns the refined ``(2, 3)`` affine matrix mapping ``moving`` into the
    reference frame, or ``None`` when ECC fails to converge.
    """
    cv2 = require_cv2()
    import numpy as np

    ref8 = normalize_to_uint8(reference).astype(np.float32) / 255.0
    mov8 = normalize_to_uint8(moving).astype(np.float32) / 255.0
    warp = (
        np.asarray(initial_matrix, dtype=np.float32)
        if initial_matrix is not None
        else np.eye(2, 3, dtype=np.float32)
    )
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        _, warp = cv2.findTransformECC(ref8, mov8, warp, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        return None
    return np.asarray(warp, dtype=np.float64)


def _window_grid(
    height: int, width: int, n_windows: int, window_size: int
) -> list[tuple[int, int]]:
    """Top-left corners of an approximately square window grid."""
    import math

    per_side = max(1, int(math.sqrt(n_windows)))
    rows = max(1, height - window_size)
    cols = max(1, width - window_size)
    corners: list[tuple[int, int]] = []
    for i in range(per_side):
        for j in range(per_side):
            r = int(i * rows / max(1, per_side - 1)) if per_side > 1 else rows // 2
            c = int(j * cols / max(1, per_side - 1)) if per_side > 1 else cols // 2
            corners.append((min(r, rows), min(c, cols)))
    return corners


def grid_phase_correlation(
    reference: "np.ndarray",
    moving: "np.ndarray",
    hardware: HardwareInfo | None = None,
    window_size: int | None = None,
    min_response: float = 0.05,
    refine_with_ecc: bool = True,
) -> AreaRegistrationResult:
    """Measure shifts on a distributed window grid and refine with ECC.

    Parameters
    ----------
    reference / moving:
        2-D same-shape band arrays.
    hardware:
        Used to adapt the number of grid windows (more on faster machines).
    window_size:
        Correlation window edge; defaults to an image-size-derived value.
    min_response:
        Windows below this phase-correlation response are discarded.
    refine_with_ecc:
        Run a final whole-image ECC affine refinement.

    Raises
    ------
    CoregistrationError
        When no window produces a usable correlation response.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float32)
    mov = np.asarray(moving, dtype=np.float32)
    if ref.shape != mov.shape or ref.ndim != 2:
        raise CoregistrationError(
            f"Grid correlation needs same-shape 2-D arrays; got {ref.shape} vs {mov.shape}.",
            f"يتطلب الارتباط الشبكي مصفوفتين ثنائيتي الأبعاد بنفس الشكل؛ وجد {ref.shape} و{mov.shape}.",
        )
    h, w = ref.shape
    if window_size is None:
        window_size = int(min(256, max(32, min(h, w) // 4)))
    window_size = min(window_size, h, w)
    n_windows = adaptive_window_count(hardware)

    shifts: list[WindowShift] = []
    for r, c in _window_grid(h, w, n_windows, window_size):
        ref_win = ref[r : r + window_size, c : c + window_size]
        mov_win = mov[r : r + window_size, c : c + window_size]
        if ref_win.std() < 1e-6 or mov_win.std() < 1e-6:
            continue
        (dx, dy), response = phase_correlation_shift(ref_win, mov_win)
        if response < min_response:
            continue
        center = (c + window_size / 2.0, r + window_size / 2.0)
        shifts.append(WindowShift(center_xy=center, shift_xy=(dx, dy), response=response))

    if not shifts:
        raise CoregistrationError(
            "No correlation window produced a usable response.",
            "لم تنتج أي نافذة ارتباط استجابة قابلة للاستخدام.",
            suggestion_en="Check image overlap and contrast, or use feature-based registration.",
            suggestion_ar="تحقق من تداخل الصورتين وتباينهما، أو استخدم التسجيل القائم على الميزات.",
        )

    weights = np.asarray([s.response for s in shifts], dtype=np.float64)
    dxs = np.asarray([s.shift_xy[0] for s in shifts], dtype=np.float64)
    dys = np.asarray([s.shift_xy[1] for s in shifts], dtype=np.float64)
    weights_sum = float(weights.sum())
    global_shift = (
        float(np.dot(weights, dxs) / weights_sum),
        float(np.dot(weights, dys) / weights_sum),
    )

    ecc_matrix: "np.ndarray | None" = None
    if refine_with_ecc:
        initial = np.array(
            [[1.0, 0.0, -global_shift[0]], [0.0, 1.0, -global_shift[1]]], dtype=np.float32
        )
        ecc_matrix = ecc_refine(ref, mov, initial_matrix=initial)

    return AreaRegistrationResult(
        global_shift_xy=global_shift, window_shifts=shifts, ecc_matrix=ecc_matrix
    )
