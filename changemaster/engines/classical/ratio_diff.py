"""Log-ratio (SAR) and normalized difference (optical) change indices.

For SAR pairs the standard operator is the absolute log-ratio
``|log(moving / reference)|`` (computed in dB via the Phase-2 converter),
which is robust to multiplicative speckle. For optical pairs the operator
is the per-band normalized absolute difference
``|moving - reference| / (|moving| + |reference| + eps)`` averaged over
bands. Both are rescaled into ``[0, 1]`` by a robust percentile computed
over valid pixels only, accumulated chunk by chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import EngineError
from changemaster.core.hardware import HardwareInfo
from changemaster.preprocessing._common import adaptive_tile_size

if TYPE_CHECKING:
    import numpy as np


@dataclass
class RatioDiffResult:
    """Outcome of the ratio/difference change index.

    Attributes
    ----------
    probability:
        Float32 ``(H, W)`` normalized change index in ``[0, 1]``
        (``NaN`` on invalid pixels).
    index:
        Raw (unnormalized) change index.
    operator:
        ``"log_ratio"`` (SAR) or ``"normalized_difference"`` (optical).
    warnings:
        Bilingual warnings.
    """

    probability: "np.ndarray"
    index: "np.ndarray"
    operator: str
    warnings: list[str] = field(default_factory=list)


def log_ratio_index(
    reference: "np.ndarray", moving: "np.ndarray", floor: float = 1e-10
) -> "np.ndarray":
    """Absolute log-ratio ``|10*log10(mov) - 10*log10(ref)|`` per pixel.

    Multi-band inputs are reduced by the per-band mean. Values are computed
    through the Phase-2 dB converter so the floor handling is identical.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` linear-scale (intensity/amplitude) SAR arrays. If
        the input is already in dB (negative-mean heuristic), the plain
        difference is used instead.
    floor:
        Linear floor forwarded to the dB conversion.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` absolute log-ratio in dB.
    """
    import numpy as np

    from changemaster.preprocessing.sar.convert import to_db

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    # Heuristic: dB images have negative means; linear intensities don't.
    if float(np.nanmean(ref)) < 0.0 or float(np.nanmean(mov)) < 0.0:
        diff = mov - ref
    else:
        diff = np.stack([to_db(b, floor=floor) for b in mov]) - np.stack(
            [to_db(b, floor=floor) for b in ref]
        )
    return np.abs(diff).mean(axis=0)


def normalized_difference_index(
    reference: "np.ndarray", moving: "np.ndarray", eps: float = 1e-9
) -> "np.ndarray":
    """Per-band normalized absolute difference, averaged across bands.

    ``d_b = |mov_b - ref_b| / (|mov_b| + |ref_b| + eps)`` and the result is
    the mean over bands — bounded in ``[0, 1]`` by construction.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` optical arrays.
    eps:
        Stabilizer against division by zero.

    Returns
    -------
    numpy.ndarray
        ``(H, W)`` normalized difference index.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    d = np.abs(mov - ref) / (np.abs(mov) + np.abs(ref) + eps)
    return d.mean(axis=0)


def ratio_diff_change(
    reference: "np.ndarray",
    moving: "np.ndarray",
    valid_mask: "np.ndarray",
    mode: str = "optical",
    hardware: HardwareInfo | None = None,
    clip_percentile: float = 99.0,
) -> RatioDiffResult:
    """Compute the ratio/difference change probability for a pair.

    Selects the log-ratio operator for SAR and the normalized difference for
    optical imagery, computes it chunk by chunk, and rescales to ``[0, 1]``
    with a robust percentile of the valid values.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` co-registered, normalized arrays.
    valid_mask:
        Boolean ``(H, W)`` mask; invalid pixels are excluded from the
        normalization statistics and are ``NaN`` in the output.
    mode:
        ``"optical"`` or ``"sar"``.
    hardware:
        Hardware snapshot for adaptive chunk sizing.
    clip_percentile:
        Percentile of valid index values mapped to probability 1.0.

    Returns
    -------
    RatioDiffResult
        Probability map, raw index and the operator used.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.ndim != 3:
        raise EngineError(
            f"Ratio/diff needs identical (bands, H, W) arrays; got "
            f"{ref.shape} vs {mov.shape}.",
            f"يتطلب مؤشر النسبة/الفرق مصفوفتين متطابقتين (bands, H, W)؛ وجد "
            f"{ref.shape} و{mov.shape}.",
        )
    if mode not in ("optical", "sar"):
        raise EngineError(
            f"Unknown mode '{mode}' for ratio/diff.",
            f"وضع غير معروف '{mode}' لمؤشر النسبة/الفرق.",
            suggestion_en="Use 'optical' or 'sar'.",
            suggestion_ar="استخدم 'optical' أو 'sar'.",
        )
    bands, height, width = ref.shape
    chunk_rows = max(64, adaptive_tile_size(hardware) // max(1, bands))
    operator = "log_ratio" if mode == "sar" else "normalized_difference"

    index = np.full((height, width), np.nan, dtype=np.float32)
    for r0 in range(0, height, chunk_rows):
        sl = slice(r0, min(height, r0 + chunk_rows))
        if mode == "sar":
            chunk = log_ratio_index(ref[:, sl], mov[:, sl])
        else:
            chunk = normalized_difference_index(ref[:, sl], mov[:, sl])
        index[sl] = np.where(valid_mask[sl], chunk, np.nan).astype(np.float32)

    finite = index[np.isfinite(index)]
    if finite.size == 0:
        raise EngineError(
            "No valid pixels for the ratio/difference index.",
            "لا توجد بكسلات صالحة لمؤشر النسبة/الفرق.",
            suggestion_en="Check the validity mask coverage.",
            suggestion_ar="تحقق من تغطية قناع الصلاحية.",
        )
    scale = float(np.percentile(finite, clip_percentile))
    if scale <= 0:
        scale = 1.0
    probability = np.clip(index / scale, 0.0, 1.0).astype(np.float32)
    probability[~valid_mask] = np.nan
    return RatioDiffResult(probability=probability, index=index, operator=operator)
