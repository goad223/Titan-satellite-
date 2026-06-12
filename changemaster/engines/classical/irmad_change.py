"""IR-MAD based change probability (reuses the Phase-2 implementation).

This module does **not** reimplement IR-MAD. It imports
:func:`changemaster.preprocessing.radiometric.irmad.compute_irmad` (the full
Phase-2 iteratively reweighted MAD with tiled covariance accumulation) and
converts its chi-square statistic into a per-pixel change probability via
the chi-square cumulative distribution function:
``P(change) = CDF_chi2(chi_square; df = bands)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import EngineError, RadiometricError
from changemaster.core.hardware import HardwareInfo
from changemaster.preprocessing._common import adaptive_tile_size
from changemaster.preprocessing.radiometric.irmad import compute_irmad

if TYPE_CHECKING:
    import numpy as np


@dataclass
class IRMADChangeResult:
    """Outcome of IR-MAD change detection.

    Attributes
    ----------
    probability:
        Float32 ``(H, W)`` change probability ``P(change) = 1 - P(no
        change)`` from the chi-square distribution (``NaN`` on invalid
        pixels).
    chi_square:
        Raw per-pixel chi-square statistic.
    canonical_correlations:
        Final canonical correlations of the IR-MAD iteration.
    iterations:
        Number of IR-MAD reweighting iterations executed.
    converged:
        ``True`` when IR-MAD converged.
    warnings:
        Bilingual warnings.
    """

    probability: "np.ndarray"
    chi_square: "np.ndarray"
    canonical_correlations: "np.ndarray"
    iterations: int
    converged: bool
    warnings: list[str] = field(default_factory=list)


def irmad_change(
    reference: "np.ndarray",
    moving: "np.ndarray",
    valid_mask: "np.ndarray",
    max_iterations: int = 30,
    hardware: HardwareInfo | None = None,
) -> IRMADChangeResult:
    """Compute IR-MAD change probability for a preprocessed pair.

    Runs the Phase-2 :func:`compute_irmad` (MAD variates + iterative
    chi-square reweighting, tiled covariance accumulation) and converts the
    chi-square statistic of each pixel into ``P(change)``.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` co-registered, normalized arrays.
    valid_mask:
        Boolean ``(H, W)`` mask; invalid pixels never enter the IR-MAD
        statistics and are ``NaN`` in the output.
    max_iterations:
        IR-MAD reweighting iteration cap.
    hardware:
        Hardware snapshot for adaptive chunk sizing.

    Returns
    -------
    IRMADChangeResult
        Change probability plus the raw chi-square map and convergence info.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.ndim != 3:
        raise EngineError(
            f"IR-MAD change needs identical (bands, H, W) arrays; got "
            f"{ref.shape} vs {mov.shape}.",
            f"يتطلب كشف IR-MAD مصفوفتين متطابقتين (bands, H, W)؛ وجد "
            f"{ref.shape} و{mov.shape}.",
        )
    bands = ref.shape[0]
    chunk_rows = max(64, adaptive_tile_size(hardware) // max(1, bands))
    try:
        result = compute_irmad(
            ref,
            mov,
            valid_mask=valid_mask,
            max_iterations=max_iterations,
            chunk_rows=chunk_rows,
        )
    except RadiometricError as exc:
        raise EngineError(
            f"IR-MAD change detection failed: {exc.message_en}",
            f"فشل كشف التغيرات بطريقة IR-MAD: {exc.message_ar}",
            suggestion_en="Check the mask coverage and band count of the pair.",
            suggestion_ar="تحقق من تغطية القناع وعدد نطاقات الزوج.",
        ) from exc

    probability = np.asarray(
        1.0 - result.no_change_probability, dtype=np.float32
    )
    probability = np.clip(probability, 0.0, 1.0)
    probability[~valid_mask] = np.nan
    warnings = list(result.warnings)
    if not result.converged:
        warnings.append(
            f"IR-MAD did not converge within {result.iterations} iterations; "
            "the probability map may be less stable. | لم يتقارب IR-MAD خلال "
            f"{result.iterations} تكراراً؛ قد تكون خريطة الاحتمالات أقل استقراراً."
        )
    return IRMADChangeResult(
        probability=probability,
        chi_square=np.asarray(result.chi_square, dtype=np.float32),
        canonical_correlations=result.canonical_correlations,
        iterations=result.iterations,
        converged=result.converged,
        warnings=warnings,
    )
