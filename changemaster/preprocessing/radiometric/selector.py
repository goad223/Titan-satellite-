"""Automatic selection of the best radiometric normalization method.

Chooses between histogram matching, PIF linear normalization and IR-MAD
based on the sensor pair, band count, valid-pixel budget and acquisition
gap — then runs the chosen method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import RadiometricError
from changemaster.preprocessing.radiometric.histogram_matching import match_histograms
from changemaster.preprocessing.radiometric.irmad import (
    apply_irmad_normalization,
    compute_irmad,
)
from changemaster.preprocessing.radiometric.pif_normalization import (
    apply_pif_normalization,
    fit_pif_linear,
)

if TYPE_CHECKING:
    import numpy as np

#: Method identifiers.
METHOD_HISTOGRAM = "histogram_matching"
METHOD_PIF = "pif_normalization"
METHOD_IRMAD = "irmad"


@dataclass
class RadiometricSelection:
    """Chosen radiometric method, its rationale and the normalized output.

    Attributes
    ----------
    method:
        One of :data:`METHOD_HISTOGRAM`, :data:`METHOD_PIF`, :data:`METHOD_IRMAD`.
    reason:
        Bilingual explanation of why this method was selected.
    normalized:
        Normalized moving image ``(bands, H, W)``.
    details:
        Method-specific diagnostics (e.g. IR-MAD iterations, PIF count).
    warnings:
        Accumulated bilingual warnings.
    """

    method: str
    reason: str
    normalized: "np.ndarray"
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def choose_method(
    band_count: int,
    valid_pixel_count: int,
    same_sensor: bool,
    acquisition_gap_days: float | None,
) -> tuple[str, str]:
    """Pick the most suitable method for a pair; returns ``(method, reason)``.

    Heuristics:
      * >= 2 bands and enough pixels -> IR-MAD (statistically strongest).
      * single band but enough pixels -> PIF linear normalization.
      * same sensor with a short revisit gap, or tiny pixel budget ->
        histogram matching (cheap and adequate).
    """
    if same_sensor and acquisition_gap_days is not None and acquisition_gap_days <= 30:
        return (
            METHOD_HISTOGRAM,
            "Same sensor and short acquisition gap; histogram matching suffices. | "
            "نفس المستشعر وفارق زمني قصير؛ مطابقة الهيستوغرام كافية.",
        )
    if band_count >= 2 and valid_pixel_count >= 1000 * band_count:
        return (
            METHOD_IRMAD,
            "Multi-band pair with a large valid-pixel budget; IR-MAD is the most robust. | "
            "زوج متعدد النطاقات ببكسلات صالحة وفيرة؛ IR-MAD هو الأكثر متانة.",
        )
    if valid_pixel_count >= 500:
        return (
            METHOD_PIF,
            "Limited bands; PIF linear normalization chosen. | "
            "نطاقات محدودة؛ اختير التطبيع الخطي عبر PIF.",
        )
    return (
        METHOD_HISTOGRAM,
        "Small valid-pixel budget; histogram matching is the safest default. | "
        "بكسلات صالحة قليلة؛ مطابقة الهيستوغرام هي الخيار الآمن.",
    )


def normalize_pair(
    reference: "np.ndarray",
    moving: "np.ndarray",
    valid_mask: "np.ndarray | None" = None,
    reference_sensor: str | None = None,
    moving_sensor: str | None = None,
    reference_datetime: datetime | None = None,
    moving_datetime: datetime | None = None,
    method: str | None = None,
) -> RadiometricSelection:
    """Normalize ``moving`` to ``reference`` using the best (or given) method.

    Parameters
    ----------
    reference / moving:
        Co-registered ``(bands, H, W)`` arrays.
    valid_mask:
        Boolean ``(H, W)`` usable-pixel mask.
    reference_sensor / moving_sensor:
        Sensor identifiers used in method selection.
    reference_datetime / moving_datetime:
        Acquisition times used to compute the gap.
    method:
        Force a specific method; ``None`` selects automatically.

    Returns
    -------
    RadiometricSelection
        Chosen method, normalized image and diagnostics.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.ndim == 2:
        ref = ref[np.newaxis]
    if mov.ndim == 2:
        mov = mov[np.newaxis]
    if ref.shape != mov.shape:
        raise RadiometricError(
            f"Pair shapes differ: {ref.shape} vs {mov.shape}.",
            f"شكلا الزوج مختلفان: {ref.shape} مقابل {mov.shape}.",
            suggestion_en="Run harmonization and co-registration first.",
            suggestion_ar="نفّذ التوحيد والتسجيل الهندسي أولاً.",
        )

    height, width = ref.shape[1:]
    valid = (
        np.ones((height, width), dtype=bool) if valid_mask is None else valid_mask
    )
    valid = valid & np.all(np.isfinite(ref), axis=0) & np.all(np.isfinite(mov), axis=0)
    valid_count = int(valid.sum())

    same_sensor = (
        reference_sensor is not None
        and moving_sensor is not None
        and reference_sensor == moving_sensor
    )
    gap_days: float | None = None
    if reference_datetime is not None and moving_datetime is not None:
        gap_days = abs((reference_datetime - moving_datetime).total_seconds()) / 86400.0

    if method is None:
        method, reason = choose_method(ref.shape[0], valid_count, same_sensor, gap_days)
    else:
        if method not in (METHOD_HISTOGRAM, METHOD_PIF, METHOD_IRMAD):
            raise RadiometricError(
                f"Unknown radiometric method '{method}'.",
                f"طريقة تطبيع إشعاعي غير معروفة '{method}'.",
                suggestion_en=(
                    f"Use one of: {METHOD_HISTOGRAM}, {METHOD_PIF}, {METHOD_IRMAD}."
                ),
                suggestion_ar=(
                    f"استخدم إحدى: {METHOD_HISTOGRAM}, {METHOD_PIF}, {METHOD_IRMAD}."
                ),
            )
        reason = "Method forced by caller. | فُرضت الطريقة من المستدعي."

    warnings: list[str] = []
    details: dict[str, Any] = {"valid_pixel_count": valid_count}
    normalized: "np.ndarray | None" = None

    if method == METHOD_IRMAD:
        try:
            result = compute_irmad(ref, mov, valid_mask=valid)
            normalized = apply_irmad_normalization(mov, result.gains, result.offsets)
            warnings.extend(result.warnings)
            details.update(
                {
                    "iterations": result.iterations,
                    "converged": result.converged,
                    "canonical_correlations": result.canonical_correlations.tolist(),
                    "pif_count": int(result.pif_mask.sum()),
                    "gains": result.gains.tolist(),
                    "offsets": result.offsets.tolist(),
                }
            )
        except RadiometricError as exc:
            warnings.append(
                f"IR-MAD failed ({exc.message_en}); falling back to histogram matching. | "
                f"فشل IR-MAD ({exc.message_ar})؛ يتم التحول إلى مطابقة الهيستوغرام."
            )
            method = METHOD_HISTOGRAM
            normalized = match_histograms(ref, mov)
    elif method == METHOD_PIF:
        try:
            normalization = fit_pif_linear(ref, mov, valid_mask=valid)
            normalized = apply_pif_normalization(mov, normalization)
            warnings.extend(normalization.warnings)
            details.update(
                {
                    "pif_count": normalization.pif_count,
                    "gains": normalization.gains.tolist(),
                    "offsets": normalization.offsets.tolist(),
                    "r_squared": normalization.r_squared.tolist(),
                }
            )
        except RadiometricError as exc:
            warnings.append(
                f"PIF normalization failed ({exc.message_en}); falling back to "
                f"histogram matching. | فشل تطبيع PIF ({exc.message_ar})؛ يتم "
                "التحول إلى مطابقة الهيستوغرام."
            )
            method = METHOD_HISTOGRAM
            normalized = match_histograms(ref, mov)
    if normalized is None:
        method = METHOD_HISTOGRAM
        normalized = match_histograms(ref, mov)

    return RadiometricSelection(
        method=method,
        reason=reason,
        normalized=normalized,
        details=details,
        warnings=warnings,
    )
