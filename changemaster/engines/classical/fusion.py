"""Fusion of the four classical change-probability maps.

Combines the per-method probability maps with a configurable weighted mean
(documented default weights: CVA 0.3, PCA-KMeans 0.25, IR-MAD 0.3,
Ratio/Diff 0.15 — adjustable through the Phase-1 config or the ``weights``
argument), a per-pixel majority vote, and an **agreement map** counting how
many methods exceeded their own threshold at each pixel (0-4) — later shown
in the GUI as a confidence layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from changemaster.core.exceptions import EngineError

if TYPE_CHECKING:
    import numpy as np

#: Documented default fusion weights per method (sum = 1.0). CVA and IR-MAD
#: carry the most weight because they exploit the full multi-band statistics;
#: PCA-KMeans adds local-texture evidence; the simple ratio/difference index
#: is the weakest single voter.
DEFAULT_WEIGHTS: dict[str, float] = {
    "cva": 0.30,
    "pca_kmeans": 0.25,
    "irmad": 0.30,
    "ratio_diff": 0.15,
}


@dataclass
class FusionResult:
    """Outcome of multi-method fusion.

    Attributes
    ----------
    fused_probability:
        Float32 ``(H, W)`` weighted-mean probability in ``[0, 1]``
        (``NaN`` on invalid pixels).
    agreement_map:
        Uint8 ``(H, W)`` count of methods voting change at each pixel.
    majority_map:
        Boolean ``(H, W)``: ``True`` where a strict majority of available
        methods voted change.
    weights:
        Normalized weights actually used (only for the methods present).
    warnings:
        Bilingual warnings.
    """

    fused_probability: "np.ndarray"
    agreement_map: "np.ndarray"
    majority_map: "np.ndarray"
    weights: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def fuse_probabilities(
    probability_maps: Mapping[str, "np.ndarray"],
    thresholds: Mapping[str, float],
    valid_mask: "np.ndarray",
    weights: Mapping[str, float] | None = None,
) -> FusionResult:
    """Fuse per-method probability maps into one consensus probability.

    Parameters
    ----------
    probability_maps:
        Mapping ``method name -> (H, W)`` probability map in ``[0, 1]``
        (``NaN`` allowed on invalid pixels).
    thresholds:
        Per-method decision threshold (same keys) used for the agreement
        count and the majority vote.
    valid_mask:
        Boolean ``(H, W)`` mask of evaluated pixels.
    weights:
        Optional per-method weights; missing methods fall back to
        :data:`DEFAULT_WEIGHTS` (then to a uniform weight). Weights are
        re-normalized over the methods actually present.

    Returns
    -------
    FusionResult
        Weighted-mean probability, agreement map and majority vote.
    """
    import numpy as np

    if not probability_maps:
        raise EngineError(
            "At least one probability map is required for fusion.",
            "مطلوب خريطة احتمالات واحدة على الأقل للدمج.",
        )
    shape = valid_mask.shape
    names = list(probability_maps)
    for name in names:
        if probability_maps[name].shape != shape:
            raise EngineError(
                f"Probability map '{name}' shape {probability_maps[name].shape} "
                f"does not match {shape}.",
                f"شكل خريطة الاحتمالات '{name}' ‏{probability_maps[name].shape} "
                f"لا يطابق {shape}.",
            )
        if name not in thresholds:
            raise EngineError(
                f"Missing threshold for method '{name}'.",
                f"عتبة الطريقة '{name}' مفقودة.",
            )

    raw_weights = {
        name: float(
            (weights or {}).get(name, DEFAULT_WEIGHTS.get(name, 1.0 / len(names)))
        )
        for name in names
    }
    total = sum(raw_weights.values())
    if total <= 0:
        raise EngineError(
            "Fusion weights must sum to a positive value.",
            "يجب أن يكون مجموع أوزان الدمج موجباً.",
            suggestion_en="Provide at least one positive weight.",
            suggestion_ar="وفر وزناً موجباً واحداً على الأقل.",
        )
    norm_weights = {name: w / total for name, w in raw_weights.items()}

    fused = np.zeros(shape, dtype=np.float64)
    weight_sum = np.zeros(shape, dtype=np.float64)
    agreement = np.zeros(shape, dtype=np.uint8)
    for name in names:
        pmap = np.asarray(probability_maps[name], dtype=np.float64)
        finite = np.isfinite(pmap)
        w = norm_weights[name]
        fused[finite] += w * pmap[finite]
        weight_sum[finite] += w
        agreement += (finite & (pmap > thresholds[name])).astype(np.uint8)

    with np.errstate(invalid="ignore", divide="ignore"):
        fused = np.where(weight_sum > 0, fused / weight_sum, np.nan)
    fused = fused.astype(np.float32)
    fused[~valid_mask] = np.nan
    agreement[~valid_mask] = 0
    majority = valid_mask & (agreement.astype(np.int32) * 2 > len(names))
    return FusionResult(
        fused_probability=np.clip(fused, 0.0, 1.0),
        agreement_map=agreement,
        majority_map=majority,
        weights=norm_weights,
    )
