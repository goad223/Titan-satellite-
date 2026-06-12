"""Automatic thresholding on tile-accumulated histograms.

Implements three classical histogram thresholding criteria — Otsu's
between-class variance, Kapur's maximum entropy and the
Kittler-Illingworth (KI) minimum-error criterion — computed on a histogram
accumulated window by window (never the full image at once), plus an
automatic selector:

* when the three thresholds agree (small relative spread) their **median**
  is used;
* when they diverge, **KI** is preferred for bimodal distributions (typical
  of change-magnitude images) and a bilingual warning is recorded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from changemaster.core.exceptions import EngineError

if TYPE_CHECKING:
    import numpy as np

#: Default number of histogram bins.
DEFAULT_BINS = 256


@dataclass
class ThresholdSelection:
    """Outcome of automatic threshold selection.

    Attributes
    ----------
    value:
        The selected threshold (data units).
    method:
        ``"median"`` (consensus of the three) or ``"ki"`` (divergence).
    candidates:
        Per-method thresholds ``{"otsu": ..., "kapur": ..., "ki": ...}``.
    spread:
        Relative spread of the candidates over the data range.
    warnings:
        Bilingual warnings (recorded when methods diverge).
    """

    value: float
    method: str
    candidates: dict[str, float] = field(default_factory=dict)
    spread: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class HistogramAccumulator:
    """Fixed-range histogram accumulated incrementally over windows.

    Parameters
    ----------
    minimum / maximum:
        Fixed value range of the histogram.
    bins:
        Number of equal-width bins.
    """

    minimum: float
    maximum: float
    bins: int = DEFAULT_BINS

    def __post_init__(self) -> None:
        import numpy as np

        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise EngineError(
                "Histogram range must be finite.",
                "يجب أن يكون مدى الهيستوغرام منتهياً.",
            )
        if self.maximum <= self.minimum:
            self.maximum = self.minimum + 1.0
        if self.bins < 2:
            raise EngineError(
                f"Histogram needs at least 2 bins, got {self.bins}.",
                f"يحتاج الهيستوغرام إلى حاويتين على الأقل، وجد {self.bins}.",
            )
        self.counts: "np.ndarray" = np.zeros(self.bins, dtype=np.int64)
        self.edges: "np.ndarray" = np.linspace(
            self.minimum, self.maximum, self.bins + 1
        )

    def update(self, values: "np.ndarray") -> None:
        """Accumulate finite values from one window into the histogram."""
        import numpy as np

        data = np.asarray(values, dtype=np.float64).ravel()
        data = data[np.isfinite(data)]
        if data.size == 0:
            return
        counts, _ = np.histogram(data, bins=self.bins, range=(self.minimum, self.maximum))
        self.counts += counts

    @property
    def centers(self) -> "np.ndarray":
        """Bin centre values."""
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def total(self) -> int:
        """Total accumulated sample count."""
        return int(self.counts.sum())


def accumulate_histogram(
    chunks: Iterable["np.ndarray"],
    minimum: float,
    maximum: float,
    bins: int = DEFAULT_BINS,
) -> HistogramAccumulator:
    """Build a histogram from an iterable of value windows (tiled).

    Parameters
    ----------
    chunks:
        Iterable of arrays (already masked to valid pixels).
    minimum / maximum:
        Fixed histogram range.
    bins:
        Number of bins.

    Returns
    -------
    HistogramAccumulator
        The populated accumulator.
    """
    acc = HistogramAccumulator(minimum=minimum, maximum=maximum, bins=bins)
    for chunk in chunks:
        acc.update(chunk)
    return acc


def otsu_threshold(hist: HistogramAccumulator) -> float:
    """Otsu's threshold: maximize the between-class variance.

    Returns the bin-centre value of the optimal split.
    """
    import numpy as np

    counts = hist.counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise EngineError(
            "Cannot threshold an empty histogram.",
            "لا يمكن حساب عتبة لهيستوغرام فارغ.",
        )
    p = counts / total
    centers = hist.centers
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.nanargmax(sigma_b2[:-1]))
    return float(hist.edges[idx + 1])


def kapur_threshold(hist: HistogramAccumulator) -> float:
    """Kapur's maximum-entropy threshold.

    Maximizes the sum of the Shannon entropies of the background and
    foreground partitions of the normalized histogram.
    """
    import numpy as np

    counts = hist.counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise EngineError(
            "Cannot threshold an empty histogram.",
            "لا يمكن حساب عتبة لهيستوغرام فارغ.",
        )
    p = counts / total
    cumsum = np.cumsum(p)
    # Entropy partial sums with safe log.
    plogp = np.where(p > 0, p * np.log(p), 0.0)
    cum_plogp = np.cumsum(plogp)
    best_idx = 0
    best_entropy = -np.inf
    for t in range(hist.bins - 1):
        w0 = cumsum[t]
        w1 = 1.0 - w0
        if w0 <= 0 or w1 <= 0:
            continue
        h0 = math.log(w0) - cum_plogp[t] / w0
        h1 = math.log(w1) - (cum_plogp[-1] - cum_plogp[t]) / w1
        entropy = h0 + h1
        if entropy > best_entropy:
            best_entropy = entropy
            best_idx = t
    return float(hist.edges[best_idx + 1])


def ki_threshold(hist: HistogramAccumulator) -> float:
    """Kittler-Illingworth minimum-error threshold.

    Models the histogram as a mixture of two Gaussians and minimizes the
    classification-error criterion
    ``J(t) = 1 + 2 (w0 ln s0 + w1 ln s1) - 2 (w0 ln w0 + w1 ln w1)``.
    """
    import numpy as np

    counts = hist.counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise EngineError(
            "Cannot threshold an empty histogram.",
            "لا يمكن حساب عتبة لهيستوغرام فارغ.",
        )
    p = counts / total
    centers = hist.centers
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0_num = np.cumsum(p * centers)
    m1_num = m0_num[-1] - m0_num
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = m0_num / w0
        m1 = m1_num / w1
        v0_num = np.cumsum(p * centers**2)
        v1_num = v0_num[-1] - v0_num
        v0 = v0_num / w0 - m0**2
        v1 = v1_num / w1 - m1**2
        s0 = np.sqrt(np.maximum(v0, 1e-12))
        s1 = np.sqrt(np.maximum(v1, 1e-12))
        j = (
            1.0
            + 2.0 * (w0 * np.log(s0) + w1 * np.log(s1))
            - 2.0 * (w0 * _safe_log(w0) + w1 * _safe_log(w1))
        )
    j[(w0 <= 0) | (w1 <= 0)] = np.inf
    j[~np.isfinite(j)] = np.inf
    idx = int(np.argmin(j[:-1]))
    return float(hist.edges[idx + 1])


def _safe_log(values: "np.ndarray") -> "np.ndarray":
    """Elementwise ``log`` that returns 0 for non-positive entries."""
    import numpy as np

    out = np.zeros_like(values)
    positive = values > 0
    out[positive] = np.log(values[positive])
    return out


def select_threshold(
    hist: HistogramAccumulator, agreement_tolerance: float = 0.10
) -> ThresholdSelection:
    """Compute Otsu, Kapur and KI thresholds and select automatically.

    Selection rule:

    * relative spread = (max - min of the three) / histogram range;
    * spread <= ``agreement_tolerance`` → the **median** of the three;
    * otherwise → **KI** (best suited to the bimodal change/no-change
      mixtures of difference images) and a bilingual warning is recorded.

    Parameters
    ----------
    hist:
        The accumulated histogram.
    agreement_tolerance:
        Maximum relative spread considered an agreement.

    Returns
    -------
    ThresholdSelection
        Selected threshold, method, all candidates and warnings.
    """
    import numpy as np

    candidates = {
        "otsu": otsu_threshold(hist),
        "kapur": kapur_threshold(hist),
        "ki": ki_threshold(hist),
    }
    values = np.array(list(candidates.values()))
    data_range = hist.maximum - hist.minimum
    spread = float((values.max() - values.min()) / data_range) if data_range > 0 else 0.0
    warnings: list[str] = []
    if spread <= agreement_tolerance:
        value = float(np.median(values))
        method = "median"
    else:
        value = candidates["ki"]
        method = "ki"
        warnings.append(
            f"Thresholding methods diverge (spread {spread:.1%} of range); "
            "preferring Kittler-Illingworth for the bimodal distribution. | "
            f"تتباعد طرق العتبات (التباعد {spread:.1%} من المدى)؛ "
            "تُفضَّل طريقة Kittler-Illingworth للتوزيع ثنائي النمط."
        )
    return ThresholdSelection(
        value=value,
        method=method,
        candidates=candidates,
        spread=spread,
        warnings=warnings,
    )
