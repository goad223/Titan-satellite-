"""Band-by-band histogram matching radiometric normalization.

Maps the moving image's per-band cumulative distribution onto the
reference's, using quantile lookup tables. Works tiled: histograms are
accumulated incrementally across windows before building the mapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from changemaster.core.exceptions import RadiometricError

if TYPE_CHECKING:
    import numpy as np


def _accumulate_histogram(
    chunks: Iterable["np.ndarray"],
    bins: int,
    value_range: tuple[float, float],
    nodata: float | None,
) -> "np.ndarray":
    """Accumulate a histogram across an iterable of 2-D chunks."""
    import numpy as np

    hist = np.zeros(bins, dtype=np.float64)
    for chunk in chunks:
        data = np.asarray(chunk, dtype=np.float64).ravel()
        data = data[np.isfinite(data)]
        if nodata is not None:
            data = data[data != nodata]
        if data.size:
            h, _ = np.histogram(data, bins=bins, range=value_range)
            hist += h
    return hist


def build_matching_lut(
    reference_hist: "np.ndarray",
    moving_hist: "np.ndarray",
    value_range: tuple[float, float],
) -> tuple["np.ndarray", "np.ndarray"]:
    """Build a histogram-matching lookup table from two histograms.

    Returns ``(bin_centers, mapped_values)`` so that interpolating a pixel
    value over ``bin_centers -> mapped_values`` matches the reference CDF.
    """
    import numpy as np

    if reference_hist.shape != moving_hist.shape:
        raise RadiometricError(
            "Reference and moving histograms must have identical bin counts.",
            "يجب أن يتطابق عدد الحاويات في هيستوغرامي الصورتين.",
        )
    bins = reference_hist.shape[0]
    lo, hi = value_range
    centers = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins

    ref_total = reference_hist.sum()
    mov_total = moving_hist.sum()
    if ref_total <= 0 or mov_total <= 0:
        raise RadiometricError(
            "Cannot histogram-match: one of the images has no valid pixels.",
            "تعذرت مطابقة الهيستوغرام: إحدى الصورتين لا تحتوي بكسلات صالحة.",
            suggestion_en="Check nodata configuration and mask coverage.",
            suggestion_ar="تحقق من إعدادات nodata وتغطية الأقنعة.",
        )
    ref_cdf = np.cumsum(reference_hist) / ref_total
    mov_cdf = np.cumsum(moving_hist) / mov_total
    # For each moving bin, the matched value is where the reference CDF
    # reaches the same quantile.
    mapped = np.interp(mov_cdf, ref_cdf, centers)
    return centers, mapped


def apply_lut(
    band: "np.ndarray",
    centers: "np.ndarray",
    mapped: "np.ndarray",
    nodata: float | None = None,
) -> "np.ndarray":
    """Apply a histogram-matching LUT to a 2-D band (nodata preserved)."""
    import numpy as np

    data = np.asarray(band, dtype=np.float64)
    out = np.interp(data, centers, mapped)
    if nodata is not None:
        out[data == nodata] = nodata
    out[~np.isfinite(data)] = np.nan
    return out


def match_histograms(
    reference: "np.ndarray",
    moving: "np.ndarray",
    bins: int = 1024,
    nodata: float | None = None,
) -> "np.ndarray":
    """Histogram-match ``moving`` to ``reference`` band-by-band.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` or 2-D arrays with the same band count.
    bins:
        Histogram resolution.
    nodata:
        Value excluded from statistics and preserved in the output.

    Returns
    -------
    np.ndarray
        Matched image as float64 with the input's shape.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    squeeze = False
    if ref.ndim == 2:
        ref = ref[np.newaxis]
        squeeze = True
    if mov.ndim == 2:
        mov = mov[np.newaxis]
    if ref.shape[0] != mov.shape[0]:
        raise RadiometricError(
            f"Band count mismatch: {ref.shape[0]} vs {mov.shape[0]}.",
            f"عدم تطابق عدد النطاقات: {ref.shape[0]} مقابل {mov.shape[0]}.",
            suggestion_en="Harmonize the band lists of the pair first.",
            suggestion_ar="وحّد قائمتي النطاقات للزوج أولاً.",
        )

    out = np.empty_like(mov)
    for b in range(mov.shape[0]):
        both = np.concatenate(
            [
                ref[b][np.isfinite(ref[b])].ravel(),
                mov[b][np.isfinite(mov[b])].ravel(),
            ]
        )
        if nodata is not None:
            both = both[both != nodata]
        if both.size == 0:
            raise RadiometricError(
                f"Band {b + 1} has no valid pixels for histogram matching.",
                f"النطاق {b + 1} لا يحتوي بكسلات صالحة لمطابقة الهيستوغرام.",
            )
        lo, hi = float(both.min()), float(both.max())
        if hi <= lo:
            out[b] = mov[b]
            continue
        value_range = (lo, hi)
        ref_hist = _accumulate_histogram([ref[b]], bins, value_range, nodata)
        mov_hist = _accumulate_histogram([mov[b]], bins, value_range, nodata)
        centers, mapped = build_matching_lut(ref_hist, mov_hist, value_range)
        out[b] = apply_lut(mov[b], centers, mapped, nodata)
    return out[0] if squeeze else out


def match_histograms_tiled(
    reference_tiles: Iterable["np.ndarray"],
    moving_tiles: Iterable["np.ndarray"],
    value_range: tuple[float, float],
    bins: int = 1024,
    nodata: float | None = None,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Build a matching LUT by accumulating histograms across tile streams.

    For giant images: iterate both images tile-by-tile (single band), then
    apply the returned LUT with :func:`apply_lut` in a second pass.

    Returns ``(centers, mapped)``.
    """
    ref_hist = _accumulate_histogram(reference_tiles, bins, value_range, nodata)
    mov_hist = _accumulate_histogram(moving_tiles, bins, value_range, nodata)
    return build_matching_lut(ref_hist, mov_hist, value_range)
