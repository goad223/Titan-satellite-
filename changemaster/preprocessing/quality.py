"""Input quality gate executed before any preprocessing step.

Checks every input image for corruption, saturation, blur, nodata coverage
and a first-pass cloud fraction estimate, then produces a
:class:`QualityReport` with an overall 0-100 score and a recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import QualityGateError
from changemaster.core.hardware import HardwareInfo
from changemaster.io_engine.base_reader import BaseImageReader, open_image
from changemaster.io_engine.tiled_access import TiledImageAccessor
from changemaster.preprocessing._common import adaptive_tile_size

if TYPE_CHECKING:
    import numpy as np

#: Recommendation labels.
RECOMMEND_PROCEED = "proceed"
RECOMMEND_WARN = "warn"
RECOMMEND_REJECT = "reject"


@dataclass
class QualityReport:
    """Quality assessment of a single input image.

    Attributes
    ----------
    path:
        Source image path.
    readable:
        ``False`` when sampled windows failed to read (corrupt file).
    corrupt_window_count:
        Number of sampled windows that raised read errors.
    sampled_window_count:
        Total number of sparse windows sampled.
    saturation_fraction:
        Fraction of sampled pixels at the dtype saturation value.
    blur_metric:
        Variance of the Laplacian (higher = sharper); ``0`` when unreadable.
    nodata_fraction:
        Fraction of sampled pixels equal to the nodata value.
    cloud_fraction_estimate:
        First-pass bright-pixel cloud fraction estimate (0-1).
    score:
        Overall quality score 0-100.
    recommendation:
        ``"proceed"``, ``"warn"`` or ``"reject"``.
    warnings:
        Human-readable bilingual warning strings.
    """

    path: str
    readable: bool
    corrupt_window_count: int
    sampled_window_count: int
    saturation_fraction: float
    blur_metric: float
    nodata_fraction: float
    cloud_fraction_estimate: float
    score: float
    recommendation: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)


def _laplacian_variance(band: "np.ndarray") -> float:
    """Variance of a 4-neighbour Laplacian of a 2-D band (blur metric)."""
    import numpy as np

    data = np.asarray(band, dtype=np.float64)
    if data.shape[0] < 3 or data.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * data[1:-1, 1:-1]
        + data[:-2, 1:-1]
        + data[2:, 1:-1]
        + data[1:-1, :-2]
        + data[1:-1, 2:]
    )
    return float(np.var(lap))


def _saturation_value(dtype_name: str) -> float | None:
    """Return the saturation (max) value for integer dtypes, else ``None``."""
    import numpy as np

    dtype = np.dtype(dtype_name)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


def _estimate_cloud_fraction(data: "np.ndarray", valid: "np.ndarray") -> float:
    """Rough bright-pixel cloud fraction over valid pixels of a tile stack."""
    import numpy as np

    if not np.any(valid):
        return 0.0
    band = np.asarray(data[0], dtype=np.float64)
    values = band[valid]
    if values.size == 0:
        return 0.0
    spread = np.percentile(values, 99.0) - np.percentile(values, 1.0)
    if spread <= 0:
        return 0.0
    threshold = np.percentile(values, 1.0) + 0.85 * spread
    return float(np.mean(values > threshold))


def _select_sample_tiles(accessor: TiledImageAccessor, max_windows: int) -> list[int]:
    """Pick up to ``max_windows`` tile indices spread across the raster."""
    total = len(accessor.tiles)
    if total <= max_windows:
        return list(range(total))
    step = total / max_windows
    return sorted({int(i * step) for i in range(max_windows)})


def assess_quality(
    source: BaseImageReader | Path | str,
    hardware: HardwareInfo | None = None,
    max_windows: int = 16,
    reject_threshold: float = 30.0,
    warn_threshold: float = 60.0,
) -> QualityReport:
    """Assess input quality by sampling sparse windows across the raster.

    Parameters
    ----------
    source:
        An opened reader, or a path which is opened (and closed) here.
    hardware:
        Hardware snapshot used to adapt the sampling tile size.
    max_windows:
        Maximum number of sparse windows to sample.
    reject_threshold / warn_threshold:
        Score boundaries for the recommendation.

    Raises
    ------
    QualityGateError
        When the file cannot be opened at all.
    """
    import numpy as np

    owns_reader = not isinstance(source, BaseImageReader)
    if owns_reader:
        try:
            reader = open_image(source)  # type: ignore[arg-type]
        except Exception as exc:
            raise QualityGateError(
                f"Cannot open image for quality check: {source}: {exc}",
                f"تعذر فتح الصورة لفحص الجودة: {source}: {exc}",
                suggestion_en="Verify the file exists and is not corrupted, then retry.",
                suggestion_ar="تحقق من وجود الملف وسلامته ثم أعد المحاولة.",
            ) from exc
    else:
        reader = source

    try:
        meta = reader.metadata
        tile_size = min(adaptive_tile_size(hardware), max(meta.height, meta.width))
        accessor = TiledImageAccessor(reader, tile_size=tile_size)
        indices = _select_sample_tiles(accessor, max_windows)

        corrupt = 0
        sat_pixels = 0
        nodata_pixels = 0
        total_pixels = 0
        blur_values: list[float] = []
        cloud_fractions: list[float] = []
        sat_value = _saturation_value(meta.dtype)

        for idx in indices:
            tile = accessor.tiles[idx]
            try:
                data = accessor.read_tile(tile)
            except Exception:  # noqa: BLE001 - any read failure marks corruption
                corrupt += 1
                continue
            band0 = np.asarray(data[0], dtype=np.float64)
            n = band0.size
            total_pixels += n
            valid = np.isfinite(band0)
            if meta.nodata is not None:
                nodata_here = band0 == meta.nodata
                nodata_pixels += int(np.count_nonzero(nodata_here))
                valid &= ~nodata_here
            nodata_pixels += int(np.count_nonzero(~np.isfinite(band0)))
            if sat_value is not None:
                sat_pixels += int(np.count_nonzero(band0 == sat_value))
            blur_values.append(_laplacian_variance(band0))
            cloud_fractions.append(_estimate_cloud_fraction(data, valid))

        sampled = len(indices)
        readable = corrupt < sampled
        saturation_fraction = sat_pixels / total_pixels if total_pixels else 0.0
        nodata_fraction = nodata_pixels / total_pixels if total_pixels else 1.0
        blur_metric = float(np.median(blur_values)) if blur_values else 0.0
        cloud_estimate = float(np.mean(cloud_fractions)) if cloud_fractions else 0.0

        warnings: list[str] = []
        score = 100.0
        if corrupt > 0:
            score -= 40.0 * (corrupt / sampled)
            warnings.append(
                f"{corrupt}/{sampled} sampled windows failed to read (possible corruption). | "
                f"فشلت قراءة {corrupt}/{sampled} من النوافذ المعاينة (احتمال تلف)."
            )
        if saturation_fraction > 0.05:
            score -= min(20.0, 200.0 * saturation_fraction)
            warnings.append(
                f"High saturation fraction: {saturation_fraction:.1%}. | "
                f"نسبة تشبع مرتفعة: {saturation_fraction:.1%}."
            )
        if nodata_fraction > 0.30:
            score -= min(25.0, 50.0 * nodata_fraction)
            warnings.append(
                f"High nodata fraction: {nodata_fraction:.1%}. | "
                f"نسبة nodata مرتفعة: {nodata_fraction:.1%}."
            )
        if blur_metric < 1.0 and readable:
            score -= 15.0
            warnings.append(
                "Image appears blurred (low Laplacian variance). | "
                "تبدو الصورة ضبابية (تباين لابلاس منخفض)."
            )
        if cloud_estimate > 0.50:
            score -= min(20.0, 30.0 * cloud_estimate)
            warnings.append(
                f"Estimated cloud fraction is high: {cloud_estimate:.1%}. | "
                f"نسبة الغيوم المقدرة مرتفعة: {cloud_estimate:.1%}."
            )
        if not readable:
            score = 0.0
            warnings.append(
                "File is unreadable in all sampled windows. | "
                "تعذرت قراءة الملف في كل النوافذ المعاينة."
            )
        score = max(0.0, min(100.0, score))

        if score < reject_threshold:
            recommendation = RECOMMEND_REJECT
        elif score < warn_threshold:
            recommendation = RECOMMEND_WARN
        else:
            recommendation = RECOMMEND_PROCEED

        return QualityReport(
            path=str(meta.path),
            readable=readable,
            corrupt_window_count=corrupt,
            sampled_window_count=sampled,
            saturation_fraction=float(saturation_fraction),
            blur_metric=blur_metric,
            nodata_fraction=float(nodata_fraction),
            cloud_fraction_estimate=cloud_estimate,
            score=float(score),
            recommendation=recommendation,
            warnings=warnings,
        )
    finally:
        if owns_reader:
            reader.close()
