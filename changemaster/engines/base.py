"""Unified contract for all ChangeMaster change-detection engines.

This module defines the engine contract that **every** engine — the Phase-3
classical engine and the Phase-4 deep engine alike — must honour:

* Input: a :class:`PreprocessedPair` produced by the Phase-2
  :class:`~changemaster.preprocessing.pipeline.PreprocessingPipeline`
  (co-registered, radiometrically normalized, same shape) together with a
  :class:`~changemaster.preprocessing.masking.combiner.ValidityMask`.
* Output: a :class:`~changemaster.engines.results.ChangeResult` whose
  ``probability_map`` is a float32 ``(H, W)`` array in ``[0, 1]``.

Masked pixels (clouds, shadows, nodata, ...) **never** enter any statistical
computation (thresholds, PCA, KMeans, covariances) and are never labelled as
change — they appear in the result as *unevaluated*
(:data:`~changemaster.engines.results.BINARY_UNEVALUATED`).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from changemaster.core.exceptions import EngineError
from changemaster.io_engine.metadata import GeoReference

if TYPE_CHECKING:
    import numpy as np

    from changemaster.engines.results import ChangeResult
    from changemaster.preprocessing.masking.combiner import ValidityMask


@dataclass
class PreprocessedPair:
    """A co-registered, normalized image pair ready for change detection.

    This is the hand-off object between Phase 2 (preprocessing) and the
    change-detection engines. Both arrays must share an identical
    ``(bands, H, W)`` shape — exactly what
    :class:`~changemaster.preprocessing.pipeline.PreprocessingPipeline`
    produces after harmonization, co-registration and radiometric
    normalization.

    Attributes
    ----------
    reference:
        Reference (earlier) image, ``(bands, H, W)``.
    moving:
        Moving (later) image aligned and normalized to the reference,
        ``(bands, H, W)``.
    georef:
        Georeferencing shared by both arrays (may be empty).
    mode:
        ``"optical"`` or ``"sar"`` — selects the difference operator used by
        ratio/difference engines.
    sensor_id:
        Sensor identifier (e.g. ``"sentinel2"``) used to look up band roles
        for change-type hints; ``None`` when unknown.
    band_names:
        Human-readable band names (one per band).
    nodata:
        No-data value when defined.
    """

    reference: "np.ndarray"
    moving: "np.ndarray"
    georef: GeoReference = field(default_factory=GeoReference)
    mode: str = "optical"
    sensor_id: str | None = None
    band_names: list[str] = field(default_factory=list)
    nodata: float | None = None

    def __post_init__(self) -> None:
        import numpy as np

        ref = np.asarray(self.reference)
        mov = np.asarray(self.moving)
        if ref.ndim == 2:
            ref = ref[np.newaxis]
        if mov.ndim == 2:
            mov = mov[np.newaxis]
        if ref.ndim != 3 or mov.ndim != 3 or ref.shape != mov.shape:
            raise EngineError(
                f"PreprocessedPair needs identical (bands, H, W) arrays; "
                f"got {ref.shape} vs {mov.shape}.",
                f"يتطلب PreprocessedPair مصفوفتين متطابقتين (bands, H, W)؛ "
                f"وجد {ref.shape} و{mov.shape}.",
                suggestion_en="Run the Phase-2 PreprocessingPipeline on the pair first.",
                suggestion_ar="شغّل أنبوب المعالجة المسبقة (المرحلة 2) على الزوج أولاً.",
            )
        if self.mode not in ("optical", "sar"):
            raise EngineError(
                f"Unknown pair mode '{self.mode}'.",
                f"وضع زوج غير معروف '{self.mode}'.",
                suggestion_en="Use 'optical' or 'sar'.",
                suggestion_ar="استخدم 'optical' أو 'sar'.",
            )
        self.reference = ref
        self.moving = mov
        if not self.band_names:
            self.band_names = [f"Band {i + 1}" for i in range(ref.shape[0])]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Pair shape as ``(bands, H, W)``."""
        return tuple(self.reference.shape)  # type: ignore[return-value]

    @property
    def pixel_size_m(self) -> float | None:
        """Ground pixel size in metres from the geotransform (``None`` if unknown)."""
        if self.georef.transform is None:
            return None
        size = abs(self.georef.transform[0])
        return size if size > 0 else None

    @classmethod
    def from_pipeline_workdir(
        cls, workdir: Path | str, mode: str = "optical"
    ) -> tuple["PreprocessedPair", "ValidityMask | None"]:
        """Load a pair (and its validity mask) from Phase-2 checkpoints.

        Reads the last completed checkpoint of a
        :class:`~changemaster.preprocessing.pipeline.PreprocessingPipeline`
        run (``radiometric`` preferred, then ``coregistration``) plus the
        ``masking`` checkpoint when present.

        Parameters
        ----------
        workdir:
            The pipeline working directory containing ``checkpoint_*.npz``.
        mode:
            ``"optical"`` or ``"sar"``.

        Returns
        -------
        tuple
            ``(pair, validity_mask_or_None)``.
        """
        import numpy as np

        from changemaster.preprocessing.masking.combiner import (
            CODE_NODATA,
            ValidityMask,
            combine_masks,
        )

        wd = Path(workdir)
        arrays: dict[str, "np.ndarray"] | None = None
        for step in ("radiometric", "coregistration", "harmonize", "speckle"):
            path = wd / f"checkpoint_{step}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    arrays = {k: data[k] for k in data.files}
                break
        if arrays is None or "reference" not in arrays or "moving" not in arrays:
            raise EngineError(
                f"No usable preprocessing checkpoint found in {wd}.",
                f"لم يُعثر على نقطة استئناف معالجة مسبقة صالحة في {wd}.",
                suggestion_en="Run scripts/titan_preprocess.py on the pair first.",
                suggestion_ar="شغّل scripts/titan_preprocess.py على الزوج أولاً.",
            )
        validity: "ValidityMask | None" = None
        mask_path = wd / "checkpoint_masking.npz"
        if mask_path.exists():
            with np.load(mask_path, allow_pickle=False) as data:
                if "validity_mask" in data.files:
                    raw = data["validity_mask"]
                    validity = combine_masks(raw.shape, nodata=raw == CODE_NODATA)
                    validity.mask = raw
        pair = cls(reference=arrays["reference"], moving=arrays["moving"], mode=mode)
        return pair, validity


def resolve_valid_mask(
    pair: PreprocessedPair, mask: "ValidityMask | None"
) -> "np.ndarray":
    """Return the boolean ``(H, W)`` mask of pixels usable for statistics.

    Combines the explicit :class:`ValidityMask` (code 0 = valid) with
    non-finite pixel screening on both images. Raises when shapes mismatch
    or when no valid pixel remains.

    Parameters
    ----------
    pair:
        The preprocessed pair.
    mask:
        Optional Phase-2 validity mask; ``None`` means all pixels are valid.

    Returns
    -------
    numpy.ndarray
        Boolean ``(H, W)`` array, ``True`` where the pixel may be used.
    """
    import numpy as np

    height, width = pair.reference.shape[1], pair.reference.shape[2]
    valid = np.ones((height, width), dtype=bool)
    if mask is not None:
        arr = np.asarray(mask.mask)
        if arr.shape != (height, width):
            raise EngineError(
                f"Validity mask shape {arr.shape} does not match pair "
                f"shape {(height, width)}.",
                f"شكل قناع الصلاحية {arr.shape} لا يطابق شكل الزوج {(height, width)}.",
                suggestion_en="Use the mask produced by the same pipeline run.",
                suggestion_ar="استخدم القناع الناتج من نفس تشغيل الأنبوب.",
            )
        valid &= arr == 0
    valid &= np.all(np.isfinite(pair.reference), axis=0)
    valid &= np.all(np.isfinite(pair.moving), axis=0)
    if pair.nodata is not None:
        valid &= ~np.all(pair.reference == pair.nodata, axis=0)
        valid &= ~np.all(pair.moving == pair.nodata, axis=0)
    if not valid.any():
        raise EngineError(
            "No valid pixels remain after masking; change detection is impossible.",
            "لا توجد بكسلات صالحة بعد تطبيق الأقنعة؛ كشف التغيرات مستحيل.",
            suggestion_en="Check cloud/nodata coverage of the input pair.",
            suggestion_ar="تحقق من تغطية الغيوم/البيانات المفقودة في الزوج المدخل.",
        )
    return valid


class ChangeEngine(abc.ABC):
    """Abstract base class for all change-detection engines.

    The contract (honoured by the Phase-3 classical engine and required of
    the Phase-4 deep engine and any future ensemble member):

    * :meth:`detect` receives a :class:`PreprocessedPair` and an optional
      :class:`ValidityMask` and returns a
      :class:`~changemaster.engines.results.ChangeResult`.
    * ``probability_map`` is float32 ``(H, W)`` in ``[0, 1]``.
    * Masked pixels are excluded from all statistics and are reported as
      unevaluated, never as change.
    * Engines must work tiled/chunked so memory stays bounded on huge
      images, adapting window sizes (not accuracy) to the
      :class:`~changemaster.core.hardware.HardwareTier`.
    """

    #: Unique engine identifier (used in fusion weights and reports).
    name: str = "abstract"

    @abc.abstractmethod
    def detect(
        self, pair: PreprocessedPair, mask: "ValidityMask | None" = None
    ) -> "ChangeResult":
        """Detect changes on a preprocessed pair.

        Parameters
        ----------
        pair:
            Co-registered, normalized image pair from Phase 2.
        mask:
            Optional unified validity mask; masked pixels are excluded from
            all statistics and marked unevaluated in the result.

        Returns
        -------
        ChangeResult
            Probability map in ``[0, 1]`` plus binary, uncertainty and
            agreement layers with metadata and statistics.
        """
