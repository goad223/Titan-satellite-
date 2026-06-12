"""Unified ValidityMask combination with documented uint8 codes.

Mask codes (uint8):
    0 = VALID            valid pixel | بكسل صالح
    1 = CLOUD            cloud | غيمة
    2 = SHADOW           cloud shadow | ظل غيمة
    3 = SNOW             snow/ice | ثلج
    4 = WATER            water | ماء
    5 = NODATA_SATURATED nodata or saturated | بلا بيانات أو مشبع
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import MaskingError

if TYPE_CHECKING:
    import numpy as np

#: Documented validity-mask codes.
CODE_VALID = 0
CODE_CLOUD = 1
CODE_SHADOW = 2
CODE_SNOW = 3
CODE_WATER = 4
CODE_NODATA = 5

#: Human-readable bilingual labels per code.
MASK_LABELS: dict[int, str] = {
    CODE_VALID: "valid | صالح",
    CODE_CLOUD: "cloud | غيمة",
    CODE_SHADOW: "shadow | ظل",
    CODE_SNOW: "snow | ثلج",
    CODE_WATER: "water | ماء",
    CODE_NODATA: "nodata/saturated | بلا بيانات/مشبع",
}


@dataclass
class ValidityMask:
    """Unified uint8 validity mask plus coverage statistics.

    Attributes
    ----------
    mask:
        ``(H, W)`` uint8 array with the documented codes.
    fractions:
        Coverage fraction per code label.
    invalid_fraction:
        Fraction of all non-valid pixels.
    low_reliability:
        ``True`` when the masked fraction exceeds the reliability threshold
        (default 60%) — downstream change detection becomes unreliable.
    warnings:
        Bilingual warnings (includes the explicit low-reliability warning).
    """

    mask: "np.ndarray"
    fractions: dict[str, float] = field(default_factory=dict)
    invalid_fraction: float = 0.0
    low_reliability: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (mask array excluded)."""
        data = asdict(self)
        data.pop("mask")
        return data


def combine_masks(
    shape: tuple[int, int],
    cloud: "np.ndarray | None" = None,
    shadow: "np.ndarray | None" = None,
    snow: "np.ndarray | None" = None,
    water: "np.ndarray | None" = None,
    nodata: "np.ndarray | None" = None,
    reliability_threshold: float = 0.60,
) -> ValidityMask:
    """Combine boolean class masks into one uint8 :class:`ValidityMask`.

    Priority (highest wins): nodata > cloud > shadow > snow > water.

    Parameters
    ----------
    shape:
        Expected ``(H, W)`` of all masks.
    cloud / shadow / snow / water / nodata:
        Optional boolean masks (``None`` = absent).
    reliability_threshold:
        Invalid fraction beyond which a low-reliability warning is issued
        (the specification mandates a warning above 60%).

    Returns
    -------
    ValidityMask
        Combined mask with statistics and warnings.
    """
    import numpy as np

    mask = np.zeros(shape, dtype=np.uint8)

    def _check(name: str, m: "np.ndarray | None") -> "np.ndarray | None":
        if m is None:
            return None
        arr = np.asarray(m, dtype=bool)
        if arr.shape != shape:
            raise MaskingError(
                f"{name} mask shape {arr.shape} does not match expected {shape}.",
                f"شكل قناع {name} ‏{arr.shape} لا يطابق الشكل المتوقع {shape}.",
            )
        return arr

    # Apply in increasing priority so higher-priority classes overwrite.
    for code, m in (
        (CODE_WATER, _check("water", water)),
        (CODE_SNOW, _check("snow", snow)),
        (CODE_SHADOW, _check("shadow", shadow)),
        (CODE_CLOUD, _check("cloud", cloud)),
        (CODE_NODATA, _check("nodata", nodata)),
    ):
        if m is not None:
            mask[m] = code

    total = mask.size
    fractions = {
        MASK_LABELS[code]: float(np.count_nonzero(mask == code)) / total
        for code in MASK_LABELS
    }
    invalid_fraction = float(np.count_nonzero(mask != CODE_VALID)) / total
    warnings: list[str] = []
    low_reliability = invalid_fraction > reliability_threshold
    if low_reliability:
        warnings.append(
            f"Masks cover {invalid_fraction:.1%} of the overlap (> "
            f"{reliability_threshold:.0%}); any subsequent change detection has LOW "
            "RELIABILITY. | تغطي الأقنعة "
            f"{invalid_fraction:.1%} من منطقة التداخل (أكثر من "
            f"{reliability_threshold:.0%})؛ أي كشف تغيرات لاحق منخفض الموثوقية."
        )
    return ValidityMask(
        mask=mask,
        fractions=fractions,
        invalid_fraction=invalid_fraction,
        low_reliability=low_reliability,
        warnings=warnings,
    )
