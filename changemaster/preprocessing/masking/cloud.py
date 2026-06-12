"""Cloud detection: multi-spectral thresholds fused with SCL / QA_PIXEL.

Smart priority: when a Sentinel-2 SCL layer or a Landsat QA_PIXEL band is
available it is decoded first and *complemented* by spectral tests;
otherwise spectral detection runs alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import MaskingError

if TYPE_CHECKING:
    import numpy as np

#: Sentinel-2 Scene Classification Layer class values.
SCL_NODATA = (0, 1)  # no data / saturated-defective
SCL_CLOUD_SHADOW = (3,)
SCL_WATER = (6,)
SCL_CLOUD = (8, 9, 10)  # medium prob, high prob, thin cirrus
SCL_SNOW = (11,)

#: Landsat Collection 2 QA_PIXEL bit positions.
QA_BIT_FILL = 0
QA_BIT_DILATED_CLOUD = 1
QA_BIT_CIRRUS = 2
QA_BIT_CLOUD = 3
QA_BIT_CLOUD_SHADOW = 4
QA_BIT_SNOW = 5
QA_BIT_WATER = 7


@dataclass
class CloudDetectionResult:
    """Cloud (and auxiliary) detection masks.

    Attributes
    ----------
    cloud:
        Boolean ``(H, W)`` cloud mask.
    shadow / snow / water / nodata:
        Auxiliary boolean masks decoded from SCL/QA when available.
    source:
        ``"scl"``, ``"qa_pixel"``, ``"spectral"`` or combinations like
        ``"scl+spectral"``.
    cloud_fraction:
        Cloud fraction over all pixels.
    """

    cloud: "np.ndarray"
    shadow: "np.ndarray"
    snow: "np.ndarray"
    water: "np.ndarray"
    nodata: "np.ndarray"
    source: str
    cloud_fraction: float = 0.0
    warnings: list[str] = field(default_factory=list)


def decode_scl(scl: "np.ndarray") -> CloudDetectionResult:
    """Decode a Sentinel-2 L2A SCL band into boolean class masks."""
    import numpy as np

    data = np.asarray(scl)
    if data.ndim == 3:
        data = data[0]
    cloud = np.isin(data, SCL_CLOUD)
    shadow = np.isin(data, SCL_CLOUD_SHADOW)
    snow = np.isin(data, SCL_SNOW)
    water = np.isin(data, SCL_WATER)
    nodata = np.isin(data, SCL_NODATA)
    return CloudDetectionResult(
        cloud=cloud,
        shadow=shadow,
        snow=snow,
        water=water,
        nodata=nodata,
        source="scl",
        cloud_fraction=float(cloud.mean()),
    )


def decode_qa_pixel(qa: "np.ndarray") -> CloudDetectionResult:
    """Decode a Landsat Collection-2 QA_PIXEL band into boolean masks."""
    import numpy as np

    data = np.asarray(qa).astype(np.uint16)
    if data.ndim == 3:
        data = data[0]

    def bit(position: int) -> "np.ndarray":
        return (data >> position) & 1 == 1

    cloud = bit(QA_BIT_CLOUD) | bit(QA_BIT_DILATED_CLOUD) | bit(QA_BIT_CIRRUS)
    shadow = bit(QA_BIT_CLOUD_SHADOW)
    snow = bit(QA_BIT_SNOW)
    water = bit(QA_BIT_WATER)
    nodata = bit(QA_BIT_FILL)
    return CloudDetectionResult(
        cloud=cloud,
        shadow=shadow,
        snow=snow,
        water=water,
        nodata=nodata,
        source="qa_pixel",
        cloud_fraction=float(cloud.mean()),
    )


def detect_clouds_spectral(
    blue: "np.ndarray",
    red: "np.ndarray | None" = None,
    nir: "np.ndarray | None" = None,
    swir: "np.ndarray | None" = None,
    cirrus: "np.ndarray | None" = None,
    reflectance_scale: float = 1.0,
    blue_threshold: float = 0.30,
    brightness_threshold: float = 0.35,
    cirrus_threshold: float = 0.012,
    ndsi_cloud_max: float = 0.8,
) -> "np.ndarray":
    """Multi-threshold spectral cloud test (Fmask-inspired, simplified).

    Tests applied (logical AND of the available ones):
      * high blue reflectance (clouds are bright and white),
      * high mean visible/NIR brightness,
      * NDSI below ``ndsi_cloud_max`` to avoid flagging snow,
      * optional cirrus-band test (logical OR with the result).

    Parameters
    ----------
    blue / red / nir / swir / cirrus:
        2-D reflectance bands (``None`` skips the related test).
    reflectance_scale:
        Divider mapping DN to reflectance (e.g. 10000 for Sentinel-2 L2A).

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` cloud mask.
    """
    import numpy as np

    if reflectance_scale <= 0:
        raise MaskingError(
            f"reflectance_scale must be positive, got {reflectance_scale}.",
            f"يجب أن يكون معامل الانعكاسية موجباً، وجد {reflectance_scale}.",
        )
    b = np.asarray(blue, dtype=np.float64) / reflectance_scale
    mask = b > blue_threshold

    stack = [b]
    for band in (red, nir):
        if band is not None:
            stack.append(np.asarray(band, dtype=np.float64) / reflectance_scale)
    brightness = np.mean(np.stack(stack), axis=0)
    mask &= brightness > brightness_threshold

    if swir is not None:
        s = np.asarray(swir, dtype=np.float64) / reflectance_scale
        # Visible band for the NDSI test: red when available, else blue.
        vis_band = stack[1] if len(stack) > 1 else b
        denom = vis_band + s
        denom[denom == 0] = 1.0
        ndsi = (vis_band - s) / denom
        mask &= ndsi < ndsi_cloud_max

    if cirrus is not None:
        c = np.asarray(cirrus, dtype=np.float64) / reflectance_scale
        mask |= c > cirrus_threshold
    return mask


def detect_clouds(
    bands: dict[str, "np.ndarray"],
    scl: "np.ndarray | None" = None,
    qa_pixel: "np.ndarray | None" = None,
    reflectance_scale: float = 1.0,
) -> CloudDetectionResult:
    """Fused cloud detection with smart source priority.

    Parameters
    ----------
    bands:
        Mapping of available spectral bands by role: any of ``"blue"``,
        ``"green"``, ``"red"``, ``"nir"``, ``"swir"``, ``"cirrus"``.
    scl:
        Sentinel-2 SCL classification band (decoded first when given).
    qa_pixel:
        Landsat QA_PIXEL band (decoded first when given and no SCL).
    reflectance_scale:
        DN-to-reflectance divisor for the spectral tests.

    Returns
    -------
    CloudDetectionResult
        Fused masks; ``source`` documents which inputs contributed.

    Raises
    ------
    MaskingError
        When neither QA layers nor a blue band are available.
    """
    import numpy as np

    base: CloudDetectionResult | None = None
    if scl is not None:
        base = decode_scl(scl)
    elif qa_pixel is not None:
        base = decode_qa_pixel(qa_pixel)

    spectral: "np.ndarray | None" = None
    if "blue" in bands:
        spectral = detect_clouds_spectral(
            blue=bands["blue"],
            red=bands.get("red"),
            nir=bands.get("nir"),
            swir=bands.get("swir"),
            cirrus=bands.get("cirrus"),
            reflectance_scale=reflectance_scale,
        )

    if base is None and spectral is None:
        raise MaskingError(
            "Cloud detection needs an SCL/QA_PIXEL layer or at least a blue band.",
            "يتطلب كشف الغيوم طبقة SCL/QA_PIXEL أو نطاقاً أزرق على الأقل.",
            suggestion_en="Provide the product's QA layer or map the blue band.",
            suggestion_ar="وفر طبقة الجودة الخاصة بالمنتج أو حدد النطاق الأزرق.",
        )

    if base is not None and spectral is not None:
        cloud = base.cloud | spectral
        source = f"{base.source}+spectral"
        result = CloudDetectionResult(
            cloud=cloud,
            shadow=base.shadow,
            snow=base.snow,
            water=base.water,
            nodata=base.nodata,
            source=source,
            cloud_fraction=float(cloud.mean()),
            warnings=base.warnings,
        )
        return result
    if base is not None:
        return base
    assert spectral is not None
    empty = np.zeros(spectral.shape, dtype=bool)
    return CloudDetectionResult(
        cloud=spectral,
        shadow=empty.copy(),
        snow=empty.copy(),
        water=empty.copy(),
        nodata=empty.copy(),
        source="spectral",
        cloud_fraction=float(spectral.mean()),
    )
