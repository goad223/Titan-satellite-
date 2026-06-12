"""Masking subpackage (Phase 2): clouds, shadows, snow/water, nodata.

Produces one unified uint8 :class:`~.combiner.ValidityMask` with documented
codes: 0=valid, 1=cloud, 2=shadow, 3=snow, 4=water, 5=nodata/saturated.
"""

from changemaster.preprocessing.masking.cloud import (
    CloudDetectionResult,
    decode_qa_pixel,
    decode_scl,
    detect_clouds,
    detect_clouds_spectral,
)
from changemaster.preprocessing.masking.combiner import (
    CODE_CLOUD,
    CODE_NODATA,
    CODE_SHADOW,
    CODE_SNOW,
    CODE_VALID,
    CODE_WATER,
    MASK_LABELS,
    ValidityMask,
    combine_masks,
)
from changemaster.preprocessing.masking.nodata import (
    detect_edges_nodata,
    detect_nodata,
    detect_saturation,
)
from changemaster.preprocessing.masking.shadow import (
    ShadowDetectionResult,
    detect_shadows,
    project_cloud_shadow,
)
from changemaster.preprocessing.masking.snow_water import (
    detect_snow,
    detect_water,
    mndwi,
    ndsi,
    ndwi,
)

__all__ = [
    "CODE_CLOUD",
    "CODE_NODATA",
    "CODE_SHADOW",
    "CODE_SNOW",
    "CODE_VALID",
    "CODE_WATER",
    "MASK_LABELS",
    "CloudDetectionResult",
    "ShadowDetectionResult",
    "ValidityMask",
    "combine_masks",
    "decode_qa_pixel",
    "decode_scl",
    "detect_clouds",
    "detect_clouds_spectral",
    "detect_edges_nodata",
    "detect_nodata",
    "detect_saturation",
    "detect_shadows",
    "detect_snow",
    "detect_water",
    "mndwi",
    "ndsi",
    "ndwi",
    "project_cloud_shadow",
]
