"""SAR preprocessing subpackage (Phase 2).

Sigma0 calibration from Sentinel-1 metadata, speckle filtering
(Refined Lee default, Frost, Gamma-MAP) and dB conversion utilities.
"""

from changemaster.preprocessing.sar.calibration import (
    CalibrationVector,
    build_calibration_lut,
    calibrate_sigma0,
    parse_calibration_xml,
)
from changemaster.preprocessing.sar.convert import from_db, percentile_clip, to_db
from changemaster.preprocessing.sar.speckle import (
    SPECKLE_FILTERS,
    apply_speckle_filter,
    frost,
    gamma_map,
    refined_lee,
)

__all__ = [
    "SPECKLE_FILTERS",
    "CalibrationVector",
    "apply_speckle_filter",
    "build_calibration_lut",
    "calibrate_sigma0",
    "from_db",
    "frost",
    "gamma_map",
    "parse_calibration_xml",
    "percentile_clip",
    "refined_lee",
    "to_db",
]
