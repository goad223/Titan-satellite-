"""Radiometric normalization subpackage (Phase 2).

Histogram matching, full IR-MAD (CCA + iterative chi-square weighting),
PIF linear normalization, and automatic method selection.
"""

from changemaster.preprocessing.radiometric.histogram_matching import (
    apply_lut,
    build_matching_lut,
    match_histograms,
    match_histograms_tiled,
)
from changemaster.preprocessing.radiometric.irmad import (
    IncrementalStats,
    IRMADResult,
    apply_irmad_normalization,
    compute_irmad,
    fit_pif_normalization,
)
from changemaster.preprocessing.radiometric.pif_normalization import (
    PIFNormalization,
    apply_pif_normalization,
    fit_pif_linear,
    select_pifs_statistical,
)
from changemaster.preprocessing.radiometric.selector import (
    METHOD_HISTOGRAM,
    METHOD_IRMAD,
    METHOD_PIF,
    RadiometricSelection,
    choose_method,
    normalize_pair,
)

__all__ = [
    "METHOD_HISTOGRAM",
    "METHOD_IRMAD",
    "METHOD_PIF",
    "IRMADResult",
    "IncrementalStats",
    "PIFNormalization",
    "RadiometricSelection",
    "apply_irmad_normalization",
    "apply_lut",
    "apply_pif_normalization",
    "build_matching_lut",
    "choose_method",
    "compute_irmad",
    "fit_pif_linear",
    "fit_pif_normalization",
    "match_histograms",
    "match_histograms_tiled",
    "normalize_pair",
    "select_pifs_statistical",
]
