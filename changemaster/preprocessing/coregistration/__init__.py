"""Geometric co-registration subpackage (Phase 2).

Exposes the full coarse-to-fine registration stack:
metadata coarse alignment, SIFT+ORB+AKAZE feature matching, phase
correlation + ECC, transform models (Affine/Homography/TPS), pyramid
strategy and independent-checkpoint evaluation.
"""

from changemaster.preprocessing.coregistration.area_based import (
    AreaRegistrationResult,
    WindowShift,
    ecc_refine,
    grid_phase_correlation,
    phase_correlation_shift,
)
from changemaster.preprocessing.coregistration.coarse import (
    CoarseAlignment,
    coarse_align_from_metadata,
)
from changemaster.preprocessing.coregistration.evaluator import (
    RegistrationEvaluation,
    displacement_map,
    evaluate_registration,
    split_matches,
)
from changemaster.preprocessing.coregistration.feature_based import (
    MatchResult,
    extract_and_match_features,
    pick_matching_band,
)
from changemaster.preprocessing.coregistration.pyramid import (
    PyramidRegistrationResult,
    build_pyramid,
    pyramid_levels_for,
    register_pyramid,
)
from changemaster.preprocessing.coregistration.transforms import (
    AffineTransform,
    GeometricTransform,
    HomographyTransform,
    ThinPlateSplineTransform,
    select_transform_model,
)

__all__ = [
    "AffineTransform",
    "AreaRegistrationResult",
    "CoarseAlignment",
    "GeometricTransform",
    "HomographyTransform",
    "MatchResult",
    "PyramidRegistrationResult",
    "RegistrationEvaluation",
    "ThinPlateSplineTransform",
    "WindowShift",
    "build_pyramid",
    "coarse_align_from_metadata",
    "displacement_map",
    "ecc_refine",
    "evaluate_registration",
    "extract_and_match_features",
    "grid_phase_correlation",
    "phase_correlation_shift",
    "pick_matching_band",
    "pyramid_levels_for",
    "register_pyramid",
    "select_transform_model",
    "split_matches",
]
