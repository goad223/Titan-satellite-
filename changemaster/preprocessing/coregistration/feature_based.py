"""Feature-based matching: SIFT + ORB + AKAZE combined, FLANN + RANSAC.

All three detectors run on the same band pair; their matches are merged so
the registration benefits from SIFT's accuracy, ORB's speed-features and
AKAZE's robustness. A FLANN matcher with Lowe's ratio test filters the raw
matches, and RANSAC rejects the remaining outliers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import CoregistrationError
from changemaster.preprocessing._common import normalize_to_uint8, require_cv2

if TYPE_CHECKING:
    import numpy as np


@dataclass
class MatchResult:
    """Point correspondences produced by feature matching.

    Attributes
    ----------
    src_points:
        ``(N, 2)`` matched ``(x, y)`` points in the moving image.
    dst_points:
        ``(N, 2)`` matched ``(x, y)`` points in the reference image.
    detector_counts:
        Raw match count contributed by each detector.
    inlier_count:
        Matches surviving RANSAC.
    """

    src_points: "np.ndarray"
    dst_points: "np.ndarray"
    detector_counts: dict[str, int] = field(default_factory=dict)
    inlier_count: int = 0


def _detect_and_match(
    detector: Any,
    norm_type: int,
    ref8: "np.ndarray",
    mov8: "np.ndarray",
    ratio: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Run one detector on both images and FLANN-match with ratio test."""
    cv2 = require_cv2()
    import numpy as np

    kp_ref, desc_ref = detector.detectAndCompute(ref8, None)
    kp_mov, desc_mov = detector.detectAndCompute(mov8, None)
    if desc_ref is None or desc_mov is None or len(kp_ref) < 2 or len(kp_mov) < 2:
        return [], []

    if norm_type == cv2.NORM_HAMMING:
        # FLANN with LSH index for binary descriptors (ORB / AKAZE).
        index_params = {"algorithm": 6, "table_number": 6, "key_size": 12, "multi_probe_level": 1}
        d_ref, d_mov = desc_ref, desc_mov
    else:
        # FLANN with KD-tree for float descriptors (SIFT).
        index_params = {"algorithm": 1, "trees": 5}
        d_ref = np.asarray(desc_ref, dtype=np.float32)
        d_mov = np.asarray(desc_mov, dtype=np.float32)
    matcher = cv2.FlannBasedMatcher(index_params, {"checks": 64})
    try:
        knn = matcher.knnMatch(d_mov, d_ref, k=2)
    except cv2.error:
        return [], []

    src: list[tuple[float, float]] = []
    dst: list[tuple[float, float]] = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            src.append(kp_mov[m.queryIdx].pt)
            dst.append(kp_ref[m.trainIdx].pt)
    return src, dst


def extract_and_match_features(
    reference_band: "np.ndarray",
    moving_band: "np.ndarray",
    ratio: float = 0.75,
    ransac_threshold: float = 1.0,
    max_features: int = 4000,
) -> MatchResult:
    """Match a band pair using SIFT + ORB + AKAZE merged, FLANN and RANSAC.

    Parameters
    ----------
    reference_band / moving_band:
        2-D arrays (any dtype); internally percentile-stretched to uint8.
    ratio:
        Lowe's ratio-test threshold.
    ransac_threshold:
        RANSAC reprojection threshold (pixels).
    max_features:
        Per-detector feature budget.

    Returns
    -------
    MatchResult
        RANSAC-inlier correspondences (moving -> reference).

    Raises
    ------
    CoregistrationError
        When too few matches survive for a geometric model.
    """
    cv2 = require_cv2()
    import numpy as np

    ref8 = normalize_to_uint8(reference_band)
    mov8 = normalize_to_uint8(moving_band)

    detectors: list[tuple[str, Any, int]] = []
    detectors.append(("sift", cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2))
    detectors.append(("orb", cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING))
    detectors.append(("akaze", cv2.AKAZE_create(), cv2.NORM_HAMMING))

    all_src: list[tuple[float, float]] = []
    all_dst: list[tuple[float, float]] = []
    counts: dict[str, int] = {}
    for name, detector, norm in detectors:
        src, dst = _detect_and_match(detector, norm, ref8, mov8, ratio)
        counts[name] = len(src)
        all_src.extend(src)
        all_dst.extend(dst)

    if len(all_src) < 4:
        raise CoregistrationError(
            f"Only {len(all_src)} feature matches found; at least 4 are required.",
            f"وُجد {len(all_src)} تطابقات فقط؛ المطلوب 4 على الأقل.",
            suggestion_en=(
                "Try a larger overlap area, different bands (NIR/SWIR) or "
                "area-based registration."
            ),
            suggestion_ar=(
                "جرّب منطقة تداخل أكبر أو نطاقات مختلفة (NIR/SWIR) أو التسجيل "
                "القائم على المساحة."
            ),
        )

    src_arr = np.asarray(all_src, dtype=np.float64)
    dst_arr = np.asarray(all_dst, dtype=np.float64)

    _, inlier_mask = cv2.findHomography(
        src_arr.astype(np.float32),
        dst_arr.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )
    if inlier_mask is None:
        raise CoregistrationError(
            "RANSAC failed to find a consistent geometric model.",
            "فشل RANSAC في إيجاد نموذج هندسي متسق.",
            suggestion_en="Increase the RANSAC threshold or check that the images overlap.",
            suggestion_ar="زد عتبة RANSAC أو تحقق من تداخل الصورتين.",
        )
    inliers = inlier_mask.ravel().astype(bool)
    if int(inliers.sum()) < 3:
        raise CoregistrationError(
            f"Only {int(inliers.sum())} RANSAC inliers; registration unreliable.",
            f"عدد النقاط الداخلية بعد RANSAC هو {int(inliers.sum())} فقط؛ التسجيل غير موثوق.",
            suggestion_en="Use coarse georeferenced alignment first, then retry feature matching.",
            suggestion_ar="استخدم المحاذاة التقريبية من الميتاداتا أولاً ثم أعد مطابقة الميزات.",
        )
    return MatchResult(
        src_points=src_arr[inliers],
        dst_points=dst_arr[inliers],
        detector_counts=counts,
        inlier_count=int(inliers.sum()),
    )


def pick_matching_band(
    band_names: list[str], band_count: int
) -> int:
    """Pick the best band (1-based) for matching: prefer SWIR, then NIR.

    Vegetation-insensitive bands (SWIR/NIR) are more stable across seasons,
    so feature matching prefers them when present.
    """
    lowered = [name.lower() for name in band_names]
    for keyword in ("swir", "nir"):
        for i, name in enumerate(lowered):
            if keyword in name:
                return i + 1
    for keyword in ("red",):
        for i, name in enumerate(lowered):
            if keyword in name:
                return i + 1
    return 1 if band_count >= 1 else 1
