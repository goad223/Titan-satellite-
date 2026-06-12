"""Coarse initial alignment from geographic metadata.

When both images carry CRS + affine transforms in the *same* CRS, the
pixel-level offset between them can be computed analytically before any
image-content matching. This shrinks the search range for the fine stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from changemaster.core.exceptions import CoregistrationError
from changemaster.io_engine.metadata import GeoReference, ImageMetadata


@dataclass
class CoarseAlignment:
    """Coarse pixel offset between two georeferenced rasters.

    Attributes
    ----------
    offset_xy:
        ``(dx, dy)`` such that moving pixel ``(col + dx, row + dy)``
        corresponds to reference pixel ``(col, row)``.
    pixel_size_ratio:
        Moving / reference ground pixel size ratio (resampling hint).
    same_crs:
        ``True`` when both rasters share the same CRS string.
    """

    offset_xy: tuple[float, float]
    pixel_size_ratio: float
    same_crs: bool


def _pixel_size(georef: GeoReference) -> tuple[float, float]:
    """Absolute ground pixel size ``(x_size, y_size)`` from a transform."""
    assert georef.transform is not None
    a, b, _, d, e, _ = georef.transform
    return (abs(a) + abs(b), abs(d) + abs(e))


def coarse_align_from_metadata(
    reference: ImageMetadata, moving: ImageMetadata
) -> CoarseAlignment:
    """Compute the coarse pixel offset between two georeferenced rasters.

    Parameters
    ----------
    reference / moving:
        Metadata of the reference and moving images. Both must be
        georeferenced; for differing CRS strings only a same-CRS fast path
        is implemented and a :class:`CoregistrationError` is raised
        suggesting harmonization first.

    Returns
    -------
    CoarseAlignment
        Pixel offset of the moving image relative to the reference grid.

    Raises
    ------
    CoregistrationError
        When either image lacks georeferencing, or CRSs differ.
    """
    if not reference.georef.is_georeferenced or not moving.georef.is_georeferenced:
        raise CoregistrationError(
            "Coarse alignment requires georeferencing on both images.",
            "تتطلب المحاذاة التقريبية مرجعية جغرافية في كلتا الصورتين.",
            suggestion_en=(
                "Skip the coarse step and rely on the feature-based pyramid "
                "registration for non-georeferenced imagery."
            ),
            suggestion_ar=(
                "تجاوز خطوة المحاذاة التقريبية واعتمد على التسجيل الهرمي "
                "القائم على الميزات للصور غير المرجعة جغرافياً."
            ),
        )
    same_crs = (reference.georef.crs or "").strip().upper() == (
        moving.georef.crs or ""
    ).strip().upper()
    if not same_crs:
        raise CoregistrationError(
            f"CRS mismatch: {reference.georef.crs} vs {moving.georef.crs}.",
            f"عدم تطابق نظام الإحداثيات: {reference.georef.crs} مقابل {moving.georef.crs}.",
            suggestion_en="Run harmonization (reprojection) before coarse alignment.",
            suggestion_ar="نفّذ التوحيد (إعادة الإسقاط) قبل المحاذاة التقريبية.",
        )

    # World coordinates of each image's pixel (0, 0).
    ref_x0, ref_y0 = reference.georef.pixel_to_coords(0, 0)
    mov_x0, mov_y0 = moving.georef.pixel_to_coords(0, 0)
    ref_sx, ref_sy = _pixel_size(reference.georef)
    mov_sx, mov_sy = _pixel_size(moving.georef)
    if ref_sx <= 0 or ref_sy <= 0:
        raise CoregistrationError(
            "Reference pixel size is zero or negative.",
            "حجم بكسل الصورة المرجعية صفر أو سالب.",
        )
    # Offset of the moving origin inside the reference pixel grid.
    a, b, c, d, e, f = reference.georef.transform  # type: ignore[misc]
    det = a * e - b * d
    if det == 0:
        raise CoregistrationError(
            "Reference affine transform is singular.",
            "التحويل الأفيني للصورة المرجعية منفرد.",
        )
    dx = (e * (mov_x0 - c) - b * (mov_y0 - f)) / det
    dy = (-d * (mov_x0 - c) + a * (mov_y0 - f)) / det
    ratio = ((mov_sx / ref_sx) + (mov_sy / ref_sy)) / 2.0
    return CoarseAlignment(
        offset_xy=(float(dx), float(dy)), pixel_size_ratio=float(ratio), same_crs=True
    )
