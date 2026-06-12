"""Pair harmonization: CRS, resampling, overlap cropping and band lists.

Brings a reference/moving pair onto a single grid: reproject the moving
image to the reference CRS (rasterio, lazy import), resample to the
reference pixel size, crop both to their geographic intersection and align
the band lists by name. A pixel-space fallback handles non-georeferenced
imagery by resizing to the reference shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import DependencyMissingError, PreprocessingError
from changemaster.io_engine.metadata import GeoReference, ImageMetadata

if TYPE_CHECKING:
    import numpy as np


@dataclass
class HarmonizedPair:
    """A reference/moving pair on one common grid.

    Attributes
    ----------
    reference / moving:
        ``(bands, H, W)`` arrays on the shared grid.
    georef:
        Georeferencing of the common grid (empty for pixel-space pairs).
    band_names:
        Common band names (aligned across both images).
    nodata:
        Propagated nodata value, if any.
    warnings:
        Bilingual warnings accumulated during harmonization.
    """

    reference: "np.ndarray"
    moving: "np.ndarray"
    georef: GeoReference
    band_names: list[str] = field(default_factory=list)
    nodata: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (arrays excluded)."""
        return {
            "shape": list(self.reference.shape),
            "crs": self.georef.crs,
            "transform": list(self.georef.transform) if self.georef.transform else None,
            "band_names": self.band_names,
            "nodata": self.nodata,
            "warnings": self.warnings,
        }


def common_bands(
    reference_names: list[str], moving_names: list[str]
) -> tuple[list[int], list[int], list[str]]:
    """Find band indices (1-based) shared by name between two band lists.

    Returns ``(ref_indices, mov_indices, names)``; when no names match,
    falls back to pairing the first ``min(n1, n2)`` bands positionally.
    """
    ref_lower = [n.lower() for n in reference_names]
    mov_lower = [n.lower() for n in moving_names]
    ref_idx: list[int] = []
    mov_idx: list[int] = []
    names: list[str] = []
    for i, name in enumerate(ref_lower):
        if name in mov_lower:
            ref_idx.append(i + 1)
            mov_idx.append(mov_lower.index(name) + 1)
            names.append(reference_names[i])
    if not ref_idx:
        n = min(len(reference_names), len(moving_names))
        ref_idx = list(range(1, n + 1))
        mov_idx = list(range(1, n + 1))
        names = reference_names[:n]
    return ref_idx, mov_idx, names


def _resize_to(image: "np.ndarray", shape: tuple[int, int]) -> "np.ndarray":
    """Bilinear-resize a ``(bands, H, W)`` array to ``shape`` via OpenCV."""
    from changemaster.preprocessing._common import require_cv2

    cv2 = require_cv2()
    import numpy as np

    h, w = shape
    out = np.empty((image.shape[0], h, w), dtype=np.float64)
    for b in range(image.shape[0]):
        out[b] = cv2.resize(
            image[b].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
    return out


def harmonize_arrays(
    reference: "np.ndarray",
    moving: "np.ndarray",
    reference_meta: ImageMetadata | None = None,
    moving_meta: ImageMetadata | None = None,
) -> HarmonizedPair:
    """Harmonize two in-memory images onto the reference grid.

    Georeferenced same-CRS pairs are cropped to their overlap window; other
    pairs are aligned in pixel space (moving resized to the reference
    shape, with a warning).

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` (or 2-D) arrays.
    reference_meta / moving_meta:
        Optional metadata enabling geographic overlap computation and
        band-name alignment.

    Raises
    ------
    PreprocessingError
        When the pair has no geographic overlap.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.ndim == 2:
        ref = ref[np.newaxis]
    if mov.ndim == 2:
        mov = mov[np.newaxis]
    warnings: list[str] = []

    # Band alignment by name when metadata is present.
    band_names = [f"Band {i + 1}" for i in range(min(ref.shape[0], mov.shape[0]))]
    if reference_meta is not None and moving_meta is not None:
        ref_idx, mov_idx, band_names = common_bands(
            reference_meta.band_names, moving_meta.band_names
        )
        ref = ref[[i - 1 for i in ref_idx]]
        mov = mov[[i - 1 for i in mov_idx]]
    else:
        n = min(ref.shape[0], mov.shape[0])
        if ref.shape[0] != mov.shape[0]:
            warnings.append(
                f"Band counts differ ({ref.shape[0]} vs {mov.shape[0]}); using the "
                f"first {n} bands of each. | عدد النطاقات مختلف؛ تُستخدم أول {n} نطاقات."
            )
        ref = ref[:n]
        mov = mov[:n]

    georef = GeoReference()
    nodata = reference_meta.nodata if reference_meta is not None else None

    geo_ok = (
        reference_meta is not None
        and moving_meta is not None
        and reference_meta.georef.is_georeferenced
        and moving_meta.georef.is_georeferenced
        and (reference_meta.georef.crs or "").upper() == (moving_meta.georef.crs or "").upper()
    )
    if geo_ok:
        assert reference_meta is not None and moving_meta is not None
        ref_g, mov_g = reference_meta.georef, moving_meta.georef
        # Compute the overlap rectangle in reference pixel space.
        from changemaster.preprocessing.coregistration.coarse import (
            coarse_align_from_metadata,
        )

        coarse = coarse_align_from_metadata(reference_meta, moving_meta)
        dx, dy = coarse.offset_xy
        if abs(coarse.pixel_size_ratio - 1.0) > 1e-6:
            new_h = max(1, int(round(mov.shape[1] * coarse.pixel_size_ratio)))
            new_w = max(1, int(round(mov.shape[2] * coarse.pixel_size_ratio)))
            mov = _resize_to(mov, (new_h, new_w))
            warnings.append(
                f"Moving image resampled by factor {coarse.pixel_size_ratio:.3f} to the "
                f"reference pixel size. | أعيدت معاينة الصورة المتحركة بمعامل "
                f"{coarse.pixel_size_ratio:.3f} لمطابقة حجم بكسل المرجع."
            )
        # Overlap window in reference pixel coordinates.
        r0 = max(0, int(round(dy)))
        c0 = max(0, int(round(dx)))
        r1 = min(ref.shape[1], int(round(dy)) + mov.shape[1])
        c1 = min(ref.shape[2], int(round(dx)) + mov.shape[2])
        if r1 <= r0 or c1 <= c0:
            raise PreprocessingError(
                "The image pair has no geographic overlap.",
                "لا يوجد تداخل جغرافي بين الصورتين.",
                suggestion_en="Verify both products cover the same area and CRS.",
                suggestion_ar="تحقق من أن المنتجين يغطيان نفس المنطقة ونظام الإحداثيات.",
            )
        mr0 = r0 - int(round(dy))
        mc0 = c0 - int(round(dx))
        ref = ref[:, r0:r1, c0:c1]
        mov = mov[:, mr0 : mr0 + (r1 - r0), mc0 : mc0 + (c1 - c0)]
        # New transform anchored at the overlap origin.
        a, b, c, d, e, f = ref_g.transform  # type: ignore[misc]
        x0, y0 = ref_g.pixel_to_coords(r0, c0)
        georef = GeoReference(crs=ref_g.crs, transform=(a, b, x0, d, e, y0))
        _ = mov_g
    else:
        if ref.shape[1:] != mov.shape[1:]:
            warnings.append(
                "Pair not on a common geographic grid; the moving image was resized "
                f"from {mov.shape[1:]} to {ref.shape[1:]} in pixel space. | الزوج ليس "
                f"على شبكة جغرافية موحدة؛ غُيّر حجم الصورة المتحركة من {mov.shape[1:]} "
                f"إلى {ref.shape[1:]} في فضاء البكسل."
            )
            mov = _resize_to(mov, (ref.shape[1], ref.shape[2]))
        if reference_meta is not None and reference_meta.georef.is_georeferenced:
            georef = reference_meta.georef

    return HarmonizedPair(
        reference=ref,
        moving=mov,
        georef=georef,
        band_names=band_names,
        nodata=nodata,
        warnings=warnings,
    )


def reproject_to_reference(
    moving_path: str,
    reference_meta: ImageMetadata,
    resampling: str = "bilinear",
) -> "np.ndarray":
    """Reproject a raster file onto the reference CRS/grid via rasterio.

    Parameters
    ----------
    moving_path:
        Path of the moving raster (any CRS).
    reference_meta:
        Metadata defining the target CRS, transform and shape.
    resampling:
        ``"nearest"``, ``"bilinear"`` or ``"cubic"``.

    Returns
    -------
    np.ndarray
        ``(bands, H, W)`` array on the reference grid.

    Raises
    ------
    DependencyMissingError
        When rasterio is not installed.
    PreprocessingError
        When the reference is not georeferenced.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from rasterio.warp import reproject
    except ImportError as exc:
        raise DependencyMissingError(
            "rasterio", "CRS reprojection", "إعادة الإسقاط بين أنظمة الإحداثيات"
        ) from exc
    import numpy as np

    if not reference_meta.georef.is_georeferenced:
        raise PreprocessingError(
            "Reference image is not georeferenced; cannot reproject onto it.",
            "الصورة المرجعية غير مرجعة جغرافياً؛ تعذر إعادة الإسقاط عليها.",
            suggestion_en="Use harmonize_arrays for pixel-space alignment instead.",
            suggestion_ar="استخدم harmonize_arrays للمحاذاة في فضاء البكسل بدلاً من ذلك.",
        )
    method = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }.get(resampling)
    if method is None:
        raise PreprocessingError(
            f"Unknown resampling '{resampling}'.",
            f"طريقة إعادة معاينة غير معروفة '{resampling}'.",
            suggestion_en="Use nearest, bilinear or cubic.",
            suggestion_ar="استخدم nearest أو bilinear أو cubic.",
        )
    a, b, c, d, e, f = reference_meta.georef.transform  # type: ignore[misc]
    dst_transform = Affine(a, b, c, d, e, f)
    with rasterio.open(moving_path) as src:
        out = np.zeros(
            (src.count, reference_meta.height, reference_meta.width), dtype=np.float64
        )
        for band in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=out[band - 1],
                dst_transform=dst_transform,
                dst_crs=reference_meta.georef.crs,
                resampling=method,
            )
    return out
