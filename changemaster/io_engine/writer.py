"""Raster writers: georeferenced GeoTIFF output and PNG export.

GeoTIFF writing preserves CRS and affine transform via rasterio (lazy
import). PNG export works without heavy dependencies using Pillow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from changemaster.core.exceptions import DependencyMissingError, ImageWriteError
from changemaster.io_engine.metadata import GeoReference

if TYPE_CHECKING:
    import numpy as np


def _normalize_array(array: "np.ndarray") -> "np.ndarray":
    """Validate and normalize an array to ``(bands, height, width)`` shape."""
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ImageWriteError(
            f"Expected a 2-D or 3-D array, got {arr.ndim}-D.",
            f"المتوقع مصفوفة ثنائية أو ثلاثية الأبعاد، وجد {arr.ndim} أبعاد.",
        )
    if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ImageWriteError(
            f"Invalid array shape {arr.shape}.",
            f"شكل مصفوفة غير صالح {arr.shape}.",
        )
    return arr


def write_geotiff(
    path: Path | str,
    array: "np.ndarray",
    georef: GeoReference | None = None,
    nodata: float | None = None,
    compress: str = "deflate",
) -> Path:
    """Write a ``(bands, height, width)`` array to a GeoTIFF file.

    Parameters
    ----------
    path:
        Output file path (parent directories are created).
    array:
        2-D ``(height, width)`` or 3-D ``(bands, height, width)`` array.
    georef:
        Optional georeferencing (CRS + affine transform) to preserve.
    nodata:
        Optional no-data value stored in the file.
    compress:
        GeoTIFF compression (``"deflate"``, ``"lzw"``, ``"none"``).

    Returns
    -------
    Path
        The written file path.
    """
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise DependencyMissingError(
            "rasterio", "GeoTIFF writing", "كتابة GeoTIFF"
        ) from exc

    arr = _normalize_array(array)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile: dict[str, object] = {
        "driver": "GTiff",
        "width": arr.shape[2],
        "height": arr.shape[1],
        "count": arr.shape[0],
        "dtype": str(arr.dtype),
        "compress": compress,
        "BIGTIFF": "IF_SAFER",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    if georef is not None and georef.crs is not None:
        profile["crs"] = georef.crs
    if georef is not None and georef.transform is not None:
        a, b, c, d, e, f = georef.transform
        profile["transform"] = Affine(a, b, c, d, e, f)

    try:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)
    except Exception as exc:
        raise ImageWriteError(
            f"Failed to write GeoTIFF {out_path}: {exc}",
            f"فشل في كتابة ملف GeoTIFF ‏{out_path}: {exc}",
        ) from exc
    return out_path


def export_png(
    path: Path | str,
    array: "np.ndarray",
    percentile_stretch: tuple[float, float] | None = (2.0, 98.0),
) -> Path:
    """Export an array as an 8-bit PNG (visualisation/quick-look).

    Parameters
    ----------
    path:
        Output PNG path (parent directories are created).
    array:
        2-D grayscale or 3-D array; for 3-D input the first 1, 3 or 4 bands
        are written as L/RGB/RGBA respectively.
    percentile_stretch:
        ``(low, high)`` percentiles used for contrast stretching to 0-255.
        ``None`` disables stretching (values are clipped to 0-255).

    Returns
    -------
    Path
        The written file path.
    """
    import numpy as np

    try:
        from PIL import Image
    except ImportError as exc:
        raise DependencyMissingError("Pillow", "PNG export", "تصدير PNG") from exc

    arr = _normalize_array(array).astype(np.float64)
    bands = arr.shape[0]
    if bands == 2 or bands > 4:
        arr = arr[:1]
        bands = 1

    if percentile_stretch is not None:
        low_p, high_p = percentile_stretch
        scaled = np.empty_like(arr)
        for i in range(bands):
            band = arr[i]
            low, high = np.percentile(band, [low_p, high_p])
            if high <= low:
                scaled[i] = np.zeros_like(band)
            else:
                scaled[i] = (np.clip(band, low, high) - low) / (high - low) * 255.0
        arr = scaled
    arr8 = np.clip(arr, 0, 255).astype(np.uint8)

    if bands == 1:
        image = Image.fromarray(arr8[0], mode="L")
    else:
        image = Image.fromarray(np.transpose(arr8, (1, 2, 0)))

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.save(out_path, format="PNG")
    except OSError as exc:
        raise ImageWriteError(
            f"Failed to write PNG {out_path}: {exc}",
            f"فشل في كتابة ملف PNG ‏{out_path}: {exc}",
        ) from exc
    return out_path
