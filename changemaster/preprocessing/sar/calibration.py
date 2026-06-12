"""SAR radiometric calibration: sigma0 from Sentinel-1 metadata.

Sentinel-1 GRD calibration applies ``sigma0 = DN^2 / A^2`` where ``A`` is
the sigmaNought calibration LUT interpolated from the sparse calibration
vectors (line/pixel grids) in the product's calibration annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from changemaster.core.exceptions import SARCalibrationError

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class CalibrationVector:
    """One calibration vector (a row of LUT samples at a given line).

    Attributes
    ----------
    line:
        Image line (row) of this vector.
    pixels:
        Sample pixel (column) positions.
    sigma_nought:
        Calibration gain ``A`` at each sample pixel.
    """

    line: int
    pixels: tuple[int, ...]
    sigma_nought: tuple[float, ...]


def build_calibration_lut(
    vectors: Sequence[CalibrationVector], height: int, width: int
) -> "np.ndarray":
    """Build a dense per-pixel sigmaNought LUT by bilinear interpolation.

    Parameters
    ----------
    vectors:
        Sparse calibration vectors sorted (or sortable) by line.
    height / width:
        Output image dimensions.

    Returns
    -------
    np.ndarray
        ``(height, width)`` float64 array of calibration gains.

    Raises
    ------
    SARCalibrationError
        When fewer than 2 vectors exist or pixel grids are invalid.
    """
    import numpy as np

    if len(vectors) < 2:
        raise SARCalibrationError(
            f"At least 2 calibration vectors are required, got {len(vectors)}.",
            f"يلزم متجها معايرة على الأقل، وجد {len(vectors)}.",
            suggestion_en="Parse all calibrationVector elements from the annotation XML.",
            suggestion_ar="حلّل كل عناصر calibrationVector من ملف XML الخاص بالمعايرة.",
        )
    ordered = sorted(vectors, key=lambda v: v.line)
    lines = np.asarray([v.line for v in ordered], dtype=np.float64)
    # Interpolate each vector to the full width first.
    rows = np.empty((len(ordered), width), dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    for i, vec in enumerate(ordered):
        px = np.asarray(vec.pixels, dtype=np.float64)
        sn = np.asarray(vec.sigma_nought, dtype=np.float64)
        if px.size != sn.size or px.size < 2:
            raise SARCalibrationError(
                f"Calibration vector at line {vec.line} is malformed.",
                f"متجه المعايرة عند السطر {vec.line} غير سليم.",
            )
        if np.any(sn <= 0):
            raise SARCalibrationError(
                f"Non-positive sigmaNought gain at line {vec.line}.",
                f"معامل sigmaNought غير موجب عند السطر {vec.line}.",
            )
        rows[i] = np.interp(cols, px, sn)
    # Then interpolate between vector lines for every output row.
    lut = np.empty((height, width), dtype=np.float64)
    row_idx = np.arange(height, dtype=np.float64)
    for c in range(width):
        lut[:, c] = np.interp(row_idx, lines, rows[:, c])
    return lut


def calibrate_sigma0(
    dn: "np.ndarray",
    lut: "np.ndarray",
    nodata_value: float | None = 0.0,
) -> "np.ndarray":
    """Apply sigma0 calibration ``sigma0 = DN^2 / A^2``.

    Parameters
    ----------
    dn:
        2-D digital-number amplitude image.
    lut:
        ``(H, W)`` sigmaNought gain LUT from :func:`build_calibration_lut`.
    nodata_value:
        Input DN value treated as nodata (output becomes NaN).

    Returns
    -------
    np.ndarray
        Linear-power sigma0 image (float64, NaN where nodata).
    """
    import numpy as np

    data = np.asarray(dn, dtype=np.float64)
    gains = np.asarray(lut, dtype=np.float64)
    if data.shape != gains.shape:
        raise SARCalibrationError(
            f"DN shape {data.shape} does not match LUT shape {gains.shape}.",
            f"شكل DN ‏{data.shape} لا يطابق شكل LUT ‏{gains.shape}.",
            suggestion_en="Build the LUT with the image's exact height and width.",
            suggestion_ar="ابنِ جدول المعايرة بنفس أبعاد الصورة تماماً.",
        )
    if np.any(gains <= 0):
        raise SARCalibrationError(
            "Calibration LUT contains non-positive gains.",
            "جدول المعايرة يحتوي معاملات غير موجبة.",
        )
    sigma0 = (data**2) / (gains**2)
    if nodata_value is not None:
        sigma0[data == nodata_value] = np.nan
    return sigma0


def parse_calibration_xml(xml_text: str) -> list[CalibrationVector]:
    """Parse Sentinel-1 calibration annotation XML into vectors.

    Extracts every ``<calibrationVector>``'s ``line``, ``pixel`` list and
    ``sigmaNought`` list.

    Raises
    ------
    SARCalibrationError
        When the document contains no usable calibration vectors.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)  # noqa: S314 - local product files only
    except Exception as exc:  # noqa: BLE001 - parse errors reported uniformly
        raise SARCalibrationError(
            f"Cannot parse calibration XML: {exc}",
            f"تعذر تحليل XML المعايرة: {exc}",
            suggestion_en="Check the annotation/calibration/*.xml file integrity.",
            suggestion_ar="تحقق من سلامة ملف annotation/calibration/*.xml.",
        ) from exc

    vectors: list[CalibrationVector] = []
    for vec in root.iter("calibrationVector"):
        line_el = vec.find("line")
        pixel_el = vec.find("pixel")
        sigma_el = vec.find("sigmaNought")
        if line_el is None or pixel_el is None or sigma_el is None:
            continue
        try:
            line = int(line_el.text or "")
            pixels = tuple(int(p) for p in (pixel_el.text or "").split())
            sigma = tuple(float(s) for s in (sigma_el.text or "").split())
        except ValueError:
            continue
        if pixels and len(pixels) == len(sigma):
            vectors.append(CalibrationVector(line=line, pixels=pixels, sigma_nought=sigma))
    if not vectors:
        raise SARCalibrationError(
            "No calibration vectors found in the XML document.",
            "لم يُعثر على متجهات معايرة في وثيقة XML.",
            suggestion_en="Pass the calibration-*.xml (not the noise or product XML).",
            suggestion_ar="مرر ملف calibration-*.xml وليس ملف الضوضاء أو المنتج.",
        )
    return vectors
