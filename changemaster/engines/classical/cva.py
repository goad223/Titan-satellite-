"""Change Vector Analysis (CVA): multi-band magnitude + spectral direction.

The change vector at each pixel is the per-band standardized difference
``(moving - reference)``. Its Euclidean norm gives the change magnitude and
its angle in a (brightness, greenness)-like plane gives a direction that is
quantized into 8 sectors with bilingual semantic hints (vegetation gain /
loss, new construction, removal, ...) derived from the sensor band roles in
the Phase-1 sensor profiles.

All statistics (per-band mean/std, magnitude normalization percentile) are
accumulated over row chunks so the algorithm stays memory-bounded on huge
images; the chunk height adapts to the hardware tier (smaller windows on
weak machines — never lower accuracy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from changemaster.core.exceptions import EngineError, SensorProfileError
from changemaster.core.hardware import HardwareInfo
from changemaster.preprocessing._common import adaptive_tile_size

if TYPE_CHECKING:
    import numpy as np

#: Bilingual semantic hints for the 8 direction sectors when the angle plane
#: is (d_brightness, d_greenness). Sector k covers angles
#: [k*45° - 22.5°, k*45° + 22.5°) measured counter-clockwise from +x.
SECTOR_LABELS: dict[int, str] = {
    0: "brightening / new construction | سطوع / بناء جديد",
    1: "vegetation gain with brightening | زيادة نباتات مع سطوع",
    2: "vegetation gain | زيادة نباتات",
    3: "vegetation gain with darkening | زيادة نباتات مع إعتام",
    4: "darkening / water or shadow increase | إعتام / زيادة ماء أو ظل",
    5: "vegetation loss with darkening | فقدان نباتات مع إعتام",
    6: "vegetation loss / clearing | فقدان نباتات / إزالة",
    7: "vegetation loss with brightening / construction | فقدان نباتات مع سطوع / بناء",
}

#: Fallback labels when no red/NIR roles are known (plain band-space angle).
GENERIC_SECTOR_LABELS: dict[int, str] = {
    k: f"spectral direction sector {k} ({k * 45}°) | قطاع الاتجاه الطيفي {k}"
    for k in range(8)
}


@dataclass
class CVAResult:
    """Outcome of Change Vector Analysis.

    Attributes
    ----------
    probability:
        Float32 ``(H, W)`` normalized change magnitude in ``[0, 1]``
        (``NaN`` on invalid pixels).
    magnitude:
        Raw multi-band Euclidean magnitude of the standardized change vector.
    direction_sectors:
        Int16 ``(H, W)`` direction sector 0-7 (-1 on invalid pixels).
    sector_labels:
        Bilingual meaning of every sector code.
    warnings:
        Bilingual warnings (e.g. missing band roles).
    """

    probability: "np.ndarray"
    magnitude: "np.ndarray"
    direction_sectors: "np.ndarray"
    sector_labels: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _find_role_bands(
    sensor_id: str | None, n_bands: int, band_names: list[str] | None
) -> tuple[int | None, int | None]:
    """Locate (red_index, nir_index) 0-based band positions, best effort.

    Uses the sensor profile band wavelengths when available, then falls back
    to band-name matching. Returns ``(None, None)`` when undetectable.
    """
    red_idx: int | None = None
    nir_idx: int | None = None
    if sensor_id is not None:
        try:
            from changemaster.sensors.profiles import sensor_registry

            profile = sensor_registry.get(sensor_id)
            for i, band in enumerate(profile.bands[:n_bands]):
                if band.wavelength_nm is None:
                    continue
                if red_idx is None and 620 <= band.wavelength_nm <= 700:
                    red_idx = i
                if nir_idx is None and 760 <= band.wavelength_nm <= 920:
                    nir_idx = i
        except SensorProfileError:
            pass
    if (red_idx is None or nir_idx is None) and band_names:
        for i, name in enumerate(band_names[:n_bands]):
            lowered = name.lower()
            if red_idx is None and "red" in lowered and "edge" not in lowered:
                red_idx = i
            if nir_idx is None and ("nir" in lowered or "near infrared" in lowered):
                nir_idx = i
    return red_idx, nir_idx


def compute_cva(
    reference: "np.ndarray",
    moving: "np.ndarray",
    valid_mask: "np.ndarray",
    sensor_id: str | None = None,
    band_names: list[str] | None = None,
    hardware: HardwareInfo | None = None,
    clip_percentile: float = 99.0,
) -> CVAResult:
    """Run full multi-band Change Vector Analysis on a preprocessed pair.

    Steps:

    1. Per-band standardization: each band of each image is centred and
       scaled by the mean/std of the **valid** pixels of the reference image
       so all bands contribute comparably to the magnitude.
    2. Magnitude: Euclidean norm of the standardized difference vector.
    3. Probability: magnitude rescaled to ``[0, 1]`` by the
       ``clip_percentile`` of valid magnitudes (robust to outliers).
    4. Direction: the angle of the change vector in the
       (brightness, greenness) plane — built from the red/NIR roles when
       known, otherwise the first two bands — quantized into 8 sectors of
       45° with semantic labels.

    Parameters
    ----------
    reference / moving:
        ``(bands, H, W)`` co-registered, normalized arrays.
    valid_mask:
        Boolean ``(H, W)`` mask; invalid pixels are excluded from all
        statistics and marked ``NaN`` / -1 in the outputs.
    sensor_id:
        Sensor identifier used to resolve red/NIR band roles.
    band_names:
        Band names used as a role-detection fallback.
    hardware:
        Hardware snapshot for adaptive chunk sizing.
    clip_percentile:
        Percentile of valid magnitudes mapped to probability 1.0.

    Returns
    -------
    CVAResult
        Probability, raw magnitude, direction sectors and their meanings.
    """
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    if ref.shape != mov.shape or ref.ndim != 3:
        raise EngineError(
            f"CVA needs identical (bands, H, W) arrays; got {ref.shape} vs {mov.shape}.",
            f"يتطلب CVA مصفوفتين متطابقتين (bands, H, W)؛ وجد {ref.shape} و{mov.shape}.",
        )
    bands, height, width = ref.shape
    chunk_rows = max(64, adaptive_tile_size(hardware) // max(1, bands))
    warnings: list[str] = []

    # Pass 1: per-band mean/std on valid reference pixels (chunked).
    count = 0.0
    sums = np.zeros(bands)
    sumsq = np.zeros(bands)
    for r0 in range(0, height, chunk_rows):
        sl = slice(r0, min(height, r0 + chunk_rows))
        vm = valid_mask[sl]
        n = int(vm.sum())
        if n == 0:
            continue
        chunk = ref[:, sl][:, vm]
        count += n
        sums += chunk.sum(axis=1)
        sumsq += (chunk**2).sum(axis=1)
    if count < 2:
        raise EngineError(
            "Too few valid pixels for CVA statistics.",
            "عدد البكسلات الصالحة غير كافٍ لإحصاءات CVA.",
            suggestion_en="Check the validity mask coverage.",
            suggestion_ar="تحقق من تغطية قناع الصلاحية.",
        )
    means = sums / count
    stds = np.sqrt(np.maximum(sumsq / count - means**2, 0.0))
    stds[stds < 1e-12] = 1.0

    red_idx, nir_idx = _find_role_bands(sensor_id, bands, band_names)
    use_roles = red_idx is not None and nir_idx is not None and red_idx != nir_idx
    if use_roles:
        labels = dict(SECTOR_LABELS)
    else:
        labels = dict(GENERIC_SECTOR_LABELS)
        if bands >= 2:
            warnings.append(
                "Red/NIR roles unknown; direction sectors use the first two "
                "bands without vegetation semantics. | أدوار الأحمر/تحت الأحمر "
                "القريب غير معروفة؛ تستخدم قطاعات الاتجاه أول نطاقين دون دلالات نباتية."
            )

    # Pass 2: magnitude + direction, chunked.
    magnitude = np.full((height, width), np.nan, dtype=np.float32)
    sectors = np.full((height, width), -1, dtype=np.int16)
    for r0 in range(0, height, chunk_rows):
        sl = slice(r0, min(height, r0 + chunk_rows))
        diff = (mov[:, sl] - ref[:, sl]) / stds[:, None, None]
        mag = np.sqrt((diff**2).sum(axis=0))
        vm = valid_mask[sl]
        magnitude[sl] = np.where(vm, mag, np.nan).astype(np.float32)

        if bands >= 2:
            if use_roles:
                d_red = diff[red_idx]
                d_nir = diff[nir_idx]
                # Brightness axis ~ (d_red + d_nir), greenness ~ (d_nir - d_red).
                x_axis = (d_red + d_nir) / math.sqrt(2.0)
                y_axis = (d_nir - d_red) / math.sqrt(2.0)
            else:
                x_axis = diff[0]
                y_axis = diff[1]
            angle = np.arctan2(y_axis, x_axis)  # [-pi, pi]
            sector = np.floor(((angle + math.pi / 8.0) % (2.0 * math.pi)) / (math.pi / 4.0))
            sector = sector.astype(np.int16) % 8
            sectors[sl] = np.where(vm, sector, np.int16(-1))
        else:
            # Single band: only "increase" (0) vs "decrease" (4) directions.
            direction = np.where(diff[0] >= 0, np.int16(0), np.int16(4))
            sectors[sl] = np.where(vm, direction, np.int16(-1))

    if bands == 1:
        labels = {
            0: "value increase | زيادة القيمة",
            4: "value decrease | نقصان القيمة",
        }

    finite = magnitude[np.isfinite(magnitude)]
    scale = float(np.percentile(finite, clip_percentile)) if finite.size else 1.0
    if scale <= 0:
        scale = 1.0
    probability = np.clip(magnitude / scale, 0.0, 1.0).astype(np.float32)
    probability[~valid_mask] = np.nan
    return CVAResult(
        probability=probability,
        magnitude=magnitude,
        direction_sectors=sectors,
        sector_labels=labels,
        warnings=warnings,
    )
