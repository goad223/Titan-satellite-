"""Sensor profile model, registry and automatic sensor detection.

A :class:`SensorProfile` describes a satellite sensor: its bands, spatial
resolution and filename patterns used for auto-detection. Twelve built-in
profiles live in :mod:`changemaster.sensors.definitions`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from changemaster.core.exceptions import SensorProfileError


@dataclass(frozen=True)
class BandDefinition:
    """A single spectral band of a sensor.

    Attributes
    ----------
    band_id:
        Sensor-specific identifier, e.g. ``"B04"``.
    name:
        Human-readable name, e.g. ``"Red"``.
    wavelength_nm:
        Central wavelength in nanometres (``None`` for SAR/unknown).
    resolution_m:
        Ground sample distance in metres.
    """

    band_id: str
    name: str
    wavelength_nm: float | None
    resolution_m: float


@dataclass(frozen=True)
class SensorProfile:
    """Complete description of a satellite sensor.

    Attributes
    ----------
    sensor_id:
        Unique lowercase identifier, e.g. ``"sentinel2"``.
    display_name:
        English display name.
    display_name_ar:
        Arabic display name.
    platform:
        Spacecraft/constellation name.
    sensor_type:
        ``"optical"`` or ``"sar"``.
    bands:
        Tuple of :class:`BandDefinition`.
    default_resolution_m:
        Typical ground resolution in metres.
    filename_patterns:
        Regex patterns (case-insensitive) matched against product names for
        auto-detection.
    rgb_bands:
        Band IDs forming a natural-colour composite, when applicable.
    """

    sensor_id: str
    display_name: str
    display_name_ar: str
    platform: str
    sensor_type: str
    bands: tuple[BandDefinition, ...]
    default_resolution_m: float
    filename_patterns: tuple[str, ...] = ()
    rgb_bands: tuple[str, str, str] | None = None

    def __post_init__(self) -> None:
        if self.sensor_type not in ("optical", "sar"):
            raise SensorProfileError(
                f"Invalid sensor type '{self.sensor_type}' for {self.sensor_id}.",
                f"نوع مستشعر غير صالح '{self.sensor_type}' للمستشعر {self.sensor_id}.",
            )
        if self.default_resolution_m <= 0:
            raise SensorProfileError(
                f"Resolution must be positive for {self.sensor_id}.",
                f"يجب أن تكون الدقة موجبة للمستشعر {self.sensor_id}.",
            )

    def band(self, band_id: str) -> BandDefinition:
        """Return the band with ``band_id`` (case-insensitive).

        Raises :class:`SensorProfileError` for unknown bands.
        """
        for b in self.bands:
            if b.band_id.upper() == band_id.upper():
                return b
        raise SensorProfileError(
            f"Band '{band_id}' not defined for sensor {self.sensor_id}.",
            f"النطاق '{band_id}' غير معرف للمستشعر {self.sensor_id}.",
        )

    def matches_filename(self, name: str) -> bool:
        """True when any filename pattern matches ``name``."""
        return any(re.search(p, name, re.IGNORECASE) for p in self.filename_patterns)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return asdict(self)


class SensorRegistry:
    """Registry of :class:`SensorProfile` objects with auto-detection."""

    def __init__(self) -> None:
        self._profiles: dict[str, SensorProfile] = {}

    def register(self, profile: SensorProfile) -> SensorProfile:
        """Register a profile; raises on duplicate ``sensor_id``."""
        if profile.sensor_id in self._profiles:
            raise SensorProfileError(
                f"Sensor '{profile.sensor_id}' is already registered.",
                f"المستشعر '{profile.sensor_id}' مسجل مسبقاً.",
            )
        self._profiles[profile.sensor_id] = profile
        return profile

    @property
    def profiles(self) -> tuple[SensorProfile, ...]:
        """All registered profiles (in registration order)."""
        return tuple(self._profiles.values())

    def get(self, sensor_id: str) -> SensorProfile:
        """Return the profile for ``sensor_id`` (case-insensitive).

        Raises :class:`SensorProfileError` for unknown sensors.
        """
        profile = self._profiles.get(sensor_id.lower())
        if profile is None:
            raise SensorProfileError(
                f"Unknown sensor '{sensor_id}'. Known: {sorted(self._profiles)}",
                f"مستشعر غير معروف '{sensor_id}'. المعروف: {sorted(self._profiles)}",
            )
        return profile

    def detect(self, path: Path | str) -> SensorProfile:
        """Detect the sensor for a product file/directory name.

        Falls back to the ``"generic"`` profile when nothing matches.
        """
        name = Path(path).name
        for profile in self._profiles.values():
            if profile.sensor_id != "generic" and profile.matches_filename(name):
                return profile
        return self.get("generic")


#: Global sensor registry populated by the definitions package.
sensor_registry = SensorRegistry()

# Populate built-in profiles on import.
from changemaster.sensors import definitions as _definitions  # noqa: E402,F401
