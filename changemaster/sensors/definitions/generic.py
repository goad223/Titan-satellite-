"""Generic fallback sensor profile for unrecognized imagery."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="generic",
        display_name="Generic imagery",
        display_name_ar="صور عامة",
        platform="Unknown",
        sensor_type="optical",
        bands=(
            BandDefinition("B1", "Band 1", None, 1.0),
            BandDefinition("B2", "Band 2", None, 1.0),
            BandDefinition("B3", "Band 3", None, 1.0),
        ),
        default_resolution_m=1.0,
        filename_patterns=(),
        rgb_bands=("B1", "B2", "B3"),
    )
)
