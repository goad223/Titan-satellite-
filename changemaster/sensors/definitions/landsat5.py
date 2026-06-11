"""Landsat 5 TM optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="landsat5",
        display_name="Landsat 5 (TM)",
        display_name_ar="لاندسات 5",
        platform="Landsat 5",
        sensor_type="optical",
        bands=(
            BandDefinition("B1", "Blue", 485.0, 30.0),
            BandDefinition("B2", "Green", 560.0, 30.0),
            BandDefinition("B3", "Red", 660.0, 30.0),
            BandDefinition("B4", "NIR", 830.0, 30.0),
            BandDefinition("B5", "SWIR 1", 1650.0, 30.0),
            BandDefinition("B6", "Thermal", 11450.0, 120.0),
            BandDefinition("B7", "SWIR 2", 2215.0, 30.0),
        ),
        default_resolution_m=30.0,
        filename_patterns=(r"^LT05_", r"^LT5", r"LANDSAT[-_]?5"),
        rgb_bands=("B3", "B2", "B1"),
    )
)
