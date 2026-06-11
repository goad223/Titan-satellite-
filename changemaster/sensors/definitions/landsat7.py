"""Landsat 7 ETM+ optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="landsat7",
        display_name="Landsat 7 (ETM+)",
        display_name_ar="لاندسات 7",
        platform="Landsat 7",
        sensor_type="optical",
        bands=(
            BandDefinition("B1", "Blue", 485.0, 30.0),
            BandDefinition("B2", "Green", 560.0, 30.0),
            BandDefinition("B3", "Red", 660.0, 30.0),
            BandDefinition("B4", "NIR", 835.0, 30.0),
            BandDefinition("B5", "SWIR 1", 1650.0, 30.0),
            BandDefinition("B6", "Thermal", 11450.0, 60.0),
            BandDefinition("B7", "SWIR 2", 2220.0, 30.0),
            BandDefinition("B8", "Panchromatic", 710.0, 15.0),
        ),
        default_resolution_m=30.0,
        filename_patterns=(r"^LE07_", r"^LE7", r"LANDSAT[-_]?7"),
        rgb_bands=("B3", "B2", "B1"),
    )
)
