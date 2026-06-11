"""Landsat 8 OLI/TIRS optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="landsat8",
        display_name="Landsat 8 (OLI/TIRS)",
        display_name_ar="لاندسات 8",
        platform="Landsat 8",
        sensor_type="optical",
        bands=(
            BandDefinition("B1", "Coastal aerosol", 443.0, 30.0),
            BandDefinition("B2", "Blue", 482.0, 30.0),
            BandDefinition("B3", "Green", 561.5, 30.0),
            BandDefinition("B4", "Red", 654.5, 30.0),
            BandDefinition("B5", "NIR", 865.0, 30.0),
            BandDefinition("B6", "SWIR 1", 1608.5, 30.0),
            BandDefinition("B7", "SWIR 2", 2200.5, 30.0),
            BandDefinition("B8", "Panchromatic", 589.5, 15.0),
            BandDefinition("B9", "Cirrus", 1373.5, 30.0),
            BandDefinition("B10", "Thermal 1", 10895.0, 100.0),
            BandDefinition("B11", "Thermal 2", 12005.0, 100.0),
        ),
        default_resolution_m=30.0,
        filename_patterns=(r"^LC08_", r"^LC8", r"LANDSAT[-_]?8"),
        rgb_bands=("B4", "B3", "B2"),
    )
)
