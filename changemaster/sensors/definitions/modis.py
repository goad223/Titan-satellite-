"""MODIS (Terra/Aqua) optical sensor profile (first 7 land bands)."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="modis",
        display_name="MODIS (Terra/Aqua)",
        display_name_ar="موديس (تيرا/أكوا)",
        platform="Terra/Aqua",
        sensor_type="optical",
        bands=(
            BandDefinition("B1", "Red", 645.0, 250.0),
            BandDefinition("B2", "NIR", 858.5, 250.0),
            BandDefinition("B3", "Blue", 469.0, 500.0),
            BandDefinition("B4", "Green", 555.0, 500.0),
            BandDefinition("B5", "SWIR 1", 1240.0, 500.0),
            BandDefinition("B6", "SWIR 2", 1640.0, 500.0),
            BandDefinition("B7", "SWIR 3", 2130.0, 500.0),
        ),
        default_resolution_m=250.0,
        filename_patterns=(r"^MOD\d{2}", r"^MYD\d{2}", r"MODIS"),
        rgb_bands=("B1", "B4", "B3"),
    )
)
