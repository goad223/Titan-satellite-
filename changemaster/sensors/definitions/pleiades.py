"""Pleiades 1A/1B high-resolution optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="pleiades",
        display_name="Pleiades 1A/1B",
        display_name_ar="بلياديس 1A/1B",
        platform="Pleiades",
        sensor_type="optical",
        bands=(
            BandDefinition("B0", "Blue", 490.0, 2.0),
            BandDefinition("B1", "Green", 555.0, 2.0),
            BandDefinition("B2", "Red", 650.0, 2.0),
            BandDefinition("B3", "NIR", 840.0, 2.0),
            BandDefinition("P", "Panchromatic", 650.0, 0.5),
        ),
        default_resolution_m=2.0,
        filename_patterns=(r"PHR1[AB]", r"PLEIADES"),
        rgb_bands=("B2", "B1", "B0"),
    )
)
