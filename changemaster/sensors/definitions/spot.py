"""SPOT 6/7 optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="spot",
        display_name="SPOT 6/7",
        display_name_ar="سبوت 6/7",
        platform="SPOT",
        sensor_type="optical",
        bands=(
            BandDefinition("B0", "Blue", 485.0, 6.0),
            BandDefinition("B1", "Green", 560.0, 6.0),
            BandDefinition("B2", "Red", 660.0, 6.0),
            BandDefinition("B3", "NIR", 825.0, 6.0),
            BandDefinition("P", "Panchromatic", 625.0, 1.5),
        ),
        default_resolution_m=6.0,
        filename_patterns=(r"SPOT[-_]?[67]", r"^SPOT"),
        rgb_bands=("B2", "B1", "B0"),
    )
)
