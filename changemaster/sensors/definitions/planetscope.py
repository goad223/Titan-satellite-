"""PlanetScope (Dove constellation) optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="planetscope",
        display_name="PlanetScope (Dove)",
        display_name_ar="بلانت سكوب (دوف)",
        platform="PlanetScope",
        sensor_type="optical",
        bands=(
            BandDefinition("B1", "Blue", 490.0, 3.0),
            BandDefinition("B2", "Green", 565.0, 3.0),
            BandDefinition("B3", "Red", 665.0, 3.0),
            BandDefinition("B4", "NIR", 865.0, 3.0),
        ),
        default_resolution_m=3.0,
        filename_patterns=(r"PLANETSCOPE", r"PSScene", r"_psscene", r"\dPS[BD]?_"),
        rgb_bands=("B3", "B2", "B1"),
    )
)
