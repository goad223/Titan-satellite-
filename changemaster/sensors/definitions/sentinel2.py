"""Sentinel-2 MSI optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="sentinel2",
        display_name="Sentinel-2 (MSI)",
        display_name_ar="سنتينل-2 (بصري)",
        platform="Sentinel-2",
        sensor_type="optical",
        bands=(
            BandDefinition("B01", "Coastal aerosol", 443.0, 60.0),
            BandDefinition("B02", "Blue", 490.0, 10.0),
            BandDefinition("B03", "Green", 560.0, 10.0),
            BandDefinition("B04", "Red", 665.0, 10.0),
            BandDefinition("B05", "Red edge 1", 705.0, 20.0),
            BandDefinition("B06", "Red edge 2", 740.0, 20.0),
            BandDefinition("B07", "Red edge 3", 783.0, 20.0),
            BandDefinition("B08", "NIR", 842.0, 10.0),
            BandDefinition("B8A", "Narrow NIR", 865.0, 20.0),
            BandDefinition("B09", "Water vapour", 945.0, 60.0),
            BandDefinition("B10", "Cirrus", 1375.0, 60.0),
            BandDefinition("B11", "SWIR 1", 1610.0, 20.0),
            BandDefinition("B12", "SWIR 2", 2190.0, 20.0),
        ),
        default_resolution_m=10.0,
        filename_patterns=(r"^S2[ABCD]_", r"SENTINEL[-_]?2"),
        rgb_bands=("B04", "B03", "B02"),
    )
)
