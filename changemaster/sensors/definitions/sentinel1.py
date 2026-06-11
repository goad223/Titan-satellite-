"""Sentinel-1 C-band SAR sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="sentinel1",
        display_name="Sentinel-1 (C-SAR)",
        display_name_ar="سنتينل-1 (رادار)",
        platform="Sentinel-1",
        sensor_type="sar",
        bands=(
            BandDefinition("VV", "VV polarisation", None, 10.0),
            BandDefinition("VH", "VH polarisation", None, 10.0),
            BandDefinition("HH", "HH polarisation", None, 10.0),
            BandDefinition("HV", "HV polarisation", None, 10.0),
        ),
        default_resolution_m=10.0,
        filename_patterns=(r"^S1[ABCD]_", r"SENTINEL[-_]?1"),
        rgb_bands=None,
    )
)
