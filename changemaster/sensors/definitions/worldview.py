"""WorldView-2/3 high-resolution optical sensor profile."""

from __future__ import annotations

from changemaster.sensors.profiles import BandDefinition, SensorProfile, sensor_registry

PROFILE: SensorProfile = sensor_registry.register(
    SensorProfile(
        sensor_id="worldview",
        display_name="WorldView-2/3",
        display_name_ar="وورلد فيو 2/3",
        platform="WorldView",
        sensor_type="optical",
        bands=(
            BandDefinition("C", "Coastal", 425.0, 1.84),
            BandDefinition("B", "Blue", 480.0, 1.84),
            BandDefinition("G", "Green", 545.0, 1.84),
            BandDefinition("Y", "Yellow", 605.0, 1.84),
            BandDefinition("R", "Red", 660.0, 1.84),
            BandDefinition("RE", "Red edge", 725.0, 1.84),
            BandDefinition("N1", "NIR 1", 832.5, 1.84),
            BandDefinition("N2", "NIR 2", 950.0, 1.84),
            BandDefinition("P", "Panchromatic", 625.0, 0.46),
        ),
        default_resolution_m=1.84,
        filename_patterns=(r"WV0?[23]", r"WORLDVIEW"),
        rgb_bands=("R", "G", "B"),
    )
)
