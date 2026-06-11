"""Tests for sensor profiles, registry and auto-detection."""

from __future__ import annotations

import pytest

from changemaster.core.exceptions import SensorProfileError
from changemaster.sensors.profiles import (
    BandDefinition,
    SensorProfile,
    SensorRegistry,
    sensor_registry,
)

EXPECTED_SENSORS = {
    "sentinel1",
    "sentinel2",
    "landsat5",
    "landsat7",
    "landsat8",
    "landsat9",
    "modis",
    "worldview",
    "pleiades",
    "spot",
    "planetscope",
    "generic",
}


class TestBuiltinProfiles:
    def test_all_twelve_registered(self) -> None:
        ids = {p.sensor_id for p in sensor_registry.profiles}
        assert ids == EXPECTED_SENSORS

    def test_profiles_have_bilingual_names_and_bands(self) -> None:
        for profile in sensor_registry.profiles:
            assert profile.display_name
            assert profile.display_name_ar
            assert profile.bands
            assert profile.default_resolution_m > 0

    def test_sentinel2_band_lookup(self) -> None:
        s2 = sensor_registry.get("sentinel2")
        red = s2.band("b04")
        assert red.name == "Red"
        assert red.wavelength_nm == 665.0
        assert s2.rgb_bands == ("B04", "B03", "B02")

    def test_sar_bands_have_no_wavelength(self) -> None:
        s1 = sensor_registry.get("sentinel1")
        assert s1.sensor_type == "sar"
        assert all(b.wavelength_nm is None for b in s1.bands)

    def test_unknown_band_raises(self) -> None:
        with pytest.raises(SensorProfileError):
            sensor_registry.get("sentinel2").band("B99")

    def test_unknown_sensor_raises(self) -> None:
        with pytest.raises(SensorProfileError):
            sensor_registry.get("voyager1")


class TestDetection:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("S2B_MSIL2A_20240115T083301.SAFE", "sentinel2"),
            ("S1A_IW_GRDH_1SDV_20240101.SAFE", "sentinel1"),
            ("LT05_L1TP_174038_19900715.tar", "landsat5"),
            ("LE07_L1TP_174038_20000715", "landsat7"),
            ("LC08_L1TP_174038_20240220.tar", "landsat8"),
            ("LC09_L1TP_174038_20240220", "landsat9"),
            ("MOD09GA.A2024001.h20v05.hdf", "modis"),
            ("WV03_20240101_PAN.tif", "worldview"),
            ("PHR1A_20240101_MS.jp2", "pleiades"),
            ("SPOT6_202401_ORT.tif", "spot"),
            ("20240101_psscene_analytic.tif", "planetscope"),
            ("holiday_photo.png", "generic"),
        ],
    )
    def test_filename_detection(self, name: str, expected: str) -> None:
        assert sensor_registry.detect(name).sensor_id == expected


class TestProfileValidation:
    def test_invalid_sensor_type_raises(self) -> None:
        with pytest.raises(SensorProfileError):
            SensorProfile(
                sensor_id="x",
                display_name="X",
                display_name_ar="س",
                platform="X",
                sensor_type="thermal",
                bands=(BandDefinition("B1", "b", None, 1.0),),
                default_resolution_m=1.0,
            )

    def test_invalid_resolution_raises(self) -> None:
        with pytest.raises(SensorProfileError):
            SensorProfile(
                sensor_id="x",
                display_name="X",
                display_name_ar="س",
                platform="X",
                sensor_type="optical",
                bands=(BandDefinition("B1", "b", None, 1.0),),
                default_resolution_m=0.0,
            )

    def test_duplicate_registration_raises(self) -> None:
        registry = SensorRegistry()
        profile = SensorProfile(
            sensor_id="dup",
            display_name="Dup",
            display_name_ar="مكرر",
            platform="X",
            sensor_type="optical",
            bands=(BandDefinition("B1", "b", None, 1.0),),
            default_resolution_m=1.0,
        )
        registry.register(profile)
        with pytest.raises(SensorProfileError):
            registry.register(profile)

    def test_to_dict_serialisable(self) -> None:
        import json

        json.dumps(sensor_registry.get("landsat8").to_dict())

    def test_detect_falls_back_to_generic(self) -> None:
        registry = SensorRegistry()
        registry.register(sensor_registry.get("generic"))
        assert registry.detect("anything.tif").sensor_id == "generic"
