"""Sensor profiles: satellite/sensor definitions and auto-detection.

بروفايلات المستشعرات: تعريفات الأقمار الصناعية والكشف التلقائي.
"""

from changemaster.sensors.profiles import (
    BandDefinition,
    SensorProfile,
    SensorRegistry,
    sensor_registry,
)

__all__ = ["BandDefinition", "SensorProfile", "SensorRegistry", "sensor_registry"]
