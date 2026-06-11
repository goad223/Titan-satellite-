"""Built-in sensor profile definitions (12 sensors).

Importing this package registers every profile with
:data:`changemaster.sensors.profiles.sensor_registry`.

تعريفات بروفايلات المستشعرات المدمجة (12 مستشعراً).
"""

from changemaster.sensors.definitions import (  # noqa: F401
    generic,
    landsat5,
    landsat7,
    landsat8,
    landsat9,
    modis,
    planetscope,
    pleiades,
    sentinel1,
    sentinel2,
    spot,
    worldview,
)

__all__ = [
    "generic",
    "landsat5",
    "landsat7",
    "landsat8",
    "landsat9",
    "modis",
    "planetscope",
    "pleiades",
    "sentinel1",
    "sentinel2",
    "spot",
    "worldview",
]
