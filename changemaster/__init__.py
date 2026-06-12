"""ChangeMaster Ultimate — offline satellite image change detection suite.

Phase 1 foundation: hardware detection, configuration, logging, a unified
multi-format image I/O engine and a sensor profile registry.

برنامج ChangeMaster Ultimate — كشف التغيرات بين صور الأقمار الصناعية،
يعمل دون اتصال بالإنترنت. المرحلة الأولى: الأساس.
"""

from __future__ import annotations

__version__ = "0.1.0"
__app_name__ = "ChangeMaster Ultimate"

from changemaster.core.exceptions import (
    ChangeMasterError,
    ConfigError,
    DependencyMissingError,
    FormatNotSupportedError,
    HardwareDetectionError,
    ImageReadError,
    ImageWriteError,
    MetadataError,
    SensorProfileError,
)
from changemaster.io_engine.base_reader import open_image, reader_registry
from changemaster.io_engine.metadata import GeoReference, ImageMetadata
from changemaster.sensors.profiles import SensorProfile, sensor_registry

__all__ = [
    "__app_name__",
    "__version__",
    "ChangeMasterError",
    "ConfigError",
    "DependencyMissingError",
    "FormatNotSupportedError",
    "GeoReference",
    "HardwareDetectionError",
    "ImageMetadata",
    "ImageReadError",
    "ImageWriteError",
    "MetadataError",
    "SensorProfile",
    "SensorProfileError",
    "open_image",
    "reader_registry",
    "sensor_registry",
]
