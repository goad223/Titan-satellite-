"""Unified multi-format image I/O engine.

محرك القراءة/الكتابة الموحد لصيغ صور الأقمار الصناعية المتعددة.
"""

from changemaster.io_engine.base_reader import (
    BaseImageReader,
    ReaderRegistry,
    open_image,
    reader_registry,
)
from changemaster.io_engine.base_reader import (
    _ensure_builtin_readers_registered as _register_builtins,
)
from changemaster.io_engine.metadata import GeoReference, ImageMetadata

# Register all built-in readers on package import. The reader modules use
# lazy imports internally, so this never pulls in heavy dependencies.
_register_builtins()

__all__ = [
    "BaseImageReader",
    "GeoReference",
    "ImageMetadata",
    "ReaderRegistry",
    "open_image",
    "reader_registry",
]
