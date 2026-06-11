"""Abstract reader interface plus the reader registry/factory.

Each concrete reader registers itself with :data:`reader_registry`. Readers
whose heavy dependency (rasterio, h5py, netCDF4) is missing remain
registered but report ``is_available() == False`` so the application can
list them as installable extras without ever crashing at import time.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from changemaster.core.exceptions import FormatNotSupportedError
from changemaster.io_engine.metadata import ImageMetadata

if TYPE_CHECKING:
    import numpy as np


class BaseImageReader(abc.ABC):
    """Abstract base class for all image readers.

    Concrete subclasses must define the class attributes
    ``format_name``, ``extensions`` and ``required_package`` and implement
    the abstract methods. Use a reader as a context manager::

        with SomeReader(path) as reader:
            meta = reader.metadata
            data = reader.read()
    """

    #: Human readable format name, e.g. "GeoTIFF".
    format_name: ClassVar[str] = ""
    #: Lower-case extensions handled, e.g. (".tif", ".tiff").
    extensions: ClassVar[tuple[str, ...]] = ()
    #: PyPI package needed, or None when the reader has no heavy dependency.
    required_package: ClassVar[str | None] = None

    def __init__(self, path: Path | str) -> None:
        self.path: Path = Path(path)
        self._metadata: ImageMetadata | None = None

    # -- availability / matching ------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True when the reader's dependency is importable."""
        if cls.required_package is None:
            return True
        import importlib.util

        return importlib.util.find_spec(cls.required_package) is not None

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Cheap check (extension/structure) whether this reader fits ``path``."""
        return path.suffix.lower() in cls.extensions

    # -- lifecycle ----------------------------------------------------------------

    @abc.abstractmethod
    def open(self) -> None:
        """Open the underlying source and populate :attr:`metadata`."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release any underlying resources (idempotent)."""

    def __enter__(self) -> "BaseImageReader":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- data access ----------------------------------------------------------------

    @property
    def metadata(self) -> ImageMetadata:
        """Normalized metadata; opens the source lazily when needed."""
        if self._metadata is None:
            self.open()
        assert self._metadata is not None
        return self._metadata

    @abc.abstractmethod
    def read(
        self,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Read pixel data as a ``(bands, height, width)`` numpy array.

        Parameters
        ----------
        bands:
            1-based band indices to read; ``None`` reads all bands.
        window:
            ``(row_off, col_off, height, width)`` pixel window;
            ``None`` reads the full raster.
        """


class ReaderRegistry:
    """Registry and factory for :class:`BaseImageReader` subclasses."""

    def __init__(self) -> None:
        self._readers: list[type[BaseImageReader]] = []

    def register(self, reader_cls: type[BaseImageReader]) -> type[BaseImageReader]:
        """Register a reader class (usable as a class decorator)."""
        if reader_cls not in self._readers:
            self._readers.append(reader_cls)
        return reader_cls

    @property
    def readers(self) -> tuple[type[BaseImageReader], ...]:
        """All registered reader classes (in registration order)."""
        return tuple(self._readers)

    def available_readers(self) -> tuple[type[BaseImageReader], ...]:
        """Reader classes whose dependencies are currently installed."""
        return tuple(r for r in self._readers if r.is_available())

    def find_reader(self, path: Path | str) -> type[BaseImageReader]:
        """Return the first available reader class that can read ``path``.

        Raises
        ------
        FormatNotSupportedError
            When no available reader matches the path.
        """
        p = Path(path)
        for reader_cls in self._readers:
            if reader_cls.is_available() and reader_cls.can_read(p):
                return reader_cls
        raise FormatNotSupportedError(str(p))

    def format_report(self) -> list[dict[str, object]]:
        """Describe every registered format and its availability.

        Returns a list of dictionaries with keys ``format``, ``extensions``,
        ``available`` and ``requires``.
        """
        return [
            {
                "format": r.format_name,
                "extensions": list(r.extensions),
                "available": r.is_available(),
                "requires": r.required_package,
            }
            for r in self._readers
        ]


#: Global registry used by the application.
reader_registry = ReaderRegistry()


def open_image(path: Path | str) -> BaseImageReader:
    """Open ``path`` with the best matching available reader.

    Returns an *opened* reader instance; the caller is responsible for
    closing it (or using it as a context manager).
    """
    _ensure_builtin_readers_registered()
    reader_cls = reader_registry.find_reader(path)
    reader = reader_cls(path)
    reader.open()
    return reader


_builtins_registered = False


def _ensure_builtin_readers_registered() -> None:
    """Import all built-in reader modules so they self-register (once)."""
    global _builtins_registered
    if _builtins_registered:
        return
    # Importing the modules triggers their @reader_registry.register decorators.
    from changemaster.io_engine import (  # noqa: F401
        hdf_reader,
        landsat_reader,
        netcdf_reader,
        raster_reader,
        safe_reader,
        simple_reader,
    )

    _builtins_registered = True
