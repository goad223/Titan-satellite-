"""Reader for NetCDF files via netCDF4 (lazy import).

Reads the first (or a named) 2-D/3-D variable inside a NetCDF file and
exposes it through the unified reader interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from changemaster.core.exceptions import DependencyMissingError, ImageReadError
from changemaster.io_engine.base_reader import BaseImageReader, reader_registry
from changemaster.io_engine.metadata import GeoReference, ImageMetadata

if TYPE_CHECKING:
    import numpy as np


def _import_netcdf4() -> Any:
    """Import and return netCDF4, raising a bilingual error when missing."""
    try:
        import netCDF4
    except ImportError as exc:
        raise DependencyMissingError("netCDF4", "NetCDF reading", "قراءة NetCDF") from exc
    return netCDF4


@reader_registry.register
class NetCDFReader(BaseImageReader):
    """netCDF4-backed reader exposing 2-D/3-D variables as rasters.

    Parameters
    ----------
    path:
        NetCDF file path.
    variable:
        Optional variable name. When omitted the first 2-D or 3-D variable
        is used.
    """

    format_name: ClassVar[str] = "NetCDF"
    extensions: ClassVar[tuple[str, ...]] = (".nc", ".nc4", ".cdf")
    required_package: ClassVar[str | None] = "netCDF4"

    def __init__(self, path: Path | str, variable: str | None = None) -> None:
        super().__init__(path)
        self.variable_name: str | None = variable
        self._file: Any = None
        self._var: Any = None

    @staticmethod
    def list_variables(path: Path | str) -> list[str]:
        """Return names of all 2-D/3-D variables in the file."""
        netCDF4 = _import_netcdf4()
        with netCDF4.Dataset(str(Path(path)), "r") as f:
            return [name for name, var in f.variables.items() if var.ndim in (2, 3)]

    def open(self) -> None:
        """Open the NetCDF file, locate the target variable, build metadata."""
        netCDF4 = _import_netcdf4()
        try:
            self._file = netCDF4.Dataset(str(self.path), "r")
        except OSError as exc:
            raise ImageReadError(
                f"Failed to open NetCDF file {self.path}: {exc}",
                f"فشل في فتح ملف NetCDF ‏{self.path}: {exc}",
            ) from exc

        if self.variable_name is not None:
            if self.variable_name not in self._file.variables:
                self.close()
                raise ImageReadError(
                    f"Variable '{self.variable_name}' not found in {self.path}.",
                    f"المتغير '{self.variable_name}' غير موجود في {self.path}.",
                )
            self._var = self._file.variables[self.variable_name]
        else:
            for name, var in self._file.variables.items():
                if var.ndim in (2, 3):
                    self._var = var
                    self.variable_name = name
                    break
            if self._var is None:
                self.close()
                raise ImageReadError(
                    f"No 2-D/3-D variable found in NetCDF file {self.path}.",
                    f"لا يوجد متغير ثنائي/ثلاثي الأبعاد في ملف NetCDF ‏{self.path}.",
                )

        shape = self._var.shape
        if len(shape) == 2:
            band_count, height, width = 1, shape[0], shape[1]
        else:
            band_count, height, width = shape[0], shape[1], shape[2]
        attrs = {k: _safe_attr(self._var.getncattr(k)) for k in self._var.ncattrs()}
        self._metadata = ImageMetadata(
            path=self.path,
            driver="NetCDF",
            width=width,
            height=height,
            band_count=band_count,
            dtype=str(self._var.dtype),
            georef=GeoReference(),
            extra={"variable": self.variable_name, **attrs},
        )

    def close(self) -> None:
        """Close the NetCDF file (idempotent)."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._var = None

    def read(
        self,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Read pixels as ``(bands, height, width)``; see base class."""
        import numpy as np

        if self._var is None:
            self.open()
        assert self._var is not None and self._metadata is not None
        count = self._metadata.band_count
        indexes = bands if bands is not None else list(range(1, count + 1))
        for b in indexes:
            if b < 1 or b > count:
                raise ImageReadError(
                    f"Band index {b} out of range 1..{count} for {self.path}.",
                    f"رقم النطاق {b} خارج المدى 1..{count} للملف {self.path}.",
                )
        if window is not None:
            row_off, col_off, height, width = window
            rows = slice(row_off, row_off + height)
            cols = slice(col_off, col_off + width)
        else:
            rows = slice(None)
            cols = slice(None)

        if self._var.ndim == 2:
            return np.ma.filled(np.ma.asanyarray(self._var[rows, cols]))[np.newaxis, :, :]
        band_idx = [b - 1 for b in indexes]
        return np.stack(
            [np.ma.filled(np.ma.asanyarray(self._var[i, rows, cols])) for i in band_idx]
        )


def _safe_attr(value: Any) -> Any:
    """Convert a NetCDF attribute value to a JSON-friendly Python object."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
