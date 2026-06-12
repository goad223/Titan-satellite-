"""Reader for HDF5 files via h5py (lazy import).

Reads the first (or a named) 2-D/3-D dataset inside an HDF5 file and exposes
it through the unified reader interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from changemaster.core.exceptions import DependencyMissingError, ImageReadError
from changemaster.io_engine.base_reader import BaseImageReader, reader_registry
from changemaster.io_engine.metadata import GeoReference, ImageMetadata

if TYPE_CHECKING:
    import numpy as np


def _import_h5py() -> Any:
    """Import and return h5py, raising a bilingual error when missing."""
    try:
        import h5py
    except ImportError as exc:
        raise DependencyMissingError("h5py", "HDF5 reading", "قراءة HDF5") from exc
    return h5py


@reader_registry.register
class HDFReader(BaseImageReader):
    """h5py-backed reader exposing 2-D/3-D HDF5 datasets as rasters.

    Parameters
    ----------
    path:
        HDF5 file path.
    dataset:
        Optional internal dataset path (e.g. ``"group/band1"``). When omitted
        the first 2-D or 3-D dataset found (depth-first) is used.
    """

    format_name: ClassVar[str] = "HDF5"
    extensions: ClassVar[tuple[str, ...]] = (".h5", ".hdf", ".hdf5", ".he5")
    required_package: ClassVar[str | None] = "h5py"

    def __init__(self, path: Path | str, dataset: str | None = None) -> None:
        super().__init__(path)
        self.dataset_name: str | None = dataset
        self._file: Any = None
        self._dset: Any = None

    @staticmethod
    def list_datasets(path: Path | str) -> list[str]:
        """Return internal paths of all 2-D/3-D datasets in the file."""
        h5py = _import_h5py()
        names: list[str] = []
        with h5py.File(Path(path), "r") as f:

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                    names.append(name)

            f.visititems(visit)
        return names

    def open(self) -> None:
        """Open the HDF5 file, locate the target dataset and build metadata."""
        h5py = _import_h5py()
        try:
            self._file = h5py.File(self.path, "r")
        except OSError as exc:
            raise ImageReadError(
                f"Failed to open HDF5 file {self.path}: {exc}",
                f"فشل في فتح ملف HDF5 {self.path}: {exc}",
            ) from exc

        if self.dataset_name is not None:
            if self.dataset_name not in self._file:
                self.close()
                raise ImageReadError(
                    f"Dataset '{self.dataset_name}' not found in {self.path}.",
                    f"مجموعة البيانات '{self.dataset_name}' غير موجودة في {self.path}.",
                )
            self._dset = self._file[self.dataset_name]
        else:
            found: list[Any] = []

            def visit(name: str, obj: Any) -> None:
                if not found and isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                    found.append(obj)

            self._file.visititems(visit)
            if not found:
                self.close()
                raise ImageReadError(
                    f"No 2-D/3-D dataset found in HDF5 file {self.path}.",
                    f"لا توجد مجموعة بيانات ثنائية/ثلاثية الأبعاد في ملف HDF5 ‏{self.path}.",
                )
            self._dset = found[0]
            self.dataset_name = self._dset.name.lstrip("/")

        shape = self._dset.shape
        if len(shape) == 2:
            band_count, height, width = 1, shape[0], shape[1]
        else:
            band_count, height, width = shape[0], shape[1], shape[2]
        attrs = {k: _safe_attr(v) for k, v in self._dset.attrs.items()}
        self._metadata = ImageMetadata(
            path=self.path,
            driver="HDF5",
            width=width,
            height=height,
            band_count=band_count,
            dtype=str(self._dset.dtype),
            georef=GeoReference(),
            extra={"dataset": self.dataset_name, **attrs},
        )

    def close(self) -> None:
        """Close the HDF5 file (idempotent)."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._dset = None

    def read(
        self,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Read pixels as ``(bands, height, width)``; see base class."""
        import numpy as np

        if self._dset is None:
            self.open()
        assert self._dset is not None and self._metadata is not None
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

        if self._dset.ndim == 2:
            data = np.asarray(self._dset[rows, cols])[np.newaxis, :, :]
        else:
            band_idx = [b - 1 for b in indexes]
            data = np.stack([np.asarray(self._dset[i, rows, cols]) for i in band_idx])
            return data
        return data


def _safe_attr(value: Any) -> Any:
    """Convert an HDF5 attribute value to a JSON-friendly Python object."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
