"""Reader for Landsat products: MTL.txt metadata plus band files.

Supports both an extracted product directory and a ``.tar`` archive (the
archive is indexed and band files are extracted on demand to a temporary
directory). Band files are read via rasterio.
"""

from __future__ import annotations

import re
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from changemaster.core.exceptions import ImageReadError, MetadataError
from changemaster.io_engine.base_reader import BaseImageReader, reader_registry
from changemaster.io_engine.metadata import ImageMetadata
from changemaster.io_engine.raster_reader import RasterReader, _import_rasterio

if TYPE_CHECKING:
    import numpy as np

_BAND_FILE_RE = re.compile(r"_(B\d{1,2})\.TIF$", re.IGNORECASE)
_MTL_VALUE_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*$")

_SPACECRAFT_TO_SENSOR = {
    "LANDSAT_5": "landsat5",
    "LANDSAT_7": "landsat7",
    "LANDSAT_8": "landsat8",
    "LANDSAT_9": "landsat9",
}


def parse_mtl(text: str) -> dict[str, str]:
    """Parse Landsat MTL key/value text into a flat dictionary.

    Group markers (``GROUP``/``END_GROUP``/``END``) are skipped and quoted
    values are unquoted.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _MTL_VALUE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if key in ("GROUP", "END_GROUP") or key == "END":
            continue
        values[key] = value.strip().strip('"')
    return values


def _acquisition_datetime(mtl: dict[str, str]) -> datetime | None:
    """Combine DATE_ACQUIRED and SCENE_CENTER_TIME MTL fields when present."""
    date_str = mtl.get("DATE_ACQUIRED")
    if not date_str:
        return None
    time_str = mtl.get("SCENE_CENTER_TIME", "00:00:00").strip('"').split(".")[0].rstrip("Z")
    try:
        return datetime.fromisoformat(f"{date_str}T{time_str}")
    except ValueError:
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            return None


@reader_registry.register
class LandsatReader(BaseImageReader):
    """Reader for Landsat scenes (directory or ``.tar`` archive).

    Parameters
    ----------
    path:
        Scene directory containing ``*_MTL.txt`` and ``*_B*.TIF`` files, or a
        ``.tar`` archive of such a scene.
    band:
        Optional band identifier (e.g. ``"B4"``). Defaults to the first
        discovered band.
    """

    format_name: ClassVar[str] = "Landsat"
    extensions: ClassVar[tuple[str, ...]] = (".tar",)
    required_package: ClassVar[str | None] = "rasterio"

    def __init__(self, path: Path | str, band: str | None = None) -> None:
        super().__init__(path)
        self.band: str | None = band.upper() if band else None
        self.mtl: dict[str, str] = {}
        self._band_files: dict[str, Path] = {}
        self._reader: RasterReader | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Match Landsat scene directories or ``.tar`` archives."""
        if path.is_dir():
            return any(path.glob("*_MTL.txt")) or any(path.glob("*_MTL.TXT"))
        return path.suffix.lower() == ".tar"

    def _load_from_directory(self, directory: Path) -> None:
        """Index the MTL file and band files inside ``directory``."""
        mtl_files = sorted(directory.glob("*_MTL.txt")) + sorted(directory.glob("*_MTL.TXT"))
        if not mtl_files:
            raise MetadataError(
                f"No MTL metadata file found in Landsat scene {self.path}.",
                f"لم يتم العثور على ملف MTL في مشهد لاندسات {self.path}.",
            )
        self.mtl = parse_mtl(mtl_files[0].read_text(encoding="utf-8", errors="replace"))
        for tif in sorted(directory.glob("*.TIF")) + sorted(directory.glob("*.tif")):
            match = _BAND_FILE_RE.search(tif.name)
            if match:
                self._band_files.setdefault(match.group(1).upper(), tif)

    def _extract_tar(self) -> Path:
        """Safely extract the scene tar archive to a temporary directory."""
        self._tmpdir = tempfile.TemporaryDirectory(prefix="changemaster_landsat_")
        target = Path(self._tmpdir.name)
        try:
            with tarfile.open(self.path) as tar:
                for member in tar.getmembers():
                    member_path = (target / member.name).resolve()
                    if not str(member_path).startswith(str(target.resolve())):
                        raise ImageReadError(
                            f"Unsafe path in tar archive: {member.name}",
                            f"مسار غير آمن داخل أرشيف tar: {member.name}",
                        )
                tar.extractall(target, filter="data")
        except (tarfile.TarError, OSError) as exc:
            raise ImageReadError(
                f"Failed to extract Landsat archive {self.path}: {exc}",
                f"فشل في فك أرشيف لاندسات {self.path}: {exc}",
            ) from exc
        return target

    def open(self) -> None:
        """Index the scene, open the selected band and build metadata."""
        _import_rasterio()  # fail early with a bilingual message
        if self.path.is_dir():
            self._load_from_directory(self.path)
        elif self.path.suffix.lower() == ".tar" and self.path.exists():
            self._load_from_directory(self._extract_tar())
        else:
            raise ImageReadError(
                f"Landsat scene not found: {self.path}",
                f"مشهد لاندسات غير موجود: {self.path}",
            )
        if not self._band_files:
            raise ImageReadError(
                f"No band files found in Landsat scene {self.path}.",
                f"لم يتم العثور على ملفات نطاقات في مشهد لاندسات {self.path}.",
            )
        band = self.band if self.band is not None else sorted(self._band_files)[0]
        if band not in self._band_files:
            raise ImageReadError(
                f"Band '{band}' not found in Landsat scene {self.path}. "
                f"Available: {sorted(self._band_files)}",
                f"النطاق '{band}' غير موجود في مشهد لاندسات {self.path}. "
                f"المتاح: {sorted(self._band_files)}",
            )
        self.band = band
        self._reader = RasterReader(self._band_files[band])
        self._reader.open()
        inner = self._reader.metadata

        spacecraft = self.mtl.get("SPACECRAFT_ID", "")
        self._metadata = ImageMetadata(
            path=self.path,
            driver="Landsat",
            width=inner.width,
            height=inner.height,
            band_count=inner.band_count,
            dtype=inner.dtype,
            nodata=inner.nodata,
            georef=inner.georef,
            acquisition_datetime=_acquisition_datetime(self.mtl),
            sensor_id=_SPACECRAFT_TO_SENSOR.get(spacecraft),
            band_names=[band],
            extra={
                "mtl": dict(self.mtl),
                "available_bands": sorted(self._band_files),
                "band_file": str(self._band_files[band]),
            },
        )

    def close(self) -> None:
        """Close the band reader and clean up any temp extraction dir."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def read(
        self,
        bands: list[int] | None = None,
        window: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Read the selected band's pixels; see base class."""
        if self._reader is None:
            self.open()
        assert self._reader is not None
        return self._reader.read(bands=bands, window=window)
