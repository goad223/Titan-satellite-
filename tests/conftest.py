"""Shared pytest fixtures: all test data is generated programmatically.

No large binary files are stored in the repository — every fixture builds
its data on the fly inside ``tmp_path`` directories.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def rgb_array() -> np.ndarray:
    """Deterministic 3-band uint8 array shaped (3, 32, 48)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(3, 32, 48), dtype=np.uint8)


@pytest.fixture()
def gray_array() -> np.ndarray:
    """Deterministic single-band uint16 array shaped (1, 40, 30)."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 4096, size=(1, 40, 30), dtype=np.uint16)


@pytest.fixture()
def png_file(tmp_path: Path, rgb_array: np.ndarray) -> Path:
    """A small RGB PNG file written with Pillow."""
    from PIL import Image

    path = tmp_path / "sample.png"
    Image.fromarray(np.transpose(rgb_array, (1, 2, 0))).save(path)
    return path


@pytest.fixture()
def jpeg_file(tmp_path: Path, rgb_array: np.ndarray) -> Path:
    """A small RGB JPEG file written with Pillow."""
    from PIL import Image

    path = tmp_path / "sample.jpg"
    Image.fromarray(np.transpose(rgb_array, (1, 2, 0))).save(path, quality=95)
    return path


@pytest.fixture()
def geotiff_file(tmp_path: Path, gray_array: np.ndarray) -> Path:
    """A georeferenced single-band GeoTIFF written with rasterio."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "sample.tif"
    transform = from_origin(500000.0, 4100000.0, 10.0, 10.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=gray_array.shape[2],
        height=gray_array.shape[1],
        count=1,
        dtype=str(gray_array.dtype),
        crs="EPSG:32636",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(gray_array)
    return path


@pytest.fixture()
def hdf5_file(tmp_path: Path) -> Path:
    """An HDF5 file with one 2-D and one 3-D dataset."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(3)
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("scalar", data=1.5)
        grp = f.create_group("data")
        d2 = grp.create_dataset("band2d", data=rng.integers(0, 100, size=(20, 25), dtype=np.int32))
        d2.attrs["units"] = "DN"
        grp.create_dataset(
            "cube3d", data=rng.integers(0, 100, size=(3, 20, 25), dtype=np.int32)
        )
    return path


@pytest.fixture()
def netcdf_file(tmp_path: Path) -> Path:
    """A NetCDF file with one 2-D and one 3-D variable."""
    netCDF4 = pytest.importorskip("netCDF4")
    rng = np.random.default_rng(5)
    path = tmp_path / "sample.nc"
    with netCDF4.Dataset(str(path), "w") as f:
        f.createDimension("band", 2)
        f.createDimension("y", 15)
        f.createDimension("x", 18)
        v2 = f.createVariable("temp", "f4", ("y", "x"))
        v2[:] = rng.random((15, 18)).astype(np.float32)
        v2.units = "K"
        v3 = f.createVariable("refl", "f4", ("band", "y", "x"))
        v3[:] = rng.random((2, 15, 18)).astype(np.float32)
    return path


def _write_band_jp2_or_tif(directory: Path, name: str, seed: int) -> Path:
    """Write a tiny georeferenced GeoTIFF band file (JP2 writing needs extra drivers)."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(seed)
    data = rng.integers(0, 10000, size=(1, 12, 14), dtype=np.uint16)
    path = directory / name
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=14,
        height=12,
        count=1,
        dtype="uint16",
        crs="EPSG:32636",
        transform=from_origin(300000.0, 3900000.0, 10.0, 10.0),
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture()
def safe_product(tmp_path: Path) -> Path:
    """A minimal synthetic Sentinel-2 SAFE product directory."""
    pytest.importorskip("rasterio")
    product = tmp_path / "S2A_MSIL2A_20240115T083301_N0510_R021_T36RUU_20240115T120000.SAFE"
    granule = product / "GRANULE" / "L2A_T36RUU" / "IMG_DATA"
    granule.mkdir(parents=True)
    b02 = _write_band_jp2_or_tif(granule, "T36RUU_20240115T083301_B02.tif", 11)
    b03 = _write_band_jp2_or_tif(granule, "T36RUU_20240115T083301_B03.tif", 12)
    rel02 = b02.relative_to(product).as_posix()
    rel03 = b03.relative_to(product).as_posix()
    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"
           xmlns:safe="http://www.esa.int/safe/sentinel/1.1">
  <metadataSection>
    <metadataObject ID="acquisitionPeriod">
      <metadataWrap><xmlData>
        <safe:acquisitionPeriod>
          <safe:startTime>2024-01-15T08:33:01.024Z</safe:startTime>
        </safe:acquisitionPeriod>
      </xmlData></metadataWrap>
    </metadataObject>
    <metadataObject ID="platform">
      <metadataWrap><xmlData>
        <safe:platform>
          <safe:familyName>SENTINEL</safe:familyName>
          <safe:number>2A</safe:number>
        </safe:platform>
      </xmlData></metadataWrap>
    </metadataObject>
  </metadataSection>
  <dataObjectSection>
    <dataObject ID="IMG_DATA_Band_B02">
      <byteStream><fileLocation locatorType="URL" href="./{rel02}"/></byteStream>
    </dataObject>
    <dataObject ID="IMG_DATA_Band_B03">
      <byteStream><fileLocation locatorType="URL" href="./{rel03}"/></byteStream>
    </dataObject>
  </dataObjectSection>
</xfdu:XFDU>
"""
    (product / "manifest.safe").write_text(manifest, encoding="utf-8")
    return product


_MTL_TEXT = """GROUP = LANDSAT_METADATA_FILE
  GROUP = IMAGE_ATTRIBUTES
    SPACECRAFT_ID = "LANDSAT_8"
    DATE_ACQUIRED = 2024-02-20
    SCENE_CENTER_TIME = "08:15:32.0250000Z"
    CLOUD_COVER = 3.12
  END_GROUP = IMAGE_ATTRIBUTES
END_GROUP = LANDSAT_METADATA_FILE
END
"""


@pytest.fixture()
def landsat_scene(tmp_path: Path) -> Path:
    """A minimal synthetic Landsat 8 scene directory (MTL + 2 bands)."""
    pytest.importorskip("rasterio")
    scene = tmp_path / "LC08_L1TP_174038_20240220_20240228_02_T1"
    scene.mkdir()
    (scene / "LC08_L1TP_174038_20240220_20240228_02_T1_MTL.txt").write_text(
        _MTL_TEXT, encoding="utf-8"
    )
    _write_band_jp2_or_tif(scene, "LC08_L1TP_174038_20240220_20240228_02_T1_B4.TIF", 21)
    _write_band_jp2_or_tif(scene, "LC08_L1TP_174038_20240220_20240228_02_T1_B5.TIF", 22)
    return scene


@pytest.fixture()
def landsat_tar(tmp_path: Path, landsat_scene: Path) -> Path:
    """The synthetic Landsat scene packed into a ``.tar`` archive."""
    tar_path = tmp_path / "LC08_L1TP_174038_20240220_20240228_02_T1.tar"
    with tarfile.open(tar_path, "w") as tar:
        for item in sorted(landsat_scene.iterdir()):
            tar.add(item, arcname=item.name)
    return tar_path


# ---------------------------------------------------------------------------
# Phase 2 (preprocessing) fixtures — all data generated programmatically.
# ---------------------------------------------------------------------------


@pytest.fixture()
def textured_pair() -> tuple[np.ndarray, np.ndarray]:
    """Structured reference band and a (dx=5.3, dy=-3.7) shifted copy.

    The scene mixes sinusoidal terrain, Gaussian blobs and fine texture so
    feature detectors, phase correlation and ECC all behave as they do on
    real imagery.
    """
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:300, 0:300].astype(np.float32)
    scene = 50 * np.sin(x / 23) + 40 * np.cos(y / 17) + 30 * np.sin((x + y) / 31)
    for _ in range(40):
        cx, cy = rng.uniform(20, 280, size=2)
        radius, amplitude = rng.uniform(5, 25), rng.uniform(30, 90)
        scene += amplitude * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2)))
    scene += cv2.GaussianBlur(rng.random((300, 300)).astype(np.float32), (0, 0), 1.5) * 25
    base = scene.astype(np.float32)
    matrix = np.array([[1, 0, 5.3], [0, 1, -3.7]], dtype=np.float32)
    moving = cv2.warpAffine(base, matrix, (300, 300))
    return base, moving


@pytest.fixture()
def multiband_pair() -> tuple[np.ndarray, np.ndarray]:
    """Co-registered 3-band pair with a linear radiometric distortion + change."""
    rng = np.random.default_rng(1)
    ref = rng.normal(100.0, 20.0, size=(3, 90, 100))
    mov = ref * 1.3 + 15.0 + rng.normal(0.0, 2.0, size=(3, 90, 100))
    mov[:, 20:40, 30:50] += 80.0  # a changed region
    return ref, mov


@pytest.fixture()
def optical_pair_png(tmp_path: Path) -> tuple[Path, Path]:
    """A reference/moving PNG pair with a known sub-pixel shift."""
    cv2 = pytest.importorskip("cv2")
    from PIL import Image

    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(rng.random((200, 200)).astype(np.float32), (0, 0), 2)
    base = (base - base.min()) / (base.max() - base.min())
    ref = np.stack([base * 200, base * 180, base * 220]).astype(np.uint8)
    matrix = np.array([[1, 0, 4.0], [0, 1, -2.5]], dtype=np.float32)
    mov = np.stack(
        [cv2.warpAffine(b.astype(np.float32), matrix, (200, 200)) for b in ref]
    ).astype(np.uint8)
    ref_path = tmp_path / "ref.png"
    mov_path = tmp_path / "mov.png"
    Image.fromarray(np.transpose(ref, (1, 2, 0))).save(ref_path)
    Image.fromarray(np.transpose(mov, (1, 2, 0))).save(mov_path)
    return ref_path, mov_path


@pytest.fixture()
def speckled_image() -> np.ndarray:
    """A gamma-distributed single-look speckle field over a two-level scene."""
    rng = np.random.default_rng(7)
    scene = np.full((64, 64), 0.1)
    scene[:, 32:] = 0.5  # an edge between two homogeneous regions
    return scene * rng.gamma(1.0, 1.0, size=(64, 64))
