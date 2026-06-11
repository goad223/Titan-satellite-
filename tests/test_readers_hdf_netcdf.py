"""Tests for HDF5 and NetCDF readers (skipped when libraries missing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import ImageReadError


class TestHDFReader:
    @pytest.fixture(autouse=True)
    def _require(self) -> None:
        pytest.importorskip("h5py")

    def test_auto_selects_first_dataset(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file) as reader:
            meta = reader.metadata
            assert meta.driver == "HDF5"
            assert reader.dataset_name == "data/band2d"
            assert meta.band_count == 1
            assert (meta.height, meta.width) == (20, 25)
            assert meta.extra["units"] == "DN"
            data = reader.read()
            assert data.shape == (1, 20, 25)

    def test_named_3d_dataset(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file, dataset="data/cube3d") as reader:
            assert reader.metadata.band_count == 3
            full = reader.read()
            assert full.shape == (3, 20, 25)
            band2 = reader.read(bands=[2])
            np.testing.assert_array_equal(band2[0], full[1])
            win = reader.read(window=(2, 3, 5, 6))
            np.testing.assert_array_equal(win, full[:, 2:7, 3:9])

    def test_list_datasets(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        names = HDFReader.list_datasets(hdf5_file)
        assert set(names) == {"data/band2d", "data/cube3d"}

    def test_missing_dataset_raises(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with pytest.raises(ImageReadError):
            HDFReader(hdf5_file, dataset="nope").open()

    def test_band_out_of_range(self, hdf5_file: Path) -> None:
        from changemaster.io_engine.hdf_reader import HDFReader

        with HDFReader(hdf5_file) as reader:
            with pytest.raises(ImageReadError):
                reader.read(bands=[2])

    def test_no_image_dataset_raises(self, tmp_path: Path) -> None:
        import h5py

        from changemaster.io_engine.hdf_reader import HDFReader

        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("scalar", data=2.0)
        with pytest.raises(ImageReadError):
            HDFReader(path).open()


class TestNetCDFReader:
    @pytest.fixture(autouse=True)
    def _require(self) -> None:
        pytest.importorskip("netCDF4")

    def test_auto_selects_first_variable(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with NetCDFReader(netcdf_file) as reader:
            meta = reader.metadata
            assert meta.driver == "NetCDF"
            assert reader.variable_name == "temp"
            assert meta.band_count == 1
            assert (meta.height, meta.width) == (15, 18)
            assert meta.extra["units"] == "K"
            assert reader.read().shape == (1, 15, 18)

    def test_named_3d_variable(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with NetCDFReader(netcdf_file, variable="refl") as reader:
            assert reader.metadata.band_count == 2
            full = reader.read()
            assert full.shape == (2, 15, 18)
            band2 = reader.read(bands=[2])
            np.testing.assert_array_equal(band2[0], full[1])
            win = reader.read(window=(1, 2, 4, 5))
            np.testing.assert_array_equal(win, full[:, 1:5, 2:7])

    def test_list_variables(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        assert set(NetCDFReader.list_variables(netcdf_file)) == {"temp", "refl"}

    def test_missing_variable_raises(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with pytest.raises(ImageReadError):
            NetCDFReader(netcdf_file, variable="nope").open()

    def test_band_out_of_range(self, netcdf_file: Path) -> None:
        from changemaster.io_engine.netcdf_reader import NetCDFReader

        with NetCDFReader(netcdf_file) as reader:
            with pytest.raises(ImageReadError):
                reader.read(bands=[3])


class TestDependencyMessages:
    def test_missing_dependency_error_is_bilingual(self) -> None:
        from changemaster.core.exceptions import DependencyMissingError

        exc = DependencyMissingError("h5py", "HDF5 reading", "قراءة HDF5")
        assert "pip install h5py" in exc.message_en
        assert "h5py" in exc.message_ar
