"""Tests for tiled access to large rasters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from changemaster.core.exceptions import ImageReadError
from changemaster.io_engine.base_reader import open_image
from changemaster.io_engine.tiled_access import Tile, TiledImageAccessor, compute_tiles


class TestComputeTiles:
    def test_exact_grid(self) -> None:
        tiles = compute_tiles(100, 200, 50)
        assert len(tiles) == 2 * 4
        assert all(t.height == 50 and t.width == 50 for t in tiles)

    def test_edge_tiles_smaller(self) -> None:
        tiles = compute_tiles(105, 130, 50)
        assert len(tiles) == 3 * 3
        last = tiles[-1]
        assert last.height == 5 and last.width == 30

    def test_full_coverage_no_overlap(self) -> None:
        height, width = 73, 91
        cover = np.zeros((height, width), dtype=int)
        for t in compute_tiles(height, width, 32):
            cover[t.row_off : t.row_off + t.height, t.col_off : t.col_off + t.width] += 1
        assert (cover == 1).all()

    def test_overlap_expands_tiles(self) -> None:
        tiles = compute_tiles(100, 100, 50, overlap=10)
        inner = tiles[3]  # bottom-right tile of 2x2 grid
        assert inner.row_off == 40 and inner.col_off == 40
        assert tiles[0].row_off == 0 and tiles[0].col_off == 0
        assert tiles[0].height == 60 and tiles[0].width == 60

    def test_huge_virtual_raster_tile_count(self) -> None:
        tiles = compute_tiles(100_000, 100_000, 4096)
        assert len(tiles) == 25 * 25
        assert tiles[-1].row_off + tiles[-1].height == 100_000

    @pytest.mark.parametrize(
        ("height", "width", "tile_size", "overlap"),
        [(0, 10, 4, 0), (10, 10, 0, 0), (10, 10, 4, -1), (10, 10, 4, 4)],
    )
    def test_invalid_args_raise(self, height: int, width: int, tile_size: int, overlap: int) -> None:
        with pytest.raises(ImageReadError):
            compute_tiles(height, width, tile_size, overlap)

    def test_tile_window_property(self) -> None:
        tile = Tile(index=0, row_off=3, col_off=4, height=5, width=6)
        assert tile.window == (3, 4, 5, 6)


class TestTiledImageAccessor:
    def test_reassemble_png(self, png_file: Path, rgb_array: np.ndarray) -> None:
        with open_image(png_file) as reader:
            accessor = TiledImageAccessor(reader, tile_size=10)
            assert len(accessor) == 4 * 5  # 32x48 px in 10px tiles
            out = np.zeros_like(rgb_array)
            for tile, data in accessor.iter_tiles():
                out[:, tile.row_off : tile.row_off + tile.height,
                    tile.col_off : tile.col_off + tile.width] = data
            np.testing.assert_array_equal(out, rgb_array)

    def test_reassemble_geotiff(self, geotiff_file: Path, gray_array: np.ndarray) -> None:
        pytest.importorskip("rasterio")
        with open_image(geotiff_file) as reader:
            accessor = TiledImageAccessor(reader, tile_size=16)
            out = np.zeros_like(gray_array)
            for tile, data in accessor.iter_tiles():
                out[:, tile.row_off : tile.row_off + tile.height,
                    tile.col_off : tile.col_off + tile.width] = data
            np.testing.assert_array_equal(out, gray_array)

    def test_read_single_tile_with_bands(self, png_file: Path, rgb_array: np.ndarray) -> None:
        with open_image(png_file) as reader:
            accessor = TiledImageAccessor(reader, tile_size=10)
            tile = accessor.tiles[0]
            data = accessor.read_tile(tile, bands=[1])
            np.testing.assert_array_equal(data[0], rgb_array[0, :10, :10])
