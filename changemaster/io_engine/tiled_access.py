"""Tiled access to huge rasters (up to 100,000 x 100,000 pixels and beyond).

:class:`TiledImageAccessor` iterates over a raster in fixed-size tiles via
any :class:`~changemaster.io_engine.base_reader.BaseImageReader`, so memory
use stays bounded regardless of image size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from changemaster.core.exceptions import ImageReadError
from changemaster.io_engine.base_reader import BaseImageReader

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Tile:
    """A rectangular tile window inside a raster.

    Attributes
    ----------
    index:
        Sequential tile number (row-major order, starting at 0).
    row_off / col_off:
        Pixel offsets of the tile's top-left corner.
    height / width:
        Tile dimensions in pixels (edge tiles may be smaller).
    """

    index: int
    row_off: int
    col_off: int
    height: int
    width: int

    @property
    def window(self) -> tuple[int, int, int, int]:
        """Window tuple ``(row_off, col_off, height, width)`` for readers."""
        return (self.row_off, self.col_off, self.height, self.width)


def compute_tiles(
    height: int, width: int, tile_size: int, overlap: int = 0
) -> list[Tile]:
    """Compute the tile grid for a raster of ``height`` x ``width`` pixels.

    Parameters
    ----------
    height, width:
        Raster dimensions in pixels (must be positive).
    tile_size:
        Nominal tile edge length in pixels (must be positive).
    overlap:
        Pixels of overlap added on every side of each tile (clamped to the
        raster bounds). Must satisfy ``0 <= overlap < tile_size``.

    Returns
    -------
    list[Tile]
        Tiles in row-major order covering the whole raster.
    """
    if height <= 0 or width <= 0:
        raise ImageReadError(
            f"Invalid raster size {height}x{width} for tiling.",
            f"حجم نقطي غير صالح {height}x{width} للتجزئة.",
        )
    if tile_size <= 0:
        raise ImageReadError(
            f"Tile size must be positive, got {tile_size}.",
            f"يجب أن يكون حجم البلاطة موجباً، وجد {tile_size}.",
        )
    if overlap < 0 or overlap >= tile_size:
        raise ImageReadError(
            f"Overlap must satisfy 0 <= overlap < tile_size, got {overlap}.",
            f"يجب أن يحقق التداخل 0 <= overlap < tile_size، وجد {overlap}.",
        )

    tiles: list[Tile] = []
    n_rows = math.ceil(height / tile_size)
    n_cols = math.ceil(width / tile_size)
    index = 0
    for tr in range(n_rows):
        for tc in range(n_cols):
            row_start = max(0, tr * tile_size - overlap)
            col_start = max(0, tc * tile_size - overlap)
            row_end = min(height, (tr + 1) * tile_size + overlap)
            col_end = min(width, (tc + 1) * tile_size + overlap)
            tiles.append(
                Tile(
                    index=index,
                    row_off=row_start,
                    col_off=col_start,
                    height=row_end - row_start,
                    width=col_end - col_start,
                )
            )
            index += 1
    return tiles


class TiledImageAccessor:
    """Iterate over a raster in memory-bounded tiles.

    Parameters
    ----------
    reader:
        An opened :class:`BaseImageReader`.
    tile_size:
        Tile edge length in pixels.
    overlap:
        Overlap in pixels added around each tile (useful for window-based
        filters in later phases).
    """

    def __init__(
        self,
        reader: BaseImageReader,
        tile_size: int = 1024,
        overlap: int = 0,
    ) -> None:
        self.reader: BaseImageReader = reader
        meta = reader.metadata
        self.tiles: list[Tile] = compute_tiles(meta.height, meta.width, tile_size, overlap)
        self.tile_size: int = tile_size
        self.overlap: int = overlap

    def __len__(self) -> int:
        return len(self.tiles)

    def read_tile(self, tile: Tile, bands: list[int] | None = None) -> "np.ndarray":
        """Read pixel data for a single :class:`Tile`."""
        return self.reader.read(bands=bands, window=tile.window)

    def iter_tiles(
        self, bands: list[int] | None = None
    ) -> Iterator[tuple[Tile, "np.ndarray"]]:
        """Yield ``(tile, data)`` pairs covering the full raster."""
        for tile in self.tiles:
            yield tile, self.read_tile(tile, bands=bands)
