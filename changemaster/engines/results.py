"""ChangeResult: the unified output object of every change-detection engine.

Holds the probability / binary / uncertainty / agreement maps plus
change-type hints, computes area statistics from the pixel size, and exports
results as GeoTIFF (Phase-1 writer), GeoJSON (written manually) and ESRI
Shapefile (via the optional ``pyshp`` dependency, lazily imported).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from changemaster.core.exceptions import DependencyMissingError, EngineError
from changemaster.io_engine.metadata import GeoReference

if TYPE_CHECKING:
    import numpy as np

#: Binary-map code: pixel evaluated, no change detected.
BINARY_NO_CHANGE = 0
#: Binary-map code: pixel evaluated, change detected.
BINARY_CHANGE = 1
#: Binary-map code: pixel masked (cloud/shadow/nodata) — not evaluated.
BINARY_UNEVALUATED = 255


@dataclass
class ChangeStatistics:
    """Aggregate statistics of a change map.

    Attributes
    ----------
    changed_pixels:
        Number of pixels labelled change.
    evaluated_pixels:
        Number of evaluated (non-masked) pixels.
    unevaluated_pixels:
        Number of masked / unevaluated pixels.
    change_fraction:
        ``changed / evaluated`` (0 when nothing was evaluated).
    change_area_m2 / change_area_ha:
        Changed area in square metres / hectares (``None`` when the pixel
        size is unknown).
    n_regions:
        Number of connected change regions (8-connectivity).
    """

    changed_pixels: int = 0
    evaluated_pixels: int = 0
    unevaluated_pixels: int = 0
    change_fraction: float = 0.0
    change_area_m2: float | None = None
    change_area_ha: float | None = None
    n_regions: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "changed_pixels": self.changed_pixels,
            "evaluated_pixels": self.evaluated_pixels,
            "unevaluated_pixels": self.unevaluated_pixels,
            "change_fraction": self.change_fraction,
            "change_area_m2": self.change_area_m2,
            "change_area_ha": self.change_area_ha,
            "n_regions": self.n_regions,
        }


@dataclass
class ChangePolygon:
    """One vectorized connected change region.

    Attributes
    ----------
    rings:
        Polygon rings in CRS coordinates: the first ring is the exterior
        boundary, the rest are interior holes. Each ring is a closed list of
        ``(x, y)`` points.
    area_m2:
        Region area in square metres (``None`` when pixel size unknown).
    mean_probability:
        Mean fused change probability over the region pixels.
    mean_agreement:
        Mean engine-agreement count over the region pixels.
    change_type:
        Dominant change-type hint label for the region (bilingual).
    pixel_count:
        Number of pixels in the region.
    """

    rings: list[list[tuple[float, float]]]
    area_m2: float | None
    mean_probability: float
    mean_agreement: float
    change_type: str
    pixel_count: int

    def to_geojson_feature(self) -> dict[str, Any]:
        """Return the region as a GeoJSON ``Feature`` dictionary."""
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[float(x), float(y)] for x, y in ring] for ring in self.rings
                ],
            },
            "properties": {
                "area_m2": self.area_m2,
                "mean_probability": round(self.mean_probability, 4),
                "mean_agreement": round(self.mean_agreement, 2),
                "change_type": self.change_type,
                "pixel_count": self.pixel_count,
            },
        }


def _trace_rings(
    component: "np.ndarray", row_off: int = 0, col_off: int = 0
) -> list[list[tuple[int, int]]]:
    """Trace the boundary rings of a boolean pixel component.

    Each ``True`` pixel contributes its four unit edges; edges shared by two
    component pixels cancel, leaving only boundary edges. The surviving
    directed edges (component kept on the left) are chained into closed
    rings — the exterior ring plus one ring per hole.

    Parameters
    ----------
    component:
        Boolean ``(h, w)`` array of the connected component (cropped).
    row_off / col_off:
        Offsets of the crop inside the full raster.

    Returns
    -------
    list
        Closed rings as lists of ``(row, col)`` pixel-corner vertices.
    """
    import numpy as np

    comp = np.asarray(component, dtype=bool)
    h, w = comp.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = comp

    # Directed boundary edges, component on the left of travel direction.
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def _add(a: tuple[int, int], b: tuple[int, int]) -> None:
        edges.setdefault(a, []).append(b)

    rows, cols = np.nonzero(comp)
    for r, c in zip(rows.tolist(), cols.tolist()):
        pr, pc = r + 1, c + 1
        if not padded[pr - 1, pc]:  # top edge: travel left -> right
            _add((r, c), (r, c + 1))
        if not padded[pr, pc + 1]:  # right edge: top -> bottom
            _add((r, c + 1), (r + 1, c + 1))
        if not padded[pr + 1, pc]:  # bottom edge: right -> left
            _add((r + 1, c + 1), (r + 1, c))
        if not padded[pr, pc - 1]:  # left edge: bottom -> top
            _add((r + 1, c), (r, c))

    rings: list[list[tuple[int, int]]] = []
    while edges:
        start = next(iter(edges))
        ring = [start]
        current = start
        prev: tuple[int, int] | None = None
        while True:
            nexts = edges.get(current)
            if not nexts:
                break
            if len(nexts) == 1:
                nxt = nexts.pop(0)
            else:
                # At a corner touching: prefer the turn that keeps the
                # component on the left (right-hand turn relative to travel).
                nxt = nexts[0]
                if prev is not None:
                    dr, dc = current[0] - prev[0], current[1] - prev[1]
                    best = None
                    for cand in nexts:
                        cdr, cdc = cand[0] - current[0], cand[1] - current[1]
                        cross = dr * cdc - dc * cdr
                        if best is None or cross < best[0]:
                            best = (cross, cand)
                    nxt = best[1] if best is not None else nexts[0]
                nexts.remove(nxt)
            if not nexts:
                edges.pop(current, None)
            prev = current
            current = nxt
            if current == start:
                break
            ring.append(current)
        ring.append(start)
        # Simplify collinear runs along the ring.
        simplified: list[tuple[int, int]] = []
        for pt in ring:
            if (
                len(simplified) >= 2
                and (simplified[-1][0] - simplified[-2][0])
                * (pt[1] - simplified[-1][1])
                == (simplified[-1][1] - simplified[-2][1]) * (pt[0] - simplified[-1][0])
            ):
                simplified[-1] = pt
            else:
                simplified.append(pt)
        rings.append(
            [(r + row_off, c + col_off) for r, c in simplified]
        )
    return rings


@dataclass
class ChangeResult:
    """Unified result of a change-detection run.

    Attributes
    ----------
    probability_map:
        Float32 ``(H, W)`` fused change probability in ``[0, 1]``
        (``NaN`` on unevaluated pixels).
    binary_map:
        Uint8 ``(H, W)``: 0 = no change, 1 = change, 255 = unevaluated.
    uncertainty_map:
        Float32 ``(H, W)`` uncertainty in ``[0, 1]``.
    agreement_map:
        Uint8 ``(H, W)`` count of engines that voted change (0-4).
    change_type_hints:
        Int16 ``(H, W)`` CVA direction sector per changed pixel
        (-1 = none / unevaluated).
    change_type_labels:
        Bilingual label per sector code.
    georef:
        Georeferencing of all maps.
    pixel_size_m:
        Ground pixel size in metres (``None`` when unknown).
    engine_name:
        Name of the engine that produced the result.
    metadata:
        Free-form metadata (per-method thresholds, weights, warnings...).
    statistics:
        Computed :class:`ChangeStatistics`.
    """

    probability_map: "np.ndarray"
    binary_map: "np.ndarray"
    uncertainty_map: "np.ndarray"
    agreement_map: "np.ndarray"
    change_type_hints: "np.ndarray"
    change_type_labels: dict[int, str] = field(default_factory=dict)
    georef: GeoReference = field(default_factory=GeoReference)
    pixel_size_m: float | None = None
    engine_name: str = "classical"
    metadata: dict[str, Any] = field(default_factory=dict)
    statistics: ChangeStatistics = field(default_factory=ChangeStatistics)

    def __post_init__(self) -> None:
        import numpy as np

        if self.probability_map.ndim != 2:
            raise EngineError(
                f"probability_map must be 2-D, got {self.probability_map.ndim}-D.",
                f"يجب أن تكون خريطة الاحتمالات ثنائية الأبعاد، وجد {self.probability_map.ndim} أبعاد.",
            )
        self.probability_map = np.asarray(self.probability_map, dtype=np.float32)
        self.uncertainty_map = np.asarray(self.uncertainty_map, dtype=np.float32)
        self.binary_map = np.asarray(self.binary_map, dtype=np.uint8)
        self.agreement_map = np.asarray(self.agreement_map, dtype=np.uint8)
        self.change_type_hints = np.asarray(self.change_type_hints, dtype=np.int16)

    # -- statistics --------------------------------------------------------

    def compute_statistics(self) -> ChangeStatistics:
        """Compute (and store) area/region statistics from the binary map."""
        import numpy as np

        binary = self.binary_map
        changed = int(np.count_nonzero(binary == BINARY_CHANGE))
        unevaluated = int(np.count_nonzero(binary == BINARY_UNEVALUATED))
        evaluated = int(binary.size - unevaluated)
        area_m2: float | None = None
        area_ha: float | None = None
        if self.pixel_size_m is not None:
            area_m2 = changed * self.pixel_size_m**2
            area_ha = area_m2 / 10_000.0
        labels, n_regions = _label_components(binary == BINARY_CHANGE)
        _ = labels
        self.statistics = ChangeStatistics(
            changed_pixels=changed,
            evaluated_pixels=evaluated,
            unevaluated_pixels=unevaluated,
            change_fraction=changed / evaluated if evaluated else 0.0,
            change_area_m2=area_m2,
            change_area_ha=area_ha,
            n_regions=int(n_regions),
        )
        return self.statistics

    # -- raster export -------------------------------------------------------

    def to_geotiff(self, directory: Path | str, prefix: str = "change") -> list[Path]:
        """Write all result layers as GeoTIFFs via the Phase-1 writer.

        Parameters
        ----------
        directory:
            Output directory (created if missing).
        prefix:
            Filename prefix for the written layers.

        Returns
        -------
        list[Path]
            Paths of the written files (probability, binary, uncertainty,
            agreement, change-type hints).
        """
        from changemaster.io_engine.writer import write_geotiff

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        layers: list[tuple[str, "np.ndarray", float | None]] = [
            ("probability", self.probability_map, None),
            ("binary", self.binary_map, float(BINARY_UNEVALUATED)),
            ("uncertainty", self.uncertainty_map, None),
            ("agreement", self.agreement_map, None),
            ("change_type", self.change_type_hints, -1.0),
        ]
        for suffix, array, nodata in layers:
            written.append(
                write_geotiff(
                    out_dir / f"{prefix}_{suffix}.tif",
                    array,
                    georef=self.georef,
                    nodata=nodata,
                )
            )
        return written

    # -- vector export ---------------------------------------------------------

    def to_vectors(self, min_pixels: int = 1) -> list[ChangePolygon]:
        """Vectorize connected change regions into georeferenced polygons.

        Connected components (8-connectivity) of the binary change map are
        traced into polygon rings (exterior + holes) at pixel corners, then
        transformed to CRS coordinates via the geotransform (pixel
        coordinates are kept when no geotransform exists).

        Parameters
        ----------
        min_pixels:
            Regions smaller than this pixel count are skipped.

        Returns
        -------
        list[ChangePolygon]
            One polygon per region with area, mean probability, mean
            agreement and the dominant change-type hint.
        """
        import numpy as np

        change = self.binary_map == BINARY_CHANGE
        labels, n_regions = _label_components(change)
        polygons: list[ChangePolygon] = []
        if n_regions == 0:
            return polygons
        transform = self.georef.transform
        pixel_area = self.pixel_size_m**2 if self.pixel_size_m is not None else None

        def _to_crs(r: float, c: float) -> tuple[float, float]:
            if transform is None:
                return float(c), float(r)
            a, b, cc, d, e, f = transform
            return a * c + b * r + cc, d * c + e * r + f

        objects = _find_objects(labels, n_regions)
        for region_id, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            rs, cs = sl
            comp = labels[rs, cs] == region_id
            count = int(comp.sum())
            if count < min_pixels:
                continue
            rings_px = _trace_rings(comp, row_off=rs.start, col_off=cs.start)
            rings = [[_to_crs(r, c) for r, c in ring] for ring in rings_px]
            region_mask = labels == region_id
            probs = self.probability_map[region_mask]
            probs = probs[np.isfinite(probs)]
            hints = self.change_type_hints[region_mask]
            hints = hints[hints >= 0]
            if hints.size:
                values, counts = np.unique(hints, return_counts=True)
                dominant = int(values[int(np.argmax(counts))])
                type_label = self.change_type_labels.get(dominant, str(dominant))
            else:
                type_label = "unknown | غير معروف"
            polygons.append(
                ChangePolygon(
                    rings=rings,
                    area_m2=count * pixel_area if pixel_area is not None else None,
                    mean_probability=float(probs.mean()) if probs.size else 0.0,
                    mean_agreement=float(self.agreement_map[region_mask].mean()),
                    change_type=type_label,
                    pixel_count=count,
                )
            )
        return polygons

    def save_geojson(self, path: Path | str, min_pixels: int = 1) -> Path:
        """Write change polygons as a GeoJSON ``FeatureCollection`` (manual).

        Parameters
        ----------
        path:
            Output ``.geojson`` path.
        min_pixels:
            Minimum region size in pixels.

        Returns
        -------
        Path
            The written file path.
        """
        polygons = self.to_vectors(min_pixels=min_pixels)
        collection: dict[str, Any] = {
            "type": "FeatureCollection",
            "features": [p.to_geojson_feature() for p in polygons],
        }
        if self.georef.crs is not None:
            collection["crs"] = {
                "type": "name",
                "properties": {"name": str(self.georef.crs)},
            }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(collection, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def save_shapefile(self, path: Path | str, min_pixels: int = 1) -> Path:
        """Write change polygons as an ESRI Shapefile with attributes.

        Uses the optional ``pyshp`` dependency (lazy import). The attribute
        table contains area (m²), mean probability, mean agreement,
        change-type hint and pixel count for every polygon.

        Parameters
        ----------
        path:
            Output ``.shp`` path.
        min_pixels:
            Minimum region size in pixels.

        Returns
        -------
        Path
            The written ``.shp`` file path.
        """
        try:
            import shapefile
        except ImportError as exc:
            raise DependencyMissingError(
                "pyshp", "Shapefile export", "تصدير Shapefile"
            ) from exc

        polygons = self.to_vectors(min_pixels=min_pixels)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with shapefile.Writer(str(out), shapeType=shapefile.POLYGON) as writer:
            writer.field("AREA_M2", "N", decimal=2)
            writer.field("MEAN_PROB", "N", decimal=4)
            writer.field("MEAN_AGREE", "N", decimal=2)
            writer.field("CHG_TYPE", "C", size=80)
            writer.field("PIXELS", "N")
            for poly in polygons:
                # Shapefile rings: exterior clockwise, holes counter-clockwise.
                parts = [_orient_ring(poly.rings[0], clockwise=True)]
                parts.extend(
                    _orient_ring(ring, clockwise=False) for ring in poly.rings[1:]
                )
                writer.poly(parts)
                writer.record(
                    poly.area_m2 if poly.area_m2 is not None else 0.0,
                    poly.mean_probability,
                    poly.mean_agreement,
                    poly.change_type[:80],
                    poly.pixel_count,
                )
        if self.georef.crs is not None:
            _write_prj(out.with_suffix(".prj"), str(self.georef.crs))
        return out

    def summary(self) -> str:
        """Human-readable bilingual result summary."""
        stats = self.statistics
        lines = [
            f"Change detection result ({self.engine_name}) | نتيجة كشف التغيرات",
            "=" * 50,
            f"Changed pixels: {stats.changed_pixels} "
            f"({stats.change_fraction:.2%} of evaluated)",
            f"Unevaluated (masked) pixels: {stats.unevaluated_pixels}",
            f"Change regions: {stats.n_regions}",
        ]
        if stats.change_area_m2 is not None and stats.change_area_ha is not None:
            lines.append(
                f"Changed area: {stats.change_area_m2:,.0f} m² "
                f"({stats.change_area_ha:,.2f} ha)"
            )
        return "\n".join(lines)


def _label_components(binary: "np.ndarray") -> tuple["np.ndarray", int]:
    """8-connected component labelling via SciPy (lazy import)."""
    from changemaster.preprocessing._common import require_scipy

    require_scipy()
    import numpy as np
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=bool)
    labels, n = ndimage.label(binary, structure=structure)
    return labels, int(n)


def _find_objects(labels: "np.ndarray", n_regions: int) -> list[Any]:
    """Bounding-box slices per labelled region via SciPy."""
    from scipy import ndimage

    return list(ndimage.find_objects(labels, max_label=n_regions))


def _ring_signed_area(ring: list[tuple[float, float]]) -> float:
    """Signed shoelace area of a closed ring (positive = counter-clockwise)."""
    area = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        area += x0 * y1 - x1 * y0
    return area / 2.0


def _orient_ring(
    ring: list[tuple[float, float]], clockwise: bool
) -> list[tuple[float, float]]:
    """Return ``ring`` oriented clockwise or counter-clockwise as requested."""
    is_ccw = _ring_signed_area(ring) > 0
    if clockwise == is_ccw:
        return list(reversed(ring))
    return list(ring)


def _write_prj(path: Path, crs: str) -> None:
    """Best-effort ``.prj`` sidecar: WKT passthrough or EPSG lookup via rasterio."""
    wkt = crs
    if crs.upper().startswith("EPSG:"):
        try:
            from rasterio.crs import CRS

            wkt = CRS.from_string(crs).to_wkt()
        except Exception:  # noqa: BLE001 - prj sidecar is best-effort
            return
    path.write_text(wkt, encoding="utf-8")
