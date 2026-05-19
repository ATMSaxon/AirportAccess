"""Vectorised numpy SDF on the airport's ENU voxel grid.

The SDF is computed as the *minimum signed distance* across all OLS prisms:
  - inside the union of prisms → SDF < 0 (negative depth)
  - outside the union → SDF > 0 (positive clearance)

Per-prism distance uses a "box SDF" decomposition:
  d_xy = 2-D signed distance from (x,y) to the footprint polygon (positive outside),
  d_z  = max(z - z_top(x,y), z_low - z)   — positive if outside vertical range,
                                            negative if inside.
  inside  : d_xy ≤ 0 AND d_z ≤ 0  → SDF = max(d_xy, d_z)   (both negative; returns the
                                                            distance to the nearest face)
  outside : SDF = sqrt(max(d_xy,0)^2 + max(d_z,0)^2)

`min` over prisms yields the union SDF in sign; the magnitude is the distance to the
*nearest* prism face (which equals the union-boundary distance everywhere except near
shared faces, where it remains a tight upper bound).
"""
from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import shapely
import geopandas as gpd
from scipy.ndimage import distance_transform_edt

from ..utils.grid import VoxelGrid
from ..utils.logs import get_logger
from .ols_surfaces import Prism

logger = get_logger(__name__)

_BIG = np.float32(1e9)


def _signed_2d_distance(poly, xx: np.ndarray, yy: np.ndarray,
                        sampling: Tuple[float, float]) -> np.ndarray:
    """Approximate 2-D signed distance: positive outside polygon, negative inside.

    Uses cell-centre membership (shapely 2 contains_xy) + scipy's exact EDT.
    Accuracy ≈ ±0.5 cell near the polygon boundary.
    """
    nx, ny = xx.shape
    inside = shapely.contains_xy(poly, xx.ravel(), yy.ravel()).reshape(nx, ny)
    if not inside.any():
        return np.full((nx, ny), _BIG, dtype=np.float32)
    if inside.all():
        return np.full((nx, ny), -_BIG, dtype=np.float32)
    d_out = distance_transform_edt(~inside, sampling=sampling)
    d_in = distance_transform_edt(inside, sampling=sampling)
    return (d_out - d_in).astype(np.float32)


def build_sdf(gdf: gpd.GeoDataFrame, grid: VoxelGrid) -> Tuple[np.ndarray, dict]:
    """Build (nx, ny, nz) float32 SDF from a GeoDataFrame of prisms.

    Returns (sdf, meta) where meta = dict(grid_x=..., grid_y=..., grid_z=...).
    """
    xs, ys, zs = grid.coords()
    nx, ny, nz = grid.shape
    sampling = (grid.dx, grid.dy)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    sdf = np.full((nx, ny, nz), _BIG, dtype=np.float32)

    n_prisms = len(gdf)
    logger.info("Building SDF on %d×%d×%d grid from %d prisms (sampling %sm × %sm × %sm)",
                nx, ny, nz, n_prisms, grid.dx, grid.dy, grid.dz)

    for i, row in enumerate(gdf.itertuples(index=False)):
        prism = Prism.from_row(row._asdict())
        d2d = _signed_2d_distance(prism.footprint, xx, yy, sampling)
        ztop2d = prism.evaluate_z_top(xx, yy).astype(np.float32)
        z_low = np.float32(prism.z_low)
        for k, z in enumerate(zs):
            zf = np.float32(z)
            dz_above = zf - ztop2d
            dz_below = z_low - zf
            dz = np.maximum(dz_above, dz_below)
            inside_mask = (d2d <= 0) & (dz <= 0)
            outside_part = np.sqrt(
                np.maximum(d2d, 0.0).astype(np.float32) ** 2
                + np.maximum(dz, 0.0).astype(np.float32) ** 2
            )
            inside_part = np.maximum(d2d, dz).astype(np.float32)
            this = np.where(inside_mask, inside_part, outside_part).astype(np.float32)
            np.minimum(sdf[:, :, k], this, out=sdf[:, :, k])
        if (i + 1) % 10 == 0 or (i + 1) == n_prisms:
            logger.info("  prism %d/%d (%s, %s)", i + 1, n_prisms, prism.surface, prism.name)

    meta = {
        "grid_x": xs.astype(np.float32),
        "grid_y": ys.astype(np.float32),
        "grid_z": zs.astype(np.float32),
    }
    return sdf, meta


def save_sdf(path, sdf: np.ndarray, meta: dict):
    np.savez_compressed(path, sdf=sdf, **meta)
