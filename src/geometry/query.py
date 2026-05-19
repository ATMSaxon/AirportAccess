"""Trilinear-interpolated queries over the OLS SDF and per-surface distance helpers.

Exports:
  - SDFQuery: 3-D trilinear interpolation over `data/processed/<ICAO>/sdf.npz`.
       .clearance_m(x, y, z)  → signed clearance (positive = clear)
       .is_clear(x, y, z)     → bool array

  - SurfaceDistance: 2-D distance from (x,y) to the union of selected surface
    footprints (uses the GeoPackage written by `build_ols.py`).
       .d_OLS(x, y)        → 2-D distance to union of *all* OLS prism footprints
       .d_runway(x, y)     → distance to nearest runway-strip footprint
       .d_approach(x, y)   → distance to nearest approach-surface footprint
       .d_departure(x, y)  → distance to nearest takeoff-climb footprint

These three (plus the 3-D SDF for d_OLS depth) are the features the ml-engineer
needs (`d_OLS, d_runway, d_approach, d_departure`).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import shapely
from shapely import unary_union

from ..utils.paths import airport_dir
from ..utils.logs import get_logger
from .ols_surfaces import (
    APPROACH, TAKEOFF, RUNWAY_STRIP,
)

logger = get_logger(__name__)


# ============================================================================
# 3-D SDF query (trilinear)
# ============================================================================

class SDFQuery:
    """Trilinear interpolation over a regular ENU SDF grid."""

    def __init__(self, sdf: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray):
        self.sdf = np.asarray(sdf)
        self.grid_x = np.asarray(grid_x, dtype=np.float64)
        self.grid_y = np.asarray(grid_y, dtype=np.float64)
        self.grid_z = np.asarray(grid_z, dtype=np.float64)
        self.dx = float(self.grid_x[1] - self.grid_x[0]) if len(self.grid_x) > 1 else 1.0
        self.dy = float(self.grid_y[1] - self.grid_y[0]) if len(self.grid_y) > 1 else 1.0
        self.dz = float(self.grid_z[1] - self.grid_z[0]) if len(self.grid_z) > 1 else 1.0

    @classmethod
    def load(cls, path: str | Path) -> "SDFQuery":
        d = np.load(path)
        return cls(d["sdf"], d["grid_x"], d["grid_y"], d["grid_z"])

    @classmethod
    def from_airport(cls, icao: str) -> "SDFQuery":
        return cls.load(airport_dir(icao, "processed") / "sdf.npz")

    def _interp(self, x, y, z):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        nx, ny, nz = self.sdf.shape
        fx = (x - self.grid_x[0]) / self.dx
        fy = (y - self.grid_y[0]) / self.dy
        fz = (z - self.grid_z[0]) / self.dz
        ix0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        iy0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        iz0 = np.clip(np.floor(fz).astype(np.int64), 0, nz - 2)
        tx = np.clip(fx - ix0, 0.0, 1.0)
        ty = np.clip(fy - iy0, 0.0, 1.0)
        tz = np.clip(fz - iz0, 0.0, 1.0)
        s = self.sdf
        c000 = s[ix0,     iy0,     iz0]
        c100 = s[ix0 + 1, iy0,     iz0]
        c010 = s[ix0,     iy0 + 1, iz0]
        c110 = s[ix0 + 1, iy0 + 1, iz0]
        c001 = s[ix0,     iy0,     iz0 + 1]
        c101 = s[ix0 + 1, iy0,     iz0 + 1]
        c011 = s[ix0,     iy0 + 1, iz0 + 1]
        c111 = s[ix0 + 1, iy0 + 1, iz0 + 1]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        out = c0 * (1 - tz) + c1 * tz
        return out

    def clearance_m(self, x, y, z):
        """Trilinearly-interpolated SDF value. Positive = clear (outside OLS protection)."""
        out = self._interp(x, y, z)
        return float(out[0]) if out.size == 1 else out

    def is_clear(self, x, y, z):
        """`clearance_m > 0`. Same shape as inputs."""
        v = self._interp(x, y, z)
        result = v > 0
        return bool(result[0]) if result.size == 1 else result

    # ---- d_OLS feature (depth/clearance, 3-D) — alias for `clearance_m` ----
    def d_OLS(self, x, y, z):
        """3-D signed distance to the OLS-protection union. Positive = clear."""
        return self.clearance_m(x, y, z)


# ============================================================================
# 2-D surface-family distance (for ML features and visualisation)
# ============================================================================

class SurfaceDistance:
    """2-D distance from (x, y) to the union of selected surface footprints.

    Construct from the OLS GeoPackage written by `scripts/build_ols.py`.
    Per-family unions are cached on first use.
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self._family_cache: dict[tuple, object] = {}

    @classmethod
    def from_airport(cls, icao: str) -> "SurfaceDistance":
        import geopandas as gpd
        gpkg = airport_dir(icao, "processed") / "ols.gpkg"
        gdf = gpd.read_file(gpkg, layer="ols")
        return cls(gdf)

    def _union(self, surfaces: Sequence[str]):
        key = tuple(sorted(surfaces))
        if key not in self._family_cache:
            sub = self.gdf[self.gdf["surface"].isin(surfaces)]
            if len(sub) == 0:
                self._family_cache[key] = None
            else:
                self._family_cache[key] = unary_union(sub.geometry.tolist())
        return self._family_cache[key]

    def _distance(self, surfaces: Sequence[str], x, y):
        union = self._union(surfaces)
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if union is None:
            out = np.full_like(x, np.inf)
            return float(out[0]) if out.size == 1 else out
        pts = shapely.points(x, y)
        out = shapely.distance(union, pts)
        return float(out[0]) if out.size == 1 else out

    def d_OLS(self, x, y):
        """2-D distance to the union of *all* OLS prism footprints (positive everywhere)."""
        return self._distance(tuple(sorted(self.gdf["surface"].unique().tolist())), x, y)

    def d_runway(self, x, y):
        return self._distance((RUNWAY_STRIP,), x, y)

    def d_approach(self, x, y):
        return self._distance((APPROACH,), x, y)

    def d_departure(self, x, y):
        return self._distance((TAKEOFF,), x, y)
