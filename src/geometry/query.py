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

  - PrismIndex: per-prism membership and *runway-configuration-aware* filtered
    SDF (computes signed distance to the union of prisms restricted to the
    currently-active arrival / departure runways).
       .point_in_approach_prism(x, y, z, rwy_id=None)   → bool array
       .point_in_departure_prism(x, y, z, rwy_id=None)  → bool array
       .point_in_missed_approach(x, y, z, rwy_id=None)  → bool array
       .sdf_at(x, y, z, active_arrivals=None, active_departures=None) → float array
       .distance_to_active_approach(x, y, z, active_arrivals=None)    → float array
       .distance_to_active_departure(x, y, z, active_departures=None) → float array

The first three groups (3-D SDF, 2-D family distance) are static (run-config
agnostic). PrismIndex layers on top so the ml-lane / planning-lane can sample
counterfactual conflicts under a specific runway configuration (e.g. KLAX
{arr: 24R/25L, dep: 24L/25R}) without rebuilding the global SDF.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Sequence
import numpy as np
import shapely
from shapely import unary_union

from ..utils.paths import airport_dir
from ..utils.logs import get_logger
from .ols_surfaces import (
    APPROACH, TAKEOFF, TRANSITIONAL, INNER_HORIZONTAL, CONICAL,
    RUNWAY_STRIP, RESA, OFZ_INNER_APPROACH, OFZ_INNER_TRANSITIONAL,
    Prism,
)

# Surfaces that are *always* protected regardless of the active runway-config.
# (Inner-horizontal, conical, transitional, OFZs, runway strip, RESA do not flip
# direction with traffic flow; only the approach/takeoff prisms are selectable.)
_STATIC_SURFACES = (
    RUNWAY_STRIP, INNER_HORIZONTAL, CONICAL, TRANSITIONAL,
    OFZ_INNER_APPROACH, OFZ_INNER_TRANSITIONAL, RESA,
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


# ============================================================================
# Runway-config-aware prism index (per-prism membership + filtered SDF)
# ============================================================================

def _scalar_or_array(out: np.ndarray, was_scalar: bool):
    if was_scalar:
        return out.item() if out.size == 1 else float(out.ravel()[0])
    return out


class PrismIndex:
    """Per-prism membership and runway-configuration-aware filtered SDF.

    Loads the OLS GeoPackage written by `scripts/build_ols.py` and lets you
    ask:

      * is point P inside the *approach* / *takeoff* / *missed-approach* prism
        of a specific runway (or any runway if ``rwy_id`` is None)?
      * what is the signed distance to the *active* protection union — i.e.
        the union of approach prisms for ``active_arrivals``, takeoff prisms
        for ``active_departures``, plus the always-protected static surfaces
        (runway strip, transitional, inner-horizontal, conical, OFZs, RESA)?

    The "missed approach" geometry is modelled as the takeoff-climb prism of
    the *same* runway-ID: when an aircraft on the approach to RWY 24L goes
    around, it climbs straight ahead (along +runway-heading) — which is
    exactly the takeoff-climb surface anchored at that runway's stop-end.
    """

    def __init__(self, gdf):
        self.gdf = gdf
        self._prisms: list[Prism] = [Prism.from_row(r) for _, r in gdf.iterrows()]
        # Pre-index by surface and by (runway_id, surface) for fast lookups.
        self._by_surface: dict[str, list[Prism]] = {}
        self._by_rwy_surface: dict[tuple[str, str], list[Prism]] = {}
        for p in self._prisms:
            self._by_surface.setdefault(p.surface, []).append(p)
            self._by_rwy_surface.setdefault((p.runway_id, p.surface), []).append(p)
        self._static_prisms = [p for p in self._prisms if p.surface in _STATIC_SURFACES]

    # ---------------------------------------------------------------- load
    @classmethod
    def from_airport(cls, icao: str) -> "PrismIndex":
        import geopandas as gpd
        gpkg = airport_dir(icao, "processed") / "ols.gpkg"
        gdf = gpd.read_file(gpkg, layer="ols")
        return cls(gdf)

    @classmethod
    def from_gdf(cls, gdf) -> "PrismIndex":
        return cls(gdf)

    # ---------------------------------------------------------------- helpers
    def runway_ids(self) -> list[str]:
        return sorted({p.runway_id for p in self._prisms if p.runway_id != "-"})

    def _prisms_for(self, surface: str, rwy_id: Optional[str]) -> list[Prism]:
        if rwy_id is None:
            return list(self._by_surface.get(surface, []))
        return list(self._by_rwy_surface.get((rwy_id, surface), []))

    def prisms_for_surface(self, surface: str,
                           runway_ids: Optional[Iterable[str]] = None) -> list[Prism]:
        """Public lookup: return all prisms of a given surface (optionally filtered to runways).

        Example: `idx.prisms_for_surface("approach", active_arrivals)` returns the
        approach prisms (each runway has 3 sub-prisms in the Annex-14 3-section model)
        for every active arrival runway.
        """
        if runway_ids is None:
            return list(self._by_surface.get(surface, []))
        out: list[Prism] = []
        for rid in runway_ids:
            out += self._by_rwy_surface.get((rid, surface), [])
        return out

    def static_prisms(self) -> list[Prism]:
        """Return the runway-config-agnostic prism set (strip, transitional, IH, conical, OFZs, RESA)."""
        return list(self._static_prisms)

    # ---------------------------------------------------------------- per-prism membership
    @staticmethod
    def _point_in_prism(prism: Prism, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        inside_xy = shapely.contains_xy(prism.footprint, x, y)
        z_top = prism.evaluate_z_top(x, y)
        inside_z = (z >= prism.z_low) & (z <= z_top)
        return inside_xy & inside_z

    def _any_prism_contains(self, prisms: list[Prism], x, y, z):
        was_scalar = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        out = np.zeros_like(x, dtype=bool)
        for p in prisms:
            out |= self._point_in_prism(p, x, y, z)
        return _scalar_or_array(out, was_scalar)

    # Public membership API ----------------------------------------------------
    def point_in_approach_prism(self, x, y, z, rwy_id: Optional[str] = None):
        """True if (x,y,z) is inside any *approach* prism (optionally restricted to one runway)."""
        return self._any_prism_contains(self._prisms_for(APPROACH, rwy_id), x, y, z)

    def point_in_departure_prism(self, x, y, z, rwy_id: Optional[str] = None):
        """True if (x,y,z) is inside any *takeoff-climb* prism (optionally restricted to one runway)."""
        return self._any_prism_contains(self._prisms_for(TAKEOFF, rwy_id), x, y, z)

    def point_in_missed_approach(self, x, y, z, rwy_id: Optional[str] = None):
        """True if (x,y,z) is inside the missed-approach prism (takeoff-climb of the same RWY ID).

        Modelling note: in the absence of an explicit missed-approach geometry
        in `code4_precision.yaml`, we use the takeoff-climb surface for the
        same runway designation — i.e. straight-ahead climb from the stop end,
        which is the standard ICAO PANS-OPS construct for non-published GA.
        """
        return self.point_in_departure_prism(x, y, z, rwy_id)

    def point_in_static_protection(self, x, y, z):
        """True if (x,y,z) is inside any always-on (runway-config-agnostic) prism."""
        return self._any_prism_contains(self._static_prisms, x, y, z)

    # ---------------------------------------------------------------- per-prism signed 3-D distance
    @staticmethod
    def _signed_3d_distance_to_prism(prism: Prism, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Box-SDF signed distance to a single prism. Negative inside, positive outside."""
        pts = shapely.points(x, y)
        # NB: shapely.distance(polygon, point) returns 0 when the point is INSIDE
        # the polygon (the geometries intersect). For a signed lateral distance we
        # need the unsigned distance to the boundary instead.
        d_xy_abs = shapely.distance(prism.footprint.boundary, pts)
        inside_xy = shapely.contains_xy(prism.footprint, x, y)
        d_xy = np.where(inside_xy, -d_xy_abs, d_xy_abs)

        z_top = prism.evaluate_z_top(x, y)
        z_low = prism.z_low
        dz_above = z - z_top
        dz_below = z_low - z
        d_z = np.maximum(dz_above, dz_below)               # >=0 outside [z_low, z_top]

        inside_box = (d_xy <= 0) & (d_z <= 0)
        # Inside: both negative → closest face is the *least* negative ⇒ max
        inside_part = np.maximum(d_xy, d_z)
        # Outside: Euclidean combination of positive parts, plus any lone negative.
        outside_lat = np.maximum(d_xy, 0.0)
        outside_vert = np.maximum(d_z, 0.0)
        outside_part = np.sqrt(outside_lat ** 2 + outside_vert ** 2) \
                       + np.minimum(np.maximum(d_xy, d_z), 0.0)
        return np.where(inside_box, inside_part, outside_part).astype(np.float64)

    def _min_signed_distance(self, prisms: list[Prism], x, y, z) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        if not prisms:
            return np.full_like(x, np.inf, dtype=np.float64)
        out = np.full_like(x, np.inf, dtype=np.float64)
        for p in prisms:
            d = self._signed_3d_distance_to_prism(p, x, y, z)
            np.minimum(out, d, out=out)
        return out

    # ---------------------------------------------------------------- filtered SDF API
    def sdf_at(self, x, y, z,
               active_arrivals: Optional[Iterable[str]] = None,
               active_departures: Optional[Iterable[str]] = None):
        """Signed distance to the *runway-config-aware* protection union.

        Negative = inside the active protection volume. The union is:
            (∪ approach prisms for active_arrivals)
          ∪ (∪ takeoff prisms for active_departures)
          ∪ (always-on static prisms: runway-strip, transitional, inner-horiz,
                                       conical, OFZ-inapp, OFZ-intr, RESA)

        If both ``active_arrivals`` and ``active_departures`` are None the
        filter degenerates to "all approach + all takeoff" (equivalent to the
        static global SDF — but recomputed from prisms, so values can differ
        from `SDFQuery.clearance_m` by ≲ 0.5 cell-spacing).
        """
        was_scalar = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
        keep: list[Prism] = list(self._static_prisms)
        if active_arrivals is None:
            keep += self._by_surface.get(APPROACH, [])
        else:
            for rid in active_arrivals:
                keep += self._by_rwy_surface.get((rid, APPROACH), [])
        if active_departures is None:
            keep += self._by_surface.get(TAKEOFF, [])
        else:
            for rid in active_departures:
                keep += self._by_rwy_surface.get((rid, TAKEOFF), [])
        out = self._min_signed_distance(keep, x, y, z)
        return _scalar_or_array(out, was_scalar)

    def distance_to_active_approach(self, x, y, z,
                                    active_arrivals: Optional[Iterable[str]] = None):
        """Signed 3-D distance to the union of approach prisms of the active arrival runways.

        Negative inside an approach prism; positive elsewhere. If
        ``active_arrivals`` is None, all approach prisms are used.
        """
        was_scalar = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
        if active_arrivals is None:
            prisms = list(self._by_surface.get(APPROACH, []))
        else:
            prisms = []
            for rid in active_arrivals:
                prisms += self._by_rwy_surface.get((rid, APPROACH), [])
        out = self._min_signed_distance(prisms, x, y, z)
        return _scalar_or_array(out, was_scalar)

    def distance_to_active_departure(self, x, y, z,
                                     active_departures: Optional[Iterable[str]] = None):
        """Signed 3-D distance to the union of takeoff-climb prisms of the active departure runways."""
        was_scalar = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
        if active_departures is None:
            prisms = list(self._by_surface.get(TAKEOFF, []))
        else:
            prisms = []
            for rid in active_departures:
                prisms += self._by_rwy_surface.get((rid, TAKEOFF), [])
        out = self._min_signed_distance(prisms, x, y, z)
        return _scalar_or_array(out, was_scalar)

    def distance_to_active_missed_approach(self, x, y, z,
                                           active_arrivals: Optional[Iterable[str]] = None):
        """Missed-approach climb-surface distance for the currently-arriving runways.

        Aliases :meth:`distance_to_active_departure` with ``active_arrivals``
        (an aircraft aborting an approach climbs straight ahead on the same
        runway's takeoff-climb surface — see :meth:`point_in_missed_approach`).
        """
        return self.distance_to_active_departure(x, y, z, active_arrivals)

    # ---------------------------------------------------------------- grid-mode evaluator
    def eval_on_grid(self, grid, prisms: Optional[Iterable[Prism]] = None,
                     out: Optional[np.ndarray] = None,
                     log_every: Optional[int] = None) -> np.ndarray:
        """Min-reduced signed-distance ndarray on a :class:`VoxelGrid` over `prisms`.

        Mathematically equivalent to evaluating
        :meth:`_signed_3d_distance_to_prism` at every voxel-centre and taking
        min over `prisms`, but uses ``scipy.ndimage.distance_transform_edt``
        for the lateral 2-D term — orders of magnitude faster on large grids
        (~0.4 s/prism on a 600×600 grid vs minutes for the per-point shapely
        path).

        Use this when you need a *whole-grid* config-aware SDF (e.g. the
        traffic lane's `ConfigStaticCache`). The recommended decomposition
        pattern bakes the static-only SDF *once* per airport, then unions in
        only the active approach/takeoff prisms per config:

            # Bake once per airport (cache the result):
            sdf_static_only = idx.eval_on_grid(grid, idx.static_prisms())

            # Per (active_arrivals, active_departures) config:
            arr_prisms = idx.prisms_for_surface(APPROACH, active_arrivals)
            dep_prisms = idx.prisms_for_surface(TAKEOFF, active_departures)
            sdf_t = idx.eval_on_grid(
                grid, arr_prisms + dep_prisms,
                out=sdf_static_only.astype(np.float32, copy=True),
            )
            A_static_t = sdf_t > 0

        On LAX (600×600×117, 100 m × 100 m × 30 m): ~23 s static bake,
        ~5 s per config (10 active prisms typical). Matches :meth:`sdf_at`
        to within ≲ 0.5 · √(dx² + dy²) ≈ 70 m laterally (EDT accuracy).

        .. warning::
           Do **not** seed `out` with the baked `sdf.npz` artefact — that file
           already contains *all* approach/takeoff prisms (it's the global
           static SDF), so min-reducing the active subset on top of it just
           yields the global SDF, not the config-aware one. Use
           ``idx.eval_on_grid(grid, idx.static_prisms())`` to bake the
           runway-config-agnostic baseline instead.

        Parameters
        ----------
        grid : VoxelGrid
        prisms : iterable of Prism, optional
            If None, uses every prism in the index (≡ rebuilds the full static SDF).
        out : ndarray, optional
            Pre-allocated (nx, ny, nz) float32 buffer to min-reduce into. When
            supplied, the result is the union with whatever is already in the
            buffer — convenient for chaining static + active deltas.
        log_every : int or None
            Progress logging cadence (None = silent; default None to avoid log
            spam in the M3 slice loop).

        Returns
        -------
        sdf : np.ndarray, shape `grid.shape`, dtype float32
            Negative inside the prism union, positive outside.
        """
        # Local import keeps the import graph shallow at module load time.
        from .sdf import build_sdf_from_prisms
        if prisms is None:
            prisms = self._prisms
        return build_sdf_from_prisms(prisms, grid, out=out, log_every=log_every)
