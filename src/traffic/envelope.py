"""Dynamic-envelope closure function ``E_t = A_static \\ C_t``.

``C_t`` is the set of voxels excluded by current operations:
  * within 3 NM lateral × 1500 ft vertical of any **active approach** centreline,
  * within 3 NM lateral × 1500 ft vertical of any **active departure** centreline,
  * below ``LOW_ALT_CAP_AGL_M`` (default 5000 ft AGL — the closure only applies
    inside the airport airspace; above this the dynamic exclusion no longer adds
    over the Annex-14 envelope).

Under IMC (visibility < 5 km OR ceiling < 1000 ft) the lateral buffer is expanded
by ``IMC_LATERAL_EXPANSION`` (default +25 %).

The full dynamic envelope per slice is

    E_t = A_static  AND  (NOT C_t)

where ``A_static`` is the geometry-engineer's ``is_clear`` boolean grid (loaded
via ``src.geometry.query.is_clear`` when available; otherwise this module
treats ``A_static`` as all-clear and warns once).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import json
import numpy as np
import pandas as pd

from ..utils.grid import VoxelGrid
from ..utils.logs import get_logger

log = get_logger(__name__)

NM_M = 1852.0
FT_M = 0.3048

LATERAL_BUFFER_M: float = 3.0 * NM_M          # 3 NM
VERTICAL_BUFFER_M: float = 1500.0 * FT_M      # 1500 ft
LOW_ALT_CAP_AGL_M: float = 5000.0 * FT_M      # 5000 ft AGL
APPROACH_LENGTH_M: float = 15_000.0           # length of approach cone from threshold
DEPARTURE_LENGTH_M: float = 15_000.0
IMC_LATERAL_EXPANSION: float = 0.25           # +25%
IMC_VIS_SM: float = 5.0 / 1.609                # 5 km in statute miles ≈ 3.1 sm
IMC_CEIL_FT: float = 1000.0


@dataclass
class WeatherState:
    vis_sm: float | None
    ceiling_ft: float | None
    flight_rule: str | None = None

    @property
    def is_imc(self) -> bool:
        vis_imc = self.vis_sm is not None and np.isfinite(self.vis_sm) and self.vis_sm < IMC_VIS_SM
        ceil_imc = self.ceiling_ft is not None and np.isfinite(self.ceiling_ft) and self.ceiling_ft < IMC_CEIL_FT
        rule_imc = self.flight_rule in ("IFR", "LIFR")
        return bool(vis_imc or ceil_imc or rule_imc)


# ---------------------------------------------------------------------------
# Core mask building
# ---------------------------------------------------------------------------

def build_closure_mask(grid: VoxelGrid,
                       runway_ends: list,                   # list[classify.RunwayEnd]
                       arrivals_active: Iterable[str],
                       departures_active: Iterable[str],
                       weather: WeatherState) -> np.ndarray:
    """Return ``C_t`` — boolean grid of voxels excluded by current ops."""
    nx, ny, nz = grid.shape
    closed = np.zeros((nx, ny, nz), dtype=bool)
    lat = LATERAL_BUFFER_M
    if weather.is_imc:
        lat *= (1.0 + IMC_LATERAL_EXPANSION)

    z_centres = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
    z_keep = (z_centres < LOW_ALT_CAP_AGL_M)
    if not z_keep.any():
        return closed
    # We carve a 2-D mask per relevant z-slice then broadcast vertically by the
    # vertical-buffer threshold.
    by_id = {r.runway_id: r for r in runway_ends}
    arr_ends = [by_id[a] for a in arrivals_active if a in by_id]
    dep_ends = [by_id[d] for d in departures_active if d in by_id]

    if not arr_ends and not dep_ends:
        return closed

    xs = grid.x_min + (np.arange(nx) + 0.5) * grid.dx
    ys = grid.y_min + (np.arange(ny) + 0.5) * grid.dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")        # [nx, ny]

    flat_mask_2d = np.zeros((nx, ny), dtype=bool)
    for r in arr_ends:
        flat_mask_2d |= _approach_corridor_mask(X, Y, r, length_m=APPROACH_LENGTH_M, lateral_m=lat)
    for r in dep_ends:
        flat_mask_2d |= _departure_corridor_mask(X, Y, r, length_m=DEPARTURE_LENGTH_M, lateral_m=lat)

    # Vertical buffer is symmetric around the runway-level corridor; voxels are
    # excluded if their AGL z is < ``LOW_ALT_CAP_AGL_M``. Inside that band, the
    # vertical extent of the corridor is taken to be [0, VERTICAL_BUFFER_M].
    z_in_corridor = z_centres < VERTICAL_BUFFER_M
    z_in_cap = z_centres < LOW_ALT_CAP_AGL_M
    # Always restrict closure to LOW_ALT_CAP; corridor heights stay within
    # VERTICAL_BUFFER_M (1500 ft).
    z_use = z_in_corridor & z_in_cap
    closed[:, :, z_use] = flat_mask_2d[:, :, None]
    return closed


def _approach_corridor_mask(X: np.ndarray, Y: np.ndarray, r,
                            length_m: float, lateral_m: float) -> np.ndarray:
    """2-D mask of the approach corridor for runway end ``r``.

    Approach is the cone OUTSIDE the threshold (along the -axis direction) — i.e.
    aircraft are coming IN toward the threshold from outside. We carve the
    rectangular strip from the threshold extending ``length_m`` opposite to the
    runway heading, with half-width ``lateral_m``.
    """
    axis = r.axis                                   # heading of operating direction
    # vector from threshold to each grid point
    px = X - r.thr_x; py = Y - r.thr_y
    along = px * axis[0] + py * axis[1]
    perp = px * (-axis[1]) + py * axis[0]           # right-hand perpendicular
    # Approach is BEFORE the threshold in the direction of flight: along ∈ [-length, 0]
    in_along = (along >= -length_m) & (along <= 0.0)
    in_perp = np.abs(perp) <= lateral_m
    return in_along & in_perp


def _departure_corridor_mask(X: np.ndarray, Y: np.ndarray, r,
                             length_m: float, lateral_m: float) -> np.ndarray:
    """2-D mask of the departure corridor — from threshold OUT, along runway axis."""
    axis = r.axis
    px = X - r.thr_x; py = Y - r.thr_y
    along = px * axis[0] + py * axis[1]
    perp = px * (-axis[1]) + py * axis[0]
    in_along = (along >= 0.0) & (along <= length_m + r.length_m)
    in_perp = np.abs(perp) <= lateral_m
    return in_along & in_perp


# ---------------------------------------------------------------------------
# Combine with A_static (geometry-engineer's is_clear)
# ---------------------------------------------------------------------------

_WARN_NO_GEOMETRY = False


def load_static_mask(icao: str, grid: VoxelGrid) -> Optional[np.ndarray]:
    """Load ``A_static`` (the Annex-14-clear voxel mask) via the canonical
    ``src.geometry.query.SDFQuery`` (per ``src/geometry/INTERFACES.md`` §5).

    We deliberately do **not** read ``sdf.npz`` directly — per geometry-engineer's
    standing guidance, going through ``SDFQuery`` ensures future grid metadata
    (terrain bottoms, lateral safety buffers, OFV interactions, …) is picked up
    automatically.

    Returns
    -------
    ndarray[bool] | None
        Boolean voxel mask (``q.sdf > 0``) with the same shape as ``grid``, or
        ``None`` if ``SDFQuery`` could not be loaded (in which case the caller
        treats the envelope as 'all-clear' and a one-shot warning is logged).
    """
    global _WARN_NO_GEOMETRY
    try:
        from ..geometry.query import SDFQuery         # type: ignore
    except Exception as e:
        if not _WARN_NO_GEOMETRY:
            log.warning("A_static: SDFQuery import failed (%s); treating envelope "
                        "as all-clear. Run `scripts/build_ols.py --airport %s` first.",
                        e, icao)
            _WARN_NO_GEOMETRY = True
        return None
    try:
        q = SDFQuery.from_airport(icao)
    except FileNotFoundError as e:
        if not _WARN_NO_GEOMETRY:
            log.warning("A_static: no sdf.npz for %s (%s); treating envelope as "
                        "all-clear. Run `scripts/build_ols.py --airport %s` first.",
                        icao, e, icao)
            _WARN_NO_GEOMETRY = True
        return None
    if q.sdf.shape != tuple(grid.shape):
        log.error("A_static: SDFQuery shape %s != VoxelGrid shape %s for %s — "
                  "treating envelope as all-clear. This indicates a stale SDF; "
                  "rerun `scripts/build_ols.py --airport %s`.",
                  q.sdf.shape, grid.shape, icao, icao)
        return None
    return (q.sdf > 0.0).astype(bool, copy=False)


def load_static_sdf(icao: str, grid: VoxelGrid) -> Optional[np.ndarray]:
    """Return the raw float **global** ``A_static`` signed-distance field.

    Same source as :func:`load_static_mask` — only the thresholding step is
    skipped. This is the *global* OLS union from ``sdf.npz``, i.e. it already
    contains every approach + takeoff prism in the airport. Suitable for the
    default (non-config-aware) envelope path and for visualisation, but
    **NOT** as a base for the runway-config decomposition — for that, use
    :class:`ConfigStaticCache`, which bakes a separate *static-only* baseline
    via ``PrismIndex.eval_on_grid(grid, prism_index.static_prisms())``.
    """
    try:
        from ..geometry.query import SDFQuery             # type: ignore
        q = SDFQuery.from_airport(icao)
    except Exception:
        return None
    if q.sdf.shape != tuple(grid.shape):
        log.error("static SDF shape %s != VoxelGrid shape %s for %s — ignoring.",
                  q.sdf.shape, grid.shape, icao)
        return None
    return np.asarray(q.sdf, dtype=np.float32)


# ---------------------------------------------------------------------------
# Opt-in: runway-config-aware A_static via geometry.PrismIndex
# ---------------------------------------------------------------------------

_WARN_NO_PRISM = False


def load_prism_index(icao: str):
    """Best-effort loader for ``src.geometry.query.PrismIndex``.

    PrismIndex (introduced post-M2) yields a *runway-config-filtered* SDF: only
    the approach prisms of currently-arriving runways and takeoff prisms of
    currently-departing runways count toward the protection union, in addition
    to the always-on static surfaces (strip, transitional, inner-horizontal,
    conical, OFZs, RESA). Returns ``None`` and warns once if PrismIndex is not
    available (in which case callers must fall back to ``load_static_mask``).
    """
    global _WARN_NO_PRISM
    try:
        from ..geometry.query import PrismIndex          # type: ignore
    except Exception as e:
        if not _WARN_NO_PRISM:
            log.warning("PrismIndex unavailable (%s); config-aware A_static disabled.", e)
            _WARN_NO_PRISM = True
        return None
    try:
        return PrismIndex.from_airport(icao)
    except (FileNotFoundError, Exception) as e:           # pragma: no cover
        if not _WARN_NO_PRISM:
            log.warning("PrismIndex.from_airport(%s) failed (%s); config-aware "
                        "A_static disabled — falling back to static SDFQuery.",
                        icao, e)
            _WARN_NO_PRISM = True
        return None


def _config_key(arrivals_active: Iterable[str],
                departures_active: Iterable[str]) -> tuple:
    """Hashable cache key for an active runway config (order-insensitive)."""
    return (frozenset(arrivals_active or ()), frozenset(departures_active or ()))


class ConfigStaticCache:
    """Per-airport, per-config cache of ``A_static_t`` masks.

    Implements the two perf wins suggested by geometry-engineer:

    1. **Static-base + active-delta decomposition (corrected).** The
       always-on surfaces (strip, transitional, IH, conical, OFZs, RESA) are
       baked once into a *static-only* SDF via
       ``PrismIndex.eval_on_grid(grid, prism_index.static_prisms())``. Per
       slice, only the active approach + takeoff prisms are evaluated and
       min-reduced into a copy of the baked baseline, via
       ``eval_on_grid(grid, arr+dep_prisms, out=static.copy())``.

       Note: this **deliberately does not** seed from ``sdf.npz`` /
       :func:`load_static_sdf`, because that file already includes every
       approach + takeoff prism in the airport — using it as the base would
       defeat the point of being runway-config-aware.

    2. **Config-keyed memoisation.** Real LAX days typically have ≲ 6–8 distinct
       (arrivals, departures) tuples across 96 slices; the cache turns those
       96 evaluations into ~6–8.

    Fallback ladder (in case an older geometry build is loaded):
        eval_on_grid path (preferred, ~30× faster on the LAX grid)
        → distance_to_active_* deltas (with no static baseline — config-only)
        → sdf_at(...) on the full meshgrid (legacy, slow)
    """

    def __init__(self, grid: VoxelGrid, prism_index,
                 sdf_static: Optional[np.ndarray] = None,
                 max_entries: int = 32):
        if prism_index is None:
            raise ValueError("ConfigStaticCache requires a non-None prism_index")
        self.grid = grid
        self.prism_index = prism_index
        self._cache: dict[tuple, np.ndarray] = {}
        self._max_entries = max_entries
        self.hits = 0
        self.misses = 0
        # Pick the fastest available eval path.
        self._has_eval_on_grid = (
            hasattr(prism_index, "eval_on_grid")
            and hasattr(prism_index, "static_prisms")
            and hasattr(prism_index, "prisms_for_surface")
        )
        self._has_delta_api = (
            hasattr(prism_index, "distance_to_active_approach")
            and hasattr(prism_index, "distance_to_active_departure")
        )
        # Bake the *static-only* baseline once if we have eval_on_grid. Caller
        # may also inject one explicitly (useful in tests) — but it MUST be
        # static-only, not the global sdf.npz. We document this on
        # `load_static_sdf` so callers don't accidentally pass the wrong thing.
        self._sdf_static_only: Optional[np.ndarray] = None
        if sdf_static is not None:
            self._sdf_static_only = np.asarray(sdf_static, dtype=np.float32)
        elif self._has_eval_on_grid:
            statics = prism_index.static_prisms()
            self._sdf_static_only = np.asarray(
                prism_index.eval_on_grid(grid, statics), dtype=np.float32,
            )
        # Voxel-centre meshgrid (only needed by legacy delta_api / sdf_at paths).
        self._X = self._Y = self._Z = None
        if not self._has_eval_on_grid:
            nx, ny, nz = grid.shape
            xs = grid.x_min + (np.arange(nx) + 0.5) * grid.dx
            ys = grid.y_min + (np.arange(ny) + 0.5) * grid.dy
            zs = grid.z_min + (np.arange(nz) + 0.5) * grid.dz
            self._X, self._Y, self._Z = np.meshgrid(xs, ys, zs, indexing="ij")

    @property
    def eval_path(self) -> str:
        if self._has_eval_on_grid:
            return "eval_on_grid"
        if self._has_delta_api:
            return "active_delta"
        return "sdf_at"

    def mask_for(self, arrivals_active: Iterable[str],
                 departures_active: Iterable[str]) -> np.ndarray:
        key = _config_key(arrivals_active, departures_active)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        mask = self._compute(arrivals_active, departures_active)
        # Bounded cache — for real LAX days max_entries=32 is well above the
        # observed unique-config count (~6–8) so we never actually evict.
        if len(self._cache) >= self._max_entries:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = mask
        return mask

    def _compute(self, arrivals_active, departures_active) -> np.ndarray:
        arr = list(arrivals_active) if arrivals_active else []
        dep = list(departures_active) if departures_active else []
        if self._has_eval_on_grid:
            from ..geometry.query import APPROACH, TAKEOFF      # local to keep import cheap
            prisms: list = []
            if arr:
                prisms.extend(self.prism_index.prisms_for_surface(APPROACH, arr))
            if dep:
                prisms.extend(self.prism_index.prisms_for_surface(TAKEOFF, dep))
            # Seed from the baked static-only baseline (copy — eval_on_grid mutates `out`).
            if self._sdf_static_only is not None:
                out = self._sdf_static_only.copy()
            else:
                out = np.full(tuple(self.grid.shape), np.inf, dtype=np.float32)
            if prisms:
                sdf_t = np.asarray(
                    self.prism_index.eval_on_grid(self.grid, prisms, out=out),
                    dtype=np.float32,
                )
            else:
                sdf_t = out                                       # no active runways → static-only
            return (sdf_t > 0.0).astype(bool, copy=False)
        if self._has_delta_api:
            d_arr = np.asarray(self.prism_index.distance_to_active_approach(
                self._X, self._Y, self._Z, active_arrivals=arr or None))
            d_dep = np.asarray(self.prism_index.distance_to_active_departure(
                self._X, self._Y, self._Z, active_departures=dep or None))
            stack = [d_arr, d_dep]
            if self._sdf_static_only is not None and self._sdf_static_only.shape == d_arr.shape:
                stack.append(self._sdf_static_only)
            sdf_t = np.minimum.reduce(stack)
            return (sdf_t > 0.0).astype(bool, copy=False)
        # Last resort.
        sdf_t = np.asarray(self.prism_index.sdf_at(
            self._X, self._Y, self._Z,
            active_arrivals=arr or None,
            active_departures=dep or None,
        ))
        return (sdf_t > 0.0).astype(bool, copy=False)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "unique_configs": len(self._cache),
            "hit_rate": (self.hits / total) if total else 0.0,
            "eval_path": self.eval_path,
        }


def static_mask_for_config(grid: VoxelGrid,
                           prism_index,
                           arrivals_active: Iterable[str],
                           departures_active: Iterable[str],
                           sdf_static: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ``A_static_t`` for one (arrivals, departures) config.

    Thin wrapper around :class:`ConfigStaticCache` for one-shot callers. For
    repeated evaluation across many slices, construct a ``ConfigStaticCache``
    once and call ``mask_for(...)`` per slice to amortise the meshgrid +
    benefit from the config-keyed cache.
    """
    cache = ConfigStaticCache(grid, prism_index, sdf_static=sdf_static)
    return cache.mask_for(arrivals_active, departures_active)


def envelope_for_slice(grid: VoxelGrid,
                       runway_ends: list,
                       arrivals_active: Iterable[str],
                       departures_active: Iterable[str],
                       weather: WeatherState,
                       a_static: Optional[np.ndarray] = None,
                       prism_index=None,
                       static_cache: Optional[ConfigStaticCache] = None) -> np.ndarray:
    """Return ``E_t``: boolean voxel grid where eVTOL is permissible.

    Parameters
    ----------
    a_static
        Pre-loaded *global* static mask from :func:`load_static_mask` (the
        canonical and historically-stable path). If ``None`` and no
        ``prism_index``/``static_cache`` is given, A_static is treated as all-clear.
    prism_index
        Optional :class:`src.geometry.query.PrismIndex`. When supplied (and
        ``static_cache`` is not), a fresh cache is built per call — fine for a
        one-off but wasteful in a loop. Prefer ``static_cache``.
    static_cache
        Optional :class:`ConfigStaticCache`. When supplied, ``A_static`` is
        recomputed *per active runway config* (cached) using the static-base
        + active-delta decomposition. ``a_static`` and ``prism_index`` are then
        ignored. This is the recommended path for full "dynamic envelope"
        semantics in a loop.
    """
    closed = build_closure_mask(grid, runway_ends, arrivals_active, departures_active, weather)
    if static_cache is not None:
        a_static_t = static_cache.mask_for(arrivals_active, departures_active)
        return a_static_t & (~closed)
    if prism_index is not None:
        a_static_t = static_mask_for_config(grid, prism_index, arrivals_active, departures_active)
        return a_static_t & (~closed)
    if a_static is None:
        return ~closed
    if a_static.shape != closed.shape:
        raise ValueError(f"A_static shape {a_static.shape} != grid shape {closed.shape}")
    return a_static & (~closed)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_envelope_zarr(envelopes: dict[str, np.ndarray],
                       out_path: Path,
                       grid: VoxelGrid,
                       slice_times: list[pd.Timestamp]) -> Path:
    """Persist a dict ``{slice_iso: mask}`` to Zarr (preferred) or NPZ (fallback).

    Zarr layout::

        envelope_<date>.zarr/
            mask  (T, X, Y, Z) bool, chunks=(1, ny, nx, nz)
            time  (T,)         string
            grid  attrs        VoxelGrid spec
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = len(envelopes)
    if T == 0:
        return out_path
    nx, ny, nz = grid.shape
    # Order envelopes by slice_times.
    ordered_keys = [k for k in (t.isoformat() for t in slice_times) if k in envelopes]

    try:
        import zarr
        zver = tuple(int(p) for p in zarr.__version__.split(".")[:2])
        store = zarr.open(str(out_path), mode="w")
        if zver >= (3, 0):
            mask_arr = store.create_array(
                name="mask", shape=(T, nx, ny, nz), dtype="bool",
                chunks=(1, nx, ny, nz),
            )
            for i, k in enumerate(ordered_keys):
                mask_arr[i] = envelopes[k]
            time_arr = np.array(ordered_keys, dtype="U32")
            store.create_array(name="time", shape=time_arr.shape, dtype=time_arr.dtype)
            store["time"][:] = time_arr
        else:                                          # Zarr v2 path
            mask_arr = store.create_dataset(
                "mask", shape=(T, nx, ny, nz), dtype=bool,
                chunks=(1, nx, ny, nz),
            )
            for i, k in enumerate(ordered_keys):
                mask_arr[i] = envelopes[k]
            store.create_dataset("time", data=np.array(ordered_keys, dtype="U32"))
        store.attrs["grid"] = {
            "x_min": grid.x_min, "x_max": grid.x_max, "dx": grid.dx,
            "y_min": grid.y_min, "y_max": grid.y_max, "dy": grid.dy,
            "z_min": grid.z_min, "z_max": grid.z_max, "dz": grid.dz,
        }
        log.info("envelope: wrote Zarr %s (T=%d, %s)", out_path, T, (nx, ny, nz))
        return out_path
    except Exception as e:                          # zarr missing → npz fallback
        log.warning("zarr unavailable (%s) — falling back to compressed npz", e)
        npz_path = out_path.with_suffix(".npz")
        stack = np.stack([envelopes[k] for k in ordered_keys], axis=0)
        np.savez_compressed(npz_path, mask=stack, time=np.array(ordered_keys, dtype="U32"))
        # Sibling JSON keeps the grid spec discoverable.
        with npz_path.with_suffix(".grid.json").open("w") as f:
            json.dump({
                "x_min": grid.x_min, "x_max": grid.x_max, "dx": grid.dx,
                "y_min": grid.y_min, "y_max": grid.y_max, "dy": grid.dy,
                "z_min": grid.z_min, "z_max": grid.z_max, "dz": grid.dz,
                "slices": ordered_keys,
            }, f, indent=2)
        return npz_path
