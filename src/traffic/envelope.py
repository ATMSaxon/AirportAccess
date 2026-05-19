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
    """Best-effort load of ``A_static`` (the Annex-14-clear voxel mask).

    Tries ``src.geometry.query.is_clear`` first (if the geometry-engineer has
    published that interface). Falls back to ``data/processed/<ICAO>/sdf.npz``
    (the M2 artefact) and treats positive-SDF cells as 'clear'. If neither is
    available, returns ``None`` and the caller treats A_static as all-clear.
    """
    global _WARN_NO_GEOMETRY
    try:                                            # preferred path
        from ..geometry import query as gquery     # type: ignore
        if hasattr(gquery, "is_clear"):
            return gquery.is_clear(icao=icao, grid=grid)
    except Exception as e:
        log.debug("geometry.query.is_clear unavailable: %s", e)

    from ..utils.paths import airport_dir
    sdf_path = airport_dir(icao, "processed") / "sdf.npz"
    if sdf_path.exists():
        try:
            data = np.load(sdf_path)
            sdf = data["sdf"] if "sdf" in data.files else data[data.files[0]]
            if sdf.shape == tuple(grid.shape):
                return (sdf > 0.0)
            log.warning("sdf.npz shape %s mismatches grid %s; ignoring", sdf.shape, grid.shape)
        except Exception as e:
            log.warning("sdf.npz unreadable: %s", e)

    if not _WARN_NO_GEOMETRY:
        log.warning("A_static unavailable for %s; treating envelope as 'all-clear'", icao)
        _WARN_NO_GEOMETRY = True
    return None


def envelope_for_slice(grid: VoxelGrid,
                       runway_ends: list,
                       arrivals_active: Iterable[str],
                       departures_active: Iterable[str],
                       weather: WeatherState,
                       a_static: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ``E_t``: boolean voxel grid where eVTOL is permissible."""
    closed = build_closure_mask(grid, runway_ends, arrivals_active, departures_active, weather)
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
        store = zarr.open(str(out_path), mode="w")
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
