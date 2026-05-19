"""Disk-I/O loaders for the planner.

The planner is built so its algorithmic core (Planner.plan) never touches disk: it consumes
a pre-built ``PlannerInputs``. This module is the *only* place where artefact files
(``sdf.npz``, ``ofv_*.npz``, ``envelope_*.zarr``, ``risk_grid_*.zarr``, density arrays) are
read in. When an artefact is missing in production code paths, ``MissingArtifactError`` is
raised with an actionable next-command. The corridor orchestrator catches missing optional
artefacts (envelope, risk, density) and degrades gracefully.

Schemas (mirror what teammates publish):

* ``data/processed/{icao}/sdf.npz``         — npz with keys ``sdf`` (nx,ny,nz float32) and
                                              optional ``grid`` (json-encoded VoxelGrid dict).
* ``data/processed/{icao}/ofv_V?.npz``       — same layout as sdf.
* ``data/processed/{icao}/envelope_{date}.zarr`` — zarr array (T,nx,ny,nz) bool with attrs
                                                    ``time_index`` (UTC ISO strings).
* ``data/processed/{icao}/risk_grid_{model}.zarr`` — float32 (T,nx,ny,nz) in [0,1].
* ``data/processed/{icao}/traffic_density_{date}.zarr`` — float32 (T,nx,ny,nz) aircraft·s / voxel.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.config import load_airport
from src.utils.grid import VoxelGrid
from src.utils.logs import get_logger

LOG = get_logger(__name__)


class MissingArtifactError(FileNotFoundError):
    """Raised when a required artefact (SDF, OFV) is missing.

    The error message always includes the next command the user should run.
    """


@dataclass
class EnvelopeSlice:
    """A single 15-minute slice of the dynamic envelope, or A_static fallback."""
    grid: VoxelGrid
    mask: np.ndarray            # bool (nx, ny, nz)
    dynamic_used: bool          # False if we fell back to A_static
    slice_time_utc: pd.Timestamp | None


# ---------------------------------------------------------------------------
# SDF / OFV
# ---------------------------------------------------------------------------

def _grid_from_npz(npz: np.lib.npyio.NpzFile, fallback_cfg: dict | None = None) -> VoxelGrid:
    """Extract a VoxelGrid from a sidecar 'grid' field, or fall back to airport cfg."""
    if "grid" in npz.files:
        raw = npz["grid"]
        meta = json.loads(str(raw)) if raw.shape == () else json.loads(raw.item())
        return VoxelGrid(**meta)
    if fallback_cfg is not None:
        return VoxelGrid.from_airport_cfg(fallback_cfg)
    raise ValueError("npz lacks 'grid' field and no fallback config provided")


def load_sdf(icao: str) -> Tuple[VoxelGrid, np.ndarray]:
    """Load the static OLS SDF for an airport.

    Returns a (grid, sdf) tuple where ``sdf > 0`` outside obstacles (clear airspace).
    """
    fp = paths.airport_dir(icao, kind="processed") / "sdf.npz"
    if not fp.exists():
        raise MissingArtifactError(
            f"Missing SDF for {icao}: {fp}. "
            f"Run `python scripts/build_ols.py --airport {icao}` first."
        )
    cfg = load_airport(icao)
    with np.load(fp, allow_pickle=False) as z:
        grid = _grid_from_npz(z, fallback_cfg=cfg)
        sdf = z["sdf"].astype(np.float32, copy=False)
    LOG.debug("loaded SDF %s shape=%s min=%.1f max=%.1f", fp, sdf.shape, sdf.min(), sdf.max())
    return grid, sdf


def load_ofv(icao: str, vertiport_id: str) -> Tuple[VoxelGrid, np.ndarray]:
    """Load the per-vertiport obstacle-free volume mask (positive sdf -> clear OFV airspace)."""
    fp = paths.airport_dir(icao, kind="processed") / f"ofv_{vertiport_id}.npz"
    if not fp.exists():
        raise MissingArtifactError(
            f"Missing OFV {vertiport_id} for {icao}: {fp}. "
            f"Run `python scripts/build_ols.py --airport {icao}` first."
        )
    cfg = load_airport(icao)
    with np.load(fp, allow_pickle=False) as z:
        grid = _grid_from_npz(z, fallback_cfg=cfg)
        ofv = z["sdf"].astype(np.float32, copy=False)
    return grid, ofv


def ofv_mask(icao: str, vertiport_id: str) -> Tuple[VoxelGrid, np.ndarray]:
    """Same as load_ofv but already collapsed to a boolean clear-cells mask."""
    grid, ofv = load_ofv(icao, vertiport_id)
    return grid, ofv > 0


# ---------------------------------------------------------------------------
# Dynamic envelope
# ---------------------------------------------------------------------------

def _slice_index_for_hour(hour: int, n_slices: int) -> int:
    """Pick the 15-minute slice that covers H:30 of `hour`. Assumes slices start at 00:00 UTC."""
    target_minute = hour * 60 + 30
    idx = target_minute // 15
    return int(np.clip(idx, 0, max(n_slices - 1, 0)))


def load_envelope_slice(icao: str, date: str, hour: int) -> EnvelopeSlice | None:
    """Load a single 15-minute envelope slice.

    Returns ``None`` if no envelope file exists for the given date (caller falls back to
    A_static). Sets ``dynamic_used=True`` when the load succeeded, ``False`` otherwise.
    """
    fp = paths.airport_dir(icao, kind="processed") / f"envelope_{date}.zarr"
    if not fp.exists():
        LOG.info("envelope file missing for %s %s (%s) — falling back to A_static", icao, date, fp)
        return None
    try:
        import zarr  # local import to keep planner usable without zarr at unit-test time
    except ImportError as e:  # pragma: no cover — zarr is in requirements.txt
        LOG.warning("zarr not importable (%s); cannot load envelope", e)
        return None

    cfg = load_airport(icao)
    try:
        arr = zarr.open(str(fp), mode="r")
        data = np.asarray(arr)
        T = int(data.shape[0])
        idx = _slice_index_for_hour(hour, T)
        mask = np.asarray(data[idx]).astype(bool, copy=False)
        # Pull time index if stored as attribute.
        slice_t = None
        try:
            time_index = arr.attrs.get("time_index")
            if time_index is not None and idx < len(time_index):
                slice_t = pd.Timestamp(time_index[idx], tz="UTC")
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        LOG.warning("failed to read envelope %s: %s — falling back to A_static", fp, e)
        return None
    grid = VoxelGrid.from_airport_cfg(cfg)
    return EnvelopeSlice(grid=grid, mask=mask, dynamic_used=True, slice_time_utc=slice_t)


# ---------------------------------------------------------------------------
# Risk grid
# ---------------------------------------------------------------------------

def load_risk_slice(icao: str, model: str, date: str, hour: int) -> np.ndarray | None:
    """Return the risk field slice (nx,ny,nz) float32 in [0,1], or None if missing."""
    fp = paths.airport_dir(icao, kind="processed") / f"risk_grid_{model}.zarr"
    if not fp.exists():
        # Also accept date-stamped variants.
        fp2 = paths.airport_dir(icao, kind="processed") / f"risk_grid_{model}_{date}.zarr"
        if fp2.exists():
            fp = fp2
        else:
            LOG.info("risk grid missing for %s/%s (%s)", icao, model, fp)
            return None
    try:
        import zarr
    except ImportError:  # pragma: no cover
        return None
    try:
        arr = zarr.open(str(fp), mode="r")
        data = np.asarray(arr)
        if data.ndim == 4:
            T = data.shape[0]
            idx = _slice_index_for_hour(hour, T)
            slc = np.asarray(data[idx]).astype(np.float32, copy=False)
        elif data.ndim == 3:
            slc = data.astype(np.float32, copy=False)
        else:
            LOG.warning("unexpected risk grid ndim=%d", data.ndim)
            return None
        np.clip(slc, 0.0, 1.0, out=slc)
        return slc
    except Exception as e:  # noqa: BLE001
        LOG.warning("failed to read risk grid %s: %s", fp, e)
        return None


def load_density_slice(icao: str, date: str, hour: int) -> np.ndarray | None:
    """Return a per-voxel traffic-density slice as a population/noise proxy."""
    fp = paths.airport_dir(icao, kind="processed") / f"traffic_density_{date}.zarr"
    if not fp.exists():
        return None
    try:
        import zarr
    except ImportError:  # pragma: no cover
        return None
    try:
        arr = zarr.open(str(fp), mode="r")
        data = np.asarray(arr)
        if data.ndim == 4:
            T = data.shape[0]
            idx = _slice_index_for_hour(hour, T)
            slc = np.asarray(data[idx]).astype(np.float32, copy=False)
        else:
            slc = data.astype(np.float32, copy=False)
        return slc
    except Exception as e:  # noqa: BLE001
        LOG.warning("failed to read density %s: %s", fp, e)
        return None


# ---------------------------------------------------------------------------
# Coarsening (downsample to a planning resolution)
# ---------------------------------------------------------------------------

def _block_reduce(arr: np.ndarray, factors: tuple[int, int, int], op: str) -> np.ndarray:
    """Block-reduce a 3-D array by integer factors via reshape + reduction."""
    fx, fy, fz = factors
    nx, ny, nz = arr.shape
    nx2 = (nx // fx) * fx
    ny2 = (ny // fy) * fy
    nz2 = (nz // fz) * fz
    arr = arr[:nx2, :ny2, :nz2]
    new_shape = (nx2 // fx, fx, ny2 // fy, fy, nz2 // fz, fz)
    blk = arr.reshape(new_shape)
    if op == "mean":
        return blk.mean(axis=(1, 3, 5))
    if op == "min":
        return blk.min(axis=(1, 3, 5))
    if op == "max":
        return blk.max(axis=(1, 3, 5))
    if op == "all":
        return blk.all(axis=(1, 3, 5))
    raise ValueError(f"unknown block-reduce op: {op}")


def coarsen(
    fine_grid: VoxelGrid,
    *,
    planning_xy_m: float,
    planning_z_m: float,
    sdf: np.ndarray | None = None,
    envelope: np.ndarray | None = None,
    risk: np.ndarray | None = None,
    density: np.ndarray | None = None,
    ofv: np.ndarray | None = None,
) -> Tuple[VoxelGrid, dict[str, np.ndarray | None]]:
    """Block-reduce fine-resolution arrays to a coarser planning grid.

    XY block factor = round(planning_xy_m / fine_grid.dx). Z block factor analogous.
    SDF coarsened by min (worst-case obstacle proximity).
    Envelope coarsened by AND (a coarse cell is clear only if all sub-voxels are clear).
    Risk / density coarsened by mean.
    OFV coarsened by max (a coarse cell is "in OFV" if any sub-voxel is in OFV).
    """
    fx = max(int(round(planning_xy_m / fine_grid.dx)), 1)
    fy = max(int(round(planning_xy_m / fine_grid.dy)), 1)
    fz = max(int(round(planning_z_m / fine_grid.dz)), 1)
    if (fx, fy, fz) == (1, 1, 1):
        return fine_grid, {"sdf": sdf, "envelope": envelope, "risk": risk,
                            "density": density, "ofv": ofv}

    new_nx = fine_grid.shape[0] // fx
    new_ny = fine_grid.shape[1] // fy
    new_nz = fine_grid.shape[2] // fz
    coarse_grid = VoxelGrid(
        x_min=fine_grid.x_min, x_max=fine_grid.x_min + new_nx * fine_grid.dx * fx,
        dx=fine_grid.dx * fx,
        y_min=fine_grid.y_min, y_max=fine_grid.y_min + new_ny * fine_grid.dy * fy,
        dy=fine_grid.dy * fy,
        z_min=fine_grid.z_min, z_max=fine_grid.z_min + new_nz * fine_grid.dz * fz,
        dz=fine_grid.dz * fz,
    )

    out: dict[str, np.ndarray | None] = {}
    out["sdf"] = _block_reduce(sdf, (fx, fy, fz), "min") if sdf is not None else None
    out["envelope"] = _block_reduce(envelope, (fx, fy, fz), "all") if envelope is not None else None
    out["risk"] = _block_reduce(risk, (fx, fy, fz), "mean") if risk is not None else None
    out["density"] = _block_reduce(density, (fx, fy, fz), "mean") if density is not None else None
    out["ofv"] = _block_reduce(ofv, (fx, fy, fz), "max") if ofv is not None else None
    return coarse_grid, out
