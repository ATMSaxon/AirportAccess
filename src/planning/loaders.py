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


def _grid_from_traffic_attrs(attrs: dict | None, fallback_cfg: dict) -> VoxelGrid:
    """Build a VoxelGrid from traffic-engineer's envelope ``grid`` attribute dict.

    Shape: ``{"x_min", "x_max", "dx", "y_min", "y_max", "dy", "z_min", "z_max", "dz"}``.
    Falls back to the airport config grid if attrs are absent/malformed.
    """
    if attrs:
        try:
            return VoxelGrid(
                x_min=float(attrs["x_min"]), x_max=float(attrs["x_max"]),
                dx=float(attrs["dx"]),
                y_min=float(attrs["y_min"]), y_max=float(attrs["y_max"]),
                dy=float(attrs["dy"]),
                z_min=float(attrs["z_min"]), z_max=float(attrs["z_max"]),
                dz=float(attrs["dz"]),
            )
        except (KeyError, TypeError, ValueError):
            pass
    return VoxelGrid.from_airport_cfg(fallback_cfg)


def _pick_time_slice(times: np.ndarray | list[str] | None, hour: int, T: int) -> int:
    """Pick the time index whose UTC hour best covers H:30 of `hour`.

    Falls back to ``_slice_index_for_hour`` if `times` is unavailable.
    """
    if times is None or len(times) == 0:
        return _slice_index_for_hour(hour, T)
    try:
        ts = pd.to_datetime([str(t) for t in times], utc=True, errors="coerce")
        target = hour * 60 + 30
        minute_of_day = ts.hour * 60 + ts.minute
        # Closest slice to H:30.
        diff = np.abs(minute_of_day.to_numpy() - target)
        return int(diff.argmin())
    except Exception:  # noqa: BLE001
        return _slice_index_for_hour(hour, T)


def load_envelope_slice(icao: str, date: str, hour: int) -> EnvelopeSlice | None:
    """Load a single 15-minute envelope slice.

    Returns ``None`` if no envelope file exists for the given date (caller falls back to
    A_static). Sets ``dynamic_used=True`` when the load succeeded, ``False`` otherwise.

    Supports two on-disk layouts (per ``src/traffic/SCHEMAS.md``):

    * ``envelope_<date>.zarr`` — zarr group with ``mask`` (T,nx,ny,nz) bool, ``time``
      (T,) UTF strings, and a root ``grid`` attribute dict.
    * ``envelope_<date>.npz``  — fallback npz with key ``mask`` and a sibling
      ``envelope_<date>.grid.json``.
    """
    proc = paths.airport_dir(icao, kind="processed")
    zfp = proc / f"envelope_{date}.zarr"
    nfp = proc / f"envelope_{date}.npz"
    cfg = load_airport(icao)

    if zfp.exists():
        try:
            import zarr  # local import
        except ImportError as e:  # pragma: no cover
            LOG.warning("zarr not importable (%s); cannot load envelope", e)
            return None
        try:
            root = zarr.open(str(zfp), mode="r")
            # Prefer the group's `mask` array; fall back to root array.
            if hasattr(root, "__contains__") and "mask" in root:
                data_arr = root["mask"]
            else:
                data_arr = root
            T = int(data_arr.shape[0])
            times = None
            if hasattr(root, "__contains__") and "time" in root:
                try:
                    times = np.asarray(root["time"])
                except Exception:  # noqa: BLE001
                    times = None
            idx = _pick_time_slice(times, hour, T)
            mask = np.asarray(data_arr[idx]).astype(bool, copy=False)
            # Time
            slice_t: pd.Timestamp | None = None
            if times is not None and idx < len(times):
                try:
                    slice_t = pd.Timestamp(str(times[idx]), tz="UTC")
                except Exception:  # noqa: BLE001
                    slice_t = None
            elif hasattr(root, "attrs"):
                try:
                    time_index = root.attrs.get("time_index")
                    if time_index is not None and idx < len(time_index):
                        slice_t = pd.Timestamp(time_index[idx], tz="UTC")
                except Exception:  # noqa: BLE001
                    pass
            grid_attrs = None
            try:
                grid_attrs = root.attrs.get("grid") if hasattr(root, "attrs") else None
            except Exception:  # noqa: BLE001
                grid_attrs = None
            grid = _grid_from_traffic_attrs(grid_attrs, fallback_cfg=cfg)
            return EnvelopeSlice(grid=grid, mask=mask, dynamic_used=True, slice_time_utc=slice_t)
        except Exception as e:  # noqa: BLE001
            LOG.warning("failed to read envelope zarr %s: %s — trying npz fallback", zfp, e)

    if nfp.exists():
        try:
            with np.load(nfp, allow_pickle=False) as z:
                data = z["mask"] if "mask" in z.files else z[z.files[0]]
                T = int(data.shape[0])
                idx = _slice_index_for_hour(hour, T)
                mask = np.asarray(data[idx]).astype(bool, copy=False)
            grid_attrs = None
            sidecar = proc / f"envelope_{date}.grid.json"
            if sidecar.exists():
                try:
                    grid_attrs = json.loads(sidecar.read_text())
                except Exception:  # noqa: BLE001
                    grid_attrs = None
            grid = _grid_from_traffic_attrs(grid_attrs, fallback_cfg=cfg)
            return EnvelopeSlice(grid=grid, mask=mask, dynamic_used=True, slice_time_utc=None)
        except Exception as e:  # noqa: BLE001
            LOG.warning("failed to read envelope npz %s: %s", nfp, e)
            return None

    LOG.info("envelope file missing for %s %s (%s/%s) — falling back to A_static",
             icao, date, zfp, nfp)
    return None


# ---------------------------------------------------------------------------
# Risk grid
# ---------------------------------------------------------------------------

def load_risk_slice(icao: str, model: str, date: str, hour: int) -> np.ndarray | None:
    """Return the risk field slice (nx,ny,nz) float32 in [0,1], or None if missing.

    Reads ml-engineer's C3 zarr-group layout::

        /rho           (T, NX, NY, NZ) float16 (or uint8-quantised), values in [0, 1]
        /x_m, /y_m, /z_msl_m
        /time_utc_ns   int64 unix nanoseconds (UTC)

    Falls back to a flat-array layout (legacy) if `/rho` is absent.
    """
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
        root = zarr.open(str(fp), mode="r")
        # Group with /rho is the canonical layout.
        if hasattr(root, "__contains__") and "rho" in root:
            rho = root["rho"]
            # Time-slice selection via /time_utc_ns if present, else hour-of-day.
            T = int(rho.shape[0])
            idx = _slice_index_for_hour(hour, T)
            if "time_utc_ns" in root:
                try:
                    t_ns = np.asarray(root["time_utc_ns"])
                    if t_ns.size > 0:
                        # Pick slice whose UTC hour is closest to H:30.
                        ts = pd.to_datetime(t_ns, unit="ns", utc=True)
                        target = hour * 60 + 30
                        mod = ts.hour * 60 + ts.minute
                        diff = np.abs(mod.to_numpy() - target)
                        idx = int(diff.argmin())
                except Exception:  # noqa: BLE001
                    pass
            slc = np.asarray(rho[idx]).astype(np.float32, copy=False)
            # Dequantise: ml-engineer stores either float16 directly (range [0,1])
            # or uint8-quantised (range [0,255] → divide by 255). Heuristic:
            if slc.max() > 1.5:
                slc = slc / 255.0
        else:
            data = np.asarray(root)
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
