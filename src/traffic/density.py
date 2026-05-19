"""3-D arrival/departure density fields on the airport ``VoxelGrid``.

We binarise each cleaned ADS-B sample (already in ENU) into the airport voxel
grid, weighted by its category (arrival/departure/overflight), then smooth with
a 3-D Gaussian (≈ 500 m horizontal, 60 m vertical defaults). The result is a
density field per category per 15-min slice.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..utils.grid import VoxelGrid
from ..utils.logs import get_logger

log = get_logger(__name__)

GAUSS_SIGMA_XY_M: float = 500.0
GAUSS_SIGMA_Z_M: float = 60.0


@dataclass
class DensityFields:
    arrivals: np.ndarray         # float32 [nx, ny, nz]
    departures: np.ndarray
    overflights: np.ndarray
    all_traffic: np.ndarray
    n_arr_pts: int
    n_dep_pts: int
    n_over_pts: int


def compute_density(adsb: pd.DataFrame,
                    track_categories: pd.DataFrame,
                    grid: VoxelGrid) -> DensityFields:
    """Compute density fields from per-sample observations and per-track categories.

    Parameters
    ----------
    adsb
        Cleaned ADS-B in airport ENU (must have ``icao24, x_m, y_m, z_agl_m``).
    track_categories
        Output of ``classify.classify_tracks`` (one row per ``icao24``).
    grid
        Airport ``VoxelGrid`` (must match the OLS SDF grid).
    """
    # Convert AGL to ENU z (z grid is referenced to the airport reference, see VoxelGrid)
    # The VoxelGrid z-axis runs 0..z_max from field elevation upward — i.e. AGL.
    cats = track_categories.set_index("icao24")["category"].to_dict() if not track_categories.empty else {}

    if adsb.empty:
        zeros = np.zeros(grid.shape, dtype=np.float32)
        return DensityFields(zeros, zeros.copy(), zeros.copy(), zeros.copy(), 0, 0, 0)

    cat_per_row = adsb["icao24"].map(cats).fillna("unknown").to_numpy()
    x = adsb["x_m"].to_numpy(dtype=np.float64)
    y = adsb["y_m"].to_numpy(dtype=np.float64)
    z = adsb["z_agl_m"].to_numpy(dtype=np.float64)

    def _hist(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros(grid.shape, dtype=np.float32)
        ix, iy, iz = grid.world_to_index(x[mask], y[mask], z[mask])
        # In-bound check (grid clips, but we want to drop OOB points to avoid edge piling).
        nx, ny, nz = grid.shape
        in_x = (x[mask] >= grid.x_min) & (x[mask] < grid.x_max)
        in_y = (y[mask] >= grid.y_min) & (y[mask] < grid.y_max)
        in_z = (z[mask] >= grid.z_min) & (z[mask] < grid.z_max)
        keep = in_x & in_y & in_z
        if not keep.any():
            return np.zeros(grid.shape, dtype=np.float32)
        ix, iy, iz = ix[keep], iy[keep], iz[keep]
        flat = np.bincount(np.ravel_multi_index((ix, iy, iz), (nx, ny, nz)),
                           minlength=nx * ny * nz).astype(np.float32)
        return flat.reshape(nx, ny, nz)

    arr_h = _hist(cat_per_row == "arrival")
    dep_h = _hist(cat_per_row == "departure")
    over_h = _hist(cat_per_row == "overflight")

    arr = _gauss_smooth(arr_h, grid)
    dep = _gauss_smooth(dep_h, grid)
    over = _gauss_smooth(over_h, grid)
    return DensityFields(
        arrivals=arr, departures=dep, overflights=over,
        all_traffic=(arr + dep + over).astype(np.float32),
        n_arr_pts=int((cat_per_row == "arrival").sum()),
        n_dep_pts=int((cat_per_row == "departure").sum()),
        n_over_pts=int((cat_per_row == "overflight").sum()),
    )


def _gauss_smooth(hist: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """Anisotropic 3-D Gaussian smoothing of a histogram in voxel units."""
    if hist.sum() == 0:
        return hist.astype(np.float32, copy=False)
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:                       # pragma: no cover
        return hist.astype(np.float32, copy=False)
    sx = GAUSS_SIGMA_XY_M / grid.dx
    sy = GAUSS_SIGMA_XY_M / grid.dy
    sz = GAUSS_SIGMA_Z_M / grid.dz
    return gaussian_filter(hist, sigma=(sx, sy, sz), mode="nearest").astype(np.float32)
