"""Synthetic SDF/envelope/risk inputs for tests and the sanity hook.

These factories build a tiny `PlannerInputs` bundle (30x30x10 voxels at 200m x 200m x 60m)
with optional planted obstacles and envelope blocks. Used by:

* `tests/test_planning.py` to validate A* correctness without real upstream data
* `src/planning/__init__.py::sanity_check` for the M0 smoke run

All synthetic artefacts are explicitly labelled `source=synthetic` in any JSON they leak into.
They are NEVER used for evaluation against real data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid

if TYPE_CHECKING:
    from .astar import PlannerInputs


_SYN_GRID_KW = dict(
    x_min=-3000.0, x_max=3000.0, dx=200.0,
    y_min=-3000.0, y_max=3000.0, dy=200.0,
    z_min=0.0,     z_max=600.0,  dz=60.0,
)


def synthetic_grid() -> VoxelGrid:
    """Return the canonical 30 x 30 x 10 synthetic VoxelGrid."""
    return VoxelGrid(**_SYN_GRID_KW)


def synthetic_frame() -> AirportFrame:
    """Origin-anchored ENU frame for the synthetic airport."""
    return AirportFrame(icao="KSYN", lat0=0.0, lon0=0.0, elev_m=0.0, utm_epsg=32631)


def _corner_ball(shape: tuple[int, int, int], ijk: tuple[int, int, int], radius: int = 2) -> np.ndarray:
    """A boolean ball of voxels around `ijk` clipped to `shape`."""
    i0, j0, k0 = ijk
    nx, ny, nz = shape
    out = np.zeros(shape, dtype=bool)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            for dk in range(-radius, radius + 1):
                if di * di + dj * dj + dk * dk > radius * radius:
                    continue
                i, j, k = i0 + di, j0 + dj, k0 + dk
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    out[i, j, k] = True
    return out


def make_synthetic_inputs(
    *,
    with_obstacle: bool = False,
    with_envelope_block: bool = False,
    with_risk_hotspot: bool = False,
    seed: int = 0,
) -> "PlannerInputs":
    """Build a `PlannerInputs` for the synthetic planning problem.

    Parameters
    ----------
    with_obstacle:
        Plant a 6x6x5-voxel block of negative SDF in the middle of the grid
        (1.2 km x 1.2 km x 300 m in metric units). The A* planner must detour around it.
    with_envelope_block:
        Set ``envelope[:, 14:16, :3] = False`` — a 400 m strip along x at low altitude,
        simulating an "active approach" runway slot. B1 ignores this; B3+ must respect it.
    with_risk_hotspot:
        Place a localised high-risk region in the upper half of the grid (for B4 tests).
    seed:
        RNG seed for any stochastic component (currently unused, reserved for future
        noise on the SDF surface).
    """
    from .astar import PlannerInputs  # avoid circular import

    rng = np.random.default_rng(seed)  # noqa: F841 — reserved for future use
    grid = synthetic_grid()
    frame = synthetic_frame()
    nx, ny, nz = grid.shape

    sdf = np.full((nx, ny, nz), 500.0, dtype=np.float32)
    if with_obstacle:
        # Centre indices (15,15,4); block spans [12:18, 12:18, 2:7].
        sdf[12:18, 12:18, 2:7] = -50.0

    envelope = np.ones((nx, ny, nz), dtype=bool)
    if with_envelope_block:
        envelope[:, 14:16, :3] = False

    risk = np.zeros((nx, ny, nz), dtype=np.float32)
    if with_risk_hotspot:
        risk[10:20, 10:20, 5:10] = 0.7

    density = np.zeros((nx, ny, nz), dtype=np.float32)

    # OFVs: small boolean balls near the two opposite corners.
    ofv_start = _corner_ball((nx, ny, nz), (2, 2, 1), radius=2)
    ofv_end = _corner_ball((nx, ny, nz), (nx - 3, ny - 3, 5), radius=2)

    return PlannerInputs(
        grid=grid,
        frame=frame,
        sdf=sdf,
        envelope=envelope,
        risk=risk,
        density=density,
        ofv_start=ofv_start,
        ofv_end=ofv_end,
        source="synthetic",
    )


def synthetic_endpoints() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Canonical start/end ENU points for the synthetic problem (matches the OFV balls)."""
    grid = synthetic_grid()
    # Voxel-centre coordinates for (2, 2, 1) and (27, 27, 5).
    sx = grid.x_min + (2 + 0.5) * grid.dx
    sy = grid.y_min + (2 + 0.5) * grid.dy
    sz = grid.z_min + (1 + 0.5) * grid.dz
    ex = grid.x_min + (27 + 0.5) * grid.dx
    ey = grid.y_min + (27 + 0.5) * grid.dy
    ez = grid.z_min + (5 + 0.5) * grid.dz
    return (float(sx), float(sy), float(sz)), (float(ex), float(ey), float(ez))
