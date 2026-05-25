"""3-D voxel-graph helpers for the envelope-constrained A* planner.

Exposes:

* ``NEIGHBOURS_26``    — (26, 3) int8 offsets for 26-connectivity (no zero-offset).
* ``is_feasible_edge`` — feasibility test for a single edge (bounds, A_static, envelope,
                         climb/descent rate caps).
* ``snap_to_voxel``    — WGS lat/lon/z to a feasible grid-index, searching outward up to a
                         configurable radius with a small BFS through the supplied mask.
* ``EndpointInfeasibleError`` — raised when no feasible voxel can be found.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid


class EndpointInfeasibleError(ValueError):
    """Raised when a corridor endpoint cannot be snapped to a feasible voxel."""


# Pre-computed 26-connectivity offsets (excluding (0,0,0)).
NEIGHBOURS_26 = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ],
    dtype=np.int8,
)
assert NEIGHBOURS_26.shape == (26, 3)


def in_bounds(ijk: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    i, j, k = ijk
    return 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]


def edge_geom(d: np.ndarray, grid: VoxelGrid) -> tuple[float, float, float]:
    """Return (horiz_m, dz_m, length_m) for the offset row `d = (di,dj,dk)`."""
    dx_m = float(d[0]) * grid.dx
    dy_m = float(d[1]) * grid.dy
    dz_m = float(d[2]) * grid.dz
    horiz_m = float(np.hypot(dx_m, dy_m))
    length_m = float(np.hypot(horiz_m, dz_m))
    return horiz_m, dz_m, length_m


def is_feasible_edge(
    target_ijk: tuple[int, int, int],
    *,
    shape: tuple[int, int, int],
    sdf: np.ndarray | None,
    envelope: np.ndarray | None,
    use_a_static: bool,
    use_envelope: bool,
    dz_m: float,
    edge_dt_s: float,
    max_climb_rate_mps: float,
    max_descent_rate_mps: float,
    static_closure: np.ndarray | None = None,
    use_static_closure: bool = False,
) -> bool:
    """Hard-reject infeasible edges.

    * `target_ijk` must be in bounds.
    * If `use_a_static`, `sdf[target] > 0`.
    * If `use_envelope`, `envelope[target] is True`.
    * If `use_static_closure` AND `static_closure` is provided, `static_closure[target]`
      must be False (i.e. the voxel is NOT inside the pessimistic both-sides-closed
      runway corridor used by baseline B2). Missing `static_closure` is treated as
      "no closure" so the synthetic problem (no runways) still works.
    * Climb-rate / descent-rate caps respected.
    """
    if not in_bounds(target_ijk, shape):
        return False
    i, j, k = target_ijk
    if use_a_static:
        if sdf is None or sdf[i, j, k] <= 0:
            return False
    if use_envelope:
        if envelope is None or not envelope[i, j, k]:
            return False
    if use_static_closure and static_closure is not None:
        if static_closure[i, j, k]:
            return False
    if edge_dt_s > 0:
        rate = dz_m / edge_dt_s
        if rate > max_climb_rate_mps:
            return False
        if rate < -max_descent_rate_mps:
            return False
    return True


def angle_between_offsets(a: np.ndarray, b: np.ndarray, grid: VoxelGrid) -> float:
    """Return the angle in radians between two voxel-offset vectors in metric ENU space."""
    av = np.array([a[0] * grid.dx, a[1] * grid.dy, a[2] * grid.dz], dtype=np.float64)
    bv = np.array([b[0] * grid.dx, b[1] * grid.dy, b[2] * grid.dz], dtype=np.float64)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip(av.dot(bv) / (na * nb), -1.0, 1.0))
    return float(np.arccos(cos))


def world_to_ijk(grid: VoxelGrid, x: float, y: float, z: float) -> tuple[int, int, int]:
    """Convert ENU metres to voxel indices clipped to grid bounds."""
    ix = int(np.clip(np.floor((x - grid.x_min) / grid.dx), 0, grid.shape[0] - 1))
    iy = int(np.clip(np.floor((y - grid.y_min) / grid.dy), 0, grid.shape[1] - 1))
    iz = int(np.clip(np.floor((z - grid.z_min) / grid.dz), 0, grid.shape[2] - 1))
    return ix, iy, iz


def ijk_to_world(grid: VoxelGrid, ijk: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return ENU voxel-centre metres for an `(i,j,k)` index."""
    i, j, k = ijk
    return (
        grid.x_min + (i + 0.5) * grid.dx,
        grid.y_min + (j + 0.5) * grid.dy,
        grid.z_min + (k + 0.5) * grid.dz,
    )


def snap_to_voxel(
    grid: VoxelGrid,
    *,
    x_m: float,
    y_m: float,
    z_m: float,
    mask: np.ndarray | None = None,
    max_radius: int = 5,
) -> tuple[int, int, int]:
    """Snap an ENU point to its nearest feasible voxel via BFS through ``mask``.

    Parameters
    ----------
    mask:
        Boolean (nx,ny,nz). A voxel is feasible iff ``mask[i,j,k] is True``. If ``None``,
        the raw index is returned (no feasibility check). Typical use: pass
        ``(sdf > 0) & ofv_mask`` for endpoints.
    max_radius:
        BFS expansion radius in voxels. 5 -> up to ~5*dx in metric space.
    """
    ijk0 = world_to_ijk(grid, x_m, y_m, z_m)
    if mask is None or mask[ijk0]:
        return ijk0

    shape = grid.shape
    visited = np.zeros(shape, dtype=bool)
    visited[ijk0] = True
    q: deque[tuple[int, int, int, int]] = deque([(ijk0[0], ijk0[1], ijk0[2], 0)])
    while q:
        i, j, k, r = q.popleft()
        if r > max_radius:
            break
        if mask[i, j, k]:
            return (i, j, k)
        for d in NEIGHBOURS_26:
            ni, nj, nk = i + int(d[0]), j + int(d[1]), k + int(d[2])
            if not in_bounds((ni, nj, nk), shape):
                continue
            if visited[ni, nj, nk]:
                continue
            visited[ni, nj, nk] = True
            q.append((ni, nj, nk, r + 1))
    raise EndpointInfeasibleError(
        f"No feasible voxel within {max_radius}-voxel radius of "
        f"({x_m:.1f}, {y_m:.1f}, {z_m:.1f}) m ENU."
    )


def vertiport_anchor_enu(
    frame: AirportFrame, vertiport_cfg: dict, *, ground_clearance_m: float = 30.0
) -> tuple[float, float, float]:
    """Convert a vertiport YAML entry to an ENU anchor point.

    `ground_clearance_m` is the AGL hover-out altitude where the corridor starts; we add it
    on top of the vertiport's MSL elevation derived from `elev_ft`.
    """
    lon, lat = float(vertiport_cfg["lon"]), float(vertiport_cfg["lat"])
    elev_ft = float(vertiport_cfg.get("elev_ft", 0.0))
    elev_m = elev_ft * 0.3048
    x, y = frame.wgs_to_enu(np.array([lon]), np.array([lat]))
    z = float(elev_m + ground_clearance_m)
    return (float(x[0]), float(y[0]), z)
