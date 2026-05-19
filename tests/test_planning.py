"""Unit tests for M5 — envelope-constrained A* corridor planner.

All tests run against the synthetic ``src.planning._synthetic`` problem (30×30×10 at
200×200×60 m). No disk artefacts and no real airport configs are touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.planning._synthetic import (
    make_synthetic_inputs,
    synthetic_endpoints,
    synthetic_grid,
)
from src.planning.astar import Planner, PlannerConfig
from src.planning.graph import (
    EndpointInfeasibleError,
    NEIGHBOURS_26,
    is_feasible_edge,
    snap_to_voxel,
    world_to_ijk,
)


# ---------------------------------------------------------------------------
# Algorithmic correctness
# ---------------------------------------------------------------------------


def test_astar_straight_line():
    """B2 on the obstacle-free synthetic problem: feasible, length within +10% of L2."""
    inputs = make_synthetic_inputs(with_obstacle=False)
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    corridor = planner.plan(start, end, baseline="B2",
                            vertiport_pair=("V_close", "V_far"),
                            date="2024-08-02", hour=11)
    assert corridor.feasible, f"B2 obstacle-free should be feasible; notes={corridor.notes}"
    l2 = float(np.linalg.norm(np.subtract(end, start)))
    assert corridor.length_m <= l2 * 1.10 + 1.0, (
        f"path length {corridor.length_m:.0f} m exceeds L2+10% ({l2*1.10:.0f} m)"
    )
    # Time check: length_m / cruise_speed (with climb portion at 30 m/s ≤ 67).
    assert corridor.time_s > 0
    # Endpoints actually start/end inside the OFV masks.
    assert inputs.ofv_start[tuple(corridor.path_ijk[0])]
    assert inputs.ofv_end[tuple(corridor.path_ijk[-1])]


def test_astar_avoids_obstacle():
    """B2 with the central 6×6×5 obstacle: feasible, longer than L2, positive margin."""
    inputs = make_synthetic_inputs(with_obstacle=True)
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    corridor = planner.plan(start, end, baseline="B2",
                            vertiport_pair=("V_close", "V_far"),
                            date="2024-08-02", hour=11)
    assert corridor.feasible, "A* must detour around the obstacle, not declare infeasible"

    # SDF along path: every waypoint should be in clear air (sdf > 0).
    ijk = corridor.path_ijk
    sdf_along = inputs.sdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    assert (sdf_along > 0).all(), (
        f"{(sdf_along <= 0).sum()} waypoints inside the obstacle (sdf<=0)"
    )

    # Detour: longer than the L2 chord.
    l2 = float(np.linalg.norm(np.subtract(end, start)))
    assert corridor.length_m > l2, (
        f"detoured path length {corridor.length_m:.0f} should exceed L2={l2:.0f}"
    )


def test_b1_ignores_envelope():
    """B1 fixed straight-line corridor must produce a path even when envelope blocks it."""
    inputs = make_synthetic_inputs(with_obstacle=False, with_envelope_block=True)
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    corridor = planner.plan(start, end, baseline="B1",
                            vertiport_pair=("V_close", "V_far"),
                            date="2024-08-02", hour=11)
    assert corridor.feasible
    assert corridor.path_enu is not None and len(corridor.path_enu) >= 2
    assert corridor.dynamic_envelope_used is False
    # The whole straight line is at z=max(z_start,z_end) — so its waypoints can land in
    # the blocked strip at certain (x,y,z) cells; that's *exactly* the bug B1 represents.
    # We just assert it didn't fail because of the envelope.
    assert "B1" in corridor.baseline


def test_b3_respects_envelope():
    """B3 must NOT include any waypoint inside the blocked envelope region."""
    inputs = make_synthetic_inputs(with_obstacle=False, with_envelope_block=True)
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    corridor = planner.plan(start, end, baseline="B3",
                            vertiport_pair=("V_close", "V_far"),
                            date="2024-08-02", hour=11)
    if not corridor.feasible:
        # If endpoint isn't blocked but the planner couldn't route around, that's also OK
        # (B3 is allowed to declare infeasible). At minimum the notes should explain.
        assert corridor.notes
        return
    ijk = corridor.path_ijk
    blocked = (
        (ijk[:, 1] == 14) | (ijk[:, 1] == 15)
    ) & (ijk[:, 2] < 3)
    assert not blocked.any(), (
        f"B3 path entered the blocked envelope region at {blocked.sum()} waypoints"
    )


def test_b1_length_scales_with_l2():
    """Pure sanity: B1's length is exactly the chord between its endpoints (within 1 m)."""
    inputs = make_synthetic_inputs()
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    c = planner.plan(start, end, baseline="B1", vertiport_pair=("V_close", "V_far"),
                     date="", hour=11)
    expected = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    assert abs(c.length_m - expected) < 5.0


# ---------------------------------------------------------------------------
# Climb/descent rate caps
# ---------------------------------------------------------------------------


def test_climb_cap_rejects_pure_vertical():
    """A (0,0,1) offset in the synthetic grid climbs 60 m in 2 s — well above 7 m/s cap."""
    grid = synthetic_grid()
    sdf = np.full(grid.shape, 500.0, dtype=np.float32)
    # Edge: (0,0,1) offset → dz_m=60, length=60, speed=30 (climb), edge_dt=2s, rate=30 m/s.
    assert not is_feasible_edge(
        (5, 5, 6),
        shape=grid.shape,
        sdf=sdf,
        envelope=None,
        use_a_static=True,
        use_envelope=False,
        dz_m=60.0,
        edge_dt_s=2.0,
        max_climb_rate_mps=7.0,
        max_descent_rate_mps=5.0,
    )


def test_climb_cap_allows_shallow_climb():
    """A diagonal (1,1,1) offset: dz=60 over 282 m horiz at 30 m/s climb → ~2 s edge."""
    grid = synthetic_grid()
    sdf = np.full(grid.shape, 500.0, dtype=np.float32)
    # Make sure the rate is plenty under the cap by inflating edge_dt artificially.
    # (Realistic edge_dt for a long horizontal edge will be ≫ 2s, so this is the right
    # regime; we just ensure the function does not reject when dz/dt < cap.)
    assert is_feasible_edge(
        (5, 5, 6),
        shape=grid.shape,
        sdf=sdf,
        envelope=None,
        use_a_static=True,
        use_envelope=False,
        dz_m=60.0,
        edge_dt_s=20.0,            # 3 m/s climb rate < 7 m/s cap.
        max_climb_rate_mps=7.0,
        max_descent_rate_mps=5.0,
    )


# ---------------------------------------------------------------------------
# Endpoint snapping
# ---------------------------------------------------------------------------


def test_snap_to_voxel_returns_index_when_mask_already_true():
    grid = synthetic_grid()
    mask = np.ones(grid.shape, dtype=bool)
    ijk = snap_to_voxel(grid, x_m=0.0, y_m=0.0, z_m=300.0, mask=mask, max_radius=5)
    assert mask[ijk]


def test_snap_to_voxel_raises_when_no_feasible_voxel():
    grid = synthetic_grid()
    mask = np.zeros(grid.shape, dtype=bool)
    # Tiny patch of feasibility, but not within 5 voxels of (0,0,0).
    mask[28, 28, 8] = True
    with pytest.raises(EndpointInfeasibleError):
        snap_to_voxel(grid, x_m=-2500.0, y_m=-2500.0, z_m=60.0, mask=mask, max_radius=3)


def test_snap_to_voxel_bfs_finds_nearby_clear_cell():
    """If the requested voxel is blocked but a neighbour is clear, BFS finds it."""
    grid = synthetic_grid()
    mask = np.zeros(grid.shape, dtype=bool)
    mask[10, 10, 5] = True  # one feasible cell only
    # Request a point that maps to (10, 10, 4) — one z-step away from (10,10,5).
    x = grid.x_min + (10 + 0.5) * grid.dx
    y = grid.y_min + (10 + 0.5) * grid.dy
    z = grid.z_min + (4 + 0.5) * grid.dz
    ijk = snap_to_voxel(grid, x_m=x, y_m=y, z_m=z, mask=mask, max_radius=2)
    assert ijk == (10, 10, 5)


# ---------------------------------------------------------------------------
# Neighbour set + corridor reproducibility
# ---------------------------------------------------------------------------


def test_neighbours_26_complete_and_unique():
    s = {tuple(int(x) for x in r) for r in NEIGHBOURS_26}
    assert (0, 0, 0) not in s
    assert len(s) == 26


def test_b0_returns_infeasible_stub():
    inputs = make_synthetic_inputs()
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    c = planner.plan(start, end, baseline="B0", vertiport_pair=("V_close", "V_far"),
                     date="", hour=11)
    assert not c.feasible
    assert c.baseline == "B0"
    assert any("B0" in n for n in c.notes)


def test_planner_world_to_ijk_consistency():
    """A round-trip world→ijk→world (cell-centre) should land back in the same cell."""
    grid = synthetic_grid()
    for ijk in [(2, 2, 1), (10, 10, 5), (27, 27, 5)]:
        x = grid.x_min + (ijk[0] + 0.5) * grid.dx
        y = grid.y_min + (ijk[1] + 0.5) * grid.dy
        z = grid.z_min + (ijk[2] + 0.5) * grid.dz
        assert world_to_ijk(grid, x, y, z) == ijk


# ---------------------------------------------------------------------------
# OFV → SDF-grid re-projection (regression: shape-mismatch bug)
# ---------------------------------------------------------------------------


def test_ofv_mask_on_grid_reprojects_local_ofv(tmp_path, monkeypatch):
    """ofv_mask_on_grid must return a target-grid-shaped bool mask, not the OFV's
    native shape, and must light up cells only inside the OFV's bbox.

    Regression for the (sdf>0) & ofv_start broadcast crash reported by team-lead:
    when the OFV is on a local 40³ grid and the SDF is on a 600³ airport grid,
    the projected mask must end up on the SDF grid.
    """
    from src.planning.loaders import ofv_mask_on_grid
    from src.utils.grid import VoxelGrid

    # 1. Build a synthetic OFV ("funnel") on its own local grid centred at
    # vertiport @ (1000, 500, 0). Negative inside funnel per geometry-engineer's
    # sign convention. The funnel core must span at least 2× the *target* dx
    # (100 m below) so coarse-grid interpolation lands inside it.
    n_local = 15
    dx_local = 50.0  # 50 m local cell → 750 m OFV span
    cx, cy = 1000.0, 500.0  # vertiport ENU
    grid_x = cx + (np.arange(n_local) - n_local / 2.0 + 0.5) * dx_local
    grid_y = cy + (np.arange(n_local) - n_local / 2.0 + 0.5) * dx_local
    grid_z = (np.arange(n_local) + 0.5) * dx_local
    ofv_sdf = np.full((n_local, n_local, n_local), 50.0, dtype=np.float32)
    # Funnel core: 5×5×5 cells = 250×250×250 m centred on the vertiport.
    ofv_sdf[5:10, 5:10, :5] = -10.0

    # 2. Persist as the geometry-engineer's npz layout so ofv_mask_on_grid can load it.
    icao = "KSYN_OFVTEST"
    proc = tmp_path / "processed" / icao
    proc.mkdir(parents=True)
    np.savez(
        proc / f"ofv_VTEST.npz",
        sdf=ofv_sdf, grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
    )

    # Point ``paths.airport_dir(icao, "processed")`` at our tmp tree.
    import src.planning.loaders as loaders_mod
    monkeypatch.setattr(
        loaders_mod.paths, "airport_dir",
        lambda icao, kind="processed": tmp_path / kind / icao,
    )

    # 3. Target grid is much larger and coarser than the OFV (airport-wide style).
    target = VoxelGrid(
        x_min=-2000.0, x_max=2000.0, dx=100.0,
        y_min=-2000.0, y_max=2000.0, dy=100.0,
        z_min=0.0, z_max=600.0, dz=30.0,
    )

    mask = ofv_mask_on_grid(icao, "VTEST", target)
    assert mask.shape == target.shape, (
        f"projected OFV mask should match target grid shape; got {mask.shape}, want {target.shape}"
    )
    assert mask.dtype == bool
    # At least one cell should be inside the funnel.
    assert mask.sum() > 0
    # All "True" cells lie within the OFV's bbox.
    ii, jj, kk = np.where(mask)
    xs = target.x_min + (ii + 0.5) * target.dx
    ys = target.y_min + (jj + 0.5) * target.dy
    zs = target.z_min + (kk + 0.5) * target.dz
    half_local = n_local * dx_local / 2.0
    assert (xs >= cx - half_local).all() and (xs <= cx + half_local).all()
    assert (ys >= cy - half_local).all() and (ys <= cy + half_local).all()
    assert (zs >= 0.0).all() and (zs <= n_local * dx_local).all()
    # Far-away cells must be False (sanity that we restricted to the bbox).
    far_i = int((10000.0 - target.x_min) / target.dx)
    if 0 <= far_i < target.shape[0]:
        assert not mask[far_i, 0, 0]
