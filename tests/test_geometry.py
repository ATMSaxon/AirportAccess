"""Unit tests for M2 — Annex 14 OLS → 3-D SDF.

All tests run on the synthetic KSYN airport (configs/sanity.yaml) so they're fast and
deterministic. The KSYN runway runs east at lat=0 from (0,0) to ~(3000, 0) ENU.
"""
from __future__ import annotations
import numpy as np
import pytest

from src.utils.config import load_yaml
from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid
from src.utils.paths import CONFIGS
from src.geometry.ols_surfaces import (
    build_airport_surfaces,
    APPROACH, RUNWAY_STRIP, TAKEOFF, INNER_HORIZONTAL,
)
from src.geometry.sdf import build_sdf
from src.geometry.query import SDFQuery


@pytest.fixture(scope="module")
def synthetic():
    cfg = load_yaml(CONFIGS / "sanity.yaml")
    frame = AirportFrame.from_cfg(cfg)
    grid = VoxelGrid.from_airport_cfg(cfg)
    ax14 = load_yaml(CONFIGS / "annex14" / "code4_precision.yaml")
    gdf = build_airport_surfaces(cfg, frame, ax14)
    sdf, meta = build_sdf(gdf, grid)
    q = SDFQuery(sdf, meta["grid_x"], meta["grid_y"], meta["grid_z"])
    return dict(cfg=cfg, frame=frame, grid=grid, gdf=gdf, sdf=sdf, meta=meta, q=q)


# -------------------------------------------------------------------- prisms

def test_prism_count_and_surfaces(synthetic):
    """KSYN has 2 runway records → expect ~9 surfaces per runway + 2 airport-level."""
    gdf = synthetic["gdf"]
    surfaces = set(gdf["surface"].unique())
    assert APPROACH in surfaces
    assert TAKEOFF in surfaces
    assert RUNWAY_STRIP in surfaces
    assert INNER_HORIZONTAL in surfaces
    # 2 runways × (3 approach + 1-2 takeoff + 2 transitional + 1 strip + 1 resa
    #            + 1 ofz_inapp + 2 ofz_intr) = 2 × ~11 = ~22, plus 1 inner-horiz + 1 conical
    assert 18 <= len(gdf) <= 30, f"unexpected prism count {len(gdf)}"


def test_prism_normals_outward(synthetic):
    """Every prism footprint has a CCW exterior (outward-pointing lateral normals)."""
    gdf = synthetic["gdf"]
    bad = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "Polygon":
            if not geom.exterior.is_ccw:
                bad.append(row["name"])
        elif geom.geom_type == "MultiPolygon":
            for p in geom.geoms:
                if not p.exterior.is_ccw:
                    bad.append(row["name"])
                    break
    assert not bad, f"non-CCW prisms: {bad}"


# -------------------------------------------------------------------- SDF tests

def test_runway_centreline_at_z0_has_small_sdf(synthetic):
    """A point on the runway centreline at z=0 has |SDF| < grid spacing."""
    grid = synthetic["grid"]
    q = synthetic["q"]
    val = float(q.clearance_m(1500.0, 0.0, 0.0))
    spacing = max(grid.dx, grid.dy, grid.dz)
    assert abs(val) < spacing, f"|SDF|={abs(val):.2f} not < spacing={spacing}"


def test_high_above_runway_is_clear(synthetic):
    """A point well above the runway (near grid ceiling) is clear (SDF > 0)."""
    q = synthetic["q"]
    grid = synthetic["grid"]
    # KSYN grid z_max = 2000 m. Pick a point near top, above conical+inner_horizontal.
    z = grid.z_max - 1.5 * grid.dz
    val = float(q.clearance_m(1500.0, 0.0, z))
    assert val > 0, f"SDF at high altitude should be positive; got {val}"


def test_inside_approach_near_threshold_is_unclear(synthetic):
    """A point inside the approach surface near runway threshold has SDF < 0."""
    q = synthetic["q"]
    # KSYN runway "09": thr at (0,0), approach extends WEST. Inner edge at x=-60.
    # Section 1 slope 0.02 → at s=240 m (x=-300), z_top ≈ 4.8 m. Pick z=2 m → inside.
    val = float(q.clearance_m(-300.0, 0.0, 2.0))
    assert val < 0, f"expected inside (SDF<0); got {val}"


def test_far_from_airport_is_clear(synthetic):
    """A point well outside all surfaces (corner of grid at altitude) is clear."""
    q = synthetic["q"]
    grid = synthetic["grid"]
    val = float(q.clearance_m(grid.x_max - 1.5 * grid.dx,
                              grid.y_max - 1.5 * grid.dy,
                              grid.z_max - 1.5 * grid.dz))
    assert val > 0, f"corner of grid should be clear; got {val}"


def test_round_trip_random_samples(synthetic):
    """Trilinear interp at cell-centre coords reproduces stored SDF values."""
    sdf = synthetic["sdf"]
    meta = synthetic["meta"]
    q = synthetic["q"]
    rng = np.random.default_rng(42)
    nx, ny, nz = sdf.shape
    ix = rng.integers(0, nx, 100)
    iy = rng.integers(0, ny, 100)
    iz = rng.integers(0, nz, 100)
    xs = meta["grid_x"][ix]
    ys = meta["grid_y"][iy]
    zs = meta["grid_z"][iz]
    stored = sdf[ix, iy, iz].astype(np.float64)
    interp = np.asarray(q.clearance_m(xs, ys, zs), dtype=np.float64)
    np.testing.assert_allclose(stored, interp, atol=1e-3,
                               err_msg="round-trip values differ at cell centres")


def test_sdf_sign_consistency(synthetic):
    """SDF is signed: at least some cells are inside (< 0), most are outside (> 0)."""
    sdf = synthetic["sdf"]
    assert (sdf < 0).any(), "no inside cells — protection union is empty?"
    assert (sdf > 0).any(), "no outside cells — entire grid inside protection?"
    inside_frac = (sdf < 0).mean()
    assert 0.001 < inside_frac < 0.95, f"unrealistic inside fraction {inside_frac:.3f}"


def test_sdf_finite_everywhere(synthetic):
    """SDF has no NaN; finite coverage ≥ 99 %."""
    sdf = synthetic["sdf"]
    finite_frac = float(np.isfinite(sdf).mean())
    assert finite_frac >= 0.99, f"only {finite_frac:.3f} of cells finite"


# -------------------------------------------------------------------- vertiport OFV

def test_vertiport_ofv_funnel_sign(tmp_path):
    """Inside the funnel (small lateral, mid-height above FATO) is negative; outside is positive."""
    from src.utils.config import load_yaml
    from src.utils.crs import AirportFrame
    from src.utils.paths import CONFIGS
    from src.geometry.vertiport_ofv import build_vertiport_ofv
    cfg = load_yaml(CONFIGS / "sanity.yaml")
    ax14 = load_yaml(CONFIGS / "annex14" / "code4_precision.yaml")
    frame = AirportFrame.from_cfg(cfg)
    vid, vp = next(iter(cfg["vertiports"].items()))
    ofv = build_vertiport_ofv(vid, vp, frame, ax14, arp_elev_m=float(cfg["arp"]["elev_m"]))
    sdf = ofv["sdf"]
    gx, gy, gz = ofv["grid_x"], ofv["grid_y"], ofv["grid_z"]
    cx, cy, z_base = ofv["centre"]
    height = ofv["params"]["height"]

    # Inside funnel: directly above FATO at half-height (axial; small radius).
    # Pick the cell closest to (cx, cy, z_base + height/2).
    target_z = z_base + height / 2.0
    ix = int(np.argmin(np.abs(gx - cx)))
    iy = int(np.argmin(np.abs(gy - cy)))
    iz = int(np.argmin(np.abs(gz - target_z)))
    assert sdf[ix, iy, iz] < 0, "centre of funnel should be inside (negative SDF)"

    # Outside funnel: far from centre at low altitude.
    ix_far = int(np.argmin(np.abs(gx - (cx + 1000.0))))
    iy_far = int(np.argmin(np.abs(gy - (cy + 1000.0))))
    iz_low = int(np.argmin(np.abs(gz - (z_base + 10.0))))
    assert sdf[ix_far, iy_far, iz_low] > 0, "far-from-funnel point should be outside (positive SDF)"


# -------------------------------------------------------------------- PrismIndex (runway-config-aware)

def test_prism_index_membership_and_filtered_sdf(synthetic):
    """PrismIndex per-prism membership + filtered SDF behave consistently with the static SDF.

    KSYN runway 09: thr at (0,0), aircraft land coming from the WEST, so the
    approach prism extends to ``x < 0`` and the takeoff-climb prism extends
    east of the stop end (``x > 3000``).
    """
    from src.geometry.query import PrismIndex
    gdf = synthetic["gdf"]
    idx = PrismIndex(gdf)
    rwy_ids = idx.runway_ids()
    assert "09" in rwy_ids and "27" in rwy_ids, f"unexpected runway ids {rwy_ids}"

    # Approach to RWY 09 is WEST of the threshold (x < 0).
    assert idx.point_in_approach_prism(-300.0, 0.0, 2.0, rwy_id="09") is True
    # That same point is NOT in RWY 09's takeoff-climb (which lies east of x=3000).
    assert idx.point_in_departure_prism(-300.0, 0.0, 2.0, rwy_id="09") is False
    # Missed-approach for RWY 09 == takeoff-climb of RWY 09; ~3500 m east of thr is inside.
    assert idx.point_in_missed_approach(3500.0, 0.0, 5.0, rwy_id="09") is True

    # Pick a point inside the RWY 09 approach prism but ABOVE the inner-horizontal
    # ceiling (45 m AGL) so the static-prism set is *clear* of it. At axial 2940 m
    # the section-1 approach top is z = 0.02 * 2940 ≈ 58.8 m, so z=50 m is inside.
    pt = (-3000.0, 0.0, 50.0)
    assert idx.point_in_approach_prism(*pt, rwy_id="09") is True

    # Filtered SDF with active arrivals=09 → negative (point inside approach).
    val = float(idx.sdf_at(*pt, active_arrivals=["09"], active_departures=["09"]))
    assert val < 0, f"point inside RWY 09 approach should give negative filtered SDF; got {val}"

    # Filtered SDF with empty arrivals/departures excludes that approach prism;
    # the same point lies above the inner-horizontal and outside all other static
    # prisms, so it should now be clear (positive).
    val_off = float(idx.sdf_at(*pt, active_arrivals=[], active_departures=[]))
    assert val_off > 0, f"with arrivals={{}} the same point should be clear; got {val_off}"

    # distance_to_active_approach respects the filter — None means all approaches.
    d_all = float(idx.distance_to_active_approach(*pt))
    d_09 = float(idx.distance_to_active_approach(*pt, active_arrivals=["09"]))
    assert d_all == d_09, "single-runway and all-arrivals should match here"
    assert d_09 < 0, f"inside RWY 09 approach → negative distance; got {d_09}"


def test_prism_index_vector_inputs(synthetic):
    """PrismIndex queries accept ndarray inputs and return arrays of matching shape."""
    from src.geometry.query import PrismIndex
    idx = PrismIndex(synthetic["gdf"])
    xs = np.array([-300.0, 1500.0, 5000.0])
    ys = np.array([0.0, 0.0, 0.0])
    zs = np.array([2.0, 0.0, 1500.0])
    membership = idx.point_in_approach_prism(xs, ys, zs, rwy_id="09")
    assert membership.shape == (3,), f"expected shape (3,); got {membership.shape}"
    assert membership[0] == True and membership[2] == False
    sdfs = idx.sdf_at(xs, ys, zs)
    assert sdfs.shape == (3,)
    assert np.all(np.isfinite(sdfs))


def test_prism_index_eval_on_grid_matches_build_sdf(synthetic):
    """`eval_on_grid` with every prism reproduces the global SDF from build_sdf exactly."""
    from src.geometry.query import PrismIndex
    grid = synthetic["grid"]
    idx = PrismIndex(synthetic["gdf"])
    sdf_via_eval = idx.eval_on_grid(grid)               # all prisms
    sdf_via_build = synthetic["sdf"]                    # from build_sdf
    assert sdf_via_eval.shape == sdf_via_build.shape
    # Same primitive on both sides → should match to f32 floating-point precision.
    np.testing.assert_array_equal(sdf_via_eval, sdf_via_build)


def test_prism_index_eval_on_grid_active_subset(synthetic):
    """`eval_on_grid` restricted to a subset matches the global SDF only where those prisms dominate."""
    from src.geometry.query import PrismIndex
    from src.geometry.ols_surfaces import APPROACH, TAKEOFF
    grid = synthetic["grid"]
    idx = PrismIndex(synthetic["gdf"])

    arr_prisms = idx.prisms_for_surface(APPROACH, ["09"])
    dep_prisms = idx.prisms_for_surface(TAKEOFF, ["09"])
    sdf_subset = idx.eval_on_grid(grid, arr_prisms + dep_prisms)

    # Subset SDF is min over a *subset* of the prisms used by the full SDF, so it
    # must be pointwise ≥ the full SDF (allowing f32 round-off slack).
    assert np.all(sdf_subset >= synthetic["sdf"] - 1e-3), \
        "subset SDF must be ≥ full SDF (fewer prisms ⇒ less negative)"

    # Spot check: a voxel deep in RWY 09 section-2 approach (axial ≈ 3 km in,
    # ceiling z_top ≈ 109 m at section-2 slope). z=50 m is comfortably inside.
    gx, gy, gz = synthetic["meta"]["grid_x"], synthetic["meta"]["grid_y"], synthetic["meta"]["grid_z"]
    ix = int(np.argmin(np.abs(gx - (-5000.0))))
    iy = int(np.argmin(np.abs(gy - 0.0)))
    iz = int(np.argmin(np.abs(gz - 50.0)))
    assert sdf_subset[ix, iy, iz] < 0, \
        f"deep-in-approach voxel should be negative; got {sdf_subset[ix, iy, iz]}"

    # A far-corner voxel: outside every approach/takeoff (and the subset contains no
    # static prisms) ⇒ positive.
    ix_far = int(np.argmin(np.abs(gx - (grid.x_max - 1.5 * grid.dx))))
    iy_far = int(np.argmin(np.abs(gy - (grid.y_max - 1.5 * grid.dy))))
    iz_far = int(np.argmin(np.abs(gz - (grid.z_max - 1.5 * grid.dz))))
    assert sdf_subset[ix_far, iy_far, iz_far] > 0


def test_prism_index_eval_on_grid_seeded_out_buffer(synthetic):
    """`out=` seeded buffer unions in additional prisms (the M3 decomposition pattern)."""
    from src.geometry.query import PrismIndex
    from src.geometry.ols_surfaces import APPROACH
    grid = synthetic["grid"]
    idx = PrismIndex(synthetic["gdf"])

    BIG = np.float32(1e9)
    seed = np.full(grid.shape, BIG, dtype=np.float32)

    # Empty subset on an all-positive seed → unchanged.
    out_empty = idx.eval_on_grid(grid, [], out=seed.copy())
    np.testing.assert_array_equal(out_empty, seed)

    # Approach-only subset → introduces negative voxels (the approach footprint exists in the grid).
    arr_prisms = idx.prisms_for_surface(APPROACH, ["09"])
    out_unioned = idx.eval_on_grid(grid, arr_prisms, out=seed.copy())
    assert np.any(out_unioned < 0), "approach prisms should add negative voxels"
    assert np.any(out_unioned < seed - 1e-3), "approach prisms should reduce some voxels"

    # Re-seeding with a baked static SDF and adding the same approach prisms can never
    # *increase* a value (min-reduce monotonicity).
    static_seed = synthetic["sdf"].astype(np.float32, copy=True)
    out_chained = idx.eval_on_grid(grid, arr_prisms, out=static_seed.copy())
    assert np.all(out_chained <= static_seed + 1e-3), "min-reduce can never increase values"
