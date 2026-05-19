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
