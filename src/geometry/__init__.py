"""Annex 14 OLS geometry, 3-D SDF construction, vertiport OFV.

Public API (stable — see `src/geometry/INTERFACES.md`):

    from src.geometry.ols_surfaces import build_airport_surfaces
    from src.geometry.sdf import build_sdf, save_sdf
    from src.geometry.vertiport_ofv import build_vertiport_ofv, build_all_vertiport_ofvs
    from src.geometry.query import SDFQuery, SurfaceDistance

Sanity hook (for `scripts/run_sanity.py`):

    from src.geometry import sanity_check
    info = sanity_check(out_dir, airport_cfg)
"""
from __future__ import annotations
from pathlib import Path


def sanity_check(out_dir: Path, airport_cfg: dict) -> dict:
    """Offline smoke test for the geometry lane on the synthetic KSYN airport.

    Builds OLS prisms → 3-D SDF → per-vertiport OFV from `airport_cfg` and the shipped
    Code-4-precision annex14 parameterisation; verifies basic SDF properties; and writes
    `ols.gpkg`, `sdf.npz`, `ofv_<VID>.npz` (+ manifests) under ``out_dir``.

    Args:
        out_dir: writable directory for artefacts (auto-created).
        airport_cfg: parsed airport YAML (e.g. `configs/sanity.yaml` for KSYN).

    Returns:
        {"ok": True, "outputs": [paths…], "metrics": {...}} on success. Raises on hard errors.
    """
    # Local imports to keep the top-level package light (and avoid forcing geopandas
    # at import-time when the caller only needs the sanity hook).
    import time
    import numpy as np

    from ..utils.config import load_annex14
    from ..utils.crs import AirportFrame
    from ..utils.grid import VoxelGrid
    from ..utils.io import write_manifest
    from .ols_surfaces import build_airport_surfaces
    from .sdf import build_sdf, save_sdf
    from .vertiport_ofv import build_all_vertiport_ofvs
    from .query import SDFQuery

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ax14 = load_annex14("code4_precision")
    frame = AirportFrame.from_cfg(airport_cfg)
    grid = VoxelGrid.from_airport_cfg(airport_cfg)

    # 1) OLS prisms → GeoPackage
    gdf = build_airport_surfaces(airport_cfg, frame, ax14)
    gpkg = out_dir / "ols.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    gdf.to_file(gpkg, driver="GPKG", layer="ols")
    write_manifest(gpkg, source="annex14 code4_precision + airport YAML",
                   params={"icao": airport_cfg["icao"], "n_prisms": int(len(gdf))})

    # 2) 3-D SDF
    sdf, meta = build_sdf(gdf, grid)
    sdf_path = out_dir / "sdf.npz"
    save_sdf(sdf_path, sdf, meta)
    write_manifest(sdf_path, source="OLS prisms (sanity_check)",
                   params={"icao": airport_cfg["icao"], "shape": list(sdf.shape),
                           "dx_m": grid.dx, "dy_m": grid.dy, "dz_m": grid.dz})

    # 3) Per-vertiport OFV
    ofv_paths = build_all_vertiport_ofvs(airport_cfg, frame, ax14, out_dir=out_dir)

    # 4) Validate basic SDF properties (raise if anything is wrong)
    q = SDFQuery(sdf, meta["grid_x"], meta["grid_y"], meta["grid_z"])
    grid_spacing = max(grid.dx, grid.dy, grid.dz)
    # Runway centreline at z=0 should sit ≈ on the (degenerate) runway-strip prism
    rwy = airport_cfg["runways"][0]
    tx, ty = frame.wgs_to_enu(np.array([rwy["thr_lon"]]), np.array([rwy["thr_lat"]]))
    ex, ey = frame.wgs_to_enu(np.array([rwy["end_lon"]]), np.array([rwy["end_lat"]]))
    cx_rwy = 0.5 * (float(tx[0]) + float(ex[0]))
    cy_rwy = 0.5 * (float(ty[0]) + float(ey[0]))
    sdf_centreline = float(q.clearance_m(cx_rwy, cy_rwy, 0.0))
    if abs(sdf_centreline) >= grid_spacing:
        raise RuntimeError(f"runway centreline SDF |{sdf_centreline:.2f}| ≥ spacing {grid_spacing:.2f}")

    # High altitude above runway should be clear (positive SDF)
    z_high = grid.z_max - 1.5 * grid.dz
    sdf_high = float(q.clearance_m(cx_rwy, cy_rwy, z_high))
    if sdf_high <= 0:
        raise RuntimeError(f"high-altitude SDF {sdf_high:.2f} ≤ 0 (should be clear)")

    inside_frac = float((sdf < 0).mean())
    finite_frac = float(np.isfinite(sdf).mean())
    if finite_frac < 0.99:
        raise RuntimeError(f"only {finite_frac:.3f} of SDF cells are finite")
    if not (0.0 < inside_frac < 0.95):
        raise RuntimeError(f"inside fraction {inside_frac:.3f} outside (0, 0.95)")

    outputs = [str(gpkg), str(sdf_path)] + [str(p) for p in ofv_paths]
    metrics = {
        "n_prisms": int(len(gdf)),
        "sdf_shape": list(sdf.shape),
        "sdf_min_m": float(sdf.min()),
        "sdf_max_m": float(sdf.max()),
        "inside_fraction": inside_frac,
        "finite_fraction": finite_frac,
        "sdf_centreline_at_z0_m": sdf_centreline,
        "sdf_high_alt_m": sdf_high,
        "grid_dx_m": grid.dx, "grid_dy_m": grid.dy, "grid_dz_m": grid.dz,
        "n_ofv": len(ofv_paths),
        "elapsed_s": float(time.time() - t0),
    }
    return {"ok": True, "outputs": outputs, "metrics": metrics}
