"""Build OLS GeoPackage + 3-D SDF + per-vertiport OFV for an airport.

Usage:
    python scripts/build_ols.py --airport KLAX
    python scripts/build_ols.py --airport KSFO
    python scripts/build_ols.py --airport KSYN   # synthetic sanity airport

Outputs (in `data/processed/<ICAO>/`):
    ols.gpkg          — OLS prism footprints + z-height params (one layer 'ols')
    sdf.npz           — float32 3-D SDF on the airport voxel grid
    ofv_<VID>.npz     — per-vertiport local-OFV SDF (one per vertiport in the cfg)
    *_manifest.json   — provenance manifest next to each artefact

Convention: SDF is positive *outside* all OLS protection volumes (clear for eVTOL),
negative inside.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

# Allow `python scripts/build_ols.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.utils.config import load_airport, load_annex14, load_yaml
from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid
from src.utils.paths import CONFIGS, airport_dir
from src.utils.io import write_manifest
from src.utils.logs import setup_logging, get_logger
from src.geometry.ols_surfaces import build_airport_surfaces
from src.geometry.sdf import build_sdf, save_sdf
from src.geometry.vertiport_ofv import build_all_vertiport_ofvs


def _load_cfg(icao: str) -> dict:
    """Airports live in configs/airports/<ICAO>.yaml. KSYN lives in configs/sanity.yaml."""
    if icao == "KSYN":
        return load_yaml(CONFIGS / "sanity.yaml")
    return load_airport(icao)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--airport", required=True, help="ICAO code (KLAX, KSFO, KSYN)")
    p.add_argument("--params", default=None,
                   help="Path to an annex14 YAML profile (default: code4_precision)")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--seed", default=42, type=int, help="(reserved; this stage is deterministic)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)
    setup_logging("DEBUG" if args.debug else "INFO")
    log = get_logger("build_ols")

    log.info("M2 build_ols for %s — seed=%d", args.airport, args.seed)

    cfg = _load_cfg(args.airport)
    ax14 = load_yaml(args.params) if args.params else load_annex14("code4_precision")
    frame = AirportFrame.from_cfg(cfg)
    grid = VoxelGrid.from_airport_cfg(cfg)
    out = Path(args.output_dir) if args.output_dir else airport_dir(args.airport, "processed")
    out.mkdir(parents=True, exist_ok=True)

    # 1) Build OLS prisms + GeoPackage
    t0 = time.time()
    gdf = build_airport_surfaces(cfg, frame, ax14)
    gpkg = out / "ols.gpkg"
    # Remove existing layer so re-runs don't conflict
    if gpkg.exists():
        gpkg.unlink()
    gdf.to_file(gpkg, driver="GPKG", layer="ols")
    write_manifest(
        gpkg,
        source="ICAO Annex 14 (Code 4 precision parameterisation) + airport YAML",
        source_url="configs/annex14/code4_precision.yaml",
        params={"airport": args.airport, "n_prisms": int(len(gdf)),
                "n_runways": len(cfg["runways"])},
    )
    log.info("  ols.gpkg  : %d prisms → %s  (%.1fs)", len(gdf), gpkg, time.time() - t0)

    # 2) Build 3-D SDF on the airport voxel grid
    t0 = time.time()
    sdf, meta = build_sdf(gdf, grid)
    sdf_path = out / "sdf.npz"
    save_sdf(sdf_path, sdf, meta)
    inside_frac = float((sdf < 0).mean())
    write_manifest(
        sdf_path,
        source="OLS prisms (this script)",
        source_url=str(gpkg),
        params={
            "airport": args.airport,
            "shape": list(sdf.shape),
            "dx_m": grid.dx, "dy_m": grid.dy, "dz_m": grid.dz,
            "x_range": [float(grid.x_min), float(grid.x_max)],
            "y_range": [float(grid.y_min), float(grid.y_max)],
            "z_range": [float(grid.z_min), float(grid.z_max)],
            "inside_fraction": inside_frac,
            "sdf_min_m": float(sdf.min()), "sdf_max_m": float(sdf.max()),
        },
    )
    log.info("  sdf.npz   : %s → %s  (%.1fs; inside_frac=%.3f, range %.1f..%.1f m)",
             sdf.shape, sdf_path, time.time() - t0, inside_frac,
             float(sdf.min()), float(sdf.max()))

    # 3) Per-vertiport OFV
    t0 = time.time()
    ofv_paths = build_all_vertiport_ofvs(cfg, frame, ax14, out_dir=out)
    log.info("  ofv files : %d → %s  (%.1fs)", len(ofv_paths), out, time.time() - t0)

    log.info("M2 build_ols done for %s.", args.airport)


if __name__ == "__main__":
    main()
