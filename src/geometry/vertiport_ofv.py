"""Vertiport Obstacle-Free Volume (OFV) builder.

Per FAA EB-105A, each vertiport has:
  - a FATO square (side `fato_side_m`),
  - an OFV funnel above the FATO that widens linearly from FATO half-side at z=base
    to `ofv_top_radius_m` at z=base+`ofv_height_m`,
  - an approach/departure surface (1:8) — not yet rendered here; the OFV captures the
    *near-vertiport* protected airspace which is what the corridor planner needs.

Each per-vertiport SDF is saved on a *local* high-resolution grid (centred on the
vertiport, expressed in airport-local ENU metres) with the convention:
  SDF < 0  ⇔  inside the OFV funnel  (the only volume where eVTOL terminal-area ops are allowed)
  SDF > 0  ⇔  outside the funnel
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np

from ..utils.crs import AirportFrame
from ..utils.io import write_manifest
from ..utils.logs import get_logger

logger = get_logger(__name__)


def _ofv_sdf_on_grid(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                     cx: float, cy: float, z_base: float,
                     fato_half: float, top_r: float, height: float) -> np.ndarray:
    """Compute funnel SDF on a regular grid (nx, ny, nz)."""
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    lat = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    nx, ny = xx.shape
    nz = len(grid_z)
    sdf = np.empty((nx, ny, nz), dtype=np.float32)
    z_top = z_base + height
    for k, z in enumerate(grid_z):
        z = float(z)
        if z < z_base:
            r_z = fato_half
        elif z > z_top:
            r_z = top_r
        else:
            t = (z - z_base) / height
            r_z = fato_half + (top_r - fato_half) * t
        dr = lat - np.float32(r_z)
        dz_above = np.float32(z - z_top)
        dz_below = np.float32(z_base - z)
        dz = max(float(dz_above), float(dz_below))   # scalar
        if dr.size == 0:
            continue
        inside = (dr <= 0) & (dz <= 0)
        outside_part = np.sqrt(np.maximum(dr, 0.0) ** 2 + max(dz, 0.0) ** 2).astype(np.float32)
        inside_part = np.maximum(dr, np.float32(dz)).astype(np.float32)
        sdf[:, :, k] = np.where(inside, inside_part, outside_part).astype(np.float32)
    return sdf


def build_vertiport_ofv(vid: str, vp: dict, frame: AirportFrame, ax14: dict,
                        arp_elev_m: float,
                        local_half_m: float = 200.0,
                        dx: float = 10.0, dz: float = 10.0,
                        z_top_m: float = 360.0) -> Dict:
    """Compute one vertiport's local-OFV SDF on a small, dense grid.

    Defaults give ±200 m laterally at 10 m cells and ~360 m vertically at 10 m cells —
    a ~40×40×~40 grid that fully resolves the FATO + funnel (FATO half-side 8 m,
    top radius 80 m, funnel height 300 m).
    """
    ofv = ax14["vertiport_ofv"]
    fato_half = float(ofv["fato_side_m"]) / 2.0
    top_r = float(ofv["ofv_top_radius_m"])
    height = float(ofv["ofv_height_m"])

    xv_a, yv_a = frame.wgs_to_enu(np.array([vp["lon"]]), np.array([vp["lat"]]))
    cx = float(xv_a[0])
    cy = float(yv_a[0])
    z_base = float(vp["elev_ft"]) * 0.3048 - float(arp_elev_m)   # AGL above ARP

    nx = int(2 * local_half_m / dx)
    grid_x = cx + (-local_half_m + (np.arange(nx) + 0.5) * dx)
    grid_y = cy + (-local_half_m + (np.arange(nx) + 0.5) * dx)
    # z grid spans [min(0, z_base) - 30, max(z_base+height, z_top_m) + 30]
    z_lo = min(0.0, z_base) - dz
    z_hi = max(z_base + height, z_top_m) + dz
    nz = int(np.ceil((z_hi - z_lo) / dz))
    grid_z = z_lo + (np.arange(nz) + 0.5) * dz

    sdf = _ofv_sdf_on_grid(grid_x, grid_y, grid_z, cx, cy, z_base,
                           fato_half, top_r, height)
    return {
        "vid": vid,
        "sdf": sdf,
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
        "grid_z": grid_z.astype(np.float32),
        "centre": np.array([cx, cy, z_base], dtype=np.float32),
        "params": {"fato_half": fato_half, "top_r": top_r, "height": height},
    }


def build_all_vertiport_ofvs(cfg: dict, frame: AirportFrame, ax14: dict,
                             out_dir: Path) -> List[Path]:
    """Build + save one OFV file per vertiport. Returns list of written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arp_elev_m = float(cfg["arp"]["elev_m"])
    paths: List[Path] = []
    for vid, vp in cfg["vertiports"].items():
        ofv = build_vertiport_ofv(vid, vp, frame, ax14, arp_elev_m)
        path = out_dir / f"ofv_{vid}.npz"
        np.savez_compressed(
            path, sdf=ofv["sdf"],
            grid_x=ofv["grid_x"], grid_y=ofv["grid_y"], grid_z=ofv["grid_z"],
            centre=ofv["centre"],
            fato_half=np.float32(ofv["params"]["fato_half"]),
            top_r=np.float32(ofv["params"]["top_r"]),
            height=np.float32(ofv["params"]["height"]),
        )
        write_manifest(path, source="annex14+vertiport_yaml",
                       params={"vid": vid, "vp_lat": vp["lat"], "vp_lon": vp["lon"],
                               "vp_elev_ft": vp.get("elev_ft"),
                               **ofv["params"]})
        paths.append(path)
        logger.info("OFV %s → %s  (shape %s, inside_frac=%.3f)",
                    vid, path, ofv["sdf"].shape, float((ofv["sdf"] < 0).mean()))
    return paths
