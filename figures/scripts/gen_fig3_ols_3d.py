"""Fig 3 — 3-D rendering of LAX Annex 14 OLS prisms.

Reads `data/processed/KLAX/ols.gpkg` (98 prisms × 9 surface families) and
projects each polygon footprint with its z_low / z_top into a 3-D wireframe.
ARP marker + 8 runway thresholds annotated.

Surface families coloured per family (approach, takeoff-climb, transitional,
inner-horizontal, conical, runway strip, RESA, OFZ inner-approach,
OFZ inner-transitional). Drawn as top-of-prism polygons at z_top (so the
mesh is readable from above).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from figures.scripts._style import save_fig

ICAO = "KLAX"
SURFACE_PALETTE = {
    "approach": "#5ec962",
    "takeoff_climb": "#21918c",
    "transitional": "#3b528b",
    "inner_horizontal": "#440154",
    "conical": "#9c179e",
    "runway_strip": "#fde725",
    "resa": "#ff7f0e",
    "ofz_inner_approach": "#9aef9a",
    "ofz_inner_transitional": "#bccff7",
}


def main() -> int:
    gpkg = ROOT / "data" / "processed" / ICAO / "ols.gpkg"
    g = gpd.read_file(gpkg, layer="ols")

    fig = plt.figure(figsize=(6.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    z_for_top: list[float] = []
    for _, row in g.iterrows():
        poly = row["geometry"]
        if poly is None or poly.is_empty:
            continue
        xs, ys = poly.exterior.coords.xy
        xs = np.asarray(xs, dtype=float) / 1000.0   # km for readability
        ys = np.asarray(ys, dtype=float) / 1000.0
        z_top = float(row.get("z_top_c", 0.0)) or 0.0
        z_for_top.append(z_top)
        verts = [list(zip(xs, ys, np.full_like(xs, z_top)))]
        c = SURFACE_PALETTE.get(row["surface"], "#999999")
        poly3d = Poly3DCollection(verts, alpha=0.18, edgecolor=c, facecolor=c)
        ax.add_collection3d(poly3d)

    # ARP marker
    ax.scatter([0], [0], [0], c="black", marker="*", s=80, label="ARP")

    # Runway thresholds — read from KLAX yaml-like positions in the gpkg
    # Use the gpkg's per-runway strip polygons to find centerlines
    strips = g[g["surface"] == "runway_strip"]
    for _, s in strips.iterrows():
        if s["geometry"] is None or s["geometry"].is_empty:
            continue
        cx = s["geometry"].centroid.x / 1000.0
        cy = s["geometry"].centroid.y / 1000.0
        ax.scatter([cx], [cy], [0], c="red", marker="s", s=8)

    ax.set_xlabel("East from ARP (km)")
    ax.set_ylabel("North from ARP (km)")
    ax.set_zlabel("Height above ARP (m)")
    ax.set_xlim(-18, 18)
    ax.set_ylim(-18, 18)
    ax.set_zlim(0, max(160, (max(z_for_top) if z_for_top else 0) + 20))
    ax.view_init(elev=22, azim=-55)

    # Legend keyed by surface family (proxy artists)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=c, edgecolor=c, alpha=0.4, label=k.replace("_", "-"))
        for k, c in SURFACE_PALETTE.items() if k in g["surface"].unique()
    ]
    handles.append(plt.Line2D([0], [0], marker="*", color="black", linestyle="None",
                               markersize=8, label="ARP"))
    handles.append(plt.Line2D([0], [0], marker="s", color="red", linestyle="None",
                               markersize=5, label="Runway centroid"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=6.5, frameon=False, ncol=1)

    fig.tight_layout()
    save_fig(fig, "fig3_ols_3d_klax")
    return 0


if __name__ == "__main__":
    sys.exit(main())
