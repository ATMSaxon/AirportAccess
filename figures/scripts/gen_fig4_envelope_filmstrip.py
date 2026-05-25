"""Fig 4 — 4-panel envelope filmstrip for KLAX 2024-08-02 at 1500 ft AGL.

Reads `data/processed/KLAX/envelope_2024-08-02.zarr` (T=96 × 600 × 600 × 117 bool)
and shows the dynamic-closure mask at four representative UTC hours
(8, 11, 17, 23 → arrival-rush / midday / pm peak / night).

For each panel, slice the envelope at z corresponding to ~1500 ft AGL
(z = 457 m above ARP for KLAX), then plot True = clear (yellow) /
False = closed (purple) over the airport ENU.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import zarr
import matplotlib.pyplot as plt

from figures.scripts._style import save_fig

ICAO = "KLAX"
DATE = "2024-08-02"
HOURS_UTC = [8, 11, 17, 23]
TARGET_Z_M = 457.0   # ~1500 ft AGL


def _z_index(grid_z: np.ndarray, z_target: float) -> int:
    return int(np.argmin(np.abs(grid_z - z_target)))


def main() -> int:
    zpath = ROOT / "data" / "processed" / ICAO / f"envelope_{DATE}.zarr"
    if not zpath.exists():
        print(f"missing {zpath}; skipping fig 4")
        return 1
    z = zarr.open(str(zpath), mode="r")

    mask = z["mask"][:]                                  # (T, nx, ny, nz) bool
    t_arr = z["time"][:] if "time" in z else None
    T, nx, ny, nz = mask.shape

    # Build grid_z from airport config
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "airports" / f"{ICAO}.yaml").read_text())
    box = cfg["extract_box_m"]; res = cfg["grid_resolution_m"]
    grid_z = box["z_min"] + (np.arange(nz) + 0.5) * res["z"]
    kz = _z_index(grid_z, TARGET_Z_M)
    xs = np.linspace(-box["half_x"], box["half_x"], nx) / 1000.0
    ys = np.linspace(-box["half_y"], box["half_y"], ny) / 1000.0
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    # Pick slice indices for each requested hour
    # 96 slices = 15 min each; hour h begins at slice h*4.
    slice_indices = [min(T - 1, h * 4) for h in HOURS_UTC]

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.1), sharey=True)
    for ax, h, k in zip(axes, HOURS_UTC, slice_indices):
        # Top-down view: collapse y as image, use imshow with proper extent.
        m = mask[k, :, :, kz].T                                  # (ny, nx)
        ax.imshow(
            m.astype(float),
            origin="lower",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        ax.scatter([0], [0], c="white", marker="*", s=30, edgecolor="black", linewidth=0.4)
        ax.set_title(f"{h:02d}:00 UTC")
        ax.set_xlabel("E km")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
    axes[0].set_ylabel("N km")

    # Single colour bar on the right
    cbar_ax = fig.add_axes([0.93, 0.18, 0.012, 0.65])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, ticks=[0, 1])
    cb.ax.set_yticklabels(["closed", "open"])

    fig.suptitle(f"{ICAO} dynamic envelope at $\\sim$1500 ft AGL, {DATE}", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    save_fig(fig, "fig4_envelope_filmstrip_klax")
    return 0


if __name__ == "__main__":
    sys.exit(main())
