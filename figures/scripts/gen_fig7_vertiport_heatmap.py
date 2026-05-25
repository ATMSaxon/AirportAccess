"""Fig 7 — Per-vertiport-pair feasibility heatmap, B2 vs B3, LAX vs SFO.

For each airport and baseline, compute mean feasibility over all (date, hour)
for each (V_src, V_dst) pair. Plot a 2×2 grid of heatmaps:

    LAX × B2  |  LAX × B3
    -----------+----------
    SFO × B2  |  SFO × B3

The pattern of B3 lifting V3-rooftop pairs is the qualitative H3 / H4 evidence.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures.scripts._style import save_fig


def _heatmap(ax, df: pd.DataFrame, baseline: str, vertiports: list[str], title: str):
    sub = df[df["baseline"] == baseline]
    pivot = sub.pivot_table(
        index="vertiport_src", columns="vertiport_dst",
        values="feasible", aggfunc="mean",
    ).reindex(index=vertiports, columns=vertiports)
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(vertiports)))
    ax.set_xticklabels(vertiports)
    ax.set_yticks(range(len(vertiports)))
    ax.set_yticklabels(vertiports)
    ax.set_xlabel("V$_{dst}$")
    ax.set_ylabel("V$_{src}$")
    ax.set_title(title)
    for i in range(len(vertiports)):
        for j in range(len(vertiports)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                txt = "—" if i == j else f"{v*100:.0f}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    return im


def main() -> int:
    frames = {}
    for ap in ("KLAX", "KSFO"):
        path = ROOT / "results" / "eval" / ap / "kpi_table.parquet"
        if path.exists():
            frames[ap] = pd.read_parquet(path)
    if not frames:
        print("No KPI tables — run quick_kpi.py first")
        return 1

    vertiports = ["V1", "V2", "V3", "V4"]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))
    im = None
    for i, ap in enumerate(("KLAX", "KSFO")):
        for j, b in enumerate(("B2", "B3")):
            im = _heatmap(axes[i, j], frames[ap], b, vertiports,
                          f"{ap} · {b}  (% feasible)")
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_yticklabels(["0", "25", "50", "75", "100"])
    cb.set_label("Feasibility (%)")
    fig.tight_layout(rect=[0, 0, 0.92, 1.0])
    save_fig(fig, "fig7_vertiport_heatmap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
