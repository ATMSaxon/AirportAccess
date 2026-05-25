#!/usr/bin/env python3
"""TR Part C figures from quick_kpi.py KPI tables — publication-grade styling."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def setup_style():
    style = ROOT / "configs" / "paper_style.mplstyle"
    if style.exists():
        plt.style.use(str(style))


def fig_feasibility_by_baseline(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    g = df.groupby(["airport", "baseline"])["feasible"].mean().reset_index()
    g["feas_pct"] = g["feasible"] * 100
    airports = sorted(g["airport"].unique())
    baselines = ["B0", "B1", "B2", "B3"]
    x = np.arange(len(baselines))
    width = 0.36
    palette = ["#3b528b", "#21918c"]
    for i, ap in enumerate(airports):
        sub = g[g["airport"] == ap].set_index("baseline").reindex(baselines)
        ax.bar(x + (i - 0.5) * width, sub["feas_pct"], width,
               label=ap, color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(baselines)
    ax.set_ylabel("Feasibility (\\%)")
    ax.set_xlabel("Baseline")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", title="Airport")
    ax.set_title("Corridor feasibility across baselines")
    out = out_dir / "fig_feasibility_by_baseline.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_h3_b2_vs_b3(df: pd.DataFrame, out_dir: Path) -> Path:
    pivot = df.pivot_table(index=["airport", "vertiport_src", "vertiport_dst"],
                            columns="baseline", values="feasible", aggfunc="mean")
    pivot["b3_minus_b2"] = pivot["B3"] - pivot["B2"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    palette = ["#3b528b", "#21918c"]
    bins = np.linspace(-0.05, 1.05, 23)
    for i, ap in enumerate(sorted(df["airport"].unique())):
        sub = pivot.xs(ap, level="airport")["b3_minus_b2"]
        ax.hist(sub.values, bins=bins, alpha=0.65,
                label=f"{ap} (n={len(sub)})", color=palette[i % len(palette)])
    ax.set_xlabel("Per-pair feasibility gain: B3 $-$ B2")
    ax.set_ylabel("Number of (V$_{src}$, V$_{dst}$) pairs")
    ax.set_title("H3: dynamic envelope reopens many B2-infeasible pairs")
    ax.legend(loc="upper right")
    out = out_dir / "fig_h3_b2_vs_b3.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_length_per_baseline(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
    palette = ["#3b528b", "#21918c"]
    baselines = ["B1", "B2", "B3"]
    for ax, ap in zip(axes, sorted(df["airport"].unique())):
        sub = df[(df["airport"] == ap) & df["feasible"]]
        data = [sub[sub["baseline"] == b]["length_m"].dropna().values / 1000 for b in baselines]
        bp = ax.boxplot(data, tick_labels=baselines, widths=0.55, patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.2))
        for patch, c in zip(bp["boxes"], palette + ["#fde725"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        ax.set_title(ap)
        ax.set_xlabel("Baseline")
    axes[0].set_ylabel("Corridor length (km)")
    fig.suptitle("Corridor lengths by baseline (feasible only)", y=1.02)
    out = out_dir / "fig_length_per_baseline.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_xgb_metrics(out_dir: Path) -> Path | None:
    import json
    rows = []
    for ap in ("KLAX", "KSFO"):
        for model in ("lr", "rf", "xgb", "mlp"):
            p = ROOT / "results" / "risk" / ap / f"{model}.json"
            if p.exists():
                d = json.loads(p.read_text())
                rows.append({"airport": ap, "model": model,
                              "auroc": d.get("auroc"),
                              "conformal_coverage": d.get("conformal_coverage")})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    models = ["lr", "rf", "xgb", "mlp"]
    airports = sorted(df["airport"].unique())
    x = np.arange(len(models))
    width = 0.36
    palette = ["#3b528b", "#21918c"]
    for i, ap in enumerate(airports):
        sub = df[df["airport"] == ap].set_index("model").reindex(models)
        ax.bar(x + (i - 0.5) * width, sub["auroc"], width, label=ap,
               color=palette[i % len(palette)])
    ax.axhline(0.80, ls="--", c="0.5", lw=0.5, label="Target ≥ 0.80")
    ax.axhline(0.50, ls=":", c="0.7", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylabel("AUROC (held-out day)")
    ax.set_ylim(0.4, 0.85)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Risk-field AUROC (4-day train / 1-day test)")
    out = out_dir / "fig_xgb_metrics.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(ROOT / "figures" / "paper"))
    args = p.parse_args()
    setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for ap in ("KLAX", "KSFO"):
        kp = ROOT / "results" / "eval" / ap / "kpi_table.parquet"
        if kp.exists():
            frames.append(pd.read_parquet(kp))
    if not frames:
        print("No KPI tables found. Run quick_kpi.py first.")
        return 1
    df = pd.concat(frames, ignore_index=True)
    print(f"loaded KPI table: {df.shape}")

    figs = [
        fig_feasibility_by_baseline(df, out_dir),
        fig_h3_b2_vs_b3(df, out_dir),
        fig_length_per_baseline(df, out_dir),
    ]
    f = fig_xgb_metrics(out_dir)
    if f:
        figs.append(f)
    for f in figs:
        print(f"  → {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
