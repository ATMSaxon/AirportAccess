"""Shared publication style for DREAM paper figures.

Loads `configs/paper_style.mplstyle` (Latin Modern + viridis + vector PDF).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_STYLE = ROOT / "configs" / "paper_style.mplstyle"
if _STYLE.exists():
    plt.style.use(str(_STYLE))

# Two-airport palette (consistent across all DREAM figures)
COLORS = {"KLAX": "#3b528b", "KSFO": "#21918c"}
# Baseline palette (viridis-style ramp)
BASELINE_COLORS = {
    "B0": "#440154",
    "B1": "#3b528b",
    "B2": "#21918c",
    "B3": "#5ec962",
    "B4": "#fde725",
}


def save_fig(fig, name: str, fmt: str = "pdf") -> Path:
    p = FIG_DIR / f"{name}.{fmt}"
    fig.savefig(p)
    print(f"  saved {p.relative_to(ROOT)}")
    plt.close(fig)
    return p
