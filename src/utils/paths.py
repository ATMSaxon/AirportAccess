"""Project-wide canonical paths.

All modules go through this module so the layout can be relocated without code-spelunking.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
AIRPORTS_CFG = CONFIGS / "airports"
ANNEX14_CFG = CONFIGS / "annex14"
SCENARIOS_CFG = CONFIGS / "scenarios"

DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
PROCESSED = DATA / "processed"

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

REFINE_LOGS = ROOT / "refine-logs"
PAPER = ROOT / "paper"


def airport_dir(icao: str, kind: str = "processed") -> Path:
    """`data/<kind>/<ICAO>/` — auto-creates."""
    base = {"raw": RAW, "cache": CACHE, "processed": PROCESSED, "results": RESULTS,
            "figures": FIGURES, "models": MODELS}[kind]
    out = base / icao
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_dir(run_id: str) -> Path:
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out
