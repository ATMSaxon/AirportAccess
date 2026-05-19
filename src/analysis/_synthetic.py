"""Synthetic helpers for the analysis sanity hook.

The KPI engines are designed to consume real corridor JSONs plus pre-built artefact
arrays (SDF, envelope-over-time, ADS-B parquet, METAR, BTS, LAWA). For the sanity hook
in ``src/analysis/__init__.py::sanity_check`` we don't have those upstream files,
so this module manufactures a *minimal* feasible bundle on the synthetic KSYN airport
using ``src.planning._synthetic``. The bundle is explicitly labelled
``source="synthetic"`` on the corridor so it never silently leaks into a real-data run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid

if TYPE_CHECKING:
    from src.planning.astar import Corridor


@dataclass
class SyntheticBundle:
    """All inputs the three KPI engines need to run end-to-end on KSYN."""

    corridor: "Corridor"
    sdf: np.ndarray
    grid: VoxelGrid
    frame: AirportFrame
    airport_cfg: dict
    adsb: pd.DataFrame
    envelopes_T: np.ndarray
    metar: pd.DataFrame
    bts_od: pd.DataFrame
    lawa_peaks: pd.DataFrame


def _synthetic_adsb(frame: AirportFrame, *, hour: int = 11, n: int = 6) -> pd.DataFrame:
    """Three arrivals approaching the KSYN runway from the east at the requested hour."""
    rng = np.random.default_rng(0)
    base = pd.Timestamp(f"2024-08-02T{hour:02d}:00:00Z")
    rows = []
    for i in range(n):
        # Approach path along the +x axis at ~3000 m altitude, gliding down to threshold.
        x = float(8000.0 - 500.0 * i + rng.normal(0, 50))
        y = float(rng.normal(0, 30))
        z = float(800.0 + 100.0 * i + rng.normal(0, 5))
        rows.append({
            "icao24": f"adsb{i:03d}",
            "time_utc": base + pd.Timedelta(minutes=10 * i),
            "x_m": x,
            "y_m": y,
            "z_msl_m": z,
            "arrival_flag": True,
        })
    return pd.DataFrame(rows)


def _synthetic_envelopes_T(grid: VoxelGrid, *, T: int = 4) -> np.ndarray:
    """Envelope-over-time stub: all-clear for every quarter-hour slice."""
    nx, ny, nz = grid.shape
    arr = np.ones((T, nx, ny, nz), dtype=bool)
    return arr


def _synthetic_metar(hour: int = 11) -> pd.DataFrame:
    """METAR DataFrame with ceiling and visibility — passes the >1000ft+>3SM test."""
    base = pd.Timestamp("2024-08-02T00:00:00Z")
    rows = []
    for h in range(24):
        rows.append({
            "time_utc": base + pd.Timedelta(hours=h),
            "ceiling_ft": 5000.0,
            "visibility_sm": 10.0,
        })
    return pd.DataFrame(rows)


def _synthetic_bts() -> pd.DataFrame:
    """One BTS DB1B-style row so the passenger-weighted KPI gets a non-trivial value."""
    return pd.DataFrame([
        {"origin": "SYN", "dest": "AAA", "pax_count": 50_000},
        {"origin": "AAA", "dest": "SYN", "pax_count": 48_000},
    ])


def _synthetic_lawa() -> pd.DataFrame:
    return pd.DataFrame([{"hour": h, "share": 0.3 if h in (8, 11, 17) else 0.05} for h in range(24)])


def build_full_synthetic(airport_cfg: dict | None = None) -> SyntheticBundle:
    """Plan a feasible corridor on KSYN and return everything the KPI engines need."""
    from src.planning._synthetic import (
        make_synthetic_inputs,
        synthetic_endpoints,
    )
    from src.planning.astar import Planner, PlannerConfig

    if airport_cfg is None:
        from src.utils.config import load_yaml
        from src.utils.paths import CONFIGS

        airport_cfg = load_yaml(CONFIGS / "sanity.yaml")

    inputs = make_synthetic_inputs(with_obstacle=True)
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    corridor = planner.plan(start, end, baseline="B2",
                            vertiport_pair=("V_close", "V_far"),
                            date="2024-08-02", hour=11)

    return SyntheticBundle(
        corridor=corridor,
        sdf=inputs.sdf,
        grid=inputs.grid,
        frame=inputs.frame,
        airport_cfg=airport_cfg,
        adsb=_synthetic_adsb(inputs.frame, hour=11),
        envelopes_T=_synthetic_envelopes_T(inputs.grid, T=4),
        metar=_synthetic_metar(hour=11),
        bts_od=_synthetic_bts(),
        lawa_peaks=_synthetic_lawa(),
    )
