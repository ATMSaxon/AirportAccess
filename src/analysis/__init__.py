"""Safety / capacity / accessibility KPIs and Pareto-ranking baseline comparison.

Public API:

* ``safety_for_corridor`` / ``SafetyKPI``        — eight safety metrics.
* ``capacity_for_corridor`` / ``CapacityKPI``    — DES-based capacity metrics.
* ``accessibility_for_corridor`` / ``AccessibilityKPI`` — passenger-weighted access.
* ``KPIResult``, ``assemble_kpi_table``, ``assert_pareto_ranking``, ``make_figures``
                                                — joint table + figures + ranking check.
* ``sanity_check(out_dir, airport_cfg) -> dict`` — exercised by ``scripts/run_sanity.py``.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import write_json
from src.utils.logs import get_logger

from .accessibility_kpis import AccessibilityKPI, accessibility_for_corridor
from .capacity_kpis import CapacityKPI, capacity_for_corridor
from .joint_eval import (
    KPIResult,
    assemble_kpi_table,
    assert_pareto_ranking,
    make_figures,
)
from .safety_kpis import SafetyKPI, safety_for_corridor

LOG = get_logger(__name__)

__all__ = [
    "AccessibilityKPI",
    "accessibility_for_corridor",
    "CapacityKPI",
    "capacity_for_corridor",
    "KPIResult",
    "assemble_kpi_table",
    "assert_pareto_ranking",
    "make_figures",
    "SafetyKPI",
    "safety_for_corridor",
    "sanity_check",
]


def sanity_check(out_dir: Path, airport_cfg: dict) -> dict[str, Any]:
    """End-to-end smoke test for the three KPI engines + joint Pareto check.

    Plans a synthetic KSYN corridor via ``src/planning``, then runs safety/capacity/
    accessibility KPIs against the in-memory ``SyntheticBundle``. Returns a dict that
    ``scripts/run_sanity.py`` uses to set ``analysis_ok=True``.
    """
    from ._synthetic import build_full_synthetic

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_full_synthetic(airport_cfg)
    corridor = bundle.corridor

    safety_kpi = safety_for_corridor(
        corridor,
        sdf=bundle.sdf,
        grid=bundle.grid,
        frame=bundle.frame,
        airport_cfg=bundle.airport_cfg,
        adsb=bundle.adsb,
    )
    capacity_kpi = capacity_for_corridor(
        corridor,
        adsb_arrivals=bundle.adsb,
        envelopes_T=bundle.envelopes_T,
        airport_cfg=bundle.airport_cfg,
        frame=bundle.frame,
    )
    access_kpi = accessibility_for_corridor(
        corridor,
        airport_cfg=bundle.airport_cfg,
        metar=bundle.metar,
        bts_od=bundle.bts_od,
        lawa_peaks=bundle.lawa_peaks,
    )

    # Build a tiny KPI table (two synthetic baselines so Pareto check has something to compare).
    rows: list[dict] = []
    for label, c in (("B1", _stub_b1(corridor)), ("B2", corridor)):
        rows.append({
            "airport": bundle.airport_cfg.get("icao", "KSYN"),
            "date": c.date,
            "hour": int(c.hour),
            "vertiport_src": c.vertiport_pair[0],
            "vertiport_dst": c.vertiport_pair[1],
            "baseline": label,
            "feasible": bool(c.feasible),
            "ols_violation_rate": 0.10 if label == "B1" else 0.00,
            "access_time_saving_min_vs_road": access_kpi.access_time_saving_min_vs_road,
        })
    df = pd.DataFrame(rows)
    pareto_ok = assert_pareto_ranking(df, safety_col="ols_violation_rate",
                                      monotone=("B1", "B2"))

    out_json = out_dir / "summary_analysis.json"
    summary = {
        "analysis_ok": bool(corridor.feasible),
        "source": "synthetic",
        "safety": safety_kpi.to_dict(),
        "capacity": capacity_kpi.to_dict(),
        "accessibility": access_kpi.to_dict(),
        "pareto_ranking_ok": bool(pareto_ok),
    }
    write_json(out_json, summary)
    LOG.info("analysis sanity → %s (feasible=%s, pareto=%s)",
             out_json, corridor.feasible, pareto_ok)
    return summary


def _stub_b1(corridor) -> "Corridor":  # noqa: F821 -- forward ref
    """Return a copy of ``corridor`` re-labelled as ``B1`` for the sanity Pareto check.

    The ``ols_violation_rate`` values used in ``sanity_check`` are synthetic (0.10 vs 0.00)
    so this stub doesn't need to *actually* model what a B1 corridor would do.
    """
    from copy import deepcopy

    c = deepcopy(corridor)
    c.baseline = "B1"
    return c
