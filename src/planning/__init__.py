"""Envelope-constrained A* corridor planner for eVTOL.

Public API:

* ``Planner``, ``PlannerConfig``, ``PlannerInputs``, ``Corridor`` — algorithmic core.
* ``plan_corridor``, ``plan_corridors_batch`` — disk-aware orchestrator.
* ``write_corridor_geojson``, ``write_corridor_json`` — I/O writers.
* ``EndpointInfeasibleError``, ``MissingArtifactError`` — production-script errors.
* ``sanity_check`` — exercised by ``scripts/run_sanity.py``.
"""
from __future__ import annotations

from pathlib import Path

from src.utils.logs import get_logger

from .astar import Corridor, Planner, PlannerConfig, PlannerInputs
from .corridor import (
    plan_corridor,
    plan_corridors_batch,
    write_corridor_geojson,
    write_corridor_json,
)
from .graph import EndpointInfeasibleError
from .loaders import MissingArtifactError

LOG = get_logger(__name__)

__all__ = [
    "Corridor",
    "Planner",
    "PlannerConfig",
    "PlannerInputs",
    "plan_corridor",
    "plan_corridors_batch",
    "write_corridor_geojson",
    "write_corridor_json",
    "EndpointInfeasibleError",
    "MissingArtifactError",
    "sanity_check",
]


def sanity_check(out_dir: Path, airport_cfg: dict) -> dict:
    """End-to-end smoke test on a tiny synthetic SDF (no disk I/O).

    Builds the canonical 30 x 30 x 10 synthetic problem with a 6 x 6 x 5 obstacle, runs
    A* under baseline B2, and writes the corridor JSON to ``out_dir``. Returns a dict
    consumed by ``scripts/run_sanity.py`` so it can mark ``planning_ok=True``.
    """
    from . import _synthetic
    from .corridor import write_corridor_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _synthetic.make_synthetic_inputs(with_obstacle=True)
    start_enu, end_enu = _synthetic.synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    corridor = planner.plan(
        start_enu, end_enu, baseline="B2",
        vertiport_pair=("V_close", "V_far"),
        date="0000-00-00",
        hour=0,
    )
    corridor_path = out_dir / "synthetic_corridor.json"
    write_corridor_json(corridor, corridor_path)
    return {
        "planning_ok": bool(corridor.feasible),
        "baseline": "B2",
        "n_path_pts": int(corridor.path_enu.shape[0]) if corridor.feasible else 0,
        "time_s": float(corridor.time_s),
        "length_m": float(corridor.length_m),
        "n_expansions": int(corridor.n_expansions),
        "output": str(corridor_path),
    }
