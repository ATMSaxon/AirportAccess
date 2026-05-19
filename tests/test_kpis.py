"""Unit tests for M6/M7 — safety / capacity / accessibility KPIs and the joint table.

All tests use the in-memory synthetic bundle (no disk artefacts touched).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.analysis import (
    accessibility_for_corridor,
    assemble_kpi_table,
    assert_pareto_ranking,
    capacity_for_corridor,
    safety_for_corridor,
)
from src.analysis._synthetic import build_full_synthetic
from src.planning._synthetic import (
    make_synthetic_inputs,
    synthetic_endpoints,
)
from src.planning.astar import Planner, PlannerConfig
from src.planning.corridor import write_corridor_geojson, write_corridor_json
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_safety_kpi_finite():
    """Every field is either finite or explicitly ``None`` (per docstring)."""
    b = build_full_synthetic()
    kpi = safety_for_corridor(
        b.corridor, sdf=b.sdf, grid=b.grid, frame=b.frame,
        airport_cfg=b.airport_cfg, adsb=b.adsb,
    )
    d = kpi.to_dict()
    for k, v in d.items():
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, float):
            assert math.isfinite(v), f"safety.{k} not finite: {v}"
        elif isinstance(v, int):
            assert v >= 0, f"safety.{k} negative: {v}"
    assert 0.0 <= d["ols_violation_rate"] <= 1.0
    assert d["obstacle_margin_min_m"] > 0  # planned through a feasible region


def test_safety_kpi_infeasible_corridor():
    """Infeasible corridor → ols_violation_rate=1.0, ofv_compliance=False, no exceptions."""
    from src.planning.astar import Corridor

    b = build_full_synthetic()
    bad = Corridor(
        feasible=False, baseline="B2", vertiport_pair=("V_close", "V_far"),
        date="2024-08-02", hour=11, notes=["test"], source="synthetic",
    )
    kpi = safety_for_corridor(
        bad, sdf=b.sdf, grid=b.grid, frame=b.frame, airport_cfg=b.airport_cfg,
    )
    assert kpi.ols_violation_rate == 1.0
    assert kpi.ofv_compliance is False


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


def test_capacity_kpi_finite():
    """Three synthetic arrivals on one runway → delay >= 0, throughput in (0, 1.5]."""
    b = build_full_synthetic()
    kpi = capacity_for_corridor(
        b.corridor, adsb_arrivals=b.adsb, envelopes_T=b.envelopes_T,
        airport_cfg=b.airport_cfg, frame=b.frame,
    )
    d = kpi.to_dict()
    assert d["evtol_ops_per_hour"] > 0  # corridor is feasible
    if d["runway_delay_extra_s"] is not None:
        assert d["runway_delay_extra_s"] >= -1e-6
    if d["throughput_preservation"] is not None:
        assert 0.0 < d["throughput_preservation"] <= 1.5


def test_capacity_kpi_with_no_adsb():
    """No ADS-B → runway_delay_extra_s + throughput_preservation are None/0.0; no exception."""
    b = build_full_synthetic()
    kpi = capacity_for_corridor(
        b.corridor, adsb_arrivals=None, envelopes_T=b.envelopes_T,
        airport_cfg=b.airport_cfg, frame=b.frame,
    )
    # Either None or 0.0 (per design) — both are acceptable.
    assert kpi.runway_delay_extra_s in (None, 0.0)
    assert kpi.evtol_ops_per_hour > 0


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------


def test_accessibility_offline():
    """OSRM/BTS/METAR all None → finite floats + weather_reliability_pct is None."""
    b = build_full_synthetic()
    kpi = accessibility_for_corridor(
        b.corridor, airport_cfg=b.airport_cfg,
        metar=None, bts_od=None, lawa_peaks=None, osrm_url=None,
    )
    d = kpi.to_dict()
    for k in (
        "access_time_saving_min_vs_road",
        "passenger_weighted_access_score",
        "vertiport_to_terminal_transfer_min",
        "peak_service_capacity_ops_per_hour",
    ):
        v = d[k]
        assert isinstance(v, float) and math.isfinite(v), f"{k} not finite: {v}"
    assert d["weather_reliability_pct"] is None


def test_accessibility_with_metar_passes_weather():
    """METAR with ceiling=5000ft, vis=10SM at peak hours → reliability_pct ≈ 100."""
    b = build_full_synthetic()
    kpi = accessibility_for_corridor(
        b.corridor, airport_cfg=b.airport_cfg, metar=b.metar,
        bts_od=b.bts_od, lawa_peaks=b.lawa_peaks,
    )
    assert kpi.weather_reliability_pct is not None
    assert kpi.weather_reliability_pct >= 95.0


# ---------------------------------------------------------------------------
# Pareto ranking
# ---------------------------------------------------------------------------


def test_pareto_ranking_monotone_true():
    df = pd.DataFrame([
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B1", "ols_violation_rate": 0.30},
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B2", "ols_violation_rate": 0.20},
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B3", "ols_violation_rate": 0.10},
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B4", "ols_violation_rate": 0.05},
    ])
    assert assert_pareto_ranking(df) is True


def test_pareto_ranking_violation_detected():
    df = pd.DataFrame([
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B1", "ols_violation_rate": 0.10},
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B2", "ols_violation_rate": 0.20},  # WORSE than B1 — bug
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B3", "ols_violation_rate": 0.15},
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B4", "ols_violation_rate": 0.05},
    ])
    assert assert_pareto_ranking(df) is False


def test_pareto_ranking_handles_missing_baselines():
    """A group with only one baseline cannot violate monotonicity."""
    df = pd.DataFrame([
        {"airport": "KLAX", "date": "d", "hour": 11,
         "vertiport_src": "V1", "vertiport_dst": "V3",
         "baseline": "B2", "ols_violation_rate": 0.20},
    ])
    assert assert_pareto_ranking(df) is True


# ---------------------------------------------------------------------------
# Joint assembly
# ---------------------------------------------------------------------------


def test_joint_eval_assembles_table(tmp_path):
    """Plan 4 synthetic corridors (B1–B4 × one pair × one hour) → 4-row joint table."""
    ksyn_cfg = load_yaml(CONFIGS / "sanity.yaml")
    inputs = make_synthetic_inputs(with_obstacle=True)
    start, end = synthetic_endpoints()
    planner = Planner(inputs, PlannerConfig())
    out = tmp_path / "KSYN"
    out.mkdir()
    sub = out / "2024-08-02"
    sub.mkdir()
    n_written = 0
    for b in ("B1", "B2", "B3", "B4"):
        c = planner.plan(start, end, baseline=b,
                         vertiport_pair=("V_close", "V_far"),
                         date="2024-08-02", hour=11)
        stem = f"V_close_V_far_11_{b}"
        write_corridor_json(c, sub / f"{stem}.json")
        write_corridor_geojson(c, sub / f"{stem}.geojson")
        n_written += 1
    df = assemble_kpi_table(
        corridor_dir=out,
        airport_cfg=ksyn_cfg,
        support_artefacts={
            "sdf": inputs.sdf,
            "grid": inputs.grid,
            "frame": inputs.frame,
        },
    )
    assert len(df) == n_written
    for col in (
        "airport", "date", "hour", "vertiport_src", "vertiport_dst", "baseline",
        "ols_violation_rate", "evtol_ops_per_hour", "access_time_saving_min_vs_road",
    ):
        assert col in df.columns
    assert set(df["baseline"].unique()) == {"B1", "B2", "B3", "B4"}
    # ols_violation_rate finite (since SDF was provided).
    assert df["ols_violation_rate"].notna().all()


def test_joint_eval_empty_dir_returns_empty_df(tmp_path):
    ksyn_cfg = load_yaml(CONFIGS / "sanity.yaml")
    df = assemble_kpi_table(tmp_path, ksyn_cfg, support_artefacts=None)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
