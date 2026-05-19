"""Tiny offline ML-lane smoke run on the synthetic KSYN airport.

Pipeline (no network, no real ADS-B / METAR):
    1. Build a 4-slice synthetic runway-config DataFrame spanning two dates
       (so the temporal holdout in `risk_field.train` has a test day).
    2. Call `counterfactual.sample_and_label(icao='KSYN', n=N, ...)` with
       `adsb_df=None` — labels are driven by geometry alone (axis-cross,
       prism/missed-approach intersection, SDF buffer).
    3. Extract features via `features.extract_features` (no METAR).
    4. Train LR via `risk_field.train(model_name='lr', …)` and persist:
         out_dir/counterfactuals.parquet
         out_dir/features.parquet
         out_dir/ml_sanity.json
    5. Return a small dict the orchestrator uses to mark `ml_ok=True`.

Designed to be quick (≤ ~3 s) and 100 % offline.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

from . import counterfactual as cf
from . import features as feats
from . import risk_field as rf
from ._geom import AirportGeom
from ..utils.io import write_json


def _synthetic_runway_config(icao: str = "KSYN") -> pd.DataFrame:
    """Two-date, 4-slice runway-config table. Activates both runway directions
    so segments near the airport have a high baseline conflict probability."""
    base = pd.Timestamp("2024-08-02T12:00:00Z")
    rows = []
    # Day 1: 12:00, 12:15 UTC. Day 2: 12:00, 12:15 UTC.
    for day_offset in (0, 1):
        for q in (0, 1):
            t = base + pd.Timedelta(days=day_offset) + pd.Timedelta(minutes=15 * q)
            rows.append({
                "time_utc": t,
                "config_id": "WEST_FLOW",
                "active_arrivals": "09",
                "active_departures": "27",
            })
    return pd.DataFrame(rows)


def run(out_dir: Path, airport_cfg: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    icao = str(airport_cfg.get("icao", "KSYN"))

    # 1. Synthetic runway-config (4 slices over 2 dates).
    rc_df = _synthetic_runway_config(icao=icao)

    # 2. Sample + label counterfactuals (offline; no ADS-B).
    n_segments = 256
    cf_path = out_dir / "counterfactuals.parquet"
    seg_df = cf.sample_and_label(
        icao=icao,
        n=n_segments,
        seed=0,
        adsb_df=None,
        runway_config_df=rc_df,
        output_path=cf_path,
    )
    n_conflicts = int(seg_df["conflict"].sum())

    # 3. Features (no ADS-B → traffic_density = 0; no METAR → defaults).
    geom = AirportGeom.from_icao(icao)
    feats_df = feats.extract_features(seg_df, geom, adsb_df=None, metar_df=None)
    feats_df.to_parquet(out_dir / "features.parquet", index=False)

    # 4. Merge features+labels, train LR with the standard pipeline.
    fl = feats.merge_features_labels(feats_df, seg_df)
    # Guard the LR path against single-class data — if conflict rate is 0 % or
    # 100 % the synthetic configuration is broken; surface that clearly.
    if int(fl["conflict"].nunique()) < 2:
        return {
            "ml_ok": False,
            "n_segments": int(len(seg_df)),
            "n_conflicts": n_conflicts,
            "error": "synthetic counterfactual data is single-class",
        }

    result = rf.train(model_name="lr", features_labels=fl, seed=0,
                      conformal_alpha=0.1)

    metrics = dict(result.metrics)
    metrics["model_name"] = result.model_name
    metrics["n_segments"] = int(len(seg_df))
    metrics["n_conflicts"] = n_conflicts
    metrics["conflict_rate"] = float(seg_df["conflict"].mean())
    metrics_path = out_dir / "ml_sanity.json"
    write_json(metrics_path, metrics)

    return {
        "ml_ok": True,
        "n_segments": int(len(seg_df)),
        "n_conflicts": n_conflicts,
        "conflict_rate": float(seg_df["conflict"].mean()),
        "auroc": float(metrics.get("auroc", float("nan"))),
        "aupr": float(metrics.get("aupr", float("nan"))),
        "logloss": float(metrics.get("logloss", float("nan"))),
        "conformal_coverage": float(metrics.get("conformal_coverage", float("nan"))),
        "conformal_width": float(metrics.get("conformal_width", float("nan"))),
        "n_train": int(metrics.get("n_train", 0)),
        "n_test": int(metrics.get("n_test", 0)),
        "outputs": [
            str(cf_path),
            str(out_dir / "features.parquet"),
            str(metrics_path),
        ],
    }
