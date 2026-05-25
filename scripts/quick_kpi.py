#!/usr/bin/env python3
"""Quick KPI assembly without the slow per-corridor ADS-B nearest-neighbor scan.

The full `scripts/eval_safety_capacity_access.py` runs `safety_for_corridor`
which does an O(N_path × N_ADS-B) `_min_separation` scan — that's ~3.7 sec
per corridor × 720 corridors = ~45 min per airport, plus capacity's SimPy DES.

For the M8 paper figures we don't need min-separation distributions — we need:
- baseline / vertiport / hour / date identity
- feasibility, length, time, n_expansions, cost
- ols_violation_rate (already computed by the planner and stored in the corridor JSON)

This script reads `results/corridors/<ICAO>/*/V*_*_*.json` and emits a
`results/eval/<ICAO>/kpi_table.parquet` with those columns. The Pareto-rank
verdict + figures follow from the parquet.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--airport", required=True)
    p.add_argument("--corridor-dir", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cdir = Path(args.corridor_dir) if args.corridor_dir else (
        ROOT / "results/corridors" / args.airport)
    odir = Path(args.output_dir) if args.output_dir else (
        ROOT / "results/eval" / args.airport)
    odir.mkdir(parents=True, exist_ok=True)

    rows = []
    json_paths = sorted(cdir.rglob("V*.json"))
    for path in json_paths:
        # Skip the manifest siblings.
        if path.name.endswith("_manifest.json"):
            continue
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "baseline" not in d:
            continue
        # Fields we'll keep
        rows.append({
            "airport": args.airport,
            "date": d.get("date"),
            "hour": d.get("hour"),
            "vertiport_src": d.get("vertiport_pair", [None, None])[0],
            "vertiport_dst": d.get("vertiport_pair", [None, None])[1],
            "baseline": d.get("baseline"),
            "feasible": bool(d.get("feasible", False)),
            "length_m": d.get("length_m"),
            "time_s": d.get("time_s"),
            "cost": d.get("cost"),
            "n_expansions": d.get("n_expansions"),
            "energy_j": d.get("energy_j"),
            "dynamic_envelope_used": d.get("dynamic_envelope_used"),
            "risk_used": d.get("risk_used"),
            "ols_violation_rate": d.get("ols_violation_rate", 1.0 if not d.get("feasible") else 0.0),
            "source": d.get("source"),
        })
    df = pd.DataFrame(rows)
    out = odir / "kpi_table.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    if len(df) > 0:
        print(df.groupby(["baseline"]).agg(
            n=("feasible", "size"),
            feas_pct=("feasible", lambda s: 100 * s.mean()),
            mean_len_m=("length_m", "mean"),
            mean_pops=("n_expansions", "mean"),
        ).round(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
