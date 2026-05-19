#!/usr/bin/env python3
"""Plan eVTOL corridors for an airport across baselines, hours, dates, vertiport pairs.

Usage:

    python scripts/plan_corridors.py \\
        --airport KLAX \\
        --vertiport V1,V2,V3,V4 \\
        --baseline B0,B1,B2,B3,B4 \\
        --hours 8,11,17 \\
        --date 2024-08-02 \\
        --output-dir results/corridors/KLAX

Multi-date convenience: ``--date all-fridays-2024-08`` expands to every Friday in August.
For a synthetic dry-run (no real artefacts), pass ``--synthetic`` to write one corridor
per baseline against the in-memory KSYN problem instead of disk-loaded inputs.
"""
from __future__ import annotations

import argparse
import calendar
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_scenario
from src.utils.io import write_summary
from src.utils.logs import get_logger, setup_logging
from src.utils.paths import RESULTS


def _parse_pairs(vertiport: str | None, pair: str | None) -> list[tuple[str, str]]:
    if pair:
        out: list[tuple[str, str]] = []
        for token in pair.split(","):
            token = token.strip()
            if "->" not in token:
                raise ValueError(f"--pair item {token!r} must be SRC->DST")
            a, b = token.split("->", 1)
            out.append((a.strip(), b.strip()))
        return out
    if not vertiport:
        raise ValueError("either --vertiport or --pair is required")
    verts = [v.strip() for v in vertiport.split(",") if v.strip()]
    # All ordered pairs (src != dst).
    return [(a, b) for a, b in itertools.permutations(verts, 2)]


def _parse_dates(date: str) -> list[str]:
    if not date:
        return [""]
    if date.startswith("all-fridays-"):
        try:
            ym = date.replace("all-fridays-", "")
            year, month = ym.split("-")
            year_i = int(year)
            month_i = int(month)
        except ValueError:
            raise ValueError(f"--date {date!r} should be 'all-fridays-YYYY-MM'")
        n_days = calendar.monthrange(year_i, month_i)[1]
        out = []
        for day in range(1, n_days + 1):
            if calendar.weekday(year_i, month_i, day) == calendar.FRIDAY:
                out.append(f"{year_i:04d}-{month_i:02d}-{day:02d}")
        return out
    return [d.strip() for d in date.split(",") if d.strip()]


def _parse_planning_resolution(s: str) -> tuple[float | None, float | None]:
    if not s:
        return None, None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--planning-resolution {s!r} must be 'XY,Z' metres")
    return float(parts[0]), float(parts[1])


def _run_synthetic(args, log) -> dict:
    """Run a B1+B2+B3+B4 sweep against the synthetic KSYN bundle."""
    from src.planning._synthetic import (
        make_synthetic_inputs,
        synthetic_endpoints,
    )
    from src.planning.astar import Planner, PlannerConfig
    from src.planning.corridor import write_corridor_geojson, write_corridor_json

    out_dir = Path(args.output_dir) if args.output_dir else (RESULTS / "corridors" / "KSYN")
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = make_synthetic_inputs(with_obstacle=True, with_envelope_block=True)
    start, end = synthetic_endpoints()
    cfg = PlannerConfig.from_cfg(load_scenario("cost_weights"))
    if args.strict_turn:
        cfg = PlannerConfig(**{**cfg.__dict__, "strict_turn": True})
    planner = Planner(inputs, cfg)
    baselines = [b.strip() for b in (args.baseline or "B1,B2,B3,B4").split(",")]
    hours = [int(h) for h in (args.hours or "11").split(",")]
    date = (args.date or "synthetic")
    n_ok = 0
    n_total = 0
    for h, b in itertools.product(hours, baselines):
        n_total += 1
        corridor = planner.plan(
            start, end, baseline=b,
            vertiport_pair=("V_close", "V_far"),
            date=date, hour=int(h),
        )
        stem = f"V_close_V_far_{h:02d}_{b}"
        sub = out_dir / date
        sub.mkdir(parents=True, exist_ok=True)
        write_corridor_json(corridor, sub / f"{stem}.json")
        write_corridor_geojson(corridor, sub / f"{stem}.geojson")
        if corridor.feasible:
            n_ok += 1
        log.info("synthetic %s hour=%d feasible=%s pops=%d", b, h, corridor.feasible,
                 corridor.n_expansions)
    summary = {
        "airport": "KSYN",
        "mode": "synthetic",
        "baselines": baselines,
        "hours": hours,
        "date": date,
        "n_total": n_total,
        "n_feasible": n_ok,
        "output_dir": str(out_dir),
    }
    write_summary(out_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--airport", required=True, help="ICAO code (KLAX, KSFO, KSYN)")
    p.add_argument("--vertiport", default=None,
                   help="Comma-separated vertiport IDs; all permutations planned")
    p.add_argument("--pair", default=None,
                   help='Comma-separated explicit pairs, e.g. "V1->V3,V2->V3"')
    p.add_argument("--baseline", default="B0,B1,B2,B3,B4",
                   help="Comma-separated baseline IDs")
    p.add_argument("--hours", default="8,11,17", help="Comma-separated UTC hours")
    p.add_argument("--date", default="2024-08-02",
                   help="Date YYYY-MM-DD, comma-list, or 'all-fridays-YYYY-MM'")
    p.add_argument("--config", default=None,
                   help="Override cost_weights YAML path")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--planning-resolution", default=None,
                   help='Coarsen to "XY_m,Z_m" before planning (default: native grid)')
    p.add_argument("--model", default="xgb",
                   help="Risk-grid model tag (only used by B4)")
    p.add_argument("--strict-turn", action="store_true",
                   help="Enforce hard turn-rate cap (default: soft penalty only)")
    p.add_argument("--no-smooth", action="store_true",
                   help="Skip RDP/Bezier smoothing of the corridor")
    p.add_argument("--synthetic", action="store_true",
                   help="Use the in-memory KSYN synthetic problem (no disk artefacts).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    setup_logging("DEBUG" if args.debug else "INFO")
    log = get_logger("plan_corridors")

    if args.synthetic or args.airport == "KSYN":
        summary = _run_synthetic(args, log)
        log.info("synthetic plan_corridors done: %s", summary)
        return 0

    # Real airport branch (needs disk artefacts to exist).
    from src.planning.astar import PlannerConfig
    from src.planning.corridor import plan_corridors_batch
    from src.planning.loaders import MissingArtifactError

    if args.config:
        from src.utils.config import load_yaml
        cw = load_yaml(args.config)
    else:
        cw = load_scenario("cost_weights")
    cfg = PlannerConfig.from_cfg(cw)
    if args.strict_turn:
        cfg = PlannerConfig(**{**cfg.__dict__, "strict_turn": True})

    pairs = _parse_pairs(args.vertiport, args.pair)
    dates = _parse_dates(args.date)
    hours = [int(h) for h in args.hours.split(",")]
    baselines = [b.strip() for b in args.baseline.split(",")]
    planning_xy, planning_z = _parse_planning_resolution(args.planning_resolution or "")

    out_dir = Path(args.output_dir) if args.output_dir else (RESULTS / "corridors" / args.airport)

    t0 = time.time()
    try:
        written = plan_corridors_batch(
            airport=args.airport,
            vertiport_pairs=pairs,
            dates=dates,
            hours=hours,
            baselines=baselines,
            cfg=cfg,
            model=args.model,
            out_dir=out_dir,
            planning_xy_m=planning_xy,
            planning_z_m=planning_z,
            smooth=not args.no_smooth,
        )
    except MissingArtifactError as e:
        log.error("Cannot plan: %s", e)
        log.error("HINT: run the prerequisite acquisition step first.")
        return 2

    summary = {
        "airport": args.airport,
        "n_corridors": len(written),
        "baselines": baselines,
        "hours": hours,
        "dates": dates,
        "vertiport_pairs": [list(p) for p in pairs],
        "duration_s": round(time.time() - t0, 1),
        "output_dir": str(out_dir),
    }
    write_summary(out_dir, summary)
    log.info("plan_corridors done: %d corridors → %s in %.1fs",
             len(written), out_dir, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
