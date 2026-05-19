#!/usr/bin/env python3
"""Run safety/capacity/accessibility KPIs against the corridors planned for an airport.

Inputs:
    --airport KLAX
    --corridor-dir results/corridors/KLAX      (default if omitted)
    --output-dir   results/eval/KLAX           (default if omitted)
    --osrm-url     <URL>                       (env OSRM_URL respected if omitted)

Outputs in ``output-dir``:
    kpi_table.parquet      — one row per corridor, all 18 metrics + identity cols
    pareto.png             — safety vs accessibility scatter
    safety_boxplot.png     — per-baseline distribution
    airport_baseline_bar.png — mean safety by airport×baseline
    hour_of_day.png        — mean safety by hour
    summary.json           — overall counts + Pareto verdict + run params

If no support artefacts are found on disk, KPIs that need them return ``None`` / ``NaN``
and the script logs WARNs but does not fail. The Pareto ranking check logs WARN on
violation but always exits 0 (the team-lead decides whether to block on it).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_airport, load_yaml
from src.utils.crs import AirportFrame
from src.utils.io import write_summary
from src.utils.logs import get_logger, setup_logging
from src.utils.paths import CONFIGS, RESULTS, airport_dir


def _load_support_artefacts(icao: str, log) -> dict[str, Any]:
    """Best-effort load of SDF, OFV, envelope-over-time, ADS-B, METAR, BTS, LAWA."""
    out: dict[str, Any] = {}
    if icao == "KSYN":
        cfg = load_yaml(CONFIGS / "sanity.yaml")
    else:
        cfg = load_airport(icao)
    try:
        out["frame"] = AirportFrame.from_cfg(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("AirportFrame.from_cfg failed: %s", e)

    proc = airport_dir(icao, "processed")

    # SDF + grid
    sdf_path = proc / "sdf.npz"
    if sdf_path.exists():
        from src.planning.loaders import load_sdf
        try:
            grid, sdf = load_sdf(icao)
            out["grid"] = grid
            out["sdf"] = sdf
            log.info("loaded SDF %s shape=%s", sdf_path, sdf.shape)
        except Exception as e:  # noqa: BLE001
            log.warning("SDF load failed: %s", e)
    else:
        log.warning("no SDF at %s — safety KPIs will be NaN", sdf_path)

    # Per-vertiport OFV
    ofv: dict[str, np.ndarray] = {}
    for vid in cfg.get("vertiports", {}).keys():
        ofv_path = proc / f"ofv_{vid}.npz"
        if ofv_path.exists():
            try:
                from src.planning.loaders import ofv_mask
                _, mask = ofv_mask(icao, vid)
                ofv[vid] = mask
            except Exception as e:  # noqa: BLE001
                log.warning("OFV load failed for %s: %s", vid, e)
    if ofv:
        out["ofv"] = ofv

    # Envelope-over-time (concat across all envelope_*.zarr in proc).
    env_paths = sorted(proc.glob("envelope_*.zarr"))
    if env_paths:
        try:
            import zarr  # noqa: F401
            slices = []
            for ep in env_paths:
                z = zarr.open(str(ep), mode="r")
                arr = np.asarray(z["envelope"]) if "envelope" in z else np.asarray(z)
                if arr.ndim == 4:
                    slices.append(arr)
                elif arr.ndim == 3:
                    slices.append(arr[np.newaxis])
            if slices:
                out["envelopes_T"] = np.concatenate(slices, axis=0).astype(bool)
                log.info("loaded envelopes_T shape=%s", out["envelopes_T"].shape)
        except Exception as e:  # noqa: BLE001
            log.warning("envelope load failed: %s", e)

    # ADS-B parquet
    ads_path = proc / "adsb.parquet"
    if ads_path.exists():
        try:
            out["adsb"] = pd.read_parquet(ads_path)
            log.info("loaded ADS-B %d rows", len(out["adsb"]))
        except Exception as e:  # noqa: BLE001
            log.warning("ADS-B parquet load failed: %s", e)

    # METAR parquet
    metar_path = proc / "metar.parquet"
    if metar_path.exists():
        try:
            out["metar"] = pd.read_parquet(metar_path)
            log.info("loaded METAR %d rows", len(out["metar"]))
        except Exception as e:  # noqa: BLE001
            log.warning("METAR parquet load failed: %s", e)

    # BTS DB1B parquet
    bts_path = proc / "bts_db1b.parquet"
    if bts_path.exists():
        try:
            out["bts_od"] = pd.read_parquet(bts_path)
        except Exception as e:  # noqa: BLE001
            log.warning("BTS parquet load failed: %s", e)

    # LAWA peaks
    lawa_path = proc / "lawa_peaks.csv"
    if lawa_path.exists():
        try:
            out["lawa_peaks"] = pd.read_csv(lawa_path)
        except Exception as e:  # noqa: BLE001
            log.warning("LAWA CSV load failed: %s", e)

    return out, cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--airport", required=True)
    p.add_argument("--corridor-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--osrm-url", default=os.environ.get("OSRM_URL"))
    p.add_argument("--safety-col", default="ols_violation_rate",
                   help="Column to verify Pareto ranking against")
    p.add_argument("--monotone", default="B1,B2,B3,B4",
                   help="Expected non-increasing-in-this-order baseline sequence")
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)

    setup_logging("DEBUG" if args.debug else "INFO")
    log = get_logger("eval_safety_capacity_access")

    corridor_dir = Path(args.corridor_dir) if args.corridor_dir else (
        RESULTS / "corridors" / args.airport
    )
    if not corridor_dir.exists():
        log.error("No corridor directory: %s", corridor_dir)
        log.error("HINT: run `python scripts/plan_corridors.py --airport %s ...` first.",
                  args.airport)
        return 2
    out_dir = Path(args.output_dir) if args.output_dir else (RESULTS / "eval" / args.airport)
    out_dir.mkdir(parents=True, exist_ok=True)

    support, airport_cfg = _load_support_artefacts(args.airport, log)
    if args.osrm_url:
        support["osrm_url"] = args.osrm_url

    from src.analysis import (
        assemble_kpi_table,
        assert_pareto_ranking,
        make_figures,
    )

    t0 = time.time()
    df = assemble_kpi_table(corridor_dir, airport_cfg, support_artefacts=support)
    kpi_path = out_dir / "kpi_table.parquet"
    try:
        df.to_parquet(kpi_path, index=False)
    except Exception as e:  # noqa: BLE001
        log.warning("parquet write failed (%s); falling back to CSV", e)
        kpi_path = out_dir / "kpi_table.csv"
        df.to_csv(kpi_path, index=False)
    log.info("wrote %s (%d rows, %.1fs)", kpi_path, len(df), time.time() - t0)

    fig_paths: list[Path] = []
    if not df.empty:
        try:
            fig_paths = make_figures(kpi_path, out_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("figure generation failed: %s", e)

    monotone = tuple(b.strip() for b in args.monotone.split(",") if b.strip())
    pareto_ok = assert_pareto_ranking(df, safety_col=args.safety_col, monotone=monotone)
    if not pareto_ok:
        log.warning("Pareto ranking %s NOT monotone in %s — flagged for review",
                    args.safety_col, monotone)

    summary = {
        "airport": args.airport,
        "n_corridors": int(len(df)),
        "corridor_dir": str(corridor_dir),
        "output_dir": str(out_dir),
        "kpi_table": str(kpi_path),
        "figures": [str(p) for p in fig_paths],
        "pareto_ranking_ok": bool(pareto_ok),
        "safety_col": args.safety_col,
        "monotone": list(monotone),
        "duration_s": round(time.time() - t0, 1),
        "support_artefacts_present": {
            k: True for k, v in support.items() if v is not None
        },
    }
    write_summary(out_dir, summary)
    log.info("eval_safety_capacity_access done → %s (pareto_ok=%s)", out_dir, pareto_ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
