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

    # Envelope-over-time. Each day's envelope zarr is ~4.2 GB uncompressed bool
    # (T=96 × 600 × 600 × 117). Loading all 5 days in memory + concat exhausts
    # 54 GB Featurize RAM. Skip envelope loading by default (set DREAM_LOAD_ENVELOPES=1
    # to opt back in). Capacity KPI's `corridor_closure_rate` becomes N/A; safety +
    # accessibility KPIs are unaffected since they don't read envelopes_T.
    if os.environ.get("DREAM_LOAD_ENVELOPES", "0") == "1":
        env_paths = sorted(proc.glob("envelope_*.zarr"))
        npz_paths = sorted(p for p in proc.glob("envelope_*.npz") if ".grid" not in p.name)
        slices: list[np.ndarray] = []
        if env_paths:
            try:
                import zarr  # noqa: F401
                for ep in env_paths:
                    z = zarr.open(str(ep), mode="r")
                    if hasattr(z, "__contains__") and "mask" in z:
                        arr = np.asarray(z["mask"])
                    elif hasattr(z, "__contains__") and "envelope" in z:
                        arr = np.asarray(z["envelope"])
                    else:
                        arr = np.asarray(z)
                    if arr.ndim == 4:
                        slices.append(arr)
                    elif arr.ndim == 3:
                        slices.append(arr[np.newaxis])
            except Exception as e:  # noqa: BLE001
                log.warning("envelope zarr load failed: %s", e)
        if npz_paths and not slices:
            for ep in npz_paths:
                try:
                    with np.load(ep, allow_pickle=False) as zz:
                        arr = zz["mask"] if "mask" in zz.files else zz[zz.files[0]]
                        if arr.ndim == 4:
                            slices.append(arr)
                        elif arr.ndim == 3:
                            slices.append(arr[np.newaxis])
                except Exception as e:  # noqa: BLE001
                    log.warning("envelope npz load failed %s: %s", ep, e)
        if slices:
            try:
                out["envelopes_T"] = np.concatenate(slices, axis=0).astype(bool)
                log.info("loaded envelopes_T shape=%s", out["envelopes_T"].shape)
            except Exception as e:  # noqa: BLE001
                log.warning("envelope concat failed: %s", e)
    else:
        log.info("envelope loading skipped (DREAM_LOAD_ENVELOPES=0); "
                 "corridor_closure_rate KPI will be N/A")

    # ADS-B parquet (per-day adsb_<YYYY-MM-DD>.parquet, or single adsb.parquet legacy).
    ads_frames: list[pd.DataFrame] = []
    for ap in sorted(proc.glob("adsb_*.parquet")):
        try:
            ads_frames.append(pd.read_parquet(ap))
        except Exception as e:  # noqa: BLE001
            log.warning("ADS-B load failed %s: %s", ap, e)
    legacy_ads = proc / "adsb.parquet"
    if legacy_ads.exists() and not ads_frames:
        try:
            ads_frames.append(pd.read_parquet(legacy_ads))
        except Exception as e:  # noqa: BLE001
            log.warning("ADS-B legacy load failed: %s", e)
    if ads_frames:
        out["adsb"] = pd.concat(ads_frames, ignore_index=True)
        log.info("loaded ADS-B %d rows across %d file(s)", len(out["adsb"]), len(ads_frames))

    # METAR parquet
    metar_path = proc / "metar.parquet"
    if metar_path.exists():
        try:
            out["metar"] = pd.read_parquet(metar_path)
            log.info("loaded METAR %d rows", len(out["metar"]))
        except Exception as e:  # noqa: BLE001
            log.warning("METAR parquet load failed: %s", e)

    # BTS DB1B parquet — accept `db1b_ond.parquet` (D8) or legacy `bts_db1b.parquet`.
    for bts_candidate in (proc / "db1b_ond.parquet", proc / "bts_db1b.parquet"):
        if bts_candidate.exists():
            try:
                out["bts_od"] = pd.read_parquet(bts_candidate)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("BTS parquet load failed %s: %s", bts_candidate, e)

    # LAWA peaks — accept `peak_hour.parquet` (D7) or legacy `lawa_peaks.csv`.
    for lawa_candidate in (proc / "peak_hour.parquet",
                           proc / "lawa_peaks.parquet",
                           proc / "lawa_peaks.csv"):
        if lawa_candidate.exists():
            try:
                if lawa_candidate.suffix == ".csv":
                    out["lawa_peaks"] = pd.read_csv(lawa_candidate)
                else:
                    out["lawa_peaks"] = pd.read_parquet(lawa_candidate)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("LAWA load failed %s: %s", lawa_candidate, e)

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
