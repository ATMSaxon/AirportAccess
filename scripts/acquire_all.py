#!/usr/bin/env python3
"""Orchestrate the eight DREAM data sources for one airport + window.

Usage:
    python scripts/acquire_all.py --airport KLAX --window 2024-08
    python scripts/acquire_all.py --airport KSFO --window 2024-08 --skip opensky

Each source is run independently; any failure is captured to a `<source>.OFFLINE.json`
sibling and does not abort the run. Writes
`data/processed/<ICAO>/_inventory.json` summarising what came back.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when this script is invoked directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (
    source_adsblol, source_bts, source_faa_dof, source_faa_nasr, source_lawa,
    source_noaa_wx, source_opensky, source_osm, source_usgs_3dep,
)
from src.data._common import FetchResult, write_offline
from src.utils import io as io_utils
from src.utils import paths as path_utils
from src.utils.config import load_airport
from src.utils.logs import get_logger, setup_logging

logger = get_logger(__name__)

# (name, module, offline-recovery checklist)
SOURCES: list[tuple[str, object, list[str]]] = [
    ("faa_nasr", source_faa_nasr, [
        "Verify configs/airports/<ICAO>.yaml has a runways block.",
        "If the YAML is empty, download FAA NFDC NASR ZIP and rerun.",
    ]),
    ("faa_dof", source_faa_dof, [
        "Manually download https://aeronav.faa.gov/Obst_Data/DAILY_DOF.ZIP",
        "Drop it under data/cache/faa_dof/DAILY_DOF.ZIP",
        "Re-run scripts/acquire_all.py",
    ]),
    ("usgs_3dep", source_usgs_3dep, [
        "Manually browse https://apps.nationalmap.gov/downloader/",
        "Download 1/3 arc-second tiles covering the ARP-centred 60 km box.",
        "Drop the .tif files under data/cache/usgs_3dep/<ICAO>/ and re-run.",
    ]),
    ("osm", source_osm, [
        "Overpass servers throttle aggressively. Wait and re-run, or",
        "Use https://overpass-turbo.eu/ to export the same bbox and drop GeoJSON.",
    ]),
    ("adsblol", source_adsblol, [
        "Large download (~2.5 GB/day) from github.com/adsblol/globe_history_2024.",
        "Re-run with stable connection; per-day resumable.",
        "For 2024 Fridays the prod-0 release tag is present.",
    ]),
    ("opensky", source_opensky, [
        "Register at https://opensky-network.org/",
        "Export OPENSKY_USERNAME and OPENSKY_PASSWORD env vars.",
        "pip install pyopensky.",
        "Re-run scripts/acquire_all.py.",
    ]),
    ("noaa_wx", source_noaa_wx, [
        "AWC API rarely fails; check network egress.",
        "For 2024-08 historical METARs, see https://www.aviationweather.gov/dataserver",
        "ERA5 needs ~/.cdsapirc credentials.",
    ]),
    ("lawa", source_lawa, [
        "Hard-coded — should never fail. If it does, inspect src/data/source_lawa.py.",
    ]),
    ("bts", source_bts, [
        "Visit https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLM",
        "Select Year=2024, Quarter=3, table DB1B Coupon, Download.",
        "Drop ZIP at data/cache/bts/Origin_and_Destination_Survey_DB1BCoupon_2024_3.zip",
        "Re-run scripts/acquire_all.py.",
    ]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airport", required=True, help="ICAO code (e.g. KLAX)")
    parser.add_argument("--window", default="2024-08", help="Time window tag (e.g. 2024-08)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output dir (default: data/processed/<ICAO>)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", default="", help="Comma-separated source names to skip")
    parser.add_argument("--only", default="", help="Comma-separated source names to run exclusively")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.debug else "INFO")
    cfg = load_airport(args.airport)
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = path_utils.airport_dir(args.airport, "processed")
    out_root.mkdir(parents=True, exist_ok=True)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    # Merge into existing inventory when running with --only (partial top-up),
    # otherwise start fresh.
    inv_path = out_root / "_inventory.json"
    if only and inv_path.exists():
        try:
            import json as _json
            inventory = _json.loads(inv_path.read_text())
            inventory.setdefault("sources", {})
            inventory["icao"] = args.airport
            inventory["window"] = args.window
            inventory["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.info("merging into existing inventory (%d prior sources)",
                        len(inventory["sources"]))
        except Exception as e:
            logger.warning("could not parse existing inventory (%s); starting fresh", e)
            inventory = {"icao": args.airport, "window": args.window,
                         "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "sources": {}}
    else:
        inventory = {
            "icao": args.airport,
            "window": args.window,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sources": {},
        }

    overall_t0 = time.time()
    for name, mod, recovery in SOURCES:
        if only and name not in only:
            continue
        if name in skip:
            logger.info("skip %s", name)
            continue
        logger.info("=== source %s ===", name)
        t0 = time.time()
        try:
            result: FetchResult = mod.fetch(cfg, window=args.window, out_dir=out_root)
            inventory["sources"][name] = result.to_inventory_entry()
        except Exception as exc:  # noqa: BLE001 — universal trap by design
            logger.exception("source %s failed", name)
            offline_path = write_offline(
                name,
                out_root,
                error=f"{exc.__class__.__name__}: {exc}",
                recovery=recovery,
                source_url=getattr(mod, "SOURCE_URL", ""),
                params={"airport": args.airport, "window": args.window},
            )
            inventory["sources"][name] = {
                "status": "offline",
                "files": [offline_path.name],
                "error": str(exc),
            }
        logger.info("source %s done in %.1fs", name, time.time() - t0)

    inventory["wall_time_s"] = round(time.time() - overall_t0, 1)
    io_utils.write_json(out_root / "_inventory.json", inventory)
    logger.info("inventory → %s", out_root / "_inventory.json")

    ok = sum(1 for v in inventory["sources"].values() if v["status"] == "ok")
    off = sum(1 for v in inventory["sources"].values() if v["status"] == "offline")
    logger.info("DONE: %d ok, %d offline (%.1fs wall)", ok, off, inventory["wall_time_s"])
    return 0  # offline manifests count as clean


if __name__ == "__main__":
    sys.exit(main())
