"""CLI: build the dynamic envelope for one airport-day.

Usage::

    python scripts/build_envelope.py --airport KLAX --window 2024-08-02 --interval 15min

Pipeline:
    1. Load airport config + cleaned ADS-B parquet + METAR parquet (if available).
       If ADS-B is offline (`adsb_<date>.OFFLINE.json` present), the script still
       runs end-to-end on whatever is available and writes an OFFLINE manifest.
    2. Clean tracks (`adsb_clean.clean_tracks`) and classify (`classify`).
    3. 15-min rolling configuration (`runway_config.rolling_config`).
    4. For each slice, build `E_t = A_static \\ C_t` via `envelope.envelope_for_slice`.
    5. Write `data/processed/<ICAO>/envelope_<date>.zarr` (or .npz fallback) and
       `runway_config_<date>.parquet`. Write `results/build_envelope_<ICAO>_<date>/summary.json`.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow `python scripts/build_envelope.py` from the repo root without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.utils import paths, io as ioutil, config as cfgutil, grid as gridmod
from src.utils.crs import AirportFrame
from src.utils.logs import setup_logging, get_logger
from src.traffic import adsb_clean, classify, runway_config, density, envelope

log = get_logger("build_envelope")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build dynamic envelope per 15-min slice.")
    p.add_argument("--airport", required=True, help="ICAO (e.g. KLAX)")
    p.add_argument("--window", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--interval", default="15min", help="slice length, e.g. 15min")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None, help="optional airport YAML override")
    p.add_argument("--output-dir", default=None,
                   help="override `data/processed/<ICAO>/`")
    p.add_argument("--config-aware-static", action="store_true",
                   help="use src.geometry.query.PrismIndex to recompute A_static "
                        "per slice from the active runways (slower but truly "
                        "runway-config-aware). Default off — uses the global "
                        "static A_static from SDFQuery.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def _load_adsb_for_date(airport: str, date: str) -> pd.DataFrame | None:
    apath = paths.airport_dir(airport, "processed") / f"adsb_{date}.parquet"
    if apath.exists():
        log.info("loading ADS-B parquet %s", apath)
        return adsb_clean.load_adsb_parquet(apath)
    offline = apath.with_suffix(".OFFLINE.json")
    if offline.exists():
        log.warning("ADS-B offline for %s/%s — see %s", airport, date, offline)
    else:
        log.warning("no ADS-B parquet at %s (no OFFLINE marker either)", apath)
    return None


def _load_metar(airport: str, date: str) -> pd.DataFrame | None:
    """Load METAR parquet matching the D6 schema, filter to the day if possible."""
    mpath = paths.airport_dir(airport, "processed") / "metar.parquet"
    if not mpath.exists():
        return None
    try:
        m = pd.read_parquet(mpath)
    except Exception as e:
        log.warning("metar parquet unreadable: %s", e)
        return None
    if "time_utc" in m.columns:
        if not pd.api.types.is_datetime64_any_dtype(m["time_utc"]):
            m["time_utc"] = pd.to_datetime(m["time_utc"], utc=True)
        d0 = pd.Timestamp(date, tz="UTC")
        m = m[(m["time_utc"] >= d0 - pd.Timedelta("1d")) &
              (m["time_utc"] < d0 + pd.Timedelta("2d"))]
    return m


def _interval_minutes(spec: str) -> int:
    spec = spec.strip().lower()
    if spec.endswith("min"):
        return int(spec[:-3])
    if spec.endswith("m"):
        return int(spec[:-1])
    return int(spec)


def main(argv=None) -> int:
    args = _parse_args(argv)
    setup_logging("DEBUG" if args.debug else "INFO")
    np.random.seed(args.seed)

    if args.config:
        airport_cfg = cfgutil.load_yaml(args.config)
    else:
        airport_cfg = cfgutil.load_airport(args.airport)
    # Inject the local_crs_epsg fallback because AirportFrame requires it
    airport_cfg.setdefault("arp", {})
    airport_cfg["arp"].setdefault("elev_m",
                                  float(airport_cfg["arp"].get("elev_ft", 0.0)) * 0.3048)
    airport_cfg.setdefault("icao", args.airport)
    airport_cfg.setdefault("local_crs_epsg",
                           airport_cfg.get("local_crs_epsg", 32611))
    frame = AirportFrame.from_cfg(airport_cfg)
    grid = gridmod.VoxelGrid.from_airport_cfg(airport_cfg)
    runway_ends = classify.airport_runway_ends(airport_cfg, frame)
    bearings = {r.runway_id: r.bearing_deg for r in runway_ends}

    out_dir = Path(args.output_dir) if args.output_dir else paths.airport_dir(args.airport, "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    adsb_raw = _load_adsb_for_date(args.airport, args.window)
    metar = _load_metar(args.airport, args.window)

    summary: dict = {
        "airport": args.airport, "window": args.window,
        "interval": args.interval, "seed": args.seed,
        "adsb_present": adsb_raw is not None, "metar_present": metar is not None,
        "n_runway_ends": len(runway_ends),
        "grid_shape": list(grid.shape),
    }

    if adsb_raw is None or adsb_raw.empty:
        offline_manifest = out_dir / f"envelope_{args.window}.OFFLINE.json"
        ioutil.write_json(offline_manifest, {
            "reason": "no_adsb_parquet",
            "airport": args.airport, "window": args.window,
            "manual_recovery": "Run scripts/acquire_all.py to fetch OpenSky data, "
                               "or re-run with --config pointing at synthetic ADS-B.",
        })
        log.error("no ADS-B parquet for %s/%s; wrote offline manifest %s",
                  args.airport, args.window, offline_manifest)
        summary["envelope_file"] = None
        ioutil.write_summary(paths.run_dir(f"build_envelope_{args.airport}_{args.window}"), summary)
        return 1

    cleaned, clean_stats = adsb_clean.clean_tracks(adsb_raw, frame, metar=metar)
    tracks = classify.classify_tracks(cleaned, runway_ends)
    log.info("classified %d tracks: %s", len(tracks),
             dict(tracks["category"].value_counts()) if not tracks.empty else "{}")

    day_start = pd.Timestamp(args.window, tz="UTC")
    day_end = day_start + pd.Timedelta("1D")
    interval_min = _interval_minutes(args.interval)
    rconf = runway_config.rolling_config(
        tracks, day_start, day_end,
        interval_min=interval_min, metar=metar, runway_bearings=bearings,
    )

    # Persist runway config table.
    rconf_path = out_dir / f"runway_config_{args.window}.parquet"
    # Lists -> JSON-encoded strings for parquet portability.
    # Persist with canonical (list-as-CSV) + downstream alias columns.
    # `active_arrivals` / `active_departures` / `time_utc` / `config_id` are
    # already populated by `runway_config.rolling_config` per SCHEMAS.md.
    rconf_to_write = rconf.copy()
    for col in ("arrivals_active", "departures_active"):
        if col in rconf_to_write.columns:
            rconf_to_write[col] = rconf_to_write[col].apply(
                lambda v: ",".join(v) if isinstance(v, (list, tuple)) else v)
    rconf_to_write.to_parquet(rconf_path, index=False)
    ioutil.write_manifest(rconf_path, source="src.traffic.runway_config",
                           params={"interval_min": interval_min,
                                    "active_share_threshold": runway_config.ACTIVE_SHARE_THRESHOLD,
                                    "wind_bearing_tol_deg": runway_config.WIND_BEARING_TOL_DEG})
    summary["runway_config_file"] = str(rconf_path)
    summary["metar_match_rate"] = runway_config.metar_match_rate(rconf)
    summary["n_slices"] = int(len(rconf))

    # Static mask (A_static): default → global SDFQuery; opt-in → PrismIndex per slice.
    prism_index = None
    a_static = None
    if args.config_aware_static:
        prism_index = envelope.load_prism_index(args.airport)
        if prism_index is None:
            log.warning("--config-aware-static requested but PrismIndex unavailable; "
                        "falling back to global SDFQuery A_static.")
    if prism_index is None:
        a_static = envelope.load_static_mask(args.airport, grid)
    summary["a_static_available"] = a_static is not None or prism_index is not None
    summary["a_static_mode"] = "prism_index" if prism_index is not None else (
        "sdf_query" if a_static is not None else "all_clear")

    # Build per-slice envelope masks.
    envelopes: dict[str, np.ndarray] = {}
    slice_times: list[pd.Timestamp] = []
    for _, row in rconf.iterrows():
        wx = envelope.WeatherState(
            vis_sm=row.get("visibility_sm"),
            ceiling_ft=row.get("ceiling_ft"),
            flight_rule=row.get("flight_rule"),
        )
        e_t = envelope.envelope_for_slice(
            grid=grid,
            runway_ends=runway_ends,
            arrivals_active=row["arrivals_active"] or [],
            departures_active=row["departures_active"] or [],
            weather=wx,
            a_static=a_static,
            prism_index=prism_index,
        )
        envelopes[row["slice_start"].isoformat()] = e_t
        slice_times.append(row["slice_start"])

    env_path = out_dir / f"envelope_{args.window}.zarr"
    written = envelope.save_envelope_zarr(envelopes, env_path, grid, slice_times)
    ioutil.write_manifest(written, source="src.traffic.envelope", params={
        "lateral_buffer_m": envelope.LATERAL_BUFFER_M,
        "vertical_buffer_m": envelope.VERTICAL_BUFFER_M,
        "low_alt_cap_agl_m": envelope.LOW_ALT_CAP_AGL_M,
        "imc_lateral_expansion": envelope.IMC_LATERAL_EXPANSION,
        "approach_length_m": envelope.APPROACH_LENGTH_M,
        "departure_length_m": envelope.DEPARTURE_LENGTH_M,
    })
    summary["envelope_file"] = str(written)
    summary["clean_stats"] = clean_stats.as_dict()

    # Compact KPIs.
    if envelopes:
        kept_fractions = [float(e.mean()) for e in envelopes.values()]
        summary["envelope_kept_fraction_mean"] = float(np.mean(kept_fractions))
        summary["envelope_kept_fraction_min"] = float(np.min(kept_fractions))
        summary["envelope_kept_fraction_max"] = float(np.max(kept_fractions))

    ioutil.write_summary(paths.run_dir(f"build_envelope_{args.airport}_{args.window}"), summary)
    log.info("done — wrote %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
