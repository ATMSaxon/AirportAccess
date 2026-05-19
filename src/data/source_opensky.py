"""OpenSky Network ADS-B trajectories.

Two paths:

1. **Trino (history)** — preferred. Requires `OPENSKY_USERNAME` + `OPENSKY_PASSWORD` env
   vars and the `pyopensky.trino` package (optional dependency). Pulls hourly state-vector
   batches covering the requested Friday days of August 2024.

2. **REST `/states/all`** — fallback. The public anonymous endpoint exposes a single
   *current* snapshot. We poll it once and write a same-day snapshot parquet so the
   pipeline has something to operate on; for historical days we mark the day OFFLINE
   with the manual recovery flow (use the authenticated `/tracks/all` endpoint or
   `pyopensky.trino`).

Either way the output schema matches `src/data/SCHEMAS.md`.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import io as io_utils
from src.utils import paths as path_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

from ._common import FetchResult, bbox_around_arp, http_get, write_offline

logger = get_logger(__name__)

REST_BASE = "https://opensky-network.org/api"
SOURCE_URL = REST_BASE

# 30 NM around ARP ≈ 55.56 km
DEFAULT_HALF_KM = 56.0

# LAX August 2024 Fridays (project window).
DEFAULT_DAYS_2024_AUG = [
    "2024-08-02", "2024-08-09", "2024-08-16", "2024-08-23", "2024-08-30",
]

STATE_COLS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]


def _enrich(df: pd.DataFrame, frame: AirportFrame, field_elev_m: float) -> pd.DataFrame:
    """Add x_m, y_m, z_msl_m, z_agl_m columns and normalise dtypes."""
    df = df.copy()
    df["icao24"] = df["icao24"].astype(str).str.strip()
    df["callsign"] = df.get("callsign", "").astype(str).str.strip()
    for c in ("longitude", "latitude", "baro_altitude", "geo_altitude",
              "velocity", "true_track", "vertical_rate"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["on_ground"] = df.get("on_ground", False).astype(bool)
    df = df.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)

    x_m, y_m = frame.wgs_to_enu(df["longitude"].to_numpy(),
                                df["latitude"].to_numpy())
    df["x_m"] = x_m.astype(np.float32)
    df["y_m"] = y_m.astype(np.float32)
    # Prefer GNSS altitude; fall back to barometric pressure altitude
    z = df["geo_altitude"].where(df["geo_altitude"].notna(), df["baro_altitude"])
    df["z_msl_m"] = z.astype("float32")
    df["z_agl_m"] = (df["z_msl_m"] - field_elev_m).astype("float32")

    out = pd.DataFrame({
        "icao24": df["icao24"].astype(str),
        "callsign": df["callsign"].astype(str),
        "lon_wgs": df["longitude"].astype("float32"),
        "lat_wgs": df["latitude"].astype("float32"),
        "baro_alt_m": df["baro_altitude"].astype("float32"),
        "geo_alt_m": df["geo_altitude"].astype("float32"),
        "velocity_ms": df["velocity"].astype("float32"),
        "track_deg": df["true_track"].astype("float32"),
        "vert_rate_ms": df["vertical_rate"].astype("float32"),
        "on_ground": df["on_ground"].astype(bool),
        "x_m": df["x_m"],
        "y_m": df["y_m"],
        "z_msl_m": df["z_msl_m"],
        "z_agl_m": df["z_agl_m"],
    })
    if "time_utc" in df.columns:
        out.insert(0, "time_utc", pd.to_datetime(df["time_utc"], utc=True))
    return out


def _fetch_rest_snapshot(bbox: tuple[float, float, float, float],
                         auth: tuple[str, str] | None) -> pd.DataFrame:
    """One-shot snapshot from the public REST API."""
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {"lamin": lat_min, "lomin": lon_min,
              "lamax": lat_max, "lomax": lon_max}
    r = http_get(f"{REST_BASE}/states/all", params=params, timeout=60, auth=auth)
    r.raise_for_status()
    payload = r.json()
    states = payload.get("states") or []
    if not states:
        return pd.DataFrame(columns=STATE_COLS + ["time_utc"])
    df = pd.DataFrame(states, columns=STATE_COLS)
    df["time_utc"] = pd.to_datetime(payload.get("time"), unit="s", utc=True)
    return df


def _fetch_trino_day(day: str, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Pull a full UTC day of state vectors via pyopensky.trino. Imports lazily."""
    try:
        from pyopensky.trino import Trino  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pyopensky.trino not installed; install `pip install pyopensky` "
            "and set OPENSKY_USERNAME / OPENSKY_PASSWORD"
        ) from e

    start = pd.Timestamp(day, tz="UTC")
    stop = start + pd.Timedelta(days=1)
    tr = Trino()
    lon_min, lat_min, lon_max, lat_max = bbox
    df = tr.history(
        start=start, stop=stop,
        bounds=(lon_min, lat_min, lon_max, lat_max),
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()
    # pyopensky returns lower-case columns mostly matching REST schema. Normalise.
    rename = {
        "lon": "longitude", "lat": "latitude",
        "baroaltitude": "baro_altitude", "geoaltitude": "geo_altitude",
        "groundspeed": "velocity", "heading": "true_track",
        "vertrate": "vertical_rate", "onground": "on_ground",
        "timestamp": "time_utc",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    if "time_utc" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "time_utc"})
    return df


def fetch(airport_cfg: dict, *, window: str, out_dir: Path,
          days: list[str] | None = None,
          half_km: float = DEFAULT_HALF_KM) -> FetchResult:
    icao = airport_cfg["icao"]
    frame = AirportFrame.from_cfg(airport_cfg)
    bbox = bbox_around_arp(frame, half_km=half_km)
    field_elev_m = float(airport_cfg["arp"]["elev_m"])

    days = days or DEFAULT_DAYS_2024_AUG
    out_dir.mkdir(parents=True, exist_ok=True)

    user = os.environ.get("OPENSKY_USERNAME")
    pw = os.environ.get("OPENSKY_PASSWORD")
    auth = (user, pw) if user and pw else None

    files: list[str] = []
    per_day: dict = {}
    trino_failed = False

    for day in days:
        day_path = out_dir / f"adsb_{day}.parquet"
        if day_path.exists():
            per_day[day] = {"status": "cached", "rows": None}
            files.append(day_path.name)
            continue

        # 1) Try Trino if creds present and prior Trino attempt hadn't already failed
        if auth and not trino_failed:
            try:
                raw = _fetch_trino_day(day, bbox)
                if len(raw) > 0:
                    enriched = _enrich(raw, frame, field_elev_m)
                    enriched.to_parquet(day_path, index=False)
                    io_utils.write_manifest(
                        day_path, source="opensky",
                        source_url="https://trino.opensky-network.org/",
                        params={"airport": icao, "day": day, "bbox": list(bbox),
                                "method": "pyopensky.trino"},
                        extra={"row_count": int(len(enriched))},
                    )
                    files.append(day_path.name)
                    per_day[day] = {"status": "ok", "rows": int(len(enriched)),
                                    "method": "trino"}
                    continue
                else:
                    logger.warning("Trino returned 0 rows for %s; falling back", day)
            except Exception as e:
                logger.warning("Trino fetch failed for %s: %s — disabling Trino", day, e)
                trino_failed = True

        # 2) REST fallback — only useful for the current day; mark historical days OFFLINE
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        if day == today:
            try:
                df = _fetch_rest_snapshot(bbox, auth=auth)
                if len(df) > 0:
                    enriched = _enrich(df, frame, field_elev_m)
                    enriched.to_parquet(day_path, index=False)
                    io_utils.write_manifest(
                        day_path, source="opensky",
                        source_url=f"{REST_BASE}/states/all",
                        params={"airport": icao, "day": day, "bbox": list(bbox),
                                "method": "rest_snapshot"},
                        extra={"row_count": int(len(enriched))},
                    )
                    files.append(day_path.name)
                    per_day[day] = {"status": "ok", "rows": int(len(enriched)),
                                    "method": "rest_snapshot"}
                    continue
            except Exception as e:
                logger.warning("REST snapshot failed for %s: %s", day, e)

        # 3) Day-level OFFLINE
        write_offline(
            f"opensky_{day}", out_dir,
            error=f"No live data for {day}. "
                  "Trino requires OPENSKY_USERNAME/OPENSKY_PASSWORD; REST exposes only "
                  "the current snapshot, not history.",
            source_url=SOURCE_URL,
            recovery=[
                "Register at https://opensky-network.org/ and obtain credentials.",
                "Export OPENSKY_USERNAME and OPENSKY_PASSWORD.",
                "Install pyopensky: pip install pyopensky.",
                f"Re-run: python scripts/acquire_all.py --airport {icao} --window {window}",
                "Trino throughput is roughly 1–5 M rows/min — full LAX-day pull ~10 min.",
            ],
            params={"airport": icao, "day": day, "bbox": list(bbox)},
        )
        files.append(f"opensky_{day}.OFFLINE.json")
        per_day[day] = {"status": "offline"}

    # Final status: OK if at least one day succeeded, otherwise OFFLINE (the per-day
    # offline manifests carry the recovery flow).
    ok_days = sum(1 for v in per_day.values() if v.get("status") in ("ok", "cached"))
    if ok_days == 0:
        raise RuntimeError(
            f"OpenSky: 0/{len(days)} requested days produced data. "
            "Need OPENSKY_USERNAME/OPENSKY_PASSWORD env vars."
        )
    return FetchResult(
        name="opensky",
        status="ok",
        files=files,
        extra={"days": per_day, "ok_days": ok_days, "total_days": len(days)},
    )
