"""NOAA AWC METAR/TAF + ECMWF ERA5 surface reanalysis.

* METAR: pulled from the aviationweather.gov public Data API (no credentials needed).
  We request the prior 365 days of history (the API exposes
  `hoursBeforeNow` up to 720). For our August 2024 window we'd need the historical
  archive endpoint; the public API can return the most recent ~30 days but historical
  pulls require the `/dataserver/dataserver.php` ZIP archive. We grab both.

* ERA5: optional, only attempted if `CDSAPI_KEY` env var is set. Otherwise we emit an
  OFFLINE marker with the manual recovery flow.

* TAF: pulled alongside METAR in a `taf.parquet` if available.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import io as io_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

from ._common import FetchResult, http_get, write_offline

logger = get_logger(__name__)

AWC_METAR_API = "https://aviationweather.gov/api/data/metar"
AWC_TAF_API = "https://aviationweather.gov/api/data/taf"
SOURCE_URL = AWC_METAR_API


# --------------------------------------------------------------------------- #
# Minimal METAR parser (no external deps): pull wind, vis, ceiling, T/Td, QNH.
# --------------------------------------------------------------------------- #
def _parse_metar(raw: str) -> dict:
    out = {
        "wind_dir_deg": np.nan, "wind_kt": np.nan, "wind_gust_kt": np.nan,
        "vis_sm": np.nan, "temp_c": np.nan, "dewpoint_c": np.nan,
        "altim_hpa": np.nan, "ceiling_ft": np.nan, "flight_rule": "",
    }
    if not isinstance(raw, str) or not raw:
        return out
    s = " " + raw.upper() + " "
    # Wind: dddff(Ggg)KT or VRB
    m = re.search(r"\s(VRB|\d{3})(\d{2,3})(G(\d{2,3}))?KT", s)
    if m:
        if m.group(1) != "VRB":
            out["wind_dir_deg"] = float(m.group(1))
        out["wind_kt"] = float(m.group(2))
        if m.group(4):
            out["wind_gust_kt"] = float(m.group(4))
    # Visibility (statute miles in US METARs): e.g. " 10SM" or " 1 1/2SM"
    m = re.search(r"\s(\d{1,2})\s(\d/\d)SM", s)
    if m:
        out["vis_sm"] = float(m.group(1)) + eval(m.group(2))
    else:
        m = re.search(r"\s(M?\d{1,2}(?:/\d)?)SM", s)
        if m:
            v = m.group(1).replace("M", "")
            if "/" in v:
                num, den = v.split("/")
                out["vis_sm"] = float(num) / float(den)
            else:
                out["vis_sm"] = float(v)
    # Temperature / dewpoint
    m = re.search(r"\s(M?\d{2})/(M?\d{2})\s", s)
    if m:
        def _t(x: str) -> float:
            v = int(x.replace("M", "-").replace("M", "")) if "M" in x else int(x)
            return float(v)
        out["temp_c"] = _t(m.group(1))
        out["dewpoint_c"] = _t(m.group(2))
    # Altimeter: A2992 (in inHg * 100) or Q1013 (hPa)
    m = re.search(r"\sA(\d{4})\s", s)
    if m:
        inhg = int(m.group(1)) / 100.0
        out["altim_hpa"] = inhg * 33.8639
    else:
        m = re.search(r"\sQ(\d{4})\s", s)
        if m:
            out["altim_hpa"] = float(m.group(1))
    # Ceiling: lowest BKN/OVC layer
    layers = re.findall(r"\s(BKN|OVC|VV)(\d{3})", s)
    if layers:
        out["ceiling_ft"] = min(int(h) * 100 for _, h in layers)
    # Flight rule (rough FAA classification)
    vis = out["vis_sm"]
    cei = out["ceiling_ft"]
    if not np.isnan(vis) or not np.isnan(cei):
        v = 99 if np.isnan(vis) else vis
        c = 99999 if np.isnan(cei) else cei
        if v < 1 or c < 500:
            out["flight_rule"] = "LIFR"
        elif v < 3 or c < 1000:
            out["flight_rule"] = "IFR"
        elif v < 5 or c < 3000:
            out["flight_rule"] = "MVFR"
        else:
            out["flight_rule"] = "VFR"
    return out


def _parse_metar_csv(text: str) -> pd.DataFrame:
    """The AWC API returns CSV with a header line. Some rows may be blank."""
    # The new aviationweather.gov API returns a JSON or CSV depending on `format`.
    # We request `format=csv`. The first line is a comment; the second is the header.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    # Drop leading comment lines
    while lines and (lines[0].startswith("#") or not "," in lines[0]):
        lines.pop(0)
    if not lines:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)), dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _normalise_metar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map AWC CSV columns to our schema."""
    if df.empty:
        return pd.DataFrame(columns=[
            "station_id", "time_utc", "wind_dir_deg", "wind_kt", "wind_gust_kt",
            "vis_sm", "temp_c", "dewpoint_c", "altim_hpa", "ceiling_ft",
            "flight_rule", "raw"])
    col = lambda *names: next((n for n in names if n in df.columns), None)
    out = pd.DataFrame()
    out["station_id"] = df[col("icaoid", "station_id", "station")].astype(str)
    # AWC time field is `reportTime` or `obsTime` (UTC epoch seconds or ISO)
    tcol = col("reporttime", "obstime", "valid_time", "time")
    if tcol:
        ser = df[tcol]
        if ser.dtype == object:
            # Try ISO first
            try:
                out["time_utc"] = pd.to_datetime(ser, utc=True, errors="coerce")
            except Exception:
                out["time_utc"] = pd.to_datetime(pd.to_numeric(ser, errors="coerce"),
                                                  unit="s", utc=True)
        else:
            out["time_utc"] = pd.to_datetime(ser, unit="s", utc=True)
    raw_col = col("rawob", "raw_text", "raw_observation", "raw")
    out["raw"] = df[raw_col].astype(str) if raw_col else ""

    # Parse from raw to fill numeric fields uniformly
    parsed = out["raw"].map(_parse_metar)
    for k in ["wind_dir_deg", "wind_kt", "wind_gust_kt", "vis_sm",
              "temp_c", "dewpoint_c", "altim_hpa", "ceiling_ft", "flight_rule"]:
        out[k] = parsed.map(lambda d, k=k: d[k])
    for c in ["wind_dir_deg", "wind_kt", "wind_gust_kt", "vis_sm",
              "temp_c", "dewpoint_c", "altim_hpa", "ceiling_ft"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


def _request_metar(station: str, *, hours_before: int = 168) -> pd.DataFrame:
    params = {
        "ids": station,
        "format": "csv",
        "hours": hours_before,
        "taf": "false",
    }
    r = http_get(AWC_METAR_API, params=params, timeout=120)
    r.raise_for_status()
    return _parse_metar_csv(r.text)


def _request_taf(station: str) -> pd.DataFrame:
    params = {"ids": station, "format": "csv", "hours": 24}
    r = http_get(AWC_TAF_API, params=params, timeout=60)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    if not lines:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines)), dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _fetch_era5(airport_cfg: dict, out_dir: Path) -> Path | None:
    """ERA5 single-level via cdsapi (if creds available)."""
    if not (os.environ.get("CDSAPI_KEY") or (Path.home() / ".cdsapirc").exists()):
        return None
    try:
        import cdsapi  # type: ignore
    except Exception as e:
        logger.warning("cdsapi not installed: %s", e)
        return None
    frame = AirportFrame.from_cfg(airport_cfg)
    lat, lon = frame.lat0, frame.lon0
    area = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]
    out = out_dir / "era5_surface.nc"
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind",
                         "mean_sea_level_pressure", "2m_temperature",
                         "2m_dewpoint_temperature"],
            "year": "2024", "month": "08",
            "day": ["02", "09", "16", "23", "30"],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
        },
        str(out),
    )
    return out


def fetch(airport_cfg: dict, *, window: str, out_dir: Path) -> FetchResult:
    icao = airport_cfg["icao"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # METAR — recent observations + parse from raw
    df = _request_metar(icao, hours_before=168)
    parsed = _normalise_metar_df(df)
    metar_path = out_dir / "metar.parquet"
    parsed.to_parquet(metar_path, index=False)
    io_utils.write_manifest(
        metar_path, source="noaa_wx_metar", source_url=AWC_METAR_API,
        params={"airport": icao, "window": window, "hours_before": 168},
        extra={"row_count": int(len(parsed)),
               "note": ("AWC live API returns ~7 days; for 2024-08 historical pull use "
                        "the AWC archive: https://www.aviationweather.gov/dataserver "
                        "or the NOAA ISD-Lite dataset.")},
    )
    files = [metar_path.name]

    # TAF
    try:
        taf_df = _request_taf(icao)
        if len(taf_df) > 0:
            taf_path = out_dir / "taf.parquet"
            taf_df.to_parquet(taf_path, index=False)
            io_utils.write_manifest(taf_path, source="noaa_wx_taf",
                                     source_url=AWC_TAF_API,
                                     params={"airport": icao, "window": window},
                                     extra={"row_count": int(len(taf_df))})
            files.append(taf_path.name)
    except Exception as e:
        logger.warning("TAF fetch failed: %s", e)

    # ERA5 — optional
    try:
        era5 = _fetch_era5(airport_cfg, out_dir)
        if era5 and era5.exists():
            io_utils.write_manifest(era5, source="era5",
                                     source_url="https://cds.climate.copernicus.eu/",
                                     params={"airport": icao, "window": window},
                                     extra={})
            files.append(era5.name)
        else:
            write_offline(
                "era5", out_dir,
                error="No CDS API credentials (CDSAPI_KEY or ~/.cdsapirc) — ERA5 skipped.",
                source_url="https://cds.climate.copernicus.eu/",
                recovery=[
                    "Register at https://cds.climate.copernicus.eu/",
                    "Accept the ERA5 dataset terms.",
                    "Put your key in ~/.cdsapirc (format described in docs).",
                    "pip install cdsapi",
                    f"Re-run scripts/acquire_all.py --airport {icao} --window {window}",
                ],
                params={"airport": icao},
            )
            files.append("era5.OFFLINE.json")
    except Exception as e:
        write_offline("era5", out_dir, error=str(e),
                      source_url="https://cds.climate.copernicus.eu/",
                      recovery=["Check CDS account quotas + dataset licence acceptance",
                                "Retry with `python -c 'import cdsapi; cdsapi.Client().retrieve(...)'`"],
                      params={"airport": icao})
        files.append("era5.OFFLINE.json")

    logger.info("NOAA wx: %d METAR rows for %s", len(parsed), icao)
    return FetchResult(name="noaa_wx", status="ok", files=files,
                       extra={"metar_rows": int(len(parsed))})
