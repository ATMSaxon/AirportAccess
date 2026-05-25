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
# Iowa State ASOS archive — free, station-by-station historical METAR back decades.
# Reference: https://mesonet.agron.iastate.edu/request/download.phtml
ASOS_API = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
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


def _parse_metar_json(payload: list[dict]) -> pd.DataFrame:
    """The AWC API JSON endpoint returns a list[dict] with structured METAR fields."""
    if not payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    df.columns = [c.strip() for c in df.columns]
    return df


def _parse_metar_raw(text: str) -> pd.DataFrame:
    """`format=raw` returns one METAR per line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()
    return pd.DataFrame({"rawOb": lines})


def _normalise_metar_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map AWC JSON columns to our schema."""
    EMPTY_COLS = ["station_id", "time_utc", "wind_dir_deg", "wind_kt", "wind_gust_kt",
                  "vis_sm", "temp_c", "dewpoint_c", "altim_hpa", "ceiling_ft",
                  "flight_rule", "raw"]
    if df.empty:
        return pd.DataFrame(columns=EMPTY_COLS)
    col = lambda *names: next((n for n in names if n in df.columns), None)
    out = pd.DataFrame()
    sid = col("icaoId", "icaoid", "station_id", "station")
    out["station_id"] = df[sid].astype(str) if sid else ""
    # AWC time field is `reportTime` (ISO) or `obsTime` (UTC epoch seconds).
    # ASOS archive path pre-parses to `time_utc` (already datetime64[ns, UTC]); we
    # accept it as well as the IEM raw `valid` column (ISO strings like
    # "2024-08-01 00:53") for safety if `_request_asos_archive` is bypassed.
    tcol = col("time_utc", "reportTime", "reporttime", "obsTime", "obstime",
               "valid_time", "valid", "time")
    if tcol:
        ser = df[tcol]
        if pd.api.types.is_datetime64_any_dtype(ser):
            # Already parsed — just ensure UTC tz.
            if getattr(ser.dt, "tz", None) is None:
                out["time_utc"] = pd.to_datetime(ser, utc=True, errors="coerce")
            else:
                out["time_utc"] = ser.dt.tz_convert("UTC")
        elif pd.api.types.is_numeric_dtype(ser):
            # AWC `obsTime` is UTC epoch seconds.
            out["time_utc"] = pd.to_datetime(ser, unit="s", utc=True, errors="coerce")
        else:
            # Object/string column — try ISO parse first, fall back to epoch-seconds.
            parsed = pd.to_datetime(ser, utc=True, errors="coerce")
            if parsed.isna().all():
                parsed = pd.to_datetime(pd.to_numeric(ser, errors="coerce"),
                                        unit="s", utc=True, errors="coerce")
            out["time_utc"] = parsed
    else:
        out["time_utc"] = pd.NaT
    raw_col = col("rawOb", "rawob", "raw_text", "raw_observation", "raw")
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
    """Hit aviationweather.gov `format=json` (only `raw` and `json` are accepted)."""
    params = {
        "ids": station,
        "format": "json",
        "hours": hours_before,
        "taf": "false",
    }
    r = http_get(AWC_METAR_API, params=params, timeout=120)
    r.raise_for_status()
    try:
        payload = r.json()
    except Exception:
        # Fallback: try `raw` and synthesise a minimal frame
        params["format"] = "raw"
        r = http_get(AWC_METAR_API, params=params, timeout=120)
        r.raise_for_status()
        return _parse_metar_raw(r.text)
    return _parse_metar_json(payload)


def _parse_window(window: str) -> tuple[dt.date, dt.date] | None:
    """Translate a window tag (`2024-08`, `2024-08-02`, `2024-08-01..2024-09-01`) → date pair.

    For month tags we span the whole calendar month; for single-day we use [day, day+1].
    For range strings (`A..B`) we use A inclusive, B exclusive."""
    w = window.strip()
    if ".." in w:
        a, b = w.split("..", 1)
        try:
            return dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        except ValueError:
            return None
    parts = w.split("-")
    try:
        if len(parts) == 3:
            d = dt.date.fromisoformat(w)
            return d, d + dt.timedelta(days=1)
        if len(parts) == 2:
            y, m = int(parts[0]), int(parts[1])
            start = dt.date(y, m, 1)
            end = dt.date(y + (m // 12), (m % 12) + 1, 1)
            return start, end
    except (ValueError, IndexError):
        return None
    return None


def _request_asos_archive(station: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Pull historical METAR from the Iowa State ASOS archive (free, no auth)."""
    # Strip leading 'K' for the IEM station code (4-letter ICAO with country prefix
    # removed for US stations).
    sid = station[1:] if station.startswith("K") and len(station) == 4 else station
    params = {
        "station": sid,
        "data": "metar",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "yes",
        "report_type": "3",  # MADIS-grade ASOS hourly + special METARs
        "report_type2": "4",
    }
    r = http_get(ASOS_API, params=params, timeout=180)
    r.raise_for_status()
    text = r.text
    if not text or text.strip().startswith("#"):
        # IEM returns a comment-only response when no data
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False, comment="#")
    if df.empty or "metar" not in df.columns:
        return pd.DataFrame()
    # IEM exposes the raw METAR string under the `metar` column; the rest are derived.
    out = pd.DataFrame()
    out["station_id"] = df["station"].astype(str)
    out["time_utc"] = pd.to_datetime(df["valid"], utc=True, errors="coerce")
    out["raw"] = df["metar"].astype(str)
    return out


def _request_taf(station: str) -> pd.DataFrame:
    params = {"ids": station, "format": "json", "hours": 24}
    r = http_get(AWC_TAF_API, params=params, timeout=60)
    r.raise_for_status()
    try:
        payload = r.json()
    except Exception:
        return pd.DataFrame()
    if not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload)


def _fetch_era5(airport_cfg: dict, out_dir: Path,
                 max_wait_s: float = 120.0) -> Path | None:
    """ERA5 single-level via cdsapi (if creds available).

    Wraps the blocking ``cdsapi.Client().retrieve(...)`` in a daemon thread with a
    wall-clock timeout so the orchestrator doesn't stall when CDS is in maintenance
    (the retrieve() call internally polls forever until the request completes).
    """
    if not (os.environ.get("CDSAPI_KEY") or (Path.home() / ".cdsapirc").exists()):
        return None
    try:
        import cdsapi  # type: ignore
    except Exception as e:
        logger.warning("cdsapi not installed: %s", e)
        return None
    import threading
    frame = AirportFrame.from_cfg(airport_cfg)
    lat, lon = frame.lat0, frame.lon0
    area = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]
    out = out_dir / "era5_surface.nc"

    err_slot: list[BaseException | None] = [None]

    def _work() -> None:
        try:
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
        except BaseException as e:  # noqa: BLE001 — capture any thread-local error
            err_slot[0] = e

    t = threading.Thread(target=_work, daemon=True, name="era5-cdsapi")
    t.start()
    t.join(timeout=max_wait_s)
    if t.is_alive():
        raise TimeoutError(
            f"ERA5 retrieval exceeded {max_wait_s}s wall-clock budget "
            "(CDS likely in maintenance / queue saturated). "
            "Re-run later to obtain the netCDF — the orchestrator continues without it."
        )
    if err_slot[0] is not None:
        raise err_slot[0]
    return out


def fetch(airport_cfg: dict, *, window: str, out_dir: Path) -> FetchResult:
    icao = airport_cfg["icao"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # METAR — historical archive (Iowa State ASOS) for windows in the past,
    # AWC live API for current-week windows. The Iowa State archive serves the
    # exact same FAA stations as ICAO codes (with 'K' stripped) with date-range
    # query params and emits raw METAR strings.
    span = _parse_window(window)
    today = dt.date.today()
    source_used = AWC_METAR_API
    archive_used = False
    if span and span[1] < today - dt.timedelta(days=2):
        try:
            df = _request_asos_archive(icao, span[0], span[1])
            if not df.empty:
                archive_used = True
                source_used = ASOS_API
                logger.info("ASOS archive: %d raw METAR rows for %s [%s..%s]",
                            len(df), icao, span[0], span[1])
            else:
                logger.warning("ASOS archive returned 0 rows for %s; "
                               "falling back to AWC live API", icao)
                df = _request_metar(icao, hours_before=168)
        except Exception as e:
            logger.warning("ASOS archive failed (%s); falling back to AWC live API", e)
            df = _request_metar(icao, hours_before=168)
    else:
        df = _request_metar(icao, hours_before=168)
    parsed = _normalise_metar_df(df)
    metar_path = out_dir / "metar.parquet"
    parsed.to_parquet(metar_path, index=False)
    metar_note = (
        f"Window {window} ({span[0]}..{span[1]}) from Iowa State ASOS archive."
        if archive_used else
        "AWC live API (~7-day window). For pre-2026 windows use Iowa State ASOS archive."
    )
    io_utils.write_manifest(
        metar_path, source="noaa_wx_metar", source_url=source_used,
        params={"airport": icao, "window": window,
                "archive": "iowa_state_asos" if archive_used else "awc_live"},
        extra={"row_count": int(len(parsed)), "note": metar_note},
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
