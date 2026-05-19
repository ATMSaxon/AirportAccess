"""ADS-B preprocessing: outlier rejection, resampling, ENU re-projection, geometric altitude.

Input contract: OpenSky parquet matching `src/data/SCHEMAS.md` §D5 (one file per UTC day).

Pipeline per file:
  1. Drop rows missing position or duplicate ``(icao24, time_utc)``.
  2. Outlier rejection per ``icao24`` track:
       * speed jump  > ``SPEED_JUMP_MAX_MS`` (m/s) between consecutive samples
       * altitude jump > ``ALT_JUMP_MAX_M`` (m) between consecutive samples
       * impossible negative geometric altitude
  3. Resample to a fixed grid (``RESAMPLE_S`` seconds, default 5 s) per ``icao24`` by
     time-bin median (robust to jitter).
  4. Re-project ``(lon_wgs, lat_wgs)`` to the airport's local ENU frame, overwriting
     ``x_m``, ``y_m`` (we don't trust upstream projection assumptions).
  5. Derive geometric altitude:
        * if ``geo_alt_m`` (GNSS) is valid, use it.
        * else convert ``baro_alt_m`` (pressure) using the contemporaneous METAR
          altimeter setting (``altim_hpa``) via standard ICAO formula. Fallback to
          ISA-standard QNH (1013.25 hPa) if METAR is unavailable.
  6. Compute ``z_agl_m`` = ``z_msl_m`` − field elevation (DEM lookup is a future
     enhancement; field-elev is sufficient for the dynamic envelope masks because
     they operate primarily within the airport horizon).

Returns a ``pd.DataFrame`` with the columns of §D5 plus the boolean ``cleaned``.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from ..utils.crs import AirportFrame
from ..utils.logs import get_logger

log = get_logger(__name__)

# Tunables (kept module-level so callers can monkeypatch in tests).
SPEED_JUMP_MAX_MS: float = 200.0     # m/s between consecutive samples
ALT_JUMP_MAX_M: float = 1000.0       # m between consecutive samples (~3300 ft)
RESAMPLE_S: int = 5                  # resample bin (seconds)
ISA_QNH_HPA: float = 1013.25

# ICAO state vector schema (kept here so a user can inspect at import time).
REQUIRED_COLUMNS = (
    "time_utc", "icao24", "lon_wgs", "lat_wgs",
    "baro_alt_m", "geo_alt_m", "velocity_ms",
    "track_deg", "vert_rate_ms", "on_ground",
)


@dataclass
class CleanStats:
    n_in: int
    n_after_dedup: int
    n_after_outliers: int
    n_after_resample: int
    n_tracks: int
    n_offline_qnh: int       # rows that used ISA QNH fallback

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_adsb_parquet(path: str | Path) -> pd.DataFrame:
    """Load OpenSky D5 parquet, normalising types and ensuring required cols exist."""
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time_utc"]):
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    elif df["time_utc"].dt.tz is None:
        df["time_utc"] = df["time_utc"].dt.tz_localize("UTC")
    df["icao24"] = df["icao24"].astype(str).str.lower()
    return df


def clean_tracks(df: pd.DataFrame,
                 frame: AirportFrame,
                 metar: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, CleanStats]:
    """Top-level pipeline. Returns the cleaned dataframe and a CleanStats record."""
    n_in = len(df)
    df = _drop_invalid_and_duplicates(df)
    n_after_dedup = len(df)

    df = _reject_outliers(df)
    n_after_outliers = len(df)

    df = _resample_per_track(df, period_s=RESAMPLE_S)
    n_after_resample = len(df)

    df = _project_to_enu(df, frame)
    df, n_offline_qnh = _derive_geometric_altitude(df, metar, frame)

    stats = CleanStats(
        n_in=n_in,
        n_after_dedup=n_after_dedup,
        n_after_outliers=n_after_outliers,
        n_after_resample=n_after_resample,
        n_tracks=int(df["icao24"].nunique()),
        n_offline_qnh=int(n_offline_qnh),
    )
    log.info("adsb_clean: %s", stats.as_dict())
    return df, stats


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _drop_invalid_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["lon_wgs", "lat_wgs"]).copy()
    df = df[(df["lat_wgs"].between(-90, 90)) & (df["lon_wgs"].between(-180, 180))]
    df = df.drop_duplicates(subset=["icao24", "time_utc"], keep="first")
    df = df.sort_values(["icao24", "time_utc"]).reset_index(drop=True)
    return df


def _reject_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that exhibit unphysical step changes from the previous obs of the same icao24."""
    if df.empty:
        return df
    df = df.copy()
    g = df.groupby("icao24", sort=False, group_keys=False)
    df["_dt_s"] = g["time_utc"].diff().dt.total_seconds()
    # Use best altitude available for step check.
    z_pref = df["geo_alt_m"].where(df["geo_alt_m"].notna(), df["baro_alt_m"])
    df["_alt_step"] = z_pref.groupby(df["icao24"]).diff().abs()
    df["_v_step"] = g["velocity_ms"].diff().abs()

    bad = pd.Series(False, index=df.index)
    # Only apply when dt is sane (>0 and < 1 hr).
    sane_dt = (df["_dt_s"] > 0) & (df["_dt_s"] < 3600)
    bad |= sane_dt & (df["_alt_step"] > ALT_JUMP_MAX_M)
    bad |= sane_dt & (df["_v_step"] > SPEED_JUMP_MAX_MS)
    # Negative altitudes (below sea level by > 200 m) are unphysical.
    bad |= (z_pref < -200.0)
    df = df.loc[~bad].drop(columns=["_dt_s", "_alt_step", "_v_step"]).reset_index(drop=True)
    return df


def _resample_per_track(df: pd.DataFrame, period_s: int = 5) -> pd.DataFrame:
    """Bin to ``period_s`` seconds per icao24 using the median of fields within the bin."""
    if df.empty:
        return df
    df = df.copy()
    rule = f"{int(period_s)}s"
    out = []
    num_cols = ["lon_wgs", "lat_wgs", "baro_alt_m", "geo_alt_m",
                "velocity_ms", "track_deg", "vert_rate_ms"]
    for icao24, g in df.groupby("icao24", sort=False):
        gi = g.set_index("time_utc").sort_index()
        agg = {c: "median" for c in num_cols if c in gi.columns}
        if "on_ground" in gi.columns:
            agg["on_ground"] = "max"   # 'sticky' if on ground in any sub-interval
        if "callsign" in gi.columns:
            agg["callsign"] = "last"
        r = gi.resample(rule).agg(agg).dropna(subset=["lon_wgs", "lat_wgs"])
        r["icao24"] = icao24
        out.append(r.reset_index())
    if not out:
        return df.iloc[0:0]
    return pd.concat(out, ignore_index=True).sort_values(["icao24", "time_utc"]).reset_index(drop=True)


def _project_to_enu(df: pd.DataFrame, frame: AirportFrame) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        for c in ("x_m", "y_m"):
            df[c] = pd.Series(dtype=np.float32)
        return df
    df = df.copy()
    x, y = frame.wgs_to_enu(df["lon_wgs"].to_numpy(), df["lat_wgs"].to_numpy())
    df["x_m"] = x.astype(np.float32)
    df["y_m"] = y.astype(np.float32)
    return df


def _baro_to_geometric_m(baro_alt_m: np.ndarray, qnh_hpa: np.ndarray) -> np.ndarray:
    """Convert pressure altitude to geometric altitude using a station QNH (hPa).

    Geometric ≈ baro + (QNH − 1013.25) × 27 ft  =  baro + (QNH − 1013.25) × 8.2296 m.
    This is the standard ICAO altimeter correction for the low-altitude regime
    relevant to airport airspace (< ~5000 ft); it's good to ±a few metres.
    """
    return baro_alt_m + (qnh_hpa - ISA_QNH_HPA) * 8.2296


def _derive_geometric_altitude(df: pd.DataFrame,
                               metar: Optional[pd.DataFrame],
                               frame: AirportFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        for c in ("z_msl_m", "z_agl_m"):
            df[c] = pd.Series(dtype=np.float32)
        return df, 0

    qnh_series = None
    n_offline_qnh = 0
    if metar is not None and not metar.empty and "altim_hpa" in metar.columns:
        m = metar.dropna(subset=["altim_hpa"]).copy()
        if not m.empty:
            if not pd.api.types.is_datetime64_any_dtype(m["time_utc"]):
                m["time_utc"] = pd.to_datetime(m["time_utc"], utc=True)
            elif m["time_utc"].dt.tz is None:
                m["time_utc"] = m["time_utc"].dt.tz_localize("UTC")
            m = m.sort_values("time_utc")
            qnh_series = pd.merge_asof(
                df[["time_utc"]].sort_values("time_utc").reset_index(),
                m[["time_utc", "altim_hpa"]],
                on="time_utc", direction="nearest",
                tolerance=pd.Timedelta("90min"),
            ).set_index("index")["altim_hpa"].reindex(df.index)

    if qnh_series is None:
        qnh = np.full(len(df), ISA_QNH_HPA, dtype=np.float32)
        n_offline_qnh = len(df)
    else:
        qnh = qnh_series.fillna(ISA_QNH_HPA).to_numpy(dtype=np.float32)
        n_offline_qnh = int(qnh_series.isna().sum())

    baro = df["baro_alt_m"].to_numpy(dtype=np.float32)
    geo = df["geo_alt_m"].to_numpy(dtype=np.float32)
    derived = _baro_to_geometric_m(baro, qnh).astype(np.float32)
    z_msl = np.where(np.isfinite(geo), geo, derived)
    df = df.copy()
    df["z_msl_m"] = z_msl.astype(np.float32)
    df["z_agl_m"] = (df["z_msl_m"] - float(frame.elev_m)).astype(np.float32)
    return df, n_offline_qnh
