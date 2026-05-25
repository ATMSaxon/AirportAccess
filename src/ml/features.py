"""Feature extraction for the risk-field learner.

Per-segment (or per-grid-cell) features at the midpoint:

| name                  | source                                              |
|-----------------------|-----------------------------------------------------|
| d_OLS                 | `AirportGeom.sdf` magnitude → distance to nearest   |
|                       |  active OLS surface (m, positive outside)           |
| d_runway              | `AirportGeom.distance_to_nearest_runway` (m)        |
| d_approach            | distance to nearest *active* approach prism (m)     |
| d_departure           | distance to nearest *active* departure prism (m)    |
| traffic_density       | ADS-B counts/m³/s in a box around mid-point         |
| wind_dir              | METAR wind direction (deg, from)                    |
| wind_speed            | METAR wind speed (m/s)                              |
| visibility            | METAR visibility (m)                                |
| ceiling               | METAR ceiling height (m AGL, NaN→11000)             |
| runway_config_one_hot | 0/1 over the known configurations                   |
| hour_sin / hour_cos   | cyclical hour-of-day                                |

All extraction functions are pure; the orchestrator in this module loads the
right artefacts and produces a feature parquet aligned with the counterfactual
parquet on `seg_id`.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import logging
import math
import numpy as np
import pandas as pd

from ..utils.logs import get_logger
from ._geom import AirportGeom, adsb_density_box, NM_M

logger = get_logger(__name__)


KT_MS = 0.514444
SM_M = 1609.344
FT_M = 0.3048


# Canonical numeric feature ordering. Categorical configs are appended at the
# end as `cfg_<id>` one-hot columns.
NUMERIC_FEATURES = [
    "d_OLS_m", "d_runway_m", "d_approach_m", "d_departure_m",
    "traffic_density", "wind_dir_sin", "wind_dir_cos", "wind_speed_mps",
    "visibility_m", "ceiling_m",
    "hour_sin", "hour_cos",
]


# ---------------------------------------------------------------------------
# Weather joining
# ---------------------------------------------------------------------------
def _join_metar(times: pd.Series, metar_df: pd.DataFrame | None) -> pd.DataFrame:
    """Nearest-time merge of METAR onto the supplied times.

    Returns a DataFrame with columns `wind_dir_deg, wind_speed_mps, vis_m,
    ceiling_m`. Missing values are filled with conservative defaults:
    wind_dir_deg=NaN (encoded as sin=cos=0), wind_speed=0, vis=16000 m, ceiling=11000 m.
    """
    n = len(times)
    out = pd.DataFrame(
        {
            "wind_dir_deg": np.full(n, np.nan, dtype=np.float64),
            "wind_speed_mps": np.zeros(n, dtype=np.float64),
            "vis_m": np.full(n, 16000.0, dtype=np.float64),
            "ceiling_m": np.full(n, 11000.0, dtype=np.float64),
        }
    )
    if metar_df is None or len(metar_df) == 0:
        return out
    m = metar_df.copy()
    # Force tz-aware UTC; ASOS source writes naive datetime64[ns] with possible NaT rows.
    if pd.api.types.is_datetime64_any_dtype(m["time_utc"]):
        if m["time_utc"].dt.tz is None:
            m["time_utc"] = m["time_utc"].dt.tz_localize("UTC")
        else:
            m["time_utc"] = m["time_utc"].dt.tz_convert("UTC")
    else:
        m["time_utc"] = pd.to_datetime(m["time_utc"], utc=True)
    m = m.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)
    if len(m) == 0:
        return out

    t = pd.to_datetime(times.to_numpy(), utc=True)
    idx = np.searchsorted(m["time_utc"].to_numpy(), t)
    idx = np.clip(idx, 1, len(m) - 1)
    prev = m.iloc[idx - 1].reset_index(drop=True)
    nxt = m.iloc[idx].reset_index(drop=True)
    # Choose the closer of prev/next per row.
    dprev = np.abs((t - prev["time_utc"]).astype("timedelta64[s]").astype(np.int64))
    dnext = np.abs((nxt["time_utc"] - t).astype("timedelta64[s]").astype(np.int64))
    take_next = dnext < dprev
    chosen = nxt.where(take_next, prev)
    out["wind_dir_deg"] = chosen.get("wind_dir_deg", out["wind_dir_deg"]).to_numpy()
    if "wind_kt" in chosen.columns:
        out["wind_speed_mps"] = chosen["wind_kt"].to_numpy() * KT_MS
    if "vis_sm" in chosen.columns:
        out["vis_m"] = chosen["vis_sm"].to_numpy() * SM_M
    if "ceiling_ft" in chosen.columns:
        out["ceiling_m"] = chosen["ceiling_ft"].to_numpy() * FT_M
    out["wind_speed_mps"] = out["wind_speed_mps"].fillna(0.0)
    out["vis_m"] = out["vis_m"].fillna(16000.0)
    out["ceiling_m"] = out["ceiling_m"].fillna(11000.0)
    return out


# ---------------------------------------------------------------------------
# Feature engineering for a counterfactual segment table
# ---------------------------------------------------------------------------
def _build_fast_density_fn(adsb_df: pd.DataFrame,
                           half_xy_m: float = 1500.0,
                           half_z_m: float = 150.0,
                           half_t_s: float = 300.0):
    """Precompute sorted ADS-B arrays once; return a closure that answers
    density queries via ``np.searchsorted`` + vectorised xyz mask.

    Roughly 50-200× faster than `adsb_density_box` on multi-day ADS-B
    (which scans the whole DataFrame per call). Kept compatible with the
    ``density_fn(x, y, z, mt) -> float`` interface used in `extract_features`.
    """
    a = adsb_df.dropna(subset=["time_utc", "x_m", "y_m", "z_msl_m"])
    if len(a) == 0:
        return lambda x, y, z, mt: 0.0
    a = a.sort_values("time_utc").reset_index(drop=True)
    t_utc = pd.to_datetime(a["time_utc"], utc=True)
    if t_utc.dt.tz is not None:
        t_utc = t_utc.dt.tz_convert("UTC").dt.tz_localize(None)
    adsb_t = t_utc.to_numpy(dtype="datetime64[ns]")
    adsb_x = a["x_m"].to_numpy(dtype=np.float64)
    adsb_y = a["y_m"].to_numpy(dtype=np.float64)
    adsb_z = a["z_msl_m"].to_numpy(dtype=np.float64)
    half_t = np.timedelta64(int(half_t_s), "s")
    vol = (2 * half_xy_m) ** 2 * (2 * half_z_m)
    dt_window = 2 * half_t_s
    denom = vol * dt_window + 1e-12

    def density_fn(x: float, y: float, z: float, mt) -> float:
        t = pd.Timestamp(mt)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        t64 = np.datetime64(t.to_datetime64(), "ns")
        i_lo = int(np.searchsorted(adsb_t, t64 - half_t, side="left"))
        i_hi = int(np.searchsorted(adsb_t, t64 + half_t, side="right"))
        if i_hi <= i_lo:
            return 0.0
        xx = adsb_x[i_lo:i_hi]
        yy = adsb_y[i_lo:i_hi]
        zz = adsb_z[i_lo:i_hi]
        m = ((np.abs(xx - x) < half_xy_m)
             & (np.abs(yy - y) < half_xy_m)
             & (np.abs(zz - z) < half_z_m))
        return float(m.sum()) / denom

    return density_fn


def extract_features(seg_df: pd.DataFrame, geom: AirportGeom,
                     adsb_df: pd.DataFrame | None = None,
                     metar_df: pd.DataFrame | None = None,
                     density_fn=None) -> pd.DataFrame:
    """Vectorised feature extraction. Returns a DataFrame keyed on seg_id."""
    feats: dict[str, list[float]] = {k: [] for k in NUMERIC_FEATURES}
    cfg_ids: list[str] = []
    seg_ids: list[str] = []

    arrivals = seg_df["active_arrivals"].astype(str).str.split(";").apply(
        lambda L: [s for s in L if s])
    departures = seg_df["active_departures"].astype(str).str.split(";").apply(
        lambda L: [s for s in L if s])

    # Fast spatial+temporal ADS-B index — built once, queried per-segment.
    # Avoids the O(N) pandas filter inside `adsb_density_box` on every call,
    # which dominates runtime for multi-day ADS-B (5M+ rows × 200k segs).
    if density_fn is None and adsb_df is not None and len(adsb_df) > 0:
        density_fn = _build_fast_density_fn(adsb_df)

    for i, row in seg_df.reset_index(drop=True).iterrows():
        x, y, z = float(row["mid_x_m"]), float(row["mid_y_m"]), float(row["mid_z_m"])
        arr = arrivals.iloc[i]; dep = departures.iloc[i]
        d_ols = abs(geom.sdf(x, y, z, arr, dep))
        d_rwy = geom.distance_to_nearest_runway(x, y, z)
        d_app = geom.distance_to_active_approach(x, y, z, arr)
        d_dep = geom.distance_to_active_departure(x, y, z, dep)
        feats["d_OLS_m"].append(float(d_ols))
        feats["d_runway_m"].append(float(d_rwy))
        feats["d_approach_m"].append(float(d_app if math.isfinite(d_app) else 1e6))
        feats["d_departure_m"].append(float(d_dep if math.isfinite(d_dep) else 1e6))

        mt = pd.Timestamp(row["mid_t_utc"])
        if density_fn is not None:
            rho = float(density_fn(x, y, z, mt))
        else:
            rho = float(adsb_density_box(adsb_df, x, y, z, mt))
        feats["traffic_density"].append(rho)

        h = mt.hour + mt.minute / 60.0
        feats["hour_sin"].append(math.sin(2 * math.pi * h / 24.0))
        feats["hour_cos"].append(math.cos(2 * math.pi * h / 24.0))
        cfg_ids.append(str(row.get("config_id", "UNKNOWN")))
        seg_ids.append(str(row["seg_id"]))

    # Wind/vis/ceiling: bulk-merge to METAR by mid_t_utc.
    metar_join = _join_metar(seg_df["mid_t_utc"], metar_df)
    wind_dir_rad = np.deg2rad(metar_join["wind_dir_deg"].fillna(0.0).to_numpy())
    valid = ~metar_join["wind_dir_deg"].isna().to_numpy()
    feats["wind_dir_sin"] = (np.sin(wind_dir_rad) * valid).tolist()
    feats["wind_dir_cos"] = (np.cos(wind_dir_rad) * valid).tolist()
    feats["wind_speed_mps"] = metar_join["wind_speed_mps"].to_numpy().tolist()
    feats["visibility_m"] = metar_join["vis_m"].to_numpy().tolist()
    feats["ceiling_m"] = metar_join["ceiling_m"].to_numpy().tolist()

    # One-hot encode configs.
    cfg_series = pd.Series(cfg_ids, name="config_id")
    one_hot = pd.get_dummies(cfg_series, prefix="cfg", dtype=np.float32)

    out = pd.DataFrame({"seg_id": seg_ids, **feats})
    out = pd.concat([out, one_hot.reset_index(drop=True)], axis=1)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c == "seg_id" or c in NUMERIC_FEATURES or c.startswith("cfg_")]


def merge_features_labels(features_df: pd.DataFrame,
                          labels_df: pd.DataFrame,
                          label_col: str = "conflict") -> pd.DataFrame:
    """Inner join on seg_id, keep features + label + mid_t_utc + sample_date.

    ``sample_date`` is carried through when present so the trainer can build a
    per-day holdout (first N-1 sample_dates → train, last sample_date → test).
    """
    keep_cols = ["seg_id", "mid_t_utc", label_col]
    if "sample_date" in labels_df.columns:
        keep_cols.append("sample_date")
    out = features_df.merge(
        labels_df[keep_cols],
        on="seg_id", how="inner")
    return out


__all__ = [
    "NUMERIC_FEATURES",
    "extract_features",
    "feature_columns",
    "merge_features_labels",
    "_join_metar",
]
