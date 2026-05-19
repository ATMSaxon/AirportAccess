"""15-min rolling runway-configuration inference from classified tracks.

For each 15-min slice ``t``, count operations per runway end and report the
*active configuration*: a sorted tuple of runway ends that account for
``ACTIVE_SHARE_THRESHOLD`` of operations, separately for arrivals and departures.

Confidence = winner share (top-1 share of operations among assigned tracks).

METAR cross-check (per slice):
  predicted active landing runway bearing should be within ±60° of the METAR
  mean wind direction (winds blow FROM that direction; aircraft land INTO it).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd

from ..utils.logs import get_logger

log = get_logger(__name__)

ACTIVE_SHARE_THRESHOLD: float = 0.5    # at least 50% of ops on this runway to be 'active'
WIND_BEARING_TOL_DEG: float = 60.0


@dataclass
class ConfigSlice:
    slice_start: pd.Timestamp
    slice_end: pd.Timestamp
    arrivals_active: tuple[str, ...]
    departures_active: tuple[str, ...]
    n_arrivals: int
    n_departures: int
    arr_share: float                   # top-1 share among arrivals
    dep_share: float                   # top-1 share among departures
    metar_wind_dir_deg: float | None
    metar_wind_kt: float | None
    metar_match: bool | None
    flight_rule: str | None
    visibility_sm: float | None
    ceiling_ft: float | None


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

def rolling_config(tracks: pd.DataFrame,
                   day_start: pd.Timestamp,
                   day_end: pd.Timestamp,
                   interval_min: int = 15,
                   metar: Optional[pd.DataFrame] = None,
                   runway_bearings: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """Return a DataFrame with one row per ``interval_min`` slice.

    Parameters
    ----------
    tracks
        Output of ``classify.classify_tracks`` — must contain ``t_end``, ``t_start``,
        ``category``, ``runway_end_id``.
    day_start, day_end
        UTC bracketing the day of interest.
    interval_min
        Slice length in minutes (default 15).
    metar
        Optional METAR DataFrame (D6 schema) — used for cross-check.
    runway_bearings
        ``{runway_id: bearing_deg}`` (heading of the runway from threshold toward
        the far end). Required if you want METAR cross-check.
    """
    if day_start.tzinfo is None:
        day_start = day_start.tz_localize("UTC")
    if day_end.tzinfo is None:
        day_end = day_end.tz_localize("UTC")
    edges = pd.date_range(day_start, day_end, freq=f"{int(interval_min)}min", tz="UTC")
    if len(edges) < 2:
        return pd.DataFrame()
    rows: list[ConfigSlice] = []

    if tracks is None or tracks.empty:
        tracks = pd.DataFrame(columns=["t_start", "t_end", "category", "runway_end_id"])

    if not tracks.empty:
        tracks = tracks.copy()
        for c in ("t_start", "t_end"):
            if not pd.api.types.is_datetime64_any_dtype(tracks[c]):
                tracks[c] = pd.to_datetime(tracks[c], utc=True)
            elif tracks[c].dt.tz is None:
                tracks[c] = tracks[c].dt.tz_localize("UTC")

    if metar is not None and not metar.empty:
        metar = metar.copy()
        if not pd.api.types.is_datetime64_any_dtype(metar["time_utc"]):
            metar["time_utc"] = pd.to_datetime(metar["time_utc"], utc=True)
        elif metar["time_utc"].dt.tz is None:
            metar["time_utc"] = metar["time_utc"].dt.tz_localize("UTC")
        metar = metar.sort_values("time_utc")

    for s, e in zip(edges[:-1], edges[1:]):
        # An "operation" belongs to a slice when t_end (touchdown for arrivals,
        # liftoff for departures) falls in [s, e).
        in_slice = tracks[(tracks["t_end"] >= s) & (tracks["t_end"] < e)]
        arr = in_slice[(in_slice["category"] == "arrival") & in_slice["runway_end_id"].notna()]
        dep = in_slice[(in_slice["category"] == "departure") & in_slice["runway_end_id"].notna()]
        arr_active, arr_share = _winners(arr["runway_end_id"], ACTIVE_SHARE_THRESHOLD)
        dep_active, dep_share = _winners(dep["runway_end_id"], ACTIVE_SHARE_THRESHOLD)
        m = _metar_for_slice(metar, s, e) if metar is not None else None
        wind_dir = float(m["wind_dir_deg"]) if m is not None and "wind_dir_deg" in m else None
        wind_kt = float(m["wind_kt"]) if m is not None and "wind_kt" in m else None
        vis_sm = float(m["vis_sm"]) if m is not None and "vis_sm" in m else None
        ceiling = float(m["ceiling_ft"]) if m is not None and "ceiling_ft" in m else None
        flight_rule = str(m["flight_rule"]) if m is not None and "flight_rule" in m else None

        match = None
        if wind_dir is not None and arr_active and runway_bearings:
            # Compute the minimum bearing-difference between any active landing
            # runway heading and the wind FROM-direction.
            diffs = [abs(_bearing_diff(runway_bearings.get(r, np.nan), wind_dir))
                     for r in arr_active if not np.isnan(runway_bearings.get(r, np.nan))]
            if diffs:
                match = bool(min(diffs) <= WIND_BEARING_TOL_DEG)

        rows.append(ConfigSlice(
            slice_start=s, slice_end=e,
            arrivals_active=tuple(arr_active),
            departures_active=tuple(dep_active),
            n_arrivals=int(len(arr)),
            n_departures=int(len(dep)),
            arr_share=float(arr_share),
            dep_share=float(dep_share),
            metar_wind_dir_deg=wind_dir,
            metar_wind_kt=wind_kt,
            metar_match=match,
            flight_rule=flight_rule,
            visibility_sm=vis_sm,
            ceiling_ft=ceiling,
        ))
    df = pd.DataFrame([_to_record(r) for r in rows])
    return df


def _to_record(s: ConfigSlice) -> dict:
    d = s.__dict__.copy()
    # Tuples → JSON-friendly comma string for parquet (lists also fine).
    d["arrivals_active"] = list(s.arrivals_active)
    d["departures_active"] = list(s.departures_active)
    return d


def _winners(series: pd.Series, share_threshold: float) -> tuple[list[str], float]:
    """Return the runways that together cover ``share_threshold`` of operations.

    Algorithm: rank runways by descending count; keep adding to the active set
    until cumulative share ≥ threshold. ``winner_share`` is the top-1 share.
    """
    if series.empty:
        return [], 0.0
    counts = series.value_counts()
    total = int(counts.sum())
    cum = (counts.cumsum() / total)
    keep = []
    for rwy, c_share in cum.items():
        keep.append(str(rwy))
        if c_share >= share_threshold:
            break
    winner_share = float(counts.iloc[0] / total)
    return sorted(keep), winner_share


def _metar_for_slice(metar: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp):
    """Return a single METAR-like record (median of fields in slice, or nearest)."""
    sub = metar[(metar["time_utc"] >= s - pd.Timedelta(minutes=30)) &
                (metar["time_utc"] < e + pd.Timedelta(minutes=30))]
    if sub.empty:
        return None
    rec = {}
    for c in ("wind_dir_deg", "wind_kt", "vis_sm", "ceiling_ft"):
        if c in sub.columns and sub[c].notna().any():
            if c == "wind_dir_deg":
                # circular mean
                rad = np.deg2rad(sub[c].dropna().to_numpy())
                rec[c] = float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)
            else:
                rec[c] = float(sub[c].median())
    if "flight_rule" in sub.columns and sub["flight_rule"].notna().any():
        rec["flight_rule"] = str(sub["flight_rule"].mode().iloc[0])
    return rec if rec else None


def _bearing_diff(a: float, b: float) -> float:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(d)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def metar_match_rate(slices: pd.DataFrame) -> float:
    """Fraction of slices (with both predicted runways and METAR wind) that match.

    Returns NaN if there's no scoreable slice.
    """
    s = slices.dropna(subset=["metar_match"])
    if s.empty:
        return float("nan")
    return float(s["metar_match"].astype(bool).mean())
