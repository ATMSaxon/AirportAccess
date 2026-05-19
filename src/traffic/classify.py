"""Per-track arrival/departure/overflight classification and runway-end assignment.

We work with cleaned ADS-B (see ``adsb_clean.py``) in airport ENU coordinates.

Track ⇒ category:
    * **arrival**   — descends overall, ends at/near field elevation within the
                      airport horizon (≈ 25 km), final speed is approach-speed
                      regime, and the trajectory enters from outside.
    * **departure** — ascends overall, starts at/near field elevation within the
                      airport horizon, initial speed is takeoff-speed regime.
    * **overflight**— stays high (> ``OVERFLIGHT_MIN_AGL_M``) for the whole pass
                      and never gets within the runway-axis cone of any threshold.

Runway-end assignment uses a simple geometric test:
    * Pick the candidate threshold whose **bearing-to-track** is most consistent
      with the runway centreline:
        - for an arrival, the *segment from outside-approach-point to threshold*
          should be aligned with the runway bearing within ±``BEARING_TOL_DEG``°.
        - for a departure, the *segment from threshold to outside-departure-point*
          should be aligned with the runway bearing within ±``BEARING_TOL_DEG``°.
    * Among candidates that pass the bearing test, pick the one with smallest
      lateral distance to the runway centreline at the touchdown/lift-off point.

If nothing matches, the track is left with ``runway_end_id = None``.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np
import pandas as pd

from ..utils.crs import AirportFrame
from ..utils.logs import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (module-level so tests can override)
# ---------------------------------------------------------------------------
ARRIVAL_MAX_DIST_M: float = 25_000.0      # within 25 km of ARP at touchdown
DEPARTURE_MAX_DIST_M: float = 25_000.0
LOW_AGL_M: float = 50.0                   # "on/near ground" threshold
OVERFLIGHT_MIN_AGL_M: float = 1500.0      # always above this → overflight
BEARING_TOL_DEG: float = 15.0
RUNWAY_AXIS_LATERAL_TOL_M: float = 2_000.0
APPROACH_SPEED_MIN_MS: float = 30.0       # ~ 60 kt — captures GA + jets
APPROACH_SPEED_MAX_MS: float = 120.0      # ~ 230 kt
DEPARTURE_SPEED_MIN_MS: float = 40.0
MIN_TRACK_POINTS: int = 4


@dataclass
class RunwayEnd:
    """One end of a runway — the (thr, end) pair is the *direction of operation*.

    For an arrival, aircraft land **toward** ``end`` from outside; for a departure,
    aircraft lift off **from** ``thr`` toward ``end``. Bearing is the heading
    *along* the operating direction.
    """
    runway_id: str
    thr_x: float; thr_y: float
    end_x: float; end_y: float
    bearing_deg: float                    # heading from thr toward end (deg true)
    length_m: float

    @property
    def axis(self) -> np.ndarray:
        v = np.array([self.end_x - self.thr_x, self.end_y - self.thr_y], dtype=float)
        n = np.linalg.norm(v)
        return v / max(n, 1e-9)

    @property
    def thr_xy(self) -> np.ndarray:
        return np.array([self.thr_x, self.thr_y], dtype=float)

    @property
    def end_xy(self) -> np.ndarray:
        return np.array([self.end_x, self.end_y], dtype=float)


@dataclass
class TrackRecord:
    icao24: str
    callsign: str
    n_points: int
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    entry_x: float; entry_y: float; entry_z_agl: float; entry_speed: float
    exit_x: float;  exit_y: float;  exit_z_agl: float;  exit_speed: float
    min_dist_arp_m: float
    mean_vert_rate_ms: float
    category: str                  # 'arrival' / 'departure' / 'overflight' / 'unknown'
    runway_end_id: str | None
    runway_assign_lateral_m: float

    def as_dict(self) -> dict:
        return {**asdict(self), "t_start": self.t_start.isoformat(),
                "t_end": self.t_end.isoformat()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def airport_runway_ends(airport_cfg: dict, frame: AirportFrame) -> list[RunwayEnd]:
    """Build one ``RunwayEnd`` per `(thr, end)` direction from the airport YAML.

    The airport YAML lists each runway end as a separate entry (e.g. 06L + 24R);
    each one corresponds to one *direction of operation*.
    """
    ends: list[RunwayEnd] = []
    for r in airport_cfg["runways"]:
        thr_x, thr_y = frame.wgs_to_enu(np.array([r["thr_lon"]]), np.array([r["thr_lat"]]))
        end_x, end_y = frame.wgs_to_enu(np.array([r["end_lon"]]), np.array([r["end_lat"]]))
        length_m = float(r.get("length_ft", 0.0)) * 0.3048
        ends.append(RunwayEnd(
            runway_id=str(r["id"]),
            thr_x=float(thr_x[0]), thr_y=float(thr_y[0]),
            end_x=float(end_x[0]), end_y=float(end_y[0]),
            bearing_deg=float(r.get("bearing_deg", _bearing(thr_x[0], thr_y[0], end_x[0], end_y[0]))),
            length_m=length_m or _dist(thr_x[0], thr_y[0], end_x[0], end_y[0]),
        ))
    return ends


def classify_tracks(adsb: pd.DataFrame,
                    runway_ends: Iterable[RunwayEnd]) -> pd.DataFrame:
    """Classify every icao24 track and (if applicable) assign a runway end.

    Returns a DataFrame indexed by ``icao24`` with the fields of ``TrackRecord``.
    """
    runway_ends = list(runway_ends)
    recs: list[TrackRecord] = []
    if adsb.empty:
        return pd.DataFrame(columns=[f.name for f in TrackRecord.__dataclass_fields__.values()])

    for icao24, g in adsb.groupby("icao24", sort=False):
        g = g.sort_values("time_utc").reset_index(drop=True)
        if len(g) < MIN_TRACK_POINTS:
            continue
        rec = _classify_one(icao24, g, runway_ends)
        recs.append(rec)
    if not recs:
        return pd.DataFrame(columns=[f.name for f in TrackRecord.__dataclass_fields__.values()])
    out = pd.DataFrame([r.as_dict() for r in recs])
    out["t_start"] = pd.to_datetime(out["t_start"], utc=True)
    out["t_end"] = pd.to_datetime(out["t_end"], utc=True)
    return out


# ---------------------------------------------------------------------------
# Per-track classification
# ---------------------------------------------------------------------------

def _classify_one(icao24: str, g: pd.DataFrame, runway_ends: list[RunwayEnd]) -> TrackRecord:
    xs = g["x_m"].to_numpy(); ys = g["y_m"].to_numpy()
    zs_agl = g["z_agl_m"].to_numpy()
    speed = g["velocity_ms"].to_numpy()
    vrate = g["vert_rate_ms"].to_numpy()
    on_ground = g["on_ground"].to_numpy() if "on_ground" in g.columns else np.zeros(len(g), bool)

    d_arp = np.hypot(xs, ys)
    entry_idx, exit_idx = 0, len(g) - 1

    near_ground_start = zs_agl[entry_idx] < LOW_AGL_M or bool(on_ground[entry_idx])
    near_ground_end = zs_agl[exit_idx] < LOW_AGL_M or bool(on_ground[exit_idx])
    near_arp_start = d_arp[entry_idx] < ARRIVAL_MAX_DIST_M
    near_arp_end = d_arp[exit_idx] < ARRIVAL_MAX_DIST_M
    overall_descent = zs_agl[exit_idx] - zs_agl[entry_idx] < -150.0
    overall_climb = zs_agl[exit_idx] - zs_agl[entry_idx] > 150.0
    speed_ok_arrival = APPROACH_SPEED_MIN_MS <= np.nan_to_num(speed[exit_idx], nan=70) <= APPROACH_SPEED_MAX_MS
    speed_ok_depart = np.nan_to_num(speed[entry_idx], nan=70) >= DEPARTURE_SPEED_MIN_MS
    sliding_min_d = float(_sliding_min(d_arp, window=5))

    category = "unknown"
    if (near_ground_end and near_arp_end and overall_descent and sliding_min_d < ARRIVAL_MAX_DIST_M):
        category = "arrival"
    elif (near_ground_start and near_arp_start and overall_climb and speed_ok_depart):
        category = "departure"
    elif np.nanmin(zs_agl) > OVERFLIGHT_MIN_AGL_M:
        category = "overflight"
    elif near_arp_end and overall_descent and speed_ok_arrival:
        category = "arrival"

    rwy_id, lateral = None, float("nan")
    if category == "arrival":
        rwy_id, lateral = _assign_runway_for_arrival(xs, ys, runway_ends, exit_idx)
    elif category == "departure":
        rwy_id, lateral = _assign_runway_for_departure(xs, ys, runway_ends, entry_idx)

    return TrackRecord(
        icao24=str(icao24),
        callsign=str(g["callsign"].iloc[-1]) if "callsign" in g.columns else "",
        n_points=int(len(g)),
        t_start=g["time_utc"].iloc[0],
        t_end=g["time_utc"].iloc[-1],
        entry_x=float(xs[entry_idx]), entry_y=float(ys[entry_idx]),
        entry_z_agl=float(zs_agl[entry_idx]), entry_speed=float(np.nan_to_num(speed[entry_idx])),
        exit_x=float(xs[exit_idx]), exit_y=float(ys[exit_idx]),
        exit_z_agl=float(zs_agl[exit_idx]), exit_speed=float(np.nan_to_num(speed[exit_idx])),
        min_dist_arp_m=float(np.nanmin(d_arp)),
        mean_vert_rate_ms=float(np.nanmean(vrate)) if np.any(np.isfinite(vrate)) else float("nan"),
        category=category,
        runway_end_id=rwy_id,
        runway_assign_lateral_m=float(lateral),
    )


# ---------------------------------------------------------------------------
# Runway-end assignment
# ---------------------------------------------------------------------------

def _assign_runway_for_arrival(xs: np.ndarray, ys: np.ndarray,
                               runway_ends: list[RunwayEnd], exit_idx: int) -> tuple[str | None, float]:
    """Pick the runway end whose centreline is best matched by the final approach segment.

    Final-approach direction = bearing of last 5 ENU points (or available).
    A landing on rwy ``XX`` heads INTO the runway bearing; we test bearing alignment
    between the approach direction and the runway bearing within ±15°.
    """
    seg_x, seg_y = xs[max(0, exit_idx - 4):exit_idx + 1], ys[max(0, exit_idx - 4):exit_idx + 1]
    if len(seg_x) < 2:
        return None, float("nan")
    approach_bearing = _bearing(seg_x[0], seg_y[0], seg_x[-1], seg_y[-1])
    px, py = xs[exit_idx], ys[exit_idx]
    return _best_runway(approach_bearing, px, py, runway_ends, landing=True)


def _assign_runway_for_departure(xs: np.ndarray, ys: np.ndarray,
                                 runway_ends: list[RunwayEnd], entry_idx: int) -> tuple[str | None, float]:
    seg_x, seg_y = xs[entry_idx:entry_idx + 5], ys[entry_idx:entry_idx + 5]
    if len(seg_x) < 2:
        return None, float("nan")
    depart_bearing = _bearing(seg_x[0], seg_y[0], seg_x[-1], seg_y[-1])
    px, py = xs[entry_idx], ys[entry_idx]
    return _best_runway(depart_bearing, px, py, runway_ends, landing=False)


def _best_runway(track_bearing: float, px: float, py: float,
                 runway_ends: list[RunwayEnd], landing: bool) -> tuple[str | None, float]:
    best_id, best_score = None, float("inf")
    best_lateral = float("nan")
    for r in runway_ends:
        # Landing into runway bearing — track direction ≈ runway bearing.
        # Departure climbing out — track direction ≈ runway bearing (departing FROM thr toward end).
        delta = _bearing_diff(track_bearing, r.bearing_deg)
        if abs(delta) > BEARING_TOL_DEG:
            continue
        lateral = _lateral_offset(px, py, r)
        # touchdown/lift-off point should be on the runway slab — strong penalty if far
        if lateral > RUNWAY_AXIS_LATERAL_TOL_M:
            continue
        # along-runway position (positive = past threshold, negative = pre-threshold)
        along = _along_track(px, py, r)
        # For arrivals we expect along ∈ [0, length]; for departures roughly same.
        if landing:
            band_penalty = max(0.0, -along) + max(0.0, along - 1.2 * r.length_m)
        else:
            band_penalty = max(0.0, -along) + max(0.0, along - 1.5 * r.length_m)
        score = lateral + 0.3 * band_penalty + 50.0 * abs(delta)
        if score < best_score:
            best_score = score
            best_id = r.runway_id
            best_lateral = lateral
    return best_id, best_lateral


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _bearing(x0, y0, x1, y1) -> float:
    """Bearing (deg true, 0=N, clockwise) from (x0,y0) to (x1,y1) in ENU."""
    dx, dy = x1 - x0, y1 - y0
    return float((np.degrees(np.arctan2(dx, dy))) % 360.0)


def _bearing_diff(a, b) -> float:
    """Signed minimal angular distance a-b in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(d)


def _dist(x0, y0, x1, y1) -> float:
    return float(np.hypot(x1 - x0, y1 - y0))


def _lateral_offset(px: float, py: float, r: RunwayEnd) -> float:
    v = np.array([px - r.thr_x, py - r.thr_y], dtype=float)
    axis = r.axis
    proj = v @ axis
    perp = v - proj * axis
    return float(np.linalg.norm(perp))


def _along_track(px: float, py: float, r: RunwayEnd) -> float:
    v = np.array([px - r.thr_x, py - r.thr_y], dtype=float)
    return float(v @ r.axis)


def _sliding_min(arr: np.ndarray, window: int = 5) -> float:
    if len(arr) < window:
        return float(np.nanmin(arr))
    s = np.lib.stride_tricks.sliding_window_view(arr, window)
    return float(np.nanmin(np.nanmin(s, axis=-1)))
