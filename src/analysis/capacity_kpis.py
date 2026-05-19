"""Capacity KPIs — runway delay via a SimPy DES, throughput preservation, etc.

The DES is intentionally tiny: each active runway end is a one-capacity SimPy ``Resource``,
ADS-B-derived arrivals seize it for a service window, and the corridor — if it crosses an
active runway centreline — injects an additional 30-second resource-blocking event at the
appropriate simulated time. We run the simulation twice (with and without the eVTOL event)
to extract the *extra* delay attributable to the corridor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

LOG = get_logger(__name__)


@dataclass
class CapacityKPI:
    runway_delay_extra_s: float | None
    throughput_preservation: float | None
    evtol_ops_per_hour: float
    corridor_closure_rate_pct: float | None
    atc_intervention_proxy: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def _arrivals_in_hour(adsb_arrivals: pd.DataFrame | None, hour: int) -> list[float]:
    """Return arrival times in seconds-from-hour-start.

    Expects ``adsb_arrivals`` to be a DataFrame with at least ``time_utc`` and an arrival
    flag. We allow the column to be named ``arrival_flag`` (bool) or use the entire
    DataFrame if no such column exists (caller's responsibility to pre-filter).
    """
    if adsb_arrivals is None or len(adsb_arrivals) == 0:
        return []
    df = adsb_arrivals
    if "arrival_flag" in df.columns:
        df = df[df["arrival_flag"].astype(bool)]
    if "time_utc" not in df.columns:
        return []
    t = pd.to_datetime(df["time_utc"], utc=True)
    # Use the calendar day of the first arrival; isolate the requested hour.
    day = pd.Timestamp(t.iloc[0].date()).tz_localize("UTC") if len(t) else pd.Timestamp.utcnow()
    hour_start = day + pd.Timedelta(hours=hour)
    hour_end = hour_start + pd.Timedelta(hours=1)
    in_hour = t[(t >= hour_start) & (t < hour_end)]
    return sorted([(ts - hour_start).total_seconds() for ts in in_hour])


def _evtol_crossing_time(corridor, *, frame: AirportFrame, airport_cfg: dict, cruise_mps: float) -> float | None:
    """Return the arc-length time (s) at which the corridor first crosses any runway centreline."""
    if not corridor.feasible or corridor.path_enu is None or len(corridor.path_enu) < 2:
        return None
    px = corridor.path_enu[:, 0].astype(np.float64)
    py = corridor.path_enu[:, 1].astype(np.float64)
    seg = np.diff(corridor.path_enu[:, :2], axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    for rwy in airport_cfg.get("runways", []):
        tx, ty = frame.wgs_to_enu(np.array([rwy["thr_lon"]]), np.array([rwy["thr_lat"]]))
        ex, ey = frame.wgs_to_enu(np.array([rwy["end_lon"]]), np.array([rwy["end_lat"]]))
        thr = np.array([float(tx[0]), float(ty[0])])
        end = np.array([float(ex[0]), float(ey[0])])
        unit = end - thr
        L = float(np.linalg.norm(unit))
        if L < 1.0:
            continue
        unit /= L
        rel_x = px - thr[0]
        rel_y = py - thr[1]
        along = rel_x * unit[0] + rel_y * unit[1]
        perp = rel_x * (-unit[1]) + rel_y * unit[0]
        sgn = np.sign(perp)
        for i in range(len(sgn) - 1):
            if sgn[i] != sgn[i + 1] and 0 <= along[i] <= L:
                return float(arc[i] / max(cruise_mps, 1e-6))
    return None


def _run_des(arrival_times_s: list[float], service_s: float,
             evtol_event_s: float | None, evtol_block_s: float) -> tuple[float, int]:
    """Run the mini SimPy DES; return (total_wait_s, served_count)."""
    import simpy

    env = simpy.Environment()
    runway = simpy.Resource(env, capacity=1)
    wait_total = {"v": 0.0}
    served = {"n": 0}

    def arrival(t_target: float):
        yield env.timeout(t_target)
        t_req = env.now
        with runway.request() as req:
            yield req
            wait = env.now - t_req
            wait_total["v"] += wait
            served["n"] += 1
            yield env.timeout(service_s)

    def evtol(t_target: float):
        yield env.timeout(t_target)
        with runway.request() as req:
            yield req
            yield env.timeout(evtol_block_s)

    for t in arrival_times_s:
        env.process(arrival(float(t)))
    if evtol_event_s is not None:
        env.process(evtol(float(evtol_event_s)))
    env.run()
    return float(wait_total["v"]), int(served["n"])


def _corridor_closure_rate(corridor, envelopes_T: np.ndarray | None) -> float | None:
    """% of time slices in which the corridor would be infeasible under E_t.

    Counts a slice as "closed" if any waypoint along the corridor has
    ``envelopes_T[t, i, j, k] == False``.
    """
    if envelopes_T is None or corridor.path_ijk is None:
        return None
    if envelopes_T.ndim != 4:
        return None
    T = envelopes_T.shape[0]
    n_closed = 0
    ijk = corridor.path_ijk
    for t in range(T):
        slc = envelopes_T[t]
        try:
            if (~slc[ijk[:, 0], ijk[:, 1], ijk[:, 2]]).any():
                n_closed += 1
        except IndexError:
            continue
    return 100.0 * n_closed / max(T, 1)


def _atc_proxy(corridor, adsb_arrivals, frame: AirportFrame,
               lateral_thresh_nm: float = 0.5, vertical_thresh_ft: float = 500.0) -> int | None:
    """Count distinct ADS-B aircraft whose minimum separation drops below thresholds."""
    if adsb_arrivals is None or corridor.path_enu is None or len(corridor.path_enu) == 0:
        return None
    if not {"x_m", "y_m", "z_msl_m", "icao24"}.issubset(set(adsb_arrivals.columns)):
        return None
    px = corridor.path_enu[:, 0].astype(np.float64)
    py = corridor.path_enu[:, 1].astype(np.float64)
    pz = corridor.path_enu[:, 2].astype(np.float64)
    lateral_m = lateral_thresh_nm * 1852.0
    vertical_m = vertical_thresh_ft * 0.3048
    bad = set()
    stride = max(1, len(px) // 50)
    grouped = adsb_arrivals.groupby("icao24")
    for icao, sub in grouped:
        x = sub["x_m"].to_numpy(dtype=np.float64)
        y = sub["y_m"].to_numpy(dtype=np.float64)
        z = sub["z_msl_m"].to_numpy(dtype=np.float64)
        for k in range(0, len(px), stride):
            d_lat = np.hypot(x - px[k], y - py[k])
            d_vert = np.abs(z - pz[k])
            if np.any((d_lat < lateral_m) & (d_vert < vertical_m)):
                bad.add(icao)
                break
    return int(len(bad))


def capacity_for_corridor(
    corridor,
    *,
    adsb_arrivals: pd.DataFrame | None,
    envelopes_T: np.ndarray | None,
    airport_cfg: dict,
    frame: AirportFrame,
    service_time_s: float = 60.0,
    evtol_block_s: float = 30.0,
    cruise_mps: float = 67.0,
    seed: int = 0,
) -> CapacityKPI:
    """Compute capacity KPIs for a single corridor."""
    hour = int(corridor.hour or 0)

    # Throughput preservation = (with_evtol_throughput) / (without_evtol_throughput) over the hour.
    runway_delay_extra: float | None = None
    throughput_pres: float | None = None
    if adsb_arrivals is not None and corridor.feasible:
        arrivals = _arrivals_in_hour(adsb_arrivals, hour)
        if arrivals:
            cross_t = _evtol_crossing_time(corridor, frame=frame, airport_cfg=airport_cfg,
                                            cruise_mps=cruise_mps)
            wait_no, served_no = _run_des(arrivals, service_time_s, None, evtol_block_s)
            wait_yes, served_yes = _run_des(arrivals, service_time_s,
                                            cross_t if cross_t is not None else 0.0, evtol_block_s)
            runway_delay_extra = wait_yes - wait_no
            throughput_pres = served_yes / max(served_no, 1)
    elif corridor.feasible:
        runway_delay_extra = 0.0
        throughput_pres = 1.0

    # eVTOL ops/h (cycle time = corridor.time_s + 60 s ground turn).
    evtol_ops_per_hour = 0.0
    if corridor.feasible and corridor.time_s > 0:
        evtol_ops_per_hour = 3600.0 / (corridor.time_s + 60.0)

    closure_pct = _corridor_closure_rate(corridor, envelopes_T)
    atc_n = _atc_proxy(corridor, adsb_arrivals, frame)

    # If corridor.feasible is False overall, override.
    if not corridor.feasible:
        evtol_ops_per_hour = 0.0
        if closure_pct is None:
            closure_pct = 100.0

    return CapacityKPI(
        runway_delay_extra_s=runway_delay_extra,
        throughput_preservation=throughput_pres,
        evtol_ops_per_hour=float(evtol_ops_per_hour),
        corridor_closure_rate_pct=closure_pct,
        atc_intervention_proxy=atc_n,
    )
