"""Accessibility KPIs — eVTOL vs road, passenger-weighted, weather, peak service.

Default behaviour is fully offline-friendly:

* If ``OSRM_URL`` (env or arg) is unset → fall back to great-circle × 1.4 detour @ 50 km/h.
* If BTS DB1B parquet absent → uniform weight 1.0 per OD pair.
* If LAWA peaks CSV absent → eVTOL time savings not scaled by demand profile.
* If METAR absent → ``weather_reliability_pct = None`` and logged.

The 'access time saving vs road' is computed against a vertiport → terminal centroid trip.
Terminal centroid is approximated as the airport ARP unless overridden.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import pandas as pd

from src.data._common import great_circle_nm
from src.utils.logs import get_logger

LOG = get_logger(__name__)


@dataclass
class AccessibilityKPI:
    access_time_saving_min_vs_road: float
    passenger_weighted_access_score: float
    vertiport_to_terminal_transfer_min: float
    weather_reliability_pct: float | None
    peak_service_capacity_ops_per_hour: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Road time helpers
# ---------------------------------------------------------------------------

def _road_time_via_osrm(osrm_url: str, src_wgs: tuple[float, float], dst_wgs: tuple[float, float]) -> float | None:
    try:
        import requests
        u = osrm_url.rstrip("/")
        url = (
            f"{u}/route/v1/driving/"
            f"{src_wgs[0]},{src_wgs[1]};{dst_wgs[0]},{dst_wgs[1]}?overview=false"
        )
        r = requests.get(url, timeout=2.0)
        if r.status_code != 200:
            return None
        j = r.json()
        sec = float(j["routes"][0]["duration"])
        return sec / 60.0
    except Exception as e:  # noqa: BLE001
        LOG.debug("OSRM fallback (%s): %s", osrm_url, e)
        return None


def _road_time_proxy_min(src_wgs: tuple[float, float], dst_wgs: tuple[float, float],
                        *, surface_kmh: float = 50.0, detour: float = 1.4) -> float:
    """Great-circle × detour / surface speed in minutes."""
    nm = great_circle_nm(src_wgs[0], src_wgs[1], dst_wgs[0], dst_wgs[1])
    km = nm * 1.852
    return km * detour / surface_kmh * 60.0


def _walk_time_min(src_wgs: tuple[float, float], dst_wgs: tuple[float, float],
                   *, kmh: float = 5.0, buffer_min: float = 1.0) -> float:
    nm = great_circle_nm(src_wgs[0], src_wgs[1], dst_wgs[0], dst_wgs[1])
    km = nm * 1.852
    return km / kmh * 60.0 + buffer_min


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def accessibility_for_corridor(
    corridor,
    *,
    airport_cfg: dict,
    metar: pd.DataFrame | None = None,
    bts_od: pd.DataFrame | None = None,
    lawa_peaks: pd.DataFrame | None = None,
    osrm_url: str | None = None,
    terminal_centroid_wgs: tuple[float, float] | None = None,
    peak_hours: tuple[int, ...] = (8, 11, 17),
) -> AccessibilityKPI:
    """Compute all five accessibility KPIs."""
    if osrm_url is None:
        osrm_url = os.environ.get("OSRM_URL") or None

    # Source vertiport WGS (use first endpoint).
    src_id, dst_id = corridor.vertiport_pair
    verts = airport_cfg.get("vertiports", {})
    src_v = verts.get(src_id, {})
    dst_v = verts.get(dst_id, {})

    src_lon = float(src_v.get("lon", 0.0))
    src_lat = float(src_v.get("lat", 0.0))
    if terminal_centroid_wgs is None:
        term = (float(airport_cfg["arp"]["lon"]), float(airport_cfg["arp"]["lat"]))
    else:
        term = (float(terminal_centroid_wgs[0]), float(terminal_centroid_wgs[1]))

    # Road time: try OSRM, fallback to great-circle proxy.
    road_min: float | None = None
    if osrm_url:
        road_min = _road_time_via_osrm(osrm_url, (src_lon, src_lat), term)
    if road_min is None:
        road_min = _road_time_proxy_min((src_lon, src_lat), term)
        proxy_logged = True
    else:
        proxy_logged = False

    # eVTOL time = corridor time + 60s ground turn for fair comparison.
    evtol_min = (corridor.time_s + 60.0) / 60.0 if corridor.feasible else float("inf")
    saving_min = float(road_min - evtol_min) if corridor.feasible else 0.0

    # Walking transfer vertiport→terminal.
    walk_min = _walk_time_min((src_lon, src_lat), term)

    # Passenger weighting from BTS, LAWA peak share.
    weight = 1.0
    # BTS: accept both new `passengers` (D8) and legacy `pax_count` column names.
    if bts_od is not None:
        try:
            pax_col = None
            for cand in ("passengers", "pax_count"):
                if cand in bts_od.columns:
                    pax_col = cand
                    break
            if pax_col is not None:
                sub = bts_od
                iata = airport_cfg.get("iata", "")
                if iata and "origin" in sub.columns and "dest" in sub.columns:
                    sub = sub[
                        (sub["origin"].astype(str) == iata)
                        | (sub["dest"].astype(str) == iata)
                    ]
                total = float(sub[pax_col].sum())
                weight = max(total / 1e6, 0.1)
        except Exception as e:  # noqa: BLE001
            LOG.debug("BTS weighting failed (%s); falling back to uniform", e)
    # LAWA: accept both new `peak_hour` string ("08-09") + `trips`/`direction` (D7)
    # and legacy `hour` int + `share` columns.
    if lawa_peaks is not None and len(lawa_peaks) > 0:
        try:
            cols = set(lawa_peaks.columns)
            h = int(corridor.hour or 0)
            if {"peak_hour", "trips"}.issubset(cols):
                # Parse peak_hour like "08-09" → start hour.
                def _start_hour(s: str) -> int:
                    try:
                        return int(str(s).split("-", 1)[0])
                    except (ValueError, IndexError):
                        return -1
                df = lawa_peaks.copy()
                df["_h"] = df["peak_hour"].map(_start_hour)
                row = df[df["_h"] == h]
                if "direction" in row.columns:
                    row = row[row["direction"].astype(str).str.lower() == "in"]
                if len(row):
                    trips = float(row["trips"].sum())
                    total = float(df["trips"].sum()) or 1.0
                    share = trips / total
                    weight = float(weight * (1.0 + share))
            elif {"hour", "share"}.issubset(cols):
                row = lawa_peaks[lawa_peaks["hour"].astype(int) == h]
                if len(row):
                    weight = float(weight * (1.0 + float(row.iloc[0]["share"])))
        except Exception as e:  # noqa: BLE001
            LOG.debug("LAWA scaling failed (%s)", e)
    pax_score = float(saving_min) * float(weight)

    # Weather reliability.
    weather_pct: float | None = None
    if metar is not None and len(metar) > 0:
        try:
            df = metar.copy()
            df["hour"] = pd.to_datetime(df.get("time_utc", df.get("ts")), utc=True).dt.hour
            df = df[df["hour"].isin(peak_hours)]
            ceiling = df.get("ceiling_ft", pd.Series(dtype=float))
            vis = df.get("visibility_sm", pd.Series(dtype=float))
            if len(ceiling) == 0 or len(vis) == 0:
                weather_pct = None
            else:
                ok = (ceiling.fillna(0) > 1000) & (vis.fillna(0) > 3)
                weather_pct = float(ok.mean() * 100.0)
        except Exception as e:  # noqa: BLE001
            LOG.warning("weather KPI failed (%s)", e)

    # Peak service capacity (mirrors capacity_kpis.evtol_ops_per_hour but stated as accessibility).
    peak_cap = 3600.0 / (corridor.time_s + 60.0) if (corridor.feasible and corridor.time_s > 0) else 0.0

    if proxy_logged and corridor.feasible:
        LOG.debug("OSRM_FALLBACK road=50km/h detour=1.4")
    return AccessibilityKPI(
        access_time_saving_min_vs_road=float(saving_min),
        passenger_weighted_access_score=float(pax_score),
        vertiport_to_terminal_transfer_min=float(walk_min),
        weather_reliability_pct=weather_pct,
        peak_service_capacity_ops_per_hour=float(peak_cap),
    )
