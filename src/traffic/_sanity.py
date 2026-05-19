"""Internal sanity harness for the traffic lane. Called by `src/traffic/__init__.py`.

No internet, no real data. Synthesises a tiny 1-hour ADS-B dataset on the
``KSYN`` sanity airport, runs the full M3 pipeline (clean → classify → rolling
config → envelope), and writes a parquet + Zarr/NPZ under ``out_dir``.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.crs import AirportFrame
from ..utils.grid import VoxelGrid
from ..utils.logs import get_logger
from . import adsb_clean, classify, runway_config, density, envelope

log = get_logger(__name__)


def run(out_dir: Path, airport_cfg: dict) -> dict:
    icao = airport_cfg.get("icao", "KSYN")
    cfg = dict(airport_cfg)
    cfg.setdefault("arp", {})
    cfg["arp"].setdefault("elev_m",
                          float(cfg["arp"].get("elev_ft", 0.0)) * 0.3048)
    cfg.setdefault("local_crs_epsg", 32631)
    frame = AirportFrame.from_cfg(cfg)
    grid = VoxelGrid.from_airport_cfg(cfg)
    runway_ends = classify.airport_runway_ends(cfg, frame)
    bearings = {r.runway_id: r.bearing_deg for r in runway_ends}

    # ---- synthetic ADS-B (3 arrivals to RWY 27 + 1 departure off RWY 09 + 1 overflight) ----
    t0 = pd.Timestamp("2026-05-19 12:00:00", tz="UTC")
    frames: list[pd.DataFrame] = []
    for i in range(3):
        frames.append(_make_arrival(f"san_arr{i}", frame, t0 + pd.Timedelta(minutes=10 * i)))
    frames.append(_make_departure("san_dep0", frame, t0 + pd.Timedelta(minutes=35)))
    frames.append(_make_overflight("san_ovr0", frame, t0 + pd.Timedelta(minutes=45)))
    # An outlier track that should be heavily filtered (5 km altitude jump).
    bad = _make_arrival("san_bad0", frame, t0 + pd.Timedelta(minutes=50))
    bad.loc[5, "baro_alt_m"] += 5000.0
    bad.loc[5, "geo_alt_m"] = bad.loc[5, "baro_alt_m"]
    frames.append(bad)
    raw = pd.concat(frames, ignore_index=True)

    # ---- METAR: light west wind (270°), VFR.
    metar = pd.DataFrame([{
        "station_id": icao,
        "time_utc": t0,
        "wind_dir_deg": 270.0, "wind_kt": 10.0, "wind_gust_kt": np.nan,
        "vis_sm": 10.0, "temp_c": 22.0, "dewpoint_c": 12.0,
        "altim_hpa": 1013.0, "ceiling_ft": np.nan,
        "flight_rule": "VFR", "raw": "",
    }])

    cleaned, clean_stats = adsb_clean.clean_tracks(raw, frame, metar=metar)
    tracks = classify.classify_tracks(cleaned, runway_ends)

    day_start = t0.floor("D")
    day_end = day_start + pd.Timedelta("1D")
    rconf = runway_config.rolling_config(
        tracks, day_start, day_end,
        interval_min=15, metar=metar, runway_bearings=bearings,
    )

    a_static = envelope.load_static_mask(icao, grid)
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
            weather=wx, a_static=a_static,
        )
        envelopes[row["slice_start"].isoformat()] = e_t
        slice_times.append(row["slice_start"])

    # Persist a *small* slice of artefacts for evidence.
    rconf_path = out_dir / "runway_config_sanity.parquet"
    rconf_to_write = rconf.copy()
    for col in ("arrivals_active", "departures_active"):
        if col in rconf_to_write.columns:
            rconf_to_write[col] = rconf_to_write[col].apply(
                lambda v: ",".join(v) if isinstance(v, (list, tuple)) else v)
    # Downstream alias columns (`active_arrivals`, `active_departures`, `time_utc`,
    # `config_id`) are already populated by `rolling_config` — see SCHEMAS.md.
    rconf_to_write.to_parquet(rconf_path, index=False)

    env_path = out_dir / "envelope_sanity.zarr"
    written = envelope.save_envelope_zarr(envelopes, env_path, grid, slice_times)

    # Densities — cheap on the sanity grid (60×60×40 by default).
    df = density.compute_density(cleaned, tracks, grid)

    arrivals = tracks[tracks["category"] == "arrival"]
    departures = tracks[tracks["category"] == "departure"]
    overflights = tracks[tracks["category"] == "overflight"]

    kept = np.array([float(e.mean()) for e in envelopes.values()])
    metrics = {
        "n_tracks_raw": int(raw["icao24"].nunique()),
        "n_tracks_after_clean": int(clean_stats.n_tracks),
        "n_arrivals": int(len(arrivals)),
        "n_departures": int(len(departures)),
        "n_overflights": int(len(overflights)),
        "arrival_assignments": arrivals["runway_end_id"].dropna().tolist(),
        "departure_assignments": departures["runway_end_id"].dropna().tolist(),
        "n_slices": int(len(rconf)),
        "slices_with_ops": int(((rconf["n_arrivals"] + rconf["n_departures"]) > 0).sum()),
        "envelope_kept_fraction_mean": float(kept.mean()) if kept.size else 1.0,
        "envelope_kept_fraction_min": float(kept.min()) if kept.size else 1.0,
        "density_arr_sum": float(df.arrivals.sum()),
        "density_dep_sum": float(df.departures.sum()),
    }

    return {
        "ok": True,
        "outputs": [str(rconf_path), str(written)],
        "metrics": metrics,
        "grid_shape": list(grid.shape),
        "n_runway_ends": len(runway_ends),
    }


# ---------------------------------------------------------------------------
# Track synthesis helpers (local to sanity — not part of the public traffic API)
# ---------------------------------------------------------------------------

def _to_df(icao24: str, frame: AirportFrame, t0: pd.Timestamp,
           xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
           vs: np.ndarray, vrate: np.ndarray, on_ground: np.ndarray) -> pd.DataFrame:
    n = len(xs)
    times = pd.to_datetime([t0 + pd.Timedelta(seconds=5 * i) for i in range(n)], utc=True)
    lon, lat = frame.enu_to_wgs(xs, ys)
    return pd.DataFrame({
        "time_utc": times,
        "icao24": icao24,
        "callsign": ["SAN"] * n,
        "lon_wgs": lon.astype(np.float32),
        "lat_wgs": lat.astype(np.float32),
        "baro_alt_m": zs.astype(np.float32),
        "geo_alt_m": zs.astype(np.float32),
        "velocity_ms": vs.astype(np.float32),
        "track_deg": np.full(n, 270.0, dtype=np.float32),
        "vert_rate_ms": vrate.astype(np.float32),
        "on_ground": on_ground.astype(bool),
        "x_m": xs.astype(np.float32),
        "y_m": ys.astype(np.float32),
        "z_msl_m": zs.astype(np.float32),
        "z_agl_m": zs.astype(np.float32),
    })


def _make_arrival(icao24: str, frame: AirportFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Arrival to RWY 27 — flies west from x≈+10 km to threshold at x≈+1500 m."""
    n = 25
    x = np.linspace(+10000.0, +1500.0, n)
    y = np.full(n, 0.0)
    z = np.linspace(700.0, 0.0, n)
    v = np.linspace(95.0, 65.0, n)
    vrate = np.full(n, -3.0)
    og = np.zeros(n, dtype=bool); og[-1] = True
    return _to_df(icao24, frame, t0, x, y, z, v, vrate, og)


def _make_departure(icao24: str, frame: AirportFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Departure off RWY 09 — climbs east from threshold."""
    n = 25
    x = np.linspace(-1500.0, +10000.0, n)
    y = np.full(n, 0.0)
    z = np.linspace(0.0, 700.0, n)
    v = np.linspace(45.0, 130.0, n)
    vrate = np.full(n, +3.0)
    og = np.zeros(n, dtype=bool); og[0] = True
    return _to_df(icao24, frame, t0, x, y, z, v, vrate, og)


def _make_overflight(icao24: str, frame: AirportFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """Cruise overhead at 1800 m AGL (≈ 5900 ft)."""
    n = 25
    x = np.linspace(-10000.0, +10000.0, n)
    y = np.full(n, 2500.0)
    z = np.full(n, 1800.0)
    v = np.full(n, 140.0)
    vrate = np.zeros(n)
    og = np.zeros(n, dtype=bool)
    return _to_df(icao24, frame, t0, x, y, z, v, vrate, og)
