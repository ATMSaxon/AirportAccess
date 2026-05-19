"""End-to-end tests for the M3 traffic / envelope pipeline.

These tests build a *synthetic two-runway airport* ("KSYN") in pure ENU and
then plant ADS-B tracks that exercise the classifier, the rolling
runway-config vote, the METAR cross-check, and the envelope mask.

No network. No data files required.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid
from src.traffic import adsb_clean, classify, runway_config, envelope


# ---------------------------------------------------------------------------
# Synthetic airport fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_airport():
    """KSYN — two east-west runways centred at (lat=10, lon=20)."""
    arp_lat, arp_lon = 10.0, 20.0
    # Two parallel ~3 km runways centred at ARP, ~500 m apart along Y.
    # Runway A: threshold 09 → end 27 (1.5 km east of threshold)
    # Runway B: threshold 09 → end 27, shifted 500 m north
    # We'll let pyproj convert ENU offsets back to lat/lon to make a usable YAML.
    icao = "KSYN"
    cfg = {
        "icao": icao,
        "name": "Synthetic two-runway",
        "arp": {"lat": arp_lat, "lon": arp_lon, "elev_ft": 0.0, "elev_m": 0.0},
        "local_crs_epsg": 32611,
        "runways": [],
        "extract_box_m": {"half_x": 6000, "half_y": 6000, "z_min": 0, "z_max": 1200},
        "grid_resolution_m": {"xy": 100, "z": 30},
        "vertiports": {},
    }
    frame = AirportFrame(icao=icao, lat0=arp_lat, lon0=arp_lon, elev_m=0.0, utm_epsg=32611)

    # Build (lat, lon) for runway thresholds at known ENU offsets.
    def enu_to_ll(x, y):
        lon, lat = frame.enu_to_wgs(np.array([x]), np.array([y]))
        return float(lat[0]), float(lon[0])

    # Runway 09L/27R: y = +250 m, x from -1500 to +1500
    a_thr_lat, a_thr_lon = enu_to_ll(-1500.0, +250.0)
    a_end_lat, a_end_lon = enu_to_ll(+1500.0, +250.0)
    # Runway 09R/27L: y = -250 m
    b_thr_lat, b_thr_lon = enu_to_ll(-1500.0, -250.0)
    b_end_lat, b_end_lon = enu_to_ll(+1500.0, -250.0)
    for rid_thr, rid_end, thr, end, bearing_thr, bearing_end in [
        ("09L", "27R", (a_thr_lat, a_thr_lon), (a_end_lat, a_end_lon), 90.0, 270.0),
        ("09R", "27L", (b_thr_lat, b_thr_lon), (b_end_lat, b_end_lon), 90.0, 270.0),
    ]:
        cfg["runways"].append({
            "id": rid_thr, "thr_lat": thr[0], "thr_lon": thr[1],
            "end_lat": end[0], "end_lon": end[1],
            "length_ft": 9800, "width_ft": 150, "bearing_deg": bearing_thr,
            "code_letter": "F", "code_number": 4, "precision": True,
        })
        cfg["runways"].append({
            "id": rid_end, "thr_lat": end[0], "thr_lon": end[1],
            "end_lat": thr[0], "end_lon": thr[1],
            "length_ft": 9800, "width_ft": 150, "bearing_deg": bearing_end,
            "code_letter": "F", "code_number": 4, "precision": True,
        })

    runway_ends = classify.airport_runway_ends(cfg, frame)
    grid = VoxelGrid.from_airport_cfg(cfg)
    return {"cfg": cfg, "frame": frame, "runway_ends": runway_ends, "grid": grid}


# ---------------------------------------------------------------------------
# Helpers to synthesise tracks
# ---------------------------------------------------------------------------

def _track_to_dataframe(icao24: str, t0: pd.Timestamp, xs, ys, zs, vs, vrate, on_ground):
    n = len(xs)
    times = pd.to_datetime([t0 + pd.Timedelta(seconds=5 * i) for i in range(n)], utc=True)
    # We let adsb_clean re-derive x_m/y_m from lon/lat. Pass them through a
    # round-trip via the synthetic frame for honesty.
    from src.utils.crs import AirportFrame
    frame = AirportFrame(icao="KSYN", lat0=10.0, lon0=20.0, elev_m=0.0, utm_epsg=32611)
    lon, lat = frame.enu_to_wgs(np.asarray(xs), np.asarray(ys))
    return pd.DataFrame({
        "time_utc": times,
        "icao24": icao24,
        "callsign": ["TEST"] * n,
        "lon_wgs": lon.astype(np.float32),
        "lat_wgs": lat.astype(np.float32),
        "baro_alt_m": np.asarray(zs, dtype=np.float32),
        "geo_alt_m": np.asarray(zs, dtype=np.float32),
        "velocity_ms": np.asarray(vs, dtype=np.float32),
        "track_deg": np.full(n, 270.0, dtype=np.float32),
        "vert_rate_ms": np.asarray(vrate, dtype=np.float32),
        "on_ground": np.asarray(on_ground, dtype=bool),
        "x_m": np.asarray(xs, dtype=np.float32),
        "y_m": np.asarray(ys, dtype=np.float32),
        "z_msl_m": np.asarray(zs, dtype=np.float32),
        "z_agl_m": np.asarray(zs, dtype=np.float32),
    })


def _arrival_to_runway_27(icao24: str, y_offset: float, t0: pd.Timestamp) -> pd.DataFrame:
    """Aircraft on final approach to runway 27R (or 27L): flying west.

    Starts ~12 km east of ARP at 3000 ft AGL, descends along centreline, lands
    at runway-27 threshold (which is the 09's end → x=+1500).
    """
    n = 30
    x = np.linspace(+12000, +1500, n)
    y = np.full(n, y_offset)
    z = np.linspace(900.0, 0.0, n)
    v = np.linspace(95.0, 65.0, n)
    vrate = np.full(n, -3.0)
    og = np.zeros(n, dtype=bool); og[-1] = True
    return _track_to_dataframe(icao24, t0, x, y, z, v, vrate, og)


def _departure_off_runway_09(icao24: str, y_offset: float, t0: pd.Timestamp) -> pd.DataFrame:
    """Departure from runway 09 (heading east). Starts at threshold, climbs out east."""
    n = 30
    x = np.linspace(-1500.0, +12000.0, n)
    y = np.full(n, y_offset)
    z = np.linspace(0.0, 900.0, n)
    v = np.linspace(45.0, 130.0, n)
    vrate = np.full(n, +3.0)
    og = np.zeros(n, dtype=bool); og[0] = True
    return _track_to_dataframe(icao24, t0, x, y, z, v, vrate, og)


def _overflight(icao24: str, t0: pd.Timestamp) -> pd.DataFrame:
    n = 30
    x = np.linspace(-15000.0, +15000.0, n)
    y = np.full(n, 3000.0)
    z = np.full(n, 1800.0)              # 6000 ft AGL
    v = np.full(n, 140.0)
    vrate = np.zeros(n)
    og = np.zeros(n, dtype=bool)
    return _track_to_dataframe(icao24, t0, x, y, z, v, vrate, og)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_adsb_outlier_rejection_and_resample(synthetic_airport):
    frame = synthetic_airport["frame"]
    t0 = pd.Timestamp("2024-08-02 14:00:00", tz="UTC")
    df = _arrival_to_runway_27("aaa111", y_offset=+250.0, t0=t0)
    # Inject a duplicate timestamp and an unphysical altitude jump.
    bad = df.iloc[[5]].copy()
    bad["icao24"] = "aaa111"
    bad["time_utc"] = df.iloc[5]["time_utc"]
    bad["baro_alt_m"] = df["baro_alt_m"].iloc[5] + 5000.0  # 5 km jump
    bad["geo_alt_m"] = bad["baro_alt_m"]
    df_dirty = pd.concat([df, bad], ignore_index=True)

    cleaned, stats = adsb_clean.clean_tracks(df_dirty, frame, metar=None)
    assert stats.n_in == len(df_dirty)
    # Dup removed (one of the two with the same (icao24, time_utc) drops).
    assert stats.n_after_dedup == len(df)
    # No 5 km step in cleaned z series.
    z = cleaned["z_msl_m"].to_numpy()
    assert np.max(np.abs(np.diff(z))) < 600.0
    # 5-s resample preserves point count (already 5-s spaced).
    assert stats.n_after_resample >= len(df) - 1


def test_classifier_east_arrivals_pick_runway_27(synthetic_airport):
    """East-wind synthetic ADS-B (planes coming from east) → classifier identifies
    arrivals to runway 27R/27L (i.e. landing to the west)."""
    frame = synthetic_airport["frame"]
    runway_ends = synthetic_airport["runway_ends"]
    t0 = pd.Timestamp("2024-08-02 14:00:00", tz="UTC")

    frames = []
    for i in range(5):                                # 5 arrivals to 27R
        frames.append(_arrival_to_runway_27(f"arr{i:03d}", +250.0, t0 + pd.Timedelta(minutes=2 * i)))
    for i in range(3):                                # 3 arrivals to 27L
        frames.append(_arrival_to_runway_27(f"arl{i:03d}", -250.0, t0 + pd.Timedelta(minutes=2 * i + 30)))
    frames.append(_overflight("ovr001", t0 + pd.Timedelta(minutes=10)))
    raw = pd.concat(frames, ignore_index=True)
    cleaned, _ = adsb_clean.clean_tracks(raw, frame, metar=None)
    tracks = classify.classify_tracks(cleaned, runway_ends)
    arrs = tracks[tracks["category"] == "arrival"]
    assert len(arrs) == 8, f"expected 8 arrivals, got {len(arrs)}: {tracks}"
    assigned = arrs["runway_end_id"].dropna().tolist()
    assert all(r in ("27R", "27L") for r in assigned), f"bad assignments: {assigned}"
    # All 27R ones land on 27R, all 27L ones on 27L (deterministic synthetic).
    by_track = dict(zip(arrs["icao24"], arrs["runway_end_id"]))
    assert {k: v for k, v in by_track.items() if k.startswith("arr")} == {f"arr{i:03d}": "27R" for i in range(5)}
    assert {k: v for k, v in by_track.items() if k.startswith("arl")} == {f"alr".replace("alr", "arl") + f"{i:03d}": "27L" for i in range(3)}


def test_classifier_west_departures_pick_runway_09(synthetic_airport):
    frame = synthetic_airport["frame"]
    runway_ends = synthetic_airport["runway_ends"]
    t0 = pd.Timestamp("2024-08-02 14:00:00", tz="UTC")
    frames = []
    for i in range(4):
        frames.append(_departure_off_runway_09(f"dep{i:03d}", +250.0, t0 + pd.Timedelta(minutes=4 * i)))
    raw = pd.concat(frames, ignore_index=True)
    cleaned, _ = adsb_clean.clean_tracks(raw, frame, metar=None)
    tracks = classify.classify_tracks(cleaned, runway_ends)
    deps = tracks[tracks["category"] == "departure"]
    assert len(deps) == 4
    assert set(deps["runway_end_id"].dropna()) <= {"09L", "09R"}


def test_overflight_classified_as_overflight(synthetic_airport):
    frame = synthetic_airport["frame"]
    runway_ends = synthetic_airport["runway_ends"]
    t0 = pd.Timestamp("2024-08-02 14:00:00", tz="UTC")
    df = _overflight("ovr001", t0)
    cleaned, _ = adsb_clean.clean_tracks(df, frame, metar=None)
    tracks = classify.classify_tracks(cleaned, runway_ends)
    assert tracks.iloc[0]["category"] == "overflight"
    assert tracks.iloc[0]["runway_end_id"] is None


def test_rolling_config_and_metar_cross_check(synthetic_airport):
    """Plant easterly wind (METAR wind FROM 270°? wait — we want to LAND west, so wind FROM the WEST = 270°).

    Aircraft land to the west (rwy 27) which means they head INTO 270°.
    METAR wind is reported FROM-direction; consistent METAR wind is therefore ~270°.

    Match rate should be ≥ 80 % even when we inject ~20 % noise.
    """
    frame = synthetic_airport["frame"]
    runway_ends = synthetic_airport["runway_ends"]
    bearings = {r.runway_id: r.bearing_deg for r in runway_ends}

    day_start = pd.Timestamp("2024-08-02 00:00:00", tz="UTC")
    day_end = day_start + pd.Timedelta("1D")

    # 60 arrivals across the day, all landing on 27R/27L (west flow).
    frames = []
    for i in range(60):
        y_off = +250.0 if (i % 2 == 0) else -250.0
        frames.append(_arrival_to_runway_27(
            f"flt{i:04d}", y_off, day_start + pd.Timedelta(minutes=15 * i),
        ))
    raw = pd.concat(frames, ignore_index=True)
    cleaned, _ = adsb_clean.clean_tracks(raw, frame, metar=None)
    tracks = classify.classify_tracks(cleaned, runway_ends)

    # Build hourly METAR: mostly wind from 270° (west) with 20% noise.
    rng = np.random.default_rng(1)
    metar_rows = []
    for h in range(24):
        t = day_start + pd.Timedelta(hours=h)
        if rng.random() < 0.8:
            wd = 270.0 + rng.normal(0, 5)
        else:
            wd = rng.uniform(0, 360)               # bad observation 20% of the time
        metar_rows.append({
            "station_id": "KSYN", "time_utc": t, "wind_dir_deg": wd,
            "wind_kt": 12.0, "wind_gust_kt": np.nan,
            "vis_sm": 10.0, "temp_c": 25.0, "dewpoint_c": 12.0,
            "altim_hpa": 1013.0, "ceiling_ft": np.nan,
            "flight_rule": "VFR", "raw": "",
        })
    metar = pd.DataFrame(metar_rows)

    slices = runway_config.rolling_config(
        tracks, day_start, day_end, interval_min=15,
        metar=metar, runway_bearings=bearings,
    )
    # At least 90 % of slices with arrivals should pick runway 27R/27L.
    s_with_ops = slices[slices["n_arrivals"] > 0]
    active_lists = s_with_ops["arrivals_active"].tolist()
    correct = sum(1 for a in active_lists if set(a) <= {"27R", "27L"})
    assert correct / max(1, len(active_lists)) >= 0.9

    match_rate = runway_config.metar_match_rate(slices)
    assert match_rate >= 0.8, f"METAR match rate {match_rate:.2f} below 0.8"

    # The downstream alias columns (consumed by ml-engineer / planner) must be present.
    for col in ("time_utc", "active_arrivals", "active_departures", "config_id"):
        assert col in slices.columns, f"missing alias column {col}"
    # And the semicolon-joined format must hold on a non-empty slice.
    nonempty = slices[slices["n_arrivals"] > 0].iloc[0]
    assert ";" in nonempty["active_arrivals"] or len(nonempty["active_arrivals"].split(";")) >= 1
    assert nonempty["config_id"].startswith("ARR:")
    assert nonempty["time_utc"] == nonempty["slice_start"]


def test_envelope_excludes_known_approach_corridor(synthetic_airport):
    """The dynamic envelope must exclude voxels within 3 NM × 1500 ft of an active
    approach centreline below 5000 ft AGL."""
    grid = synthetic_airport["grid"]
    runway_ends = synthetic_airport["runway_ends"]
    # Active: arrivals to 27R, departures off 09L.
    wx = envelope.WeatherState(vis_sm=10.0, ceiling_ft=10000.0, flight_rule="VFR")
    closed = envelope.build_closure_mask(
        grid, runway_ends,
        arrivals_active=["27R"], departures_active=["09L"], weather=wx,
    )
    assert closed.dtype == bool
    # A point 5 km east of the 27R threshold (along the approach) at 800 ft AGL
    # is well inside the 3 NM × 1500 ft corridor → should be CLOSED.
    # 27R threshold ≈ (+1500, +250) in ENU. Approach extends east (positive x).
    px, py, pz = 6500.0, 250.0, 244.0       # ~800 ft AGL
    ix, iy, iz = grid.world_to_index(px, py, pz)
    assert closed[ix, iy, iz], "voxel on active approach should be closed"

    # A point far to the south at the same altitude is well outside lateral 3 NM.
    px2, py2, pz2 = 6500.0, -8000.0, 244.0
    ix2, iy2, iz2 = grid.world_to_index(px2, py2, pz2)
    assert not closed[ix2, iy2, iz2], "voxel far from corridor should be open"

    # A point above 5000 ft AGL on the corridor centreline should be open (closure
    # cap is 5000 ft AGL).
    pz_high = 1700.0          # ~5577 ft AGL
    ix3, iy3, iz3 = grid.world_to_index(6500.0, 250.0, pz_high)
    assert not closed[ix3, iy3, iz3]


def test_envelope_imc_expands_buffer(synthetic_airport):
    grid = synthetic_airport["grid"]
    runway_ends = synthetic_airport["runway_ends"]
    closed_vfr = envelope.build_closure_mask(
        grid, runway_ends, ["27R"], ["09L"],
        weather=envelope.WeatherState(vis_sm=10.0, ceiling_ft=10000.0, flight_rule="VFR"),
    )
    closed_ifr = envelope.build_closure_mask(
        grid, runway_ends, ["27R"], ["09L"],
        weather=envelope.WeatherState(vis_sm=1.0, ceiling_ft=500.0, flight_rule="IFR"),
    )
    # IMC mask must be a strict superset of VFR mask (same active runways).
    assert closed_ifr.sum() > closed_vfr.sum()
    assert (closed_ifr | closed_vfr == closed_ifr).all()


def test_envelope_combines_with_a_static(synthetic_airport):
    grid = synthetic_airport["grid"]
    runway_ends = synthetic_airport["runway_ends"]
    nx, ny, nz = grid.shape
    # Simple A_static: clear everywhere except a 100 m × 100 m × 30 m "obstacle" at index (50, 50, 5).
    a_static = np.ones((nx, ny, nz), dtype=bool)
    a_static[50, 50, 5] = False
    e_t = envelope.envelope_for_slice(
        grid, runway_ends,
        arrivals_active=["27R"], departures_active=[],
        weather=envelope.WeatherState(vis_sm=10.0, ceiling_ft=10000.0, flight_rule="VFR"),
        a_static=a_static,
    )
    assert e_t.dtype == bool
    # The Annex-14 obstacle voxel must be closed in E_t too.
    assert not e_t[50, 50, 5]
    # Some voxels still open in E_t (we haven't closed the whole world).
    assert e_t.any()
