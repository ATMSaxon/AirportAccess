"""Offline-safe tests for src/data/source_*.py.

These tests use small fixture data + monkeypatched HTTP; no internet required.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.data import (
    source_bts, source_faa_dof, source_faa_nasr, source_lawa,
    source_noaa_wx, source_opensky, source_osm, source_usgs_3dep,
)
from src.data._common import bbox_around_arp, great_circle_nm
from src.utils.config import load_airport
from src.utils.crs import AirportFrame


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def klax_cfg():
    return load_airport("KLAX")


@pytest.fixture()
def out_dir(tmp_path):
    return tmp_path / "processed"


# --------------------------------------------------------------------------- #
# Common helpers
# --------------------------------------------------------------------------- #
def test_great_circle_nm_self_zero():
    assert great_circle_nm(0, 0, 0, 0) == pytest.approx(0.0, abs=1e-6)


def test_great_circle_nm_known_pair():
    # JFK→LAX great circle is ≈ 2147 NM
    d = great_circle_nm(-73.7781, 40.6413, -118.4081, 33.9425)
    assert 2050 < d < 2250


def test_bbox_around_arp(klax_cfg):
    frame = AirportFrame.from_cfg(klax_cfg)
    bbox = bbox_around_arp(frame, half_km=30.0)
    lon_min, lat_min, lon_max, lat_max = bbox
    assert lon_min < klax_cfg["arp"]["lon"] < lon_max
    assert lat_min < klax_cfg["arp"]["lat"] < lat_max
    # 60 km wide ≈ 0.54 deg lat
    assert 0.4 < (lat_max - lat_min) < 0.7


# --------------------------------------------------------------------------- #
# D1: FAA NASR (no network — reads shipped YAML)
# --------------------------------------------------------------------------- #
def test_faa_nasr_lax_writes_parquet(klax_cfg, out_dir):
    res = source_faa_nasr.fetch(klax_cfg, window="2024-08", out_dir=out_dir)
    assert res.status == "ok"
    p = out_dir / "runways.parquet"
    assert p.exists()
    df = pd.read_parquet(p)
    assert {"runway_id", "thr_x_m", "thr_y_m", "bearing_deg", "length_m"} <= set(df.columns)
    assert len(df) >= 4
    # Bearings in [0, 360); lengths sensible
    assert (df["bearing_deg"] >= 0).all() and (df["bearing_deg"] < 360).all()
    assert (df["length_m"] > 1500).all()
    # Manifest exists
    assert (out_dir / "runways.parquet_manifest.json").exists()


# --------------------------------------------------------------------------- #
# D2: FAA DOF — synthetic fixed-width file
# --------------------------------------------------------------------------- #
def _make_dof_fixture(tmp_path: Path) -> Path:
    """Two synthetic DOF records, fixed-width."""
    rows = []
    # OAS lat/lon for a point right next to LAX ARP
    rec = (
        "01-12345  O  US CA Los Angeles      "  # 0..33
        "33 56 33.00N "                          # 35..47
        "118 24 29.00W "                         # 49..61
        "TOWER             "                     # 62..80
        " 0001"                                   # 83..88 quantity
        "  150"                                   # 89..94 agl_ft
        "  275"                                   # 95..100 msl_ft
        " R"                                      # 102..103 lighting
        " 2"                                      # 105..106 horiz acc
        " B"                                      # 108..109 vert acc
        "M"                                       # 110..111 marked
    )
    rows.append(rec.ljust(120))
    p = tmp_path / "DAILY_DOF.DAT"
    p.write_text("\n".join(rows) + "\n", encoding="latin-1")
    return p


def test_faa_dof_parses_fixedwidth(klax_cfg, out_dir, tmp_path, monkeypatch):
    fixture = _make_dof_fixture(tmp_path)
    text = fixture.read_text(encoding="latin-1")
    df = source_faa_dof._parse_fixed_width(text)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["oas_number"] == "01-12345"
    assert row["lat_wgs"] == pytest.approx(33 + 56 / 60 + 33 / 3600, abs=1e-3)
    assert -118.5 < row["lon_wgs"] < -118.3
    assert int(row["agl_ft"]) == 150


def test_faa_dof_fetch_with_cached_zip(klax_cfg, out_dir, tmp_path, monkeypatch):
    # Place a synthetic ZIP in the cache so fetch() short-circuits to no-download
    from src.utils import paths as p_mod
    monkeypatch.setattr(p_mod, "CACHE", tmp_path / "cache")
    cache_dir = tmp_path / "cache" / "faa_dof"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fixture = _make_dof_fixture(tmp_path)
    zip_path = cache_dir / "DAILY_DOF.ZIP"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(fixture, arcname="DAILY_DOF.DAT")

    # Patch the module's paths import to point at the patched CACHE
    monkeypatch.setattr(source_faa_dof, "path_utils", p_mod)

    res = source_faa_dof.fetch(klax_cfg, window="2024-08", out_dir=out_dir)
    assert res.status == "ok"
    df = pd.read_parquet(out_dir / "obstacles.parquet")
    assert len(df) == 1
    assert "x_m" in df.columns and "y_m" in df.columns


# --------------------------------------------------------------------------- #
# D7: LAWA (no network — hard-coded)
# --------------------------------------------------------------------------- #
def test_lawa_lax_writes_rows(klax_cfg, out_dir):
    res = source_lawa.fetch(klax_cfg, window="2024-08", out_dir=out_dir)
    assert res.status == "ok"
    df = pd.read_parquet(out_dir / "peak_hour.parquet")
    assert len(df) > 30
    assert set(df["peak_hour"]) == {"08-09", "11-12", "17-18"}
    assert (df["trips"] > 0).all()


def test_lawa_sfo_empty_but_manifested(out_dir):
    sfo_cfg = load_airport("KSFO")
    res = source_lawa.fetch(sfo_cfg, window="2024-08", out_dir=out_dir)
    assert res.status == "ok"  # still ok — empty parquet + manifest with note
    df = pd.read_parquet(out_dir / "peak_hour.parquet")
    assert len(df) == 0
    # Manifest carries the SFO-specific note
    mfst = json.loads((out_dir / "peak_hour.parquet_manifest.json").read_text())
    assert "SFO" in mfst["extra"]["notes"]


# --------------------------------------------------------------------------- #
# D6: NOAA METAR — exercise the parser
# --------------------------------------------------------------------------- #
def test_metar_parser_handles_canonical():
    raw = ("KLAX 020853Z 24008KT 10SM FEW015 SCT200 19/14 A2992 RMK AO2 SLP132 "
           "T01890144")
    p = source_noaa_wx._parse_metar(raw)
    assert p["wind_dir_deg"] == 240.0
    assert p["wind_kt"] == 8.0
    assert p["vis_sm"] == 10.0
    assert 1010 < p["altim_hpa"] < 1015
    assert p["flight_rule"] == "VFR"


def test_metar_parser_low_ifr():
    raw = "KSFO 020853Z 18003KT 1SM BR OVC003 14/13 A2997"
    p = source_noaa_wx._parse_metar(raw)
    assert p["vis_sm"] == 1.0
    assert p["ceiling_ft"] == 300.0
    assert p["flight_rule"] in ("LIFR", "IFR")


# --------------------------------------------------------------------------- #
# D5: OpenSky — enrichment from a tiny synthetic state batch
# --------------------------------------------------------------------------- #
def test_opensky_enrich(klax_cfg):
    frame = AirportFrame.from_cfg(klax_cfg)
    df = pd.DataFrame({
        "icao24": ["abc123", "def456"],
        "callsign": ["UAL1", "AAL2"],
        "longitude": [-118.40, -118.39],
        "latitude": [33.94, 33.95],
        "baro_altitude": [1000.0, 1500.0],
        "geo_altitude": [1010.0, np.nan],
        "velocity": [80.0, 90.0],
        "true_track": [270.0, 90.0],
        "vertical_rate": [-2.0, 3.0],
        "on_ground": [False, False],
        "time_utc": pd.to_datetime(["2024-08-02T00:00:00Z", "2024-08-02T00:00:01Z"]),
    })
    out = source_opensky._enrich(df, frame, float(klax_cfg["arp"]["elev_m"]))
    assert {"x_m", "y_m", "z_msl_m", "z_agl_m"} <= set(out.columns)
    # Within 5 km of ARP for both points
    assert (np.hypot(out["x_m"], out["y_m"]) < 5000).all()
    # z_agl_m = z_msl_m - field elev
    assert out["z_agl_m"].iloc[0] == pytest.approx(1010 - klax_cfg["arp"]["elev_m"], abs=1.0)


def test_opensky_no_creds_marks_offline(klax_cfg, out_dir, monkeypatch):
    # Ensure no creds
    monkeypatch.delenv("OPENSKY_USERNAME", raising=False)
    monkeypatch.delenv("OPENSKY_PASSWORD", raising=False)
    # Skip historical days: REST snapshot only works for today
    days = ["2024-08-02"]
    with pytest.raises(RuntimeError):
        source_opensky.fetch(klax_cfg, window="2024-08", out_dir=out_dir, days=days)
    # Day-level OFFLINE manifest should still exist
    assert (out_dir / "opensky_2024-08-02.OFFLINE.json").exists()


# --------------------------------------------------------------------------- #
# D4: OSM — exercise the geometry coercion helpers (no network)
# --------------------------------------------------------------------------- #
def test_osm_coerce_height():
    assert source_osm._coerce_height({"height": "45.5"}) == pytest.approx(45.5)
    assert source_osm._coerce_height({"height": "30 m"}) == pytest.approx(30.0)
    assert source_osm._coerce_height({"building:levels": "5"}) == pytest.approx(15.0)
    assert np.isnan(source_osm._coerce_height({}))


# --------------------------------------------------------------------------- #
# D8: BTS DB1B — exercise the normalisation
# --------------------------------------------------------------------------- #
def test_bts_normalise_minimal():
    raw = pd.DataFrame({
        "ItinID": ["1", "2"],
        "Origin": ["LAX", "JFK"],
        "Dest": ["JFK", "LAX"],
        "Reporting_Airline": ["AA", "UA"],
        "Passengers": ["1", "2"],
        "MktFare": ["350.5", "410.0"],
        "MktDistance": ["2450", "2450"],
        "Quarter": ["3", "3"],
        "Year": ["2024", "2024"],
    })
    out = source_bts._normalise(raw)
    assert len(out) == 2
    assert set(out.columns) >= {"itin_id", "origin", "dest", "passengers",
                                "market_fare", "market_distance"}
    assert out["passengers"].sum() == 3
    assert out["market_fare"].iloc[0] == pytest.approx(350.5)


# --------------------------------------------------------------------------- #
# Orchestrator import sanity (no network — runs --only lawa)
# --------------------------------------------------------------------------- #
def test_acquire_all_lawa_only(tmp_path, monkeypatch):
    """End-to-end exercise of scripts/acquire_all.py via --only lawa (no network)."""
    import sys
    import importlib

    monkeypatch.setattr(sys, "argv", [
        "acquire_all.py", "--airport", "KLAX", "--window", "2024-08",
        "--only", "lawa,faa_nasr",
        "--output-dir", str(tmp_path / "out"),
    ])
    spec = importlib.util.spec_from_file_location(
        "acquire_all",
        Path(__file__).resolve().parents[1] / "scripts" / "acquire_all.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main()
    assert rc == 0
    inv = json.loads((tmp_path / "out" / "_inventory.json").read_text())
    assert inv["sources"]["lawa"]["status"] == "ok"
    assert inv["sources"]["faa_nasr"]["status"] == "ok"
