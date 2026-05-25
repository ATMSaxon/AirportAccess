"""Tests for the M4 risk-field stack.

Each test is self-contained: no real data on disk required. The KSYN config
that ships in the repo is used as the airport.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml import counterfactual as cf
from src.ml._geom import AirportGeom, NM_M, FT_M
from src.ml.features import extract_features, merge_features_labels, NUMERIC_FEATURES
from src.ml.conformal import SplitConformal, empirical_coverage, calibrate_split
from src.ml.risk_field import train, _holdout_temporal_split
from src.utils import config as cfg_io


# ---------------------------------------------------------------------------
# Helpers: KSYN-like synthetic airport using the shipped sanity.yaml
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ksyn_geom(tmp_path_factory):
    """Drop the shipped configs/sanity.yaml into configs/airports/KSYN.yaml
    so the loader picks it up, then build an AirportGeom.
    """
    sanity = ROOT / "configs" / "sanity.yaml"
    airports_dir = ROOT / "configs" / "airports"
    ksyn = airports_dir / "KSYN.yaml"
    if not ksyn.exists():
        ksyn.write_bytes(sanity.read_bytes())
    return AirportGeom.from_icao("KSYN")


def _runway_config_df(ksyn_geom):
    """Single-slice runway-config table for KSYN."""
    return pd.DataFrame({
        "time_utc": pd.to_datetime(["2024-08-02 17:00:00"], utc=True),
        "config_id": ["EAST"],
        "active_arrivals": ["09"],
        "active_departures": ["09"],
    })


# ---------------------------------------------------------------------------
# Geometry sanity
# ---------------------------------------------------------------------------
def test_runway_projection(ksyn_geom):
    """Projection at threshold returns (0, 0); at end returns (length, 0)."""
    r = ksyn_geom.runway_by_id("09")
    along, cross = r.project(r.thr_x, r.thr_y)
    assert abs(along) < 1.0 and abs(cross) < 1.0
    along, cross = r.project(r.end_x, r.end_y)
    # WGS84-derived end vs declared length can disagree by ≤1 % on synthetic configs.
    assert abs(along - r.length_m) < max(20.0, 0.01 * r.length_m) and abs(cross) < 1.0


def test_segment_axis_crossing(ksyn_geom):
    r = ksyn_geom.runway_by_id("09")
    # A segment going across the runway centreline at low altitude → crossing.
    p0 = np.array([0.5 * r.length_m, -1000.0, 50.0])
    p1 = np.array([0.5 * r.length_m, +1000.0, 50.0])
    assert ksyn_geom.segment_crosses_runway_axis(p0, p1, ["09"], 2000.0 * FT_M)
    # Same crossing at high altitude (3000 ft = 914 m) → no conflict.
    p0[2] = p1[2] = 3000.0 * FT_M
    assert not ksyn_geom.segment_crosses_runway_axis(p0, p1, ["09"], 2000.0 * FT_M)


# ---------------------------------------------------------------------------
# Counterfactual injection on a planted-conflict scenario
# ---------------------------------------------------------------------------
def test_counterfactual_planted_conflict(ksyn_geom):
    """Inject a known ADS-B aircraft right under a candidate segment; the
    counterfactual labeller must mark at least one segment as conflict=1."""
    rng = np.random.default_rng(0)
    r = ksyn_geom.runway_by_id("09")
    # ADS-B: a single aircraft hovering on the runway axis at 500 m AGL.
    t0 = pd.Timestamp("2024-08-02 17:05:00", tz="UTC")
    adsb = pd.DataFrame({
        "time_utc": pd.date_range(t0, periods=120, freq="1s"),
        "x_m": np.full(120, 0.5 * r.length_m),
        "y_m": np.zeros(120),
        "z_msl_m": np.full(120, 500.0),
    })
    rc = _runway_config_df(ksyn_geom)

    df = cf.sample_and_label(
        icao="KSYN", n=200, seed=42,
        adsb_df=adsb, runway_config_df=rc,
    )
    # At least one segment must trigger SOME conflict cause (axis-cross is the
    # most likely from vertiport-to-anywhere sampling around the synthetic
    # airport).
    assert df["conflict"].sum() > 0
    # Mandatory schema columns must all be present.
    for c in cf.SEGMENT_COLUMNS:
        assert c in df.columns


def test_counterfactual_explicit_axis_cross(ksyn_geom):
    """Hand-craft one segment that *must* cross the runway axis at low alt
    and verify the label."""
    r = ksyn_geom.runway_by_id("09")
    seg = {
        "x0_m": 0.5 * r.length_m, "y0_m": -2000.0, "z0_m": 200.0,
        "x1_m": 0.5 * r.length_m, "y1_m": 2000.0,  "z1_m": 200.0,
        "active_arrivals": "09", "active_departures": "09",
        "t_start_utc": pd.Timestamp("2024-08-02 17:00:00", tz="UTC"),
        "duration_s": 60.0,
    }
    labels = cf.label_segment(seg, ksyn_geom, cf.EvtolKinematics.from_cfg(
        cfg_io.load_scenario("cost_weights")), adsb_window_df=None)
    assert labels["conflict"] == 1
    assert labels["axis_cross"] is True


# ---------------------------------------------------------------------------
# Features pipeline
# ---------------------------------------------------------------------------
def test_feature_extraction_runs(ksyn_geom):
    rc = _runway_config_df(ksyn_geom)
    df = cf.sample_and_label(
        icao="KSYN", n=64, seed=1,
        adsb_df=None, runway_config_df=rc,
    )
    feats = extract_features(df, ksyn_geom, adsb_df=None, metar_df=None)
    for c in NUMERIC_FEATURES:
        assert c in feats.columns
    assert len(feats) == len(df)
    joined = merge_features_labels(feats, df)
    assert "conflict" in joined.columns
    assert (joined["d_OLS_m"] >= 0).all()


# ---------------------------------------------------------------------------
# Conformal — synthetic
# ---------------------------------------------------------------------------
def test_conformal_coverage_synthetic():
    rng = np.random.default_rng(0)
    n = 4000
    p_true = rng.uniform(0, 1, n)
    # Noisy predictions: add Gaussian noise then clip.
    p_pred = np.clip(p_true + rng.normal(0, 0.1, n), 0, 1)
    y = (rng.uniform(0, 1, n) < p_true).astype(int)
    cp, info = calibrate_split(p_pred, y, cal_fraction=0.5, alpha=0.1, seed=0)
    assert 0.85 <= info["empirical_coverage"] <= 0.95
    assert info["mean_interval_width"] > 0
    assert info["mean_interval_width"] < 1.0


# ---------------------------------------------------------------------------
# Risk-field training — synthetic-separable features → AUROC > 0.95
# ---------------------------------------------------------------------------
def test_lr_high_auroc_on_separable():
    rng = np.random.default_rng(0)
    n = 4000
    # 12 numeric features + 2 cfg one-hots; only one feature carries signal.
    X = rng.normal(0, 1, (n, len(NUMERIC_FEATURES)))
    y = (X[:, 0] > 0.5).astype(int)
    # Build a fake features+labels DataFrame matching the columns the trainer
    # expects.
    df = pd.DataFrame(X, columns=NUMERIC_FEATURES)
    df["cfg_A"] = (rng.uniform(0, 1, n) > 0.5).astype(np.float32)
    df["cfg_B"] = 1.0 - df["cfg_A"]
    df["conflict"] = y
    # Two distinct calendar days so temporal holdout picks the later one.
    days = np.where(rng.uniform(0, 1, n) > 0.5,
                    "2024-08-02", "2024-08-09")
    df["mid_t_utc"] = pd.to_datetime(days, utc=True)

    res = train(model_name="lr", features_labels=df)
    assert res.metrics["auroc"] > 0.95
    # Conformal coverage should land near 0.9 on this synthetic task.
    assert "conformal_coverage" in res.metrics
    assert 0.80 <= res.metrics["conformal_coverage"] <= 1.0


# ---------------------------------------------------------------------------
# Multi-day temporal-holdout split (the bug fix)
# ---------------------------------------------------------------------------
def test_sample_date_stamped_on_segments(ksyn_geom):
    """`sample_and_label(sample_date=...)` must stamp every row with the
    supplied date string and use it in the seg_id prefix."""
    rc = _runway_config_df(ksyn_geom)
    df = cf.sample_and_label(
        icao="KSYN", n=32, seed=7,
        adsb_df=None, runway_config_df=rc,
        sample_date="2024-08-09",
    )
    assert "sample_date" in df.columns
    assert (df["sample_date"] == "2024-08-09").all()
    assert df["seg_id"].str.startswith("KSYN-2024-08-09-").all()


def test_temporal_split_uses_sample_date():
    """`_holdout_temporal_split` must split by ``sample_date`` when present,
    independent of UTC date of ``mid_t_utc``. This is the bug fix: a single
    LAX Friday's data straddles UTC midnight (UTC-7) and the old splitter
    put 294 rows in train and 49706 in test."""
    rng = np.random.default_rng(0)
    n = 1000
    # Construct a df where mid_t_utc dates correlate with sample_date only
    # for half the rows, to prove the splitter uses sample_date, not mid_t_utc.
    df = pd.DataFrame({
        "feat": rng.normal(0, 1, n),
        "conflict": rng.integers(0, 2, n),
    })
    df["sample_date"] = np.where(np.arange(n) < 800, "2024-08-02", "2024-08-09")
    # Intentionally MIS-aligned mid_t_utc: all rows on 2024-08-15 UTC.
    df["mid_t_utc"] = pd.Timestamp("2024-08-15 12:00:00", tz="UTC")
    train_df, test_df = _holdout_temporal_split(df)
    assert (train_df["sample_date"] == "2024-08-02").all()
    assert (test_df["sample_date"] == "2024-08-09").all()
    assert len(train_df) == 800
    assert len(test_df) == 200


def test_temporal_split_falls_back_to_utc_when_no_sample_date():
    """Backwards-compat: if `sample_date` is absent, fall back to UTC date."""
    df = pd.DataFrame({
        "feat": [0.0] * 6,
        "conflict": [0, 1, 0, 1, 0, 1],
        "mid_t_utc": pd.to_datetime([
            "2024-08-02 12:00:00", "2024-08-02 13:00:00", "2024-08-02 14:00:00",
            "2024-08-09 12:00:00", "2024-08-09 13:00:00", "2024-08-09 14:00:00",
        ], utc=True),
    })
    train_df, test_df = _holdout_temporal_split(df)
    assert len(train_df) == 3
    assert len(test_df) == 3
