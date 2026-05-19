"""Unit tests for the shared utility layer."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import crs, config, grid, io  # noqa: E402


def _lax_cfg():
    return config.load_airport("KLAX")


def test_airport_frame_arp_is_origin():
    af = crs.AirportFrame.from_cfg(_lax_cfg())
    x, y = af.wgs_to_enu(np.array([af.lon0]), np.array([af.lat0]))
    assert abs(x[0]) < 0.001 and abs(y[0]) < 0.001


def test_airport_frame_roundtrip():
    af = crs.AirportFrame.from_cfg(_lax_cfg())
    lons = np.array([-118.4081, -118.4350, -118.2370])
    lats = np.array([33.9425, 33.9275, 34.0560])
    x, y = af.wgs_to_enu(lons, lats)
    lon2, lat2 = af.enu_to_wgs(x, y)
    assert np.allclose(lon2, lons, atol=1e-6)
    assert np.allclose(lat2, lats, atol=1e-6)


def test_voxel_grid_indices():
    g = grid.VoxelGrid.from_airport_cfg(_lax_cfg())
    # origin point maps to centre index
    ix, iy, iz = g.world_to_index(0.0, 0.0, 0.0)
    assert ix == g.shape[0] // 2
    assert iy == g.shape[1] // 2
    assert iz == 0


def test_voxel_grid_clipping():
    g = grid.VoxelGrid.from_airport_cfg(_lax_cfg())
    ix, iy, iz = g.world_to_index(1e9, 1e9, 1e9)
    assert ix == g.shape[0] - 1
    assert iy == g.shape[1] - 1
    assert iz == g.shape[2] - 1


def test_to_local_msl_m():
    assert abs(crs.to_local_msl_m(125.0) - 38.1) < 0.05


def test_summary_writer(tmp_path):
    out = io.write_summary(tmp_path, {"foo": 1, "bar": [1, 2]})
    assert out.exists()
    body = io.read_json(out)
    assert body["foo"] == 1
    assert "timestamp_utc" in body


def test_manifest_writer(tmp_path):
    data_file = tmp_path / "x.parquet"
    data_file.write_bytes(b"hello")
    manifest = io.write_manifest(data_file, source="unit-test", source_url="https://example.com",
                                  params={"k": 1})
    assert manifest.exists()
    body = io.read_json(manifest)
    assert body["source"] == "unit-test"
    assert body["sha256"] is not None
    assert body["size_bytes"] == 5
