"""Smoke tests — the project skeleton imports and the sanity script runs (with at least the
team-lead's portion of the pipeline). Specialist lanes that haven't landed yet are *not* required.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_project_root_layout_exists():
    for d in ("src", "configs", "scripts", "tests", "refine-logs"):
        assert (ROOT / d).is_dir(), f"missing dir: {d}"


def test_utils_importable():
    sys.path.insert(0, str(ROOT))
    from src.utils import paths, crs, io, grid, logs, config  # noqa: F401


def test_airport_configs_parse():
    sys.path.insert(0, str(ROOT))
    from src.utils import config
    for icao in ("KLAX", "KSFO", "KSYN"):
        cfg = config.load_airport(icao) if icao != "KSYN" else config.load_yaml(ROOT / "configs/sanity.yaml")
        assert cfg["icao"] == icao
        assert "runways" in cfg
        assert len(cfg["runways"]) >= 2


def test_voxel_grid_sane():
    sys.path.insert(0, str(ROOT))
    from src.utils import config
    from src.utils.grid import VoxelGrid
    cfg = config.load_airport("KLAX")
    g = VoxelGrid.from_airport_cfg(cfg)
    nx, ny, nz = g.shape
    assert nx == ny == 600
    assert nz == int(round(3500 / 30))


def test_sanity_script_runs_without_lanes():
    """Sanity script should at least run and produce a summary file even with zero lanes."""
    out = subprocess.run(
        [sys.executable, "scripts/run_sanity.py", "--output-dir", "results/sanity_test"],
        capture_output=True, cwd=ROOT,
    )
    # Exit code may be 0 or 1 depending on whether any lanes are present; the file must exist.
    summary = ROOT / "results/sanity_test/summary.json"
    assert summary.exists(), f"sanity didn't write summary.json: stderr={out.stderr.decode()[-500:]}"
    body = json.loads(summary.read_text())
    assert "lanes" in body
    assert "overall_ok" in body
