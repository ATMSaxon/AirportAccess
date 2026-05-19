"""Shared pytest fixtures."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import config  # noqa: E402


@pytest.fixture(scope="session")
def ksyn_cfg() -> dict:
    """Synthetic single-runway airport used by sanity tests."""
    return config.load_yaml(ROOT / "configs/sanity.yaml")


@pytest.fixture(scope="session")
def klax_cfg() -> dict:
    return config.load_airport("KLAX")


@pytest.fixture(scope="session")
def ksfo_cfg() -> dict:
    return config.load_airport("KSFO")


@pytest.fixture(scope="session")
def annex14_cfg() -> dict:
    return config.load_annex14("code4_precision")


@pytest.fixture
def tmp_out(tmp_path) -> Path:
    """Per-test output directory."""
    p = tmp_path / "out"
    p.mkdir()
    return p
