"""Load YAML config files (airport, annex14, scenarios)."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

from . import paths


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def load_airport(icao: str) -> dict[str, Any]:
    """Load `configs/airports/<ICAO>.yaml`."""
    p = paths.AIRPORTS_CFG / f"{icao}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"No config for airport {icao}: {p}")
    return load_yaml(p)


def load_annex14(profile: str = "code4_precision") -> dict[str, Any]:
    p = paths.ANNEX14_CFG / f"{profile}.yaml"
    return load_yaml(p)


def load_scenario(name: str) -> dict[str, Any]:
    p = paths.SCENARIOS_CFG / f"{name}.yaml"
    return load_yaml(p)
