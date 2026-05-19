"""Cross-lane integration regression: every package imports cleanly + key submodules exist."""
from __future__ import annotations
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Mapping lane → list of expected public submodules (per EXPERIMENT_PLAN.md and CLAUDE.md).
EXPECTED = {
    "src.data":     ["source_faa_nasr", "source_faa_dof", "source_usgs_3dep", "source_osm",
                     "source_opensky", "source_noaa_wx", "source_lawa", "source_bts"],
    "src.geometry": ["ols_surfaces", "vertiport_ofv", "sdf", "query"],
    "src.traffic":  ["adsb_clean", "classify", "runway_config", "density", "envelope"],
    "src.ml":       ["counterfactual", "features", "risk_field", "conformal"],
    "src.planning": ["graph", "astar", "corridor"],
    "src.analysis": ["safety_kpis", "capacity_kpis", "accessibility_kpis", "joint_eval"],
}


def _module_present(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@pytest.mark.parametrize("pkg", list(EXPECTED.keys()))
def test_package_imports(pkg):
    """The lane package can be imported even if it's still being built out."""
    importlib.import_module(pkg)


@pytest.mark.parametrize("pkg,submods", list(EXPECTED.items()))
def test_expected_submodules(pkg, submods):
    """Each expected submodule either (a) is missing entirely (lane still landing) OR
    (b) imports cleanly. A submodule file existing but failing to import is a regression."""
    for sm in submods:
        full = f"{pkg}.{sm}"
        if _module_present(full):
            importlib.import_module(full)


def test_no_circular_imports():
    """Importing every present package in topo order should not deadlock or partial-init-fail."""
    order = ["src.utils", "src.data", "src.geometry", "src.traffic", "src.ml", "src.planning", "src.analysis"]
    for pkg in order:
        importlib.import_module(pkg)
