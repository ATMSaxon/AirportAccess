#!/usr/bin/env python3
"""Team-lead dashboard: summarise what each lane has shipped.

Reports per lane:
  * modules (file LoC)
  * key API symbols present
  * docs (SCHEMAS.md / INTERFACES.md)
  * tests
  * sanity-check status
"""
from __future__ import annotations
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LANES = {
    "data":     ("source_faa_nasr", "source_faa_dof", "source_usgs_3dep", "source_osm",
                 "source_opensky", "source_noaa_wx", "source_lawa", "source_bts"),
    "geometry": ("ols_surfaces", "vertiport_ofv", "sdf", "query"),
    "traffic":  ("adsb_clean", "classify", "runway_config", "density", "envelope"),
    "ml":       ("counterfactual", "features", "risk_field", "conformal"),
    "planning": ("graph", "astar", "corridor"),
    "analysis": ("safety_kpis", "capacity_kpis", "accessibility_kpis", "joint_eval"),
}

EXPECTED_DOCS = {
    "data":     ["SCHEMAS.md"],
    "geometry": ["INTERFACES.md", "SCHEMAS.md"],
    "traffic":  ["SCHEMAS.md", "INTERFACES.md"],
    "ml":       ["SCHEMAS.md", "INTERFACES.md"],
    "planning": ["INTERFACES.md"],
    "analysis": ["INTERFACES.md"],
}


def loc(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())


def lane_status(lane: str, modules: tuple[str, ...]) -> dict:
    pkg_dir = ROOT / "src" / lane
    pkg_loc = sum(loc(p) for p in pkg_dir.glob("*.py"))
    module_status = {}
    total_module_loc = 0
    for m in modules:
        p = pkg_dir / f"{m}.py"
        L = loc(p)
        module_status[m] = {"present": p.exists(), "loc": L}
        total_module_loc += L

    docs = []
    for doc_name in EXPECTED_DOCS.get(lane, []):
        if (pkg_dir / doc_name).exists():
            docs.append(doc_name)

    tests = sorted([str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob(f"test_{lane}*.py")])

    sanity_ok = False
    sanity_err = None
    try:
        mod = importlib.import_module(f"src.{lane}")
        fn = getattr(mod, "sanity_check", None)
        sanity_ok = callable(fn)
    except Exception as e:
        sanity_err = str(e)

    return {
        "lane": lane,
        "pkg_loc": pkg_loc,
        "modules": module_status,
        "completed_modules": sum(1 for s in module_status.values() if s["present"]),
        "expected_modules": len(modules),
        "docs": docs,
        "tests": tests,
        "sanity_check_present": sanity_ok,
        "sanity_check_err": sanity_err,
    }


def main() -> int:
    rows = []
    print(f"{'lane':10s}  {'modules':>9}  {'LoC':>6}  {'docs':>4}  {'tests':>6}  sanity")
    print("-" * 60)
    total_loc = 0
    for lane, modules in LANES.items():
        s = lane_status(lane, modules)
        total_loc += s["pkg_loc"]
        rows.append(s)
        m_summary = f"{s['completed_modules']}/{s['expected_modules']}"
        t_summary = f"{len(s['tests'])}"
        d_summary = f"{len(s['docs'])}"
        sanity = "✓" if s["sanity_check_present"] else "·"
        print(f"{lane:10s}  {m_summary:>9}  {s['pkg_loc']:6d}  {d_summary:>4}  {t_summary:>6}  {sanity}")
    print("-" * 60)
    print(f"{'TOTAL':10s}  {'':9}  {total_loc:6d}")
    print()
    # missing-modules call-out
    for s in rows:
        missing = [m for m, x in s["modules"].items() if not x["present"]]
        if missing:
            print(f"  {s['lane']:10s} missing: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
