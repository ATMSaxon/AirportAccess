#!/usr/bin/env python3
"""M1 orchestrator — call every `src.data.source_*.fetch()` for a given airport + window.

Each source module is expected to expose:
    def fetch(airport_cfg: dict, window: str, out_dir: pathlib.Path) -> dict
        returns a manifest-like dict and writes its primary artefact (with `_manifest.json`)
        OR writes a `<source>.OFFLINE.json` and returns it.

Missing source modules are recorded but don't fail the run.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import paths, config, logs, io  # noqa: E402

LOG = logs.get_logger("acquire_all")

SOURCES = [
    "source_faa_nasr",
    "source_faa_dof",
    "source_usgs_3dep",
    "source_osm",
    "source_opensky",
    "source_noaa_wx",
    "source_lawa",
    "source_bts",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", required=True, help="ICAO code (e.g. KLAX)")
    parser.add_argument("--window", default="2024-08", help="month or single date")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sources", nargs="*", default=None,
                        help="restrict to a subset of source ids (default: all 8)")
    args = parser.parse_args()

    cfg = config.load_airport(args.airport)
    out_dir = Path(args.output_dir) if args.output_dir else paths.airport_dir(args.airport, "processed")
    LOG.info("Acquiring %s for %s into %s", args.window, args.airport, out_dir)

    sources = args.sources or SOURCES
    inventory: dict = {"airport": args.airport, "window": args.window, "sources": {}}
    failed: list[str] = []
    for src_id in sources:
        module_name = f"src.data.{src_id}"
        if importlib.util.find_spec(module_name) is None:
            LOG.warning("Source module not implemented yet: %s", module_name)
            inventory["sources"][src_id] = {"status": "MISSING_MODULE"}
            failed.append(src_id)
            continue
        try:
            mod = importlib.import_module(module_name)
            fetch = getattr(mod, "fetch", None)
            if fetch is None:
                LOG.warning("%s has no fetch()", module_name)
                inventory["sources"][src_id] = {"status": "NO_FETCH_FN"}
                failed.append(src_id)
                continue
            LOG.info("→ %s", src_id)
            info = fetch(cfg, args.window, out_dir) or {"status": "OK"}
            inventory["sources"][src_id] = info
        except Exception as e:
            LOG.exception("%s failed", src_id)
            inventory["sources"][src_id] = {"status": "ERROR", "error": str(e)}
            failed.append(src_id)

    io.write_json(out_dir / "_inventory.json", inventory)
    LOG.info("Wrote %s", out_dir / "_inventory.json")
    LOG.info("Done. %d/%d sources completed.", len(sources) - len(failed), len(sources))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
