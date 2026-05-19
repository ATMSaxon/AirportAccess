#!/usr/bin/env python3
"""M8 — produce the TR Part C figure set + narrative report numeric inserts.

Reads `results/eval/{KLAX,KSFO}/kpi_table.parquet` and the per-baseline corridor outputs,
emits the figures listed in `refine-logs/EXPERIMENT_PLAN.md` M8.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import paths, logs, io  # noqa: E402

LOG = logs.get_logger("paper_figs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airports", nargs="+", default=["KLAX", "KSFO"])
    args = parser.parse_args()

    figs_dir = paths.FIGURES / "eval"
    figs_dir.mkdir(parents=True, exist_ok=True)
    inserted = []

    try:
        from src.analysis import joint_eval  # owned by planning-engineer
    except ImportError as e:
        LOG.error("joint_eval not available yet: %s — run after analysis lane lands", e)
        return 1

    for icao in args.airports:
        kpi_path = paths.RESULTS / "eval" / icao / "kpi_table.parquet"
        if not kpi_path.exists():
            LOG.warning("Skipping %s — no kpi_table.parquet at %s", icao, kpi_path)
            continue
        out = joint_eval.make_figures(kpi_path, figs_dir / icao)
        inserted.append({"airport": icao, "figures": out})

    summary_path = figs_dir / "_index.json"
    io.write_json(summary_path, {"airports": inserted})
    LOG.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
