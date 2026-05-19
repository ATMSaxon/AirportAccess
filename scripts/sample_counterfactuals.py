#!/usr/bin/env python
"""Sample counterfactual eVTOL segments + label conflicts.

Example:
    python scripts/sample_counterfactuals.py --airport KLAX --n 200000 --seed 42

Inputs (auto-discovered under `data/processed/<ICAO>/`):
* `adsb_*.parquet`  (D5 schema)        — concatenated.
* `runway_config_*.parquet` (M3 output) — concatenated.

Output:
* `data/processed/<ICAO>/counterfactuals.parquet` (+ sibling `_manifest.json`)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# repo root on sys.path (so `python scripts/foo.py` works without installing).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.counterfactual import sample_and_label  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.logs import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def _discover(icao: str) -> tuple[list[Path], Path | None]:
    """Find ADS-B + runway-config parquets under `data/processed/<ICAO>/`."""
    pdir = paths.PROCESSED / icao
    adsb = sorted(pdir.glob("adsb_*.parquet"))
    rc = sorted(pdir.glob("runway_config_*.parquet"))
    if not rc:
        rc = sorted(pdir.glob("runway_config*.parquet"))
    return adsb, (rc[0] if rc else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--airport", required=True, help="ICAO code, e.g. KLAX")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scenario", default="cost_weights")
    ap.add_argument("--adsb", nargs="+", default=None,
                    help="Override ADS-B parquet path(s). Default = auto-discover.")
    ap.add_argument("--runway-config", default=None,
                    help="Override runway-config parquet path. Default = auto-discover.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    setup_logging("DEBUG" if args.debug else "INFO")

    adsb_paths, rc_path = _discover(args.airport)
    if args.adsb:
        adsb_paths = [Path(p) for p in args.adsb]
    if args.runway_config:
        rc_path = Path(args.runway_config)
    if rc_path is None:
        raise SystemExit(
            f"No runway-config parquet under data/processed/{args.airport}/. "
            f"M3 must run before M4. Pass --runway-config to override.")

    out_dir = Path(args.output_dir) if args.output_dir else (paths.PROCESSED / args.airport)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "counterfactuals.parquet"

    df = sample_and_label(
        icao=args.airport,
        n=args.n,
        seed=args.seed,
        scenario=args.scenario,
        adsb_paths=adsb_paths,
        runway_config_path=rc_path,
        output_path=out_path,
    )
    logger.info("DONE — %d rows, %d conflicts (rate=%.3f)",
                len(df), int(df["conflict"].sum()), float(df["conflict"].mean()))


if __name__ == "__main__":
    main()
