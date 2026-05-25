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


def _discover(icao: str, date: str | None = None) -> tuple[list[Path], Path | None]:
    """Find ADS-B + runway-config parquets under `data/processed/<ICAO>/`.

    When ``date`` is given, only the matching ``adsb_<date>.parquet`` and
    ``runway_config_<date>.parquet`` are selected — so the sampler can be
    invoked per-day for the temporal-day holdout.
    """
    pdir = paths.PROCESSED / icao
    if date:
        adsb = sorted(pdir.glob(f"adsb_{date}.parquet"))
        rc = sorted(pdir.glob(f"runway_config_{date}.parquet"))
        return adsb, (rc[0] if rc else None)
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
    ap.add_argument("--date", default=None,
                    help=("Single sampling date (YYYY-MM-DD). When set, "
                          "loads adsb_<date>.parquet + runway_config_<date>.parquet "
                          "and writes counterfactuals_<date>.parquet with a "
                          "`sample_date` column for temporal-day holdout."))
    ap.add_argument("--adsb", nargs="+", default=None,
                    help="Override ADS-B parquet path(s). Default = auto-discover.")
    ap.add_argument("--runway-config", default=None,
                    help="Override runway-config parquet path. Default = auto-discover.")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--output-name", default=None,
                    help="Override the output parquet filename (no path).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    setup_logging("DEBUG" if args.debug else "INFO")

    adsb_paths, rc_path = _discover(args.airport, date=args.date)
    if args.adsb:
        adsb_paths = [Path(p) for p in args.adsb]
    if args.runway_config:
        rc_path = Path(args.runway_config)
    if rc_path is None:
        date_hint = f" for date {args.date}" if args.date else ""
        raise SystemExit(
            f"No runway-config parquet under data/processed/{args.airport}/{date_hint}. "
            f"M3 must run before M4. Pass --runway-config to override.")
    if args.date and not adsb_paths:
        logger.warning("No adsb_%s.parquet under data/processed/%s/ — "
                       "labels will rely on geometry only.", args.date, args.airport)

    out_dir = Path(args.output_dir) if args.output_dir else (paths.PROCESSED / args.airport)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        out_name = args.output_name
    elif args.date:
        out_name = f"counterfactuals_{args.date}.parquet"
    else:
        out_name = "counterfactuals.parquet"
    out_path = out_dir / out_name

    df = sample_and_label(
        icao=args.airport,
        n=args.n,
        seed=args.seed,
        scenario=args.scenario,
        adsb_paths=adsb_paths,
        runway_config_path=rc_path,
        output_path=out_path,
        sample_date=args.date,
    )
    logger.info("DONE — %d rows, %d conflicts (rate=%.3f) → %s",
                len(df), int(df["conflict"].sum()), float(df["conflict"].mean()),
                out_path)


if __name__ == "__main__":
    main()
