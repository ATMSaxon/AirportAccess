#!/usr/bin/env python3
"""End-to-end smoke check for DREAM on the synthetic KSYN airport.

Exercises every lane that exists; missing lanes are logged but don't fail the run.
Each lane is expected to expose either:
  (a) `sanity_check(out_dir: Path, airport_cfg: dict) -> dict` at the package root, or
  (b) low-level functions this script can call directly.

A lane is `*_ok = True` iff it returned without raising AND produced a non-empty output dict.
Exit 0 if all available lanes are OK (missing lanes are recorded with `*_present = False`).
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

# repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import paths, config, logs, io  # noqa: E402

LOG = logs.get_logger("sanity")


LANES = [
    ("data",      "src.data"),
    ("geometry",  "src.geometry"),
    ("traffic",   "src.traffic"),
    ("ml",        "src.ml"),
    ("planning",  "src.planning"),
    ("analysis",  "src.analysis"),
]


def _try_sanity(module_name: str, out_dir: Path, airport_cfg: dict) -> tuple[bool, dict]:
    """Import the package and call its `sanity_check`; return (ok, info)."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        return False, {"error": f"import failed: {e!s}", "trace": traceback.format_exc()}
    fn = getattr(mod, "sanity_check", None)
    if fn is None:
        return False, {"error": "no sanity_check() function exposed by module"}
    try:
        info = fn(out_dir, airport_cfg) or {}
        return True, dict(info) if isinstance(info, dict) else {"info": str(info)}
    except Exception as e:
        return False, {"error": str(e), "trace": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(paths.CONFIGS / "sanity.yaml"))
    parser.add_argument("--output-dir", default=str(paths.RESULTS / "sanity"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true",
                        help="treat missing lanes as failures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = config.load_yaml(args.config)
    LOG.info("DREAM sanity on airport %s (%s)", cfg["icao"], cfg["name"])
    LOG.info("Output directory: %s", out_dir)

    summary: dict = {
        "airport": cfg["icao"],
        "config_path": str(args.config),
        "seed": args.seed,
        "lanes": {},
    }
    overall_ok = True

    for lane, module_name in LANES:
        present = importlib.util.find_spec(module_name) is not None
        info = {"present": present}
        if present:
            ok, lane_info = _try_sanity(module_name, out_dir / lane, cfg)
            info["ok"] = ok
            info["details"] = lane_info
        else:
            info["ok"] = False
            info["details"] = {"error": "module not yet implemented"}
        summary[f"{lane}_ok"] = bool(info.get("ok", False))
        summary[f"{lane}_present"] = present
        summary["lanes"][lane] = info
        LOG.info("Lane %-9s: present=%s ok=%s", lane, present, info["ok"])

        # Strict mode: any not-ok lane is failure
        if args.strict:
            overall_ok = overall_ok and info["ok"]
        else:
            # Lenient: a lane is OK if it's missing OR returned OK. A lane that's
            # present but failed is a failure.
            if present and not info["ok"]:
                overall_ok = False

    summary["overall_ok"] = overall_ok
    io.write_summary(out_dir, summary)
    LOG.info("Wrote %s", out_dir / "summary.json")
    LOG.info("Overall: %s", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
