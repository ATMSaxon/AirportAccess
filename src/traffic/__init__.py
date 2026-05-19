"""ADS-B parsing, runway-configuration inference, density fields, dynamic envelope.

Public API (kept stable — see `src/traffic/SCHEMAS.md`):

    from src.traffic import adsb_clean, classify, runway_config, density, envelope

Sanity hook (for `scripts/run_sanity.py`):

    from src.traffic import sanity_check
    info = sanity_check(out_dir, airport_cfg)
"""
from __future__ import annotations
from pathlib import Path

from . import adsb_clean, classify, runway_config, density, envelope  # noqa: F401


def sanity_check(out_dir: Path, airport_cfg: dict) -> dict:
    """Tiny offline smoke test for the traffic lane on the synthetic KSYN airport.

    Builds a synthetic 1-hour ADS-B sample (3 arrivals + 1 departure + 1
    overflight + 1 outlier-injected track), runs the full M3 pipeline through
    the envelope builder, and writes a parquet + zarr/npz under ``out_dir``.

    Returns a dict with ``ok=True`` plus per-stage metrics. Raises on hard errors.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    from . import _sanity                            # local import keeps top-level light
    return _sanity.run(out_dir, airport_cfg)
