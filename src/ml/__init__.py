"""Risk field learning, counterfactual injection, conformal calibration.

Public API:

* ``counterfactual`` — sample candidate eVTOL segments + label conflicts.
* ``features``       — feature extraction (numeric + cfg one-hot).
* ``risk_field``     — LR / RF / XGB / MLP trainers + risk-grid export.
* ``conformal``      — split conformal prediction.
* ``AirportGeom``    — local geometry helper used by the above.

Sanity hook (for ``scripts/run_sanity.py``)::

    from src.ml import sanity_check
    info = sanity_check(out_dir, airport_cfg)
"""
from __future__ import annotations
from pathlib import Path

from . import counterfactual, features, risk_field, conformal  # noqa: F401
from ._geom import AirportGeom  # re-exported convenience

__all__ = [
    "counterfactual",
    "features",
    "risk_field",
    "conformal",
    "AirportGeom",
    "sanity_check",
]


def sanity_check(out_dir: Path, airport_cfg: dict) -> dict:
    """Tiny offline ML smoke test on the synthetic KSYN airport.

    Samples counterfactual eVTOL segments inside the airport extract box,
    labels conflicts using only geometry (no ADS-B), extracts features, fits
    a baseline LR with split-conformal calibration, and writes
    ``counterfactuals.parquet``, ``features.parquet``, and ``ml_sanity.json``
    under ``out_dir``. Returns a dict consumed by ``scripts/run_sanity.py``
    so it can mark ``ml_ok=True``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from . import _sanity                       # local import keeps top-level light
    return _sanity.run(out_dir, airport_cfg)
