"""Split conformal prediction for a binary risk classifier.

We use APS-style symmetric conformal intervals on the predicted probability:

  Given a calibration set `(p_cal, y_cal)` with `p_cal ∈ [0,1]` the score is
      s_i = |p_cal_i - y_cal_i|              (absolute residual)
  Quantile  q̂ = ⌈(n+1)(1-α)⌉ / n  of {s_i}.
  Prediction set for a new score p: [p - q̂, p + q̂] ∩ [0,1].

Target nominal coverage is 1 − α; we use α = 0.1 → 0.9 nominal coverage. The
empirical coverage on a held-out test set is reported alongside the mean
interval width.

API:

    cp = SplitConformal(alpha=0.1)
    cp.fit(p_cal, y_cal)
    lo, hi = cp.predict(p_test)
    cov = empirical_coverage(lo, hi, y_test)
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class SplitConformal:
    alpha: float = 0.1                    # 1 − target coverage
    qhat: float | None = None             # calibrated half-width

    def fit(self, p_cal, y_cal) -> "SplitConformal":
        p_cal = np.asarray(p_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)
        if p_cal.shape != y_cal.shape:
            raise ValueError("p_cal and y_cal must have the same shape")
        s = np.abs(p_cal - y_cal)
        n = len(s)
        if n == 0:
            raise ValueError("calibration set must be non-empty")
        # Standard conformal quantile correction (Romano et al. 2019).
        q_level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.qhat = float(np.quantile(s, q_level, method="higher"))
        return self

    def predict(self, p_test) -> tuple[np.ndarray, np.ndarray]:
        if self.qhat is None:
            raise RuntimeError("call fit() before predict()")
        p = np.asarray(p_test, dtype=np.float64)
        lo = np.clip(p - self.qhat, 0.0, 1.0)
        hi = np.clip(p + self.qhat, 0.0, 1.0)
        return lo, hi

    def width(self, p_test) -> np.ndarray:
        lo, hi = self.predict(p_test)
        return hi - lo


def empirical_coverage(lo, hi, y_true) -> float:
    """Fraction of true labels (binary) for which `lo ≤ y ≤ hi`."""
    lo = np.asarray(lo); hi = np.asarray(hi); y = np.asarray(y_true)
    return float(np.mean((y >= lo) & (y <= hi)))


def calibrate_split(p_all, y_all, *, cal_fraction: float = 0.2,
                    alpha: float = 0.1, seed: int = 0
                    ) -> tuple[SplitConformal, dict]:
    """Random calibration/test split → fit conformal → return (cp, metrics)."""
    p_all = np.asarray(p_all, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(p_all)
    idx = rng.permutation(n)
    n_cal = max(2, int(round(cal_fraction * n)))
    cal = idx[:n_cal]; tst = idx[n_cal:]
    cp = SplitConformal(alpha=alpha).fit(p_all[cal], y_all[cal])
    lo, hi = cp.predict(p_all[tst])
    cov = empirical_coverage(lo, hi, y_all[tst])
    width = float(np.mean(hi - lo))
    return cp, {"empirical_coverage": cov,
                "nominal_coverage": 1.0 - alpha,
                "mean_interval_width": width,
                "n_calibration": int(n_cal),
                "n_test": int(len(tst)),
                "qhat": cp.qhat}


__all__ = ["SplitConformal", "empirical_coverage", "calibrate_split"]
