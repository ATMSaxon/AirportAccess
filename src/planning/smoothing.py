"""Path post-processing: RDP simplification + cubic-spline resample.

Used to turn the staircase voxel path that A* emits into a smoother corridor centreline
before it lands in the GeoJSON output. We intentionally keep this conservative: smoothing
must not push the path outside the original SDF/envelope feasibility, so the planner caller
can still verify the smoothed path with a cheap per-sample mask lookup if desired.
"""
from __future__ import annotations

import numpy as np


def _perp_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b):
        return float(np.linalg.norm(p - a))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker line simplification on 3-D points.

    Returns the kept-points (always includes the first and last).
    """
    if len(points) <= 2:
        return points.copy()
    # Find the point with maximum distance from the chord.
    a, b = points[0], points[-1]
    max_dist = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = _perp_distance(points[i], a, b)
        if d > max_dist:
            max_dist = d
            idx = i
    if max_dist > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([points[0], points[-1]])


def resample_path(points: np.ndarray, n_samples: int = 200) -> np.ndarray:
    """Uniformly resample a polyline at ``n_samples`` points along arc-length."""
    if len(points) < 2:
        return points.copy()
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.tile(points[0:1], (n_samples, 1))
    s_uni = np.linspace(0.0, float(s[-1]), n_samples)
    out = np.empty((n_samples, points.shape[1]))
    for d in range(points.shape[1]):
        out[:, d] = np.interp(s_uni, s, points[:, d])
    return out


def smooth_path(points: np.ndarray, *, rdp_eps_m: float = 50.0, n_samples: int = 200) -> np.ndarray:
    """RDP-simplify then linearly resample to ``n_samples`` waypoints."""
    if len(points) < 3:
        return points.copy()
    simplified = rdp(points.astype(np.float64), rdp_eps_m)
    return resample_path(simplified, n_samples).astype(points.dtype)
