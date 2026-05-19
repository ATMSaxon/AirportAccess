"""Safety KPIs for a single corridor.

Implements the eight metrics from §11.1 of the proposal:

1. ``ols_violation_rate`` — fraction of waypoints with SDF < 0 (inside an OLS surface).
2. ``min_separation_lateral_nm`` — minimum 3-D-lateral distance to contemporaneous ADS-B.
3. ``min_separation_vertical_ft`` — minimum vertical distance to contemporaneous ADS-B.
4. ``runway_axis_crossings`` — count of crossings of any extended runway centreline
   within ±5 NM of the runway.
5. ``approach_interference_s``, ``departure_interference_s``, ``missed_approach_overlap_s`` —
   integrated time the corridor centreline spent inside per-runway approach/departure/
   missed-approach polygons (per-airport YAML).
6. ``obstacle_margin_min_m`` — minimum SDF along the path.
7. ``ofv_compliance`` — both endpoints inside their respective OFVs.

Any metric that needs data we don't have falls back to ``None`` rather than 0, so the
joint-eval module can mask them when aggregating.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.data._common import great_circle_nm
from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid
from src.utils.logs import get_logger

LOG = get_logger(__name__)


@dataclass
class SafetyKPI:
    ols_violation_rate: float
    min_separation_lateral_nm: float | None
    min_separation_vertical_ft: float | None
    runway_axis_crossings: int
    approach_interference_s: float
    departure_interference_s: float
    missed_approach_overlap_s: float
    obstacle_margin_min_m: float
    ofv_compliance: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# OLS violation and obstacle margin (sdf-based)
# ---------------------------------------------------------------------------

def _path_sdf(corridor, sdf: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """Sample SDF at each waypoint of the corridor (returns float array, length = N waypoints).

    Tolerates the path_ijk being on a *different* grid than the sdf (e.g. coarsened
    planning vs. fine SDF) by re-snapping via world_to_index of path_enu.
    """
    if corridor.path_enu is None:
        return np.array([], dtype=np.float32)

    if (
        corridor.path_ijk is not None
        and corridor.path_ijk.shape[0] == corridor.path_enu.shape[0]
        and grid.shape[0] >= int(corridor.path_ijk[:, 0].max()) + 1
        and grid.shape[1] >= int(corridor.path_ijk[:, 1].max()) + 1
        and grid.shape[2] >= int(corridor.path_ijk[:, 2].max()) + 1
        and sdf.shape == grid.shape
    ):
        ijk = corridor.path_ijk
        return sdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float32, copy=False)

    # Re-snap from ENU.
    enu = corridor.path_enu
    ix = np.clip(np.floor((enu[:, 0] - grid.x_min) / grid.dx).astype(int), 0, grid.shape[0] - 1)
    iy = np.clip(np.floor((enu[:, 1] - grid.y_min) / grid.dy).astype(int), 0, grid.shape[1] - 1)
    iz = np.clip(np.floor((enu[:, 2] - grid.z_min) / grid.dz).astype(int), 0, grid.shape[2] - 1)
    return sdf[ix, iy, iz].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# ADS-B separation
# ---------------------------------------------------------------------------

def _min_separation(corridor, frame: AirportFrame, adsb) -> tuple[float | None, float | None]:
    """Return (lateral_NM, vertical_ft) min over contemporaneous ADS-B sample points.

    "Contemporaneous" = within ±120 s of the corridor's planned execution window. We
    don't have a corridor execution timestamp, so we use the full ADS-B set — KPI is
    a conservative (worst-case) view.
    """
    if adsb is None or corridor.path_enu is None or len(corridor.path_enu) == 0:
        return None, None
    try:
        x = np.asarray(adsb["x_m"].to_numpy(), dtype=np.float64)
        y = np.asarray(adsb["y_m"].to_numpy(), dtype=np.float64)
        z = np.asarray(adsb["z_msl_m"].to_numpy(), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None, None
    if x.size == 0:
        return None, None

    px = corridor.path_enu[:, 0].astype(np.float64)
    py = corridor.path_enu[:, 1].astype(np.float64)
    pz = corridor.path_enu[:, 2].astype(np.float64)

    min_lat_m = np.inf
    min_vert_m = np.inf
    # Stride over corridor waypoints to keep this O(N_ads x N_path / stride).
    stride = max(1, len(px) // 100)
    for k in range(0, len(px), stride):
        dlat = np.hypot(x - px[k], y - py[k])
        dvert = np.abs(z - pz[k])
        min_lat_m = min(min_lat_m, float(np.min(dlat)))
        min_vert_m = min(min_vert_m, float(np.min(dvert)))
    return float(min_lat_m / 1852.0), float(min_vert_m / 0.3048)


# ---------------------------------------------------------------------------
# Runway-axis crossings + approach/dep interference
# ---------------------------------------------------------------------------

def _runway_geometry(airport_cfg: dict, frame: AirportFrame) -> list[dict]:
    """Return per-runway ENU geometry: threshold, end, axis unit vector, length."""
    out = []
    for rwy in airport_cfg.get("runways", []):
        tx, ty = frame.wgs_to_enu(np.array([rwy["thr_lon"]]), np.array([rwy["thr_lat"]]))
        ex, ey = frame.wgs_to_enu(np.array([rwy["end_lon"]]), np.array([rwy["end_lat"]]))
        thr = np.array([float(tx[0]), float(ty[0])])
        end = np.array([float(ex[0]), float(ey[0])])
        axis = end - thr
        L = float(np.linalg.norm(axis))
        unit = axis / max(L, 1e-6)
        out.append({"id": rwy["id"], "thr": thr, "end": end, "axis": unit, "length_m": L})
    return out


def _runway_axis_crossings(corridor, runways: list[dict], window_nm: float = 5.0) -> int:
    """Count sign changes of perpendicular signed distance from each runway axis."""
    if corridor.path_enu is None or len(corridor.path_enu) < 2:
        return 0
    px = corridor.path_enu[:, 0].astype(np.float64)
    py = corridor.path_enu[:, 1].astype(np.float64)
    n = 0
    window_m = window_nm * 1852.0
    for rwy in runways:
        thr = rwy["thr"]
        unit = rwy["axis"]
        # Perpendicular sign: cross product Z-component.
        rel_x = px - thr[0]
        rel_y = py - thr[1]
        along = rel_x * unit[0] + rel_y * unit[1]
        perp = rel_x * (-unit[1]) + rel_y * unit[0]
        # Only count crossings where along ∈ [-window, length+window].
        in_window = (along >= -window_m) & (along <= rwy["length_m"] + window_m)
        sign_changes = np.diff(np.sign(perp[in_window]))
        n += int(np.sum(np.abs(sign_changes) >= 1))
    return n


def _approach_polygon_overlap(corridor, runways: list[dict], *, cruise_mps: float) -> tuple[float, float, float]:
    """Return (approach_s, departure_s, missed_s) integrated time inside per-runway polygons.

    Polygons are *triangular* approach/departure cones following the Annex-14 parameters
    that the geometry-engineer will publish; here we use a documented placeholder cone:
    inner-edge width 300 m, divergence 0.15 per side, length 5 km. This is an MVP — the
    real geometry will land via ``src.geometry`` once published, and we'll swap to it.
    """
    if corridor.path_enu is None or len(corridor.path_enu) < 2:
        return 0.0, 0.0, 0.0
    px = corridor.path_enu[:, 0].astype(np.float64)
    py = corridor.path_enu[:, 1].astype(np.float64)
    seg = np.diff(corridor.path_enu, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    seg_dt = seg_len / max(cruise_mps, 1e-6)

    inner_half = 150.0
    divergence = 0.15
    length_m = 5000.0

    approach_s = 0.0
    departure_s = 0.0
    missed_s = 0.0
    for rwy in runways:
        # Approach cone: starts at threshold, widens "away from runway" (opposite axis).
        thr = rwy["thr"]
        end = rwy["end"]
        unit = rwy["axis"]
        # Direction of approach (away from runway) is -unit.
        for sign_name, anchor, dir_unit, target in (
            ("approach", thr, -unit, "approach_s"),
            ("departure", end, unit, "departure_s"),
        ):
            rel_x = px - anchor[0]
            rel_y = py - anchor[1]
            along = rel_x * dir_unit[0] + rel_y * dir_unit[1]
            perp = np.abs(rel_x * (-dir_unit[1]) + rel_y * dir_unit[0])
            half_width = inner_half + divergence * np.maximum(along, 0.0)
            mask_pt = (along >= 0) & (along <= length_m) & (perp <= half_width)
            # Aggregate segment time using midpoint-in-polygon test.
            mask_seg = mask_pt[:-1] & mask_pt[1:]
            t = float(np.sum(seg_dt[mask_seg]))
            if target == "approach_s":
                approach_s += t
            else:
                departure_s += t
        # Missed-approach: cone over the *opposite* side of the threshold (overrun direction).
        rel_x = px - end[0]
        rel_y = py - end[1]
        along = rel_x * unit[0] + rel_y * unit[1]
        perp = np.abs(rel_x * (-unit[1]) + rel_y * unit[0])
        half_width = inner_half + divergence * np.maximum(along, 0.0)
        mask_pt = (along >= 0) & (along <= length_m) & (perp <= half_width)
        mask_seg = mask_pt[:-1] & mask_pt[1:]
        missed_s += float(np.sum(seg_dt[mask_seg]))

    return approach_s, departure_s, missed_s


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def safety_for_corridor(
    corridor,
    *,
    sdf: np.ndarray,
    grid: VoxelGrid,
    frame: AirportFrame,
    airport_cfg: dict,
    adsb=None,
    cruise_mps: float = 67.0,
    ofv_start_mask: np.ndarray | None = None,
    ofv_end_mask: np.ndarray | None = None,
) -> SafetyKPI:
    """Compute all eight safety KPIs."""
    if not corridor.feasible or corridor.path_enu is None or len(corridor.path_enu) == 0:
        return SafetyKPI(
            ols_violation_rate=1.0,
            min_separation_lateral_nm=None,
            min_separation_vertical_ft=None,
            runway_axis_crossings=0,
            approach_interference_s=0.0,
            departure_interference_s=0.0,
            missed_approach_overlap_s=0.0,
            obstacle_margin_min_m=float("nan"),
            ofv_compliance=False,
        )

    sdf_path = _path_sdf(corridor, sdf, grid)
    ols_violation = float((sdf_path <= 0).mean()) if sdf_path.size else 1.0
    obstacle_margin = float(sdf_path.min()) if sdf_path.size else float("nan")

    lat_nm, vert_ft = _min_separation(corridor, frame, adsb)
    runways = _runway_geometry(airport_cfg, frame)
    crossings = _runway_axis_crossings(corridor, runways)
    a_s, d_s, m_s = _approach_polygon_overlap(corridor, runways, cruise_mps=cruise_mps)

    # OFV compliance: only assessable when masks are supplied.
    ofv_ok: bool | None
    if (
        ofv_start_mask is not None and ofv_end_mask is not None
        and corridor.path_ijk is not None and len(corridor.path_ijk) >= 2
    ):
        start = tuple(int(x) for x in corridor.path_ijk[0])
        end = tuple(int(x) for x in corridor.path_ijk[-1])
        try:
            ofv_ok = bool(ofv_start_mask[start] and ofv_end_mask[end])
        except IndexError:
            ofv_ok = None
    else:
        ofv_ok = None

    return SafetyKPI(
        ols_violation_rate=ols_violation,
        min_separation_lateral_nm=lat_nm,
        min_separation_vertical_ft=vert_ft,
        runway_axis_crossings=int(crossings),
        approach_interference_s=float(a_s),
        departure_interference_s=float(d_s),
        missed_approach_overlap_s=float(m_s),
        obstacle_margin_min_m=obstacle_margin,
        ofv_compliance=ofv_ok,
    )
