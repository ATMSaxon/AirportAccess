"""Counterfactual eVTOL-segment sampling and conflict labelling.

Pipeline (M4-labels):

  1. For each requested sample, pick a 15-min time slice from the runway-config
     parquet (M3 output). The active runway configuration `R_t` for that slice
     determines which approach / departure / missed-approach surfaces are live.
  2. Pick a random vertiport, draw an origin uniformly from its OFV (Annex-14
     yaml: `vertiport_ofv`).
  3. Pick a random destination uniformly inside `A_static` = airport extract
     box at altitude > 200 m above field elev. Reject candidates whose climb /
     descent angle exceeds the eVTOL kinematic envelope from
     `configs/scenarios/cost_weights.yaml`.
  4. Compute the segment's midpoint and time-mapping (constant-3-D-speed at
     the eVTOL cruise speed for time-alignment purposes).
  5. Label `conflict = 1` if ANY of:
       a) lateral < 1.5 NM AND vertical < 1000 ft vs contemporaneous ADS-B
       b) crosses active runway axis below 2000 ft AGL
       c) intersects an active approach / departure prism
       d) intersects an active missed-approach surface
       e) sdf(midpoint) < safety buffer (default 30 m)

Outputs a Parquet table whose schema is `INTERFACES.md` stable.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence
import logging
import math
import numpy as np
import pandas as pd

from ..utils import config as cfg_io
from ..utils import paths
from ..utils.io import write_manifest, write_json
from ..utils.logs import get_logger
from ._geom import AirportGeom, NM_M, FT_M, adsb_density_box

logger = get_logger(__name__)


L_MIN_M = 1.5 * NM_M           # lateral conflict threshold (2778 m)
V_MIN_M = 1000.0 * FT_M        # vertical conflict threshold (304.8 m)
AXIS_Z_MAX_AGL_M = 2000.0 * FT_M
SDF_BUFFER_M = 30.0


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SEGMENT_COLUMNS = [
    "seg_id", "vertiport_id", "icao", "config_id",
    "t_start_utc", "t_end_utc", "duration_s",
    "x0_m", "y0_m", "z0_m",
    "x1_m", "y1_m", "z1_m",
    "mid_x_m", "mid_y_m", "mid_z_m", "mid_t_utc",
    "length_m", "climb_angle_deg", "cruise_speed_mps",
    "active_arrivals", "active_departures",
    "conflict", "cause",
    "min_lat_m_adsb", "min_vert_m_adsb",
    "axis_cross", "approach_hit", "departure_hit", "missed_hit",
    "sdf_mid_m",
]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@dataclass
class EvtolKinematics:
    cruise_speed_mps: float
    climb_speed_mps: float
    descent_speed_mps: float
    max_climb_rate_mps: float
    max_descent_rate_mps: float

    @property
    def max_climb_angle_rad(self) -> float:
        return math.asin(self.max_climb_rate_mps / max(self.climb_speed_mps, 1.0))

    @property
    def max_descent_angle_rad(self) -> float:
        return math.asin(self.max_descent_rate_mps / max(self.descent_speed_mps, 1.0))

    @classmethod
    def from_cfg(cls, scenario: dict) -> "EvtolKinematics":
        e = scenario["evtol"]
        return cls(
            cruise_speed_mps=float(e["cruise_speed_mps"]),
            climb_speed_mps=float(e["climb_speed_mps"]),
            descent_speed_mps=float(e["descent_speed_mps"]),
            max_climb_rate_mps=float(e["max_climb_rate_mps"]),
            max_descent_rate_mps=float(e["max_descent_rate_mps"]),
        )


def _load_runway_config(path: Path) -> pd.DataFrame:
    """Load M3 runway-config parquet.

    Expected schema (we are permissive on the input):
      time_utc (datetime64[ns, UTC]),
      config_id  (str),
      active_arrivals (str, e.g. "06L;06R"),
      active_departures (str, e.g. "07L;07R").

    If only `active_runways` is present we use it for both arrivals and
    departures (degenerate single-config case).
    """
    df = pd.read_parquet(path)
    if "time_utc" not in df.columns:
        raise ValueError(f"{path}: runway-config parquet must have time_utc column")
    if "config_id" not in df.columns:
        df["config_id"] = df.get("config", "UNKNOWN")
    if "active_arrivals" not in df.columns:
        # Fall back to active_runways or to all configured runways.
        df["active_arrivals"] = df.get("active_runways", "")
    if "active_departures" not in df.columns:
        df["active_departures"] = df.get("active_runways", df["active_arrivals"])
    return df


def _parse_active(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [str(x) for x in cell]
    return [s for s in str(cell).split(";") if s]


def _sample_segment(geom: AirportGeom, kin: EvtolKinematics,
                    vertiport_ids: Sequence[str], rng: np.random.Generator,
                    max_kinematic_retries: int = 20) -> dict:
    for _ in range(max_kinematic_retries):
        vid = vertiport_ids[rng.integers(0, len(vertiport_ids))]
        x0, y0, z0 = geom.sample_vertiport_ofv(vid, rng)
        x1, y1, z1 = geom.sample_static_volume(rng)
        dxy = math.hypot(x1 - x0, y1 - y0)
        dz = z1 - z0
        if dxy < 200.0:                    # too short to be a useful segment
            continue
        ang = math.atan2(dz, dxy)
        if ang > kin.max_climb_angle_rad or ang < -kin.max_descent_angle_rad:
            continue
        length = math.sqrt(dxy * dxy + dz * dz)
        return {
            "vertiport_id": vid,
            "x0_m": x0, "y0_m": y0, "z0_m": z0,
            "x1_m": x1, "y1_m": y1, "z1_m": z1,
            "length_m": length,
            "climb_angle_deg": math.degrees(ang),
        }
    raise RuntimeError("could not sample a kinematically feasible segment in 20 tries")


def _segment_position_at(seg: dict, t_offset_s: float, duration_s: float) -> tuple[float, float, float]:
    """Linear interpolation along the segment for time-alignment."""
    f = max(0.0, min(1.0, t_offset_s / max(duration_s, 1e-6)))
    return (seg["x0_m"] + f * (seg["x1_m"] - seg["x0_m"]),
            seg["y0_m"] + f * (seg["y1_m"] - seg["y0_m"]),
            seg["z0_m"] + f * (seg["z1_m"] - seg["z0_m"]))


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------
def label_segment(seg: dict, geom: AirportGeom, kin: EvtolKinematics,
                  adsb_window_df: pd.DataFrame | None,
                  *, l_min_m: float = L_MIN_M, v_min_m: float = V_MIN_M,
                  sdf_buffer_m: float = SDF_BUFFER_M) -> dict:
    """Compute conflict label + cause tags for one segment.

    `adsb_window_df` must already be filtered to the segment's time window.
    """
    p0 = np.array([seg["x0_m"], seg["y0_m"], seg["z0_m"]])
    p1 = np.array([seg["x1_m"], seg["y1_m"], seg["z1_m"]])
    arr = _parse_active(seg["active_arrivals"])
    dep = _parse_active(seg["active_departures"])

    # ---- a) ADS-B time-aligned 3-D separation ----
    min_lat = math.inf
    min_vert = math.inf
    if adsb_window_df is not None and len(adsb_window_df) > 0:
        t0 = pd.Timestamp(seg["t_start_utc"])
        dur = float(seg["duration_s"])
        if dur > 0:
            t_offsets = (adsb_window_df["time_utc"].to_numpy() - np.datetime64(t0)) \
                .astype("timedelta64[ms]").astype(np.int64) / 1000.0
            fs = np.clip(t_offsets / dur, 0.0, 1.0)
            xs = p0[0] + fs * (p1[0] - p0[0])
            ys = p0[1] + fs * (p1[1] - p0[1])
            zs = p0[2] + fs * (p1[2] - p0[2])
            d_lat = np.hypot(
                adsb_window_df["x_m"].to_numpy() - xs,
                adsb_window_df["y_m"].to_numpy() - ys,
            )
            d_v = np.abs(adsb_window_df["z_msl_m"].to_numpy() - zs)
            conflict_mask = (d_lat < l_min_m) & (d_v < v_min_m)
            min_lat = float(np.min(d_lat)) if len(d_lat) else math.inf
            min_vert = float(np.min(d_v)) if len(d_v) else math.inf
            adsb_conflict = bool(conflict_mask.any())
        else:
            adsb_conflict = False
    else:
        adsb_conflict = False

    # ---- b) Active runway axis crossing below 2000 ft AGL ----
    axis_cross = geom.segment_crosses_runway_axis(p0, p1, list(arr) + list(dep),
                                                  AXIS_Z_MAX_AGL_M)

    # ---- c, d) Prism / missed-approach intersection ----
    approach_hit, departure_hit, missed_hit = geom.segment_intersects_prism(
        p0, p1, arr, dep)

    # ---- e) SDF buffer at midpoint ----
    mid = 0.5 * (p0 + p1)
    sdf_mid = geom.sdf(mid[0], mid[1], mid[2], arr, dep)
    sdf_violation = sdf_mid < sdf_buffer_m

    causes = []
    if adsb_conflict:    causes.append("adsb_near")
    if axis_cross:       causes.append("axis_cross")
    if approach_hit:     causes.append("approach_prism")
    if departure_hit:    causes.append("departure_prism")
    if missed_hit:       causes.append("missed_approach")
    if sdf_violation:    causes.append("sdf_buffer")
    conflict = int(bool(causes))

    return {
        "conflict": conflict,
        "cause": ",".join(causes),
        "min_lat_m_adsb": float(min_lat) if math.isfinite(min_lat) else None,
        "min_vert_m_adsb": float(min_vert) if math.isfinite(min_vert) else None,
        "axis_cross": bool(axis_cross),
        "approach_hit": bool(approach_hit),
        "departure_hit": bool(departure_hit),
        "missed_hit": bool(missed_hit),
        "sdf_mid_m": float(sdf_mid),
    }


# ---------------------------------------------------------------------------
# Top-level sample + label entry point
# ---------------------------------------------------------------------------
def sample_and_label(*, icao: str, n: int, seed: int = 42,
                     scenario: str = "cost_weights",
                     adsb_paths: Sequence[Path] | None = None,
                     runway_config_path: Path | None = None,
                     adsb_df: pd.DataFrame | None = None,
                     runway_config_df: pd.DataFrame | None = None,
                     time_slice_minutes: int = 15,
                     output_path: Path | None = None,
                     ) -> pd.DataFrame:
    """Sample N counterfactual eVTOL segments for `icao` and label them.

    Either pass `adsb_paths` + `runway_config_path` (production path), or
    `adsb_df` + `runway_config_df` directly (tests / sanity).
    """
    rng = np.random.default_rng(seed)
    geom = AirportGeom.from_icao(icao)
    kin = EvtolKinematics.from_cfg(cfg_io.load_scenario(scenario))
    vertiport_ids = list(geom.vertiports.keys())
    if not vertiport_ids:
        raise RuntimeError(f"No vertiports configured for {icao}")

    # Load ADS-B (concatenated across requested days).
    if adsb_df is None and adsb_paths:
        frames = [pd.read_parquet(p) for p in adsb_paths if Path(p).exists()]
        adsb_df = pd.concat(frames, ignore_index=True) if frames else None
    if adsb_df is not None and len(adsb_df) > 0:
        if not pd.api.types.is_datetime64_any_dtype(adsb_df["time_utc"]):
            adsb_df["time_utc"] = pd.to_datetime(adsb_df["time_utc"], utc=True)
        adsb_df = adsb_df.sort_values("time_utc").reset_index(drop=True)
        logger.info("ADS-B rows loaded: %d (range %s → %s)",
                    len(adsb_df), adsb_df["time_utc"].min(), adsb_df["time_utc"].max())

    # Load runway-config slices.
    if runway_config_df is None and runway_config_path is not None:
        runway_config_df = _load_runway_config(Path(runway_config_path))
    if runway_config_df is None or len(runway_config_df) == 0:
        raise RuntimeError("runway_config required (provide runway_config_path or runway_config_df)")
    if not pd.api.types.is_datetime64_any_dtype(runway_config_df["time_utc"]):
        runway_config_df["time_utc"] = pd.to_datetime(runway_config_df["time_utc"], utc=True)
    runway_config_df = runway_config_df.sort_values("time_utc").reset_index(drop=True)

    rows: list[dict] = []
    half_slice = pd.Timedelta(minutes=time_slice_minutes) / 2

    for i in range(n):
        # Pick a runway-config slice uniformly at random.
        idx = int(rng.integers(0, len(runway_config_df)))
        rc = runway_config_df.iloc[idx]
        slice_t = pd.Timestamp(rc["time_utc"])
        active_arrivals = _parse_active(rc["active_arrivals"])
        active_departures = _parse_active(rc["active_departures"])

        # Sample a feasible segment geometry.
        seg = _sample_segment(geom, kin, vertiport_ids, rng)
        duration_s = seg["length_m"] / max(kin.cruise_speed_mps, 1e-3)
        # Pick a start time within the slice.
        offset = rng.uniform(0.0, time_slice_minutes * 60.0 - duration_s) \
            if duration_s < time_slice_minutes * 60.0 else 0.0
        t_start = slice_t + pd.Timedelta(seconds=float(offset)) - half_slice
        t_end = t_start + pd.Timedelta(seconds=float(duration_s))
        mid_t = t_start + (t_end - t_start) / 2
        mid = (0.5 * (seg["x0_m"] + seg["x1_m"]),
               0.5 * (seg["y0_m"] + seg["y1_m"]),
               0.5 * (seg["z0_m"] + seg["z1_m"]))
        seg.update({
            "seg_id": f"{icao}-{i:08d}",
            "icao": icao,
            "config_id": str(rc.get("config_id", "UNKNOWN")),
            "t_start_utc": t_start,
            "t_end_utc": t_end,
            "duration_s": float(duration_s),
            "mid_x_m": mid[0], "mid_y_m": mid[1], "mid_z_m": mid[2], "mid_t_utc": mid_t,
            "active_arrivals": ";".join(active_arrivals),
            "active_departures": ";".join(active_departures),
            "cruise_speed_mps": kin.cruise_speed_mps,
        })

        # Filter ADS-B to the segment's time window.
        if adsb_df is not None and len(adsb_df) > 0:
            mask = (adsb_df["time_utc"] >= t_start) & (adsb_df["time_utc"] <= t_end)
            window = adsb_df.loc[mask, ["time_utc", "x_m", "y_m", "z_msl_m"]]
        else:
            window = None

        labels = label_segment(seg, geom, kin, window)
        seg.update(labels)
        rows.append(seg)

    df = pd.DataFrame(rows, columns=SEGMENT_COLUMNS)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        write_manifest(output_path,
                       source="ml.counterfactual.sample_and_label",
                       params={"icao": icao, "n": n, "seed": seed,
                               "scenario": scenario,
                               "l_min_m": L_MIN_M, "v_min_m": V_MIN_M,
                               "axis_z_max_agl_m": AXIS_Z_MAX_AGL_M,
                               "sdf_buffer_m": SDF_BUFFER_M},
                       extra={"n_conflicts": int(df["conflict"].sum()),
                              "conflict_rate": float(df["conflict"].mean())})
        logger.info("wrote %d segments (%d conflicts, rate=%.3f) → %s",
                    len(df), int(df["conflict"].sum()), float(df["conflict"].mean()),
                    output_path)
    return df


__all__ = [
    "EvtolKinematics", "SEGMENT_COLUMNS",
    "sample_and_label", "label_segment",
    "L_MIN_M", "V_MIN_M", "AXIS_Z_MAX_AGL_M", "SDF_BUFFER_M",
]
