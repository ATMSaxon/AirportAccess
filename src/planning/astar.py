"""Envelope-constrained A* corridor planner for eVTOL.

The ``Planner`` class consumes a pre-built ``PlannerInputs`` bundle (grid + arrays) and
returns a ``Corridor`` for a chosen baseline. All disk I/O happens in
``src/planning/loaders.py`` and ``src/planning/corridor.py``; this module is pure
Python/numpy and trivially unit-testable against synthetic inputs.

Cost function (per edge, per the proposal):

    J_edge = α_T·T + α_E·E + α_ρ·ρ + α_N·N + α_I·I + α_turn·turn

with edge-time `T = length / speed`, energy `E = P_kw·1000·T`, risk `ρ` sampled at the
neighbour cell, noise/population `N = density·length`, capacity-impact proxy `I =
clip(1 - sdf/d_max, 0, 1)`, and `turn` the geometric angle change between this edge and
the previous one.

Heuristic: `α_T · L2_metric / cruise_speed` — admissible because (a) any extra cost terms
are non-negative and (b) `α_T·T` is non-negative.

Baseline gating is the only place where baselines diverge:

    B0    : skipped here; corridor.py returns Corridor(feasible=False) directly.
    B1    : straight ENU line, ignores envelope+A_static. No A* call.
    B2    : A* on A_static, no risk, no capacity-impact term.
    B3    : A* on A_static AND envelope, no risk term.
    B4    : A* on A_static AND envelope, full DREAM (all 5 terms).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid
from src.utils.logs import get_logger

from .graph import (
    NEIGHBOURS_26,
    angle_between_offsets,
    edge_geom,
    ijk_to_world,
    in_bounds,
    is_feasible_edge,
    world_to_ijk,
)

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config & input bundles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannerConfig:
    """Cost weights, vehicle envelope, and tuning knobs."""

    # Cost weights (units in the docstring at file top).
    alpha_T: float = 1.0
    alpha_E: float = 5e-4
    alpha_rho: float = 200.0
    alpha_N: float = 1e-3
    alpha_I: float = 100.0

    # Soft turn penalty in radians^-1.
    turn_penalty: float = 1.0

    # Vehicle envelope (from cost_weights.yaml `evtol`).
    cruise_mps: float = 67.0
    climb_mps: float = 30.0
    descent_mps: float = 25.0
    max_climb_rate_mps: float = 7.0
    max_descent_rate_mps: float = 5.0
    cruise_power_kw: float = 250.0
    hover_power_kw: float = 850.0
    max_bank_deg: float = 25.0

    # Capacity-impact normalization (m).
    d_max_m: float = 1500.0

    # Hard turn-cap (only enforced when `strict_turn=True`).
    strict_turn: bool = False
    max_turn_rad: float = 0.6  # ~34 deg per edge

    # Planning-grid resolution. None ⇒ use the input grid as-is.
    planning_xy_m: Optional[float] = None
    planning_z_m: Optional[float] = None

    # Safety stop: cap on heap pops before declaring infeasible.
    max_expansions: int = 5_000_000

    @classmethod
    def from_cfg(cls, cost_weights_cfg: dict) -> "PlannerConfig":
        d = cost_weights_cfg.get("default", {})
        e = cost_weights_cfg.get("evtol", {})
        return cls(
            alpha_T=float(d.get("alpha_T", cls.alpha_T)),
            alpha_E=float(d.get("alpha_E", cls.alpha_E)),
            alpha_rho=float(d.get("alpha_rho", cls.alpha_rho)),
            alpha_N=float(d.get("alpha_N", cls.alpha_N)),
            alpha_I=float(d.get("alpha_I", cls.alpha_I)),
            cruise_mps=float(e.get("cruise_speed_mps", cls.cruise_mps)),
            climb_mps=float(e.get("climb_speed_mps", cls.climb_mps)),
            descent_mps=float(e.get("descent_speed_mps", cls.descent_mps)),
            max_climb_rate_mps=float(e.get("max_climb_rate_mps", cls.max_climb_rate_mps)),
            max_descent_rate_mps=float(e.get("max_descent_rate_mps", cls.max_descent_rate_mps)),
            cruise_power_kw=float(e.get("cruise_power_kw", cls.cruise_power_kw)),
            hover_power_kw=float(e.get("hover_power_kw", cls.hover_power_kw)),
            max_bank_deg=float(e.get("max_bank_deg", cls.max_bank_deg)),
        )

    def for_baseline(self, baseline: str) -> "PlannerConfig":
        """Return a copy of self with baseline-specific weights zeroed out."""
        if baseline == "B4":
            return self
        kw = dict(self.__dict__)
        if baseline == "B1":
            kw.update(alpha_rho=0.0, alpha_I=0.0, alpha_N=0.0)
        elif baseline == "B2":
            kw.update(alpha_rho=0.0, alpha_I=0.0)
        elif baseline == "B3":
            kw.update(alpha_rho=0.0)
        else:
            raise ValueError(f"unknown baseline {baseline!r}")
        return PlannerConfig(**kw)


@dataclass
class PlannerInputs:
    """All planning inputs in one bundle.

    Arrays must share the same shape as ``grid.shape``. ``envelope``, ``risk``, ``density``,
    ``ofv_start``, ``ofv_end`` may be ``None`` to disable the corresponding feature.
    """

    grid: VoxelGrid
    frame: AirportFrame
    sdf: np.ndarray                       # float32 (nx,ny,nz), positive outside obstacle
    envelope: Optional[np.ndarray] = None # bool (nx,ny,nz), True = clear
    risk: Optional[np.ndarray] = None     # float32 (nx,ny,nz), [0,1]
    density: Optional[np.ndarray] = None  # float32 (nx,ny,nz), people·s per voxel (proxy)
    ofv_start: Optional[np.ndarray] = None  # bool (nx,ny,nz)
    ofv_end: Optional[np.ndarray] = None    # bool (nx,ny,nz)
    source: str = "real"                  # "real" or "synthetic"


# ---------------------------------------------------------------------------
# Corridor output
# ---------------------------------------------------------------------------

@dataclass
class Corridor:
    """A planned corridor and its per-corridor KPI dict."""

    feasible: bool
    baseline: str
    vertiport_pair: Tuple[str, str]
    date: str
    hour: int

    path_ijk: Optional[np.ndarray] = None     # (P, 3) int
    path_enu: Optional[np.ndarray] = None     # (P, 3) float (m)
    path_wgs: Optional[np.ndarray] = None     # (P, 3) float [lon, lat, z_msl_m]

    time_s: float = 0.0
    energy_j: float = 0.0
    risk_integral: float = 0.0
    noise_integral: float = 0.0
    capacity_impact: float = 0.0
    total_cost: float = 0.0
    length_m: float = 0.0

    n_expansions: int = 0
    dynamic_envelope_used: bool = False
    risk_used: bool = False
    notes: list[str] = field(default_factory=list)
    source: str = "real"

    def to_dict(self) -> dict:
        out = {
            "feasible": bool(self.feasible),
            "baseline": self.baseline,
            "vertiport_pair": list(self.vertiport_pair),
            "date": self.date,
            "hour": int(self.hour),
            "time_s": float(self.time_s),
            "energy_j": float(self.energy_j),
            "risk_integral": float(self.risk_integral),
            "noise_integral": float(self.noise_integral),
            "capacity_impact": float(self.capacity_impact),
            "total_cost": float(self.total_cost),
            "length_m": float(self.length_m),
            "n_expansions": int(self.n_expansions),
            "dynamic_envelope_used": bool(self.dynamic_envelope_used),
            "risk_used": bool(self.risk_used),
            "notes": list(self.notes),
            "source": self.source,
        }
        if self.path_enu is not None:
            out["n_waypoints"] = int(self.path_enu.shape[0])
        return out


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

@dataclass
class _BaselineGate:
    """Which masks/cost terms are active for a given baseline."""
    use_a_static: bool
    use_envelope: bool
    alpha_T: float
    alpha_E: float
    alpha_rho: float
    alpha_N: float
    alpha_I: float


def _baseline_gate(cfg: PlannerConfig, baseline: str) -> _BaselineGate:
    if baseline == "B1":
        return _BaselineGate(False, False, 0.0, 0.0, 0.0, 0.0, 0.0)
    if baseline == "B2":
        return _BaselineGate(True, False, cfg.alpha_T, cfg.alpha_E, 0.0, cfg.alpha_N, 0.0)
    if baseline == "B3":
        return _BaselineGate(True, True, cfg.alpha_T, cfg.alpha_E, 0.0, cfg.alpha_N, cfg.alpha_I)
    if baseline == "B4":
        return _BaselineGate(
            True, True, cfg.alpha_T, cfg.alpha_E, cfg.alpha_rho, cfg.alpha_N, cfg.alpha_I
        )
    raise ValueError(f"baseline {baseline!r} is not a planner baseline (B0 handled upstream)")


class Planner:
    """Envelope-constrained A* corridor planner."""

    def __init__(self, inputs: PlannerInputs, cfg: PlannerConfig):
        if inputs.sdf.shape != inputs.grid.shape:
            raise ValueError(
                f"SDF shape {inputs.sdf.shape} != grid shape {inputs.grid.shape}"
            )
        self.inputs = inputs
        self.cfg = cfg
        # Pre-compute per-neighbour edge geometry (length, dz) — saves work in the inner loop.
        self._edge_geom = np.array(
            [edge_geom(d, inputs.grid) for d in NEIGHBOURS_26], dtype=np.float64
        )  # (26, 3): horiz_m, dz_m, length_m

    # --- helpers ----------------------------------------------------------

    def _edge_speed(self, dz_m: float) -> float:
        if dz_m > 0:
            return self.cfg.climb_mps
        if dz_m < 0:
            return self.cfg.descent_mps
        return self.cfg.cruise_mps

    def _edge_power_kw(self, dz_m: float, edge_dt_s: float) -> float:
        if edge_dt_s <= 0:
            return self.cfg.cruise_power_kw
        ratio = abs(dz_m) / edge_dt_s / max(self.cfg.max_climb_rate_mps, 1e-6)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return self.cfg.cruise_power_kw + (self.cfg.hover_power_kw - self.cfg.cruise_power_kw) * ratio

    def _heuristic(self, ijk: tuple[int, int, int], goal_ijk: tuple[int, int, int]) -> float:
        gx, gy, gz = ijk_to_world(self.inputs.grid, goal_ijk)
        x, y, z = ijk_to_world(self.inputs.grid, ijk)
        dist_m = float(np.linalg.norm([gx - x, gy - y, gz - z]))
        return self.cfg.alpha_T * dist_m / max(self.cfg.cruise_mps, 1e-6)

    # --- main entry point -------------------------------------------------

    def plan(
        self,
        start_enu: tuple[float, float, float],
        end_enu: tuple[float, float, float],
        baseline: str,
        *,
        vertiport_pair: tuple[str, str] = ("?", "?"),
        date: str = "",
        hour: int = 0,
    ) -> Corridor:
        """Plan a corridor for the requested baseline.

        Returns a ``Corridor`` with ``feasible=True`` and a populated path on success,
        or ``feasible=False`` (with a populated ``notes`` field) on failure.
        """
        if baseline == "B0":
            return Corridor(
                feasible=False,
                baseline="B0",
                vertiport_pair=vertiport_pair,
                date=date,
                hour=hour,
                notes=["B0: no-eVTOL baseline; planner skipped."],
                source=self.inputs.source,
            )
        if baseline == "B1":
            return self._plan_b1(start_enu, end_enu, vertiport_pair, date, hour)

        gate = _baseline_gate(self.cfg, baseline)
        return self._plan_astar(start_enu, end_enu, baseline, gate, vertiport_pair, date, hour)

    # --- B1: straight ENU line, no obstacle avoidance ---------------------

    def _plan_b1(
        self,
        start_enu: tuple[float, float, float],
        end_enu: tuple[float, float, float],
        vertiport_pair: tuple[str, str],
        date: str,
        hour: int,
    ) -> Corridor:
        n = 200
        xs = np.linspace(start_enu[0], end_enu[0], n)
        ys = np.linspace(start_enu[1], end_enu[1], n)
        # "Drop to surface": fix at the higher of the two endpoint altitudes.
        z_fixed = float(max(start_enu[2], end_enu[2]))
        zs = np.full(n, z_fixed)
        path_enu = np.column_stack([xs, ys, zs]).astype(np.float32)
        seg = np.linalg.norm(np.diff(path_enu, axis=0), axis=1)
        length_m = float(seg.sum())
        time_s = length_m / max(self.cfg.cruise_mps, 1e-6)
        energy_j = self.cfg.cruise_power_kw * 1000.0 * time_s

        # Map to WGS for GeoJSON.
        lon, lat = self.inputs.frame.enu_to_wgs(path_enu[:, 0], path_enu[:, 1])
        path_wgs = np.column_stack([lon, lat, path_enu[:, 2]]).astype(np.float64)

        # Map waypoints back to indices so KPI code can sample sdf/risk along the path.
        ijk = np.array(
            [world_to_ijk(self.inputs.grid, x, y, z) for x, y, z in path_enu], dtype=np.int32
        )

        return Corridor(
            feasible=True,
            baseline="B1",
            vertiport_pair=vertiport_pair,
            date=date,
            hour=hour,
            path_ijk=ijk,
            path_enu=path_enu,
            path_wgs=path_wgs,
            time_s=time_s,
            energy_j=energy_j,
            length_m=length_m,
            risk_integral=0.0,
            noise_integral=0.0,
            capacity_impact=0.0,
            total_cost=self.cfg.alpha_T * time_s + self.cfg.alpha_E * energy_j,
            n_expansions=0,
            dynamic_envelope_used=False,
            risk_used=False,
            notes=["B1: fixed straight-line corridor; envelope and A_static ignored."],
            source=self.inputs.source,
        )

    # --- A* core (B2/B3/B4) ----------------------------------------------

    def _plan_astar(
        self,
        start_enu: tuple[float, float, float],
        end_enu: tuple[float, float, float],
        baseline: str,
        gate: _BaselineGate,
        vertiport_pair: tuple[str, str],
        date: str,
        hour: int,
    ) -> Corridor:
        grid = self.inputs.grid
        shape = grid.shape

        start_ijk = world_to_ijk(grid, *start_enu)
        end_ijk = world_to_ijk(grid, *end_enu)

        # Reject obviously-bad endpoints up-front.
        sdf = self.inputs.sdf
        envelope = self.inputs.envelope
        if gate.use_a_static and sdf[start_ijk] <= 0:
            return self._infeasible(baseline, vertiport_pair, date, hour,
                                    f"start voxel {start_ijk} not in A_static (sdf={sdf[start_ijk]:.1f})")
        if gate.use_a_static and sdf[end_ijk] <= 0:
            return self._infeasible(baseline, vertiport_pair, date, hour,
                                    f"end voxel {end_ijk} not in A_static (sdf={sdf[end_ijk]:.1f})")
        if gate.use_envelope and envelope is not None:
            if not envelope[start_ijk]:
                return self._infeasible(baseline, vertiport_pair, date, hour,
                                        f"start voxel {start_ijk} blocked by envelope")
            if not envelope[end_ijk]:
                return self._infeasible(baseline, vertiport_pair, date, hour,
                                        f"end voxel {end_ijk} blocked by envelope")

        # A* state
        # f-score heap entries: (f, counter, i, j, k, prev_dir_idx).
        risk = self.inputs.risk
        density = self.inputs.density
        d_max = max(self.cfg.d_max_m, 1.0)

        came_from: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
        g_score: dict[tuple[int, int, int], float] = {start_ijk: 0.0}
        f_start = self._heuristic(start_ijk, end_ijk)
        counter = itertools.count()
        heap: list[tuple[float, int, int, int, int, int]] = [
            (f_start, next(counter), start_ijk[0], start_ijk[1], start_ijk[2], -1)
        ]
        closed: set[tuple[int, int, int]] = set()
        n_pops = 0

        # Pre-extract weights for the inner loop.
        a_T = gate.alpha_T
        a_E = gate.alpha_E
        a_rho = gate.alpha_rho
        a_N = gate.alpha_N
        a_I = gate.alpha_I
        turn_pen = self.cfg.turn_penalty
        max_climb = self.cfg.max_climb_rate_mps
        max_descent = self.cfg.max_descent_rate_mps

        while heap:
            f, _, ci, cj, ck, prev_dir = heapq.heappop(heap)
            cur = (ci, cj, ck)
            if cur in closed:
                continue
            closed.add(cur)
            n_pops += 1
            if cur == end_ijk:
                return self._reconstruct(
                    came_from, start_ijk, end_ijk, baseline, gate, vertiport_pair, date, hour, n_pops
                )
            if n_pops > self.cfg.max_expansions:
                return self._infeasible(baseline, vertiport_pair, date, hour,
                                        f"max_expansions {self.cfg.max_expansions} hit")

            g_cur = g_score.get(cur, float("inf"))
            for d_idx, d in enumerate(NEIGHBOURS_26):
                ni, nj, nk = ci + int(d[0]), cj + int(d[1]), ck + int(d[2])
                if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                    continue
                horiz_m, dz_m, length_m = (
                    float(self._edge_geom[d_idx, 0]),
                    float(self._edge_geom[d_idx, 1]),
                    float(self._edge_geom[d_idx, 2]),
                )
                speed = self._edge_speed(dz_m)
                edge_dt = length_m / max(speed, 1e-6)
                if not is_feasible_edge(
                    (ni, nj, nk),
                    shape=shape,
                    sdf=sdf,
                    envelope=envelope,
                    use_a_static=gate.use_a_static,
                    use_envelope=gate.use_envelope,
                    dz_m=dz_m,
                    edge_dt_s=edge_dt,
                    max_climb_rate_mps=max_climb,
                    max_descent_rate_mps=max_descent,
                ):
                    continue
                # Cost terms.
                T = edge_dt
                P_kw = self._edge_power_kw(dz_m, edge_dt)
                E = P_kw * 1000.0 * edge_dt
                rho = float(risk[ni, nj, nk]) if (a_rho > 0 and risk is not None) else 0.0
                Nv = float(density[ni, nj, nk]) * length_m if (a_N > 0 and density is not None) else 0.0
                if a_I > 0:
                    I = float(np.clip(1.0 - sdf[ni, nj, nk] / d_max, 0.0, 1.0))
                else:
                    I = 0.0
                turn = 0.0
                if prev_dir >= 0:
                    turn = angle_between_offsets(NEIGHBOURS_26[prev_dir], d, self.inputs.grid)
                    if self.cfg.strict_turn and turn > self.cfg.max_turn_rad:
                        continue
                edge_cost = (
                    a_T * T + a_E * E + a_rho * rho + a_N * Nv + a_I * I + turn_pen * turn
                )
                tentative_g = g_cur + edge_cost
                nbr = (ni, nj, nk)
                if tentative_g < g_score.get(nbr, float("inf")):
                    g_score[nbr] = tentative_g
                    came_from[nbr] = (ci, cj, ck, d_idx)
                    f_score = tentative_g + self._heuristic(nbr, end_ijk)
                    heapq.heappush(heap, (f_score, next(counter), ni, nj, nk, d_idx))

        return self._infeasible(baseline, vertiport_pair, date, hour, "search exhausted; no path found",
                                n_expansions=n_pops)

    # --- reconstruction + bookkeeping ------------------------------------

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int, int], tuple[int, int, int, int]],
        start_ijk: tuple[int, int, int],
        end_ijk: tuple[int, int, int],
        baseline: str,
        gate: _BaselineGate,
        vertiport_pair: tuple[str, str],
        date: str,
        hour: int,
        n_pops: int,
    ) -> Corridor:
        # Walk came_from back to start.
        rev_ijk = [end_ijk]
        cur = end_ijk
        while cur != start_ijk:
            ci, cj, ck, _ = came_from[cur]
            cur = (ci, cj, ck)
            rev_ijk.append(cur)
        rev_ijk.reverse()
        ijk_arr = np.asarray(rev_ijk, dtype=np.int32)

        # Compute per-edge KPI sums in one pass so we can pre-fill the Corridor.
        enu = np.array(
            [ijk_to_world(self.inputs.grid, tuple(int(x) for x in row)) for row in ijk_arr],
            dtype=np.float32,
        )
        seg = np.diff(enu, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        length_m = float(seg_len.sum())

        # Per-edge time using the actual offset's vertical component.
        t_total = 0.0
        e_total = 0.0
        rho_total = 0.0
        n_total = 0.0
        i_total = 0.0
        cost_total = 0.0
        prev_d = None
        risk = self.inputs.risk
        density = self.inputs.density
        sdf = self.inputs.sdf
        d_max = max(self.cfg.d_max_m, 1.0)
        for k in range(len(seg)):
            dx_m, dy_m, dz_m = float(seg[k, 0]), float(seg[k, 1]), float(seg[k, 2])
            length = float(seg_len[k])
            speed = self._edge_speed(dz_m)
            dt = length / max(speed, 1e-6)
            P_kw = self._edge_power_kw(dz_m, dt)
            energy = P_kw * 1000.0 * dt
            ni, nj, nk = (int(ijk_arr[k + 1, 0]), int(ijk_arr[k + 1, 1]), int(ijk_arr[k + 1, 2]))
            rho_v = float(risk[ni, nj, nk]) if (gate.alpha_rho > 0 and risk is not None) else 0.0
            n_v = (
                float(density[ni, nj, nk]) * length
                if (gate.alpha_N > 0 and density is not None)
                else 0.0
            )
            i_v = float(np.clip(1.0 - sdf[ni, nj, nk] / d_max, 0.0, 1.0)) if gate.alpha_I > 0 else 0.0
            turn = 0.0
            if prev_d is not None:
                a_off = np.array([prev_d[0], prev_d[1], prev_d[2]], dtype=np.int8)
                b_off = np.array(
                    [
                        int(ijk_arr[k + 1, 0] - ijk_arr[k, 0]),
                        int(ijk_arr[k + 1, 1] - ijk_arr[k, 1]),
                        int(ijk_arr[k + 1, 2] - ijk_arr[k, 2]),
                    ],
                    dtype=np.int8,
                )
                turn = angle_between_offsets(a_off, b_off, self.inputs.grid)
            prev_d = (
                int(ijk_arr[k + 1, 0] - ijk_arr[k, 0]),
                int(ijk_arr[k + 1, 1] - ijk_arr[k, 1]),
                int(ijk_arr[k + 1, 2] - ijk_arr[k, 2]),
            )
            t_total += dt
            e_total += energy
            rho_total += rho_v * dt
            n_total += n_v
            i_total += i_v
            cost_total += (
                gate.alpha_T * dt
                + gate.alpha_E * energy
                + gate.alpha_rho * rho_v
                + gate.alpha_N * n_v
                + gate.alpha_I * i_v
                + self.cfg.turn_penalty * turn
            )

        # WGS for GeoJSON.
        lon, lat = self.inputs.frame.enu_to_wgs(enu[:, 0], enu[:, 1])
        wgs = np.column_stack([lon, lat, enu[:, 2]]).astype(np.float64)

        return Corridor(
            feasible=True,
            baseline=baseline,
            vertiport_pair=vertiport_pair,
            date=date,
            hour=hour,
            path_ijk=ijk_arr,
            path_enu=enu,
            path_wgs=wgs,
            time_s=float(t_total),
            energy_j=float(e_total),
            risk_integral=float(rho_total),
            noise_integral=float(n_total),
            capacity_impact=float(i_total),
            total_cost=float(cost_total),
            length_m=length_m,
            n_expansions=int(n_pops),
            dynamic_envelope_used=bool(gate.use_envelope and self.inputs.envelope is not None),
            risk_used=bool(gate.alpha_rho > 0 and self.inputs.risk is not None),
            notes=[f"A* succeeded baseline={baseline} pops={n_pops}"],
            source=self.inputs.source,
        )

    def _infeasible(
        self,
        baseline: str,
        vertiport_pair: tuple[str, str],
        date: str,
        hour: int,
        note: str,
        *,
        n_expansions: int = 0,
    ) -> Corridor:
        return Corridor(
            feasible=False,
            baseline=baseline,
            vertiport_pair=vertiport_pair,
            date=date,
            hour=hour,
            n_expansions=n_expansions,
            notes=[note],
            source=self.inputs.source,
        )
