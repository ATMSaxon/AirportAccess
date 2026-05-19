"""Local geometry helper used by counterfactual labelling and features.

This file deliberately mirrors the interface we expect `src.geometry.query` to
expose (M2 owner: `geometry-engineer`). When that module lands we can swap the
calls; for now this module computes OLS prism distances, approach/departure
intersection tests, missed-approach overlap, and a coarse signed-distance proxy
directly from the airport YAML + the Annex 14 generic Code-4 precision yaml.

All coordinates are in **local ENU around the airport ARP** (metres). Altitudes
are MSL metres; AGL conversions use the field elevation in the airport config.

Conventions for the runway entry in `configs/airports/<ICAO>.yaml`:
* `thr_*` is the runway **threshold** (where arrivals touch down).
* `end_*` is the **departure end** (where takeoff climb starts).
* `bearing_deg` is the takeoff direction (i.e. the bearing of the vector
  `thr → end`).
* Each runway-end is therefore one direction-of-use; opposite uses appear as
  separate entries (e.g. `06L` and `24R`).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math
import numpy as np

from ..utils import config as cfg_io
from ..utils.crs import AirportFrame


NM_M = 1852.0          # 1 nautical mile in metres
FT_M = 0.3048          # 1 foot in metres


# ---------------------------------------------------------------------------
# Runway-centric primitive
# ---------------------------------------------------------------------------
@dataclass
class Runway:
    """A single direction-of-use runway entry, projected to ENU."""
    rwy_id: str
    thr_x: float          # threshold (touchdown end) ENU east
    thr_y: float          # threshold ENU north
    thr_z: float          # threshold MSL elevation (m)
    end_x: float          # departure end ENU east
    end_y: float          # departure end ENU north
    end_z: float
    length_m: float
    width_m: float
    bearing_deg: float
    precision: bool

    @property
    def axis(self) -> np.ndarray:
        """Unit vector along runway centreline, threshold → departure-end."""
        v = np.array([self.end_x - self.thr_x, self.end_y - self.thr_y])
        n = np.linalg.norm(v)
        return v / (n + 1e-9)

    def project(self, x: float, y: float) -> tuple[float, float]:
        """Return (along_m, cross_m) coordinates relative to threshold.

        along_m increases toward the departure end; cross_m is signed
        (positive on the left looking from threshold toward the end).
        """
        ax = self.axis
        dx, dy = x - self.thr_x, y - self.thr_y
        along = dx * ax[0] + dy * ax[1]
        cross = -dx * ax[1] + dy * ax[0]
        return float(along), float(cross)


@dataclass
class AirportGeom:
    """Bundles per-airport runway geometry + Annex-14 surface parameters.

    Use :py:meth:`from_icao` to construct.

    When the geometry-engineer's `data/processed/<ICAO>/ols.gpkg` artefact is
    present, the SDF + distance methods transparently delegate to
    `src.geometry.query.PrismIndex` (which covers the full Annex-14 surface
    union including transitional / inner-horizontal / conical / OFZ / RESA).
    Otherwise they fall back to the coarse approach-only proxy implemented in
    this file, which keeps tests on a fresh repo green.
    """
    icao: str
    field_elev_m: float
    runways: list[Runway]
    annex14: dict
    vertiports: dict[str, dict]
    frame: AirportFrame
    extract_box_m: dict = field(default_factory=dict)
    _prism_index: object | None = field(default=None, repr=False)
    _prism_load_tried: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # PrismIndex integration (geometry-engineer's production API).
    # PrismIndex z-convention is **AGL above ARP**; our `z_msl_m` is MSL,
    # so we subtract `field_elev_m` before delegating.
    # ------------------------------------------------------------------
    def _to_agl(self, z_msl_m):
        # Accept scalar or ndarray transparently.
        return z_msl_m - self.field_elev_m

    @property
    def prism_index(self):
        """Lazy-loaded `PrismIndex.from_airport(icao)`. Returns None on failure
        (e.g. `ols.gpkg` missing on a fresh repo) so callers can fall back."""
        if self._prism_load_tried:
            return self._prism_index
        self._prism_load_tried = True
        try:
            from ..geometry.query import PrismIndex
            self._prism_index = PrismIndex.from_airport(self.icao)
        except Exception:                             # noqa: BLE001
            self._prism_index = None
        return self._prism_index

    @classmethod
    def from_icao(cls, icao: str, annex14_profile: str = "code4_precision") -> "AirportGeom":
        ac = cfg_io.load_airport(icao)
        ax = cfg_io.load_annex14(annex14_profile)
        frame = AirportFrame.from_cfg(ac)
        field_elev = float(ac["arp"]["elev_m"])
        runways: list[Runway] = []
        for r in ac["runways"]:
            tx, ty = frame.wgs_to_enu(np.array([r["thr_lon"]]), np.array([r["thr_lat"]]))
            ex, ey = frame.wgs_to_enu(np.array([r["end_lon"]]), np.array([r["end_lat"]]))
            runways.append(
                Runway(
                    rwy_id=str(r["id"]),
                    thr_x=float(tx[0]), thr_y=float(ty[0]), thr_z=field_elev,
                    end_x=float(ex[0]), end_y=float(ey[0]), end_z=field_elev,
                    length_m=float(r["length_ft"]) * FT_M,
                    width_m=float(r["width_ft"]) * FT_M,
                    bearing_deg=float(r["bearing_deg"]),
                    precision=bool(r.get("precision", True)),
                )
            )
        # Project vertiports too (preserve lat/lon, attach ENU).
        verts = {}
        for vid, v in ac.get("vertiports", {}).items():
            vx, vy = frame.wgs_to_enu(np.array([v["lon"]]), np.array([v["lat"]]))
            verts[vid] = {
                "name": v.get("name", vid),
                "lon": float(v["lon"]),
                "lat": float(v["lat"]),
                "elev_m": float(v.get("elev_ft", 0.0)) * FT_M,
                "x_m": float(vx[0]),
                "y_m": float(vy[0]),
            }
        return cls(
            icao=icao,
            field_elev_m=field_elev,
            runways=runways,
            annex14=ax,
            vertiports=verts,
            frame=frame,
            extract_box_m=ac.get("extract_box_m", {}),
        )

    # ------------------------------------------------------------------
    # Runway-axis helpers
    # ------------------------------------------------------------------
    def runway_by_id(self, rid: str) -> Runway:
        for r in self.runways:
            if r.rwy_id == rid:
                return r
        raise KeyError(f"runway {rid} not in airport {self.icao}")

    # ------------------------------------------------------------------
    # Approach surface  (arrival side – behind threshold)
    # ------------------------------------------------------------------
    def _approach_params(self) -> dict:
        return self.annex14["approach_surface"]

    def in_approach_prism(self, x: float, y: float, z_msl_m: float,
                          rwy: Runway) -> bool:
        """True iff (x,y,z_msl) lies inside `rwy`'s arrival approach prism."""
        ap = self._approach_params()
        along, cross = rwy.project(x, y)
        # Approach extends BEHIND threshold → along < -inner_edge_offset_m,
        # measured from threshold (negative = upwind of touchdown).
        s = -along - float(ap["inner_edge_offset_m"])
        if s < 0:
            return False
        total = float(ap["total_length_m"])
        if s > total:
            return False
        # Width: half-width grows from inner_edge_width/2 by divergence per side.
        half_w = 0.5 * float(ap["inner_edge_width_m"]) + s * float(ap["divergence_each_side"])
        if abs(cross) > half_w:
            return False
        # Vertical: two-section sloped roof + horizontal section.
        l1 = float(ap["length_first_section_m"]); s1 = float(ap["slope_first_section"])
        l2 = float(ap["length_second_section_m"]); s2 = float(ap["slope_second_section"])
        z_top = rwy.thr_z
        if s <= l1:
            z_top += s * s1
        elif s <= l1 + l2:
            z_top += l1 * s1 + (s - l1) * s2
        else:
            z_top += l1 * s1 + l2 * s2
        return rwy.thr_z <= z_msl_m <= z_top

    # ------------------------------------------------------------------
    # Takeoff climb / departure prism (departure side)
    # ------------------------------------------------------------------
    def _takeoff_params(self) -> dict:
        return self.annex14["takeoff_climb_surface"]

    def in_departure_prism(self, x: float, y: float, z_msl_m: float,
                           rwy: Runway) -> bool:
        tp = self._takeoff_params()
        along, cross = rwy.project(x, y)
        s = along - rwy.length_m - float(tp["inner_edge_offset_m"])
        if s < 0:
            return False
        total = float(tp["total_length_m"])
        if s > total:
            return False
        half_w = 0.5 * float(tp["inner_edge_width_m"]) + s * float(tp["divergence_each_side"])
        half_w = min(half_w, 0.5 * float(tp["final_width_m"]))
        if abs(cross) > half_w:
            return False
        z_top = rwy.end_z + s * float(tp["slope"])
        return rwy.end_z <= z_msl_m <= z_top

    # ------------------------------------------------------------------
    # Missed-approach surface (climb past the runway after balked landing).
    # Standard PANS-OPS practice: climb gradient ≈ 2.5 % from runway end.
    # We model it as a narrow rectangle along the runway axis past the
    # departure end, 600 m wide, 8 km long, slope 0.025.
    # ------------------------------------------------------------------
    def in_missed_approach(self, x: float, y: float, z_msl_m: float,
                           rwy: Runway, slope: float = 0.025,
                           length_m: float = 8000.0, half_w: float = 300.0) -> bool:
        along, cross = rwy.project(x, y)
        s = along - rwy.length_m
        if s < 0 or s > length_m:
            return False
        if abs(cross) > half_w:
            return False
        z_top = rwy.end_z + s * slope
        return rwy.end_z <= z_msl_m <= z_top

    # ------------------------------------------------------------------
    # Generic OLS proxy SDF (positive outside surfaces, negative inside).
    # Implemented as the signed minimum across approach + departure + missed
    # for the active runways, plus a runway-strip floor. Coarse but
    # monotonic-in-distance, which is what the risk learner needs as a
    # feature.
    # ------------------------------------------------------------------
    def sdf(self, x: float, y: float, z_msl_m: float,
            active_arrivals: Sequence[str] = (),
            active_departures: Sequence[str] = ()) -> float:
        """Signed distance proxy.

        Positive = outside any *active* OLS surface (safe-side).
        Negative = inside an active surface (protected airspace intrusion).
        Magnitude = approximate distance to nearest surface (m).
        """
        # Production path: delegate to geometry-engineer's PrismIndex when the
        # ols.gpkg artefact exists. PrismIndex's `active_*=None` matches "any"
        # while `active_*=[]` matches "none", so we always pass an explicit list.
        idx = self.prism_index
        if idx is not None:
            try:
                z_agl = float(z_msl_m) - self.field_elev_m
                return float(idx.sdf_at(
                    float(x), float(y), float(z_agl),
                    active_arrivals=list(active_arrivals),
                    active_departures=list(active_departures),
                ))
            except Exception:                          # noqa: BLE001
                pass                                   # fall through to coarse

        d_best = math.inf
        sign_inside = False
        for rid in active_arrivals:
            try:
                r = self.runway_by_id(rid)
            except KeyError:
                continue
            inside = self.in_approach_prism(x, y, z_msl_m, r) or \
                     self.in_missed_approach(x, y, z_msl_m, r)
            d = self._approach_distance_proxy(x, y, z_msl_m, r)
            if inside:
                sign_inside = True
                d_best = min(d_best, d)
            else:
                d_best = min(d_best, d)
        for rid in active_departures:
            try:
                r = self.runway_by_id(rid)
            except KeyError:
                continue
            inside = self.in_departure_prism(x, y, z_msl_m, r)
            d = self._departure_distance_proxy(x, y, z_msl_m, r)
            if inside:
                sign_inside = True
                d_best = min(d_best, d)
            else:
                d_best = min(d_best, d)
        if not math.isfinite(d_best):
            # No active surfaces — fall back to distance from any runway centreline.
            d_best = min(self._axis_distance_3d(x, y, z_msl_m, r) for r in self.runways)
        return -d_best if sign_inside else d_best

    def _axis_distance_3d(self, x, y, z, r: Runway) -> float:
        ax, ay = r.thr_x, r.thr_y
        bx, by = r.end_x, r.end_y
        # closest point on segment to (x,y) in 2-D
        t = ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / max((bx-ax)**2 + (by-ay)**2, 1e-9)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t*(bx-ax), ay + t*(by-ay)
        return math.sqrt((x-cx)**2 + (y-cy)**2 + (z - r.thr_z)**2)

    def _approach_distance_proxy(self, x, y, z, r: Runway) -> float:
        """Distance from point to the approach prism mouth (cheap proxy)."""
        ap = self._approach_params()
        along, cross = r.project(x, y)
        s = -along - float(ap["inner_edge_offset_m"])
        # Lateral distance to corridor side; if behind the prism, distance to mouth.
        half_w = 0.5 * float(ap["inner_edge_width_m"]) + max(s, 0.0) * float(ap["divergence_each_side"])
        d_lat = max(abs(cross) - half_w, 0.0)
        d_along = 0.0 if s >= 0 else -s
        # Roof distance
        l1 = float(ap["length_first_section_m"]); s1 = float(ap["slope_first_section"])
        l2 = float(ap["length_second_section_m"]); s2 = float(ap["slope_second_section"])
        z_top = r.thr_z + (min(s, l1) * s1
                           + max(min(s - l1, l2), 0.0) * s2)
        d_v = max(z - z_top, 0.0) if s >= 0 else max(z - r.thr_z, 0.0)
        return math.sqrt(d_lat*d_lat + d_along*d_along + d_v*d_v)

    def _departure_distance_proxy(self, x, y, z, r: Runway) -> float:
        tp = self._takeoff_params()
        along, cross = r.project(x, y)
        s = along - r.length_m - float(tp["inner_edge_offset_m"])
        half_w = 0.5 * float(tp["inner_edge_width_m"]) + max(s, 0.0) * float(tp["divergence_each_side"])
        half_w = min(half_w, 0.5 * float(tp["final_width_m"]))
        d_lat = max(abs(cross) - half_w, 0.0)
        d_along = 0.0 if s >= 0 else -s
        z_top = r.end_z + max(s, 0.0) * float(tp["slope"])
        d_v = max(z - z_top, 0.0) if s >= 0 else max(z - r.end_z, 0.0)
        return math.sqrt(d_lat*d_lat + d_along*d_along + d_v*d_v)

    # ------------------------------------------------------------------
    # Segment ↔ runway-axis crossing test (segment p0→p1 in ENU+z).
    # ------------------------------------------------------------------
    def segment_crosses_runway_axis(self, p0: np.ndarray, p1: np.ndarray,
                                    active_runways: Sequence[str],
                                    z_max_agl_m: float = 2000.0 * FT_M) -> bool:
        """True iff p0→p1 crosses any active runway centreline below `z_max_agl_m`.

        "Crosses" = projected positions onto runway axis change sign in the
        cross-axis direction inside the runway's length, AND the point of
        crossing is below `z_max_agl_m` above the field elevation.
        """
        for rid in active_runways:
            try:
                r = self.runway_by_id(rid)
            except KeyError:
                continue
            a0, c0 = r.project(p0[0], p0[1])
            a1, c1 = r.project(p1[0], p1[1])
            # Cross-product sign change with both points inside the strip envelope
            if c0 == c1:
                continue
            t = c0 / (c0 - c1)
            if not (0.0 <= t <= 1.0):
                continue
            a_cross = a0 + t * (a1 - a0)
            if not (-200.0 <= a_cross <= r.length_m + 200.0):
                continue
            z_cross = float(p0[2] + t * (p1[2] - p0[2]))
            if (z_cross - self.field_elev_m) <= z_max_agl_m:
                return True
        return False

    def segment_intersects_prism(self, p0: np.ndarray, p1: np.ndarray,
                                 active_arrivals: Sequence[str],
                                 active_departures: Sequence[str],
                                 n_samples: int = 16) -> tuple[bool, bool, bool]:
        """Coarse intersection test by sampling points along the segment.

        Returns (approach_hit, departure_hit, missed_approach_hit).
        """
        ts = np.linspace(0.0, 1.0, n_samples)
        approach_hit = departure_hit = missed_hit = False
        for t in ts:
            x = float(p0[0] + t * (p1[0] - p0[0]))
            y = float(p0[1] + t * (p1[1] - p0[1]))
            z = float(p0[2] + t * (p1[2] - p0[2]))
            for rid in active_arrivals:
                try:
                    r = self.runway_by_id(rid)
                except KeyError:
                    continue
                if not approach_hit and self.in_approach_prism(x, y, z, r):
                    approach_hit = True
                if not missed_hit and self.in_missed_approach(x, y, z, r):
                    missed_hit = True
            for rid in active_departures:
                try:
                    r = self.runway_by_id(rid)
                except KeyError:
                    continue
                if not departure_hit and self.in_departure_prism(x, y, z, r):
                    departure_hit = True
            if approach_hit and departure_hit and missed_hit:
                break
        return approach_hit, departure_hit, missed_hit

    # ------------------------------------------------------------------
    # Per-feature distances used by features.py
    # ------------------------------------------------------------------
    def distance_to_nearest_runway(self, x: float, y: float, z: float) -> float:
        return min(self._axis_distance_3d(x, y, z, r) for r in self.runways)

    def distance_to_active_approach(self, x: float, y: float, z: float,
                                    active_arrivals: Sequence[str]) -> float:
        """Unsigned distance (m) to nearest active approach prism.

        Delegates to `PrismIndex.distance_to_active_approach` when available;
        that returns signed (negative inside), so we abs() before returning to
        keep the column non-negative per `INTERFACES.md` (C2.d_approach_m).
        """
        if not active_arrivals:
            return math.inf
        idx = self.prism_index
        if idx is not None:
            try:
                z_agl = float(z) - self.field_elev_m
                d = float(idx.distance_to_active_approach(
                    float(x), float(y), float(z_agl),
                    active_arrivals=list(active_arrivals)))
                return abs(d)
            except Exception:                          # noqa: BLE001
                pass
        return min(self._approach_distance_proxy(x, y, z, self.runway_by_id(rid))
                   for rid in active_arrivals if self._has_runway(rid))

    def distance_to_active_departure(self, x: float, y: float, z: float,
                                     active_departures: Sequence[str]) -> float:
        """Unsigned distance (m) to nearest active departure prism."""
        if not active_departures:
            return math.inf
        idx = self.prism_index
        if idx is not None:
            try:
                z_agl = float(z) - self.field_elev_m
                d = float(idx.distance_to_active_departure(
                    float(x), float(y), float(z_agl),
                    active_departures=list(active_departures)))
                return abs(d)
            except Exception:                          # noqa: BLE001
                pass
        return min(self._departure_distance_proxy(x, y, z, self.runway_by_id(rid))
                   for rid in active_departures if self._has_runway(rid))

    def _has_runway(self, rid: str) -> bool:
        return any(r.rwy_id == rid for r in self.runways)

    # ------------------------------------------------------------------
    # Random sampling helpers
    # ------------------------------------------------------------------
    def sample_vertiport_ofv(self, vid: str, rng: np.random.Generator,
                             radius_m: float | None = None,
                             height_m: float | None = None) -> tuple[float, float, float]:
        """Uniform sample from a small ENU cylinder above the vertiport.

        Uses Annex-14 `vertiport_ofv` defaults when explicit radius/height not given.
        """
        if vid not in self.vertiports:
            raise KeyError(f"vertiport {vid} not configured for {self.icao}")
        v = self.vertiports[vid]
        ofv = self.annex14["vertiport_ofv"]
        rmax = float(radius_m) if radius_m is not None else float(ofv["ofv_top_radius_m"])
        hmax = float(height_m) if height_m is not None else float(ofv["ofv_height_m"])
        # Uniform in disk × uniform in height
        u, w = rng.uniform(), rng.uniform()
        rr = rmax * math.sqrt(u)
        theta = 2.0 * math.pi * w
        z = v["elev_m"] + rng.uniform() * hmax
        return v["x_m"] + rr * math.cos(theta), v["y_m"] + rr * math.sin(theta), z

    def sample_static_volume(self, rng: np.random.Generator,
                             z_min_m: float = 200.0,
                             z_max_m: float | None = None) -> tuple[float, float, float]:
        """Uniform sample inside the airport extraction box at altitude in
        [field_elev + z_min_m, z_max_m or extract_box.z_max]."""
        box = self.extract_box_m
        hx = float(box.get("half_x", 30000.0))
        hy = float(box.get("half_y", 30000.0))
        z_lo = self.field_elev_m + float(z_min_m)
        z_hi = float(z_max_m if z_max_m is not None else box.get("z_max", 3500.0))
        return (rng.uniform(-hx, hx),
                rng.uniform(-hy, hy),
                rng.uniform(z_lo, z_hi))


# ---------------------------------------------------------------------------
# Light traffic-density fallback used until `src.traffic.density` lands.
# ---------------------------------------------------------------------------
def adsb_density_box(adsb_df, mid_x: float, mid_y: float, mid_z: float,
                     mid_t, *, half_xy_m: float = 1500.0,
                     half_z_m: float = 150.0, half_t_s: float = 300.0) -> float:
    """Counts of ADS-B observations in a (xy, z, t) box around the midpoint,
    normalised by box volume × time-window.

    Operates on the D5 schema (columns `x_m, y_m, z_msl_m, time_utc`).
    """
    import pandas as pd  # local import; keep top-level deps light.

    if adsb_df is None or len(adsb_df) == 0:
        return 0.0
    t_lo = pd.Timestamp(mid_t) - pd.Timedelta(seconds=half_t_s)
    t_hi = pd.Timestamp(mid_t) + pd.Timedelta(seconds=half_t_s)
    sub = adsb_df[(adsb_df["time_utc"] >= t_lo) & (adsb_df["time_utc"] <= t_hi)]
    if len(sub) == 0:
        return 0.0
    m = (
        (np.abs(sub["x_m"].to_numpy() - mid_x) < half_xy_m) &
        (np.abs(sub["y_m"].to_numpy() - mid_y) < half_xy_m) &
        (np.abs(sub["z_msl_m"].to_numpy() - mid_z) < half_z_m)
    )
    n = int(m.sum())
    vol = (2 * half_xy_m) ** 2 * (2 * half_z_m)         # m^3
    dt = 2 * half_t_s                                   # s
    return n / (vol * dt + 1e-12)                       # observations / (m^3 s)
