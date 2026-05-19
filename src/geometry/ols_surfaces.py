"""Annex 14 OLS prism builder.

Each protection prism is parameterised as:
  - a 2-D footprint polygon (Shapely) in local ENU metres (CCW exterior),
  - an upper-height field z_top(x, y) — either
        z_form='affine'  →  z_top = a*x + b*y + c
        z_form='radial'  →  z_top = z_top_c + slope*(sqrt((x-cx)^2+(y-cy)^2) - r0)
  - a lower-height z_low (constant; defaults to 0 = ARP ground level AGL).

A point (x,y,z) is "inside" the prism iff (x,y) ∈ footprint AND z_low ≤ z ≤ z_top(x,y).

The PROTECTED VOLUME for the airport is the *union* of all prisms — the OLS-defined
volume below each OLS surface, within its 2-D footprint, above the ARP ground level.
The SDF is negative inside the union and positive outside.

Per CLAUDE.md: this is a *placeholder open parameterisation*. Per-airport calibration
should override `configs/annex14/code4_precision.yaml`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List
import numpy as np
import shapely
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from ..utils.crs import AirportFrame
from ..utils.logs import get_logger

logger = get_logger(__name__)

# Canonical surface names.
APPROACH = "approach"
TAKEOFF = "takeoff_climb"
TRANSITIONAL = "transitional"
INNER_HORIZONTAL = "inner_horizontal"
CONICAL = "conical"
RUNWAY_STRIP = "runway_strip"
RESA = "resa"
OFZ_INNER_APPROACH = "ofz_inner_approach"
OFZ_INNER_TRANSITIONAL = "ofz_inner_transitional"

ALL_SURFACES = (APPROACH, TAKEOFF, TRANSITIONAL, INNER_HORIZONTAL, CONICAL,
                RUNWAY_STRIP, RESA, OFZ_INNER_APPROACH, OFZ_INNER_TRANSITIONAL)


@dataclass
class Prism:
    """Single OLS protection prism (2-D footprint + height field)."""
    name: str
    surface: str
    runway_id: str
    end_id: str        # 'thr' | 'end' | 'arp'
    footprint: Polygon
    z_form: str        # 'affine' | 'radial'
    z_top_a: float = 0.0
    z_top_b: float = 0.0
    z_top_c: float = 0.0
    z_top_slope: float = 0.0
    z_top_r0: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    z_low: float = 0.0

    def evaluate_z_top(self, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        if self.z_form == "affine":
            return self.z_top_a * xx + self.z_top_b * yy + self.z_top_c
        if self.z_form == "radial":
            r = np.sqrt((xx - self.cx) ** 2 + (yy - self.cy) ** 2)
            return self.z_top_c + self.z_top_slope * (r - self.z_top_r0)
        raise ValueError(f"unknown z_form {self.z_form}")

    # ---------- GeoDataFrame round-trip ----------
    def to_row(self) -> dict:
        return {
            "name": self.name,
            "surface": self.surface,
            "runway_id": self.runway_id,
            "end_id": self.end_id,
            "z_form": self.z_form,
            "z_top_a": float(self.z_top_a),
            "z_top_b": float(self.z_top_b),
            "z_top_c": float(self.z_top_c),
            "z_top_slope": float(self.z_top_slope),
            "z_top_r0": float(self.z_top_r0),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "z_low": float(self.z_low),
            "geometry": self.footprint,
        }

    @classmethod
    def from_row(cls, row) -> "Prism":
        return cls(
            name=row["name"], surface=row["surface"],
            runway_id=row["runway_id"], end_id=row["end_id"],
            footprint=row["geometry"], z_form=row["z_form"],
            z_top_a=row["z_top_a"], z_top_b=row["z_top_b"], z_top_c=row["z_top_c"],
            z_top_slope=row["z_top_slope"], z_top_r0=row["z_top_r0"],
            cx=row["cx"], cy=row["cy"], z_low=row.get("z_low", 0.0),
        )


# ============================================================================
# Helpers
# ============================================================================

def _rwy_axes(rwy: dict, frame: AirportFrame):
    """Threshold, stop-end, axis unit (thr→end), perpendicular unit (left-hand), length (m)."""
    tx, ty = frame.wgs_to_enu(np.array([rwy["thr_lon"]]), np.array([rwy["thr_lat"]]))
    ex, ey = frame.wgs_to_enu(np.array([rwy["end_lon"]]), np.array([rwy["end_lat"]]))
    thr = np.array([float(tx[0]), float(ty[0])])
    end = np.array([float(ex[0]), float(ey[0])])
    vec = end - thr
    L = float(np.linalg.norm(vec))
    if L < 1.0:
        raise ValueError(f"Degenerate runway {rwy.get('id')} length {L} m")
    u = vec / L
    perp = np.array([-u[1], u[0]])   # 90° CCW of u
    return thr, end, u, perp, L


def _trapezoid(origin: np.ndarray, axis: np.ndarray, perp: np.ndarray,
               s0: float, s1: float, w0: float, w1: float) -> Polygon:
    """Trapezoidal footprint from axial s0→s1 along `axis`, half-widths w0→w1 along ±`perp`.

    Returned polygon has CCW exterior (outward normals).
    """
    p_inner_r = origin + s0 * axis - w0 * perp
    p_outer_r = origin + s1 * axis - w1 * perp
    p_outer_l = origin + s1 * axis + w1 * perp
    p_inner_l = origin + s0 * axis + w0 * perp
    coords = [tuple(p_inner_r), tuple(p_outer_r), tuple(p_outer_l), tuple(p_inner_l)]
    poly = Polygon(coords)
    return orient(poly, sign=1.0)


def _rect(origin: np.ndarray, axis: np.ndarray, perp: np.ndarray,
          s0: float, s1: float, half_width: float) -> Polygon:
    return _trapezoid(origin, axis, perp, s0, s1, half_width, half_width)


def _affine_along_axis(origin: np.ndarray, axis: np.ndarray, slope: float, z0: float):
    """Return (a, b, c) with z(x,y) = slope * (axis · (P - origin)) + z0 = a x + b y + c."""
    a = float(slope * axis[0])
    b = float(slope * axis[1])
    c = float(z0 - slope * (axis[0] * origin[0] + axis[1] * origin[1]))
    return a, b, c


def _affine_along_perp(origin: np.ndarray, perp: np.ndarray, slope: float,
                       offset_along_perp: float, z0: float = 0.0):
    """z(P) = slope * (perp · (P - origin) - offset_along_perp) + z0 (lateral rise from a strip edge)."""
    a = float(slope * perp[0])
    b = float(slope * perp[1])
    c = float(z0 - slope * (perp[0] * origin[0] + perp[1] * origin[1]) - slope * offset_along_perp)
    return a, b, c


def _disc(centre: np.ndarray, radius: float, n: int = 192) -> Polygon:
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([centre[0] + radius * np.cos(theta),
                           centre[1] + radius * np.sin(theta)])
    poly = Polygon(pts)
    return orient(poly, sign=1.0)


def _annulus(centre: np.ndarray, r_in: float, r_out: float, n: int = 192) -> Polygon:
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([centre[0] + r_out * np.cos(theta),
                             centre[1] + r_out * np.sin(theta)])
    inner = np.column_stack([centre[0] + r_in * np.cos(theta),
                             centre[1] + r_in * np.sin(theta)])
    poly = Polygon(outer, [list(map(tuple, inner))])
    return orient(poly, sign=1.0)


# ============================================================================
# Per-surface builders
# ============================================================================

def build_approach(rwy: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    """Approach surface: starts past the threshold (outside the runway),
    broken into 3 sub-trapezoids for the 2-slope+horizontal Annex-14 profile."""
    p = ax14["approach_surface"]
    thr, end, u, perp, L = _rwy_axes(rwy, frame)
    a_axis = -u           # approach axis points AWAY from runway, away from end
    origin = thr + p["inner_edge_offset_m"] * a_axis
    w0 = p["inner_edge_width_m"] / 2.0
    div = p["divergence_each_side"]
    L1 = p["length_first_section_m"]
    L2 = p["length_second_section_m"]
    L3 = p["horizontal_section_length_m"]
    sl1 = p["slope_first_section"]
    sl2 = p["slope_second_section"]
    z1 = L1 * sl1
    z2 = z1 + L2 * sl2

    def hw(s):  # half-width at axial offset s from origin
        return w0 + s * div

    prisms = []
    # Section 1 — slope sl1, z=0 at origin
    poly = _trapezoid(origin, a_axis, perp, 0.0, L1, hw(0.0), hw(L1))
    a, b, c = _affine_along_axis(origin, a_axis, sl1, 0.0)
    prisms.append(Prism(
        name=f"{rwy['id']}_approach_s1", surface=APPROACH,
        runway_id=rwy["id"], end_id="thr", footprint=poly, z_form="affine",
        z_top_a=a, z_top_b=b, z_top_c=c))
    # Section 2 — slope sl2, z=z1 at axial L1
    origin2 = origin + L1 * a_axis
    poly = _trapezoid(origin, a_axis, perp, L1, L1 + L2, hw(L1), hw(L1 + L2))
    a, b, c = _affine_along_axis(origin2, a_axis, sl2, z1)
    prisms.append(Prism(
        name=f"{rwy['id']}_approach_s2", surface=APPROACH,
        runway_id=rwy["id"], end_id="thr", footprint=poly, z_form="affine",
        z_top_a=a, z_top_b=b, z_top_c=c))
    # Section 3 — horizontal at z2
    poly = _trapezoid(origin, a_axis, perp, L1 + L2, L1 + L2 + L3,
                      hw(L1 + L2), hw(L1 + L2 + L3))
    prisms.append(Prism(
        name=f"{rwy['id']}_approach_s3", surface=APPROACH,
        runway_id=rwy["id"], end_id="thr", footprint=poly, z_form="affine",
        z_top_a=0.0, z_top_b=0.0, z_top_c=z2))
    return prisms


def build_takeoff_climb(rwy: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    """Takeoff-climb surface: starts past the runway stop-end, climbs at constant slope."""
    p = ax14["takeoff_climb_surface"]
    thr, end, u, perp, L = _rwy_axes(rwy, frame)
    t_axis = u            # away from threshold, past end
    origin = end + p["inner_edge_offset_m"] * t_axis
    w0 = p["inner_edge_width_m"] / 2.0
    div = p["divergence_each_side"]
    w_cap = p["final_width_m"] / 2.0
    slope = p["slope"]
    Ltot = p["total_length_m"]
    # Width cap kicks in at s_cap = (w_cap - w0) / div
    s_cap = max(0.0, (w_cap - w0) / div) if div > 0 else Ltot
    s_cap = min(s_cap, Ltot)

    prisms = []
    # First (divergent) segment
    if s_cap > 0:
        poly = _trapezoid(origin, t_axis, perp, 0.0, s_cap, w0, w0 + s_cap * div)
        a, b, c = _affine_along_axis(origin, t_axis, slope, 0.0)
        prisms.append(Prism(
            name=f"{rwy['id']}_takeoff_s1", surface=TAKEOFF,
            runway_id=rwy["id"], end_id="end", footprint=poly, z_form="affine",
            z_top_a=a, z_top_b=b, z_top_c=c))
    # Capped (constant-width) segment
    if s_cap < Ltot:
        poly = _trapezoid(origin, t_axis, perp, s_cap, Ltot, w_cap, w_cap)
        a, b, c = _affine_along_axis(origin, t_axis, slope, 0.0)
        prisms.append(Prism(
            name=f"{rwy['id']}_takeoff_s2", surface=TAKEOFF,
            runway_id=rwy["id"], end_id="end", footprint=poly, z_form="affine",
            z_top_a=a, z_top_b=b, z_top_c=c))
    return prisms


def build_transitional(rwy: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    """Two side slabs rising laterally from the runway strip edges to inner-horizontal height."""
    tp = ax14["transitional_surface"]
    sp = ax14["runway_strip"]
    thr, end, u, perp, L = _rwy_axes(rwy, frame)
    slope = tp["slope"]
    z_cap = tp["inner_horizontal_height_m"]
    half_strip = sp["width_each_side_m"]
    end_pad = sp["end_length_m"]
    lateral = z_cap / slope        # outward extent until z_top reaches z_cap

    prisms = []
    for side, sign in (("L", +1), ("R", -1)):
        # Footprint: rectangle along axis from thr-end_pad to end+end_pad,
        # offset from centreline by [half_strip, half_strip+lateral] along sign*perp.
        s0 = -end_pad
        s1 = L + end_pad
        # corners (axial s, perp offset t with t = sign*(half_strip..half_strip+lateral))
        c_inner_a = thr + s0 * u + sign * half_strip * perp
        c_outer_a = thr + s0 * u + sign * (half_strip + lateral) * perp
        c_outer_b = thr + s1 * u + sign * (half_strip + lateral) * perp
        c_inner_b = thr + s1 * u + sign * half_strip * perp
        poly = orient(Polygon([tuple(c_inner_a), tuple(c_outer_a),
                               tuple(c_outer_b), tuple(c_inner_b)]), sign=1.0)
        # z_top = slope * (sign*perp·(P-thr) - half_strip)
        a, b, c = _affine_along_perp(thr, sign * perp, slope, half_strip, z0=0.0)
        prisms.append(Prism(
            name=f"{rwy['id']}_transitional_{side}", surface=TRANSITIONAL,
            runway_id=rwy["id"], end_id="side", footprint=poly, z_form="affine",
            z_top_a=a, z_top_b=b, z_top_c=c))
    return prisms


def build_inner_horizontal(cfg: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    p = ax14["inner_horizontal_surface"]
    poly = _disc(np.array([0.0, 0.0]), p["radius_m"])
    return [Prism(name="inner_horizontal", surface=INNER_HORIZONTAL,
                  runway_id="-", end_id="arp", footprint=poly, z_form="affine",
                  z_top_a=0.0, z_top_b=0.0, z_top_c=float(p["height_above_arp_m"]))]


def build_conical(cfg: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    ihp = ax14["inner_horizontal_surface"]
    p = ax14["conical_surface"]
    r_in = float(ihp["radius_m"])
    height_above_ih = float(p["height_above_inner_horizontal_m"])
    slope = float(p["slope"])
    r_out = r_in + height_above_ih / slope
    poly = _annulus(np.array([0.0, 0.0]), r_in, r_out)
    # z_top(P) = ih_height + slope * (r - r_in)
    return [Prism(name="conical", surface=CONICAL,
                  runway_id="-", end_id="arp", footprint=poly, z_form="radial",
                  z_top_c=float(ihp["height_above_arp_m"]),
                  z_top_slope=slope, z_top_r0=r_in,
                  cx=0.0, cy=0.0)]


def build_runway_strip(rwy: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    """Ground-level strip prism (z_low=z_top=0; degenerate vertically but anchors SDF=0 on RWY)."""
    sp = ax14["runway_strip"]
    thr, end, u, perp, L = _rwy_axes(rwy, frame)
    poly = _rect(thr, u, perp, -sp["end_length_m"], L + sp["end_length_m"],
                 sp["width_each_side_m"])
    return [Prism(name=f"{rwy['id']}_strip", surface=RUNWAY_STRIP,
                  runway_id=rwy["id"], end_id="both", footprint=poly, z_form="affine",
                  z_top_a=0.0, z_top_b=0.0, z_top_c=0.0)]


def build_resa(rwy: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    """Runway-end safety area, just beyond the stop end (i.e., 'end' point in the YAML)."""
    p = ax14["resa"]
    thr, end, u, perp, L = _rwy_axes(rwy, frame)
    poly = _rect(end, u, perp, 0.0, p["length_m"], p["width_m"] / 2.0)
    return [Prism(name=f"{rwy['id']}_resa", surface=RESA,
                  runway_id=rwy["id"], end_id="end", footprint=poly, z_form="affine",
                  z_top_a=0.0, z_top_b=0.0, z_top_c=0.0)]


def build_ofz_inner_approach(rwy: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    p = ax14["ofz_inner_approach"]
    thr, end, u, perp, L = _rwy_axes(rwy, frame)
    a_axis = -u
    origin = thr + p["inner_edge_offset_m"] * a_axis
    poly = _rect(origin, a_axis, perp, 0.0, p["length_m"], p["inner_edge_width_m"] / 2.0)
    a, b, c = _affine_along_axis(origin, a_axis, p["slope"], 0.0)
    return [Prism(name=f"{rwy['id']}_ofz_inapp", surface=OFZ_INNER_APPROACH,
                  runway_id=rwy["id"], end_id="thr", footprint=poly, z_form="affine",
                  z_top_a=a, z_top_b=b, z_top_c=c)]


def build_ofz_inner_transitional(rwy: dict, frame: AirportFrame, ax14: dict) -> List[Prism]:
    """Steeper inner-transitional OFZ on each side of the runway (1:3-ish)."""
    p = ax14["ofz_inner_transitional"]
    sp = ax14["runway_strip"]
    thr, end, u, perp, L = _rwy_axes(rwy, frame)
    slope = p["slope"]
    z_cap = p["height_m"]
    half_strip = sp["width_each_side_m"]
    lateral = z_cap / slope
    prisms = []
    for side, sign in (("L", +1), ("R", -1)):
        s0, s1 = 0.0, L
        c_inner_a = thr + s0 * u + sign * half_strip * perp
        c_outer_a = thr + s0 * u + sign * (half_strip + lateral) * perp
        c_outer_b = thr + s1 * u + sign * (half_strip + lateral) * perp
        c_inner_b = thr + s1 * u + sign * half_strip * perp
        poly = orient(Polygon([tuple(c_inner_a), tuple(c_outer_a),
                               tuple(c_outer_b), tuple(c_inner_b)]), sign=1.0)
        a, b, c = _affine_along_perp(thr, sign * perp, slope, half_strip, z0=0.0)
        prisms.append(Prism(
            name=f"{rwy['id']}_ofz_intr_{side}", surface=OFZ_INNER_TRANSITIONAL,
            runway_id=rwy["id"], end_id="side", footprint=poly, z_form="affine",
            z_top_a=a, z_top_b=b, z_top_c=c))
    return prisms


# ============================================================================
# Top-level assembly
# ============================================================================

def build_airport_surfaces(cfg: dict, frame: AirportFrame, ax14: dict) -> gpd.GeoDataFrame:
    """Run every per-runway and per-airport builder, return a single GeoDataFrame.

    Geometries are in local ENU metres (origin = ARP). CRS is set to a custom
    proj string identifying the airport frame (azimuthal equidistant @ ARP).
    """
    prisms: List[Prism] = []
    for rwy in cfg["runways"]:
        prisms.extend(build_approach(rwy, frame, ax14))
        prisms.extend(build_takeoff_climb(rwy, frame, ax14))
        prisms.extend(build_transitional(rwy, frame, ax14))
        prisms.extend(build_runway_strip(rwy, frame, ax14))
        prisms.extend(build_resa(rwy, frame, ax14))
        prisms.extend(build_ofz_inner_approach(rwy, frame, ax14))
        prisms.extend(build_ofz_inner_transitional(rwy, frame, ax14))
    prisms.extend(build_inner_horizontal(cfg, frame, ax14))
    prisms.extend(build_conical(cfg, frame, ax14))

    rows = [p.to_row() for p in prisms]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    # Tag CRS as a custom local AEQD so consumers know the geometries are local-ENU metres.
    gdf.set_crs(
        f"+proj=aeqd +lat_0={frame.lat0} +lon_0={frame.lon0} +x_0=0 +y_0=0 "
        f"+datum=WGS84 +units=m +no_defs",
        inplace=True,
    )
    logger.info("Built %d OLS prisms for %s", len(gdf), frame.icao)
    return gdf
