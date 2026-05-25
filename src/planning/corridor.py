"""Corridor orchestrator: load artefacts, run the planner, write GeoJSON + JSON.

This is the production entry point invoked by ``scripts/plan_corridors.py``. It:

1. Loads airport config + cost weights.
2. Resolves vertiport endpoints in ENU and snaps them through their respective OFV masks.
3. Loads the static SDF (mandatory) and, depending on baseline, the dynamic envelope
   and risk grid (optional — graceful fallback if missing).
4. Optionally coarsens to a planning-grid resolution.
5. Calls ``Planner.plan`` for the requested baseline.
6. Writes a GeoJSON LineString and a sibling JSON KPI dict per corridor, plus a manifest.

A second public entry, ``plan_corridors_batch``, loops over (date, hour, vertiport-pair,
baseline) combinations and returns the list of written corridor JSON paths.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from src.utils import config as cfg_io
from src.utils import paths
from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid
from src.utils.io import write_json, write_manifest
from src.utils.logs import get_logger

from .astar import Corridor, Planner, PlannerConfig, PlannerInputs
from .graph import (
    EndpointInfeasibleError,
    snap_to_voxel,
    vertiport_anchor_enu,
)
from .loaders import (
    MissingArtifactError,
    load_density_slice,
    load_envelope_slice,
    load_risk_slice,
    load_sdf,
    ofv_mask_on_grid,
    runway_closure_mask,
)
from .smoothing import smooth_path

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# I/O writers
# ---------------------------------------------------------------------------


def write_corridor_geojson(corridor: Corridor, path: Path) -> Path:
    """Emit a 3-D LineString GeoJSON with KPI scalars in properties.

    For an infeasible corridor, write a Feature with ``geometry: null`` so the file always
    exists and downstream readers can detect failure deterministically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if corridor.feasible and corridor.path_wgs is not None:
        coords = [[float(lon), float(lat), float(z)] for lon, lat, z in corridor.path_wgs]
        geom = {"type": "LineString", "coordinates": coords}
    else:
        geom = None
    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": corridor.to_dict(),
            }
        ],
    }
    write_json(path, feature)
    return path


def write_corridor_json(corridor: Corridor, path: Path) -> Path:
    """Emit the per-corridor KPI dict as JSON."""
    path = Path(path)
    write_json(path, corridor.to_dict())
    return path


# ---------------------------------------------------------------------------
# Single corridor
# ---------------------------------------------------------------------------


def _planner_inputs_from_disk(
    icao: str,
    vertiport_src: str,
    vertiport_dst: str,
    *,
    date: str,
    hour: int,
    baseline: str,
    model: str,
    planning_xy_m: float | None = None,
    planning_z_m: float | None = None,
) -> tuple[PlannerInputs, dict]:
    """Load all artefacts for a single corridor; raise MissingArtifactError for hard misses."""
    airport_cfg = cfg_io.load_airport(icao)
    frame = AirportFrame.from_cfg(airport_cfg)

    # 1. Static SDF (mandatory).
    grid, sdf = load_sdf(icao)

    # 2. OFVs at endpoints. The geometry-engineer's OFV is on a tiny local grid
    # around each vertiport (40³ @ 10 m, see ``src/geometry/INTERFACES.md``); the
    # airport SDF is on a coarser airport-wide grid (e.g. 600³ @ 100 m). We project
    # each OFV onto the SDF grid via trilinear interpolation in ``ofv_mask_on_grid``
    # so downstream code can AND ``(sdf > 0) & ofv_start`` without shape mismatch.
    try:
        src_ofv = ofv_mask_on_grid(icao, vertiport_src, grid)
    except Exception as e:  # noqa: BLE001 — graceful degrade
        LOG.warning("OFV[%s] projection failed (%s); dropping OFV-source mask",
                    vertiport_src, e)
        src_ofv = None
    try:
        dst_ofv = ofv_mask_on_grid(icao, vertiport_dst, grid)
    except Exception as e:  # noqa: BLE001 — graceful degrade
        LOG.warning("OFV[%s] projection failed (%s); dropping OFV-dest mask",
                    vertiport_dst, e)
        dst_ofv = None

    # 3. Dynamic envelope (optional; B1/B2 ignore it).
    envelope = None
    dyn_used = False
    if baseline in ("B3", "B4"):
        slc = load_envelope_slice(icao, date, hour)
        if slc is not None:
            if slc.mask.shape != grid.shape:
                LOG.warning(
                    "envelope shape %s != grid shape %s; ignoring envelope",
                    slc.mask.shape, grid.shape,
                )
            else:
                envelope = slc.mask
                dyn_used = True

    # 4. Risk grid (optional; only B4).
    risk = None
    risk_used = False
    if baseline == "B4":
        r = load_risk_slice(icao, model, date, hour)
        if r is not None and r.shape == grid.shape:
            risk = r
            risk_used = True

    # 5. Density / noise proxy (optional; B2/B3/B4 use alpha_N).
    density = None
    if baseline in ("B2", "B3", "B4"):
        d = load_density_slice(icao, date, hour)
        if d is not None and d.shape == grid.shape:
            density = d

    # 5b. Static both-sides-closed runway corridor mask (B2 only).
    #     Loaded for every baseline so the cache stays warm during a sweep, but only
    #     B2's `_baseline_gate` actually enforces it via `use_static_closure=True`.
    static_closure = runway_closure_mask(icao, grid)

    # 6. Optional planning-grid coarsening.
    if planning_xy_m is not None or planning_z_m is not None:
        from .loaders import coarsen

        cx = float(planning_xy_m) if planning_xy_m else max(grid.dx, grid.dy)
        cz = float(planning_z_m) if planning_z_m else grid.dz
        coarse_grid, arrs = coarsen(
            grid,
            planning_xy_m=cx,
            planning_z_m=cz,
            sdf=sdf,
            envelope=envelope,
            risk=risk,
            density=density,
            ofv=src_ofv,
            static_closure=static_closure,
        )
        # Re-coarsen the destination OFV too.
        _, arrs2 = coarsen(grid, planning_xy_m=cx, planning_z_m=cz, ofv=dst_ofv)
        inputs = PlannerInputs(
            grid=coarse_grid,
            frame=frame,
            sdf=arrs["sdf"],
            envelope=arrs["envelope"],
            risk=arrs["risk"],
            density=arrs["density"],
            ofv_start=arrs["ofv"].astype(bool) if arrs["ofv"] is not None else None,
            ofv_end=arrs2["ofv"].astype(bool) if arrs2["ofv"] is not None else None,
            static_closure=(
                arrs["static_closure"].astype(bool)
                if arrs["static_closure"] is not None else None
            ),
        )
    else:
        inputs = PlannerInputs(
            grid=grid,
            frame=frame,
            sdf=sdf,
            envelope=envelope,
            risk=risk,
            density=density,
            ofv_start=src_ofv.astype(bool) if src_ofv is not None else None,
            ofv_end=dst_ofv.astype(bool) if dst_ofv is not None else None,
            static_closure=static_closure.astype(bool) if static_closure is not None else None,
        )

    meta = {
        "icao": icao,
        "vertiport_src": vertiport_src,
        "vertiport_dst": vertiport_dst,
        "date": date,
        "hour": hour,
        "baseline": baseline,
        "model": model,
        "dynamic_envelope_used": dyn_used,
        "risk_used": risk_used,
        "planning_xy_m": planning_xy_m,
        "planning_z_m": planning_z_m,
    }
    return inputs, meta


def plan_corridor(
    *,
    airport: str,
    vertiport_src: str,
    vertiport_dst: str,
    date: str,
    hour: int,
    baseline: str,
    cfg: PlannerConfig | None = None,
    model: str = "xgb",
    planning_xy_m: float | None = None,
    planning_z_m: float | None = None,
    smooth: bool = True,
) -> Corridor:
    """Plan a single corridor and return the populated ``Corridor``."""
    if cfg is None:
        cfg = PlannerConfig.from_cfg(cfg_io.load_scenario("cost_weights"))

    inputs, meta = _planner_inputs_from_disk(
        airport, vertiport_src, vertiport_dst,
        date=date, hour=hour, baseline=baseline, model=model,
        planning_xy_m=planning_xy_m, planning_z_m=planning_z_m,
    )

    # Endpoint resolution.
    airport_cfg = cfg_io.load_airport(airport)
    vert_cfg = airport_cfg["vertiports"]
    src_xyz = vertiport_anchor_enu(inputs.frame, vert_cfg[vertiport_src])
    dst_xyz = vertiport_anchor_enu(inputs.frame, vert_cfg[vertiport_dst])

    # OFV-aware snapping. For B2 we also exclude voxels inside the both-sides-closed
    # runway corridor so the planner doesn't land its endpoint somewhere the search will
    # immediately reject; for B3/B4 the time-varying envelope (rather than static closure)
    # gates the runway corridors.
    src_mask = (inputs.sdf > 0) & (inputs.ofv_start if inputs.ofv_start is not None else True)
    dst_mask = (inputs.sdf > 0) & (inputs.ofv_end if inputs.ofv_end is not None else True)
    if baseline == "B2" and inputs.static_closure is not None:
        src_mask = src_mask & (~inputs.static_closure)
        dst_mask = dst_mask & (~inputs.static_closure)
    try:
        src_ijk = snap_to_voxel(inputs.grid, x_m=src_xyz[0], y_m=src_xyz[1], z_m=src_xyz[2], mask=src_mask)
        dst_ijk = snap_to_voxel(inputs.grid, x_m=dst_xyz[0], y_m=dst_xyz[1], z_m=dst_xyz[2], mask=dst_mask)
    except EndpointInfeasibleError as e:
        c = Corridor(
            feasible=False,
            baseline=baseline,
            vertiport_pair=(vertiport_src, vertiport_dst),
            date=date,
            hour=hour,
            notes=[f"endpoint snap failed: {e}"],
            source=inputs.source,
        )
        return c

    # Translate snapped indices back to ENU so the planner sees actual cell centres.
    from .graph import ijk_to_world

    src_enu = ijk_to_world(inputs.grid, src_ijk)
    dst_enu = ijk_to_world(inputs.grid, dst_ijk)

    planner = Planner(inputs, cfg)
    corridor = planner.plan(
        src_enu, dst_enu, baseline,
        vertiport_pair=(vertiport_src, vertiport_dst),
        date=date,
        hour=hour,
    )

    if smooth and corridor.feasible and corridor.path_enu is not None and len(corridor.path_enu) > 4:
        smoothed = smooth_path(corridor.path_enu.astype(np.float64), rdp_eps_m=50.0, n_samples=200)
        corridor.path_enu = smoothed.astype(np.float32)
        lon, lat = inputs.frame.enu_to_wgs(smoothed[:, 0], smoothed[:, 1])
        corridor.path_wgs = np.column_stack([lon, lat, smoothed[:, 2]]).astype(np.float64)

    corridor.dynamic_envelope_used = bool(meta["dynamic_envelope_used"])
    corridor.risk_used = bool(meta["risk_used"])
    return corridor


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def plan_corridors_batch(
    *,
    airport: str,
    vertiport_pairs: Iterable[tuple[str, str]],
    dates: Iterable[str],
    hours: Iterable[int],
    baselines: Iterable[str],
    cfg: PlannerConfig | None = None,
    model: str = "xgb",
    out_dir: Path | None = None,
    planning_xy_m: float | None = None,
    planning_z_m: float | None = None,
    smooth: bool = True,
) -> list[Path]:
    """Run the planner over the (dates × hours × pairs × baselines) cartesian product.

    Returns the list of corridor JSON paths written. A sibling GeoJSON is written for each.
    """
    if cfg is None:
        cfg = PlannerConfig.from_cfg(cfg_io.load_scenario("cost_weights"))
    if out_dir is None:
        out_dir = paths.RESULTS / "corridors" / airport
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for date, hour, (src, dst), baseline in itertools.product(
        list(dates), list(hours), list(vertiport_pairs), list(baselines)
    ):
        try:
            corridor = plan_corridor(
                airport=airport,
                vertiport_src=src,
                vertiport_dst=dst,
                date=date,
                hour=hour,
                baseline=baseline,
                cfg=cfg,
                model=model,
                planning_xy_m=planning_xy_m,
                planning_z_m=planning_z_m,
                smooth=smooth,
            )
        except MissingArtifactError:
            raise  # propagate to script (clear next-command msg)
        except Exception as e:  # noqa: BLE001
            LOG.exception("planner crashed: %s", e)
            corridor = Corridor(
                feasible=False,
                baseline=baseline,
                vertiport_pair=(src, dst),
                date=date,
                hour=hour,
                notes=[f"planner exception: {e!s}"],
                source="real",
            )
        sub = out_dir / date
        sub.mkdir(parents=True, exist_ok=True)
        stem = f"{src}_{dst}_{int(hour):02d}_{baseline}"
        json_path = sub / f"{stem}.json"
        geo_path = sub / f"{stem}.geojson"
        write_corridor_json(corridor, json_path)
        write_corridor_geojson(corridor, geo_path)
        write_manifest(
            json_path,
            source="planning",
            source_url="src/planning/corridor.py",
            params={
                "airport": airport, "vertiport_src": src, "vertiport_dst": dst,
                "date": date, "hour": hour, "baseline": baseline, "model": model,
                "planning_xy_m": planning_xy_m, "planning_z_m": planning_z_m,
            },
            extra={
                "feasible": corridor.feasible,
                "length_m": corridor.length_m,
                "n_expansions": corridor.n_expansions,
            },
        )
        written.append(json_path)
        LOG.info(
            "corridor %s/%s %s %s feasible=%s len=%.0fm pops=%d",
            airport, date, stem, hour, corridor.feasible, corridor.length_m, corridor.n_expansions,
        )
    return written
