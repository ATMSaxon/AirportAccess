"""Joint evaluation: assemble a single KPI table, check Pareto ranking, plot figures.

This is the integration layer between the three KPI engines (safety/capacity/accessibility)
and the experiment-tracker rows R060/R070. The public API is small:

* ``KPIResult`` — one row of the per-corridor KPI table.
* ``assemble_kpi_table(corridor_dir, airport_cfg, support_artefacts) -> pd.DataFrame``
  Walks all corridor JSONs in ``corridor_dir``, computes the eight safety / five capacity
  / five accessibility KPIs against any artefacts present in ``support_artefacts``, and
  returns the flattened table.
* ``assert_pareto_ranking(df, safety_col, monotone) -> bool``
  Group by (airport, date, hour, pair); verify that the chosen safety column is
  monotone non-increasing across baselines in the requested order.
* ``make_figures(kpi_path, out_dir)`` — Pareto, per-baseline boxplot, KLAX/KSFO bar,
  hour-of-day line. All matplotlib (no plotly), saved as ``.png`` next to the table.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.utils.crs import AirportFrame
from src.utils.grid import VoxelGrid
from src.utils.io import read_json
from src.utils.logs import get_logger

from .accessibility_kpis import accessibility_for_corridor
from .capacity_kpis import capacity_for_corridor
from .safety_kpis import safety_for_corridor

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------


@dataclass
class KPIResult:
    """One row in the joint KPI table.

    Identity fields are first; KPI columns follow with the same names that the per-engine
    dataclasses use, so ``df[col]`` references are stable across modules.
    """

    airport: str
    date: str
    hour: int
    vertiport_src: str
    vertiport_dst: str
    baseline: str
    feasible: bool
    length_m: float
    time_s: float
    energy_j: float
    n_expansions: int
    source: str
    dynamic_envelope_used: bool
    risk_used: bool
    # Safety
    ols_violation_rate: float = float("nan")
    min_separation_lateral_nm: float | None = None
    min_separation_vertical_ft: float | None = None
    runway_axis_crossings: int = 0
    approach_interference_s: float = 0.0
    departure_interference_s: float = 0.0
    missed_approach_overlap_s: float = 0.0
    obstacle_margin_min_m: float = float("nan")
    ofv_compliance: bool | None = None
    # Capacity
    runway_delay_extra_s: float | None = None
    throughput_preservation: float | None = None
    evtol_ops_per_hour: float = 0.0
    corridor_closure_rate_pct: float | None = None
    atc_intervention_proxy: int | None = None
    # Accessibility
    access_time_saving_min_vs_road: float = 0.0
    passenger_weighted_access_score: float = 0.0
    vertiport_to_terminal_transfer_min: float = 0.0
    weather_reliability_pct: float | None = None
    peak_service_capacity_ops_per_hour: float = 0.0
    # Provenance
    corridor_json: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Corridor reloader
# ---------------------------------------------------------------------------


def _corridor_from_dict(d: dict) -> "Corridor":  # noqa: F821 -- forward reference
    """Reconstruct a minimal Corridor from its on-disk JSON dump.

    Real corridor JSONs only contain scalar KPI fields (no ``path_enu`` array — that lives
    only in the GeoJSON sibling). We re-load the GeoJSON sibling to recover the geometry.
    """
    from src.planning.astar import Corridor

    pair = d.get("vertiport_pair", ["?", "?"])
    return Corridor(
        feasible=bool(d.get("feasible", False)),
        baseline=str(d.get("baseline", "?")),
        vertiport_pair=(str(pair[0]), str(pair[1])),
        date=str(d.get("date", "")),
        hour=int(d.get("hour", 0)),
        time_s=float(d.get("time_s", 0.0)),
        energy_j=float(d.get("energy_j", 0.0)),
        risk_integral=float(d.get("risk_integral", 0.0)),
        noise_integral=float(d.get("noise_integral", 0.0)),
        capacity_impact=float(d.get("capacity_impact", 0.0)),
        total_cost=float(d.get("total_cost", 0.0)),
        length_m=float(d.get("length_m", 0.0)),
        n_expansions=int(d.get("n_expansions", 0)),
        dynamic_envelope_used=bool(d.get("dynamic_envelope_used", False)),
        risk_used=bool(d.get("risk_used", False)),
        notes=list(d.get("notes", [])),
        source=str(d.get("source", "real")),
    )


def _attach_geometry_from_geojson(corridor, geojson_path: Path) -> None:
    """Hydrate corridor.path_enu, path_wgs, path_ijk from the sibling GeoJSON, if present."""
    if not geojson_path.exists():
        return
    try:
        gj = read_json(geojson_path)
    except Exception as e:  # noqa: BLE001
        LOG.warning("could not read geojson %s: %s", geojson_path, e)
        return
    feats = gj.get("features", [])
    if not feats:
        return
    geom = feats[0].get("geometry")
    if geom is None or geom.get("type") != "LineString":
        return
    coords = np.asarray(geom["coordinates"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 3:
        return
    corridor.path_wgs = coords


def _corridor_path_enu_from_wgs(corridor, frame: AirportFrame) -> None:
    """Populate ``path_enu`` and a best-effort ``path_ijk`` from ``path_wgs``."""
    if corridor.path_wgs is None:
        return
    lon = corridor.path_wgs[:, 0]
    lat = corridor.path_wgs[:, 1]
    z = corridor.path_wgs[:, 2]
    x, y = frame.wgs_to_enu(lon, lat)
    corridor.path_enu = np.column_stack([x, y, z]).astype(np.float32)


def _corridor_path_ijk(corridor, grid: VoxelGrid) -> None:
    """Best-effort projection of ``path_enu`` onto integer voxel indices."""
    if corridor.path_enu is None:
        return
    ix, iy, iz = grid.world_to_index(
        corridor.path_enu[:, 0], corridor.path_enu[:, 1], corridor.path_enu[:, 2]
    )
    corridor.path_ijk = np.column_stack([ix, iy, iz]).astype(np.int32)


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def assemble_kpi_table(
    corridor_dir: Path,
    airport_cfg: dict,
    support_artefacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Walk corridor JSONs under ``corridor_dir`` and compute the joint KPI table.

    ``support_artefacts`` is a flat dict; recognised keys (all optional):

    * ``sdf`` (np.ndarray)         — required for safety KPIs (else ``ols_violation_rate=NaN``).
    * ``grid`` (VoxelGrid)         — required for safety KPIs.
    * ``frame`` (AirportFrame)     — required (else fall back to AirportFrame.from_cfg).
    * ``adsb`` (pd.DataFrame)      — feeds separation + DES + ATC-proxy.
    * ``envelopes_T`` (np.ndarray) — feeds corridor-closure-rate.
    * ``metar`` (pd.DataFrame)     — feeds weather reliability.
    * ``bts_od`` (pd.DataFrame)    — feeds passenger weighting.
    * ``lawa_peaks`` (pd.DataFrame)— feeds passenger weighting peak share.
    * ``osrm_url`` (str)           — feeds road-time KPI (else proxy fallback).
    * ``ofv`` (dict[VID -> np.ndarray]) — feeds OFV-compliance.
    """
    corridor_dir = Path(corridor_dir)
    support_artefacts = support_artefacts or {}

    sdf = support_artefacts.get("sdf")
    grid = support_artefacts.get("grid")
    frame: AirportFrame | None = support_artefacts.get("frame")
    if frame is None:
        try:
            frame = AirportFrame.from_cfg(airport_cfg)
        except Exception:  # noqa: BLE001
            frame = None
    adsb = support_artefacts.get("adsb")
    envelopes_T = support_artefacts.get("envelopes_T")
    metar = support_artefacts.get("metar")
    bts_od = support_artefacts.get("bts_od")
    lawa_peaks = support_artefacts.get("lawa_peaks")
    osrm_url = support_artefacts.get("osrm_url")
    ofv = support_artefacts.get("ofv") or {}

    rows: list[KPIResult] = []

    json_paths = sorted(corridor_dir.rglob("*.json"))
    json_paths = [p for p in json_paths if not p.name.endswith("_manifest.json")
                  and not p.name.startswith("summary")]

    LOG.info("assembling KPI table from %d corridor JSONs under %s",
             len(json_paths), corridor_dir)

    for path in json_paths:
        try:
            d = read_json(path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("skipping %s: %s", path, e)
            continue
        # Expect dict (corridor JSON) — skip GeoJSON-style payloads if any slipped in.
        if not isinstance(d, dict) or "baseline" not in d:
            continue
        corridor = _corridor_from_dict(d)
        geojson = path.with_suffix(".geojson")
        _attach_geometry_from_geojson(corridor, geojson)
        if frame is not None:
            _corridor_path_enu_from_wgs(corridor, frame)
        if grid is not None:
            _corridor_path_ijk(corridor, grid)

        # Safety
        if sdf is not None and grid is not None and frame is not None:
            sk = safety_for_corridor(
                corridor,
                sdf=sdf, grid=grid, frame=frame, airport_cfg=airport_cfg,
                adsb=adsb,
                ofv_start_mask=ofv.get(corridor.vertiport_pair[0]),
                ofv_end_mask=ofv.get(corridor.vertiport_pair[1]),
            )
            safety_kw = sk.to_dict()
        else:
            safety_kw = {}

        # Capacity
        if frame is not None:
            ck = capacity_for_corridor(
                corridor,
                adsb_arrivals=adsb,
                envelopes_T=envelopes_T,
                airport_cfg=airport_cfg,
                frame=frame,
            )
            capacity_kw = ck.to_dict()
        else:
            capacity_kw = {}

        # Accessibility — runs even without geometry (uses corridor.time_s & vertiport WGS).
        ak = accessibility_for_corridor(
            corridor,
            airport_cfg=airport_cfg,
            metar=metar,
            bts_od=bts_od,
            lawa_peaks=lawa_peaks,
            osrm_url=osrm_url,
        )
        access_kw = ak.to_dict()

        result = KPIResult(
            airport=str(airport_cfg.get("icao", "?")),
            date=corridor.date,
            hour=int(corridor.hour),
            vertiport_src=corridor.vertiport_pair[0],
            vertiport_dst=corridor.vertiport_pair[1],
            baseline=corridor.baseline,
            feasible=bool(corridor.feasible),
            length_m=float(corridor.length_m),
            time_s=float(corridor.time_s),
            energy_j=float(corridor.energy_j),
            n_expansions=int(corridor.n_expansions),
            source=corridor.source,
            dynamic_envelope_used=bool(corridor.dynamic_envelope_used),
            risk_used=bool(corridor.risk_used),
            corridor_json=str(path),
            **{k: v for k, v in safety_kw.items() if k in KPIResult.__annotations__},
            **{k: v for k, v in capacity_kw.items() if k in KPIResult.__annotations__},
            **{k: v for k, v in access_kw.items() if k in KPIResult.__annotations__},
        )
        rows.append(result)

    if not rows:
        return pd.DataFrame(columns=list(KPIResult.__annotations__.keys()))

    df = pd.DataFrame([r.to_dict() for r in rows])
    return df


# ---------------------------------------------------------------------------
# Pareto check
# ---------------------------------------------------------------------------


def assert_pareto_ranking(
    df: pd.DataFrame,
    safety_col: str = "ols_violation_rate",
    monotone: Iterable[str] = ("B1", "B2", "B3", "B4"),
    *,
    tol: float = 1e-9,
    groupby: Iterable[str] = ("airport", "date", "hour", "vertiport_src", "vertiport_dst"),
) -> bool:
    """Verify the safety column is monotone non-increasing along ``monotone`` per group.

    Implementation: for each (airport, date, hour, src, dst) group, look up the safety
    column at each baseline in ``monotone`` and verify ``v[i+1] <= v[i] + tol``. Missing
    baselines for a group are skipped without failing. ``tol`` allows for floating-point
    noise.

    Returns True iff every group with at least two of the listed baselines is monotone.
    Logs a WARN with the violating group when it fails.
    """
    monotone = list(monotone)
    groupby = list(groupby)
    if df.empty:
        LOG.warning("assert_pareto_ranking: empty KPI table")
        return True
    if safety_col not in df.columns:
        LOG.warning("assert_pareto_ranking: safety_col=%s not in table", safety_col)
        return False

    ok = True
    for key, grp in df.groupby(groupby, dropna=False):
        by_baseline = {row["baseline"]: row[safety_col] for _, row in grp.iterrows()}
        values: list[tuple[str, float]] = []
        for b in monotone:
            v = by_baseline.get(b)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            values.append((b, float(v)))
        if len(values) < 2:
            continue
        for (a_name, a_val), (b_name, b_val) in zip(values[:-1], values[1:]):
            if b_val > a_val + tol:
                LOG.warning(
                    "Pareto violation: group=%s baseline %s=%g > %s=%g (col=%s)",
                    key, b_name, b_val, a_name, a_val, safety_col,
                )
                ok = False
    return ok


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(kpi_path: Path, out_dir: Path) -> list[Path]:
    """Read the KPI table and emit four PNG figures next to it.

    The four figures: (a) Pareto scatter (access vs safety), (b) per-baseline boxplot of
    the safety column, (c) bar chart of mean safety by airport×baseline, (d) hour-of-day
    line plot of mean safety by baseline. All matplotlib.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kpi_path = Path(kpi_path)
    if kpi_path.suffix == ".parquet":
        df = pd.read_parquet(kpi_path)
    else:
        df = pd.read_csv(kpi_path)
    if df.empty:
        LOG.warning("KPI table empty; no figures emitted")
        return []

    written: list[Path] = []
    SAFETY = "ols_violation_rate"
    ACCESS = "access_time_saving_min_vs_road"

    # (a) Pareto: safety (x, lower=better) vs access saving (y, higher=better) by baseline.
    fig, ax = plt.subplots(figsize=(6, 4))
    for b, sub in df.groupby("baseline"):
        ax.scatter(sub[SAFETY], sub[ACCESS], label=b, alpha=0.7)
    ax.set_xlabel("OLS violation rate (lower is safer)")
    ax.set_ylabel("Access time saving vs road (min)")
    ax.set_title("Pareto: safety vs accessibility, by baseline")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "pareto.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # (b) Per-baseline boxplot of safety column.
    fig, ax = plt.subplots(figsize=(6, 4))
    baselines = sorted(df["baseline"].unique())
    data = [df.loc[df["baseline"] == b, SAFETY].dropna().to_numpy() for b in baselines]
    ax.boxplot(data, labels=baselines, showfliers=False)
    ax.set_ylabel(SAFETY)
    ax.set_title("Safety distribution per baseline")
    fig.tight_layout()
    p = out_dir / "safety_boxplot.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # (c) Bar: mean safety by airport × baseline (KLAX vs KSFO when both present).
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = df.pivot_table(index="airport", columns="baseline", values=SAFETY, aggfunc="mean")
    if not pivot.empty:
        pivot.plot.bar(ax=ax)
        ax.set_ylabel(f"mean {SAFETY}")
        ax.set_title("Mean safety by airport × baseline")
    fig.tight_layout()
    p = out_dir / "airport_baseline_bar.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # (d) Hour-of-day line by baseline.
    fig, ax = plt.subplots(figsize=(6, 4))
    if "hour" in df.columns:
        for b, sub in df.groupby("baseline"):
            agg = sub.groupby("hour")[SAFETY].mean()
            if not agg.empty:
                ax.plot(agg.index.to_numpy(), agg.to_numpy(), marker="o", label=b)
        ax.set_xlabel("Hour of day (UTC)")
        ax.set_ylabel(f"mean {SAFETY}")
        ax.set_title("Safety by hour-of-day, per baseline")
        ax.legend()
    fig.tight_layout()
    p = out_dir / "hour_of_day.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    return written
