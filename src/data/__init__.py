"""Real-data acquisition for DREAM.

Each source module (`source_faa_nasr`, `source_faa_dof`, `source_usgs_3dep`, `source_osm`,
`source_opensky`, `source_noaa_wx`, `source_lawa`, `source_bts`) exposes a `fetch(airport_cfg,
window, out_dir)` callable that downloads, validates, projects, and caches the data, writing
both the parquet/geojson/tif output AND a `_manifest.json` sibling.

Failures degrade gracefully: write `<source>.OFFLINE.json` with the error and a manual-recovery
checklist; the rest of the pipeline checks for the OFFLINE marker and continues.

The package also exposes ``sanity_check(out_dir, airport_cfg) -> dict`` for the M0 integration
gate (no network required).
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["sanity_check"]


def sanity_check(out_dir: Path, airport_cfg: dict) -> dict:
    """Offline smoke test for the data lane.

    Exercises the network-free entry points: the FAA NASR runway projector (always works
    because it reads the in-memory airport config), the METAR raw-string parser, the OSM
    height coercion, the BTS column normaliser, and the shared geodesy helpers. No HTTP.

    Args:
        out_dir: write any artefacts here (mkdir parents).
        airport_cfg: parsed `configs/sanity.yaml` for synthetic airport KSYN.
    Returns:
        Non-empty dict consumed by ``scripts/run_sanity.py``.
    """
    import numpy as np
    import pandas as pd

    from . import source_bts, source_faa_nasr, source_noaa_wx, source_osm
    from ._common import bbox_around_arp, great_circle_nm
    from src.utils.crs import AirportFrame

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    metrics: dict = {}

    # 1. FAA NASR — projects synthetic airport runways to ENU, writes parquet + geojson.
    nasr_res = source_faa_nasr.fetch(airport_cfg, window="sanity", out_dir=out_dir)
    runways_pq = out_dir / "runways.parquet"
    df_rwy = pd.read_parquet(runways_pq)
    metrics["nasr_runway_count"] = int(len(df_rwy))
    outputs.append(str(runways_pq))

    # 2. METAR parser — exercise on a canonical raw observation.
    raw = "KSYN 020853Z 24008KT 10SM FEW015 SCT200 19/14 A2992 RMK AO2"
    metar = source_noaa_wx._parse_metar(raw)
    metrics["metar_parsed_wind_kt"] = float(metar["wind_kt"])
    metrics["metar_parsed_flight_rule"] = metar["flight_rule"]

    # 3. OSM coerce_height — exercise the helper.
    h_levels = source_osm._coerce_height({"building:levels": "5"})
    metrics["osm_height_from_5_levels_m"] = float(h_levels)

    # 4. BTS normaliser — synthetic two-row frame.
    raw_db1b = pd.DataFrame({
        "ItinID": ["1", "2"], "Origin": ["LAX", "JFK"], "Dest": ["JFK", "LAX"],
        "Reporting_Airline": ["AA", "UA"], "Passengers": ["1", "2"],
        "MktFare": ["350.5", "410.0"], "MktDistance": ["2450", "2450"],
        "Quarter": ["3", "3"], "Year": ["2024", "2024"],
    })
    norm = source_bts._normalise(raw_db1b)
    metrics["bts_normalised_rows"] = int(len(norm))

    # 5. Geodesy helpers (must succeed even on the equator-centred KSYN config).
    frame = AirportFrame.from_cfg(airport_cfg)
    bbox = bbox_around_arp(frame, half_km=5.0)
    metrics["bbox_width_deg"] = float(bbox[2] - bbox[0])
    metrics["self_distance_nm"] = float(
        great_circle_nm(airport_cfg["arp"]["lon"], airport_cfg["arp"]["lat"],
                        airport_cfg["arp"]["lon"], airport_cfg["arp"]["lat"])
    )

    # 6. Write a small sanity summary alongside the artefacts.
    import json
    summary = {"ok": True, "metrics": metrics, "outputs": outputs,
               "nasr_status": nasr_res.status}
    (out_dir / "data_sanity.json").write_text(json.dumps(summary, indent=2))
    outputs.append(str(out_dir / "data_sanity.json"))

    return {
        "data_ok": True,
        "ok": True,
        "metrics": metrics,
        "outputs": outputs,
        "nasr_status": nasr_res.status,
    }
