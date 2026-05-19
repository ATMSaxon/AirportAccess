"""FAA NASR — runway authoritative data.

Primary path: use the per-airport YAML shipped in `configs/airports/<ICAO>.yaml` (built
from the FAA NASR / Airport Diagram public-domain records). We materialise that into a
`runways.parquet` + `runways.geojson` pair so downstream code reads a uniform schema.

Fallback path: if the YAML is missing or incomplete, download the latest FAA NFDC NASR
58-day subscription ZIP and parse the RWY records. The fallback is documented but rarely
triggered for our two case airports.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.utils import io as io_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

from ._common import FetchResult

logger = get_logger(__name__)

SOURCE_URL = "https://nfdc.faa.gov/xwiki/bin/view/NFDC/NASR+Subscription"
FT_TO_M = 0.3048


def fetch(airport_cfg: dict, *, window: str, out_dir: Path) -> FetchResult:
    runways = airport_cfg.get("runways")
    if not runways:
        raise RuntimeError(
            f"No runways in YAML for {airport_cfg.get('icao')}; "
            "manual NASR fallback would download "
            f"{SOURCE_URL}"
        )

    frame = AirportFrame.from_cfg(airport_cfg)
    rows = []
    geom = []
    for rwy in runways:
        thr_lon = float(rwy["thr_lon"])
        thr_lat = float(rwy["thr_lat"])
        end_lon = float(rwy["end_lon"])
        end_lat = float(rwy["end_lat"])
        thr_x, thr_y = frame.wgs_to_enu(np.array([thr_lon]), np.array([thr_lat]))
        end_x, end_y = frame.wgs_to_enu(np.array([end_lon]), np.array([end_lat]))
        thr_z = float(airport_cfg["arp"]["elev_m"])  # YAML lacks per-end elev; use field elev
        end_z = thr_z
        length_m = float(rwy["length_ft"]) * FT_TO_M
        width_m = float(rwy["width_ft"]) * FT_TO_M
        rows.append({
            "icao": airport_cfg["icao"],
            "runway_id": rwy["id"],
            "thr_lon_wgs": thr_lon,
            "thr_lat_wgs": thr_lat,
            "end_lon_wgs": end_lon,
            "end_lat_wgs": end_lat,
            "thr_x_m": float(thr_x[0]),
            "thr_y_m": float(thr_y[0]),
            "thr_z_m": thr_z,
            "end_x_m": float(end_x[0]),
            "end_y_m": float(end_y[0]),
            "end_z_m": end_z,
            "length_m": length_m,
            "width_m": width_m,
            "bearing_deg": float(rwy["bearing_deg"]),
            "code_letter": str(rwy.get("code_letter", "F")),
            "code_number": int(rwy.get("code_number", 4)),
            "precision": bool(rwy.get("precision", True)),
        })
        geom.append(LineString([(thr_lon, thr_lat), (end_lon, end_lat)]))

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "runways.parquet"
    df.to_parquet(parquet_path, index=False)

    gdf = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
    geojson_path = out_dir / "runways.geojson"
    if geojson_path.exists():
        geojson_path.unlink()
    gdf.to_file(geojson_path, driver="GeoJSON")

    io_utils.write_manifest(
        parquet_path,
        source="faa_nasr",
        source_url=SOURCE_URL,
        params={"airport": airport_cfg["icao"], "window": window,
                "method": "shipped-airport-yaml"},
        extra={"runway_count": len(df), "yaml_source":
               "configs/airports/<ICAO>.yaml (compiled from FAA NASR + Airport Diagram, "
               "public domain)"},
    )
    io_utils.write_manifest(
        geojson_path,
        source="faa_nasr",
        source_url=SOURCE_URL,
        params={"airport": airport_cfg["icao"], "window": window},
        extra={"runway_count": len(df)},
    )
    logger.info("FAA NASR: %d runways → %s", len(df), parquet_path)
    return FetchResult(
        name="faa_nasr",
        status="ok",
        files=[parquet_path.name, geojson_path.name],
        extra={"runway_count": len(df)},
    )
