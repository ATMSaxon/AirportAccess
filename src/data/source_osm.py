"""OpenStreetMap Overpass API — buildings, roads, transit, amenities.

Queries the public Overpass instance for a 30 km box centred on the airport ARP.
Falls back across multiple Overpass mirrors if one is throttled.

Outputs:
- `buildings.geojson` (Polygon) + `buildings.parquet` (ENU centroids + heights)
- `roads.geojson` (LineString) + `roads.parquet`
- `amenities.geojson` (Point) + `amenities.parquet`
"""
from __future__ import annotations

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.geometry.base import BaseGeometry

from src.utils import io as io_utils
from src.utils import paths as path_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

from ._common import FetchResult, bbox_around_arp, http_post

logger = get_logger(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
SOURCE_URL = OVERPASS_ENDPOINTS[0]


def _overpass_query(query: str) -> dict:
    last_err: Exception | None = None
    for ep in OVERPASS_ENDPOINTS:
        try:
            r = http_post(ep, data={"data": query}, timeout=180)
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"{ep} returned HTTP {r.status_code}")
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")


def _bbox_str(bbox: tuple[float, float, float, float]) -> str:
    # Overpass expects (south, west, north, east)
    lon_min, lat_min, lon_max, lat_max = bbox
    return f"{lat_min:.5f},{lon_min:.5f},{lat_max:.5f},{lon_max:.5f}"


def _build_geometry(el: dict, nodes: dict[int, tuple[float, float]]) -> BaseGeometry | None:
    """Reconstruct geometry from an Overpass element."""
    t = el.get("type")
    if t == "node":
        return Point(el["lon"], el["lat"])
    if t == "way":
        coords = [(nodes[n][1], nodes[n][0]) for n in el.get("nodes", []) if n in nodes]
        if len(coords) < 2:
            return None
        if coords[0] == coords[-1] and len(coords) >= 4:
            return Polygon(coords)
        return LineString(coords)
    return None


def _coerce_height(tags: dict) -> float:
    """Best-effort height in metres from OSM tags."""
    if "height" in tags:
        try:
            return float(str(tags["height"]).split(" ")[0].replace("m", ""))
        except Exception:
            pass
    if "building:height" in tags:
        try:
            return float(str(tags["building:height"]).split(" ")[0])
        except Exception:
            pass
    lvls = tags.get("building:levels") or tags.get("levels")
    if lvls is not None:
        try:
            return float(lvls) * 3.0  # rough storey height
        except Exception:
            pass
    return float("nan")


def fetch(airport_cfg: dict, *, window: str, out_dir: Path,
          half_km: float = 15.0) -> FetchResult:
    icao = airport_cfg["icao"]
    frame = AirportFrame.from_cfg(airport_cfg)
    bbox = bbox_around_arp(frame, half_km=half_km)
    bbox_str = _bbox_str(bbox)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = path_utils.CACHE / "osm" / icao
    cache_dir.mkdir(parents=True, exist_ok=True)

    files_out: list[str] = []
    totals: dict[str, int] = {}

    # ---- Query 1: buildings (ways tagged building=*)
    q_buildings = f"""[out:json][timeout:120];
(
  way["building"]({bbox_str});
);
out body geom;
"""
    # Geometry inline ('out geom') avoids second round-trip
    data = _overpass_query(q_buildings)
    rows_b, geoms_b = [], []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        coords = el.get("geometry") or []
        if len(coords) < 3:
            continue
        ring = [(c["lon"], c["lat"]) for c in coords]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        try:
            poly = Polygon(ring)
        except Exception:
            continue
        if not poly.is_valid or poly.area == 0:
            continue
        tags = el.get("tags", {})
        cx_lon, cy_lat = poly.centroid.x, poly.centroid.y
        x_m, y_m = frame.wgs_to_enu(np.array([cx_lon]), np.array([cy_lat]))
        # Approx area in m^2 (project centroid for scale)
        area_deg2 = poly.area
        area_m2 = area_deg2 * (111_320 ** 2) * np.cos(np.radians(frame.lat0))
        rows_b.append({
            "osm_id": el.get("id"),
            "building": tags.get("building", "yes"),
            "levels": tags.get("building:levels"),
            "height_m": _coerce_height(tags),
            "x_m": float(x_m[0]),
            "y_m": float(y_m[0]),
            "area_m2": float(area_m2),
        })
        geoms_b.append(poly)
    df_b = pd.DataFrame(rows_b)
    gdf_b = gpd.GeoDataFrame(df_b, geometry=geoms_b, crs="EPSG:4326")
    df_b.to_parquet(out_dir / "buildings.parquet", index=False)
    (out_dir / "buildings.geojson").unlink(missing_ok=True)
    gdf_b.to_file(out_dir / "buildings.geojson", driver="GeoJSON")
    io_utils.write_manifest(out_dir / "buildings.parquet",
                             source="osm", source_url=SOURCE_URL,
                             params={"airport": icao, "bbox": list(bbox),
                                     "query": "way[building]"},
                             extra={"count": len(df_b)})
    totals["buildings"] = len(df_b)
    files_out += ["buildings.parquet", "buildings.geojson"]

    # ---- Query 2: roads (way highway=*)
    q_roads = f"""[out:json][timeout:120];
(
  way["highway"]({bbox_str});
);
out body geom;
"""
    data = _overpass_query(q_roads)
    rows_r, geoms_r = [], []
    for el in data.get("elements", []):
        coords = el.get("geometry") or []
        if len(coords) < 2:
            continue
        line = LineString([(c["lon"], c["lat"]) for c in coords])
        tags = el.get("tags", {})
        # ENU length approximation
        proj = frame.enu_proj()
        xs, ys = proj([c["lon"] for c in coords], [c["lat"] for c in coords])
        length_m = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        rows_r.append({
            "osm_id": el.get("id"),
            "highway": tags.get("highway"),
            "name": tags.get("name"),
            "oneway": tags.get("oneway", "no"),
            "lanes": tags.get("lanes"),
            "length_m": length_m,
            "x_m": float(np.mean(xs)),
            "y_m": float(np.mean(ys)),
        })
        geoms_r.append(line)
    df_r = pd.DataFrame(rows_r)
    gdf_r = gpd.GeoDataFrame(df_r, geometry=geoms_r, crs="EPSG:4326")
    df_r.to_parquet(out_dir / "roads.parquet", index=False)
    (out_dir / "roads.geojson").unlink(missing_ok=True)
    gdf_r.to_file(out_dir / "roads.geojson", driver="GeoJSON")
    io_utils.write_manifest(out_dir / "roads.parquet",
                             source="osm", source_url=SOURCE_URL,
                             params={"airport": icao, "bbox": list(bbox),
                                     "query": "way[highway]"},
                             extra={"count": len(df_r)})
    totals["roads"] = len(df_r)
    files_out += ["roads.parquet", "roads.geojson"]

    # ---- Query 3: amenities + transit nodes
    q_amenities = f"""[out:json][timeout:90];
(
  node["amenity"]({bbox_str});
  node["public_transport"]({bbox_str});
  node["railway"="station"]({bbox_str});
  node["aeroway"]({bbox_str});
);
out body;
"""
    data = _overpass_query(q_amenities)
    rows_a, geoms_a = [], []
    for el in data.get("elements", []):
        if el.get("type") != "node":
            continue
        lon, lat = el.get("lon"), el.get("lat")
        if lon is None or lat is None:
            continue
        tags = el.get("tags", {})
        x_m, y_m = frame.wgs_to_enu(np.array([lon]), np.array([lat]))
        rows_a.append({
            "osm_id": el.get("id"),
            "amenity": tags.get("amenity") or tags.get("public_transport")
                       or tags.get("railway") or tags.get("aeroway"),
            "name": tags.get("name"),
            "x_m": float(x_m[0]),
            "y_m": float(y_m[0]),
        })
        geoms_a.append(Point(lon, lat))
    df_a = pd.DataFrame(rows_a)
    gdf_a = gpd.GeoDataFrame(df_a, geometry=geoms_a, crs="EPSG:4326")
    df_a.to_parquet(out_dir / "amenities.parquet", index=False)
    (out_dir / "amenities.geojson").unlink(missing_ok=True)
    gdf_a.to_file(out_dir / "amenities.geojson", driver="GeoJSON")
    io_utils.write_manifest(out_dir / "amenities.parquet",
                             source="osm", source_url=SOURCE_URL,
                             params={"airport": icao, "bbox": list(bbox),
                                     "query": "amenities + transit + aeroway"},
                             extra={"count": len(df_a)})
    totals["amenities"] = len(df_a)
    files_out += ["amenities.parquet", "amenities.geojson"]

    logger.info("OSM: buildings=%d roads=%d amenities=%d",
                totals["buildings"], totals["roads"], totals["amenities"])
    return FetchResult(name="osm", status="ok", files=files_out, extra=totals)
