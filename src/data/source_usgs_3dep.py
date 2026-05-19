"""USGS 3DEP DEM tiles via The National Map (TNM) Access API.

Fetches 1/3 arc-second elevation tiles covering a 60 km box around the airport ARP.
The TNM Access API exposes a public JSON product index; we request the
`National Elevation Dataset (NED) 1/3 arc-second` product (`13-arc-second-current`)
intersecting the airport bounding box, then download each TIFF and mosaic into a
single GeoTIFF under `data/processed/<ICAO>/dem.tif`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils import io as io_utils
from src.utils import paths as path_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

from ._common import FetchResult, bbox_around_arp, download_to, http_get

logger = get_logger(__name__)

TNM_PRODUCTS_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
SOURCE_URL = TNM_PRODUCTS_API
PRODUCT_NAMES = (
    "National Elevation Dataset (NED) 1/3 arc-second",
    "Digital Elevation Model (DEM) 1 meter",
)


def _query_tnm(bbox: tuple[float, float, float, float]) -> list[dict]:
    """Return TNM JSON items whose product matches our preferred list."""
    items: list[dict] = []
    for product in PRODUCT_NAMES:
        params = {
            "datasets": product,
            "bbox": ",".join(f"{c:.6f}" for c in bbox),
            "max": 32,
            "outputFormat": "JSON",
        }
        try:
            r = http_get(TNM_PRODUCTS_API, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning("TNM query for %r failed: %s", product, e)
            continue
        for it in data.get("items", []):
            url = it.get("downloadURL")
            if url and url.lower().endswith((".tif", ".tiff")):
                it["_product"] = product
                items.append(it)
        if items:
            break  # don't mix products
    return items


def _mosaic_tiles(tile_paths: list[Path], out_path: Path) -> None:
    import rasterio
    from rasterio.merge import merge as rio_merge

    src_files = [rasterio.open(p) for p in tile_paths]
    try:
        arr, transform = rio_merge(src_files)
        meta = src_files[0].meta.copy()
        meta.update({"height": arr.shape[1], "width": arr.shape[2],
                     "transform": transform, "count": arr.shape[0],
                     "compress": "deflate", "tiled": True})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr)
    finally:
        for s in src_files:
            s.close()


def fetch(airport_cfg: dict, *, window: str, out_dir: Path,
          half_km: float = 30.0) -> FetchResult:
    icao = airport_cfg["icao"]
    frame = AirportFrame.from_cfg(airport_cfg)
    bbox = bbox_around_arp(frame, half_km=half_km)

    cache_dir = path_utils.CACHE / "usgs_3dep" / icao
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = _query_tnm(bbox)
    if not items:
        raise RuntimeError(
            f"TNM Access API returned no DEM products for bbox {bbox}. "
            "Try the manual recovery flow (browser USGS Earth Explorer)."
        )

    tile_paths: list[Path] = []
    for it in items[:16]:  # cap to 16 tiles for safety
        url = it["downloadURL"]
        name = Path(url).name
        p = cache_dir / name
        if not p.exists():
            try:
                download_to(url, p, timeout=300)
            except Exception as e:
                logger.warning("DEM tile %s failed: %s", url, e)
                continue
        tile_paths.append(p)

    if not tile_paths:
        raise RuntimeError(
            "TNM returned items but no DEM TIFF downloaded successfully; "
            "check network egress."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    dem_path = out_dir / "dem.tif"
    if len(tile_paths) == 1:
        # single-tile shortcut: just copy
        import shutil
        shutil.copyfile(tile_paths[0], dem_path)
    else:
        _mosaic_tiles(tile_paths, dem_path)

    # Quick sanity: read shape + extent
    import rasterio
    with rasterio.open(dem_path) as ds:
        shape = (ds.height, ds.width)
        crs = str(ds.crs)
        bounds = ds.bounds
        nodata = ds.nodata

    io_utils.write_manifest(
        dem_path,
        source="usgs_3dep",
        source_url=SOURCE_URL,
        params={"airport": icao, "window": window, "bbox": list(bbox),
                "tile_count": len(tile_paths),
                "product": items[0].get("_product")},
        extra={"shape": list(shape), "crs": crs,
               "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
               "nodata": nodata,
               "tile_urls": [it["downloadURL"] for it in items[:len(tile_paths)]]},
    )
    logger.info("USGS 3DEP: %d tiles → %s (%dx%d)", len(tile_paths), dem_path, *shape)
    return FetchResult(name="usgs_3dep", status="ok", files=[dem_path.name],
                       extra={"tile_count": len(tile_paths)})
