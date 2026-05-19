"""Coordinate-reference utilities.

Working frame: local ENU around each airport's ARP, using pyproj's azimuthal-equidistant
projection. UTM is also exposed for compatibility with shapefiles. Heights are in metres
relative to MSL; AGL is derived from the airport elevation in the config.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pyproj


@dataclass(frozen=True)
class AirportFrame:
    icao: str
    lat0: float           # ARP latitude (deg, WGS-84)
    lon0: float           # ARP longitude (deg, WGS-84)
    elev_m: float         # field elevation (m MSL)
    utm_epsg: int         # local UTM zone for compatibility ops
    _enu_proj: pyproj.Proj = None  # cached azimuthal equidistant
    _utm_proj: pyproj.Proj = None

    @classmethod
    def from_cfg(cls, cfg: dict) -> "AirportFrame":
        return cls(
            icao=cfg["icao"],
            lat0=float(cfg["arp"]["lat"]),
            lon0=float(cfg["arp"]["lon"]),
            elev_m=float(cfg["arp"]["elev_m"]),
            utm_epsg=int(cfg["local_crs_epsg"]),
        )

    # ---- ENU projection helpers -------------------------------------------------
    def enu_proj(self) -> pyproj.Proj:
        proj_str = (
            f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} "
            f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        return pyproj.Proj(proj_str)

    def utm_proj(self) -> pyproj.Proj:
        return pyproj.CRS.from_epsg(self.utm_epsg)

    # ---- vectorised conversions -------------------------------------------------
    def wgs_to_enu(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(lon, lat) deg → (x_east_m, y_north_m)."""
        p = self.enu_proj()
        x, y = p(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        return x, y

    def enu_to_wgs(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.enu_proj()
        lon, lat = p(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), inverse=True)
        return lon, lat


def to_local_msl_m(elev_ft: float) -> float:
    """Feet (MSL) to metres (MSL)."""
    return float(elev_ft) * 0.3048
