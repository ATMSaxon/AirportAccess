"""FAA Digital Obstacle File (DOF).

Downloads the DAILY DOF (single national file) from the FAA digital products endpoint,
parses the fixed-width text format, projects each obstacle to ENU around the airport
ARP and filters to within `radius_nm` (default 50 NM).

DOF format reference: FAA AIS DOF User Manual (public). The DAILY DOF is a single
fixed-width text file with one obstacle per line; the column layout is documented in
the user manual under "Record Layout — Obstacle Record".
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import io as io_utils
from src.utils import paths as path_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

from ._common import FetchResult, download_to, great_circle_nm

logger = get_logger(__name__)

# Authoritative public URL (FAA "Digital Products" page).
DOF_PRIMARY = "https://aeronav.faa.gov/Obst_Data/DAILY_DOF.ZIP"
DOF_BACKUP = "https://aeronav.faa.gov/Obst_Data/DAILY_DOF.CSV"
SOURCE_URL = DOF_PRIMARY

# DOF fixed-width parsing (column ranges per FAA DOF User Manual). Columns are 1-indexed
# in the manual; Python slices are 0-indexed.
DOF_COLS = {
    "oas_number":   (0, 6),     # OAS / DOF identifier ("00-NNNNN")
    "verif_status": (8, 9),     # O = verified, U = unverified
    "country":      (10, 12),
    "state":        (13, 15),
    "city":         (16, 33),
    "lat_dms":      (35, 47),   # DD MM SS.SSH
    "lon_dms":      (49, 61),
    "obstacle_type":(62, 80),
    "quantity":     (83, 88),
    "agl_ft":       (89, 94),
    "msl_ft":       (95, 100),
    "lighting":     (102, 103),
    "accuracy_h":   (105, 106),
    "accuracy_v":   (108, 109),
    "marked":       (110, 111),
}

# DOF horizontal / vertical accuracy code translation (per User Manual Appendix).
HORIZ_ACCURACY_FT = {"1": 20, "2": 50, "3": 100, "4": 250, "5": 500, "6": 1000,
                     "7": 2640, "8": 5280, "9": float("nan")}
VERT_ACCURACY_FT = {"A": 3, "B": 10, "C": 20, "D": 50, "E": 125, "F": 250, "G": 500,
                    "H": 1000, "I": float("nan")}


def _parse_dms(s: str) -> float:
    """'DD MM SS.SSH' (or 'DD-MM-SS.SSH') → signed decimal degrees. NaN on failure."""
    s = s.strip()
    if not s:
        return float("nan")
    # DOF lat/lon: degrees, minutes, seconds, hemisphere — separators are spaces in the
    # current DAILY DOF and hyphens in some older formats. Accept both.
    m = re.match(r"^\s*(\d+)[\s\-]+(\d+)[\s\-]+([\d\.]+)\s*([NSEW])\s*$", s)
    if not m:
        return float("nan")
    d, mn, sec, hemi = m.groups()
    val = int(d) + int(mn) / 60 + float(sec) / 3600
    if hemi in ("S", "W"):
        val = -val
    return val


def _parse_csv(text: str) -> pd.DataFrame:
    """Parse the DAILY DOF CSV format."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    rename = {}
    for col, want in [
        ("OAS_NUMBER", "oas_number"), ("OASNUMBER", "oas_number"),
        ("OBSTACLE NUMBER", "oas_number"),
        ("LATITUDE", "lat_dms"), ("LATITUDE_DEC", "lat_dms"),
        ("LONGITUDE", "lon_dms"), ("LONGITUDE_DEC", "lon_dms"),
        ("OBSTACLE_TYPE", "obstacle_type"), ("TYPE", "obstacle_type"),
        ("AGL_HT", "agl_ft"), ("AGL", "agl_ft"), ("HEIGHT_AGL_FT", "agl_ft"),
        ("MSL_HT", "msl_ft"), ("AMSL_HT", "msl_ft"), ("HEIGHT_MSL_FT", "msl_ft"),
        ("HORIZ_ACY", "accuracy_h"), ("HOR_ACC", "accuracy_h"),
        ("VERT_ACY", "accuracy_v"), ("VER_ACC", "accuracy_v"),
        ("MARKING", "marked"), ("LIGHTING", "lighting"),
    ]:
        if col in df.columns:
            rename[col] = want
    df = df.rename(columns=rename)
    return df


def _parse_fixed_width(text: str) -> pd.DataFrame:
    """Parse the DAILY DOF fixed-width text format. Skips header lines."""
    rows = []
    for raw in text.splitlines():
        # Header records begin with "FILE:" or "DOF" — skip
        if len(raw) < 100 or raw.startswith(("FILE:", "DOF", "Obstacle ID", "----")):
            continue
        rec = {}
        for col, (a, b) in DOF_COLS.items():
            rec[col] = raw[a:b].strip()
        if not rec["oas_number"] or rec["oas_number"].lower().startswith("obstacle"):
            continue
        rec["lat_wgs"] = _parse_dms(rec.pop("lat_dms"))
        rec["lon_wgs"] = _parse_dms(rec.pop("lon_dms"))
        rows.append(rec)
    return pd.DataFrame(rows)


def fetch(airport_cfg: dict, *, window: str, out_dir: Path,
          radius_nm: float = 50.0) -> FetchResult:
    icao = airport_cfg["icao"]
    frame = AirportFrame.from_cfg(airport_cfg)

    cache_dir = path_utils.CACHE / "faa_dof"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "DAILY_DOF.ZIP"
    csv_path = cache_dir / "DAILY_DOF.CSV"

    # Download attempt — try ZIP first, then CSV
    df_raw: pd.DataFrame | None = None
    last_err: Exception | None = None
    if not (zip_path.exists() or csv_path.exists()):
        try:
            download_to(DOF_PRIMARY, zip_path, timeout=180)
        except Exception as e:
            last_err = e
            logger.warning("DOF ZIP download failed (%s); trying CSV", e)
            try:
                download_to(DOF_BACKUP, csv_path, timeout=180)
            except Exception as e2:
                last_err = e2

    text = None
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as zf:
                name = next((n for n in zf.namelist() if n.upper().endswith((".DAT", ".TXT", ".CSV"))),
                            zf.namelist()[0])
                with zf.open(name) as f:
                    text = f.read().decode("latin-1")
        except Exception as e:
            last_err = e
    if text is None and csv_path.exists():
        text = csv_path.read_text(encoding="latin-1")
    if text is None:
        raise RuntimeError(
            f"Could not obtain DOF data from {DOF_PRIMARY} or {DOF_BACKUP}: {last_err}"
        )

    # Decide format by header
    head = text[:2048]
    if "," in head.splitlines()[0] and head.splitlines()[0].count(",") > 5:
        df_raw = _parse_csv(text)
    else:
        df_raw = _parse_fixed_width(text)

    if "lat_wgs" not in df_raw.columns:
        # CSV path: convert if dec strings
        df_raw["lat_wgs"] = pd.to_numeric(df_raw.get("lat_dms", df_raw.get("LATITUDE", pd.Series(dtype=float))),
                                          errors="coerce")
        df_raw["lon_wgs"] = pd.to_numeric(df_raw.get("lon_dms", df_raw.get("LONGITUDE", pd.Series(dtype=float))),
                                          errors="coerce")

    df_raw = df_raw.dropna(subset=["lat_wgs", "lon_wgs"]).reset_index(drop=True)

    # Filter to within radius_nm
    nms = np.array([
        great_circle_nm(lon, lat, frame.lon0, frame.lat0)
        for lon, lat in zip(df_raw["lon_wgs"], df_raw["lat_wgs"])
    ])
    keep = nms <= radius_nm
    df = df_raw.loc[keep].copy()
    df["within_nm"] = nms[keep].astype(np.float32)

    if len(df) == 0:
        raise RuntimeError(
            f"DOF filter returned zero records within {radius_nm} NM of "
            f"{icao} ARP ({frame.lat0}, {frame.lon0}); "
            "the source likely changed format. Inspect "
            f"{cache_dir} manually."
        )

    # Project to ENU
    x_m, y_m = frame.wgs_to_enu(df["lon_wgs"].values, df["lat_wgs"].values)
    df["x_m"] = x_m.astype(np.float64)
    df["y_m"] = y_m.astype(np.float64)

    # Numeric heights
    df["agl_ft"] = pd.to_numeric(df.get("agl_ft"), errors="coerce").astype("Float32")
    df["msl_ft"] = pd.to_numeric(df.get("msl_ft"), errors="coerce").astype("Float32")
    df["agl_m"] = (df["agl_ft"] * 0.3048).astype("Float32")
    df["msl_m"] = (df["msl_ft"] * 0.3048).astype("Float32")
    df["accuracy_h_ft"] = df.get("accuracy_h", "").map(HORIZ_ACCURACY_FT).astype("Float32")
    df["accuracy_v_ft"] = df.get("accuracy_v", "").map(VERT_ACCURACY_FT).astype("Float32")

    cols = ["oas_number", "obstacle_type", "lat_wgs", "lon_wgs", "x_m", "y_m",
            "agl_ft", "msl_ft", "agl_m", "msl_m", "accuracy_h_ft", "accuracy_v_ft",
            "marked", "lighting", "within_nm"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "obstacles.parquet"
    df.to_parquet(parquet_path, index=False)

    io_utils.write_manifest(
        parquet_path,
        source="faa_dof",
        source_url=SOURCE_URL,
        params={"airport": icao, "window": window, "radius_nm": radius_nm,
                "cache_zip": str(zip_path) if zip_path.exists() else None,
                "cache_csv": str(csv_path) if csv_path.exists() else None},
        extra={"obstacle_count": int(len(df))},
    )
    logger.info("FAA DOF: %d obstacles within %.0f NM of %s",
                len(df), radius_nm, icao)
    return FetchResult(name="faa_dof", status="ok",
                       files=[parquet_path.name],
                       extra={"obstacle_count": int(len(df))})
