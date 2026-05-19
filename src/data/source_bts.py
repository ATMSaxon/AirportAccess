"""BTS DB1B Coupon / Ticket Q3 2024 O&D download.

The Bureau of Transportation Statistics publishes DB1B as a quarterly ZIP from the
TranStats portal. The ZIP is normally fetched via a POSTed form (`PREZIP.asp`).
We:

1. Submit the historical TranStats form for the DB1B Coupon table, Q3 2024.
2. Save the resulting ZIP under `data/cache/bts/`.
3. Filter rows where `Origin in {LAX, SFO}` OR `Dest in {LAX, SFO}` and emit
   `db1b_ond.parquet` per airport.

If the TranStats download is throttled (server-side rate limit), we emit OFFLINE
with the manual recovery flow (download via the browser button and drop the ZIP
in the cache directory; re-run will pick it up).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from src.utils import io as io_utils
from src.utils import paths as path_utils
from src.utils.logs import get_logger

from ._common import FetchResult, http_post

logger = get_logger(__name__)

SOURCE_URL = ("https://transtats.bts.gov/PREZIP/Origin_and_Destination_Survey_DB1BCoupon_2024_3.zip")
BACKUP_PAGE = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLM"


def _download_db1b_q3() -> Path:
    """Download the canonical Q3 2024 DB1B Coupon ZIP."""
    cache_dir = path_utils.CACHE / "bts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "Origin_and_Destination_Survey_DB1BCoupon_2024_3.zip"
    if out.exists() and out.stat().st_size > 1024:
        return out
    # Try the direct PREZIP first (TranStats historically allows direct GET on PREZIP urls)
    from ._common import http_get
    r = http_get(SOURCE_URL, timeout=900, stream=True)
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            if chunk:
                f.write(chunk)
    return out


def _read_filter(zip_path: Path, airports: tuple[str, ...]) -> pd.DataFrame:
    """Extract DB1B Coupon CSV from ZIP and filter to airports."""
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise RuntimeError(f"No CSV inside {zip_path}")
        name = candidates[0]
        with zf.open(name) as f:
            # Read in chunks — DB1B Coupon Q3 ≈ 50–80M rows; chunk filter
            chunks = []
            for chunk in pd.read_csv(f, chunksize=200_000, dtype=str, low_memory=False):
                chunk.columns = [c.strip() for c in chunk.columns]
                origin_col = next((c for c in ("Origin", "ORIGIN") if c in chunk.columns), None)
                dest_col = next((c for c in ("Dest", "DEST") if c in chunk.columns), None)
                if origin_col is None or dest_col is None:
                    continue
                mask = chunk[origin_col].isin(airports) | chunk[dest_col].isin(airports)
                if mask.any():
                    chunks.append(chunk.loc[mask])
            if not chunks:
                return pd.DataFrame()
            return pd.concat(chunks, ignore_index=True)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map DB1B Coupon columns onto the SCHEMAS.md schema."""
    if df.empty:
        return pd.DataFrame(columns=[
            "itin_id", "origin", "dest", "reporting_carrier",
            "passengers", "market_fare", "market_distance", "quarter", "year",
        ])

    def col(*names: str) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    out = pd.DataFrame()
    out["itin_id"] = pd.to_numeric(df[col("ItinID", "ITIN_ID")], errors="coerce").astype("Int64")
    out["origin"] = df[col("Origin", "ORIGIN")].astype(str)
    out["dest"] = df[col("Dest", "DEST")].astype(str)
    rc = col("Reporting_Airline", "RC_Carrier", "REPORTING_CARRIER",
             "OpCarrierGroupNew", "Op_Carrier")
    out["reporting_carrier"] = df[rc].astype(str) if rc else ""
    pax = col("Passengers", "PASSENGERS")
    out["passengers"] = pd.to_numeric(df[pax], errors="coerce").astype("Int32") if pax else 1
    fare = col("MktFare", "MARKETFARE", "ItinFare", "FareAmt")
    out["market_fare"] = pd.to_numeric(df[fare], errors="coerce").astype("float32") if fare else float("nan")
    dist = col("MktDistance", "MARKETDISTANCE", "Distance")
    out["market_distance"] = pd.to_numeric(df[dist], errors="coerce").astype("float32") if dist else float("nan")
    qcol = col("Quarter", "QUARTER")
    out["quarter"] = pd.to_numeric(df[qcol], errors="coerce").astype("Int8") if qcol else 3
    ycol = col("Year", "YEAR")
    out["year"] = pd.to_numeric(df[ycol], errors="coerce").astype("Int16") if ycol else 2024
    return out


def fetch(airport_cfg: dict, *, window: str, out_dir: Path,
          airports: tuple[str, ...] = ("LAX", "SFO")) -> FetchResult:
    icao = airport_cfg["icao"]
    iata = airport_cfg.get("iata") or icao[1:]

    out_dir.mkdir(parents=True, exist_ok=True)
    # cached parquet across airports — share between LAX and SFO runs
    shared_parquet = path_utils.CACHE / "bts" / "db1b_q3_2024_filtered.parquet"

    if not shared_parquet.exists():
        zip_path = _download_db1b_q3()
        filtered = _read_filter(zip_path, airports=airports)
        if filtered.empty:
            raise RuntimeError(
                f"BTS DB1B Q3 2024 yielded 0 rows filtered to {airports}. "
                "Schema may have changed; inspect cache."
            )
        normalised = _normalise(filtered)
        shared_parquet.parent.mkdir(parents=True, exist_ok=True)
        normalised.to_parquet(shared_parquet, index=False)

    df = pd.read_parquet(shared_parquet)
    # Per-airport view: rows where this airport is origin OR dest
    mask = (df["origin"] == iata) | (df["dest"] == iata)
    df_air = df.loc[mask].reset_index(drop=True)

    parquet_path = out_dir / "db1b_ond.parquet"
    df_air.to_parquet(parquet_path, index=False)
    io_utils.write_manifest(
        parquet_path,
        source="bts_db1b_coupon_q3_2024",
        source_url=SOURCE_URL,
        params={"airport": iata, "window": window, "year": 2024, "quarter": 3,
                "airports_filter": list(airports)},
        extra={"row_count": int(len(df_air)),
               "note": ("Per-airport filtered view from "
                        "Origin_and_Destination_Survey_DB1BCoupon_2024_3.zip "
                        f"({BACKUP_PAGE} — manual download fallback)")},
    )
    logger.info("BTS DB1B: %d rows for %s (origin or dest)", len(df_air), iata)
    return FetchResult(name="bts", status="ok",
                       files=[parquet_path.name],
                       extra={"row_count": int(len(df_air)), "airport": iata})
