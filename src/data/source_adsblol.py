"""Acquire historical ADS-B from `adsb.lol`'s `globe_history_2024` GitHub releases.

This is the no-credential alternative to the OpenSky Trino path. adsb.lol publishes
~daily releases of `readsb` aircraft trace dumps (gzipped JSON, organised by hex24)
to the GitHub repo `adsblol/globe_history_2024`. Each release is split into
``v<YYYY.MM.DD>-planes-readsb-prod-0.tar.{aa,ab}`` parts (~2 GB combined per day).

Pipeline (per requested date):
  1. Download both tarball parts to ``data/cache/adsblol/<date>/``.
  2. Concatenate → single ``.tar``.
  3. Stream-extract trace JSONs (``traces/<hex2>/trace_full_<hex6>.json``) gzipped,
     filtering by airport ARP-centred bbox (~30 NM) using each trace's first-point
     coordinate before unpacking.
  4. Parse each trace into the canonical D5 schema (see `src/data/SCHEMAS.md`),
     project to local ENU, derive AGL, write
     ``data/processed/<ICAO>/adsb_<date>.parquet``.
  5. Write a sibling ``_manifest.json``.

On any failure we write ``<airport>/opensky_<date>.OFFLINE.json`` (we reuse the
opensky OFFLINE name so the rest of the pipeline doesn't change consumers).

Module-level constant ``SOURCE_URL`` points at the GitHub releases page so the
orchestrator's manifest writer can cite it correctly.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ._common import FetchResult, bbox_around_arp, write_offline
from src.utils import paths, io as io_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

LOG = get_logger(__name__)
SOURCE_URL = "https://github.com/adsblol/globe_history_2024"

# Tag scheme — `v<YYYY.MM.DD>-planes-readsb-prod-0`. We also try `-staging-0` as a
# fallback because some early-2024 days only published staging.
_TAG_VARIANTS = ("prod-0", "staging-0")
_RELEASE_BASE = "https://github.com/adsblol/globe_history_{year}/releases/download/"
_API_BASE = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/"

_DEFAULT_RADIUS_NM = 30.0


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def fetch(airport_cfg: dict, window: str, out_dir: Path) -> FetchResult:
    """Download ADS-B for every YYYY-MM-DD in ``window`` and return a FetchResult.

    ``window`` may be ``YYYY-MM`` (in which case we expand to all Fridays in the
    month, per the proposal's design-day rule), a single ``YYYY-MM-DD`` date, or a
    comma-separated list. Per-day failure becomes a per-day OFFLINE marker; the
    overall FetchResult is "ok" if ≥1 day was acquired.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = _expand_window(window)
    LOG.info("adsblol: %s dates to fetch for %s", len(dates), airport_cfg["icao"])

    files_written: list[Path] = []
    offline_days: list[str] = []
    frame = AirportFrame.from_cfg(airport_cfg)
    bbox = _wgs_bbox_from_arp(frame, _DEFAULT_RADIUS_NM)
    arp_elev_m = float(airport_cfg["arp"]["elev_m"])

    for date in dates:
        out_path = out_dir / f"adsb_{date}.parquet"
        if out_path.exists():
            LOG.info("adsblol: %s already exists, skipping", out_path.name)
            files_written.append(out_path)
            continue
        try:
            tar_path = _ensure_tar(date)
            df = _extract_and_filter(tar_path, bbox, frame, arp_elev_m, date)
            if df.empty:
                LOG.warning("adsblol: 0 rows after bbox filter for %s", date)
                offline_days.append(date)
                write_offline(
                    f"opensky_{date}",
                    out_dir,
                    error="0 rows after bbox filter",
                    recovery=[
                        f"Inspect the cached tar at {tar_path}",
                        "Verify airport ARP / radius_nm in airport_cfg.",
                        "Re-run scripts/acquire_all.py --only adsblol.",
                    ],
                    source_url=SOURCE_URL,
                )
                continue
            df.to_parquet(out_path, compression="zstd", index=False)
            io_utils.write_manifest(
                out_path,
                source="adsb.lol globe_history (readsb trace dump)",
                source_url=SOURCE_URL,
                params={"date": date, "radius_nm": _DEFAULT_RADIUS_NM,
                        "airport": airport_cfg["icao"], "bbox_wgs": bbox},
                extra={"n_rows": int(len(df)), "n_icao24": int(df["icao24"].nunique())},
            )
            files_written.append(out_path)
            LOG.info("adsblol: %s → %s rows from %s aircraft (%.1f MB)",
                     out_path.name, len(df), df["icao24"].nunique(),
                     out_path.stat().st_size / 1e6)
        except Exception as e:  # noqa: BLE001 — per-day graceful degrade
            LOG.exception("adsblol: failed for %s", date)
            offline_days.append(date)
            write_offline(
                f"opensky_{date}",
                out_dir,
                error=f"{type(e).__name__}: {e}",
                recovery=[
                    f"Tag https://github.com/adsblol/globe_history_2024/releases/tag/v{date.replace('-', '.')}-planes-readsb-prod-0 might be missing — try the staging variant or a different day.",
                    f"Network: large download (~2.5 GB/day) — re-run with stable connection.",
                ],
                source_url=SOURCE_URL,
            )

    status = "ok" if files_written else "offline"
    return FetchResult(
        name="adsblol",
        status=status,
        files=[p.name for p in files_written],
        extra={"window": window, "radius_nm": _DEFAULT_RADIUS_NM,
               "dates_ok": [d for d in dates if d not in offline_days],
               "dates_offline": offline_days},
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((requests.RequestException, OSError)),
    reraise=True,
)
def _download_part(url: str, dest: Path) -> None:
    """Download a single .tar.aa / .tar.ab part with resumable streaming."""
    headers = {}
    mode = "wb"
    if dest.exists():
        # Resume from where we left off.
        headers["Range"] = f"bytes={dest.stat().st_size}-"
        mode = "ab"
    LOG.info("adsblol: GET %s (resume=%s)", url, headers.get("Range", "no"))
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:  # already complete
            return
        r.raise_for_status()
        with dest.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _ensure_tar(date: str) -> Path:
    """Return path to a concatenated .tar for the given ``YYYY-MM-DD`` date.

    Idempotent: re-uses cached .tar.aa / .tar.ab parts; only concatenates once.
    Avoids the GitHub API when the parts are already on disk (unauthenticated
    GitHub API is heavily rate-limited).
    """
    year = date.split("-")[0]
    tag_date = date.replace("-", ".")
    cache_dir = paths.CACHE / "adsblol" / date
    cache_dir.mkdir(parents=True, exist_ok=True)
    full_tar = cache_dir / f"v{tag_date}-planes-readsb.tar"
    if full_tar.exists() and full_tar.stat().st_size > 100 * 1024 * 1024:
        LOG.info("adsblol: reusing %s (%.1f GB)", full_tar.name,
                 full_tar.stat().st_size / 1e9)
        return full_tar

    # First, see if .tar.aa / .tar.ab / .tar are already on disk (downloaded
    # out-of-band, e.g. via parallel curl on the GPU box). If so, concatenate
    # without ever calling the GitHub API.
    cached_parts = sorted(
        p for p in cache_dir.iterdir()
        if p.name.startswith(f"v{tag_date}-planes-readsb-") and (
            p.suffix in (".tar",) or p.name.endswith(".tar.aa") or p.name.endswith(".tar.ab")
            or p.name.endswith(".tar.ac") or p.name.endswith(".tar.ad")
        )
    )
    if cached_parts and sum(p.stat().st_size for p in cached_parts) > 100 * 1024 * 1024:
        LOG.info("adsblol: found %d cached part(s), concatenating without GitHub API",
                 len(cached_parts))
        with full_tar.open("wb") as outf:
            for p in cached_parts:
                with p.open("rb") as inf:
                    while True:
                        chunk = inf.read(1 << 20)
                        if not chunk:
                            break
                        outf.write(chunk)
        LOG.info("adsblol: concatenated %d parts → %s (%.1f GB)",
                 len(cached_parts), full_tar.name, full_tar.stat().st_size / 1e9)
        return full_tar

    last_err: Exception | None = None
    for variant in _TAG_VARIANTS:
        tag = f"v{tag_date}-planes-readsb-{variant}"
        try:
            assets = _list_assets(year, tag)
        except Exception as e:
            last_err = e
            continue
        if not assets:
            continue
        # Download each .tar.aa / .tar.ab / .tar part.
        parts: list[Path] = []
        for name, url, size in assets:
            local = cache_dir / name
            _download_part(url, local)
            parts.append(local)
        # Concatenate all parts in alpha order (.tar, .tar.aa, .tar.ab, ...).
        parts_sorted = sorted(parts)
        with full_tar.open("wb") as outf:
            for p in parts_sorted:
                with p.open("rb") as inf:
                    while True:
                        chunk = inf.read(1 << 20)
                        if not chunk:
                            break
                        outf.write(chunk)
        LOG.info("adsblol: concatenated %s parts → %s (%.1f GB)",
                 len(parts_sorted), full_tar.name, full_tar.stat().st_size / 1e9)
        return full_tar

    raise RuntimeError(
        f"adsblol: no release found for {date} (tried {_TAG_VARIANTS}); "
        f"last error: {last_err}"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20),
       retry=retry_if_exception_type(requests.RequestException), reraise=True)
def _list_assets(year: str, tag: str) -> list[tuple[str, str, int]]:
    """Return [(name, url, size), …] for assets of a given GitHub release tag."""
    url = _API_BASE.format(year=year) + tag
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    data = r.json()
    return [(a["name"], a["browser_download_url"], int(a["size"]))
            for a in data.get("assets", [])]


# ---------------------------------------------------------------------------
# Extract + filter
# ---------------------------------------------------------------------------


def _wgs_bbox_from_arp(frame: AirportFrame, radius_nm: float) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) covering ``radius_nm`` around ARP."""
    radius_m = float(radius_nm) * 1852.0
    # Probe four cardinal points to get the bbox precisely.
    xs = np.array([-radius_m, +radius_m, 0.0, 0.0])
    ys = np.array([0.0, 0.0, -radius_m, +radius_m])
    lons, lats = frame.enu_to_wgs(xs, ys)
    return (float(lons.min()), float(lats.min()),
            float(lons.max()), float(lats.max()))


def _extract_and_filter(tar_path: Path, bbox: tuple[float, float, float, float],
                        frame: AirportFrame, arp_elev_m: float, date: str) -> pd.DataFrame:
    """Stream the tar, decode every ``trace_full_*.json.gz``, filter by bbox.

    Returns a DataFrame with the D5 OpenSky schema (see src/data/SCHEMAS.md §D5):

    time_utc | icao24 | callsign | lon_wgs | lat_wgs | baro_alt_m | geo_alt_m |
    velocity_ms | track_deg | vert_rate_ms | on_ground | x_m | y_m | z_msl_m | z_agl_m
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    rows: list[dict] = []
    n_traces = 0
    n_kept_traces = 0
    truncation_warning = ""

    # ``mode="r|"`` is streaming-only (no random access). It is memory-friendly for
    # the ~2.5 GB tars but fails hard if the archive is truncated mid-stream. We
    # tolerate per-member gzip errors INSIDE the loop, plus stream-level
    # truncation by wrapping the whole iteration in a try/except. Whatever we
    # processed before the truncation is salvaged.
    try:
        tar = tarfile.open(tar_path, mode="r|")
    except (OSError, tarfile.TarError) as e:
        raise RuntimeError(
            f"adsblol: tarfile.open failed on {tar_path}: {type(e).__name__}: {e}. "
            f"Likely a truncated download — delete the cached tar and re-download."
        )
    try:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            # Only consider per-aircraft `trace_full` files. readsb also publishes
            # `trace_recent_*` (last hour) — we keep both because some files only
            # have one variant.
            if "trace_full_" not in name and "trace_recent_" not in name:
                continue
            if not name.endswith(".json"):
                continue
            n_traces += 1
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            raw = fileobj.read()
            # readsb writes gzipped traces (file extension is .json but content is gz).
            # A handful of blobs per tar are truncated (EOFError) or have CRC errors
            # (gzip.BadGzipFile / zlib.error → OSError) — skip them silently so one
            # bad row doesn't abort the ~65k-trace scan.
            if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
                try:
                    raw = gzip.decompress(raw)
                except (OSError, EOFError):
                    continue
            try:
                trace = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            icao24 = str(trace.get("icao") or trace.get("hex") or "").lower()
            if not icao24:
                continue
            callsign = (trace.get("r") or trace.get("flight") or "").strip()
            # Fast bbox pre-filter using the trace's mean position.
            trace_pts = trace.get("trace") or []
            if not trace_pts:
                continue
            keep = False
            for pt in trace_pts:
                # trace point format (readsb v3): [t_offset_s, lat, lon, alt_ft, gs, track, flags, …]
                if len(pt) < 4:
                    continue
                lat = pt[1]
                lon = pt[2]
                if lat is None or lon is None:
                    continue
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    keep = True
                    break
            if not keep:
                continue
            n_kept_traces += 1
            t0_ts = trace.get("timestamp")  # epoch seconds
            for pt in trace_pts:
                if len(pt) < 5:
                    continue
                t_off = pt[0]
                lat = pt[1]
                lon = pt[2]
                alt = pt[3]
                gs = pt[4] if len(pt) > 4 else None
                trk = pt[5] if len(pt) > 5 else None
                flags = pt[6] if len(pt) > 6 else 0
                vr = pt[7] if len(pt) > 7 else None
                if lat is None or lon is None or not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    continue
                on_ground = (alt == "ground")
                alt_ft = None if (alt is None or alt == "ground") else float(alt)
                t_utc = (t0_ts or 0) + (t_off or 0)
                rows.append({
                    "time_utc": pd.Timestamp(t_utc, unit="s", tz="UTC"),
                    "icao24": icao24,
                    "callsign": callsign,
                    "lon_wgs": float(lon),
                    "lat_wgs": float(lat),
                    "baro_alt_m": (alt_ft * 0.3048) if alt_ft is not None else np.nan,
                    "geo_alt_m": np.nan,
                    "velocity_ms": (float(gs) * 0.514444) if gs is not None else np.nan,
                    "track_deg": float(trk) if trk is not None else np.nan,
                    "vert_rate_ms": (float(vr) * 0.00508) if vr is not None else np.nan,
                    "on_ground": bool(on_ground),
                })
    except (EOFError, OSError, tarfile.TarError) as e:
        # Truncated tar — salvage whatever we processed.
        truncation_warning = f"{type(e).__name__}: {e}"
        LOG.warning("adsblol: tar stream ended early at %s traces (%s): %s",
                    n_traces, tar_path.name, truncation_warning)
    finally:
        try:
            tar.close()
        except Exception:
            pass
    LOG.info("adsblol: %s scanned %s traces, kept %s in bbox; %s state-vectors%s",
             tar_path.name, n_traces, n_kept_traces, len(rows),
             f" [TRUNCATED: {truncation_warning}]" if truncation_warning else "")
    if not rows:
        return pd.DataFrame(columns=["time_utc", "icao24", "callsign", "lon_wgs", "lat_wgs",
                                     "baro_alt_m", "geo_alt_m", "velocity_ms", "track_deg",
                                     "vert_rate_ms", "on_ground", "x_m", "y_m",
                                     "z_msl_m", "z_agl_m"])

    df = pd.DataFrame(rows).sort_values(["icao24", "time_utc"]).reset_index(drop=True)
    # Project lon/lat → ENU around the airport ARP.
    xs, ys = frame.wgs_to_enu(df["lon_wgs"].to_numpy(), df["lat_wgs"].to_numpy())
    df["x_m"] = xs.astype(np.float32)
    df["y_m"] = ys.astype(np.float32)
    # Best-altitude MSL.
    df["z_msl_m"] = df["baro_alt_m"].astype(np.float32)
    df["z_agl_m"] = (df["z_msl_m"] - arp_elev_m).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Window expansion
# ---------------------------------------------------------------------------


def _expand_window(window: str) -> list[str]:
    """Accepts ``YYYY-MM`` → Fridays in that month, single ``YYYY-MM-DD``, or comma list."""
    window = window.strip()
    if "," in window:
        return [d.strip() for d in window.split(",") if d.strip()]
    if len(window) == 10 and window.count("-") == 2:
        return [window]
    if len(window) == 7 and window.count("-") == 1:
        year, month = (int(p) for p in window.split("-"))
        from calendar import Calendar
        cal = Calendar(firstweekday=0)
        return [f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
                for d in cal.itermonthdates(year, month)
                if d.month == month and d.weekday() == 4]  # Friday
    raise ValueError(f"unrecognised window spec: {window!r}")
