"""Shared helpers for source modules: retrying HTTP, manifest/offline writing, geo helpers."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils import io as io_utils
from src.utils.crs import AirportFrame
from src.utils.logs import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT_S = 60
DEFAULT_USER_AGENT = "DREAM-eVTOL/0.1 (research; contact: shinex.zhou@gmail.com)"


@dataclass
class FetchResult:
    """Outcome of a single source fetch."""

    name: str
    status: str  # "ok" | "offline"
    files: list[str]
    error: str | None = None
    extra: dict | None = None

    def to_inventory_entry(self) -> dict:
        d = {"status": self.status, "files": self.files}
        if self.error:
            d["error"] = self.error
        if self.extra:
            d["extra"] = self.extra
        return d


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=30),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
)
def http_get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    stream: bool = False,
    auth: tuple[str, str] | None = None,
) -> requests.Response:
    """GET with exponential-backoff retry on transient HTTP errors."""
    h = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.get(
        url, params=params, headers=h, timeout=timeout, stream=stream, auth=auth
    )
    if r.status_code >= 500:
        # Will be retried by tenacity
        raise requests.HTTPError(f"{r.status_code} server error from {url}")
    return r


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=30),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
)
def http_post(
    url: str,
    *,
    data: Any = None,
    json: dict | None = None,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> requests.Response:
    h = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.post(url, data=data, json=json, headers=h, timeout=timeout)
    if r.status_code >= 500:
        raise requests.HTTPError(f"{r.status_code} server error from {url}")
    return r


def download_to(url: str, dst: Path, *, params: dict | None = None, chunk: int = 1 << 16,
                timeout: int = 120) -> Path:
    """Stream a binary download to disk."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with http_get(url, params=params, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
    return dst


# ---------------------------------------------------------------------------
# Manifest / offline helpers
# ---------------------------------------------------------------------------


def write_offline(name: str, out_dir: Path, *, error: str, recovery: list[str],
                  source_url: str = "", params: dict | None = None) -> Path:
    """Write `<name>.OFFLINE.json` documenting failure + manual recovery flow."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.OFFLINE.json"
    payload = {
        "source": name,
        "status": "offline",
        "error": str(error)[:4000],
        "source_url": source_url,
        "params": params or {},
        "recovery": recovery,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": io_utils.git_commit(),
    }
    io_utils.write_json(path, payload)
    logger.warning("source %s OFFLINE → %s", name, path)
    return path


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------


def bbox_around_arp(frame: AirportFrame, *, half_km: float) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) for a `half_km`×`half_km` square around ARP."""
    # 1 deg latitude ≈ 111.32 km
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * math.cos(math.radians(frame.lat0)))
    return (frame.lon0 - dlon, frame.lat0 - dlat,
            frame.lon0 + dlon, frame.lat0 + dlat)


def great_circle_nm(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in nautical miles."""
    R_NM = 3440.065
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_NM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Convenience: source orchestrator wrapper
# ---------------------------------------------------------------------------


def run_source(
    name: str,
    fn: Callable[..., FetchResult],
    out_dir: Path,
    *,
    offline_recovery: list[str],
    source_url: str = "",
    params: dict | None = None,
    **kwargs,
) -> FetchResult:
    """Run a source fetch with universal error handling → FetchResult."""
    t0 = time.time()
    try:
        result = fn(out_dir=out_dir, **kwargs)
        logger.info("source %s OK in %.1fs → %s", name, time.time() - t0,
                    [str(f) for f in result.files])
        return result
    except Exception as exc:  # noqa: BLE001 — universal trap by design
        offline_path = write_offline(
            name,
            out_dir,
            error=f"{exc.__class__.__name__}: {exc}",
            recovery=offline_recovery,
            source_url=source_url,
            params=params,
        )
        return FetchResult(
            name=name,
            status="offline",
            files=[offline_path.name],
            error=str(exc),
        )
