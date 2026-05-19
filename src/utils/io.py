"""I/O conventions: manifests, summaries, JSON safe-dump."""
from __future__ import annotations
import json
import gzip
import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _json_default(o: Any):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def read_json(path: Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def write_jsonl_gz(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def file_sha256(path: Path, block_size: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(__file__), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def write_manifest(data_path: Path, *, source: str, source_url: str = "",
                   params: dict | None = None, extra: dict | None = None) -> Path:
    """Write `<data_path>_manifest.json` next to a data artefact."""
    data_path = Path(data_path)
    manifest_path = data_path.with_name(data_path.name + "_manifest.json")
    sha = file_sha256(data_path) if data_path.exists() else None
    size = data_path.stat().st_size if data_path.exists() else 0
    manifest = {
        "data_file": str(data_path),
        "source": source,
        "source_url": source_url,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": sha,
        "size_bytes": size,
        "params": params or {},
        "git_commit": git_commit(),
    }
    if extra:
        manifest.update(extra)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    return manifest_path


def write_summary(run_dir: Path, summary: dict) -> Path:
    """`results/<run-id>/summary.json` with environment + git info."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "summary.json"
    base = {
        "git_commit": git_commit(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    base.update(summary)
    with out.open("w") as f:
        json.dump(base, f, indent=2, default=_json_default)
    return out
