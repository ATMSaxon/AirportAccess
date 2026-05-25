#!/usr/bin/env python
"""Train a risk-field learner and (for the primary XGBoost) export the
voxel-grid risk field.

Examples:
    python scripts/train_risk_field.py --model xgb --airport KLAX
    python scripts/train_risk_field.py --model mlp --airport KLAX --gpu remote

The `--gpu remote` path rsyncs the project to the Featurize Blackwell box and
runs the same command there. SSH config:

    host:   workspace.featurize.cn
    user:   featurize
    port:   27749
    auth:   SSHPASS env var (sshpass -e)
    pip:    Aliyun mirror

The remote command is `python scripts/train_risk_field.py --model mlp --airport
<ICAO>` (i.e. without `--gpu remote`).
"""
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml import counterfactual as cf  # noqa: E402
from src.ml.features import extract_features, merge_features_labels  # noqa: E402
from src.ml.risk_field import train, save_model, predict_risk_grid  # noqa: E402
from src.ml._geom import AirportGeom  # noqa: E402
from src.utils import config as cfg_io  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils.grid import VoxelGrid  # noqa: E402
from src.utils.io import write_json, write_manifest  # noqa: E402
from src.utils.logs import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_featurize.sh"


def _run_remote(args: argparse.Namespace) -> int:
    """Delegate the remote run to ``scripts/deploy_featurize.sh full <CMD>``.

    The script handles rsync push, Aliyun pip install, ssh exec; we then call
    ``deploy_featurize.sh pull`` to bring back results/ + models/.

    Requires ``FEATURIZE_PASS`` (preferred) or ``SSHPASS`` env var.
    """
    pw = os.environ.get("FEATURIZE_PASS") or os.environ.get("SSHPASS") \
        or os.environ.get("FEATURIZE_PASSWORD")
    if not pw:
        raise SystemExit(
            "remote run requires FEATURIZE_PASS (or SSHPASS) env var")
    if not DEPLOY_SCRIPT.exists():
        raise SystemExit(f"missing deploy script: {DEPLOY_SCRIPT}")

    # Re-build the command line minus `--gpu remote`.
    cmd_tokens = ["python", "scripts/train_risk_field.py",
                  "--model", args.model, "--airport", args.airport]
    if args.no_grid:
        cmd_tokens.append("--no-grid")
    if args.debug:
        cmd_tokens.append("--debug")
    if args.seed != 42:
        cmd_tokens += ["--seed", str(args.seed)]
    remote_cmd = " ".join(shlex.quote(t) for t in cmd_tokens)

    env = os.environ.copy(); env["FEATURIZE_PASS"] = pw
    logger.info("Featurize full: %s", remote_cmd)
    res = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "full", remote_cmd],
        env=env, cwd=str(ROOT))
    if res.returncode != 0:
        return res.returncode

    logger.info("Featurize pull → results/ + models/")
    res = subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), "pull"], env=env, cwd=str(ROOT))
    return res.returncode


def _load_counterfactuals(icao: str) -> pd.DataFrame:
    """Load and concat all counterfactuals for an airport.

    Preference order:
      1. Per-day files ``counterfactuals_<date>.parquet`` (multi-day holdout).
      2. Single combined ``counterfactuals.parquet`` (legacy / single-day).

    Returns one DataFrame with a ``sample_date`` column populated for every
    row (back-filled from filename if missing).
    """
    pdir = paths.PROCESSED / icao
    per_day = sorted(pdir.glob("counterfactuals_*.parquet"))
    # Strip out the manifest-sibling pattern just in case (glob already filters by .parquet).
    if per_day:
        frames = []
        for p in per_day:
            df = pd.read_parquet(p)
            # Back-fill sample_date from filename if older parquets are missing it.
            if "sample_date" not in df.columns or df["sample_date"].isna().any():
                date_str = p.stem.replace("counterfactuals_", "")
                if "sample_date" not in df.columns:
                    df["sample_date"] = date_str
                else:
                    df["sample_date"] = df["sample_date"].fillna(date_str)
            frames.append(df)
            logger.info("counterfactuals: %s (%d rows, %d conflicts)",
                        p.name, len(df), int(df["conflict"].sum()))
        return pd.concat(frames, ignore_index=True)
    cf_path = pdir / "counterfactuals.parquet"
    if not cf_path.exists():
        raise SystemExit(
            f"missing counterfactuals under {pdir} — run "
            f"scripts/sample_counterfactuals.py first")
    seg_df = pd.read_parquet(cf_path)
    logger.info("counterfactuals: %s (%d rows, legacy single-file)",
                cf_path.name, len(seg_df))
    return seg_df


def _load_or_build_features(icao: str) -> pd.DataFrame:
    """Load `features.parquet` (cached) or build it from counterfactuals."""
    pdir = paths.PROCESSED / icao
    seg_df = _load_counterfactuals(icao)

    feat_path = pdir / "features.parquet"
    feats: pd.DataFrame
    rebuild = True
    if feat_path.exists():
        cached = pd.read_parquet(feat_path)
        # Cache is only safe to reuse when it covers exactly the same seg_ids.
        if set(cached["seg_id"]) == set(seg_df["seg_id"]):
            logger.info("loading cached features: %s (%d rows)",
                        feat_path, len(cached))
            feats = cached
            rebuild = False
        else:
            logger.info("features.parquet stale (%d cached vs %d segs) — rebuilding",
                        len(cached), len(seg_df))
    if rebuild:
        logger.info("building features for %s (%d segs)", icao, len(seg_df))
        geom = AirportGeom.from_icao(icao)
        adsb_paths = sorted(pdir.glob("adsb_*.parquet"))
        adsb_df = pd.concat([pd.read_parquet(p) for p in adsb_paths],
                            ignore_index=True) if adsb_paths else None
        metar_path = pdir / "metar.parquet"
        metar_df = pd.read_parquet(metar_path) if metar_path.exists() else None
        feats = extract_features(seg_df, geom, adsb_df, metar_df)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(feat_path, index=False)
        write_manifest(feat_path, source="ml.features.extract_features",
                       params={"icao": icao, "n": len(feats)})
    return merge_features_labels(feats, seg_df, label_col="conflict")


def _build_time_slices(rc_df: pd.DataFrame, *, every_minutes: int = 60) -> list[pd.Timestamp]:
    rc_df = rc_df.copy()
    rc_df["time_utc"] = pd.to_datetime(rc_df["time_utc"], utc=True)
    t_min = rc_df["time_utc"].min().floor(f"{every_minutes}min")
    t_max = rc_df["time_utc"].max().ceil(f"{every_minutes}min")
    return list(pd.date_range(t_min, t_max, freq=f"{every_minutes}min"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["lr", "rf", "xgb", "mlp"], required=True)
    ap.add_argument("--airport", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", choices=["local", "remote"], default="local")
    ap.add_argument("--no-grid", action="store_true",
                    help="skip the voxel-grid risk export")
    ap.add_argument("--grid-every-minutes", type=int, default=60)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    setup_logging("DEBUG" if args.debug else "INFO")

    if args.gpu == "remote":
        rc = _run_remote(args)
        sys.exit(rc)

    fl = _load_or_build_features(args.airport)
    logger.info("features+labels: %d rows, %d cols, %d conflicts",
                len(fl), fl.shape[1], int(fl["conflict"].sum()))

    result = train(model_name=args.model, features_labels=fl, seed=args.seed)
    logger.info("metrics: %s", result.metrics)

    # Persist model + metrics.
    model_dir = paths.MODELS / "risk" / args.airport
    model_dir.mkdir(parents=True, exist_ok=True)
    model_ext = ".pt" if args.model == "mlp" else ".pkl"
    save_model(result, model_dir / f"{args.model}{model_ext}")
    metrics_dir = paths.RESULTS / "risk" / args.airport
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / f"{args.model}.json", {
        **result.metrics,
        "model_name": args.model,
        "feature_cols": result.feature_cols,
    })
    logger.info("wrote metrics → %s", metrics_dir / f"{args.model}.json")

    # Risk-grid export for the primary model.
    if args.no_grid or args.model != "xgb":
        return
    pdir = paths.PROCESSED / args.airport
    rc_path_candidates = sorted(pdir.glob("runway_config_*.parquet"))
    if not rc_path_candidates:
        logger.warning("no runway_config_*.parquet found — skipping risk-grid export")
        return
    rc_df = pd.concat([pd.read_parquet(p) for p in rc_path_candidates],
                      ignore_index=True)
    metar_path = pdir / "metar.parquet"
    metar_df = pd.read_parquet(metar_path) if metar_path.exists() else None
    adsb_paths = sorted(pdir.glob("adsb_*.parquet"))
    adsb_df = pd.concat([pd.read_parquet(p) for p in adsb_paths],
                        ignore_index=True) if adsb_paths else None

    voxel = VoxelGrid.from_airport_cfg(cfg_io.load_airport(args.airport))
    times = _build_time_slices(rc_df, every_minutes=args.grid_every_minutes)
    out_zarr = pdir / f"risk_grid_{args.model}.zarr"
    predict_risk_grid(
        icao=args.airport,
        model=result.model,
        feature_cols=result.feature_cols,
        voxel_grid=voxel,
        time_slices=times,
        runway_config_df=rc_df,
        metar_df=metar_df,
        adsb_df=adsb_df,
        output_path=out_zarr,
    )
    logger.info("wrote risk grid → %s", out_zarr)


if __name__ == "__main__":
    main()
