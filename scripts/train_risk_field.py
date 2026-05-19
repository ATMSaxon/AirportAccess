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


REMOTE_HOST = "featurize@workspace.featurize.cn"
REMOTE_PORT = 27749
REMOTE_DIR = "/home/featurize/work/airportaccess"
PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple/"


def _run_remote(args: argparse.Namespace) -> int:
    """rsync this repo to Featurize and re-run the same command server-side."""
    sshpass = os.environ.get("SSHPASS") or os.environ.get("FEATURIZE_PASSWORD")
    if not sshpass:
        raise SystemExit(
            "remote run requires SSHPASS (or FEATURIZE_PASSWORD) env var")
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

    ssh_opts = ["-p", str(REMOTE_PORT), "-o", "StrictHostKeyChecking=accept-new",
                "-o", "UserKnownHostsFile=/dev/null"]
    rsync_ssh = f"sshpass -e ssh {' '.join(shlex.quote(o) for o in ssh_opts)}"
    rsync_cmd = [
        "sshpass", "-e", "rsync", "-az", "--delete",
        "--exclude", ".git", "--exclude", "data/raw", "--exclude", ".venv",
        "--exclude", "__pycache__", "--exclude", "*.zarr",
        "-e", rsync_ssh,
        f"{ROOT}/", f"{REMOTE_HOST}:{REMOTE_DIR}/",
    ]
    logger.info("rsync to Featurize: %s", " ".join(shlex.quote(t) for t in rsync_cmd))
    env = os.environ.copy(); env["SSHPASS"] = sshpass
    res = subprocess.run(rsync_cmd, env=env)
    if res.returncode != 0:
        return res.returncode

    pip_install = (
        f"pip install -i {PIP_MIRROR} -q "
        "numpy pandas pyarrow scikit-learn xgboost torch zarr pyyaml pyproj"
    )
    remote_full = (
        f"cd {REMOTE_DIR} && "
        f"{pip_install} && "
        f"PYTHONPATH={REMOTE_DIR} {remote_cmd}"
    )
    ssh_cmd = ["sshpass", "-e", "ssh"] + ssh_opts + [REMOTE_HOST, remote_full]
    logger.info("ssh exec on Featurize: %s", remote_cmd)
    res = subprocess.run(ssh_cmd, env=env)
    if res.returncode != 0:
        return res.returncode

    # Pull results back.
    pull = [
        "sshpass", "-e", "rsync", "-az",
        "-e", rsync_ssh,
        f"{REMOTE_HOST}:{REMOTE_DIR}/results/", f"{ROOT}/results/",
        f"{REMOTE_HOST}:{REMOTE_DIR}/models/", f"{ROOT}/models/",
    ]
    subprocess.run(pull, env=env)
    return 0


def _load_or_build_features(icao: str) -> pd.DataFrame:
    """Load `features.parquet` or build it from `counterfactuals.parquet`."""
    pdir = paths.PROCESSED / icao
    cf_path = pdir / "counterfactuals.parquet"
    if not cf_path.exists():
        raise SystemExit(
            f"missing {cf_path} — run scripts/sample_counterfactuals.py first")
    seg_df = pd.read_parquet(cf_path)

    feat_path = pdir / "features.parquet"
    if feat_path.exists():
        logger.info("loading cached features: %s", feat_path)
        feats = pd.read_parquet(feat_path)
    else:
        logger.info("building features from %s", cf_path)
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
