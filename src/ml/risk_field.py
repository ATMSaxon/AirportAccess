"""Risk-field learners (LR / RF / XGB / MLP) and grid-export.

Trains a binary classifier `ρ(p, t) = P(conflict | x)` and exports a per-cell
risk grid for the chosen primary model. The MLP is the GPU path; everything
else runs on CPU.

Outputs:
* `results/risk/{icao}/{model}.json` — metrics dict.
* `models/risk/{icao}/{model}.pkl` (sklearn/XGB) or `.pt` (MLP).
* `data/processed/{icao}/risk_grid_{model}.zarr` — Float16 risk on the
  VoxelGrid × time-slice axis. Schema described in INTERFACES.md.

Holdout strategy: one Friday held out for the temporal test (largest mid_t
date in the input). Calibration uses 20 % of the remainder (random split).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import json
import logging
import math
import pickle
import numpy as np
import pandas as pd

from ..utils import config as cfg_io
from ..utils import paths
from ..utils.grid import VoxelGrid
from ..utils.io import write_json, write_manifest
from ..utils.logs import get_logger
from ._geom import AirportGeom, adsb_density_box, NM_M, FT_M
from .features import NUMERIC_FEATURES, _join_metar, extract_features
from .conformal import SplitConformal, calibrate_split

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    model_name: str
    model: object                            # sklearn-like fit-predict object
    feature_cols: list[str]
    metrics: dict
    conformal: SplitConformal | None = None


def _build_model(name: str, *, seed: int = 42):
    name = name.lower()
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1, random_state=seed)),
        ])
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed,
            class_weight="balanced_subsample")
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            n_jobs=-1, eval_metric="auc", random_state=seed, tree_method="hist")
    if name == "mlp":
        return _build_torch_mlp(seed=seed)
    raise ValueError(f"unknown model {name!r}; choose lr|rf|xgb|mlp")


# ---------------------------------------------------------------------------
# PyTorch MLP — wrapped to look sklearn-ish
# ---------------------------------------------------------------------------
class TorchMLP:
    def __init__(self, *, hidden=(64, 64), epochs=30, batch_size=512,
                 lr=1e-3, device: str | None = None, seed: int = 42):
        self.hidden = tuple(hidden)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device
        self.model = None
        self.n_features_ = None

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.n_features_ = int(X.shape[1])
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        layers = []
        in_dim = self.n_features_
        for h in self.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.model = nn.Sequential(*layers).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # class-imbalance-aware pos-weight
        n_pos = max(int(y.sum()), 1)
        n_neg = max(len(y) - n_pos, 1)
        pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=dev)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        Xt = torch.from_numpy(X).to(dev)
        yt = torch.from_numpy(y).to(dev)
        n = len(X)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            ep_loss = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s+self.batch_size]
                opt.zero_grad()
                logits = self.model(Xt[idx]).squeeze(-1)
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach()) * len(idx)
            if (ep + 1) % max(1, self.epochs // 5) == 0:
                logger.info("MLP ep=%d loss=%.4f", ep + 1, ep_loss / n)
        return self

    def predict_proba(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        dev = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(dev)).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        # sklearn convention: two columns (P(class=0), P(class=1))
        return np.column_stack([1.0 - probs, probs])


def _build_torch_mlp(seed: int = 42) -> TorchMLP:
    return TorchMLP(seed=seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _holdout_temporal_split(df: pd.DataFrame, time_col: str = "mid_t_utc"
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest **calendar date** for test."""
    t = pd.to_datetime(df[time_col], utc=True)
    dates = t.dt.date
    unique_dates = sorted(pd.unique(dates))
    if len(unique_dates) < 2:
        # Fall back to a random 20 % split if we only have one day.
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(df))
        cut = int(0.8 * len(df))
        return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
    test_date = unique_dates[-1]
    is_test = (dates == test_date)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def train(*, model_name: str, features_labels: pd.DataFrame,
          label_col: str = "conflict", seed: int = 42,
          conformal_alpha: float = 0.1) -> TrainResult:
    """Train a single risk model on a (features+label) parquet.

    `features_labels` must contain `mid_t_utc`, `conflict`, and the numeric +
    one-hot config columns produced by `features.extract_features`.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

    # Choose feature columns: numeric + cfg_* one-hot.
    feature_cols = [c for c in features_labels.columns
                    if c in NUMERIC_FEATURES or c.startswith("cfg_")]
    if not feature_cols:
        raise RuntimeError("no feature columns found in features_labels")

    train_df, test_df = _holdout_temporal_split(features_labels)
    Xtr = train_df[feature_cols].to_numpy(dtype=np.float64)
    ytr = train_df[label_col].to_numpy(dtype=np.int64)
    Xte = test_df[feature_cols].to_numpy(dtype=np.float64)
    yte = test_df[label_col].to_numpy(dtype=np.int64)

    model = _build_model(model_name, seed=seed)
    if model_name.lower() == "mlp":
        model.fit(Xtr, ytr)
    else:
        model.fit(Xtr, ytr)

    p_te = model.predict_proba(Xte)[:, 1]

    metrics: dict[str, float] = {}
    if len(np.unique(yte)) > 1:
        metrics["auroc"] = float(roc_auc_score(yte, p_te))
        metrics["aupr"] = float(average_precision_score(yte, p_te))
    else:
        metrics["auroc"] = float("nan")
        metrics["aupr"] = float("nan")
        logger.warning("test set is single-class — AUROC undefined")
    metrics["logloss"] = float(log_loss(yte, np.clip(p_te, 1e-6, 1 - 1e-6),
                                        labels=[0, 1]))
    metrics["n_train"] = int(len(ytr))
    metrics["n_test"] = int(len(yte))
    metrics["pos_rate_train"] = float(np.mean(ytr))
    metrics["pos_rate_test"] = float(np.mean(yte))

    # Conformal calibration: split a slice of *train* (not the test day) and
    # evaluate empirical coverage on the test day.
    if len(np.unique(ytr)) > 1:
        rng = np.random.default_rng(seed)
        n_tr = len(Xtr)
        cal_n = max(2, int(0.2 * n_tr))
        perm = rng.permutation(n_tr)
        cal_idx = perm[:cal_n]; train_idx = perm[cal_n:]
        # Refit on the train-fraction to avoid calibration-set contamination.
        model_cal = _build_model(model_name, seed=seed)
        if model_name.lower() == "mlp":
            model_cal.fit(Xtr[train_idx], ytr[train_idx])
        else:
            model_cal.fit(Xtr[train_idx], ytr[train_idx])
        p_cal = model_cal.predict_proba(Xtr[cal_idx])[:, 1]
        y_cal = ytr[cal_idx]
        cp = SplitConformal(alpha=conformal_alpha).fit(p_cal, y_cal)
        p_te_cal = model_cal.predict_proba(Xte)[:, 1]
        lo, hi = cp.predict(p_te_cal)
        cov = float(np.mean((yte >= lo) & (yte <= hi)))
        metrics.update({
            "conformal_coverage": cov,
            "conformal_target": 1.0 - conformal_alpha,
            "conformal_width": float(np.mean(hi - lo)),
            "conformal_qhat": float(cp.qhat),
        })
    else:
        cp = None

    return TrainResult(model_name=model_name, model=model,
                       feature_cols=feature_cols, metrics=metrics,
                       conformal=cp)


# ---------------------------------------------------------------------------
# Risk-grid export
# ---------------------------------------------------------------------------
def predict_risk_grid(*, icao: str, model, feature_cols: list[str],
                      voxel_grid: VoxelGrid, time_slices: Sequence[pd.Timestamp],
                      runway_config_df: pd.DataFrame,
                      metar_df: pd.DataFrame | None,
                      adsb_df: pd.DataFrame | None,
                      output_path: Path,
                      dtype=np.float16, chunk_z: int = 8) -> Path:
    """Score `ρ(x, y, z, t)` over the airport VoxelGrid × time-slice grid
    and write to a zarr store.

    Memory-friendly: scores one (t, iz) plane at a time.
    """
    import zarr

    geom = AirportGeom.from_icao(icao)
    nx, ny, nz = voxel_grid.shape
    xs, ys, zs = voxel_grid.coords()
    nt = len(time_slices)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    root = zarr.open(str(output_path), mode="w")
    # zarr v3+ uses positional args (shape, *, chunks, dtype, ...).
    rho = root.create_array(
        "rho",
        shape=(nt, nx, ny, nz),
        chunks=(1, min(nx, 64), min(ny, 64), min(nz, chunk_z)),
        dtype=dtype,
    )
    # Sidecar coordinate arrays.
    root.create_array("x_m", shape=(nx,), dtype=np.float64, chunks=(nx,))[:] = xs
    root.create_array("y_m", shape=(ny,), dtype=np.float64, chunks=(ny,))[:] = ys
    root.create_array("z_msl_m", shape=(nz,), dtype=np.float64, chunks=(nz,))[:] = zs
    # Time slices as ISO strings (zarr does not store datetime cleanly).
    t_iso = np.array([pd.Timestamp(t).isoformat() for t in time_slices], dtype="S32")
    root.create_array("time_utc", shape=(nt,), dtype=t_iso.dtype, chunks=(nt,))[:] = t_iso
    root.attrs.update({
        "icao": icao,
        "shape_order": "t,x,y,z",
        "voxel_dx_m": voxel_grid.dx,
        "voxel_dy_m": voxel_grid.dy,
        "voxel_dz_m": voxel_grid.dz,
        "feature_cols": list(feature_cols),
    })

    # Cache per-slice active arrivals/departures + weather row.
    if not pd.api.types.is_datetime64_any_dtype(runway_config_df["time_utc"]):
        runway_config_df = runway_config_df.copy()
        runway_config_df["time_utc"] = pd.to_datetime(runway_config_df["time_utc"], utc=True)

    rc_times = runway_config_df["time_utc"].to_numpy()

    # Static per-cell features (don't depend on t).
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    static_d_runway = np.empty((nx, ny, nz), dtype=np.float32)
    for iz, z in enumerate(zs):
        for ix in range(nx):
            for iy in range(ny):
                static_d_runway[ix, iy, iz] = geom.distance_to_nearest_runway(
                    XX[ix, iy], YY[ix, iy], float(z))

    for it, t in enumerate(time_slices):
        # Match nearest runway-config slice.
        idx = int(np.argmin(np.abs(rc_times - np.datetime64(pd.Timestamp(t)))))
        rc = runway_config_df.iloc[idx]
        arr = [s for s in str(rc.get("active_arrivals", "")).split(";") if s]
        dep = [s for s in str(rc.get("active_departures", "")).split(";") if s]
        cfg_id = str(rc.get("config_id", "UNKNOWN"))

        # Weather + hour-of-day are constant over the spatial grid.
        wx = _join_metar(pd.Series([pd.Timestamp(t)]), metar_df).iloc[0]
        wind_dir_rad = math.radians(0.0 if pd.isna(wx["wind_dir_deg"]) else float(wx["wind_dir_deg"]))
        wind_dir_sin = math.sin(wind_dir_rad) if not pd.isna(wx["wind_dir_deg"]) else 0.0
        wind_dir_cos = math.cos(wind_dir_rad) if not pd.isna(wx["wind_dir_deg"]) else 0.0
        h = pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60.0
        hour_sin = math.sin(2 * math.pi * h / 24.0)
        hour_cos = math.cos(2 * math.pi * h / 24.0)

        # Build a feature row template (in feature_cols order) and stamp the
        # per-cell columns inside the inner loop.
        feat_template = {c: 0.0 for c in feature_cols}
        feat_template["wind_dir_sin"] = wind_dir_sin
        feat_template["wind_dir_cos"] = wind_dir_cos
        feat_template["wind_speed_mps"] = float(wx["wind_speed_mps"])
        feat_template["visibility_m"] = float(wx["vis_m"])
        feat_template["ceiling_m"] = float(wx["ceiling_m"])
        feat_template["hour_sin"] = hour_sin
        feat_template["hour_cos"] = hour_cos
        cfg_key = f"cfg_{cfg_id}"
        if cfg_key in feat_template:
            feat_template[cfg_key] = 1.0

        # Score in xy-planes per iz to keep memory bounded.
        for iz, z in enumerate(zs):
            n_cells = nx * ny
            X = np.tile(np.array([feat_template[c] for c in feature_cols],
                                  dtype=np.float64), (n_cells, 1))
            # Per-cell features: d_OLS, d_runway, d_approach, d_departure, traffic_density.
            d_ols_arr = np.empty(n_cells, dtype=np.float32)
            d_app_arr = np.empty(n_cells, dtype=np.float32)
            d_dep_arr = np.empty(n_cells, dtype=np.float32)
            dens_arr = np.empty(n_cells, dtype=np.float32)
            k = 0
            for ix in range(nx):
                for iy in range(ny):
                    px = float(xs[ix]); py = float(ys[iy]); pz = float(z)
                    d_ols_arr[k] = abs(geom.sdf(px, py, pz, arr, dep))
                    d_app = geom.distance_to_active_approach(px, py, pz, arr)
                    d_dep = geom.distance_to_active_departure(px, py, pz, dep)
                    d_app_arr[k] = float(d_app if math.isfinite(d_app) else 1e6)
                    d_dep_arr[k] = float(d_dep if math.isfinite(d_dep) else 1e6)
                    dens_arr[k] = float(adsb_density_box(adsb_df, px, py, pz, pd.Timestamp(t)))
                    k += 1
            idx_map = {c: feature_cols.index(c) for c in feature_cols}
            if "d_OLS_m" in idx_map:     X[:, idx_map["d_OLS_m"]] = d_ols_arr
            if "d_runway_m" in idx_map:  X[:, idx_map["d_runway_m"]] = static_d_runway[:, :, iz].reshape(-1)
            if "d_approach_m" in idx_map: X[:, idx_map["d_approach_m"]] = d_app_arr
            if "d_departure_m" in idx_map: X[:, idx_map["d_departure_m"]] = d_dep_arr
            if "traffic_density" in idx_map: X[:, idx_map["traffic_density"]] = dens_arr

            probs = model.predict_proba(X)[:, 1].astype(dtype)
            rho[it, :, :, iz] = probs.reshape(nx, ny)
        logger.info("risk-grid slice %d/%d at %s (%s) done", it+1, nt, pd.Timestamp(t), cfg_id)

    # NB: zarr-v3 hierarchies are saved incrementally; no consolidation API.
    return output_path


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_model(result: TrainResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": result.model_name,
        "feature_cols": result.feature_cols,
        "metrics": result.metrics,
        "conformal_qhat": getattr(result.conformal, "qhat", None),
    }
    if result.model_name == "mlp":
        import torch
        torch.save({
            "state_dict": result.model.model.state_dict(),
            "hidden": result.model.hidden,
            "n_features": result.model.n_features_,
            "meta": payload,
        }, path)
    else:
        with path.open("wb") as f:
            pickle.dump({"model": result.model, "meta": payload}, f)
    return path


__all__ = [
    "TrainResult", "train", "predict_risk_grid", "save_model", "TorchMLP",
]
