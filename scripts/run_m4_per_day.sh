#!/usr/bin/env bash
# Driver for M4+: per-day counterfactual sampling + train all 4 risk models.
#
# Usage:
#   bash scripts/run_m4_per_day.sh KLAX 2024-08-02 2024-08-09 2024-08-16 2024-08-23 2024-08-30
#
# Waits politely (60 s polls, up to 4 h) for each date's adsb_<date>.parquet
# AND runway_config_<date>.parquet to land on disk before sampling that day.
# Once all dates are sampled, fits LR/RF/XGB (CPU) locally. MLP is fired
# separately via scripts/deploy_featurize.sh.
#
# Exit codes:
#   0  — sampled all days + LR/RF/XGB trained
#   1  — timeout waiting for upstream data
#   2  — sampling failed for some day
#   3  — training failed
set -euo pipefail

ICAO="${1:-KLAX}"; shift || true
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ICAO> <YYYY-MM-DD> [YYYY-MM-DD ...]" >&2
  exit 2
fi
DATES=("$@")

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PROC="data/processed/$ICAO"

WAIT_S=60
TIMEOUT_S=14400  # 4 h
SEED=42
N_PER_DAY=50000

wait_for() {
  local f="$1"
  local elapsed=0
  while [[ ! -s "$f" ]]; do
    if (( elapsed >= TIMEOUT_S )); then
      echo "[wait_for] TIMEOUT after ${TIMEOUT_S}s waiting on $f" >&2
      exit 1
    fi
    sleep "$WAIT_S"
    elapsed=$((elapsed + WAIT_S))
  done
  echo "[wait_for] OK $f"
}

echo "=== Per-day counterfactual sampling for $ICAO ==="
for date in "${DATES[@]}"; do
  adsb="$PROC/adsb_${date}.parquet"
  rc="$PROC/runway_config_${date}.parquet"
  out="$PROC/counterfactuals_${date}.parquet"
  if [[ -s "$out" ]]; then
    echo "[$date] cached → $out (skip)"
    continue
  fi
  echo "[$date] waiting for $adsb + $rc"
  wait_for "$adsb"
  wait_for "$rc"
  echo "[$date] sampling n=$N_PER_DAY"
  PYTHONPATH=. python scripts/sample_counterfactuals.py \
    --airport "$ICAO" --n "$N_PER_DAY" --seed "$SEED" --date "$date" \
    || { echo "[$date] sampling failed" >&2; exit 2; }
done

echo
echo "=== Training LR / RF / XGB (local CPU) ==="
# Invalidate features cache so it gets rebuilt across all days
rm -f "$PROC/features.parquet" "$PROC/features.parquet_manifest.json"
for model in lr rf xgb; do
  echo "--- model=$model ---"
  PYTHONPATH=. python scripts/train_risk_field.py \
    --model "$model" --airport "$ICAO" --seed "$SEED" \
    || { echo "training failed for $model" >&2; exit 3; }
done

echo
echo "=== Per-day pipeline complete for $ICAO ==="
echo "Train MLP on Featurize separately:"
echo "  FEATURIZE_PASS=... FEATURIZE_PORT=57925 bash scripts/deploy_featurize.sh full \\"
echo "    \"python scripts/train_risk_field.py --model mlp --airport $ICAO --seed $SEED\""
