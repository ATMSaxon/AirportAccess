#!/usr/bin/env bash
# Run safety/capacity/access KPI eval for KLAX + KSFO in parallel.
# Outputs:
#   results/eval/KLAX/{kpi_table.parquet, summary.json, *.png}
#   results/eval/KSFO/{kpi_table.parquet, summary.json, *.png}
#
# Usage on Featurize:
#   cd /home/featurize/work/airportaccess && bash scripts/run_eval_both.sh
set -uo pipefail

REPO="/home/featurize/work/airportaccess"
cd "$REPO"

PY="/environment/miniconda3/bin/python"
LOG_DIR="$REPO/logs_eval"
mkdir -p "$LOG_DIR"

PIDS=()
for AP in KLAX KSFO; do
  LOG="$LOG_DIR/${AP}.log"
  echo "[$(date -u +%H:%M:%S)] launching eval for $AP → $LOG"
  PYTHONPATH=. "$PY" -u scripts/eval_safety_capacity_access.py \
    --airport "$AP" \
    >"$LOG" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAIL=$((FAIL+1))
done
echo "[$(date -u +%H:%M:%S)] eval done; failed=$FAIL"
exit "$FAIL"
