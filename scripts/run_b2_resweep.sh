#!/usr/bin/env bash
# Re-fire B2-only corridors after the both-sides-closed static mask redesign.
# 10 (airport × date) jobs run in parallel. Each job: 12 pairs × 3 hours × 1 baseline = 36 corridors.
#
# Usage on Featurize:
#   cd /home/featurize/work/airportaccess && bash scripts/run_b2_resweep.sh
set -uo pipefail

REPO="/home/featurize/work/airportaccess"
cd "$REPO"

PY="/environment/miniconda3/bin/python"
LOG_DIR="$REPO/logs_b2_resweep"
mkdir -p "$LOG_DIR"

DATES=(2024-08-02 2024-08-09 2024-08-16 2024-08-23 2024-08-30)
AIRPORTS=(KLAX KSFO)

PIDS=()
for AP in "${AIRPORTS[@]}"; do
  for D in "${DATES[@]}"; do
    LOG="$LOG_DIR/${AP}_${D}.log"
    echo "[$(date -u +%H:%M:%S)] launching $AP $D → $LOG"
    PYTHONPATH=. "$PY" -u scripts/plan_corridors.py \
      --airport "$AP" \
      --vertiport V1,V2,V3,V4 \
      --baseline B2 \
      --hours 8,11,17 \
      --date "$D" \
      --planning-resolution 300,90 \
      --max-pops 200000 \
      --output-dir "$REPO/results/corridors/$AP" \
      >"$LOG" 2>&1 &
    PIDS+=($!)
  done
done

echo "Launched ${#PIDS[@]} jobs; waiting..."
FAIL=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || FAIL=$((FAIL+1))
done
echo "[$(date -u +%H:%M:%S)] B2 re-sweep done; failed=$FAIL"
exit "$FAIL"
