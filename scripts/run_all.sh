#!/usr/bin/env bash
# DREAM full-pipeline replay (M0 → M8).
# Run from the project root.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "=== M0 sanity ==="
python scripts/run_sanity.py

echo "=== M1 data acquisition (LAX) ==="
python scripts/acquire_all.py --airport KLAX --window 2024-08 || echo "[warn] LAX acquisition partial — see OFFLINE manifests"

echo "=== M1 data acquisition (SFO) ==="
python scripts/acquire_all.py --airport KSFO --window 2024-08 || echo "[warn] SFO acquisition partial"

echo "=== M2 OLS / SDF ==="
python scripts/build_ols.py --airport KLAX
python scripts/build_ols.py --airport KSFO

echo "=== M3 dynamic envelope ==="
for day in 2024-08-02 2024-08-09 2024-08-16 2024-08-23 2024-08-30; do
  python scripts/build_envelope.py --airport KLAX --window "$day" --interval 15min
done
python scripts/build_envelope.py --airport KSFO --window 2024-08-02 --interval 15min

echo "=== M4 risk field ==="
python scripts/sample_counterfactuals.py --airport KLAX --n 200000 --seed 42
python scripts/train_risk_field.py --model xgb --airport KLAX
# Optional GPU run (requires FEATURIZE_PASS in env)
# scripts/deploy_featurize.sh full "python scripts/train_risk_field.py --model mlp --airport KLAX"

echo "=== M5 corridor planning ==="
for V in V1 V2 V3 V4; do
  for B in B0 B1 B2 B3 B4; do
    python scripts/plan_corridors.py --airport KLAX --vertiport "$V" --baseline "$B" --hours 8,11,17 || true
  done
done

echo "=== M6 joint evaluation (LAX) ==="
python scripts/eval_safety_capacity_access.py --airport KLAX

echo "=== M7 generalisation (SFO) ==="
for V in V1 V2 V3 V4; do
  for B in B0 B1 B2 B3 B4; do
    python scripts/plan_corridors.py --airport KSFO --vertiport "$V" --baseline "$B" --hours 8,11,17 || true
  done
done
python scripts/eval_safety_capacity_access.py --airport KSFO

echo "=== M8 paper figures ==="
python scripts/make_paper_figures.py

echo "=== DREAM pipeline done ==="
