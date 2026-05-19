# EXPERIMENT_TRACKER.md — DREAM build runs

> Updated by team-lead as M0–M8 complete. One row per command + run-id.

| Run ID | Milestone | Command | Owner | Status | Result file | Notes |
|--------|-----------|---------|-------|--------|-------------|-------|
| R001 | M0 | `python scripts/run_sanity.py` | team-lead | pending | `results/sanity/summary.json` | Synthetic, no internet needed |
| R010 | M1 | `python scripts/acquire_all.py --airport KLAX --window 2024-08` | data-engineer | pending | `data/processed/KLAX/_inventory.json` | Eight sources, offline-safe |
| R011 | M1 | `python scripts/acquire_all.py --airport KSFO --window 2024-08` | data-engineer | pending | `data/processed/KSFO/_inventory.json` | |
| R020 | M2 | `python scripts/build_ols.py --airport KLAX` | geometry-engineer | pending | `data/processed/KLAX/sdf.npz` | Code 4 precision params |
| R021 | M2 | `python scripts/build_ols.py --airport KSFO` | geometry-engineer | pending | `data/processed/KSFO/sdf.npz` | |
| R030 | M3 | `python scripts/build_envelope.py --airport KLAX --window 2024-08-02 --interval 15min` | traffic-engineer | code-ready (waits on M1 ADS-B) | `data/processed/KLAX/envelope_2024-08-02.zarr` | All 5 modules + CLI + 8 passing tests on KSYN; ran end-to-end on synthetic LAX (96 slices, metar_match=1.0). Real run pending OpenSky parquet from data-engineer. |
| R031 | M3 | `python scripts/build_envelope.py --airport KSFO --window 2024-08-02 --interval 15min` | traffic-engineer | code-ready (waits on M1 ADS-B) | `data/processed/KSFO/envelope_*.zarr` | Same code path; only needs adsb parquet. |
| R040 | M4 | `python scripts/sample_counterfactuals.py --airport KLAX --n 200000` | ml-engineer | pending | `data/processed/KLAX/counterfactuals.parquet` | Labels via injection |
| R041 | M4 | `python scripts/train_risk_field.py --model xgb --airport KLAX` | ml-engineer | pending | `results/risk/KLAX/xgb.json` | AUROC target ≥0.80 |
| R042 | M4 | `python scripts/train_risk_field.py --model mlp --airport KLAX --gpu remote` | ml-engineer | pending | `results/risk/KLAX/mlp.json` | RTX PRO 6000 |
| R050 | M5 | `python scripts/plan_corridors.py --airport KLAX --baseline {B0..B4}` | planning-engineer | pending | `results/corridors/KLAX/…` | All vertiports × baselines |
| R060 | M6 | `python scripts/eval_safety_capacity_access.py --airport KLAX` | planning-engineer | pending | `results/eval/KLAX/kpi_table.parquet` | Pareto ranking check |
| R070 | M7 | `python scripts/eval_safety_capacity_access.py --airport KSFO` | planning-engineer | pending | `results/eval/KSFO/kpi_table.parquet` | Generalisation |
| R080 | M8 | `python scripts/make_paper_figures.py` | team-lead | pending | `figures/*.pdf` + `paper/narrative_report.md` | TR Part C ready |
