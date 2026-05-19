# EXPERIMENT_TRACKER.md — DREAM build runs

> Updated by team-lead as M0–M8 complete. One row per command + run-id.

| Run ID | Milestone | Command | Owner | Status | Result file | Notes |
|--------|-----------|---------|-------|--------|-------------|-------|
| R001 | M0 | `python scripts/run_sanity.py` | team-lead | pending | `results/sanity/summary.json` | Synthetic, no internet needed |
| R010 | M1 | `python scripts/acquire_all.py --airport KLAX --window 2024-08` | data-engineer | pending | `data/processed/KLAX/_inventory.json` | Eight sources, offline-safe |
| R011 | M1 | `python scripts/acquire_all.py --airport KSFO --window 2024-08` | data-engineer | pending | `data/processed/KSFO/_inventory.json` | |
| R020 | M2 | `python scripts/build_ols.py --airport KLAX` | geometry-engineer | done | `data/processed/KLAX/sdf.npz` | 98 prisms, ols.gpkg + sdf.npz (600,600,117) f32 + 4 ofv_V*.npz; inside_frac=0.24%; range −135..30824 m; 41 s. 10/10 tests pass. |
| R021 | M2 | `python scripts/build_ols.py --airport KSFO` | geometry-engineer | done | `data/processed/KSFO/sdf.npz` | 98 prisms, ols.gpkg + sdf.npz (600,600,117) f32 + 4 ofv_V*.npz; inside_frac=0.30%; range −135..32343 m; 48 s. |
| R030 | M3 | `python scripts/build_envelope.py --airport KLAX --window 2024-08-02 --interval 15min` | traffic-engineer | code-ready (waits on M1 ADS-B) | `data/processed/KLAX/envelope_2024-08-02.zarr` | All 5 modules + CLI + 8 passing tests on KSYN; ran end-to-end on synthetic LAX (96 slices, metar_match=1.0). `sanity_check()` exposed and green (4 arr → 27, 1 dep → 09, env kept-frac 0.987). Real run pending OpenSky parquet from data-engineer. |
| R031 | M3 | `python scripts/build_envelope.py --airport KSFO --window 2024-08-02 --interval 15min` | traffic-engineer | code-ready (waits on M1 ADS-B) | `data/processed/KSFO/envelope_*.zarr` | Same code path; only needs adsb parquet. |
| R040 | M4 | `python scripts/sample_counterfactuals.py --airport KLAX --n 200000` | ml-engineer | code-ready (waits on M1 ADS-B + M3 runway-config) | `data/processed/KLAX/counterfactuals.parquet` | All 4 modules + 2 scripts + 7 tests landed. INTERFACES.md published. `sanity_check()` wired end-to-end on KSYN (256 segs, 7.4 % conflict, LR AUROC=0.84, conformal coverage=1.00). |
| R041 | M4 | `python scripts/train_risk_field.py --model xgb --airport KLAX` | ml-engineer | code-ready (waits on M1+M3) | `results/risk/KLAX/xgb.json` | LR/RF/XGB/MLP wired w/ temporal-day holdout + split conformal. Smoke on KSYN: XGB AUROC=0.81, conformal coverage=0.90. Risk-grid zarr write verified end-to-end. |
| R042 | M4 | `python scripts/train_risk_field.py --model mlp --airport KLAX --gpu remote` | ml-engineer | code-ready (waits on M1+M3) | `results/risk/KLAX/mlp.json` | Remote path now delegates to `scripts/deploy_featurize.sh full <cmd>` + `pull` (uses `FEATURIZE_PASS`). Ready to fire as soon as M1+M3 LAX artefacts land. |
| R050 | M5 | `python scripts/plan_corridors.py --airport KLAX --baseline {B0..B4}` | planning-engineer | pending | `results/corridors/KLAX/…` | All vertiports × baselines |
| R060 | M6 | `python scripts/eval_safety_capacity_access.py --airport KLAX` | planning-engineer | pending | `results/eval/KLAX/kpi_table.parquet` | Pareto ranking check |
| R070 | M7 | `python scripts/eval_safety_capacity_access.py --airport KSFO` | planning-engineer | pending | `results/eval/KSFO/kpi_table.parquet` | Generalisation |
| R080 | M8 | `python scripts/make_paper_figures.py` | team-lead | pending | `figures/*.pdf` + `paper/narrative_report.md` | TR Part C ready |
