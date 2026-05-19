# CLAUDE.md — project-level guidance for the DREAM eVTOL airport-access codebase

## Project
**DREAM: Dynamic Runway-configuration-aware Envelope for eVTOL Airport Mobility.**
Operationalise ICAO Annex 14 obstacle limitation surfaces (OLS) into a runway-configuration-aware
3-D safety envelope for eVTOL airport shuttle integration, validated against real airport operational data
(FAA NASR/DOF + USGS 3DEP + OSM + OpenSky ADS-B + NOAA AWC/ERA5 + LAWA + BTS).
Primary case: **LAX (KLAX, August 2024 Fridays)**. External validation: **SFO (KSFO)**.
Target venue: **Transportation Research Part C**.

## Language and tooling
* **Python 3.11+** (Featurize server has 3.11.8 + torch 2.9 + CUDA 13 on RTX PRO 6000 Blackwell, 96 GB).
* **Geo:** GeoPandas / Shapely 2.x / pyproj / rasterio / contextily / osmnx (offline-friendly).
* **Trajectories:** OpenSky `traffic` library OR direct REST/Trino client.
* **DES/sim:** SimPy (only for capacity-impact micro-simulator).
* **3-D:** trimesh + scipy.spatial + scikit-fmm (SDF), shapely 2 for prism polyhedra.
* **ML:** scikit-learn, xgboost, lightgbm, PyTorch ≥ 2.2 (when GPU needed).
* **Planning:** networkx + a custom A\* with weighted 3-D grid + ENU coordinates.
* **Figures:** matplotlib + contextily + cartopy (no Plotly / Bokeh).
* **I/O:** pyarrow Parquet for all processed tables; GeoJSON for cached vector layers.

## Coordinate systems
* WGS-84 (EPSG:4326) for raw data ingest only.
* **Local ENU around each airport's ARP** is the working frame:
  * LAX: ARP 33.9425N / 118.4081W, field elevation 125 ft / 38.1 m MSL.
  * SFO: ARP 37.6189N / 122.3750W, field elevation 13 ft / 4.0 m MSL.
* Altitudes: store both `z_msl_m` and `z_agl_m`. Pressure altitude from ADS-B is converted with ERA5 surface QNH.

## Annex 14 parameters
The repo ships an **open parameterisation** in `configs/annex14/code4_precision.yaml`
(approach surface, takeoff climb, transitional, inner-horizontal, conical, strip, RESA, OFZ).
These are *generic Code 4 precision* placeholder values with citations to public secondary sources —
they are *not* a copy of the ICAO Annex 14 normative text. Real-airport calibration uses FAA-published
runway data (NASR LID 5010-1 / Airport Diagrams) and is parameter-tunable per airport.

## Conventions
* Reproducibility: every script supports `--seed`, `--config`, `--output-dir`. Configs live in `configs/`.
* Data caches go under `data/cache/<source>/<timestamp>/` and are tagged with source URL + retrieval date.
* All processed artefacts emit a `_manifest.json` alongside the data file (source, params, hash, row count).
* Results: every run writes `results/<run-id>/summary.json` plus run-specific artefacts.
* Logging: `logging` with module-level loggers, INFO by default, DEBUG with `--debug`.
* Tests under `tests/` run via `pytest -q`.

## Layout
```
airportaccess/
├── CLAUDE.md
├── requirements.txt
├── refine-logs/                # plan + tracker + (proposal mirror)
│   ├── EXPERIMENT_PLAN.md
│   ├── EXPERIMENT_TRACKER.md
│   └── FINAL_PROPOSAL.md
├── configs/
│   ├── airports/               # per-airport runway + ARP + OFV anchors
│   ├── annex14/                # OLS parameter tables (Code 4 precision)
│   ├── scenarios/              # vertiport candidate sets V1–V4
│   └── sanity.yaml             # tiny end-to-end smoke run
├── data/
│   ├── raw/                    # untouched downloads
│   ├── processed/              # parquet / geojson, ENU-projected
│   └── cache/                  # transient HTTP caches
├── src/
│   ├── data/                   # acquisition + preprocessing (FAA/USGS/OSM/OpenSky/NOAA/LAWA/BTS)
│   ├── geometry/               # Annex 14 OLS prism builders + SDF + vertiport OFV
│   ├── traffic/                # ADS-B parsing, runway-config inference, density fields, conflict counters
│   ├── ml/                     # risk-field learners, conformal calibration, counterfactual injection
│   ├── planning/               # envelope-constrained A\* corridor planner
│   ├── analysis/               # safety / capacity / accessibility KPIs + baseline comparison
│   └── utils/                  # shared CRS, IO, time, paths
├── scripts/                    # entry-point CLIs
├── tests/
├── results/
└── figures/
```

## Run conventions
* Sanity:                    `python scripts/run_sanity.py`
* Acquire LAX data:            `python scripts/acquire_all.py --airport KLAX --window 2024-08`
* Build OLS for an airport:    `python scripts/build_ols.py --airport KLAX`
* Build dynamic envelope:      `python scripts/build_envelope.py --airport KLAX --window 2024-08-02`
* Train risk field:            `python scripts/train_risk_field.py --model xgb --airport KLAX`
* Plan corridors:              `python scripts/plan_corridors.py --airport KLAX --vertiport V2 --baseline B4`
* Joint evaluation:            `python scripts/eval_safety_capacity_access.py --airport KLAX`

## GPU
Featurize RTX PRO 6000 (Blackwell, 96 GB VRAM, torch 2.9 + CUDA 13) reachable at
`ssh featurize@workspace.featurize.cn -p 27749`. Push via rsync; install pip deps from the Aliyun mirror.
Project lives in `/home/featurize/work/airportaccess/`.

## What NOT to do
* Don't fabricate ADS-B, weather, or airport-geometry data. If a source is offline, write an offline
  manifest with the error and document the manual recovery flow — do NOT replace with synthetic.
* Don't use another model's output as ground truth for any evaluation.
* Don't hard-code Annex 14 normative values from the ICAO standard; use the documented public
  parameterisation in `configs/annex14/` and cite it.
* Don't claim a milestone passed without writing the corresponding figure/result/summary JSON and
  updating `refine-logs/EXPERIMENT_TRACKER.md`.
