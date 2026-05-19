# EXPERIMENT_PLAN.md — DREAM build for TR Part C

> Derived from `refine-logs/FINAL_PROPOSAL.md`. Primary case: **LAX**. External validation: **SFO**.
> Compute: Featurize RTX PRO 6000 (96 GB VRAM, torch 2.9 + CUDA 13).
> Run order: M0 → M1 → M2 ‖ M3 → M4 → M5 → M6 → M7 → M8.

---

## M0 — Sanity (must-run)

* **Goal:** prove the project skeleton works end-to-end on a tiny synthetic airport (single runway,
  one ADS-B track, one weather record, one obstacle).
* **Setup:** `python scripts/run_sanity.py`. Synthetic inputs ship with the repo so this passes
  with zero internet.
* **Outputs:** `results/sanity/summary.json` with all six lanes ticked (data, geometry, traffic,
  ml, planning, analysis).
* **Success criterion:** smoke run returns exit 0, JSON contains keys
  `data_ok, geometry_ok, traffic_ok, ml_ok, planning_ok, analysis_ok` all `true`.
* **Budget:** < 5 min on Mac, < 1 min on GPU server.

---

## M1 — Real data acquisition (must-run, parallel across sources)

* **Goal:** download, project to local ENU, validate and cache **all eight data sources** for
  LAX and SFO.
* **Sources:**

  | ID | Source                                  | Lane                    |
  | -- | --------------------------------------- | ----------------------- |
  | D1 | FAA NASR (runways, taxiways, ARP, RWY)  | airport geometry        |
  | D2 | FAA Digital Obstacle File               | obstacles               |
  | D3 | USGS 3DEP DEM (1/3 arc-second)          | terrain                 |
  | D4 | OSM Overpass (buildings, roads, transit)| ground access + V1-V4   |
  | D5 | OpenSky Network ADS-B (Trino / REST)    | fixed-wing trajectories |
  | D6 | NOAA AWC METAR/TAF + ERA5 reanalysis    | weather                 |
  | D7 | LAWA traffic generation (CSV/scraped)   | ground demand           |
  | D8 | BTS DB1B/DB1C O&D                       | passenger OD            |

* **Setup:** `python scripts/acquire_all.py --airport KLAX --window 2024-08`.
  Each source has its own `src/data/source_XX.py` with retries, schema validation, and
  `_manifest.json` next to every cached file.
* **Outputs:** `data/processed/{airport}/{source}.parquet|geojson|tif` and a top-level
  `data/processed/{airport}/_inventory.json`.
* **Success criterion:**
  * D1: ≥ 4 LAX runway thresholds with bearings, ≥ 2 SFO runways.
  * D2: > 0 obstacles within 50 NM of each airport.
  * D3: DEM raster covers a 60 km × 60 km box around each ARP.
  * D4: ≥ 5000 OSM building footprints inside the 30 km box.
  * D5: ≥ 1 LAX Friday August day with ≥ 50 k state-vector rows in the 30 NM box.
  * D6: ≥ 24 METAR/TAF records per airport per day; ERA5 hourly winds for the 5 days.
  * D7: peak-hour trip counts per LAWA 2024 report logged.
  * D8: ≥ 100 k O&D records for LAX origin/dest.
* **Budget:** 1–4 h wall-clock; bottlenecked by OpenSky Trino throughput.
* **Offline fallback:** any source that fails writes
  `data/processed/{airport}/{source}.OFFLINE.json` describing the failure and the manual
  recovery flow. The pipeline continues for whatever is available.

---

## M2 — Annex 14 OLS as machine-readable 3-D constraints (must-run)

* **Goal:** turn the static Annex 14 OLS into a queryable signed-distance field on a regular
  3-D ENU grid for each airport, plus a vertiport-local obstacle-free volume (OFV).
* **Surfaces:** approach, takeoff-climb, transitional, inner-horizontal, conical, runway strip,
  RESA, OFZ; vertiport approach/departure + OFV.
* **Setup:** `python scripts/build_ols.py --airport KLAX --params configs/annex14/code4_precision.yaml`.
* **Outputs:**
  * `data/processed/{airport}/ols.gpkg` — vector polyhedra per surface.
  * `data/processed/{airport}/sdf.npz` — float32 SDF on a 100 m × 100 m × 30 m grid covering
    a 50 km × 50 km × 4 km box centred on ARP.
  * `data/processed/{airport}/ofv_{V1,V2,V3,V4}.npz` — local OFV SDFs for each candidate vertiport.
* **Success criterion:**
  * SDF is **finite and signed** (positive outside, zero on surface, negative inside) over ≥ 99 %
    of the grid (NaN allowed only above coverage cap).
  * Smoke check: known points on the runway centreline at z=0 return SDF ≈ 0
    (|SDF| < grid spacing).
  * Round-trip: 100 random points sampled in `mathcal{A}_static` then re-queried report
    same sign.
* **Budget:** 5–15 min CPU per airport.

---

## M3 — Runway-configuration inference + dynamic envelope (must-run)

* **Goal:** infer the time-varying runway configuration from real ADS-B, and assemble the
  dynamic envelope `E_t = A_static \ C_t`.
* **Steps:**
  1. ADS-B → arrival/departure classification per track (`src/traffic/classify.py`).
  2. Assign each classified track to the closest runway end (extension-line + bearing test).
  3. 15-min rolling vote → runway configuration `R_t`.
  4. Combine with METAR wind `W_t` and arrival/departure density `F_t`.
  5. Closure function `g(R_t, W_t, F_t)` over the SDF grid (3-D mask).
* **Setup:** `python scripts/build_envelope.py --airport KLAX --window 2024-08-02 --interval 15min`.
* **Outputs:**
  * `data/processed/{airport}/runway_config_{date}.parquet` (time, config, top-2 confidence).
  * `data/processed/{airport}/envelope_{date}.zarr` — `E_t` boolean masks per 15-min slice.
* **Success criterion:**
  * ≥ 90 % of 15-min slices in the 5 LAX Fridays have a confidently identified configuration
    (winning runway has ≥ 70 % share of operations OR top-2 vote is consistent for 3 consecutive slices).
  * Independent METAR cross-check: predicted active landing runway is within ±60° of the
    METAR mean wind direction in ≥ 80 % of slices.
* **Budget:** ~10 min per airport-day on CPU.

---

## M4 — Risk field learning + counterfactual injection (must-run)

* **Goal:** learn `ρ(p,t) = P(conflict | p, t, R_t, W_t, F_t)` from real ADS-B + counterfactually
  injected candidate eVTOL segments.
* **Features per (cell, time):** `d_OLS, d_runway, d_approach, d_departure, traffic_density,
  wind_dir, wind_speed, visibility, ceiling, runway_config, hour_of_day`.
* **Labels:** counterfactual injection — sample candidate eVTOL segments inside `A_static`,
  evaluate min 3-D + 4-D separation against contemporaneous ADS-B; label `1` if any of:
  lateral < L_min, vertical < V_min, axis-crossing of active runway, missed-approach overlap,
  OLS critical-zone intrusion. Defaults: L_min = 1.5 NM, V_min = 1000 ft.
* **Models:** logistic regression (baseline), random forest, XGBoost (primary), MLP on GPU
  (Featurize). All wrapped in conformal prediction for calibrated `ρ̂` + interval.
* **Setup:**
  * `python scripts/sample_counterfactuals.py --airport KLAX --n 200k`
  * `python scripts/train_risk_field.py --model xgb --airport KLAX`
  * `python scripts/train_risk_field.py --model mlp --airport KLAX --gpu remote`
* **Outputs:** `results/risk/{airport}/{model}.json` (metrics), `models/risk/{airport}/{model}.pkl`,
  `data/processed/{airport}/risk_grid_{model}.zarr`.
* **Success criterion:**
  * XGBoost achieves AUROC ≥ 0.80 on a temporally-held-out test day.
  * Conformal coverage within ±0.02 of nominal 0.9 on test set.
* **Budget:** 20–30 min XGBoost + 15 min MLP on RTX PRO 6000.

---

## M5 — Envelope-constrained A\* corridor planner (must-run)

* **Goal:** find `π* = argmin Σ (α1 T + α2 E + α3 ρ + α4 N + α5 I)` over 3-D ENU grid
  inside `E_t` (and `V_OFV` at endpoints), for each (vertiport pair, time slice, baseline B0–B4).
* **Components:**
  * 3-D grid graph on the SDF grid (100 m × 100 m × 30 m by default, adjustable).
  * Edges: 26-connectivity + climb/descent caps + turn-rate caps.
  * Cost weights from `configs/scenarios/cost_weights.yaml`.
  * Baseline switch flips which masks/cost-terms are active (B1 ignores `R_t`; B2 ignores `ρ`; etc.).
* **Setup:** `python scripts/plan_corridors.py --airport KLAX --vertiport V2 --baseline B4 --hours 8,11,17`.
* **Outputs:**
  * `results/corridors/{airport}/{date}/{vertiport}_{baseline}.geojson` (polylines).
  * `results/corridors/{airport}/{date}/{vertiport}_{baseline}.json` (per-corridor KPI dict).
* **Success criterion:** A\* returns a feasible corridor for ≥ 80 % of (date × hour × vertiport ×
  baseline ∈ {B2,B3,B4}) combinations. B1 baseline may fail more often (that's the H1 evidence).
* **Budget:** ~1 s per corridor; ~5 min for the full grid per airport-day.

---

## M6 — LAX safety / capacity / accessibility assessment (must-run)

* **Goal:** evaluate the full KPI matrix from §11 of the proposal, across the 5 LAX Fridays
  × peak hours × V1–V4 × B0–B4.
* **KPIs:** safety (OLS violation rate, min-separation distribution, runway-axis crossings,
  approach/departure/missed-approach interference, obstacle margin, OFV compliance);
  capacity (runway delay, throughput preservation, eVTOL ops/h, corridor closure rate,
  ATC intervention proxy); accessibility (access-time saving vs road/Metro, passenger-weighted
  accessibility, vertiport-to-terminal transfer time, weather reliability, peak service capacity).
* **Setup:** `python scripts/eval_safety_capacity_access.py --airport KLAX`.
* **Outputs:** `results/eval/KLAX/kpi_table.parquet`, `figures/eval/KLAX/*.pdf`.
* **Success criterion:**
  * The Pareto ranking **B4 ≻ B3 ≻ B2 ≻ B1** holds on at least the *safety* axis
    (mean conflict probability decreases monotonically).
  * Trade-off plots (safety vs accessibility vs capacity) generated for V1–V4.

---

## M7 — SFO external validation (must-run)

* **Goal:** re-run M2–M6 on SFO with identical configs (parameter calibration allowed for runways
  only) and report generalisation.
* **Outputs:** mirror M6 in `results/eval/KSFO/`, side-by-side comparison in
  `figures/eval/comparison_LAX_SFO.pdf`.
* **Success criterion:** the **B4 ≻ B3 ≻ B2 ≻ B1** ordering reproduces on the safety axis
  at SFO without re-tuning cost weights.

---

## M8 — Paper-ready figures & narrative report (must-run)

* **Goal:** produce TR Part C-ready figure set + a narrative results report.
* **Figures (minimum):**
  1. System diagram (DREAM five-step pipeline).
  2. Annex 14 OLS 3-D rendering at LAX with ARP/runway thresholds.
  3. Dynamic envelope evolution across one Friday at LAX (filmstrip of 4 hours).
  4. Risk field heatmap overlaid on satellite at LAX peak hour.
  5. Corridor comparison V1–V4 × B0–B4.
  6. Pareto safety-capacity-accessibility plot.
  7. SFO vs LAX generalisation panel.
* **Output:** `paper/narrative_report.md`, all figures in `figures/`.
* **Success criterion:** every figure file exists, no broken references, README in `paper/` lists them.

---

## Compute budget (rough)

| Lane               | CPU-hrs | GPU-hrs |
| ------------------ | ------- | ------- |
| M1 acquisition     |  6      | 0       |
| M2 OLS / SDF       |  2      | 0       |
| M3 envelope        |  4      | 0       |
| M4 risk training   |  2      | 0.5     |
| M5 planning        |  3      | 0       |
| M6 + M7 eval       |  4      | 0       |
| Total              | ~21     | ~0.5    |

GPU is mostly used for the MLP variant in M4 and for sensitivity sweeps in M6/M7.

## Reproducibility

* Top-level `Makefile` (or `scripts/run_all.sh`) replays M0 → M8.
* Every result file has a sibling `summary.json` with `git_commit`, `seed`, `params`, `inputs`.
* `_manifest.json` next to every cached download.
