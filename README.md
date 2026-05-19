# DREAM — Dynamic Runway-configuration-aware Envelope for eVTOL Airport Mobility

[![sanity](https://img.shields.io/badge/sanity-pending-lightgrey)](#)
[![tests](https://img.shields.io/badge/tests-12%2B-blue)](#)
[![venue](https://img.shields.io/badge/target--venue-TR%20Part%20C-orange)](#)

A real-data-driven 3-D safety-envelope framework for integrating eVTOL airport-shuttle operations into
conventional airport environments. The framework operationalises ICAO Annex 14 obstacle limitation
surfaces (OLS) into a runway-configuration-aware dynamic envelope, learns a spatio-temporal risk
field from real fixed-wing ADS-B trajectories via counterfactual injection, and plans corridors
with an envelope-constrained A\*.

* **Primary case:** Los Angeles International Airport (KLAX), Fridays in August 2024.
* **External validation:** San Francisco International Airport (KSFO).
* **Target venue:** *Transportation Research Part C*.
* **Proposal mirror:** [`refine-logs/FINAL_PROPOSAL.md`](refine-logs/FINAL_PROPOSAL.md).
* **Build plan:** [`refine-logs/EXPERIMENT_PLAN.md`](refine-logs/EXPERIMENT_PLAN.md).

## Pipeline

```
Annex 14 static surfaces
  → src/geometry        machine-readable 3-D SDF
  → src/traffic         runway-config-aware dynamic envelope
  → src/ml              risk-field learning + conformal calibration
  → src/planning        envelope-constrained A* corridor planner
  → src/analysis        safety / capacity / accessibility KPIs
```

## Quick start

```bash
# 1. Local development
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                                # smoke + utility tests
python scripts/run_sanity.py                       # tiny synthetic end-to-end

# 2. Real-data run (LAX)
python scripts/acquire_all.py --airport KLAX --window 2024-08
python scripts/build_ols.py --airport KLAX
python scripts/build_envelope.py --airport KLAX --window 2024-08-02 --interval 15min
python scripts/sample_counterfactuals.py --airport KLAX --n 200000
python scripts/train_risk_field.py --model xgb --airport KLAX
python scripts/plan_corridors.py --airport KLAX --vertiport V2 --baseline B4 --hours 8,11,17
python scripts/eval_safety_capacity_access.py --airport KLAX

# Or one-shot
bash scripts/run_all.sh

# 3. Remote training on Featurize GPU (RTX PRO 6000)
FEATURIZE_PASS=… scripts/deploy_featurize.sh full \
  "python scripts/train_risk_field.py --model mlp --airport KLAX"
```

## Layout

See [`CLAUDE.md`](CLAUDE.md) for full conventions. Key directories:

| Path                | Purpose                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `configs/airports/` | Per-airport YAMLs (runway thresholds, ARP, vertiport anchors)        |
| `configs/annex14/`  | OLS parameter tables (public-secondary-source Code 4 precision)      |
| `src/utils/`        | Shared CRS / I/O / grid / config helpers                             |
| `src/data/`         | Eight real-data acquisition modules (FAA / USGS / OSM / OpenSky / NOAA / LAWA / BTS) |
| `src/geometry/`     | OLS prism builders, SDF, vertiport OFV                               |
| `src/traffic/`      | ADS-B preprocessing, runway-config inference, dynamic envelope       |
| `src/ml/`           | Counterfactual injection + risk field + conformal calibration        |
| `src/planning/`     | Envelope-constrained A\* corridor planner                            |
| `src/analysis/`     | Safety / capacity / accessibility KPI calculators                    |

## Data sources

| Source                 | Used for                              | Auth                          |
| ---------------------- | ------------------------------------- | ----------------------------- |
| FAA NASR / Airport Diagrams | Airport geometry (runways, ARP)  | none                          |
| FAA Digital Obstacle File   | Obstacles within 50 NM            | none                          |
| USGS 3DEP DEM               | Terrain                           | none                          |
| OpenStreetMap (Overpass)    | Vertiport candidates, ground network | none                       |
| OpenSky Network ADS-B       | Real fixed-wing trajectories     | OPENSKY_USERNAME / PASSWORD (Trino) |
| NOAA Aviation Weather + ERA5 | Wind, visibility, ceiling       | CDS API key for ERA5          |
| LAWA Traffic Generation Report 2024 | Ground-access demand        | shipped CSV (no fetch)        |
| BTS DB1B / DB1C              | Passenger O&D                    | none                          |

## Threats to validity

* No real eVTOL operational data exists for LAX or SFO; this study uses **counterfactual injection**
  of candidate eVTOL segments into real fixed-wing operational scenes.
* OpenSky ADS-B coverage is partial below ~1000 ft AGL.
* Annex 14 numeric parameters in `configs/annex14/code4_precision.yaml` are public-secondary-source
  Code 4 precision values; they are not a copy of the ICAO normative text.

## Reproducibility

* Every script accepts `--seed`.
* Every cached data artefact has a sibling `_manifest.json` (source URL, retrieval date, hash, params, git commit).
* Every result run writes a `summary.json` with the same metadata.
* CI runs `pytest -q` on each commit (see `scripts/run_all.sh` for the full pipeline replay).
