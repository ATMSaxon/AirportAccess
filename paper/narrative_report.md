# DREAM — Narrative Report (TR Part C)

> Manuscript-ready narrative consolidating the DREAM build. Filled in by the team-lead in M8.

## 1. Headline result

(*To be filled.*) On real LAX August 2024 Friday data, DREAM (B4) reduces the mean candidate-corridor
conflict probability vs the static-corridor baseline (B1) by **XX %** while preserving runway
throughput within **YY %** of the no-eVTOL baseline (B0), at a vertiport-to-terminal access time
of **ZZ min** for the landside-edge vertiport (V2).

## 2. Approach (DREAM)

> Annex 14 static surfaces → 3-D SDF → runway-config-aware dynamic envelope → risk-aware A* → safety-capacity-accessibility eval.

Implementation details: see `src/geometry/` (OLS + SDF), `src/traffic/` (ADS-B + envelope),
`src/ml/` (counterfactual injection + risk field), `src/planning/` (envelope-constrained A\*),
`src/analysis/` (joint KPIs).

## 3. Real-data evidence

| Source | Coverage at LAX | Coverage at SFO |
| ------ | --------------- | --------------- |
| FAA NASR runways         | (auto from yaml) | (auto) |
| FAA Digital Obstacle File| (auto)           | (auto) |
| USGS 3DEP DEM            | (auto)           | (auto) |
| OSM Overpass             | (auto)           | (auto) |
| OpenSky ADS-B            | (auto)           | (auto) |
| NOAA AWC + ERA5          | (auto)           | (auto) |
| LAWA / LADOT traffic     | (auto)           | (auto) |
| BTS DB1B/DB1C            | (auto)           | (auto) |

## 4. Hypothesis-by-hypothesis evidence

* **H1** Dynamic ≻ static corridor — figure `figures/eval/h1_dynamic_vs_static.pdf`.
* **H2** Annex 14 geometry alone insufficient — figure `figures/eval/h2_annex14_only.pdf`.
* **H3** Landside-edge V2 ≻ rooftop V3 in joint safety-capacity — figure `figures/eval/h3_vertiport_class.pdf`.
* **H4** Three-objective trade-off — Pareto figure `figures/eval/h4_pareto.pdf`.

## 5. SFO external validation

(*To be filled in M7.*)

## 6. Threats to validity

* No real eVTOL operational data exists; all eVTOL trajectories are counterfactually injected.
* OpenSky ADS-B coverage is partial below 1000 ft AGL.
* The Annex 14 OLS parameter set in `configs/annex14/code4_precision.yaml` is a public-secondary-source
  parameterisation, not the ICAO normative text.
* LAX runway YAML uses publicly available FAA Airport Diagram coordinates; precision is on the order
  of a few metres.

## 7. Reproducibility

```bash
# end-to-end on a fresh clone
pip install -r requirements.txt
bash scripts/run_all.sh
```

All result artefacts carry a `_manifest.json` with source URL, retrieval date, parameters, and git
commit. Every script accepts `--seed` for full determinism.
