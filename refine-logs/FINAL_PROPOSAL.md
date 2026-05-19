# DREAM — Final Proposal (mirror of user-supplied proposal)

> Source: `/experiment-bridge` command args, 2026-05-19.
> Target venue: **Transportation Research Part C**.
> Method name (fixed): **DREAM — Dynamic Runway-configuration-aware Envelope for eVTOL Airport Mobility.**

## Headline

> Operationalising ICAO Annex 14 obstacle limitation surfaces into dynamic, runway-configuration-aware
> safety envelopes for eVTOL airport shuttle integration using real airport operational data.

## Five-step pipeline

```
Annex 14 static surfaces
 → machine-readable 3D constraints (SDF)
 → runway-configuration-aware dynamic envelope
 → risk-aware eVTOL corridor planning (envelope-constrained A*)
 → real-data-based safety-capacity-accessibility assessment
```

## Hypotheses

* **H1** Dynamic envelope ≻ static corridor (lower fixed-wing conflict risk).
* **H2** Annex 14 geometry alone is insufficient; runway config + wx + flow are needed.
* **H3** Landside-edge vertiports ≻ terminal-rooftop vertiports for safety-capacity balance.
* **H4** Safety – accessibility – capacity is a non-trivial 3-objective trade-off.

## Vertiport scenarios (LAX)

| ID | Location class                                            | Expected |
| -- | --------------------------------------------------------- | -------- |
| V1 | Off-fence remote (e.g. west of LAX, near Vista Del Mar)   | Safe / long transfer |
| V2 | Landside transit centre / ConRAC / Intermodal Facility    | Balanced |
| V3 | Terminal-area rooftop (TBIT / CTA core)                   | Best access / hardest safety |
| V4 | City-end (Downtown LA / Union Station / LA Metro hub)     | Network end |

## Baselines

| ID | Method                                                |
| -- | ----------------------------------------------------- |
| B0 | No-eVTOL baseline (real fixed-wing only)              |
| B1 | Static eVTOL corridor (single fixed corridor)         |
| B2 | Annex-14-geometry-only corridor (SDF-constrained)     |
| B3 | Runway-config-aware dynamic envelope                  |
| B4 | + ADS-B-learned risk field (full DREAM)               |

## Data sources

FAA eNASR/NASR, FAA Digital Obstacle File (DOF), FAA Airport Diagrams, USGS 3DEP DEM,
OpenStreetMap (Overpass), OpenSky Network (ADS-B history via Trino / public REST),
NOAA Aviation Weather Center API (METAR/TAF/PIREP/SIGMET), ECMWF ERA5 reanalysis,
LAWA LAX Traffic Generation Report (2024), BTS DB1B / DB1C O&D.

Primary window: **Fridays in August 2024 — 2024-08-02, 08-09, 08-16, 08-23, 08-30** (LAWA design days).
Peak hours: 08–09, 11–12, 17–18 (per LAWA).

## Counterfactual injection statement

> This study uses real airport, real fixed-wing trajectory, real weather, real obstacle, real ground
> traffic and real passenger OD data, into which candidate eVTOL shuttle operations are *counterfactually
> injected* to assess safety and capacity impact. No real eVTOL operational data is claimed.

(Full proposal text is the user's command argument; this file is the project mirror.)
