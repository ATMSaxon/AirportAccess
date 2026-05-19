# DREAM ML-engineer output interfaces (M4)

> Owner: `ml-engineer`. Audience: `planning-engineer`, `traffic-engineer`,
> `geometry-engineer`. Stable contract once landed on `main`.

All file paths are relative to the repository root.

---

## C1 — Counterfactual segments parquet

`data/processed/<ICAO>/counterfactuals.parquet`

One row per sampled candidate eVTOL segment.

| column            | dtype       | unit / note                                  |
|-------------------|-------------|----------------------------------------------|
| seg_id            | str         | "<ICAO>-<8-digit>" globally unique           |
| vertiport_id      | str         | one of V1/V2/V3/V4 (configured)              |
| icao              | str         | airport ICAO                                 |
| config_id         | str         | runway-config id from M3                     |
| t_start_utc       | datetime64[ns, UTC] | segment start time                   |
| t_end_utc         | datetime64[ns, UTC] | segment end time                     |
| duration_s        | float64     | segment 3-D duration at cruise speed         |
| x0_m,y0_m,z0_m    | float64     | origin ENU + MSL                              |
| x1_m,y1_m,z1_m    | float64     | destination ENU + MSL                         |
| mid_x_m,…,mid_t_utc | mixed     | midpoint in space + time                      |
| length_m          | float64     | 3-D segment length                           |
| climb_angle_deg   | float64     | signed (positive = climb)                    |
| cruise_speed_mps  | float64     | eVTOL cruise speed                           |
| active_arrivals   | str         | ";"-joined active arrival runway ids          |
| active_departures | str         | ";"-joined active departure runway ids        |
| conflict          | int8 0/1    | binary conflict label                         |
| cause             | str         | comma-joined causes (see below)               |
| min_lat_m_adsb    | float64     | min lateral distance to any ADS-B obs (m)     |
| min_vert_m_adsb   | float64     | min vertical distance to any ADS-B obs (m)    |
| axis_cross        | bool        | crosses an active runway axis < 2000 ft AGL  |
| approach_hit      | bool        | intersects an active approach prism          |
| departure_hit     | bool        | intersects an active departure prism         |
| missed_hit        | bool        | intersects an active missed-approach surface |
| sdf_mid_m         | float64     | OLS SDF at midpoint (m, sign per `_geom.sdf`) |

`cause` values are a comma-joined subset of:
`adsb_near`, `axis_cross`, `approach_prism`, `departure_prism`,
`missed_approach`, `sdf_buffer`. Empty when `conflict = 0`.

Defaults baked into the file:
* `L_min = 1.5 NM = 2778 m`
* `V_min = 1000 ft = 304.8 m`
* `axis_z_max_agl = 2000 ft = 609.6 m`
* `sdf_buffer = 30 m`

A sibling `_manifest.json` records the seed, scenario yaml, defaults, source URLs,
and the empirical conflict rate.

---

## C2 — Features parquet

`data/processed/<ICAO>/features.parquet`

One row per segment, keyed by `seg_id`.

| column            | dtype     | note                                          |
|-------------------|-----------|-----------------------------------------------|
| seg_id            | str       | join key                                      |
| d_OLS_m           | float64   | distance to nearest **active** OLS surface (m, |sdf|) |
| d_runway_m        | float64   | 3-D distance to nearest runway centreline    |
| d_approach_m      | float64   | distance to nearest active approach prism    |
| d_departure_m     | float64   | distance to nearest active departure prism   |
| traffic_density   | float64   | obs / (m³ · s) in a 1.5 km × 1.5 km × 300 m × 10 min box |
| wind_dir_sin/cos  | float64   | cyclic encoding of METAR wind direction      |
| wind_speed_mps    | float64   | METAR sustained wind                          |
| visibility_m      | float64   | METAR visibility                              |
| ceiling_m         | float64   | METAR ceiling (AGL); 11000 m sentinel for CAVOK |
| hour_sin/cos      | float64   | cyclic encoding of midpoint UTC hour          |
| cfg_<id>          | float32   | one-hot over runway-config ids                |

---

## C3 — Risk grid zarr

`data/processed/<ICAO>/risk_grid_xgb.zarr`

Hierarchy:
```
/rho           uint8-quantised float16 — shape (T, NX, NY, NZ), range [0,1]
/x_m           float64 (NX,)
/y_m           float64 (NY,)
/z_msl_m       float64 (NZ,)
/time_utc_ns   int64 (T,)           unix nanoseconds (UTC); also in root `time_iso` attr
```
Attributes on the root group:
* `icao`           — airport ICAO
* `shape_order`    — `"t,x,y,z"`
* `voxel_dx_m/dy_m/dz_m`
* `feature_cols`   — the model's feature ordering

Storage layout uses 1-slice time chunks for incremental write/read.

---

## C4 — Models on disk

`models/risk/<ICAO>/<model>.pkl` (sklearn / xgboost) or `.pt` (MLP).
Pickled dict: `{"model": fitted_estimator, "meta": {"model_name", "feature_cols",
"metrics", "conformal_qhat"}}`.

---

## C5 — Metrics JSON

`results/risk/<ICAO>/<model>.json`

```json
{
  "model_name": "xgb",
  "auroc":  0.86,
  "aupr":   0.41,
  "logloss": 0.18,
  "n_train": 160000,
  "n_test":  40000,
  "pos_rate_train": 0.07,
  "pos_rate_test":  0.06,
  "conformal_coverage": 0.902,
  "conformal_target":   0.9,
  "conformal_width":    0.18,
  "conformal_qhat":     0.09
}
```

## Stability promise

Columns / array names may be **added** without notice but never **renamed
or removed** without a team-wide SendMessage.
