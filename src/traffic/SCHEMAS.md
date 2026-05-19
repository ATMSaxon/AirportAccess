# DREAM traffic-engineer output schemas (interface contract for M3)

> Owner: `traffic-engineer`. Audience: `ml-engineer`, `planning-engineer`,
> `team-lead`. Anything in this file is a stable promise; downstream code can
> import these column names and the `RunwayEnd` dataclass directly.
>
> All processed files live under `data/processed/<ICAO>/` with sibling
> `_manifest.json` files. ENU coordinates are around the airport ARP (metres).
> Times are UTC.

## Module map

| Module                          | Role                                                          |
| ------------------------------- | ------------------------------------------------------------- |
| `src.traffic.adsb_clean`        | OpenSky → cleaned 5-s resampled ENU + geometric altitude      |
| `src.traffic.classify`          | per-track arrival / departure / overflight + runway-end       |
| `src.traffic.runway_config`     | 15-min rolling configuration vote + METAR cross-check         |
| `src.traffic.density`           | 3-D Gaussian-smoothed traffic density on the airport voxel grid|
| `src.traffic.envelope`          | dynamic envelope `E_t = A_static \ C_t`, persistence to Zarr  |

## Cleaned ADS-B (in-memory DataFrame returned by `clean_tracks`)

Same columns as `src/data/SCHEMAS.md` §D5 ; **`x_m`, `y_m`, `z_msl_m`,
`z_agl_m`** are re-derived (`x/y` from `(lon_wgs, lat_wgs)` via the airport
`AirportFrame`; `z_msl_m` from `geo_alt_m` if finite else baro → geometric via
METAR QNH; `z_agl_m = z_msl_m − airport.elev_m`).

## Per-track classification table (output of `classify.classify_tracks`)

| column                  | dtype     | unit / note                                         |
| ----------------------- | --------- | --------------------------------------------------- |
| icao24                  | str       | 24-bit ICAO address (lowercase)                     |
| callsign                | str       | last non-empty callsign                             |
| n_points                | int       | number of resampled samples in the track            |
| t_start, t_end          | datetime64[ns, UTC] | first/last sample                         |
| entry_x, entry_y        | float     | ENU (m), first sample position                      |
| entry_z_agl             | float     | m AGL at first sample                               |
| entry_speed             | float     | m/s at first sample                                 |
| exit_x, exit_y          | float     | ENU (m), last sample position                       |
| exit_z_agl              | float     | m AGL at last sample                                |
| exit_speed              | float     | m/s at last sample                                  |
| min_dist_arp_m          | float     | minimum 2-D distance to ARP across the track        |
| mean_vert_rate_ms       | float     | m/s, positive = climb                               |
| category                | str       | `arrival` / `departure` / `overflight` / `unknown`  |
| runway_end_id           | str/None  | matches an entry in `configs/airports/<ICAO>.yaml`  |
| runway_assign_lateral_m | float     | perpendicular distance to chosen runway centreline  |

## Runway-config table (written to `runway_config_<date>.parquet`)

### Canonical columns (stable)

| column               | dtype                | description                                              |
| -------------------- | -------------------- | -------------------------------------------------------- |
| slice_start          | datetime64[ns, UTC]  | slice start (UTC) — *canonical timestamp*                |
| slice_end            | datetime64[ns, UTC]  | slice end                                                |
| arrivals_active      | str (CSV) / list[str]| **comma**-separated active landing runways (sorted)     |
| departures_active    | str (CSV) / list[str]| **comma**-separated active departing runways            |
| n_arrivals           | int                  | number of arrival ops in the slice                       |
| n_departures         | int                  | number of departure ops in the slice                     |
| arr_share            | float                | top-1 runway share of arrivals (0–1)                     |
| dep_share            | float                | top-1 runway share of departures (0–1)                   |
| metar_wind_dir_deg   | float / nan          | mean wind FROM-direction in slice                        |
| metar_wind_kt        | float / nan          | mean wind speed (kt)                                     |
| metar_match          | bool / null          | wind alignment within ±60° of any active landing rwy     |
| flight_rule          | str / null           | mode METAR flight category (VFR/MVFR/IFR/LIFR)           |
| visibility_sm        | float / nan          | median visibility (statute miles)                        |
| ceiling_ft           | float / nan          | median lowest BKN/OVC ceiling (ft AGL)                   |

### Downstream alias columns (also stable — consumed by ml/planning)

Populated by `runway_config.rolling_config` and written to the same parquet, so
downstream code can index by whichever name is convenient.

| column                | dtype                | description                                              |
| --------------------- | -------------------- | -------------------------------------------------------- |
| time_utc              | datetime64[ns, UTC]  | duplicate of `slice_start` (ml-engineer convention)      |
| active_arrivals       | str                  | **semicolon**-separated active landing runways (e.g. `"06L;06R"`) |
| active_departures     | str                  | **semicolon**-separated active departing runways         |
| config_id             | str                  | compact label `ARR:<a>+<b>\|DEP:<c>+<d>` (or `"UNKNOWN"`) |

## Dynamic envelope storage (`envelope_<date>.zarr` or `.npz` fallback)

* `mask` — bool ndarray, shape `(T, nx, ny, nz)`, chunked `(1, nx, ny, nz)`.
* `time` — UTF string array, shape `(T,)`, ISO-8601 slice-start timestamps.
* Attributes / sidecar (`grid` attr in zarr; `*.grid.json` in npz):
  ```json
  {"x_min": ..., "x_max": ..., "dx": 100.0,
   "y_min": ..., "y_max": ..., "dy": 100.0,
   "z_min": 0.0, "z_max": 3500.0, "dz": 30.0}
  ```

## Tunables (module constants — change here only)

| Constant                                  | Default               | Where                       |
| ----------------------------------------- | --------------------- | --------------------------- |
| `SPEED_JUMP_MAX_MS`                       | 200 m/s               | `adsb_clean`                |
| `ALT_JUMP_MAX_M`                          | 1000 m                | `adsb_clean`                |
| `RESAMPLE_S`                              | 5 s                   | `adsb_clean`                |
| `ARRIVAL_MAX_DIST_M`, `DEPARTURE_MAX_DIST_M` | 25 km             | `classify`                  |
| `BEARING_TOL_DEG`                         | 15°                   | `classify`                  |
| `RUNWAY_AXIS_LATERAL_TOL_M`               | 2 km                  | `classify`                  |
| `ACTIVE_SHARE_THRESHOLD`                  | 0.5                   | `runway_config`             |
| `WIND_BEARING_TOL_DEG`                    | 60°                   | `runway_config`             |
| `LATERAL_BUFFER_M`                        | 3 NM                  | `envelope`                  |
| `VERTICAL_BUFFER_M`                       | 1500 ft               | `envelope`                  |
| `LOW_ALT_CAP_AGL_M`                       | 5000 ft               | `envelope`                  |
| `IMC_LATERAL_EXPANSION`                   | 0.25                  | `envelope`                  |
| `GAUSS_SIGMA_XY_M`, `GAUSS_SIGMA_Z_M`     | 500 m / 60 m          | `density`                   |

## Dependencies

* Consumes D5 OpenSky parquet and D6 METAR parquet from `data-engineer`
  (see `src/data/SCHEMAS.md`).
* Consumes `A_static` from `geometry-engineer` via one of two paths, both
  centralised in `src.traffic.envelope`:
  1. **Default (global static):** `load_static_mask(icao, grid)` →
     `src.geometry.query.SDFQuery.from_airport(icao)` and `A_static = q.sdf > 0`.
     The same mask is reused for every 15-min slice.
  2. **Opt-in (runway-config-aware):** pass a
     `src.geometry.query.PrismIndex` (loaded via
     `envelope.load_prism_index(icao)`) to `envelope_for_slice(..., prism_index=...)`.
     `A_static_t` is then recomputed per slice as
     `prism_index.sdf_at(X, Y, Z, active_arrivals, active_departures) > 0`.
     This relaxes the protection union to only the prisms of currently-active
     runways (plus always-on static surfaces). Recommended for real LAX/SFO
     runs once benchmarked; default off to preserve the historical contract.
  Reading raw `sdf.npz` directly is deliberately not supported so future grid
  metadata (terrain bottoms, lateral safety buffers, OFV interactions) flows
  through automatically. If both paths are unavailable, the envelope falls back
  to all-clear and logs a one-shot warning advising to run
  `scripts/build_ols.py` first.

## Stability promise

Once this file lands in `main`, columns and module-level constants may be
**added** without notice but never **renamed or removed** without prior
team-wide SendMessage announcement.
