# DREAM data-engineer output schemas (interface contract for M1)

> Owner: `data-engineer`. Audience: `traffic-engineer`, `ml-engineer`, `geometry-engineer`,
> `planning-engineer`. Anything in this file is a stable promise; downstream code can
> import these column names directly.
>
> All processed files live under `data/processed/<ICAO>/` with a sibling `_manifest.json`.
> All coordinates are in **local ENU around ARP** (metres) unless a column ends in `_wgs`.
> An offline source writes `<name>.OFFLINE.json` (same directory) with the error and the
> manual-recovery flow; consumers MUST check `_inventory.json` to know which sources
> are live vs. offline.

## Inventory file (`_inventory.json`)
```json
{
  "icao": "KLAX",
  "window": "2024-08",
  "generated_utc": "2026-05-19T…Z",
  "sources": {
    "faa_nasr":  {"status": "ok", "files": ["runways.parquet", "runways.geojson"]},
    "faa_dof":   {"status": "ok", "files": ["obstacles.parquet"]},
    "usgs_3dep": {"status": "ok", "files": ["dem.tif"]},
    "osm":       {"status": "ok", "files": ["buildings.geojson", "roads.geojson", "amenities.geojson"]},
    "opensky":   {"status": "offline", "files": ["opensky.OFFLINE.json"]},
    "noaa_wx":   {"status": "ok", "files": ["metar.parquet"]},
    "lawa":      {"status": "ok", "files": ["peak_hour.parquet"]},
    "bts":       {"status": "ok", "files": ["db1b_ond.parquet"]}
  }
}
```

---

## D1 — `faa_nasr` → `runways.parquet` + `runways.geojson`
| column          | dtype     | unit / note                                  |
|-----------------|-----------|----------------------------------------------|
| icao            | str       | airport ICAO (e.g. KLAX)                     |
| runway_id       | str       | e.g. "06L"                                   |
| thr_lon_wgs     | float64   | threshold longitude (deg WGS-84)             |
| thr_lat_wgs     | float64   | threshold latitude  (deg WGS-84)             |
| end_lon_wgs     | float64   | departure-end longitude                      |
| end_lat_wgs     | float64   | departure-end latitude                       |
| thr_x_m         | float64   | threshold ENU east  (m)                      |
| thr_y_m         | float64   | threshold ENU north (m)                      |
| thr_z_m         | float64   | threshold MSL elevation (m, from airport elev)|
| end_x_m         | float64   | departure-end ENU east  (m)                  |
| end_y_m         | float64   | departure-end ENU north (m)                  |
| end_z_m         | float64   | departure-end MSL elevation (m)              |
| length_m        | float64   | runway physical length                       |
| width_m         | float64   | runway physical width                        |
| bearing_deg     | float64   | true bearing of runway centreline            |
| code_letter     | str       | ICAO aerodrome code letter (A–F)             |
| code_number     | int8      | ICAO aerodrome code number (1–4)             |
| precision       | bool      | precision-approach runway?                   |

`runways.geojson` (EPSG:4326): one `LineString` per runway with the same attributes.

## D2 — `faa_dof` → `obstacles.parquet`
| column          | dtype     | unit / note                                  |
|-----------------|-----------|----------------------------------------------|
| oas_number      | str       | OAS / DOF number                             |
| obstacle_type   | str       | TOWER, BUILDING, etc. (DOF dictionary)       |
| lon_wgs         | float64   | obstacle longitude                           |
| lat_wgs         | float64   | obstacle latitude                            |
| x_m             | float64   | ENU east of ARP (m)                          |
| y_m             | float64   | ENU north of ARP (m)                         |
| agl_ft          | float32   | height AGL (ft, raw DOF)                     |
| msl_ft          | float32   | height MSL (ft, raw DOF)                     |
| agl_m           | float32   | height AGL (m)                               |
| msl_m           | float32   | height MSL (m)                               |
| accuracy_h_ft   | float32   | DOF horizontal accuracy code → ft (DOF dict) |
| accuracy_v_ft   | float32   | DOF vertical accuracy code   → ft            |
| marked          | str       | M / U / blank                                |
| lighted         | str       | L / U / blank                                |
| within_nm       | float32   | great-circle distance to ARP (NM)            |

Filtered to records within `radius_nm` (default 50 NM) of the ARP.

## D3 — `usgs_3dep` → `dem.tif`
GeoTIFF, EPSG:4326 by default (reprojected to local UTM if `--reproject utm` flag set).
Covers the airport's `extract_box_m` (60 km × 60 km around the ARP) at the native 1/3
arc-second resolution (~10 m). Sidecar manifest records source tile URLs and concat order.

## D4 — `osm` → `buildings.geojson`, `roads.geojson`, `amenities.geojson`
GeoJSON (EPSG:4326) plus ENU-projected Parquet siblings:
- `buildings.parquet` columns: `osm_id, building, levels, height_m, x_m, y_m, area_m2`
- `roads.parquet`     columns: `osm_id, highway, name, oneway, lanes, length_m, x_m, y_m`
- `amenities.parquet` columns: `osm_id, amenity, name, x_m, y_m`

Bounding box = ARP-centred 30 km × 30 km square.

## D5 — `opensky` → `adsb_<YYYY-MM-DD>.parquet`
One parquet per **day** in the requested window. If only REST `/states/all` is reachable
the rows are 10-second snapshots; if Trino is reachable they are 1-second state vectors.
| column        | dtype   | unit / note                                                       |
|---------------|---------|-------------------------------------------------------------------|
| time_utc      | datetime64[ns, UTC] | observation UTC timestamp                              |
| icao24        | str     | aircraft ICAO 24-bit address                                      |
| callsign      | str     | callsign (may be empty)                                           |
| lon_wgs       | float32 | longitude (deg WGS-84)                                            |
| lat_wgs       | float32 | latitude  (deg WGS-84)                                            |
| baro_alt_m    | float32 | barometric altitude (m, pressure altitude)                        |
| geo_alt_m     | float32 | GNSS altitude (m, may be NaN if unavailable)                      |
| velocity_ms   | float32 | ground speed (m/s)                                                |
| track_deg     | float32 | true track over ground (deg)                                      |
| vert_rate_ms  | float32 | climb / descent rate (m/s, positive = climb)                      |
| on_ground     | bool    | OpenSky on-ground flag                                            |
| x_m           | float32 | ENU east of ARP (m)                                               |
| y_m           | float32 | ENU north of ARP (m)                                              |
| z_msl_m       | float32 | best altitude estimate (m MSL): geo_alt_m if present else baro_alt|
| z_agl_m       | float32 | z_msl_m − DEM under (x_m, y_m); fallback z_msl_m − field elev     |

## D6 — `noaa_wx` → `metar.parquet` (always) + `era5_surface.nc` (if creds)
`metar.parquet`:
| column       | dtype                | unit / note                                  |
|--------------|----------------------|----------------------------------------------|
| station_id   | str                  | ICAO id, e.g. KLAX                           |
| time_utc     | datetime64[ns, UTC]  | METAR observation time                       |
| wind_dir_deg | float32              | from-direction in degrees                    |
| wind_kt      | float32              | sustained wind in knots                      |
| wind_gust_kt | float32              | gust knots (NaN if none)                     |
| vis_sm       | float32              | visibility in statute miles                  |
| temp_c       | float32              | temperature                                  |
| dewpoint_c   | float32              | dewpoint                                     |
| altim_hpa    | float32              | altimeter setting (hPa, ≈QNH)                |
| ceiling_ft   | float32              | lowest BKN/OVC ceiling (ft AGL, NaN if none) |
| flight_rule  | str                  | VFR / MVFR / IFR / LIFR                      |
| raw          | str                  | raw METAR text                               |

`era5_surface.nc` (only present if `CDSAPI_KEY` env var set): hourly `u10`, `v10`, `msl`,
`t2m`, `d2m` for the airport's 1°×1° tile.

## D7 — `lawa` → `peak_hour.parquet`
Hard-coded from the 2024 LAWA *Surface Transportation Generation Report* (cited in the
manifest). One row per `(year, peak_hour, mode, direction)`.
| column          | dtype  | description                                  |
|-----------------|--------|----------------------------------------------|
| year            | int16  | source year                                  |
| airport         | str    | LAX / SFO (LAWA only publishes LAX)          |
| peak_hour       | str    | "08-09", "11-12", "17-18"                    |
| direction       | str    | "in" (to airport) / "out" (from airport)     |
| mode            | str    | private_vehicle / shuttle / taxi / rideshare / transit / employee_bus |
| trips           | int32  | trips per hour                               |
| source          | str    | citation                                     |

## D8 — `bts` → `db1b_ond.parquet`
Quarter Q3 2024, filtered to origin OR destination in {LAX, SFO}. Columns are the BTS
DB1B Coupon/Ticket subset:
| column          | dtype   | description                                  |
|-----------------|---------|----------------------------------------------|
| itin_id         | int64   | itinerary id                                 |
| origin          | str     | three-letter IATA                            |
| dest            | str     | three-letter IATA                            |
| reporting_carrier| str    | reporting carrier                            |
| passengers      | int32   | number of passengers on itinerary            |
| market_fare     | float32 | dollars                                      |
| market_distance | float32 | miles                                        |
| quarter         | int8    | 3                                            |
| year            | int16   | 2024                                         |

---

## Stability promise

Once this file lands in `main`, columns may be **added** without notice but never
**renamed or removed** without prior team-wide SendMessage announcement.
