# `src/geometry` — public interface

Owner: **geometry-engineer** (M2). Consumers: traffic-engineer (M3), ml-engineer (M4),
planning-engineer (M5).

## 1. Build artefacts (per airport)

`scripts/build_ols.py --airport <ICAO> [--params configs/annex14/code4_precision.yaml]`
writes everything below into `data/processed/<ICAO>/`:

| File | Content |
|------|---------|
| `ols.gpkg` (layer `ols`) | GeoPackage of all OLS prism footprints + z-height params (one row per prism). CRS = airport-local AEQD metres. Columns: `name, surface, runway_id, end_id, z_form, z_top_a, z_top_b, z_top_c, z_top_slope, z_top_r0, cx, cy, z_low, geometry`. |
| `sdf.npz` | `sdf` (nx, ny, nz, float32), `grid_x, grid_y, grid_z` (cell-centre coords, m, ENU around ARP). |
| `ofv_<VID>.npz` | One per vertiport in the airport YAML. Contains a *local* high-resolution OFV-funnel SDF. Keys: `sdf, grid_x, grid_y, grid_z, centre, fato_half, top_r, height`. |
| `<file>_manifest.json` | Source + params + sha256 + git_commit for every artefact above. |

## 2. Sign conventions

- **`sdf.npz`** — built as the union of OLS protection prisms:
  - **positive** outside all protection volumes → eVTOL is *clear*.
  - **negative** inside any protection volume → eVTOL is in conflict with OLS.
- **`ofv_<V>.npz`** — built as the FATO+funnel volume around a vertiport:
  - **negative** inside the funnel → safe near-vertiport airspace.
  - **positive** outside → forbidden without explicit clearance.

(The OFV sign is *opposite* to the OLS SDF because the OFV defines a *permissive*
volume; the OLS defines a *forbidden* volume.)

## 3. Coordinate system

All `(x, y)` are local ENU metres around the airport ARP, using a pyproj `aeqd`
projection centred on `cfg.arp.{lat,lon}` (see `src/utils/crs.AirportFrame`).
All `z` are metres **AGL above the ARP elevation** (so z=0 = field elevation MSL,
positive up).

## 4. Python API (after running `scripts/build_ols.py`)

```python
from src.geometry.query import SDFQuery, SurfaceDistance

q = SDFQuery.from_airport("KLAX")
q.clearance_m(x, y, z)   # float or ndarray; positive = clear
q.is_clear(x, y, z)      # bool or ndarray
q.d_OLS(x, y, z)         # alias for clearance_m (positive = outside protection)

sd = SurfaceDistance.from_airport("KLAX")
sd.d_runway(x, y)        # 2-D distance to nearest runway-strip footprint
sd.d_approach(x, y)      # 2-D distance to nearest approach surface
sd.d_departure(x, y)     # 2-D distance to nearest takeoff-climb surface
```

Both classes accept scalars, 1-D arrays, or broadcasted ndarrays. All inputs in
airport-local ENU metres.

## 5. Stubs for downstream teams

While the SDF is being built, downstream teams may stub against these APIs:

```python
# traffic (M3): boolean "static safety envelope" mask
A_static = q.sdf > 0          # 3-D boolean array; full clear-airspace mask

# ml (M4): per-cell feature vector
features = {
    "d_OLS": q.d_OLS(x, y, z),        # signed 3-D (positive = clear)
    "d_runway":    sd.d_runway(x, y),
    "d_approach":  sd.d_approach(x, y),
    "d_departure": sd.d_departure(x, y),
}

# planning (M5): 3-D A* graph filter
walkable[i, j, k] = sdf[i, j, k] > 0
```

## 6. Annex 14 surface catalogue

Per the open parameterisation in `configs/annex14/code4_precision.yaml`. Each
surface is implemented as one or more `Prism`s:

| Surface | Per | Sub-prisms |
|---------|-----|------------|
| `approach`               | runway end (×2 per physical RWY) | 3 (two sloped sections + horizontal) |
| `takeoff_climb`          | runway end | 1–2 (divergent + capped-width if applicable) |
| `transitional`           | runway sides | 2 |
| `runway_strip`           | physical runway | 1 (degenerate vertical → anchors SDF=0 on the RWY surface) |
| `resa`                   | stop end | 1 |
| `ofz_inner_approach`     | runway end | 1 |
| `ofz_inner_transitional` | runway sides | 2 |
| `inner_horizontal`       | airport (ARP-centred disc) | 1 |
| `conical`                | airport (ARP-centred annulus) | 1 (radial z_top) |

LAX with 8 runway records → ~80 prisms + 2 airport-level = 82 prisms.
SFO with 8 runway records → ~82 prisms.

## 7. Limitations / TODO

- Per-airport calibration of OLS parameters is by overriding the annex14 YAML;
  the LAX/SFO airport YAMLs do **not** currently override these.
- OFV approach/departure surface (1:8 funnel along a specified heading) is not
  yet rendered — only the omnidirectional FATO funnel is.
- Prism z_low is fixed at 0 (ARP ground level). Terrain-following lower bound
  is not yet incorporated (will fold in DEM from M1/D3 in a follow-up).
