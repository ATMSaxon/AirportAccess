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
| `ofv_<VID>.npz` | One per vertiport in the airport YAML. Contains a *local* high-resolution OFV-funnel SDF (~40×40×40 cells at 10 m spacing, spanning ±200 m laterally and 0–360 m vertically; resolves the 8 m FATO + 80 m top-radius funnel). Keys: `sdf, grid_x, grid_y, grid_z, centre, fato_half, top_r, height`. Typical inside-funnel fraction ≈ 3–4 % (≈ 30 cells in the funnel core). |
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
from src.geometry.query import SDFQuery, SurfaceDistance, PrismIndex

# 4.1 Static (run-config agnostic) trilinear SDF
q = SDFQuery.from_airport("KLAX")
q.clearance_m(x, y, z)   # float or ndarray; positive = clear
q.is_clear(x, y, z)      # bool or ndarray
q.d_OLS(x, y, z)         # alias for clearance_m (positive = outside protection)

# 4.2 2-D distance to per-family footprints (visualisation & ML features)
sd = SurfaceDistance.from_airport("KLAX")
sd.d_runway(x, y)        # 2-D distance to nearest runway-strip footprint
sd.d_approach(x, y)      # 2-D distance to nearest approach surface
sd.d_departure(x, y)     # 2-D distance to nearest takeoff-climb surface

# 4.3 Per-prism membership + RUNWAY-CONFIG-AWARE filtered SDF (ml / planning)
idx = PrismIndex.from_airport("KLAX")
idx.runway_ids()                                              # ['06L','06R','07L','07R','24L','24R','25L','25R']
idx.point_in_approach_prism(x, y, z, rwy_id="24R")            # bool
idx.point_in_departure_prism(x, y, z, rwy_id="25L")           # bool
idx.point_in_missed_approach(x, y, z, rwy_id="24R")           # bool (= takeoff prism of same rwy)
idx.sdf_at(x, y, z, active_arrivals=["24R","25L"],
                    active_departures=["24L","25R"])          # filtered signed distance
idx.distance_to_active_approach(x, y, z, active_arrivals=[...])    # filtered
idx.distance_to_active_departure(x, y, z, active_departures=[...]) # filtered
```

All methods accept scalars or ndarrays (broadcasted via numpy / shapely 2). All
spatial inputs are in airport-local ENU metres (see §3). `PrismIndex.sdf_at`
restricts the union to {approach prisms for *active_arrivals*} ∪ {takeoff
prisms for *active_departures*} ∪ always-on static prisms (runway-strip,
transitional, inner-horizontal, conical, OFZs, RESA) — passing `None`
degenerates to the full static union.

### 4.4 Grid-mode evaluator (for per-slice config-aware envelopes)

For whole-grid SDFs (e.g. the M3 dynamic envelope's runway-config-aware
A_static), use `PrismIndex.eval_on_grid` — it uses the same EDT primitive as
`build_sdf` and is ~30× faster than the point-wise `sdf_at` loop on the
LAX 600×600×117 grid.

```python
from src.geometry.query import PrismIndex
from src.geometry.ols_surfaces import APPROACH, TAKEOFF

idx = PrismIndex.from_airport("KLAX")

# Bake static-only SDF *once* per airport (cache it):
sdf_static_only = idx.eval_on_grid(grid, idx.static_prisms())   # ~23 s on LAX

# Per active (arrivals, departures) config:
arr_prisms = idx.prisms_for_surface(APPROACH, active_arrivals)
dep_prisms = idx.prisms_for_surface(TAKEOFF, active_departures)
sdf_t = idx.eval_on_grid(grid, arr_prisms + dep_prisms,
                         out=sdf_static_only.astype(np.float32, copy=True))
A_static_t = sdf_t > 0                                          # ~5 s per config on LAX
```

Helpers on the index:

- `idx.prisms_for_surface(surface, runway_ids=None)` — lookup
- `idx.static_prisms()` — runway-config-agnostic set (strip / transitional /
   inner-horizontal / conical / OFZ_inapp / OFZ_intr / RESA)

> **Don't seed `out` with the baked `sdf.npz`** — that artefact already
> contains every approach/takeoff prism, so seeding with it then min-reducing
> the active subset yields the *global* SDF, not the config-aware one. Always
> bake the static-only baseline via `eval_on_grid(grid, idx.static_prisms())`.

Accuracy: matches `sdf_at` to within ≲ 0.5 · √(dx² + dy²) ≈ 70 m laterally on
LAX (EDT cell-centre quantization). Median error 22 m, p95 115 m. The
boolean `> 0` threshold for A_static is therefore accurate to within ~1
voxel of the true OLS boundary.

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
