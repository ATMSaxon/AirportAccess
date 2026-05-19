"""Real-data acquisition for DREAM.

Each source module (`source_faa_nasr`, `source_faa_dof`, `source_usgs_3dep`, `source_osm`,
`source_opensky`, `source_noaa_wx`, `source_lawa`, `source_bts`) exposes a `fetch(airport_cfg,
window, out_dir)` callable that downloads, validates, projects, and caches the data, writing
both the parquet/geojson/tif output AND a `_manifest.json` sibling.

Failures degrade gracefully: write `<source>.OFFLINE.json` with the error and a manual-recovery
checklist; the rest of the pipeline checks for the OFFLINE marker and continues.
"""
