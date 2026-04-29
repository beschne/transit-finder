# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the script

```bash
# Core dependencies only
pip install skyfield requests numpy

# With map generation support
pip install matplotlib contextily pyproj

# Example run (Frankfurt, 100 km radius, 7-day forecast)
python transit_finder.py --lat 50.11 --lon 8.68 --radius 100 --name Frankfurt

# Skip map generation
python transit_finder.py --lat 50.11 --lon 8.68 --no-maps
```

`de421.bsp` (~17 MB) is downloaded automatically by Skyfield on first run and is gitignored. Maps are written to `maps/` (also gitignored).

## Architecture

The single script `transit_finder.py` implements a two-pass search algorithm:

1. **Coarse scan** (`coarse_scan`) — vectorised over the full forecast period in 10-second steps. Uses NumPy arrays for all positions; finds candidate times when the ISS comes within 2° of the Sun or Moon as seen from the search centre.

2. **Fine scan** (`fine_scan`) — 0.05-second steps over a ±70-second window around each candidate. Run for 9 observer points (centre + 8 compass points on the search-radius circle from `observer_ring`). Returns the observer with the smallest angular separation.

3. **Corridor computation** (`compute_corridor`) — for confirmed transits, traces the ISS-to-body ray back to the WGS84 ellipsoid at each timestep to produce a ground-track centre line and half-width. Uses GCRS→ITRS rotation via GAST. The Moon uses the ISS-relative body direction (not geocentric) because the geocentric approximation shifts the centre line by ~60 km.

4. **Map generation** (`generate_map`) — optional; requires matplotlib/contextily/pyproj. Renders the corridor as a PNG with a CartoDB Positron basemap (falls back to OSM).

All positions are computed with Skyfield (`EarthSatellite`, `wgs84.latlon`, DE421 ephemeris). The TLE is fetched live from Celestrak's GP API (NORAD 25544).

## Key constants (top of file)

| Constant | Value | Purpose |
|---|---|---|
| `COARSE_STEP_S` | 10 s | Coarse-scan time resolution |
| `FINE_STEP_S` | 0.05 s | Fine-scan time resolution |
| `APPROACH_DEG` | 2.0° | Angular threshold triggering fine scan |
| `DEDUP_S` | 120 s | Minimum time between distinct transits |
| `SUN_RADIUS_DEG` | 0.2665° | Mean apparent solar radius |
| `MOON_RADIUS_DEG` | 0.2575° | Mean apparent lunar radius |
