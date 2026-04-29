# ISS Transit Finder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Calculates when the International Space Station (ISS) passes in front of the **Sun** or **Moon** as seen from a defined location and search radius, for the next N days.

ISS transits across the solar or lunar disc are rare, fleeting events (typically 0.5–2 seconds) visible only from a narrow corridor on the ground. This tool finds them for any location on Earth.

## How it works

The script uses a two-pass algorithm:

1. **Coarse scan** – vectorised over the full forecast period (10-second steps) to find candidate times when the ISS approaches within 2° of the Sun or Moon as seen from the search centre.
2. **Fine scan** – high-resolution pass (0.05-second steps, ±70-second window) around each candidate, run for 9 observer points covering the search radius (centre + 8 cardinal/intercardinal points). Reports the observer with the smallest angular separation.

Positions are computed with [Skyfield](https://rhodesmill.org/skyfield/) using DE421 planetary ephemerides and a live TLE fetched from Celestrak.

## Requirements

```bash
# Core dependencies (required)
pip install skyfield requests numpy

# Map generation (optional)
pip install matplotlib contextily pyproj
```

Skyfield automatically downloads `de421.bsp` (~17 MB) on first run. Maps are written to `maps/` and can be skipped with `--no-maps`.

## Usage

```bash
python transit_finder.py --lat <latitude> --lon <longitude> [options]
```

| Argument | Description | Default |
|---|---|---|
| `--lat` | Latitude in degrees (positive = North) | required |
| `--lon` | Longitude in degrees (positive = East) | required |
| `--radius` | Search radius in km | 50 |
| `--days` | Forecast period in days | 7 |
| `--name` | Location name for display | — |
| `--no-maps` | Skip PNG map generation | maps enabled |

### Examples

```bash
# Frankfurt, 100 km radius, 7 days
python transit_finder.py --lat 50.11 --lon 8.68 --radius 100 --name Frankfurt

# Munich, 50 km radius, 14 days
python transit_finder.py --lat 48.14 --lon 11.58 --days 14 --name Munich

# London, 75 km radius
python transit_finder.py --lat 51.50 --lon -0.12 --radius 75 --name London

# Bad Homburg, 30 km radius, 7 days
python transit_finder.py --lat 50.23 --lon 8.62 --radius 30 --name "Bad Homburg"
```

### Sample output

```
ISS Transit Finder – Initialising
─────────────────────────────────────────────
  Fetching TLE (NORAD 25544) from Celestrak ... OK  [ISS (ZARYA)]
  Loading planetary ephemeris (de421.bsp)   ... OK
  TLE epoch:  2026-04-29 04:01 UTC  (age: 0.2 days)

  Coarse scan: 60,481 steps × 10 s  ... 3 candidates.
  Fine scan: analysing 3 candidates ...

═════════════════════════════════════════════════════════════════
  ISS TRANSIT FINDER  –  Solar & Lunar Transits
─────────────────────────────────────────────────────────────────
  Location: Berlin  52.5200° N,  13.4100° E
  Radius:   25 km
  Period:   2026-04-29 – 2026-05-06 UTC
═════════════════════════════════════════════════════════════════

  # 1  LUNAR TRANSIT  🌙  –  2026-04-30
─────────────────────────────────────────────────────────────────
  Time (UTC):      23:38:42
  Duration:        0.6 s
  Min. separation: 14.7"  →  CENTRAL TRANSIT ✓  (crosses disc centre)
  Chord coverage:  32 % of disc diameter
  Moon position:   alt 18.7°,  az 199.0° (SSW)
  ISS position:    alt 18.5°,  az 198.9° (SSW)
  Best location:   52.3607° N,  13.6717° E  (~25 km from centre)
```

## Output fields explained

| Field | Description |
|---|---|
| **Duration** | Time the ISS spends inside the solar/lunar disc |
| **Min. separation** | Closest angular distance between ISS centre and disc centre (arcseconds) |
| **Central transit** | ISS path crosses the disc interior (min. sep. < disc radius) |
| **Chord coverage** | Length of the ISS path chord as a percentage of the disc diameter |
| **Best location** | Observer point within the search radius with the smallest separation |

## Accuracy & limitations

- **TLE age**: ISS positional accuracy degrades roughly 1–2 km per day of TLE age. Use a fresh TLE (< 2 days old) for reliable predictions.
- **Visibility corridor**: The ground track where a given transit is visible is typically 5–15 km wide. The 9-point observer ring samples the search radius but does not guarantee coverage of every point inside it.
- **Moon distance**: The script uses a fixed mean lunar angular radius. Actual size varies by ~3 % depending on the Moon's distance.
- **Solar transits**: Always use a certified solar filter when observing. Never look at the Sun without one.

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| [skyfield](https://rhodesmill.org/skyfield/) | Yes | Satellite & planetary position calculations |
| [numpy](https://numpy.org/) | Yes | Vectorised coarse-scan arithmetic |
| [requests](https://docs.python-requests.org/) | Yes | TLE download from Celestrak |
| [matplotlib](https://matplotlib.org/) | Optional | Map rendering |
| [contextily](https://contextily.readthedocs.io/) | Optional | Basemap tile download for maps |
| [pyproj](https://pyproj4.github.io/pyproj/) | Optional | Coordinate projection for maps |
