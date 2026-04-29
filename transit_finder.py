#!/usr/bin/env python3
"""
transit_finder.py  –  ISS transits across the Sun and Moon

Calculates all ISS crossings in front of the solar or lunar disk
for the next N days, visible from a defined radius around a location.

Requirements:
    pip install skyfield requests numpy

Usage:
    python transit_finder.py --lat 50.11 --lon 8.68 --radius 100 --name Frankfurt
    python transit_finder.py --lat 48.14 --lon 11.58 --days 14 --radius 50
"""

import argparse
import math
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import requests
from skyfield.api import load, wgs84, EarthSatellite

# ── Algorithm parameters ───────────────────────────────────────────────────────
SUN_RADIUS_DEG   = 0.2665   # mean apparent radius of the Sun  (~16')
MOON_RADIUS_DEG  = 0.2575   # mean apparent radius of the Moon (~15.5')
COARSE_STEP_S    = 10       # coarse-scan time step in seconds
FINE_STEP_S      = 0.05     # fine-scan time step in seconds
FINE_WINDOW_S    = 70       # fine-scan window (±seconds around each candidate)
APPROACH_DEG     = 2.0      # angular threshold that triggers the fine scan
DEDUP_S          = 120      # events closer than this are the same transit
NORAD_ISS        = 25544


# ── TLE retrieval ──────────────────────────────────────────────────────────────

def fetch_tle(norad: int = NORAD_ISS) -> tuple[str, str, str]:
    """Fetch the current ISS TLE from the Celestrak GP API."""
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
    except Exception as e:
        sys.exit(f"[Error] TLE fetch failed: {e}")
    sys.exit("[Error] Unexpected TLE format from Celestrak")


# ── Geometry helpers ───────────────────────────────────────────────────────────

def sep_vec(alt1, az1, alt2, az2):
    """Vectorised angular separation (degrees) between two alt/az positions."""
    a1, z1 = np.radians(alt1), np.radians(az1)
    a2, z2 = np.radians(alt2), np.radians(az2)
    c = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(z1 - z2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def chord_coverage(min_sep_deg: float, r_deg: float) -> float:
    """Fraction of the disk diameter the ISS path crosses (0–100 %)."""
    if min_sep_deg >= r_deg:
        return 0.0
    return math.sqrt(max(0.0, 1.0 - (min_sep_deg / r_deg) ** 2)) * 100.0


def azimuth_label(az_deg: float) -> str:
    labels = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
              "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return labels[round(az_deg / 22.5) % 16]


# ── Observer grid ──────────────────────────────────────────────────────────────

def observer_ring(lat: float, lon: float, radius_km: float) -> list[tuple[float, float]]:
    """Centre point plus 8 points on the search-radius circle (every 45°)."""
    pts = [(lat, lon)]
    if radius_km > 0.1:
        for bearing_deg in range(0, 360, 45):
            b = math.radians(bearing_deg)
            dlat = (radius_km / 111.0) * math.cos(b)
            dlon = (radius_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(b)
            pts.append((lat + dlat, lon + dlon))
    return pts


# ── Coarse scan ────────────────────────────────────────────────────────────────

def coarse_scan(
    lat: float, lon: float, days: int, ts, eph, iss
) -> tuple[list[tuple[float, str]], float]:
    """
    Vectorised scan from the centre observer.
    Returns (candidate list, t0_tt); candidates are (tt_jd, body_name).
    """
    earth    = eph["earth"]
    observer = wgs84.latlon(lat, lon)

    t0    = datetime.now(timezone.utc)
    n     = int(days * 86400 / COARSE_STEP_S) + 1
    t0_tt = ts.from_datetime(t0).tt
    tt    = t0_tt + np.arange(n) * (COARSE_STEP_S / 86400.0)
    t_arr = ts.tt_jd(tt)

    print(f"  Coarse scan: {n:,} steps × {COARSE_STEP_S} s  ...", end=" ", flush=True)

    iss_alt, iss_az, _   = (iss - observer).at(t_arr).altaz()
    earth_at             = (earth + observer).at(t_arr)
    sun_alt, sun_az, _   = earth_at.observe(eph["sun"]).apparent().altaz()
    moon_alt, moon_az, _ = earth_at.observe(eph["moon"]).apparent().altaz()

    iss_up  = iss_alt.degrees > 0
    sun_up  = sun_alt.degrees > -5
    moon_up = moon_alt.degrees > -5

    sep_sun  = sep_vec(iss_alt.degrees, iss_az.degrees, sun_alt.degrees,  sun_az.degrees)
    sep_moon = sep_vec(iss_alt.degrees, iss_az.degrees, moon_alt.degrees, moon_az.degrees)

    hits_sun  = tt[iss_up & sun_up  & (sep_sun  < APPROACH_DEG)]
    hits_moon = tt[iss_up & moon_up & (sep_moon < APPROACH_DEG)]

    candidates = ([(float(t), "sun")  for t in hits_sun] +
                  [(float(t), "moon") for t in hits_moon])
    candidates.sort(key=lambda x: x[0])

    # Cluster consecutive hits for the same body into a single candidate
    clustered: list[tuple[float, str]] = []
    last_t: dict[str, float] = {}
    for t_c, body in candidates:
        if body not in last_t or (t_c - last_t[body]) * 86400 > COARSE_STEP_S * 2.5:
            clustered.append((t_c, body))
        last_t[body] = t_c

    print(f"{len(clustered)} candidates.")
    return clustered, t0_tt


# ── Fine scan ──────────────────────────────────────────────────────────────────

def fine_scan(
    t_cand_tt: float,
    body_name: str,
    observers: list[tuple[float, float]],
    ts, eph, iss,
) -> dict | None:
    """
    High-resolution scan in the window ±FINE_WINDOW_S around a candidate.
    Checks every observer point in the search radius; returns the best result.
    """
    earth    = eph["earth"]
    body_eph = eph["sun"] if body_name == "sun" else eph["moon"]
    r_deg    = SUN_RADIUS_DEG if body_name == "sun" else MOON_RADIUS_DEG

    n_fine = int(2 * FINE_WINDOW_S / FINE_STEP_S)
    tt = (t_cand_tt - FINE_WINDOW_S / 86400.0) + np.arange(n_fine) * (FINE_STEP_S / 86400.0)
    t_arr = ts.tt_jd(tt)

    best: dict | None = None

    for obs_lat, obs_lon in observers:
        obs = wgs84.latlon(obs_lat, obs_lon)

        iss_alt, iss_az, _ = (iss - obs).at(t_arr).altaz()
        b_alt, b_az, _     = (earth + obs).at(t_arr).observe(body_eph).apparent().altaz()

        valid = (iss_alt.degrees > 0) & (b_alt.degrees > -1)
        sep   = sep_vec(iss_alt.degrees, iss_az.degrees, b_alt.degrees, b_az.degrees)
        sep[~valid] = 999.0

        idx     = int(np.argmin(sep))
        min_sep = float(sep[idx])

        if min_sep >= r_deg:
            continue  # ISS misses the disk at this observer location

        # Duration: number of fine steps where ISS is inside the disk
        in_disk    = valid & (sep < r_deg)
        duration_s = float(np.sum(in_disk)) * FINE_STEP_S

        # Single-point positions at the transit moment (for output)
        t_transit  = ts.tt_jd(float(tt[idx]))
        t_utc      = t_transit.utc_datetime()
        ba, baz, _ = (earth + obs).at(t_transit).observe(body_eph).apparent().altaz()

        # ISS altitude/azimuth at transit moment
        ia, iaz, _ = (iss - obs).at(t_transit).altaz()

        # Distance of this observer from the search centre
        clat, clon = observers[0]
        dist_km = math.sqrt(
            ((obs_lat - clat) * 111.0) ** 2 +
            ((obs_lon - clon) * 111.0 * math.cos(math.radians(clat))) ** 2
        )

        entry = {
            "time":        t_utc,
            "tt":          float(tt[idx]),
            "body":        body_name,
            "min_sep_am":  min_sep * 60,
            "r_am":        r_deg * 60,
            "coverage":    chord_coverage(min_sep, r_deg),
            "duration_s":  duration_s,
            "body_alt":    ba.degrees,
            "body_az":     baz.degrees,
            "iss_alt":     ia.degrees,
            "iss_az":      iaz.degrees,
            "obs_lat":     obs_lat,
            "obs_lon":     obs_lon,
            "obs_dist_km": dist_km,
        }

        if best is None or min_sep < best["min_sep_am"] / 60:
            best = entry

    return best


# ── Orchestration ──────────────────────────────────────────────────────────────

def find_all_transits(
    lat: float, lon: float, radius_km: float, days: int, ts, eph, iss
) -> list[dict]:
    observers = observer_ring(lat, lon, radius_km)
    clustered, _ = coarse_scan(lat, lon, days, ts, eph, iss)

    print(f"  Fine scan: analysing {len(clustered)} candidates ...", flush=True)

    transits: list[dict] = []
    seen: list[tuple[float, str]] = []   # (tt, body) of already-found transits

    for t_cand, body in clustered:
        # Skip if a transit for this body was already found at a nearby time
        if any(abs(t_cand - st) * 86400 < DEDUP_S and sb == body for st, sb in seen):
            continue

        result = fine_scan(t_cand, body, observers, ts, eph, iss)
        if result:
            seen.append((result["tt"], body))
            transits.append(result)

    transits.sort(key=lambda x: x["time"])
    return transits


# ── Output ─────────────────────────────────────────────────────────────────────

def print_results(
    transits: list[dict],
    lat: float, lon: float, radius_km: float,
    days: int, location_name: str,
) -> None:
    W = 65
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)

    print()
    print("═" * W)
    print("  ISS TRANSIT FINDER  –  Solar & Lunar Transits")
    print("─" * W)
    loc_str = f"{location_name}  " if location_name else ""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    print(f"  Location: {loc_str}{abs(lat):.4f}° {ns},  {abs(lon):.4f}° {ew}")
    print(f"  Radius:   {radius_km:.0f} km")
    print(f"  Period:   {now.strftime('%Y-%m-%d')} – {end.strftime('%Y-%m-%d')} UTC")
    print("═" * W)

    if not transits:
        print()
        print("  No ISS transits found for this period and area.")
        print()
        print("  Note: ISS transits across the Sun or Moon are rare events.")
        print("  Typical rate: 0–3 transits per month at any given location.")
        print()
        return

    for i, tr in enumerate(transits, 1):
        dt     = tr["time"]
        is_sun = tr["body"] == "sun"
        label  = "SOLAR TRANSIT  ☀" if is_sun else "LUNAR TRANSIT  🌙"
        sep_am = tr["min_sep_am"]
        r_am   = tr["r_am"]

        print()
        print(f"  # {i}  {label}  –  {dt.strftime('%Y-%m-%d')}")
        print("─" * W)
        print(f"  Time (UTC):      {dt.strftime('%H:%M:%S')}")
        print(f"  Duration:        {tr['duration_s']:.1f} s")

        if sep_am < r_am:
            print(f"  Min. separation: {sep_am:.1f}\"  →  CENTRAL TRANSIT ✓  (crosses disc centre)")
        else:
            print(f"  Min. separation: {sep_am:.1f}\"  (disc radius: {r_am:.1f}\")")
        print(f"  Chord coverage:  {tr['coverage']:.0f} % of disc diameter")

        body_label = "Sun" if is_sun else "Moon"
        az_l = azimuth_label(tr["body_az"])
        print(f"  {body_label} position:   alt {tr['body_alt']:.1f}°,  "
              f"az {tr['body_az']:.1f}° ({az_l})")
        iss_az_l = azimuth_label(tr["iss_az"])
        print(f"  ISS position:    alt {tr['iss_alt']:.1f}°,  "
              f"az {tr['iss_az']:.1f}° ({iss_az_l})")

        if tr["obs_dist_km"] < 1.0:
            vis_str = "search centre"
        else:
            vis_str = f"~{tr['obs_dist_km']:.0f} km from centre"
        print(f"  Best location:   {tr['obs_lat']:.4f}° N,  "
              f"{tr['obs_lon']:.4f}° E  ({vis_str})")

    print()
    print("─" * W)
    print(f"  Total: {len(transits)} transit(s) found.")
    print("═" * W)
    print()
    print("  Notes:")
    print("  • Accuracy depends on TLE age (< 2 days recommended).")
    print("  • Visibility corridor is typically 5–15 km wide.")
    print("  • Solar transits require a proper solar filter!")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calculate ISS transits across the Sun and Moon for the next N days",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transit_finder.py --lat 50.11 --lon 8.68 --radius 100 --name Frankfurt
  python transit_finder.py --lat 48.14 --lon 11.58 --days 14
  python transit_finder.py --lat 52.52 --lon 13.41 --radius 25 --name Berlin
  python transit_finder.py --lat 51.50 --lon -0.12 --radius 75 --name London
""",
    )
    p.add_argument("--lat",    type=float, required=True,
                   help="Latitude in degrees (positive = North)")
    p.add_argument("--lon",    type=float, required=True,
                   help="Longitude in degrees (positive = East)")
    p.add_argument("--radius", type=float, default=50.0,
                   help="Search radius in km (default: 50)")
    p.add_argument("--days",   type=int,   default=7,
                   help="Forecast period in days (default: 7)")
    p.add_argument("--name",   type=str,   default="",
                   help="Location name for display (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print("ISS Transit Finder – Initialising")
    print("─" * 45)

    print(f"  Fetching TLE (NORAD {NORAD_ISS}) from Celestrak ...", end=" ", flush=True)
    tle_name, line1, line2 = fetch_tle()
    print(f"OK  [{tle_name.strip()}]")

    print("  Loading planetary ephemeris (de421.bsp)  ...", end=" ", flush=True)
    ts  = load.timescale()
    eph = load("de421.bsp")
    iss = EarthSatellite(line1, line2, tle_name, ts)
    print("OK")

    tle_epoch = iss.epoch.utc_datetime()
    age_days  = (datetime.now(timezone.utc) - tle_epoch).total_seconds() / 86400
    print(f"  TLE epoch:  {tle_epoch.strftime('%Y-%m-%d %H:%M UTC')}  "
          f"(age: {age_days:.1f} days)")
    if age_days > 3:
        print(f"  ⚠  TLE is more than 3 days old – positional accuracy reduced!")

    print()
    transits = find_all_transits(
        args.lat, args.lon, args.radius, args.days, ts, eph, iss
    )

    print_results(transits, args.lat, args.lon, args.radius, args.days, args.name)


if __name__ == "__main__":
    main()
