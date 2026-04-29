#!/usr/bin/env python3
"""
transit_finder.py  –  ISS transits across the Sun and Moon

Calculates all ISS crossings in front of the solar or lunar disk
for the next N days, visible from a defined radius around a location.
Optionally generates a PNG map of the visibility corridor for each transit.

Requirements (core):
    pip install skyfield requests numpy

Requirements (maps):
    pip install matplotlib contextily pyproj

Usage:
    python transit_finder.py --lat 50.11 --lon 8.68 --radius 100 --name Frankfurt
    python transit_finder.py --lat 48.14 --lon 11.58 --days 14 --radius 50
"""

import argparse
import math
import os
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
MAPS_DIR         = "maps"


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


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (degrees, 0–360) from point 1 to point 2."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    """Return the point dist_km away from (lat, lon) in the given bearing."""
    R  = 6371.0
    δ  = dist_km / R
    φ1 = math.radians(lat)
    λ1 = math.radians(lon)
    θ  = math.radians(bearing_deg)
    φ2 = math.asin(math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ))
    λ2 = λ1 + math.atan2(math.sin(θ) * math.sin(δ) * math.cos(φ1),
                          math.cos(δ) - math.sin(φ1) * math.sin(φ2))
    return math.degrees(φ2), math.degrees(λ2)


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


# ── Visibility corridor ────────────────────────────────────────────────────────

def _ray_ellipsoid_intersect(
    p_itrs: np.ndarray, d_itrs: np.ndarray, body_r_deg: float
) -> tuple[np.ndarray | None, float]:
    """
    Find where the ray from ISS position p (ITRS, km) in direction d (unit,
    ITRS) toward the body intersects the WGS84 ellipsoid.

    The transit centre line is the set of ground points from which the ISS
    appears exactly in front of the body's centre. It is found by tracing
    the ray from the ISS toward the body backward to Earth's surface.

    All arithmetic is in ITRS (ECEF) so the WGS84 ellipsoid is axis-aligned.
    Returns (Q_itrs, corridor_half_km) or (None, 0.0).
    """
    a = 6378.137    # WGS84 semi-major axis km
    b = 6356.7523   # WGS84 semi-minor axis km
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)

    A = (d_itrs[0]**2 + d_itrs[1]**2) * inv_a2 + d_itrs[2]**2 * inv_b2
    B = -2.0 * ((p_itrs[0]*d_itrs[0] + p_itrs[1]*d_itrs[1]) * inv_a2
                + p_itrs[2]*d_itrs[2] * inv_b2)
    C = (p_itrs[0]**2 + p_itrs[1]**2) * inv_a2 + p_itrs[2]**2 * inv_b2 - 1.0

    disc = B*B - 4.0*A*C
    if disc < 0:
        return None, 0.0

    sqrt_disc = math.sqrt(disc)
    t = (-B - sqrt_disc) / (2.0 * A)   # near-side (observer) solution
    if t <= 0:
        t = (-B + sqrt_disc) / (2.0 * A)
    if t <= 0:
        return None, 0.0

    Q  = p_itrs - t * d_itrs
    hw = t * math.tan(math.radians(body_r_deg))
    return Q, hw


def _itrs_to_geodetic(x: float, y: float, z: float) -> tuple[float, float]:
    """Convert ITRS ECEF (km) to WGS84 geodetic (lat_deg, lon_deg)."""
    a  = 6378.137
    f  = 1.0 / 298.257223563
    b  = a * (1.0 - f)
    e2 = 1.0 - (b / a) ** 2
    lon = math.degrees(math.atan2(y, x))
    p   = math.sqrt(x*x + y*y)
    lat = math.atan2(z, p * (1.0 - e2))   # initial geocentric estimate
    for _ in range(6):                      # Bowring iteration converges in 3–4 steps
        N   = a / math.sqrt(1.0 - e2 * math.sin(lat)**2)
        lat = math.atan2(z + e2 * N * math.sin(lat), p)
    return math.degrees(lat), lon


def compute_corridor(transit: dict, ts, eph, iss) -> dict:
    """
    Compute the transit visibility corridor.

    Uses a two-resolution time grid: 1-second steps for the ±30 s approach
    context, and 0.1-second steps for the transit zone itself.

    Returns a dict with:
        center      – list of (lat, lon) | None, one per time step
        halves      – corridor half-width in km at each step
        in_transit  – bool list, True where ISS is inside the disk
    """
    from skyfield import framelib
    from skyfield.positionlib import Geocentric

    earth    = eph["earth"]
    body_eph = eph["sun"] if transit["body"] == "sun" else eph["moon"]
    r_deg    = SUN_RADIUS_DEG if transit["body"] == "sun" else MOON_RADIUS_DEG

    ctx_off  = np.arange(-30.0, 31.0, 1.0)
    half_t   = max(transit["duration_s"] / 2.0 + 2.0, 3.0)
    fine_off = np.arange(-half_t, half_t + 0.05, 0.1)
    offsets  = np.unique(np.concatenate([ctx_off, fine_off]))

    tt    = transit["tt"] + offsets / 86400.0
    t_arr = ts.tt_jd(tt)

    # EarthSatellite.at() is already geocentric (GCRS); planetary bodies are
    # barycentric (BCRS) and need the Earth geocentre subtracted.
    iss_gcrs  = iss.at(t_arr).position.km                                     # (3, n)
    body_gcrs = body_eph.at(t_arr).position.km - earth.at(t_arr).position.km  # (3, n)

    # Use the direction FROM the ISS TO the body (not the geocentric direction).
    # For the Sun the difference is negligible; for the Moon (~384 000 km) the
    # geocentric approximation shifts the centre line by ~60 km – too large.
    iss_to_body = body_gcrs - iss_gcrs                                         # (3, n)
    body_unit   = iss_to_body / np.linalg.norm(iss_to_body, axis=0)

    # Use Skyfield's full GCRS → ITRS rotation (including precession, nutation,
    # and polar motion) instead of a simple GAST rotation, which has ~5 km
    # residual error that would shift the centre line by ~15 km.
    R = framelib.itrs.rotation_at(t_arr)   # (3, 3, n)

    # Vectorised GCRS → ITRS rotation for all time steps at once.
    # np.einsum('ijn,jn->in', R, v) applies R[:, :, n] @ v[:, n] for each n.
    iss_itrs  = np.einsum("ijn,jn->in", R, iss_gcrs)   # (3, n)
    d_itrs    = np.einsum("ijn,jn->in", R, body_unit)  # (3, n) – rotation preserves unit length

    # Vectorised ray-WGS84-ellipsoid intersection for all steps.
    a, b  = 6378.137, 6356.7523
    inv_a2, inv_b2 = 1.0 / (a * a), 1.0 / (b * b)

    p, d  = iss_itrs, d_itrs
    A_v   = (d[0]**2 + d[1]**2) * inv_a2 + d[2]**2 * inv_b2
    B_v   = -2.0 * ((p[0]*d[0] + p[1]*d[1]) * inv_a2 + p[2]*d[2] * inv_b2)
    C_v   = (p[0]**2 + p[1]**2) * inv_a2 + p[2]**2 * inv_b2 - 1.0
    disc  = B_v**2 - 4.0 * A_v * C_v
    valid = disc >= 0
    sqrt_d = np.where(valid, np.sqrt(np.maximum(disc, 0.0)), 0.0)
    t_v   = np.where(valid, (-B_v - sqrt_d) / (2.0 * A_v), np.nan)
    t_v   = np.where(valid & (t_v > 0), t_v,
                     np.where(valid, (-B_v + sqrt_d) / (2.0 * A_v), np.nan))
    valid = valid & (t_v > 0)

    Q_itrs = p - t_v * d     # (3, n); columns with valid=False contain NaN
    hw_arr = np.where(valid, t_v * np.tan(np.radians(r_deg)), 0.0)

    # Vectorised Bowring geodetic conversion of Q_itrs → (lat, lon).
    e2 = 1.0 - (b / a) ** 2
    qx, qy, qz = Q_itrs[0], Q_itrs[1], Q_itrs[2]
    lon_arr = np.degrees(np.arctan2(qy, qx))
    p2      = np.sqrt(qx**2 + qy**2)
    lat_arr = np.arctan2(qz, p2 * (1.0 - e2))       # initial geocentric estimate
    for _ in range(6):
        N_arr   = a / np.sqrt(1.0 - e2 * np.sin(lat_arr)**2)
        lat_arr = np.arctan2(qz + e2 * N_arr * np.sin(lat_arr), p2)
    lat_arr = np.degrees(lat_arr)

    center_pts: list[tuple[float, float] | None] = []
    halves: list[float] = []
    for i in range(len(offsets)):
        if not valid[i]:
            center_pts.append(None)
            halves.append(0.0)
        else:
            center_pts.append((float(lat_arr[i]), float(lon_arr[i])))
            halves.append(float(hw_arr[i]))

    # In-transit flag: reuse best observer from the transit result
    obs = wgs84.latlon(transit["obs_lat"], transit["obs_lon"])
    iss_alt, iss_az, _ = (iss - obs).at(t_arr).altaz()
    b_alt,   b_az,   _ = (earth + obs).at(t_arr).observe(body_eph).apparent().altaz()
    sep = sep_vec(iss_alt.degrees, iss_az.degrees, b_alt.degrees, b_az.degrees)
    in_transit = list((iss_alt.degrees > 0) & (b_alt.degrees > -1) & (sep < r_deg))

    return {"center": center_pts, "halves": halves, "in_transit": in_transit}


def _corridor_edges(
    center_pts: list, halves: list, mask: list
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Compute left/right edge points for the masked subset of the centre line."""
    active = [(i, pt) for i, (pt, m) in enumerate(zip(center_pts, mask)) if m and pt]
    if not active:
        return [], []

    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []

    for k, (i, (lat, lon)) in enumerate(active):
        hw   = halves[i]
        prev = active[k - 1][1] if k > 0 else None
        nxt  = active[k + 1][1] if k < len(active) - 1 else None

        if prev and nxt:
            bearing = bearing_between(prev[0], prev[1], nxt[0], nxt[1])
        elif prev:
            bearing = bearing_between(prev[0], prev[1], lat, lon)
        elif nxt:
            bearing = bearing_between(lat, lon, nxt[0], nxt[1])
        else:
            # Single point: look at neighbours in the full centre line for bearing
            prev_f = next((center_pts[j] for j in range(i - 1, -1, -1) if center_pts[j]), None)
            nxt_f  = next((center_pts[j] for j in range(i + 1, len(center_pts)) if center_pts[j]), None)
            if prev_f and nxt_f:
                bearing = bearing_between(prev_f[0], prev_f[1], nxt_f[0], nxt_f[1])
            elif prev_f:
                bearing = bearing_between(prev_f[0], prev_f[1], lat, lon)
            elif nxt_f:
                bearing = bearing_between(lat, lon, nxt_f[0], nxt_f[1])
            else:
                bearing = 0.0

        left.append(offset_point(lat, lon, (bearing - 90) % 360, hw))
        right.append(offset_point(lat, lon, (bearing + 90) % 360, hw))

    return left, right


# ── Map generation ─────────────────────────────────────────────────────────────

def generate_map(
    transit: dict,
    corridor: dict,
    output_dir: str,
    search_lat: float,
    search_lon: float,
    location_name: str,
) -> str | None:
    """
    Render a PNG map of the transit visibility corridor.

    The map shows:
      • Blue line & polygon  – approach/departure path with corridor width
      • Red line & polygon   – centre line and corridor during the transit itself
      • Green star           – search centre / location

    Requires: matplotlib, contextily, pyproj
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import contextily as ctx
        from pyproj import Transformer
    except ImportError as exc:
        print(f"  [map] Skipping – missing library ({exc}).")
        print("  [map] Install with: pip install matplotlib contextily pyproj")
        return None

    center_pts = corridor["center"]
    halves     = corridor["halves"]
    in_transit = corridor["in_transit"]

    valid_all    = [pt for pt in center_pts if pt]
    transit_pts  = [pt for pt, it in zip(center_pts, in_transit) if it and pt]

    if not valid_all:
        return None

    all_mask = [pt is not None for pt in center_pts]
    left_ctx,  right_ctx  = _corridor_edges(center_pts, halves, all_mask)
    left_tran, right_tran = _corridor_edges(center_pts, halves, in_transit)

    tfm = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def m(lat: float, lon: float) -> tuple[float, float]:
        x, y = tfm.transform(lon, lat)
        return x, y

    ref    = transit_pts[len(transit_pts) // 2] if transit_pts else valid_all[len(valid_all) // 2]
    cx, cy = m(*ref)
    margin = 80_000   # ±80 km in Web Mercator metres

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    ax.set_xlim(cx - margin, cx + margin)
    ax.set_ylim(cy - margin, cy + margin)

    # OSM background (CartoDB Positron preferred – cleaner for corridor overlay)
    for provider in [ctx.providers.CartoDB.Positron,
                     ctx.providers.OpenStreetMap.Mapnik]:
        try:
            ctx.add_basemap(ax, crs="EPSG:3857", source=provider,
                            zoom="auto", attribution_size=7)
            break
        except Exception:
            continue

    # Full approach/departure corridor polygon (translucent blue)
    if left_ctx and right_ctx:
        poly = left_ctx + right_ctx[::-1]
        xs = [m(la, lo)[0] for la, lo in poly]
        ys = [m(la, lo)[1] for la, lo in poly]
        ax.fill(xs, ys, color="steelblue", alpha=0.18, linewidth=0, zorder=3)

    # Full centre line (blue)
    if valid_all:
        xs = [m(la, lo)[0] for la, lo in valid_all]
        ys = [m(la, lo)[1] for la, lo in valid_all]
        ax.plot(xs, ys, color="steelblue", linewidth=1.4, alpha=0.75,
                zorder=4, label="Approach / departure path")

    # In-transit corridor polygon (crimson, solid)
    if left_tran and right_tran:
        poly = left_tran + right_tran[::-1]
        xs = [m(la, lo)[0] for la, lo in poly]
        ys = [m(la, lo)[1] for la, lo in poly]
        ax.fill(xs, ys, color="crimson", alpha=0.55, linewidth=0, zorder=5)

    # In-transit centre line (crimson, thick)
    if transit_pts:
        xs = [m(la, lo)[0] for la, lo in transit_pts]
        ys = [m(la, lo)[1] for la, lo in transit_pts]
        ax.plot(xs, ys, color="crimson", linewidth=4, zorder=6,
                label="Transit centre line")

    # Search-centre marker
    sx, sy = m(search_lat, search_lon)
    ax.plot(sx, sy, "*", color="limegreen", markersize=16,
            markeredgecolor="black", markeredgewidth=0.8,
            zorder=7, label=location_name or "Search centre")

    # Title and legend
    dt   = transit["time"]
    body = "Solar" if transit["body"] == "sun" else "Lunar"
    ax.set_title(
        f"ISS {body} Transit  ·  {dt.strftime('%Y-%m-%d  %H:%M:%S UTC')}\n"
        f"Duration {transit['duration_s']:.1f} s   ·   "
        f"Min. separation {transit['min_sep_am']:.1f}\"   ·   "
        f"Chord coverage {transit['coverage']:.0f} %",
        fontsize=11, pad=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_axis_off()

    os.makedirs(output_dir, exist_ok=True)
    fname = f"transit_{dt.strftime('%Y%m%d_%H%M%S')}_{transit['body']}.png"
    fpath = os.path.join(output_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return fpath


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
  python transit_finder.py --lat 50.23 --lon 8.62 --radius 30 --name "Bad Homburg"
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
    p.add_argument("--no-maps", action="store_true",
                   help="Skip map generation")
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

    if transits and not args.no_maps:
        print("  Generating maps ...")
        map_paths = []
        for tr in transits:
            corridor = compute_corridor(tr, ts, eph, iss)
            path = generate_map(tr, corridor, MAPS_DIR, args.lat, args.lon, args.name)
            if path:
                map_paths.append(path)
        if map_paths:
            print()
            for p in map_paths:
                print(f"  Map saved: {p}")
            print()


if __name__ == "__main__":
    main()
