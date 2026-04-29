#!/usr/bin/env python3
"""
transit_finder.py  –  ISS-Transite vor Sonne und Mond

Berechnet für die nächsten N Tage alle Überquerungen der ISS vor Sonnenscheibe
oder Mondscheibe, sichtbar von einem definierten Radius um einen Standort.

Voraussetzungen:
    pip install skyfield requests numpy

Aufruf:
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

# ── Algorithmus-Parameter ──────────────────────────────────────────────────────
SUN_RADIUS_DEG   = 0.2665   # mittlerer scheinbarer Radius Sonne  (~16')
MOON_RADIUS_DEG  = 0.2575   # mittlerer scheinbarer Radius Mond   (~15.5')
COARSE_STEP_S    = 10       # Grobscan-Schrittweite in Sekunden
FINE_STEP_S      = 0.05     # Feinscan-Schrittweite in Sekunden
FINE_WINDOW_S    = 70       # Feinscan-Fenster (±Sekunden um Kandidaten)
APPROACH_DEG     = 2.0      # Grad-Schwelle zum Auslösen des Feinscans
DEDUP_S          = 120      # Ereignisse näher als dies = selber Transit
NORAD_ISS        = 25544


# ── TLE-Abruf ─────────────────────────────────────────────────────────────────

def fetch_tle(norad: int = NORAD_ISS) -> tuple[str, str, str]:
    """Aktuelles ISS-TLE von Celestrak GP-API laden."""
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad}&FORMAT=TLE"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
    except Exception as e:
        sys.exit(f"[Fehler] TLE-Abruf gescheitert: {e}")
    sys.exit("[Fehler] Unerwartetes TLE-Format von Celestrak")


# ── Geometrie-Hilfen ───────────────────────────────────────────────────────────

def sep_vec(alt1, az1, alt2, az2):
    """Vektorisierter Winkelabstand (Grad) zwischen zwei Altazimut-Positionen."""
    a1, z1 = np.radians(alt1), np.radians(az1)
    a2, z2 = np.radians(alt2), np.radians(az2)
    c = np.sin(a1) * np.sin(a2) + np.cos(a1) * np.cos(a2) * np.cos(z1 - z2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def chord_coverage(min_sep_deg: float, r_deg: float) -> float:
    """Anteil des Durchmessers, durch den die ISS zieht (0–100 %)."""
    if min_sep_deg >= r_deg:
        return 0.0
    return math.sqrt(max(0.0, 1.0 - (min_sep_deg / r_deg) ** 2)) * 100.0


def azimuth_label(az_deg: float) -> str:
    labels = ["N","NNO","NO","ONO","O","OSO","SO","SSO",
              "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return labels[round(az_deg / 22.5) % 16]


# ── Beobachternetz ─────────────────────────────────────────────────────────────

def observer_ring(lat: float, lon: float, radius_km: float) -> list[tuple[float, float]]:
    """Mittelpunkt + 8 Punkte auf dem Radius-Kreis (Himmelsrichtungen + 45°)."""
    pts = [(lat, lon)]
    if radius_km > 0.1:
        for bearing_deg in range(0, 360, 45):
            b = math.radians(bearing_deg)
            dlat = (radius_km / 111.0) * math.cos(b)
            dlon = (radius_km / (111.0 * math.cos(math.radians(lat)))) * math.sin(b)
            pts.append((lat + dlat, lon + dlon))
    return pts


# ── Grobscan ──────────────────────────────────────────────────────────────────

def coarse_scan(
    lat: float, lon: float, days: int, ts, eph, iss
) -> tuple[list[tuple[float, str]], float]:
    """
    Vektorisierter Scan vom Zentrum-Beobachter.
    Gibt (Kandidatenliste, t0_tt) zurück; Kandidaten sind (tt_jd, körper).
    """
    earth    = eph["earth"]
    observer = wgs84.latlon(lat, lon)

    t0    = datetime.now(timezone.utc)
    n     = int(days * 86400 / COARSE_STEP_S) + 1
    t0_tt = ts.from_datetime(t0).tt
    tt    = t0_tt + np.arange(n) * (COARSE_STEP_S / 86400.0)
    t_arr = ts.tt_jd(tt)

    print(f"  Grobscan: {n:,} Schritte à {COARSE_STEP_S} s  ...", end=" ", flush=True)

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

    # Cluster: aufeinanderfolgende Treffer desselben Körpers zusammenfassen
    clustered: list[tuple[float, str]] = []
    last_t: dict[str, float] = {}
    for t_c, body in candidates:
        if body not in last_t or (t_c - last_t[body]) * 86400 > COARSE_STEP_S * 2.5:
            clustered.append((t_c, body))
        last_t[body] = t_c

    print(f"{len(clustered)} Kandidaten.")
    return clustered, t0_tt


# ── Feinscan ──────────────────────────────────────────────────────────────────

def fine_scan(
    t_cand_tt: float,
    body_name: str,
    observers: list[tuple[float, float]],
    ts, eph, iss,
) -> dict | None:
    """
    Präziser Scan im Fenster ±FINE_WINDOW_S um einen Kandidaten.
    Prüft alle Beobachterpunkte im Radius und gibt das beste Ergebnis zurück.
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
            continue  # ISS verfehlt die Scheibe an diesem Punkt

        # Dauer: Zeitschritte, in denen ISS innerhalb der Scheibe liegt
        in_disk    = valid & (sep < r_deg)
        duration_s = float(np.sum(in_disk)) * FINE_STEP_S

        # Körper-Position zum Transitzeitpunkt (Einzelschritt für Ausgabe)
        t_transit  = ts.tt_jd(float(tt[idx]))
        t_utc      = t_transit.utc_datetime()
        ba, baz, _ = (earth + obs).at(t_transit).observe(body_eph).apparent().altaz()

        # ISS-Position für Höhenwinkel
        ia, iaz, _ = (iss - obs).at(t_transit).altaz()

        # Abstand dieses Beobachters vom Suchzentrum
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


# ── Orchestrierung ─────────────────────────────────────────────────────────────

def find_all_transits(
    lat: float, lon: float, radius_km: float, days: int, ts, eph, iss
) -> list[dict]:
    observers = observer_ring(lat, lon, radius_km)
    clustered, _ = coarse_scan(lat, lon, days, ts, eph, iss)

    print(f"  Feinscan: {len(clustered)} Kandidaten analysieren ...", flush=True)

    transits: list[dict] = []
    seen: list[tuple[float, str]] = []   # (tt, körper) bereits gefundener Transite

    for t_cand, body in clustered:
        # Nicht denselben Transit zweimal zählen
        if any(abs(t_cand - st) * 86400 < DEDUP_S and sb == body for st, sb in seen):
            continue

        result = fine_scan(t_cand, body, observers, ts, eph, iss)
        if result:
            seen.append((result["tt"], body))
            transits.append(result)

    transits.sort(key=lambda x: x["time"])
    return transits


# ── Ausgabe ───────────────────────────────────────────────────────────────────

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
    print("  ISS-TRANSIT-FINDER  –  Sonnentransite & Mondtransite")
    print("─" * W)
    loc_str = f"{location_name}  " if location_name else ""
    ns = "N" if lat >= 0 else "S"
    ew = "O" if lon >= 0 else "W"
    print(f"  Ort:     {loc_str}{abs(lat):.4f}° {ns},  {abs(lon):.4f}° {ew}")
    print(f"  Radius:  {radius_km:.0f} km")
    print(f"  Periode: {now.strftime('%d.%m.%Y')} – {end.strftime('%d.%m.%Y')} UTC")
    print("═" * W)

    if not transits:
        print()
        print("  Keine ISS-Transite in diesem Zeitraum und Gebiet gefunden.")
        print()
        print("  Hinweis: ISS-Transite vor Sonne/Mond sind sehr seltene Ereignisse.")
        print("  Typisch: 0–3 Transite pro Monat an einem gegebenen Standort.")
        print()
        return

    for i, tr in enumerate(transits, 1):
        dt    = tr["time"]
        is_sun = tr["body"] == "sun"
        label  = "SONNEN-TRANSIT  ☀" if is_sun else "MOND-TRANSIT  🌙"
        sep_am = tr["min_sep_am"]
        r_am   = tr["r_am"]
        day_str = dt.strftime("%d.%m.%Y")

        print()
        print(f"  # {i}  {label}  –  {day_str}")
        print("─" * W)
        print(f"  Zeitpunkt (UTC): {dt.strftime('%H:%M:%S')}")
        print(f"  Dauer:           {tr['duration_s']:.1f} s")

        if sep_am < r_am:
            kern = "KERN-TRANSIT  ✓  (ISS zieht über die Scheibenmitte)"
            print(f"  Min. Abstand:    {sep_am:.1f}\"  →  {kern}")
        else:
            print(f"  Min. Abstand:    {sep_am:.1f}\"  (Scheiben-Radius: {r_am:.1f}\")")
        print(f"  Sehnen-Anteil:   {tr['coverage']:.0f} % des Scheibendurchmessers")

        body_label = "Sonne" if is_sun else "Mond"
        az_l = azimuth_label(tr["body_az"])
        print(f"  {body_label}-Position:  Höhe {tr['body_alt']:.1f}°,  "
              f"Azimut {tr['body_az']:.1f}° ({az_l})")
        iss_az_l = azimuth_label(tr["iss_az"])
        print(f"  ISS-Position:    Höhe {tr['iss_alt']:.1f}°,  "
              f"Azimut {tr['iss_az']:.1f}° ({iss_az_l})")

        if tr["obs_dist_km"] < 1.0:
            vis_str = "Standort-Mittelpunkt"
        else:
            vis_str = f"~{tr['obs_dist_km']:.0f} km vom Mittelpunkt"
        print(f"  Bester Punkt:    {tr['obs_lat']:.4f}° N,  "
              f"{tr['obs_lon']:.4f}° O  ({vis_str})")

    print()
    print("─" * W)
    print(f"  Gesamt: {len(transits)} Transit(e) gefunden.")
    print("═" * W)
    print()
    print("  Hinweise:")
    print("  • Genauigkeit hängt vom TLE-Alter ab (< 2 Tage empfohlen).")
    print("  • Sichtbarkeitskorridor ist typisch 5–15 km breit.")
    print("  • Bei Sonnen-Transiten: Sonnenfilter verwenden!")
    print()


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ISS-Transite vor Sonne und Mond für die nächsten N Tage berechnen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python transit_finder.py --lat 50.11 --lon 8.68 --radius 100 --name Frankfurt
  python transit_finder.py --lat 48.14 --lon 11.58 --days 14
  python transit_finder.py --lat 52.52 --lon 13.41 --radius 25 --name Berlin
""",
    )
    p.add_argument("--lat",    type=float, required=True,
                   help="Geografische Breite in Grad (N positiv)")
    p.add_argument("--lon",    type=float, required=True,
                   help="Geografische Länge in Grad (O positiv)")
    p.add_argument("--radius", type=float, default=50.0,
                   help="Suchradius in km (Standard: 50)")
    p.add_argument("--days",   type=int,   default=7,
                   help="Vorhersagezeitraum in Tagen (Standard: 7)")
    p.add_argument("--name",   type=str,   default="",
                   help="Ortsbezeichnung für die Ausgabe (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print()
    print("ISS-Transit-Finder – Initialisierung")
    print("─" * 45)

    print(f"  Lade TLE (NORAD {NORAD_ISS}) von Celestrak ...", end=" ", flush=True)
    tle_name, line1, line2 = fetch_tle()
    print(f"OK  [{tle_name.strip()}]")

    print("  Lade Planetenephemeriden (de421.bsp)  ...", end=" ", flush=True)
    ts  = load.timescale()
    eph = load("de421.bsp")
    iss = EarthSatellite(line1, line2, tle_name, ts)
    print("OK")

    tle_epoch = iss.epoch.utc_datetime()
    age_days  = (datetime.now(timezone.utc) - tle_epoch).total_seconds() / 86400
    print(f"  TLE-Epoche: {tle_epoch.strftime('%d.%m.%Y %H:%M UTC')}  "
          f"(Alter: {age_days:.1f} Tage)")
    if age_days > 3:
        print(f"  ⚠  TLE älter als 3 Tage – Positionsgenauigkeit reduziert!")

    print()
    transits = find_all_transits(
        args.lat, args.lon, args.radius, args.days, ts, eph, iss
    )

    print_results(transits, args.lat, args.lon, args.radius, args.days, args.name)


if __name__ == "__main__":
    main()
